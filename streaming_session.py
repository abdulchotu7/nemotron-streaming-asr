"""Live microphone streaming ASR session.

Pipeline:

    Microphone
        |
        v
    StreamingAudioBuffer      (bounded PCM history -> log-mel chunks)
        |
        v
    StreamingEncoder          (stateful cache-aware FastConformer encoder)
        |
        v
    StreamingDecoder          (stateful greedy RNN-T decoder)
        |
        v
    NemotronStreamingSession  (thin orchestrator; owns language + reset)
"""

import logging
from collections import deque
from time import perf_counter_ns

import numpy as np
import mlx.core as mx

from mlx_audio.stt.models.nemotron_asr.audio import (
    log_mel_spectrogram_frames,
)

from streaming_encoder import StreamingEncoder
from streaming_decoder import StreamingDecoder

logger = logging.getLogger(__name__)


class NemotronStreamingSession:
    """Thin orchestrator connecting audio buffer, encoder and decoder.

    Owns the prompt ``language`` and the three pipeline stages. Raw mono PCM is
    fed in with ``feed()``; ``step()`` streams cumulative :class:`AlignedResult`
    objects out as complete native chunks become ready. ``reset()`` restarts the
    whole pipeline (push-to-talk) without reloading the model.

        session.feed(pcm)
            |
            v
        StreamingAudioBuffer
            |
            v
        for mel in get_ready_mel_chunks():
            for prompted in encoder.feed(mel, language):
                result = decoder.feed(prompted)
                if result:
                    yield result
    """

    def __init__(self, model, language="en-US", stats=None):
        self.model = model
        self.language = language
        # Optional benchmark instrumentation. None or disabled -> zero impact.
        self._stats = stats
        self._bench = stats is not None and stats.enabled

        self.audio = StreamingAudioBuffer(model, stats=stats)
        self.encoder = StreamingEncoder(model, stats=stats)
        self.decoder = StreamingDecoder(model, stats=stats)

    def feed(self, pcm):
        """Append raw mono PCM (float32) to the audio buffer.

        When benchmarking, records the audio-buffer feed duration and the PCM
        arrival timestamp (anchor for end-to-end / token latency).
        """
        t0 = perf_counter_ns() if self._bench else None
        self.audio.feed(pcm)
        if t0 is not None:
            now = perf_counter_ns()
            self._stats.record_audio_feed(now - t0)
            self._stats.record_arrival(now)

    def step(self):
        """Process every mel chunk that is currently ready.

        Yields one cumulative AlignedResult per prompted encoder chunk. When
        benchmarking, records one ``step`` sample per call plus an end-to-end /
        token-latency sample per emitted result.
        """
        t0 = perf_counter_ns() if self._bench else None
        try:
            for mel in self.audio.get_ready_mel_chunks():
                for prompted in self.encoder.feed(mel, self.language):
                    result = self.decoder.feed(prompted)
                    if result:
                        if t0 is not None:
                            self._stats.record_result(perf_counter_ns(), result)
                        yield result
        finally:
            if t0 is not None:
                self._stats.record_step(perf_counter_ns() - t0)

    def finish(self):
        """Flush all remaining audio and encoder state at end-of-utterance.

        Emits the sub-chunk mel tail still held in the audio buffer, then
        performs the encoder's final ``is_final=True`` flush. Together these
        transcribe every mel frame, matching the reference stream end.
        """
        t0 = perf_counter_ns() if self._bench else None
        try:
            for mel in self.audio.get_tail_mel_chunks():
                for prompted in self.encoder.feed(mel, self.language):
                    result = self.decoder.feed(prompted)
                    if result:
                        if t0 is not None:
                            self._stats.record_result(perf_counter_ns(), result)
                        yield result
            for prompted in self.encoder.finish(self.language):
                result = self.decoder.feed(prompted)
                if result:
                    if t0 is not None:
                        self._stats.record_result(perf_counter_ns(), result)
                    yield result
        finally:
            if t0 is not None:
                self._stats.record_step(perf_counter_ns() - t0)

    def reset(self):
        """Clear all streaming state; the loaded model is kept.

        Safe to reuse the session for a new utterance (push-to-talk).
        """
        self.audio.reset()
        self.encoder.reset()
        self.decoder.reset()


class StreamingAudioBuffer:
    """Bounded raw-PCM history that turns microphone audio into mel chunks.

    Responsibilities:
      * accumulate incoming PCM without unbounded growth -- only the samples
        still needed by future ``log_mel_spectrogram_frames()`` calls are kept
        (the STFT window for ``next_mel_frame`` plus one preemphasis predecessor
        sample, floored to a hop boundary),
      * track global frame/sample offsets so the trimmed storage still yields
        bit-identical mel chunks to offline extraction,
      * emit native ``(1, chunk_mel, features)`` mel chunks for every complete
        native chunk -- a chunk is only emitted once the STFT window of its
        last mel frame is fully available, so the mel values match offline
        extraction exactly (no incremental zero-padding),
    """

    def __init__(self, model, stats=None):
        self.config = model.preprocessor_config
        # Optional benchmark instrumentation. None or disabled -> zero impact.
        self._stats = stats
        self._bench = stats is not None and stats.enabled

        right_context = model.default_att_context_size[1]
        subsampling = model.encoder.args.subsampling_factor

        # Native chunk size expected by stream_encode_chunks()
        self.chunk_mel = (right_context + 1) * subsampling

        # Trimming keeps a prefix of the waveform and re-indexes frames; it is
        # only safe for the settings this model uses (chunked extraction already
        # requires normalize="NA").
        self._can_trim = (
            self.config.pad_to == 0 and self.config.normalize == "NA"
        )
        if not self._can_trim:
            logger.warning(
                "StreamingAudioBuffer: waveform trimming disabled "
                "(pad_to=%s normalize=%s) -- memory is unbounded",
                self.config.pad_to,
                self.config.normalize,
            )

        # Raw waveform history: deque of float32 sample blocks. Appending never
        # copies the retained history; trimming pops from the left.
        self._chunks: deque[np.ndarray] = deque()
        self._length = 0        # samples currently held in _chunks
        self.trim_sample = 0    # global sample index of _chunks[0][0]
        self.total_samples = 0  # samples ever fed (monotonic, never trimmed)
        self.next_mel_frame = 0  # first mel frame not yet emitted (global)
        self._trim_frames = 0    # trim_sample // hop_length

        logger.debug("Chunk mel frames : %d", self.chunk_mel)

    # ------------------------------------------------------------------ input
    def feed(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self._chunks.append(pcm)
        self._length += pcm.shape[0]
        self.total_samples += pcm.shape[0]
        if self._can_trim:
            self._trim()

    @property
    def waveform(self):
        """Contiguous view of the retained waveform (samples [trim_sample, total))."""
        return self._contiguous()

    @property
    def available_frames(self):
        """Total mel frames available from all audio ever fed (global count)."""
        return self.total_samples // self.config.hop_length + 1

    def reset(self):
        """Clear all buffered audio and counters (fresh microphone session)."""
        self._chunks.clear()
        self._length = 0
        self.trim_sample = 0
        self.total_samples = 0
        self.next_mel_frame = 0
        self._trim_frames = 0

    # ------------------------------------------------------------ internals
    def _contiguous(self):
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        if len(self._chunks) == 1:
            return self._chunks[0]
        return np.concatenate(list(self._chunks))

    def _trim(self):
        """Drop every sample before the earliest one a future mel call needs.

        ``log_mel_spectrogram_frames`` for a frame range starting at
        ``next_mel_frame`` reads samples from ``next_mel_frame * hop -
        n_fft // 2`` (the centered STFT window) plus one predecessor sample for
        the preemphasis state (y[n] = x[n] - preemph * x[n-1]). The trim point
        is floored to a hop boundary so that local frame indices passed to
        ``log_mel_spectrogram_frames`` map onto the same global frame indices.
        """
        hop = self.config.hop_length
        keep_from = self.next_mel_frame * hop - self.config.n_fft // 2 - 1
        if keep_from < 0:
            keep_from = 0
        keep_from = keep_from // hop * hop

        drop = keep_from - self.trim_sample
        if drop <= 0:
            return
        while drop > 0 and self._chunks:
            first = self._chunks[0]
            if first.shape[0] <= drop:
                drop -= first.shape[0]
                self.trim_sample += first.shape[0]
                self._length -= first.shape[0]
                self._chunks.popleft()
            else:
                self._chunks[0] = first[drop:]
                self.trim_sample += drop
                self._length -= drop
                drop = 0
        self._trim_frames = self.trim_sample // hop

    # ------------------------------------------------------------- emitting
    def get_ready_mel_chunks(self):
        """Yield one (1, chunk_mel, features) mel chunk per complete native chunk.

        The retained waveform is trimmed and the mel window is computed with
        local frame indices, so the emitted mel values are bit-identical to
        computing over the full waveform. A chunk is emitted only once the
        STFT window of its last mel frame is fully available (see ``_full_chunk_ready``),
        so no frame is incrementally zero-padded.
        """
        if self._can_trim:
            self._trim()

        waveform = mx.array(self._contiguous())

        # Snapshots keep this generator immune to concurrent feed()/reset().
        total = self.total_samples
        next_mel_frame = self.next_mel_frame
        trim_frames = self._trim_frames

        logger.debug(
            "Waveform samples : %d | Total samples : %d | Next mel frame : %d",
            self._length,
            total,
            next_mel_frame,
        )

        while self._full_chunk_ready(total, next_mel_frame):
            start = next_mel_frame - trim_frames
            end = start + self.chunk_mel

            t0 = perf_counter_ns() if self._bench else None
            mel = log_mel_spectrogram_frames(
                waveform,
                self.config,
                start,
                end,
            )
            if t0 is not None:
                # Synchronize so the recorded time reflects real extraction
                # work (MLX is async); record the eval duration separately.
                t_sync = perf_counter_ns()
                mx.eval(mel)
                t1 = perf_counter_ns()
                self._stats.record_eval(t1 - t_sync)
                self._stats.record_mel(t1 - t0)

            self.next_mel_frame = next_mel_frame + self.chunk_mel
            next_mel_frame += self.chunk_mel

            logger.debug("Emit mel frames %d -> %d  shape %s", start, end, mel.shape)
            yield mel

        if self._can_trim:
            self._trim()

    def get_tail_mel_chunks(self):
        """Yield the trailing mel frames (< chunk_mel) as one final chunk.

        Called at end-of-utterance so the final partial chunk is transcribed,
        matching the reference's final ``is_final=True`` flush. The last frames
        right-zero-pad exactly like the offline extraction of the same signal.
        """
        if self._can_trim:
            self._trim()

        waveform = mx.array(self._contiguous())
        hop = self.config.hop_length

        total = self.total_samples
        next_mel_frame = self.next_mel_frame
        trim_frames = self._trim_frames
        available = total // hop + 1
        remaining = available - next_mel_frame
        if remaining <= 0:
            return

        start = next_mel_frame - trim_frames
        end = start + remaining
        t0 = perf_counter_ns() if self._bench else None
        mel = log_mel_spectrogram_frames(waveform, self.config, start, end)
        if t0 is not None:
            t_sync = perf_counter_ns()
            mx.eval(mel)
            t1 = perf_counter_ns()
            self._stats.record_eval(t1 - t_sync)
            self._stats.record_mel(t1 - t0)
        self.next_mel_frame = next_mel_frame + remaining
        logger.debug("Emit tail mel frames %d -> %d  shape %s", start, end, mel.shape)
        yield mel

    def _full_chunk_ready(self, total_samples, next_mel_frame):
        """True when the STFT window of the last frame of the next chunk is
        fully covered by received samples: frame ``next + chunk_mel - 1`` needs
        samples up to ``frame*hop + n_fft//2`` (right window edge)."""
        hop = self.config.hop_length
        last_frame = next_mel_frame + self.chunk_mel - 1
        return total_samples >= last_frame * hop + self.config.n_fft // 2
