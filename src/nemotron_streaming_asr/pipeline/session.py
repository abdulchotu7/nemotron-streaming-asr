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
from time import perf_counter_ns

from .audio_buffer import StreamingAudioBuffer
from .decoder import StreamingDecoder
from .encoder import StreamingEncoder

logger = logging.getLogger(__name__)

# NVIDIA-trained latency operating points: right context (lookahead) in 80 ms
# subsampled frames. chunk latency = (right + 1) * 80 ms (see the model card).
SUPPORTED_LOOKAHEAD = (0, 1, 3, 6, 13)


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

    def __init__(self, model, language="en-US", stats=None, att_context_size=None):
        self.model = model
        self.language = language
        self._configured_language = language
        # In "auto" prompt mode the model emits a language-ID token (e.g.
        # "<en-US>") as it recognizes speech; the session then switches the
        # encoder prompt to the detected language for the following chunks.
        self.detect_language = language == "auto"
        # Latency operating point (model card): [left, right] in 80 ms frames;
        # chunk latency = (right + 1) * 80 ms. Defaults to the model's config.
        self.att_context_size = list(att_context_size or model.default_att_context_size)
        if int(self.att_context_size[1]) not in SUPPORTED_LOOKAHEAD:
            logger.warning(
                "NemotronStreamingSession: lookahead=%s is not one of the "
                "NVIDIA-trained operating points %s",
                self.att_context_size[1],
                SUPPORTED_LOOKAHEAD,
            )
        # Optional benchmark instrumentation. None or disabled -> zero impact.
        self._stats = stats
        self._bench = stats is not None and stats.enabled

        self.audio = StreamingAudioBuffer(
            model, stats=stats, att_context_size=self.att_context_size
        )
        self.encoder = StreamingEncoder(
            model, att_context_size=self.att_context_size, stats=stats
        )
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

    def _apply_detected_language(self):
        """Switch the encoder prompt language to the decoder's first detected
        language tag (auto mode only; applies to chunks after detection)."""
        if self.detect_language and self.decoder.detected_language is not None:
            self.language = self.decoder.detected_language

    @property
    def detected_language(self):
        """First language tag the decoder emitted (e.g. ``"en-US"``), or None."""
        return self.decoder.detected_language

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
                    self._apply_detected_language()
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
                    self._apply_detected_language()
                    if result:
                        if t0 is not None:
                            self._stats.record_result(perf_counter_ns(), result)
                        yield result
            for prompted in self.encoder.finish(self.language):
                result = self.decoder.feed(prompted)
                self._apply_detected_language()
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
        self.language = self._configured_language
        self.audio.reset()
        self.encoder.reset()
        self.decoder.reset()
