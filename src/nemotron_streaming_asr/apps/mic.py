"""Live microphone demo with a time-to-first-words latency probe.

Streams raw mono PCM (16 kHz, 20 ms blocks) through the streaming pipeline and
prints cumulative transcripts as chunks complete. A lightweight energy-based
speech-onset detector measures the user-perceived gap:

    [latency] speech onset detected        X ms after mic start
    [latency] first words after            Y ms of speech | Z ms from mic start

The gap is dominated by the native chunk accumulation (~1.12 s of audio per
chunk) plus ~0.1 s of mel/encoder/decoder processing.

Run:
    nemotron-mic
    python -m nemotron_streaming_asr.apps.mic --vad-threshold 0.005
"""

import argparse
import time
from queue import Queue

import numpy as np
import sounddevice as sd

from mlx_audio.stt import load
from nemotron_streaming_asr.pipeline.session import (
    NemotronStreamingSession,
    SUPPORTED_LOOKAHEAD,
)

RATE = 16000
BLOCK = 320  # 20 ms


class LatencyProbe:
    """Measures the user-perceived gap: speech onset -> first words, and
    microphone start -> first words.

    Speech onset is latched on the first PCM block whose RMS energy exceeds
    ``vad_threshold``. This is a demo-grade voice-activity detector; it does
    not touch the ASR pipeline.
    """

    def __init__(self, vad_threshold=0.01, print_fn=print):
        self.vad_threshold = vad_threshold
        self.print = print_fn
        self.mic_start_ns = None
        self.speech_start_ns = None
        self.first_words_ns = None
        self.first_words_text = None

    def mic_started(self):
        self.mic_start_ns = time.perf_counter_ns()

    def feed_pcm(self, pcm):
        """Feed one PCM block; latch the first speech-like block."""
        if self.speech_start_ns is None:
            # float32 in-place math: no float64 copy per block, plenty of
            # precision for an energy threshold probe.
            rms = float(np.sqrt(np.mean(np.square(pcm))))
            if rms >= self.vad_threshold:
                self.speech_start_ns = time.perf_counter_ns()
                self.print(
                    f"[latency] speech onset detected "
                    f"{self._ms(self.speech_start_ns - self.mic_start_ns):.0f} ms "
                    f"after mic start"
                )

    def on_text(self, text):
        """Latch the first non-empty text and report both gaps."""
        if self.first_words_ns is None and text.strip():
            self.first_words_ns = time.perf_counter_ns()
            self.first_words_text = text
            parts = []
            if self.speech_start_ns is not None:
                parts.append(
                    f"{self._ms(self.first_words_ns - self.speech_start_ns):.0f} ms "
                    f"of speech"
                )
            if self.mic_start_ns is not None:
                parts.append(
                    f"{self._ms(self.first_words_ns - self.mic_start_ns):.0f} ms "
                    f"from mic start"
                )
            self.print(f"[latency] first words after {' | '.join(parts)}")
            self.print(f"[latency]   -> {text.strip()!r}")

    @staticmethod
    def _ms(ns):
        return (ns if ns is not None else 0) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
    )
    parser.add_argument("--language", default="en-US")
    parser.add_argument(
        "--lookahead",
        type=int,
        default=13,
        choices=SUPPORTED_LOOKAHEAD,
        help="right-context lookahead in 80 ms frames; chunk latency = "
        "(lookahead + 1) * 80 ms (default 13 -> 1.12 s, lowest latency 0 -> 80 ms)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.01,
        help="RMS energy above which speech onset is assumed (default 0.01)",
    )
    parser.add_argument(
        "--no-latency", action="store_true", help="disable the latency probe"
    )
    args = parser.parse_args()

    model = load(args.model)
    session = NemotronStreamingSession(
        model,
        language=args.language,
        att_context_size=[56, args.lookahead],
    )
    probe = None if args.no_latency else LatencyProbe(vad_threshold=args.vad_threshold)

    audio_queue = Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    last_text = ""

    def emit(result):
        """Print new cumulative text and feed the latency probe."""
        nonlocal last_text
        if result.text != last_text:
            print(result.text)
            last_text = result.text
            if probe is not None:
                probe.on_text(result.text)

    with sd.InputStream(
        samplerate=RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK,
        callback=callback,
    ):
        print("Listening... Ctrl+C to stop")
        if probe is not None:
            probe.mic_started()

        try:
            while True:
                pcm = audio_queue.get()

                if probe is not None:
                    probe.feed_pcm(pcm)

                session.feed(pcm)

                for result in session.step():
                    emit(result)
        finally:
            # Exit path: consume any blocks still queued, then flush the
            # trailing partial chunk so the last ~1.12 s of audio is not lost.
            while True:
                try:
                    pcm = audio_queue.get_nowait()
                except queue.Empty:
                    break
                session.feed(pcm)
                for result in session.step():
                    emit(result)
            for result in session.finish():
                emit(result)


if __name__ == "__main__":
    main()
