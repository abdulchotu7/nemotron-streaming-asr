"""Long-duration benchmark harness for the Nemotron streaming ASR engine.

Drives a real ``NemotronStreamingSession`` at microphone pace (or in burst
mode) and collects ``PerformanceStats``. Durations run back-to-back on one
session without restarting, so 1/5/15-minute runs verify stable latency,
bounded memory and no cache growth on a single continuous stream.

Usage:
    python -m nemotron_streaming_asr.benchmark.runner --durations 60 300 900
"""

import argparse
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..pipeline.session import NemotronStreamingSession
from .stats import PerformanceStats
from .system import SystemMonitor

def default_audio(rate, seconds=8.0, seed=7):
    """Deterministic synthetic speech-like signal (band-limited noise with a
    slow activity envelope) so runs are reproducible."""
    rng = np.random.default_rng(seed)
    n = int(rate * seconds)
    t = np.arange(n) / rate
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.9 * t)
    noise = rng.standard_normal(n)
    kernel = np.ones(int(0.02 * rate), dtype=np.float64) / (0.02 * rate)
    lp = np.convolve(noise, kernel, mode="same")
    return (0.08 * lp * env).astype(np.float32)


class StreamingBenchmark:
    """Drives a real session at microphone pace and collects PerformanceStats.

    ``durations`` are run back-to-back without restarting the session (same
    encoder/decoder state), so 1/5/15-minute runs verify stable latency,
    bounded memory and no cache growth on a single continuous stream.
    """

    def __init__(self, model, language="en-US", audio=None, block=320,
                 realtime=True, durations=(60, 300, 900), rolling_interval_s=5.0,
                 lookahead=13, tracemalloc_enabled=True):
        self.model = model
        self.language = language
        self.lookahead = lookahead
        self.block = block
        self.realtime = realtime
        self.durations = tuple(durations)
        self.rolling_interval_s = rolling_interval_s

        self.stats = PerformanceStats(enabled=True, rolling_interval_s=rolling_interval_s)
        self.session = NemotronStreamingSession(
            model,
            language=language,
            stats=self.stats,
            att_context_size=[56, lookahead],
        )

        self.audio_blocks = self._prepare_audio(audio)
        self._system = SystemMonitor()

        if tracemalloc_enabled:
            try:
                tracemalloc.start()
            except Exception:
                pass

    # ------------------------------------------------------------- setup
    def _default_audio_source(self):
        """Prefer the speech WAV shipped in ``<repo>/data`` (produces real
        tokens); fall back to a deterministic synthetic signal."""
        wav = Path(__file__).resolve().parents[3] / "data" / "linus-original-demo_4bucvKgI.wav"
        if wav.exists():
            from mlx_audio.stt.utils import load_audio

            a = load_audio(str(wav), self.model.preprocessor_config.sample_rate,
                           dtype=mx.float32)
            return np.array(a, dtype=np.float32)
        return default_audio(self.model.preprocessor_config.sample_rate)

    def _prepare_audio(self, audio):
        if audio is None:
            audio = self._default_audio_source()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.shape[0] < self.block:
            audio = np.concatenate(
                [audio, np.zeros(self.block - audio.shape[0], dtype=np.float32)]
            )
        return [
            audio[i : i + self.block]
            for i in range(0, audio.shape[0] - self.block + 1, self.block)
        ]

    def _sample_memory(self):
        enc = self.session.encoder
        audio = self.session.audio
        sample = {
            "wall_s": self.stats.wall_ns() / 1e9,
            "waveform_samples": audio._length,
            "pending_mel_frames": 0 if enc.pending is None else enc.pending.shape[1],
            "mel_cache_frames": 0 if enc.mel_cache is None else enc.mel_cache.shape[1],
            "attn_cache_elems": sum(
                c.size for c in enc.attn_cache if c is not None
            ),
            "conv_cache_elems": sum(
                c.size for c in enc.conv_cache if c is not None
            ),
            "python_heap_bytes": (
                tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else None
            ),
        }
        sample.update(self._system.sample())
        return sample

    # ------------------------------------------------------------- run
    def run(self):
        rate = self.model.preprocessor_config.sample_rate
        tick_ns = int(self.block / rate * 1e9)  # e.g. 320/16000 = 20 ms
        src = self._audio_source_name()
        print(f"Audio source: {src} | block {self.block} samples "
              f"({self.block / rate * 1e3:.1f} ms) | realtime={self.realtime}",
              flush=True)
        self.stats.start(rolling=True, memory_sampler=self._sample_memory)
        try:
            for dur in self.durations:
                print(f"\n--- duration {dur}s ---", flush=True)
                self._run_seconds(dur, tick_ns)
                self.stats.print_rolling_snapshot()
        finally:
            # Flush the trailing partial chunk so every fed sample is accounted.
            for _ in self.session.finish():
                pass
            self.stats.stop()
        self.stats.print_final_report()

    def _audio_source_name(self):
        wav = Path(__file__).resolve().parents[3] / "data" / "linus-original-demo_4bucvKgI.wav"
        if wav.exists():
            return f"{wav.name} (loop)"
        return "deterministic synthetic signal"

    def _run_seconds(self, dur_s, tick_ns):
        end_ns = time.perf_counter_ns() + int(dur_s * 1e9)
        next_tick = time.perf_counter_ns()
        blocks = self.audio_blocks
        n = 0
        while time.perf_counter_ns() < end_ns:
            self.session.feed(blocks[n % len(blocks)])
            for _ in self.session.step():
                pass
            n += 1
            next_tick += tick_ns
            if self.realtime:
                delay_ns = next_tick - time.perf_counter_ns()
                if delay_ns > 0:
                    time.sleep(delay_ns / 1e9)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Nemotron streaming ASR engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python benchmark.py --durations 60 300 900",
    )
    parser.add_argument("--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
    parser.add_argument("--language", default="en-US")
    parser.add_argument(
        "--lookahead",
        type=int,
        default=13,
        choices=[0, 1, 3, 6, 13],
        help="right-context lookahead in 80 ms frames; chunk latency = "
        "(lookahead + 1) * 80 ms (default 13 -> 1.12 s, lowest latency 0 -> 80 ms)",
    )
    parser.add_argument("--audio", default=None,
                        help="WAV path to transcribe in a loop (default: synthetic)")
    parser.add_argument("--durations", type=int, nargs="+", default=[60, 300, 900],
                        help="run lengths in seconds, back-to-back (default: 60 300 900)")
    parser.add_argument("--block", type=int, default=320, help="PCM samples per feed (default 320 = 20 ms)")
    parser.add_argument("--no-realtime", action="store_true",
                        help="feed as fast as possible instead of pacing to real time")
    parser.add_argument("--rolling", type=float, default=5.0,
                        help="rolling report interval in seconds (default 5)")
    args = parser.parse_args()

    from mlx_audio.stt import load

    model = load(args.model)
    model.eval()

    audio = None
    if args.audio:
        from mlx_audio.stt.utils import load_audio

        wav = load_audio(args.audio, model.preprocessor_config.sample_rate,
                         dtype=mx.float32)
        audio = np.array(wav, dtype=np.float32)

    bench = StreamingBenchmark(
        model,
        language=args.language,
        audio=audio,
        block=args.block,
        realtime=not args.no_realtime,
        durations=tuple(args.durations),
        rolling_interval_s=args.rolling,
        lookahead=args.lookahead,
    )
    bench.run()


if __name__ == "__main__":
    main()
