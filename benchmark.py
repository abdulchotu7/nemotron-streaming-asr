"""Production-quality benchmarking framework for the Nemotron streaming ASR engine.

Instrumentation is optional: attach a ``PerformanceStats`` to the pipeline via
``NemotronStreamingSession(model, language, stats=stats)``. When ``stats`` is
None (the default) or ``enabled=False``, the engine behaves exactly as the
uninstrumented build -- no timing, no ``mx.eval`` synchronization, no extra
allocation.

All durations are captured with ``time.perf_counter_ns()`` (nanosecond
resolution). Stages measured independently, per emitted result where relevant:

    microphone PCM arrival  (session.feed())
        -> audio feed       (StreamingAudioBuffer.feed)
        -> mel extraction   (log_mel_spectrogram_frames, mx.eval-synced)
        -> encoder          (per prompted chunk, mx.eval-synced)
        -> decoder          (per prompted chunk; argmax already syncs)
        -> session step     (one full step() call)
        -> end-to-end       (last PCM arrival -> emitted AlignedResult)
        -> token latency    (same anchor, sampled per decoded token)

``mx.eval`` and ``mx.clear_cache`` durations are recorded independently so the
reported stage times reflect real GPU/CPU execution, never lazy dispatch.

Usage:
    python benchmark.py --durations 60 300 900 --language en-US
"""

import argparse
import ctypes
import math
import resource
import threading
import time
import tracemalloc
from ctypes import POINTER, Structure, byref, c_int, c_size_t, c_uint
from pathlib import Path

import mlx.core as mx
import numpy as np

from streaming_session import NemotronStreamingSession

_NS = 1e6  # ns -> ms


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def _percentile(sorted_ns, p):
    if not sorted_ns:
        return 0
    k = max(0, min(len(sorted_ns) - 1, int(math.ceil(p * len(sorted_ns))) - 1))
    return sorted_ns[k]


def _stats(samples_ns):
    """Return {avg, median, p95, p99, max, min, count} in ns, or None."""
    if not samples_ns:
        return None
    s = sorted(samples_ns)
    n = len(s)
    return {
        "count": n,
        "avg": sum(s) / n,
        "median": s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2,
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "max": s[-1],
        "min": s[0],
    }


# --------------------------------------------------------------------------
# system monitoring (CPU / memory / unified memory / MLX)
# --------------------------------------------------------------------------
class _VMStatistics64(Structure):
    """mach/vm_statistics.h vm_statistics64 (host_statistics64, HOST_VM_INFO64)."""

    _fields_ = [
        ("free_count", c_uint),
        ("active_count", c_uint),
        ("inactive_count", c_uint),
        ("wire_count", c_uint),
        ("zero_fill_count", c_uint),
        ("reactivations", c_uint),
        ("pageins", c_uint),
        ("pageouts", c_uint),
        ("faults", c_uint),
        ("cow_faults", c_uint),
        ("lookups", c_uint),
        ("hits", c_uint),
        ("purges", c_uint),
        ("purgeable_count", c_uint),
        ("speculative_count", c_uint),
        ("decompressions", c_uint),
        ("compressions", c_uint),
        ("swapins", c_uint),
        ("swapouts", c_uint),
        ("compressor_page_count", c_uint),
        ("throttled_count", c_uint),
        ("external_page_count", c_uint),
        ("internal_page_count", c_uint),
        ("total_uncompressed_pages_in_compressor", c_uint),
    ]


def macos_unified_memory_used_bytes():
    """System-wide unified memory in use (bytes) via host_statistics64.

    Uses the Activity-Monitor-equivalent definition: active + wired +
    compressed pages (speculative is not included; it is repurposed on current
    macOS and can report bogus values). Returns None if unavailable.
    """
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        libc.mach_host_self.restype = c_uint
        host = libc.mach_host_self()
        stats = _VMStatistics64()
        count = c_uint(ctypes.sizeof(_VMStatistics64) // ctypes.sizeof(c_uint))
        libc.host_statistics64.restype = c_int
        libc.host_statistics64.argtypes = [
            c_uint, c_int, POINTER(_VMStatistics64), POINTER(c_uint),
        ]
        if libc.host_statistics64(host, 4, byref(stats), byref(count)) != 0:
            return None
        page_size = libc.sysconf(29)  # _SC_PAGESIZE
        pages = stats.active_count + stats.wire_count + stats.compressor_page_count
        return pages * page_size
    except Exception:
        return None


def mlx_memory_info():
    """MLX device memory (bytes) and device info; {} if unavailable."""
    info = {}
    try:
        info["mlx_active_bytes"] = mx.get_active_memory()
        info["mlx_peak_bytes"] = mx.get_peak_memory()
        info["mlx_cache_bytes"] = mx.get_cache_memory()
        try:
            device = mx.device_info()
            info["mlx_total_bytes"] = device.get("memory_size")
            info["gpu_name"] = device.get("device_name")
        except Exception:
            pass
    except Exception:
        pass
    return info


class SystemMonitor:
    """Samples process CPU %, RSS, peak RSS, unified memory and MLX memory.

    CPU % is derived from ``getrusage`` CPU-time deltas over wall time; if
    ``psutil`` is installed it is used for current RSS, otherwise only the
    process peak RSS (``ru_maxrss``, bytes on macOS) is reported.
    """

    def __init__(self):
        self._cpu_last = None  # (wall_s, user+sys seconds)
        self._psutil = None
        try:
            import psutil  # optional

            self._psutil = psutil.Process()
        except Exception:
            self._psutil = None
        self.sample()  # prime the CPU baseline so the first delta is valid

    def sample(self):
        now = time.perf_counter()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu_user_sys = ru.ru_utime + ru.ru_stime

        cpu_pct = 0.0
        if self._cpu_last is not None:
            dt = now - self._cpu_last[0]
            if dt > 0:
                cpu_pct = 100.0 * (cpu_user_sys - self._cpu_last[1]) / dt
        self._cpu_last = (now, cpu_user_sys)

        rss_bytes = None
        if self._psutil is not None:
            try:
                rss_bytes = self._psutil.memory_info().rss
            except Exception:
                rss_bytes = None

        info = {
            "cpu_percent": cpu_pct,
            "rss_bytes": rss_bytes,
            "peak_rss_bytes": ru.ru_maxrss,  # bytes on macOS
            "unified_used_bytes": macos_unified_memory_used_bytes(),
        }
        info.update(mlx_memory_info())
        return info


# --------------------------------------------------------------------------
# PerformanceStats
# --------------------------------------------------------------------------
_STAGE_KEYS = (
    "audio_feed", "mel", "encoder", "decoder", "step",
    "end_to_end", "token_latency", "eval", "clear_cache",
)
_COUNT_KEYS = ("mel_chunks", "encoder_chunks", "decoder_chunks", "results",
               "tokens", "words", "feeds")

# stage key -> throughput counter key
_STAGE_COUNT_KEYS = {
    "mel": "mel_chunks",
    "encoder": "encoder_chunks",
    "decoder": "decoder_chunks",
}


class PerformanceStats:
    """Thread-safe collector of streaming pipeline timings.

    Every ``record_*`` call appends one sample (nanoseconds). Statistics
    (average / median / p95 / p99 / max / min) are computed on demand. An
    optional daemon thread prints a rolling report every ``rolling_interval_s``
    seconds and samples memory stability.

    When ``enabled=False`` all recording is a no-op.
    """

    def __init__(self, enabled=True, max_samples=1_000_000, rolling_interval_s=5.0):
        self.enabled = bool(enabled)
        self.max_samples = max_samples
        self.rolling_interval_s = rolling_interval_s

        self._lock = threading.Lock()
        self._stages = {k: [] for k in _STAGE_KEYS}
        self._counts = {k: 0 for k in _COUNT_KEYS}
        self._last_arrival_ns = None
        self._start_ns = None
        self._end_ns = None

        self._rolling_thread = None
        self._rolling_stop = threading.Event()
        self._memory_sampler = None
        self._memory_samples = []
        self._system = SystemMonitor()

    # ------------------------------------------------------------- lifecycle
    def start(self, rolling=True, memory_sampler=None):
        """Start the wall clock and (optionally) the rolling reporter thread."""
        if not self.enabled:
            return
        self._start_ns = time.perf_counter_ns()
        self._end_ns = None
        self._memory_sampler = memory_sampler
        try:
            reset = getattr(mx, "reset_peak_memory", None)
            if reset is None:
                reset = mx.metal.reset_peak_memory
            reset()
        except Exception:
            pass
        if rolling:
            self._rolling_stop.clear()
            self._rolling_thread = threading.Thread(
                target=self._rolling_loop, name="perf-stats-rolling", daemon=True
            )
            self._rolling_thread.start()

    def stop(self):
        """Stop the rolling thread and fix the wall-clock end time."""
        if self._rolling_thread is not None:
            self._rolling_stop.set()
            self._rolling_thread.join(timeout=2)
            self._rolling_thread = None
        if self.enabled:
            self._end_ns = time.perf_counter_ns()
            self._sample_memory()  # final stability sample

    def wall_ns(self):
        if self._start_ns is None:
            return 0
        end = self._end_ns if self._end_ns is not None else time.perf_counter_ns()
        return end - self._start_ns

    # --------------------------------------------------------------- records
    def record_audio_feed(self, ns): self._append("audio_feed", ns)
    def record_mel(self, ns): self._append("mel", ns)
    def record_encoder(self, ns): self._append("encoder", ns)
    def record_decoder(self, ns): self._append("decoder", ns)
    def record_step(self, ns): self._append("step", ns)
    def record_eval(self, ns): self._append("eval", ns)
    def record_clear_cache(self, ns): self._append("clear_cache", ns)

    def record_arrival(self, ns):
        """Timestamp a PCM arrival; anchors end-to-end / token latency."""
        if not self.enabled:
            return
        with self._lock:
            self._last_arrival_ns = ns
            self._counts["feeds"] += 1
            if self._start_ns is None:
                self._start_ns = ns

    def record_result(self, emit_ns, result):
        """Record one emitted AlignedResult: end-to-end latency (vs the last
        PCM arrival) plus one token-latency sample per decoded token."""
        if not self.enabled:
            return
        with self._lock:
            tokens = sum(len(s.tokens) for s in result.sentences)
            words = len(result.text.split())
            self._counts["results"] += 1
            self._counts["tokens"] += tokens
            self._counts["words"] += words
            arrival = self._last_arrival_ns
            if arrival is not None:
                latency_ns = emit_ns - arrival
                self._append_locked("end_to_end", latency_ns)
                for _ in range(tokens):
                    self._append_locked("token_latency", latency_ns)

    def _append(self, key, ns):
        if not self.enabled:
            return
        with self._lock:
            self._append_locked(key, ns)

    def _append_locked(self, key, ns):
        bucket = self._stages[key]
        bucket.append(ns)
        if self.max_samples and len(bucket) > self.max_samples:
            del bucket[: len(bucket) - self.max_samples]
        count_key = _STAGE_COUNT_KEYS.get(key)
        if count_key is not None:
            self._counts[count_key] += 1

    # ------------------------------------------------------------ accessors
    def samples(self, key):
        with self._lock:
            return list(self._stages[key])

    def count(self, key):
        with self._lock:
            if key in self._counts:
                return self._counts[key]
            return len(self._stages[key])

    def summary(self, key):
        """_stats dict (ns) or None for a stage."""
        return _stats(self.samples(key))

    @property
    def memory_samples(self):
        with self._lock:
            return list(self._memory_samples)

    # ------------------------------------------------------- memory sampling
    def _sample_memory(self):
        if self._memory_sampler is None:
            return None
        try:
            sample = self._memory_sampler()
            with self._lock:
                self._memory_samples.append(sample)
            return sample
        except Exception as e:  # never let sampling break transcription
            print(f"[perf] memory sampler error: {e!r}")
            return None

    # ------------------------------------------------------- rolling reports
    def _rolling_loop(self):
        while not self._rolling_stop.wait(self.rolling_interval_s):
            try:
                self.print_rolling_snapshot()
            except Exception as e:
                print(f"[perf] rolling report error: {e!r}")

    def print_rolling_snapshot(self):
        if not self.enabled:
            return
        with self._lock:
            snap = {k: list(v) for k, v in self._stages.items()}
            counts = dict(self._counts)
        wall_s = self.wall_ns() / 1e9

        lines = ["=" * 24]
        for name, key in (("Audio", "audio_feed"), ("Mel", "mel"),
                          ("Encoder", "encoder"), ("Decoder", "decoder"),
                          ("Session", "step"), ("End-to-end", "end_to_end")):
            st = _stats(snap[key])
            if not st:
                continue
            lines.append(
                f"{name:<11s} avg {st['avg'] / _NS:7.2f} ms | "
                f"p95 {st['p95'] / _NS:6.2f} | p99 {st['p99'] / _NS:6.2f} "
                f"(n={st['count']})"
            )
        mem = self._sample_memory()
        if mem:
            lines.append(self._system_line(mem))
        if wall_s > 0:
            lines.append(
                f"mel/s {counts['mel_chunks'] / wall_s:6.2f} | "
                f"enc/s {counts['encoder_chunks'] / wall_s:6.2f} | "
                f"dec/s {counts['decoder_chunks'] / wall_s:6.2f} | "
                f"tok/s {counts['tokens'] / wall_s:6.2f} | "
                f"words/s {counts['words'] / wall_s:6.2f}"
            )
        lines.append("=" * 24)
        print("\n".join(lines), flush=True)

    @staticmethod
    def _system_line(mem):
        parts = [f"CPU {mem.get('cpu_percent', 0.0):5.1f}%"]
        if mem.get("rss_bytes") is not None:
            parts.append(f"RSS {mem['rss_bytes'] / 1e6:6.1f} MB")
        if mem.get("unified_used_bytes"):
            parts.append(f"unified {mem['unified_used_bytes'] / 1e9:5.2f} GB")
        if mem.get("mlx_active_bytes"):
            parts.append(f"MLX active {mem['mlx_active_bytes'] / 1e6:6.1f} MB")
        return " | ".join(parts)

    # ----------------------------------------------------------- final report
    def print_final_report(self):
        if not self.enabled:
            return
        with self._lock:
            snap = {k: list(v) for k, v in self._stages.items()}
            counts = dict(self._counts)
            memory_samples = list(self._memory_samples)
        wall_s = self.wall_ns() / 1e9

        print("=" * 30)
        print("Streaming Performance")
        print("=" * 30)
        for name, key, show_min in (
            ("Mel", "mel", True),
            ("Encoder", "encoder", True),
            ("Decoder", "decoder", True),
            ("Audio feed", "audio_feed", True),
            ("Session step", "step", True),
            ("End-to-end", "end_to_end", True),
            ("Token latency", "token_latency", False),
        ):
            st = _stats(snap[key])
            print(name)
            if not st:
                print("  (no samples)")
                print()
                continue
            print(f"  Average: {st['avg'] / _NS:8.2f} ms")
            print(f"  Median:  {st['median'] / _NS:8.2f} ms")
            print(f"  P95:     {st['p95'] / _NS:8.2f} ms")
            print(f"  P99:     {st['p99'] / _NS:8.2f} ms")
            print(f"  Max:     {st['max'] / _NS:8.2f} ms")
            if show_min:
                print(f"  Min:     {st['min'] / _NS:8.2f} ms")
            print(f"  Samples: {st['count']}")
            print()

        print("MLX synchronization")
        for name, key in (("mx.eval", "eval"), ("mx.clear_cache", "clear_cache")):
            st = _stats(snap[key])
            if st:
                print(f"  {name}: avg {st['avg'] / _NS:7.2f} ms | "
                      f"p95 {st['p95'] / _NS:6.2f} | p99 {st['p99'] / _NS:6.2f} "
                      f"(n={st['count']})")
            else:
                print(f"  {name}: (no samples)")
        print()

        print("Throughput")
        if wall_s > 0:
            print(f"  Mel chunks/sec:    {counts['mel_chunks'] / wall_s:8.2f}")
            print(f"  Encoder chunks/sec:{counts['encoder_chunks'] / wall_s:8.2f}")
            print(f"  Decoder chunks/sec:{counts['decoder_chunks'] / wall_s:8.2f}")
            print(f"  Tokens/sec:        {counts['tokens'] / wall_s:8.2f}")
            print(f"  Words/sec:         {counts['words'] / wall_s:8.2f}")
        print()

        if memory_samples:
            print("System (final sample)")
            print("  " + self._system_line(memory_samples[-1]))
            print()
            print("Memory stability")
            self._print_memory_stability(memory_samples)
            print()
        print("=" * 30)

    def _print_memory_stability(self, samples):
        # Only these are genuine stability signals; ``wall_s`` grows by design
        # and non-numeric keys (e.g. gpu_name) are not comparable.
        keys = (
            "waveform_samples", "pending_mel_frames", "mel_cache_frames",
            "attn_cache_elems", "conv_cache_elems", "python_heap_bytes",
            "mlx_active_bytes", "rss_bytes", "peak_rss_bytes",
        )
        growth = []
        for key in keys:
            vals = [s.get(key) for s in samples if isinstance(s.get(key), (int, float))]
            if not vals:
                continue
            first, last = vals[0], vals[-1]
            lo, hi = min(vals), max(vals)
            print(f"  {key:22s} min {self._fmt_mem(lo):>12s} "
                  f"max {self._fmt_mem(hi):>12s} final {self._fmt_mem(last):>12s}")
            if first > 0:
                # allow 25% headroom or 1 MB absolute slack
                if last > first * 1.25 + 1_000_000:
                    growth.append(key)
        if growth:
            print(f"  => GROWTH DETECTED: {', '.join(growth)}")
        else:
            print("  => bounded (no growth)")

    @staticmethod
    def _fmt_mem(v):
        if v is None:
            return "-"
        if not isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, bool):
            return str(v)
        if v >= 1e9:
            return f"{v / 1e9:.2f} GB"
        if v >= 1e6:
            return f"{v / 1e6:.1f} MB"
        if v >= 1e3:
            return f"{v / 1e3:.0f} K"
        return str(v)


# --------------------------------------------------------------------------
# benchmark runner (long-duration, real-time paced)
# --------------------------------------------------------------------------
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
                 tracemalloc_enabled=True):
        self.model = model
        self.language = language
        self.block = block
        self.realtime = realtime
        self.durations = tuple(durations)
        self.rolling_interval_s = rolling_interval_s

        self.stats = PerformanceStats(enabled=True, rolling_interval_s=rolling_interval_s)
        self.session = NemotronStreamingSession(model, language=language, stats=self.stats)

        self.audio_blocks = self._prepare_audio(audio)
        self._system = SystemMonitor()

        if tracemalloc_enabled:
            try:
                tracemalloc.start()
            except Exception:
                pass

    # ------------------------------------------------------------- setup
    def _default_audio_source(self):
        """Prefer the speech WAV shipped with the repo (produces real tokens);
        fall back to a deterministic synthetic signal."""
        wav = Path(__file__).parent / "linus-original-demo_4bucvKgI.wav"
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
        wav = Path(__file__).parent / "linus-original-demo_4bucvKgI.wav"
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
    )
    bench.run()


if __name__ == "__main__":
    main()
