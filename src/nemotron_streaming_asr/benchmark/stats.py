"""PerformanceStats: thread-safe collector of streaming pipeline timings.

All durations are captured with ``time.perf_counter_ns()`` (nanosecond
resolution). Every ``record_*`` call appends one sample; statistics
(average / median / p95 / p99 / max / min) are computed on demand. An optional
daemon thread prints a rolling report every ``rolling_interval_s`` seconds and
samples memory stability.

When ``enabled=False`` all recording is a no-op, so attaching a disabled stats
object to the pipeline is zero-impact.
"""

import math
import threading
import time

import mlx.core as mx

from .system import SystemMonitor

_NS = 1e6  # ns -> ms

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
        self._first_result_ns = None  # first non-empty result, from start

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
    def record_audio_feed(self, ns):
        self._append("audio_feed", ns)

    def record_mel(self, ns):
        self._append("mel", ns)

    def record_encoder(self, ns):
        self._append("encoder", ns)

    def record_decoder(self, ns):
        self._append("decoder", ns)

    def record_step(self, ns):
        self._append("step", ns)

    def record_eval(self, ns):
        self._append("eval", ns)

    def record_clear_cache(self, ns):
        self._append("clear_cache", ns)

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
            # User-perceived latency: audio start -> first visible words.
            if (self._first_result_ns is None and result.text.strip()
                    and self._start_ns is not None):
                self._first_result_ns = emit_ns - self._start_ns

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

    def first_result_ms(self):
        """Milliseconds from feed start to the first non-empty result, or None."""
        if self._first_result_ns is None:
            return None
        return self._first_result_ns / _NS

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

        print("User-perceived latency")
        first = self.first_result_ms()
        if first is not None:
            print(
                f"  Time to first words: {first:8.1f} ms "
                f"(audio start -> first non-empty result; bounded by the native "
                f"chunk accumulation of (lookahead+1)*80 ms plus processing)"
            )
        else:
            print("  Time to first words: (no non-empty result)")
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
