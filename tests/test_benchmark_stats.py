"""PerformanceStats unit tests and a short benchmark-runner smoke test."""

import time

import mlx.core as mx
import pytest

from nemotron_streaming_asr import PerformanceStats
from nemotron_streaming_asr.benchmark.system import (
    macos_unified_memory_used_bytes,
    mlx_memory_info,
)


class _FakeResult:
    def __init__(self, text, tokens):
        self.text = text
        self.sentences = [type("S", (), {"tokens": tokens})()]


def test_disabled_stats_are_noop():
    stats = PerformanceStats(enabled=False)
    stats.record_mel(123)
    stats.record_result(1000, _FakeResult("hi there", [1, 2, 3]))
    assert stats.count("mel") == 0
    assert stats.count("end_to_end") == 0
    assert stats.summary("mel") is None


def test_record_and_statistics():
    stats = PerformanceStats(enabled=True)
    for v in range(1, 101):  # 1..100 ms
        stats.record_mel(v * 1_000_000)
    s = stats.summary("mel")
    assert s["count"] == 100
    assert s["min"] == 1e6 and s["max"] == 100e6
    assert abs(s["avg"] - 50.5e6) < 1e3
    assert abs(s["median"] - 50.5e6) < 1e3
    assert abs(s["p95"] - 95e6) < 1e3
    assert abs(s["p99"] - 99e6) < 1e3


def test_result_counts_and_latency():
    stats = PerformanceStats(enabled=True)
    t0 = time.perf_counter_ns()
    stats.record_arrival(t0)
    stats.record_result(t0 + 5_000_000, _FakeResult("one two three", [1, 2, 3, 4]))
    assert stats.count("results") == 1
    assert stats.count("tokens") == 4
    assert stats.count("words") == 3
    e2e = stats.samples("end_to_end")
    assert e2e == [5_000_000]
    assert stats.count("token_latency") == 4  # one sample per token
    assert stats.samples("token_latency") == [5_000_000] * 4


def test_stage_counts_drive_throughput():
    stats = PerformanceStats(enabled=True)
    for _ in range(3):
        stats.record_mel(1)
        stats.record_encoder(1)
        stats.record_decoder(1)
    assert stats.count("mel_chunks") == 3
    assert stats.count("encoder_chunks") == 3
    assert stats.count("decoder_chunks") == 3


def test_first_result_latency():
    stats = PerformanceStats(enabled=True)
    stats.start(rolling=False)  # sets the wall-clock anchor
    assert stats.first_result_ms() is None

    # Empty results do not count as "first words".
    stats.record_result(stats._start_ns + 500_000_000, _FakeResult("", []))
    assert stats.first_result_ms() is None

    # First non-empty result latches the user-perceived latency.
    stats.record_result(stats._start_ns + 1_234_000_000, _FakeResult("hi", [1, 2]))
    assert stats.first_result_ms() is not None
    assert abs(stats.first_result_ms() - 1234.0) < 0.01

    # Later results do not move it.
    stats.record_result(stats._start_ns + 3_000_000_000, _FakeResult("hello", [1]))
    assert abs(stats.first_result_ms() - 1234.0) < 0.01
    stats.stop()


def test_system_monitors_return_sane_values():
    # CPU% may be 0 on the first delta but the fields must exist.
    mem = mlx_memory_info()
    assert "mlx_active_bytes" in mem
    assert mem["mlx_active_bytes"] >= 0
    # unified memory: either a positive number or None (non-macOS/ctypes failure).
    used = macos_unified_memory_used_bytes()
    assert used is None or used > 0


@pytest.mark.slow
def test_runner_smoke(tiny_model):
    """The full harness runs end-to-end for a few seconds on one session."""
    from nemotron_streaming_asr import StreamingBenchmark

    bench = StreamingBenchmark(
        tiny_model,
        language="en-US",
        durations=(4, 3),
        rolling_interval_s=2.0,
    )
    # Reuse the package-level stats so we can assert afterwards.
    bench.run()
    stats = bench.stats
    assert stats.count("mel") > 0
    assert stats.count("end_to_end") > 0
    assert stats.count("tokens") >= 0
    assert len(stats.memory_samples) >= 1
    mx.eval(mx.zeros(1))  # keep MLX state clean
