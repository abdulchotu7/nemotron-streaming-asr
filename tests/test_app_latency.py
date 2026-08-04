"""LatencyProbe (apps/mic.py): user-perceived gap measurement."""

import time

import numpy as np


def test_latency_probe_measures_speech_to_first_words():
    from nemotron_streaming_asr.apps.mic import LatencyProbe

    lines = []
    probe = LatencyProbe(vad_threshold=0.01, print_fn=lines.append)
    probe.mic_started()
    assert probe.mic_start_ns is not None

    # Silence must not trigger speech onset.
    probe.feed_pcm(np.zeros(320, dtype=np.float32))
    assert probe.speech_start_ns is None

    # A loud block latches speech onset at ~now.
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal(320) * 0.05).astype(np.float32)
    t0 = time.perf_counter_ns()
    probe.feed_pcm(loud)
    assert probe.speech_start_ns is not None
    assert abs(probe.speech_start_ns - t0) < 5e6

    # No text yet -> nothing reported.
    probe.on_text("")
    assert probe.first_words_ns is None

    # First visible words -> both gaps reported.
    probe.on_text("hello world")
    assert probe.first_words_text == "hello world"
    assert (probe.first_words_ns - probe.speech_start_ns) / 1e6 < 5.0
    assert (probe.first_words_ns - probe.mic_start_ns) / 1e6 >= 0
    assert any("first words after" in line for line in lines)
    assert any("speech onset" in line for line in lines)

    # Only the first non-empty text is latched.
    probe.on_text("hello world again")
    assert probe.first_words_text == "hello world"


def test_latency_probe_threshold():
    from nemotron_streaming_asr.apps.mic import LatencyProbe

    probe = LatencyProbe(vad_threshold=0.1)
    probe.feed_pcm((np.random.default_rng(1).standard_normal(320) * 0.05).astype(np.float32))
    assert probe.speech_start_ns is None  # below threshold
