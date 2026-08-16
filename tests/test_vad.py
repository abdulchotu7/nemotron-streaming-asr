"""EnergyVAD: speech/silence tracking and auto-stop decisions."""

import numpy as np

from nemotron_streaming_asr.apps.dictation.vad import EnergyVAD

LOUD = (np.random.default_rng(0).standard_normal(320) * 0.05).astype(np.float32)
SILENCE = np.zeros(320, dtype=np.float32)


def _clock():
    return {"t": 0.0}


def test_silence_only_never_stops():
    vad = EnergyVAD(now_fn=lambda: 0.0)
    for _ in range(500):
        vad.feed(SILENCE)
    assert vad.wants_stop() is False  # no speech seen at all


def test_single_blip_then_silence_never_stops():
    """Less than min_speech_s of speech must not arm the silence timer."""
    clock = _clock()
    vad = EnergyVAD(now_fn=lambda: clock["t"])
    vad.feed(LOUD)  # one blip at t=0
    clock["t"] += 10.0  # long silence after
    assert vad.wants_stop() is False


def test_stops_after_silence_following_speech():
    clock = _clock()
    vad = EnergyVAD(stop_silence_s=1.5, min_speech_s=0.1,
                    now_fn=lambda: clock["t"])
    for _ in range(11):  # ~0.2 s of speech at 20 ms blocks (>= min_speech_s)
        vad.feed(LOUD)
        clock["t"] += 0.02

    assert vad.wants_stop() is False  # still talking

    clock["t"] += 1.0  # 1 s of silence
    assert vad.wants_stop() is False
    clock["t"] += 0.6  # 1.6 s total
    assert vad.wants_stop() is True

    # Speech again resets the silence timer.
    vad.feed(LOUD)
    assert vad.wants_stop() is False


def test_feed_reports_speech_blocks():
    vad = EnergyVAD(now_fn=lambda: 0.0)
    assert vad.feed(SILENCE) is False
    assert vad.feed(LOUD) is True


def test_reset_clears_state():
    clock = _clock()
    vad = EnergyVAD(stop_silence_s=0.5, min_speech_s=0.0,
                    now_fn=lambda: clock["t"])
    vad.feed(LOUD)
    clock["t"] += 1.0
    assert vad.wants_stop() is True

    vad.reset()
    assert vad.wants_stop() is False
    clock["t"] = 0.0
    for _ in range(100):
        vad.feed(SILENCE)
    assert vad.wants_stop() is False
