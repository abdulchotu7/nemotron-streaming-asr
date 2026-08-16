"""Energy-based voice activity detection for auto-stop dictation.

A deliberately simple RMS-threshold VAD: speech is detected when a PCM
block's RMS energy is at/above ``threshold``. Once speech has been seen,
``wants_stop()`` becomes True after ``stop_silence_s`` of continuous
silence (with at least ``min_speech_s`` of speech first, so a stray noise
blip cannot start the silence clock).

The clock is injectable (``now_fn``) so tests can advance time
deterministically. This is demo-grade detection: it does not touch the ASR
pipeline and is not a substitute for a trained model when accuracy matters.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np


class EnergyVAD:
    """RMS-threshold VAD tracking speech/silence for auto-stop."""

    def __init__(
        self,
        threshold: float = 0.01,
        stop_silence_s: float = 1.5,
        min_speech_s: float = 0.2,
        now_fn: Callable[[], float] = time.perf_counter,
    ):
        self.threshold = threshold
        self.stop_silence_s = stop_silence_s
        self.min_speech_s = min_speech_s
        self._now = now_fn

        self.reset()

    def reset(self) -> None:
        """Clear all state (start of a new recording)."""
        self._speech_seen = False
        self._speech_started: Optional[float] = None
        self._last_speech: Optional[float] = None

    def feed(self, pcm) -> bool:
        """Feed one PCM block; returns True if this block is speech-like."""
        rms = float(np.sqrt(np.mean(np.square(pcm))))
        now = self._now()
        if rms >= self.threshold:
            if not self._speech_seen:
                self._speech_seen = True
                self._speech_started = now
            self._last_speech = now
            return True
        return False

    def wants_stop(self) -> bool:
        """True once enough speech was seen and then ``stop_silence_s`` of
        silence followed it (call after every ``feed``)."""
        if not self._speech_seen or self._last_speech is None:
            return False
        if self._last_speech - self._speech_started < self.min_speech_s:
            return False
        return self._now() - self._last_speech >= self.stop_silence_s
