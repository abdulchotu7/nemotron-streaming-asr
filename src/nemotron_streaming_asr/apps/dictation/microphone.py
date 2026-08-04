"""Microphone capture: sounddevice InputStream -> 20 ms PCM blocks.

Blocks are pushed into an internal queue from the audio callback thread and
consumed by the recording worker thread (``poll`` / ``drain``), so the audio
callback never blocks on ASR work.
"""

from __future__ import annotations

import logging
import queue
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MicrophoneRecorder:
    """Streams 20 ms mono float32 PCM blocks into an internal queue.

    Responsibilities:
      * open/close the microphone (``start`` / ``stop``),
      * hand every captured block to the caller via ``poll`` (one block per
        call, blocking briefly) or ``drain`` (all remaining blocks).

    The queue design keeps the audio callback thread free: it only does a
    queue put, and the heavy ASR work happens on the consumer thread.
    """

    def __init__(self, sample_rate: int = 16000, block_size: int = 320):
        self.sample_rate = sample_rate
        self.block_size = block_size  # 320 samples = 20 ms at 16 kHz
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("microphone status: %s", status)
        # (block_size, 1) float32; the session flattens it.
        self._queue.put(indata.copy())

    def poll(self, timeout: float = 0.02) -> Optional[np.ndarray]:
        """Return the next block, or None if none arrived within ``timeout``."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> List[np.ndarray]:
        """Return every block currently queued (used after stop)."""
        blocks: List[np.ndarray] = []
        while True:
            try:
                blocks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return blocks

    def stop(self) -> None:
        """Stop and close the stream (idempotent)."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    close = stop  # alias for symmetry with start/stop
