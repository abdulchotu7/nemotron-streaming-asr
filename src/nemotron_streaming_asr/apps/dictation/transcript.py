"""Live transcript state: keeps the newest cumulative transcript and notifies
subscribers on every change.

The recording worker feeds :meth:`update` with each cumulative
``AlignedResult`` from ``session.step()`` / ``session.finish()``. The
controller stores ``current_text`` (only the newest cumulative transcript) and
calls every listener with the new text. Listeners may run on the recording
worker thread; UI backends that must touch the main thread should marshal
accordingly.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional


class LiveTranscriptController:
    """Tracks the newest cumulative transcript and notifies listeners."""

    def __init__(self, on_update: Optional[Callable[[str], None]] = None):
        self._listeners: List[Callable[[str], None]] = []
        if on_update is not None:
            self._listeners.append(on_update)
        self._current_text = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def add_listener(self, fn: Callable[[str], None]) -> None:
        """Register a callback invoked with the new text on every change."""
        self._listeners.append(fn)

    def clear(self) -> None:
        """Reset the transcript (start of a new recording)."""
        with self._lock:
            self._current_text = ""
        for fn in list(self._listeners):
            fn("")

    # ------------------------------------------------------------- state
    @property
    def current_text(self) -> str:
        with self._lock:
            return self._current_text

    def update(self, result) -> None:
        """Store ``result.text`` if it changed and notify all listeners."""
        text = result.text
        with self._lock:
            if text == self._current_text:
                return
            self._current_text = text
        for fn in list(self._listeners):
            fn(text)
