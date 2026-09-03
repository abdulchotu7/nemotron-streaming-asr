"""Recording-display seam for the dictation app.

``RecordingDisplay`` is the single interface every display adapter satisfies —
the console printer and the floating overlay alike. The app calls these methods
unconditionally (no ``hasattr`` probing); adapters implement only what they
need, the rest are no-ops.

Thread contract (part of the interface, not just a docstring convention):

* ``show`` may arrive on the hotkey thread: it must only record intent and
  never touch the UI toolkit (the overlay builds its panel in ``tick``).
* ``hide`` / ``tick`` run on the main thread.
* ``set_level`` runs on the recording worker thread (must be lock-protected).
* ``status`` / ``on_partial`` may arrive on either thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RecordingDisplay(ABC):
    """What the dictation app needs from any recording display."""

    @abstractmethod
    def status(self, message: str) -> None:
        """Show a status line (e.g. "Listening...", errors, final text)."""

    @abstractmethod
    def on_partial(self, text: str) -> None:
        """Show the newest cumulative transcript."""

    def show(self) -> None:
        """Make the display visible (start of a recording). No-op by default."""

    def hide(self) -> None:
        """Hide the display and release visible state (end of recording)."""

    def tick(self) -> None:
        """Advance one animation frame. Called on the main thread; no-op default."""

    def set_level(self, block) -> None:
        """Accept one raw PCM block for level visualization. No-op by default."""
