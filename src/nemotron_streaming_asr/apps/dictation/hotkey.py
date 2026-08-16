"""Tap-to-toggle global hotkey (right Option by default).

``GlobalHotkey`` is a small abstract interface so the backend can be swapped
(e.g. a Quartz CGEventTap or a menu-bar Swift app) without touching the rest
of the application. The default implementation uses ``pynput``'s global
keyboard listener.

Toggle mode: a fresh press of the trigger key flips between recording
(``on_press``) and stopping (``on_release``); releasing the key alone does
nothing, and key auto-repeat is ignored.

Requires macOS **Input Monitoring / Accessibility** permission for the app
(or terminal) running it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class GlobalHotkey(ABC):
    """Abstract global hotkey exposing press/release callbacks."""

    def __init__(self):
        self.on_press: Optional[Callable[[], None]] = None
        self.on_release: Optional[Callable[[], None]] = None

    @abstractmethod
    def start(self) -> None:
        """Begin listening for the hotkey."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening; callbacks are not invoked afterwards."""


def _key_token(key) -> Optional[str]:
    """Normalize a pynput key to a comparable token."""
    from pynput import keyboard

    if isinstance(key, keyboard.Key):
        return key.name
    if isinstance(key, keyboard.KeyCode):
        return key.char
    return None


def _resolve_key(name: str):
    """Resolve 'f10' -> keyboard.Key.f10, 'v' -> KeyCode('v')."""
    from pynput import keyboard

    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    return getattr(keyboard.Key, name)


class PynputGlobalHotkey(GlobalHotkey):
    """Tap-to-toggle hotkey via pynput's global keyboard listener.

    Args:
        key: the trigger key that toggles recording, e.g. ``"alt_r"`` (right
            Option, the default) or ``"f10"``. A fresh press flips between
            ``on_press`` (start) and ``on_release`` (stop); releasing the key
            alone never stops, and auto-repeat is ignored.
    """

    def __init__(self, key: str = "alt_r"):
        super().__init__()
        self.key = key
        self._trigger_token = _key_token(_resolve_key(key))

        self._pressed: set = set()
        self._active = False
        self._listener = None

    # ------------------------------------------------------------- interface
    def start(self) -> None:
        from pynput import keyboard

        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._active = False
        self._pressed.clear()

    # ------------------------------------------------------------- internals
    def _on_press(self, key) -> None:
        token = _key_token(key)
        if token is None:
            return
        fresh = token not in self._pressed
        self._pressed.add(token)

        # Tap-to-toggle: only a fresh press of the trigger key flips state.
        if not fresh or token != self._trigger_token:
            return
        if self._active:
            self._active = False
            if self.on_release is not None:
                self.on_release()
        else:
            self._active = True
            if self.on_press is not None:
                self.on_press()

    def _on_release(self, key) -> None:
        token = _key_token(key)
        if token is not None:
            self._pressed.discard(token)
        # Stopping is driven by the next trigger press, not a release.
