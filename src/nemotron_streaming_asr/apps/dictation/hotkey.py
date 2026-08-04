"""Global hold-to-talk hotkey with press/release detection.

``GlobalHotkey`` is a small abstract interface so the backend can be swapped
(e.g. a Quartz CGEventTap or a menu-bar Swift app) without touching the rest
of the application. The default implementation uses ``pynput``'s global
keyboard listener.

Recording starts when every configured modifier (and optional trigger key) is
held; recording stops when any of them is released. Key repeat is ignored
while the hotkey is active.

Requires macOS **Input Monitoring / Accessibility** permission for the app
(or terminal) running it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence


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


# pynput key names that count as each modifier family (macOS: option == alt).
_MODIFIER_KEYS = {
    "cmd": {"cmd", "cmd_r"},
    "ctrl": {"ctrl", "ctrl_r"},
    "option": {"alt", "alt_r"},
    "alt": {"alt", "alt_r"},
    "shift": {"shift", "shift_r"},
}


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
    """Hold-to-talk hotkey via pynput's global keyboard listener.

    Args:
        modifiers: modifier families that must be held, e.g. ``("cmd", "option")``.
        key: optional single trigger key (e.g. ``"f10"``) that must also be held.
    """

    def __init__(self, modifiers: Sequence[str] = ("cmd", "option"),
                 key: Optional[str] = None):
        super().__init__()
        self.modifiers = tuple(modifiers)
        self.key = key

        self._families = [
            _MODIFIER_KEYS.get(name, {name}) for name in self.modifiers
        ]
        self._trigger_token = None
        if key is not None:
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

    # ------------------------------------------------------------- internals
    def _ready(self) -> bool:
        # Every modifier family needs at least one physical key held
        # (e.g. left or right Cmd), not every variant.
        for family in self._families:
            if not (family & self._pressed):
                return False
        if self._trigger_token is not None and self._trigger_token not in self._pressed:
            return False
        return True

    def _on_press(self, key) -> None:
        token = _key_token(key)
        if token is not None:
            self._pressed.add(token)
        if self._ready() and not self._active:
            self._active = True
            if self.on_press is not None:
                self.on_press()

    def _on_release(self, key) -> None:
        token = _key_token(key)
        if token is not None:
            self._pressed.discard(token)
        if self._active and not self._ready():
            self._active = False
            if self.on_release is not None:
                self.on_release()
