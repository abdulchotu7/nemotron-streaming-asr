"""Global hotkey with press/release detection, hold or toggle mode.

``GlobalHotkey`` is a small abstract interface so the backend can be swapped
(e.g. a Quartz CGEventTap or a menu-bar Swift app) without touching the rest
of the application. The default implementation uses ``pynput``'s global
keyboard listener.

Hold mode (default): recording starts when every configured modifier (and
optional trigger key) is held; recording stops when any of them is released.

Toggle mode (``toggle=True``): a fresh press of the trigger key flips between
recording (``on_press``) and stopping (``on_release``); releasing the key alone
does nothing, and key auto-repeat is ignored.

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
    """Hotkey via pynput's global keyboard listener, hold or toggle mode.

    Args:
        modifiers: modifier families that must be held, e.g. ``("cmd", "option")``.
        key: optional single trigger key (e.g. ``"alt_r"`` = right Option,
            ``"f10"``) that must also be held (hold mode) or that toggles
            recording (toggle mode).
        toggle: if False (default), hold-to-talk: ``on_press`` fires when the
            combo is held, ``on_release`` when anything is released. If True,
            tap-to-toggle: a fresh press of the trigger key flips between
            ``on_press`` (start) and ``on_release`` (stop); auto-repeat is
            ignored. Toggle mode requires a trigger key.
    """

    def __init__(self, modifiers: Sequence[str] = ("cmd", "option"),
                 key: Optional[str] = None, toggle: bool = False):
        super().__init__()
        self.modifiers = tuple(modifiers)
        self.key = key
        self.toggle = toggle

        if toggle and key is None:
            raise ValueError(
                "toggle mode requires a trigger key (e.g. 'alt_r' for right "
                "Option, 'f10', 'v')"
            )

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
        self._pressed.clear()

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
        if token is None:
            return
        fresh = token not in self._pressed
        self._pressed.add(token)

        if self.toggle:
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
            return

        if self._ready() and not self._active:
            self._active = True
            if self.on_press is not None:
                self.on_press()

    def _on_release(self, key) -> None:
        token = _key_token(key)
        if token is not None:
            self._pressed.discard(token)
        if self.toggle:
            return  # stopping is driven by the next trigger press, not a release
        if self._active and not self._ready():
            self._active = False
            if self.on_release is not None:
                self.on_release()
