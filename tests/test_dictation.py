"""Dictation app: hotkey logic, transcript controller, insertion guards, and
end-to-end recording lifecycle with a fake recorder (no microphone/hardware).
"""

import threading
import time

import pytest


def _result(text):
    class R:
        pass

    r = R()
    r.text = text
    return r


# ------------------------------------------------------------------ transcript
def test_transcript_controller_tracks_newest():
    from nemotron_streaming_asr.apps.dictation.transcript import (
        LiveTranscriptController,
    )

    seen = []
    tc = LiveTranscriptController(on_update=seen.append)

    tc.update(_result("I'm"))
    tc.update(_result("I'm sure"))
    assert tc.current_text == "I'm sure"
    assert seen == ["I'm", "I'm sure"]

    tc.update(_result("I'm sure"))  # unchanged -> no re-notify
    assert seen == ["I'm", "I'm sure"]

    tc.clear()
    assert tc.current_text == ""
    assert seen[-1] == ""


def test_transcript_controller_multiple_listeners():
    from nemotron_streaming_asr.apps.dictation.transcript import (
        LiveTranscriptController,
    )

    a, b = [], []
    tc = LiveTranscriptController(on_update=a.append)
    tc.add_listener(b.append)
    tc.update(_result("hello"))
    assert a == ["hello"] and b == ["hello"]


# --------------------------------------------------------------------- hotkey
def test_hotkey_modifier_combo_hold():
    from pynput import keyboard

    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    events = []
    hk = PynputGlobalHotkey(modifiers=("cmd", "option"))
    hk.on_press = lambda: events.append("press")
    hk.on_release = lambda: events.append("release")

    hk._on_press(keyboard.Key.cmd)  # only one modifier -> not yet
    assert events == []
    hk._on_press(keyboard.Key.alt)  # both held -> press
    assert events == ["press"]
    hk._on_press(keyboard.Key.cmd)  # key repeat while held -> ignored
    assert events == ["press"]
    hk._on_release(keyboard.Key.cmd)  # any modifier released -> release
    assert events == ["press", "release"]

    # Re-hold starts again.
    hk._on_press(keyboard.Key.cmd)
    hk._on_press(keyboard.Key.alt)
    assert events == ["press", "release", "press"]


def test_hotkey_single_key_hold():
    from pynput import keyboard

    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    events = []
    hk = PynputGlobalHotkey(modifiers=(), key="f10")
    hk.on_press = lambda: events.append("press")
    hk.on_release = lambda: events.append("release")

    hk._on_press(keyboard.Key.f10)
    assert events == ["press"]
    hk._on_press(keyboard.Key.f10)  # repeat ignored
    assert events == ["press"]
    hk._on_release(keyboard.Key.f10)
    assert events == ["press", "release"]


def test_hotkey_requires_trigger_key():
    from pynput import keyboard

    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    events = []
    hk = PynputGlobalHotkey(modifiers=("cmd",), key="v")
    hk.on_press = lambda: events.append("press")
    hk.on_release = lambda: events.append("release")

    hk._on_press(keyboard.Key.cmd)
    assert events == []  # 'v' not held
    hk._on_press(keyboard.KeyCode.from_char("v"))
    assert events == ["press"]
    hk._on_release(keyboard.Key.cmd)
    assert events == ["press", "release"]


def test_hotkey_toggle_right_option():
    """Right Option tap-to-toggle: tap starts, tap again stops; repeat/release
    alone never fires, and the left Option key does not trigger it."""
    from pynput import keyboard

    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    events = []
    hk = PynputGlobalHotkey(modifiers=(), key="alt_r", toggle=True)
    hk.on_press = lambda: events.append("press")
    hk.on_release = lambda: events.append("release")

    hk._on_press(keyboard.Key.alt_r)  # first tap -> start
    assert events == ["press"]
    hk._on_press(keyboard.Key.alt_r)  # auto-repeat while held -> ignored
    assert events == ["press"]
    hk._on_release(keyboard.Key.alt_r)  # releasing alone never stops
    assert events == ["press"]
    hk._on_press(keyboard.Key.alt)  # left option is a different key -> ignored
    assert events == ["press"]

    hk._on_press(keyboard.Key.alt_r)  # second tap -> stop
    assert events == ["press", "release"]
    hk._on_release(keyboard.Key.alt_r)
    hk._on_press(keyboard.Key.alt_r)  # third tap -> start again
    assert events == ["press", "release", "press"]


def test_hotkey_toggle_with_modifiers():
    """Toggle also works with a modifier combo as long as a trigger key is set."""
    from pynput import keyboard

    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    events = []
    hk = PynputGlobalHotkey(modifiers=("cmd",), key="f10", toggle=True)
    hk.on_press = lambda: events.append("press")
    hk.on_release = lambda: events.append("release")

    hk._on_press(keyboard.Key.cmd)  # modifier alone is not the trigger
    assert events == []
    hk._on_press(keyboard.Key.f10)  # combo complete -> start
    assert events == ["press"]
    hk._on_release(keyboard.Key.cmd)  # releasing a modifier does not stop
    assert events == ["press"]
    hk._on_release(keyboard.Key.f10)  # lift the trigger between taps
    hk._on_press(keyboard.Key.cmd)  # re-hold combo...
    hk._on_press(keyboard.Key.f10)  # ...and press trigger again -> stop
    assert events == ["press", "release"]


def test_hotkey_toggle_requires_trigger_key():
    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    with pytest.raises(ValueError):
        PynputGlobalHotkey(modifiers=("option",), toggle=True)


# -------------------------------------------------------------- text insertion
def test_text_insertion_empty_guard():
    """Empty text must not touch the clipboard or post any key event."""
    from nemotron_streaming_asr.apps.dictation.text_insertion import (
        TextInsertionService,
    )

    svc = TextInsertionService()
    svc.insert("")
    svc.insert(None)
    svc.insert("   ")


def test_text_insertion_clipboard_roundtrip():
    """Clipboard snapshot/set/restore must preserve the previous content."""
    from nemotron_streaming_asr.apps.dictation.text_insertion import (
        TextInsertionService,
    )

    original = TextInsertionService._snapshot_clipboard()
    try:
        TextInsertionService._set_clipboard("transcript payload")
        assert TextInsertionService._snapshot_clipboard() == "transcript payload"
    finally:
        TextInsertionService._restore_clipboard(original)
    assert TextInsertionService._snapshot_clipboard() == original


# ----------------------------------------------------------------- recorder
def test_microphone_recorder_poll_empty_without_device():
    """poll() on a never-started recorder returns None (no device needed)."""
    from nemotron_streaming_asr.apps.dictation.microphone import MicrophoneRecorder

    rec = MicrophoneRecorder()
    assert rec.poll(timeout=0.01) is None
    assert rec.drain() == []


# ---------------------------------------------------------------------- app
class _FakeRecorder:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.blocks = []  # blocks handed to the session by the worker

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        pass

    def poll(self, timeout=0.02):
        return None  # nothing captured: the worker just waits for stop

    def drain(self):
        return []


def test_app_full_lifecycle_with_fake_recorder(tiny_model, seeded_audio):
    """press -> fresh session -> worker -> stop -> finish -> ready."""
    from nemotron_streaming_asr.apps.dictation.app import DictationApp
    from nemotron_streaming_asr.apps.dictation.microphone import MicrophoneRecorder

    recorder = _FakeRecorder()
    ui_lines = []
    ui = type("UI", (), {"status": ui_lines.append,
                          "on_partial": lambda self, t: None})()

    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insert=False, ui=ui)
    assert app._recording is False

    app.start_recording()
    assert app._recording is True
    assert recorder.started
    assert app._session is not None

    app.stop_recording()
    assert app._worker is not None
    app._worker.join(timeout=10)

    assert app._recording is False
    assert recorder.stopped
    assert app._session is None  # session destroyed after the recording
    assert any("Ready for next recording" in l for l in ui_lines)


def test_recording_produces_live_transcript_updates(tiny_model, seeded_audio):
    """A recording with real audio blocks yields cumulative partials via the
    transcript controller (no microphone: blocks are injected directly)."""
    from nemotron_streaming_asr.apps.dictation.app import DictationApp
    from nemotron_streaming_asr.apps.dictation.transcript import (
        LiveTranscriptController,
    )

    recorder = MicrophoneRecorderNoDevice(seeded_audio)
    ui_lines = []
    quiet_ui = type("UI", (), {"status": ui_lines.append,
                               "on_partial": lambda self, t: None})()
    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insert=False, ui=quiet_ui)
    # Use a plain (non-console) UI so we can inspect partials.
    partials = []
    app.transcript.add_listener(partials.append)

    app.start_recording()
    app.stop_recording()
    app._worker.join(timeout=30)

    assert recorder.started and recorder.stopped
    assert len(partials) > 0, "expected cumulative partial transcripts"
    assert app.transcript.current_text != ""
    # partials are cumulative: each one is a prefix of the next
    for prev, nxt in zip(partials, partials[1:]):
        assert nxt.startswith(prev) or nxt == prev


class MicrophoneRecorderNoDevice:
    """Drives the recording worker with real audio blocks (no microphone)."""

    def __init__(self, audio):
        self._audio = audio
        self._i = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        pass

    def poll(self, timeout=0.02):
        if self._i >= self._audio.shape[0]:
            time.sleep(0.005)
            return None
        block = self._audio[self._i : self._i + 320]
        self._i += 320
        return block

    def drain(self):
        blocks = []
        while self._i < self._audio.shape[0]:
            blocks.append(self._audio[self._i : self._i + 320])
            self._i += 320
        return blocks
