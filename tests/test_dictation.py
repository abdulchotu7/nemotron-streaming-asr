"""Dictation app: hotkey logic, transcript controller, insertion guards, and
end-to-end recording lifecycle with a fake recorder (no microphone/hardware).
"""

import time

import numpy as np


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
def test_hotkey_toggle_right_option():
    """Right Option tap-to-toggle: tap starts, tap again stops; repeat/release
    alone never fires, and the left Option key does not trigger it."""
    from pynput import keyboard

    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    events = []
    hk = PynputGlobalHotkey(key="alt_r")
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


def test_hotkey_default_is_right_option_toggle():
    """The default hotkey is exactly right-Option tap-to-toggle."""
    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    hk = PynputGlobalHotkey()
    assert hk.key == "alt_r"
    assert hk._trigger_token == "alt_r"


def test_hotkey_other_trigger_key():
    """A different trigger key (f10) toggles only on fresh f10 presses."""
    from pynput import keyboard

    from nemotron_streaming_asr.apps.dictation.hotkey import PynputGlobalHotkey

    events = []
    hk = PynputGlobalHotkey(key="f10")
    hk.on_press = lambda: events.append("press")
    hk.on_release = lambda: events.append("release")

    hk._on_press(keyboard.Key.alt_r)  # unrelated key -> ignored
    assert events == []
    hk._on_press(keyboard.Key.f10)  # fresh f10 -> start
    assert events == ["press"]
    hk._on_press(keyboard.Key.f10)  # repeat while held -> ignored
    assert events == ["press"]
    hk._on_release(keyboard.Key.f10)  # release alone -> still recording
    assert events == ["press"]
    hk._on_press(keyboard.Key.f10)  # fresh f10 -> stop
    assert events == ["press", "release"]


def test_rapid_stop_then_start_is_not_swallowed(tiny_model):
    """A start tap landing while the previous recording is still finalizing
    must wait for it and start a fresh recording (not be dropped)."""
    from nemotron_streaming_asr.apps.dictation.app import DictationApp

    class SlowRecorder(_FakeRecorder):
        def drain(self):
            time.sleep(0.1)  # keep the worker alive past the next start tap
            return []

    recorder = SlowRecorder()
    ui_lines = []
    quiet_ui = type("UI", (), {"status": ui_lines.append,
                               "on_partial": lambda self, t: None})()
    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insert=False, ui=quiet_ui)

    app.start_recording()
    app.stop_recording()
    worker1, session1 = app._worker, app._session

    app.start_recording()  # immediately: must not be swallowed
    assert app._recording is True
    assert app._worker is not worker1  # a fresh recording actually started
    assert app._session is not session1

    app.stop_recording()
    app._worker.join(timeout=10)
    assert app._recording is False


def test_auto_stop_ends_recording(tiny_model):
    """Speech followed by stop_silence_s of silence ends the recording
    without any hotkey tap."""
    from nemotron_streaming_asr.apps.dictation.app import DictationApp
    from nemotron_streaming_asr.apps.dictation.vad import EnergyVAD

    clock = {"t": 0.0}
    loud = (np.random.default_rng(0).standard_normal(320) * 0.05).astype(
        np.float32
    )
    silence = np.zeros(320, dtype=np.float32)
    # ~0.24 s of speech (12 loud blocks, safely above min_speech_s) then
    # 4 s of silence.
    blocks = [loud] * 12 + [silence] * 200

    class ClockRecorder:
        def __init__(self):
            self.started = self.stopped = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            pass

        def poll(self, timeout=0.02):
            clock["t"] += timeout
            return blocks.pop(0) if blocks else None

        def drain(self):
            return []

    recorder = ClockRecorder()
    ui_lines = []
    quiet_ui = type("UI", (), {"status": ui_lines.append,
                               "on_partial": lambda self, t: None})()
    vad = EnergyVAD(now_fn=lambda: clock["t"])
    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insert=False, ui=quiet_ui, vad=vad)

    app.start_recording()
    app._worker.join(timeout=30)

    assert app._recording is False  # ended without a stop tap
    assert recorder.stopped
    assert any("auto-stop" in line for line in ui_lines)


def test_overlay_ui_shows_live_transcript():
    """OverlayUI queues text from any thread and renders it on tick()."""
    import pytest

    from nemotron_streaming_asr.apps.dictation.overlay import OverlayUI

    ui = OverlayUI()
    if ui._ensure_panel()[0] is None:
        pytest.skip("no window server available")
    ui.on_partial("He hoped there would be stew")
    ui.tick()
    assert ui._label.stringValue() == "He hoped there would be stew"
    ui.status("(auto-stop: silence detected)")
    ui.on_partial("He hoped there would be stew, turnips and carrots")
    ui.tick()
    assert ui._label.stringValue() == "He hoped there would be stew, turnips and carrots"
    ui._panel.orderOut_(None)


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
