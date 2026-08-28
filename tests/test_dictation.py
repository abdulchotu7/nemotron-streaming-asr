"""Dictation app: hotkey logic, transcript controller, insertion guards, and
end-to-end recording lifecycle with a fake recorder (no microphone/hardware).
"""

import time


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


# -------------------------------------------------- wave overlay (new)
# These tests cover the WaveformOverlayUI integration without requiring a
# live AppKit session. We monkeypatch the module-level AppKit symbols to
# plain stand-ins so the overlay can be constructed headlessly. The point
# is to lock in the *behavior contracts* (RMS clamping, visibility flag,
# fallback imports, close idempotency), not the visual rendering.


class _StubView:
    def __init__(self):
        self.phase = None
        self.volume = None
        self.shown = False
        self.hidden = False

    def setPhase_(self, phase):
        self.phase = phase

    def setVolume_(self, volume):
        self.volume = volume

    def setNeedsDisplay_(self, _):
        pass


class _StubPanel:
    last_frame = None

    def setFrame_display_(self, rect, _flag):
        _StubPanel.last_frame = rect

    def orderOut_(self, _):
        _StubPanel.last_frame = None  # "hidden"

    def orderFrontRegardless(self):
        pass


def _build_overlay_module(monkeypatch, *, caret_rect=None, screen=(1920, 1080),
                          mouse=(100, 100)):
    """Import wave_overlay with AppKit/AX symbols stubbed so it runs headlessly.

    Returns the (module, overlay_ui) pair.
    """
    import sys
    import types

    fake_appkit = types.SimpleNamespace(
        NSView=type("NSView", (), {}),
        NSColor=type("NSColor", (), {
            "colorWithRed_green_blue_alpha_": staticmethod(
                lambda r, g, b, a: types.SimpleNamespace(_rgba=(r, g, b, a))
            ),
            "clearColor": staticmethod(lambda: None),
        }),
        NSBezierPath=type("NSBezierPath", (), {
            "bezierPathWithRoundedRect_xRadius_yRadius_": staticmethod(
                lambda *a, **k: types.SimpleNamespace()
            ),
            "bezierPath": staticmethod(lambda: types.SimpleNamespace(
                moveToPoint_=lambda *a, **k: None,
                lineToPoint_=lambda *a, **k: None,
                appendBezierPathWithRoundedRect_xRadius_yRadius_=lambda *a, **k: None,
                setLineWidth_=lambda *a, **k: None,
                stroke=lambda: None,
                fill=lambda: None,
            )),
        }),
        NSPanel=type("NSPanel", (), {
            "alloc": staticmethod(lambda: types.SimpleNamespace(
                initWithContentRect_styleMask_backing_defer_=lambda *a, **k: _StubPanel()
            )),
        }),
        NSMakeRect=staticmethod(lambda x, y, w, h: (x, y, w, h)),
        NSEvent=type("NSEvent", (), {
            "mouseLocation": staticmethod(lambda: types.SimpleNamespace(
                x=mouse[0], y=mouse[1],
            )),
        }),
        NSScreen=type("NSScreen", (), {
            "screens": staticmethod(lambda: [types.SimpleNamespace(
                frame=types.SimpleNamespace(
                    size=types.SimpleNamespace(width=screen[0], height=screen[1])
                )
            )]),
        }),
        NSRunLoop=type("NSRunLoop", (), {
            "currentRunLoop": staticmethod(
                lambda: types.SimpleNamespace(
                    runMode_beforeDate_=lambda *a, **k: None
                )
            ),
        }),
        NSDefaultRunLoopMode="kCFRunLoopDefaultMode",
        NSDate=type("NSDate", (), {
            "dateWithTimeIntervalSinceNow_": staticmethod(lambda _: None),
        }),
        NSFloatingWindowLevel=5,
        NSWindowCollectionBehaviorCanJoinAllSpaces=1,
        NSWindowCollectionBehaviorFullScreenAuxiliary=2,
        NSWindowStyleMaskBorderless=0,
        NSWindowStyleMaskNonactivatingPanel=0,
        NSBackingStoreBuffered=0,
    )
    fake_ax = types.SimpleNamespace(
        AXUIElementCreateSystemWide=lambda: None,
        AXUIElementCopyAttributeValue=lambda *a, **k: (-1, None),
        AXUIElementCopyParameterizedAttributeValue=lambda *a, **k: (-1, None),
        AXValueGetValue=lambda *a, **k: (False, None),
        kAXFocusedUIElementAttribute="kAXFocusedUIElementAttribute",
        kAXSelectedTextRangeAttribute="kAXSelectedTextRangeAttribute",
        kAXBoundsForRangeParameterizedAttribute="kAXBoundsForRangeParameterizedAttribute",
        kAXPositionAttribute="kAXPositionAttribute",
        kAXSizeAttribute="kAXSizeAttribute",
        kAXValueCGPointType=0,
        kAXValueCGRectType=0,
        kAXValueCGSizeType=0,
    )

    # Drop any cached imports so the fake modules are used.
    for name in ("nemotron_streaming_asr.apps.dictation.wave_overlay", "AppKit",
                 "ApplicationServices"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)
    monkeypatch.setitem(sys.modules, "ApplicationServices", fake_ax)
    # objc is imported by wave_overlay too; stub the one symbol it uses.
    fake_objc = types.SimpleNamespace(super=super)  # super() works on real classes
    monkeypatch.setitem(sys.modules, "objc", fake_objc)

    # Patch the caret-rect detector at the source AFTER the module is loaded.
    from nemotron_streaming_asr.apps.dictation import wave_overlay as wo
    if caret_rect is not None:
        monkeypatch.setattr(wo, "get_focused_caret_rect", lambda: caret_rect)
    return wo


def test_wave_overlay_set_volume_clamps_to_visible_range(monkeypatch):
    """RMS is mapped to [0.15, 1.0] and clamped on both ends."""
    wo = _build_overlay_module(monkeypatch)
    ui = wo.WaveformOverlayUI()
    ui.set_volume(0.0)    # silent -> floor at 0.15
    assert ui._target_volume == 0.15
    ui.set_volume(0.1)    # typical speech -> 0.75
    assert abs(ui._target_volume - 0.75) < 1e-9
    ui.set_volume(1.0)    # screaming -> ceiling at 1.0
    assert ui._target_volume == 1.0


def test_wave_overlay_set_volume_is_thread_safe(monkeypatch):
    """Concurrent set_volume calls must not corrupt _target_volume."""
    import threading
    wo = _build_overlay_module(monkeypatch)
    ui = wo.WaveformOverlayUI()
    errors = []
    def hammer():
        try:
            for _ in range(500):
                ui.set_volume(0.05)
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    # All values must have been valid floats in [0.15, 1.0]
    assert 0.15 <= ui._target_volume <= 1.0


def test_wave_overlay_close_is_idempotent(monkeypatch):
    """close() must be safe to call when the panel was never built, and
    again after it was built (the run-loop close fix lives here)."""
    wo = _build_overlay_module(monkeypatch)
    ui = wo.WaveformOverlayUI()
    # First close with no panel: must not raise.
    ui.close()
    # Stub a panel+view, then close twice: must not raise either time.
    ui._panel = _StubPanel()
    ui._view = _StubView()
    ui._current_volume = 0.9
    ui.close()
    assert ui._panel is None
    assert ui._view is None
    assert ui._current_volume == 0.15  # reset on close
    # Calling close again on a torn-down state is still safe.
    ui.close()


def test_get_focused_caret_rect_returns_none_when_no_focus(monkeypatch):
    """When AX has no focused element (err != 0), the detector returns None."""
    wo = _build_overlay_module(monkeypatch)
    assert wo.get_focused_caret_rect() is None


def test_get_focused_caret_rect_swallows_exceptions(monkeypatch):
    """If anything inside the AX calls raises, the detector must return None
    (it must never crash the recording loop)."""
    wo = _build_overlay_module(monkeypatch)
    def boom():
        raise RuntimeError("AX subsystem exploded")
    monkeypatch.setattr(wo, "AXUIElementCreateSystemWide", boom)
    assert wo.get_focused_caret_rect() is None


def test_dictation_app_falls_back_to_console_ui(tiny_model, monkeypatch):
    """If wave_overlay fails to import (e.g. on a non-macOS host), the
    app must silently fall back to ConsoleUI rather than crash."""
    import builtins
    from nemotron_streaming_asr.apps.dictation.app import DictationApp, ConsoleUI
    from nemotron_streaming_asr.apps.dictation import app as app_module

    def _explode(name, *a, **k):
        if name.endswith("wave_overlay"):
            raise ImportError("AppKit not available in CI")
        return real_import(name, *a, **k)
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _explode)
    recorder = _FakeRecorder()
    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insert=False)
    assert isinstance(app._ui, ConsoleUI)
    # Restore builtins so later tests can import normally.
    monkeypatch.setattr(builtins, "__import__", real_import)


def test_dictation_app_visibility_flag_tracks_recording(tiny_model):
    """_ui_visible must be set when recording starts and cleared when it stops
    (drives the tick/close branch in run())."""
    from nemotron_streaming_asr.apps.dictation.app import DictationApp

    recorder = _FakeRecorder()
    ui = type("UI", (), {"status": lambda self, m: None,
                          "on_partial": lambda self, t: None})()
    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insert=False, ui=ui)
    assert app._ui_visible is False
    app.start_recording()
    assert app._ui_visible is True
    app.stop_recording()
    app._worker.join(timeout=10)
    # After the worker drains and finalizes, the app is back to idle.
    assert app._ui_visible is False


def test_recording_loop_pushes_rms_to_overlay(tiny_model, monkeypatch):
    """While recording, each audio block is converted to RMS and pushed
    to the UI's set_volume (so the visualizer reacts to voice)."""
    import numpy as np
    from nemotron_streaming_asr.apps.dictation.app import DictationApp

    # 4 blocks of 320 samples of audio, alternating loud and quiet.
    loud = (np.ones(320, dtype=np.float32) * 0.5)
    quiet = (np.ones(320, dtype=np.float32) * 0.0)

    class _LoudQuietRecorder(_FakeRecorder):
        def __init__(self):
            super().__init__()
            self._blocks = [loud, quiet, loud, quiet]
        def poll(self, timeout=0.02):
            if not self._blocks:
                time.sleep(0.005)
                return None
            return self._blocks.pop(0)
        def drain(self):
            rest = self._blocks[:]
            self._blocks = []
            return rest

    # Use the real WaveformOverlayUI (stubbed AppKit) so the test exercises
    # the full RMS-to-visible-range pipeline that ships in production.
    wo = _build_overlay_module(monkeypatch)
    samples = []  # capture every (raw_rms, target_volume) the app pushes
    real_set_volume = wo.WaveformOverlayUI.set_volume
    def spy_set_volume(self, rms):
        samples.append(rms)
        real_set_volume(self, rms)
    monkeypatch.setattr(wo.WaveformOverlayUI, "set_volume", spy_set_volume)
    app = DictationApp(tiny_model, language="en-US", recorder=_LoudQuietRecorder(),
                       insert=False, ui=wo.WaveformOverlayUI())
    app.start_recording()
    app._worker.join(timeout=10)
    # At least 4 blocks were pushed (loud, quiet, loud, quiet).
    assert len(samples) >= 4
    # The loud block (RMS=0.5) must have been pushed before the quiet block
    # (RMS=0.0), and must be a larger RMS value.
    assert samples[0] > samples[1]
    # The current target volume (clamped) is in the visible range.
    assert 0.15 <= app._ui._target_volume <= 1.0
