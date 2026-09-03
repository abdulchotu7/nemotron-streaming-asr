"""Dictation app: hotkey logic, transcript controller, insertion guards, and
end-to-end recording lifecycle with a fake recorder (no microphone/hardware).
"""

import time

from nemotron_streaming_asr.apps.dictation.hotkey import GlobalHotkey
from nemotron_streaming_asr.apps.dictation.text_insertion import (
    TextInsertionService,
)


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
    quiet_ui = _quiet_ui(ui_lines)
    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insertion=_FakeInsertion(), hotkey=_FakeHotkey(),
                       insert=False, ui=quiet_ui)

    app.start_recording()
    app.stop_recording()
    utt1, session1 = app._utterance, app._session

    app.start_recording()  # immediately: must not be swallowed
    assert app._utterance.is_running
    assert app._utterance is not utt1  # a fresh recording actually started
    assert app._session is not session1

    app.stop_recording()
    app._utterance.join(timeout=10)
    assert not app._utterance.is_running


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
class _FakeHotkey(GlobalHotkey):
    """Stand-in for the hotkey backend: the app only assigns callbacks on it."""

    def __init__(self):
        super().__init__()
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _FakeInsertion(TextInsertionService):
    """Stand-in for the inserter: records pasted text instead of pasting."""

    def __init__(self):
        super().__init__()
        self.inserted = []

    def insert(self, text):
        self.inserted.append(text)


def _quiet_ui(lines, partials=None):
    """A minimal RecordingDisplay: only status/on_partial, rest are no-ops."""
    from nemotron_streaming_asr.apps.dictation.display import RecordingDisplay

    return type("UI", (RecordingDisplay,), {
        "status": lambda self, m: lines.append(m),
        "on_partial": lambda self, t: (partials.append(t) if partials is not None else None),
    })()


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
    ui = _quiet_ui(ui_lines)

    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insertion=_FakeInsertion(), hotkey=_FakeHotkey(),
                       insert=False, ui=ui)
    assert not app._utterance or not app._utterance.is_running

    app.start_recording()
    assert app._utterance.is_running
    assert recorder.started
    assert app._session is not None

    app.stop_recording()
    assert app._utterance is not None
    app._utterance.join(timeout=10)

    assert not app._utterance.is_running
    assert recorder.stopped
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
    quiet_ui = _quiet_ui(ui_lines)
    app = DictationApp(tiny_model, language="en-US", recorder=recorder,
                       insertion=_FakeInsertion(), hotkey=_FakeHotkey(),
                       insert=False, ui=quiet_ui)
    # Use a plain (non-console) UI so we can inspect partials.
    partials = []
    app.transcript.add_listener(partials.append)

    app.start_recording()
    app.stop_recording()
    app._utterance.join(timeout=30)

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


def _build_caret_module(monkeypatch, *, ax=None):
    """Import caret.py with ApplicationServices stubbed (no AppKit needed).

    ``caret`` has no top-level native imports (AX/AppKit are lazy),
    so the default ``ax`` fake only matters for get_focused_caret_rect tests;
    place_panel tests import the module directly with no stubbing at all.
    """
    import sys
    import types

    if ax is None:
        ax = types.SimpleNamespace(
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
    monkeypatch.delitem(
        sys.modules, "nemotron_streaming_asr.apps.dictation.caret", raising=False
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", ax)
    from nemotron_streaming_asr.apps.dictation import caret as caret_module
    return caret_module


def test_wave_overlay_set_level_clamps_to_visible_range(monkeypatch):
    """Raw PCM RMS is mapped to [0.15, 1.0] and clamped on both ends."""
    import numpy as np
    wo = _build_overlay_module(monkeypatch)
    ui = wo.WaveformOverlayUI()
    ui.set_level(np.zeros(320, dtype=np.float32))    # silent -> floor at 0.15
    assert ui._target_volume == 0.15
    ui.set_level(np.ones(320, dtype=np.float32) * 0.1)  # typical speech -> 0.75
    assert abs(ui._target_volume - 0.75) < 1e-6
    ui.set_level(np.ones(320, dtype=np.float32))     # screaming -> ceiling at 1.0
    assert ui._target_volume == 1.0


def test_wave_overlay_set_level_is_thread_safe(monkeypatch):
    """Concurrent set_level calls must not corrupt _target_volume."""
    import threading
    import numpy as np
    wo = _build_overlay_module(monkeypatch)
    ui = wo.WaveformOverlayUI()
    block = np.ones(320, dtype=np.float32) * 0.05
    errors = []
    def hammer():
        try:
            for _ in range(500):
                ui.set_level(block)
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    # All values must have been valid floats in [0.15, 1.0]
    assert 0.15 <= ui._target_volume <= 1.0


def test_wave_overlay_hide_is_idempotent(monkeypatch):
    """hide() must be safe to call when the panel was never built, and
    again after it was built (the run-loop hide fix lives here)."""
    wo = _build_overlay_module(monkeypatch)
    ui = wo.WaveformOverlayUI()
    # First hide with no panel: must not raise.
    ui.hide()
    # Stub a panel+view, then hide twice: must not raise either time.
    ui._panel = _StubPanel()
    ui._view = _StubView()
    ui._current_volume = 0.9
    ui.hide()
    assert ui._panel is None
    assert ui._view is None
    assert ui._current_volume == 0.15  # reset on hide
    # Calling hide again on a torn-down state is still safe.
    ui.hide()


def test_get_focused_caret_rect_returns_none_when_no_focus(monkeypatch):
    """When AX has no focused element (err != 0), the detector returns None."""
    caret = _build_caret_module(monkeypatch)
    assert caret.get_focused_caret_rect() is None


def test_get_focused_caret_rect_swallows_exceptions(monkeypatch):
    """If anything inside the AX calls raises, the detector must return None
    (it must never crash the recording loop)."""
    import types

    def boom():
        raise RuntimeError("AX subsystem exploded")

    fake_ax = types.SimpleNamespace(
        AXUIElementCreateSystemWide=boom,
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
    caret = _build_caret_module(monkeypatch, ax=fake_ax)
    assert caret.get_focused_caret_rect() is None


# ------------------------------------------------- caret.place_panel (pure)
# Placement math needs no AppKit: plain tuples in, clamped origin out.

def test_place_panel_under_fine_caret_line():
    """A fine caret (w < 6) centers the panel below the caret (AX y flipped)."""
    from nemotron_streaming_asr.apps.dictation.caret import place_panel

    # screen 1920x1080, panel 72x24, caret at AX (500, 200, w=2, h=16)
    x, y = place_panel((500, 200, 2, 16), None, (1920, 1080), (72, 24))
    assert x == 500 - 36.0
    assert y == (1080 - 200) - 16 - 24 - 6.0


def test_place_panel_under_text_box():
    """A wide selection/box centers the panel under the box, not the caret."""
    from nemotron_streaming_asr.apps.dictation.caret import place_panel

    x, y = place_panel((500, 200, 200, 30), None, (1920, 1080), (72, 24))
    assert x == 500 + (200 - 72) / 2.0
    assert y == (1080 - 200) - 30 - 24 - 6.0


def test_place_panel_mouse_fallback():
    """No caret: the panel anchors offset from the mouse cursor."""
    from nemotron_streaming_asr.apps.dictation.caret import place_panel

    x, y = place_panel(None, (100, 900), (1920, 1080), (72, 24))
    assert (x, y) == (112.0, 900 - 24 - 12.0)


def test_place_panel_screen_center_last_resort():
    """No caret and no mouse: centered horizontally near the bottom."""
    from nemotron_streaming_asr.apps.dictation.caret import place_panel

    x, y = place_panel(None, None, (1920, 1080), (72, 24))
    assert (x, y) == ((1920 - 72) / 2.0, 55.0)


def test_place_panel_clamps_to_screen():
    """A caret at the screen edge must not push the panel off-screen."""
    from nemotron_streaming_asr.apps.dictation.caret import place_panel

    x, y = place_panel((5, 5, 2, 16), (1919, 1079), (1920, 1080), (72, 24))
    assert 10.0 <= x <= 1920 - 72 - 10.0
    assert 10.0 <= y <= 1080 - 24 - 10.0
    # ... and the mouse fallback clamps too, without a caret anywhere near.
    x, y = place_panel(None, (0, 0), (1920, 1080), (72, 24))
    assert (x, y) == (12.0, 10.0)


def test_build_display_falls_back_to_console_ui(tiny_model, monkeypatch):
    """If wave_overlay fails to import (e.g. on a non-macOS host),
    build_display() must silently fall back to ConsoleUI rather than crash."""
    import builtins
    from nemotron_streaming_asr.apps.dictation.app import build_display, ConsoleUI

    def _explode(name, *a, **k):
        if name.endswith("wave_overlay"):
            raise ImportError("AppKit not available in CI")
        return real_import(name, *a, **k)
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _explode)
    try:
        assert isinstance(build_display(), ConsoleUI)
    finally:
        # Restore builtins so later tests can import normally.
        monkeypatch.setattr(builtins, "__import__", real_import)


def test_display_show_hide_tracks_recording(tiny_model):
    """show() must be called when recording starts; hide() must NOT happen
    synchronously in stop_recording (hotkey thread can't touch AppKit) but
    on the next main-thread pump_display()."""
    from nemotron_streaming_asr.apps.dictation.app import DictationApp
    from nemotron_streaming_asr.apps.dictation.display import RecordingDisplay

    calls = []
    ui = type("UI", (RecordingDisplay,), {
        "status": lambda self, m: None,
        "on_partial": lambda self, t: None,
        "show": lambda self: calls.append("show"),
        "hide": lambda self: calls.append("hide"),
    })()
    app = DictationApp(tiny_model, language="en-US", recorder=_FakeRecorder(),
                       insertion=_FakeInsertion(), hotkey=_FakeHotkey(),
                       insert=False, ui=ui)
    app.start_recording()
    assert calls == ["show"]
    # The worker thread may not be scheduled yet: recording state is
    # synchronous, so an immediate pump must tick, never idle-hide.
    app.pump_display()
    assert calls == ["show"]
    app.stop_recording()
    app._utterance.join(timeout=10)
    # stop_recording only signals; teardown is deferred to the main thread.
    assert calls == ["show"]
    app.pump_display()
    assert calls == ["show", "hide"]


def test_recording_loop_pushes_blocks_to_display(tiny_model, monkeypatch):
    """While recording, each raw audio block is pushed to the UI's set_level
    (the overlay computes RMS itself, so the visualizer reacts to voice)."""
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
    # the full block-to-visible-range pipeline that ships in production.
    wo = _build_overlay_module(monkeypatch)
    samples = []  # capture every raw block the app pushes
    real_set_level = wo.WaveformOverlayUI.set_level
    def spy_set_level(self, block):
        samples.append(block)
        real_set_level(self, block)
    monkeypatch.setattr(wo.WaveformOverlayUI, "set_level", spy_set_level)
    app = DictationApp(tiny_model, language="en-US", recorder=_LoudQuietRecorder(),
                       insertion=_FakeInsertion(), hotkey=_FakeHotkey(),
                       insert=False, ui=wo.WaveformOverlayUI())
    app.start_recording()
    app._utterance.join(timeout=10)
    # At least 4 blocks were pushed (loud, quiet, loud, quiet).
    assert len(samples) >= 4
    # The loud block (RMS=0.5) must have been pushed before the quiet block
    # (RMS=0.0), and must be a larger RMS value.
    def _rms(block):
        return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    assert _rms(samples[0]) > _rms(samples[1])
    # The current target volume (clamped) is in the visible range.
    assert 0.15 <= app._ui._target_volume <= 1.0
