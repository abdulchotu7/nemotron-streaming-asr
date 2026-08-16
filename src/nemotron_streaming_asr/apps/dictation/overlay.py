"""Floating always-on-top transcript overlay (AppKit NSPanel).

Shows the newest cumulative transcript in a small translucent panel pinned to
the top-center of the main screen. The panel is non-activating and ignores
mouse events, so it never steals focus or blocks clicks.

Threading: ``status()``/``on_partial()`` may be called from any thread — they
only enqueue text. All AppKit work happens on the main thread inside
``tick()``, which the dictation app calls from its idle loop; ``tick()`` also
pumps the Cocoa run loop so the panel actually renders.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from AppKit import (
    NSApplication,
    NSBackingStoreBuffered,
    NSColor,
    NSDate,
    NSDefaultRunLoopMode,
    NSFloatingWindowLevel,
    NSFont,
    NSLineBreakByWordWrapping,
    NSMakeRect,
    NSPanel,
    NSRunLoop,
    NSScreen,
    NSTextField,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)

_PANEL_WIDTH = 640.0
_PANEL_HEIGHT = 110.0
_TOP_MARGIN = 40.0


class OverlayUI:
    """Floating overlay backend for the dictation app's UI hooks."""

    def __init__(self):
        NSApplication.sharedApplication()
        self._panel: Optional[NSPanel] = None
        self._label: Optional[NSTextField] = None
        self._lock = threading.Lock()
        self._pending: List[Tuple[str, bool]] = []  # (text, is_status)

    # ------------------------------------------------------- UI hooks (any thread)
    def status(self, message: str) -> None:
        """Queue a status message (shown like any other text)."""
        self._enqueue(message)

    def on_partial(self, text: str) -> None:
        """Queue the newest cumulative transcript."""
        self._enqueue(text)

    def _enqueue(self, text: str) -> None:
        with self._lock:
            self._pending.append((text, True))

    # ------------------------------------------------- main thread (idle loop)
    def tick(self) -> None:
        """Flush queued text onto the panel and pump the Cocoa run loop.

        Called by the dictation app's idle loop on the main thread.
        """
        with self._lock:
            pending, self._pending = self._pending, []
        if pending:
            panel, label = self._ensure_panel()
            label.setStringValue_(pending[-1][0])
            panel.displayIfNeeded()
        # Pump events so the window server redraws and the panel stays live.
        NSRunLoop.currentRunLoop().runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05)
        )

    # ------------------------------------------------------------- internals
    def _ensure_panel(self):
        if self._panel is None:
            self._panel, self._label = self._build_panel()
        return self._panel, self._label

    @staticmethod
    def _build_panel():
        screen = NSScreen.mainScreen()
        frame = screen.frame()
        x = (frame.size.width - _PANEL_WIDTH) / 2
        y = frame.size.height - _PANEL_HEIGHT - _TOP_MARGIN
        rect = NSMakeRect(x, y, _PANEL_WIDTH, _PANEL_HEIGHT)

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setFloatingPanel_(True)

        label = NSTextField.labelWithString_("")
        label.setFont_(NSFont.systemFontOfSize_(22.0))
        label.setTextColor_(NSColor.whiteColor())
        label.setAlignment_(1)  # NSCenterTextAlignment
        label.setEditable_(False)
        label.setBordered_(False)
        label.setDrawsBackground_(False)
        label.setSelectable_(False)
        label.setUsesSingleLineMode_(False)
        label.setLineBreakMode_(NSLineBreakByWordWrapping)
        label.setFrame_(NSMakeRect(24, 20, _PANEL_WIDTH - 48, _PANEL_HEIGHT - 40))
        label.setWantsLayer_(True)
        label.layer().setShadowOpacity_(0.9)
        label.layer().setShadowRadius_(4.0)
        label.layer().setShadowOffset_((0.0, -1.0))

        panel.contentView().addSubview_(label)
        panel.orderFrontRegardless()
        return panel, label
