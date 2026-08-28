"""Floating click-through modern equalizer animation that follows the active text cursor (caret).

Provides ultra-premium, Wispr Flow-grade recording feedback globally on macOS. Automatically
detects the active text caret or focused text box using macOS Accessibility APIs and positions
itself right under it. If no caret is detected, it falls back to the mouse cursor position.
"""

from __future__ import annotations

import math
import objc
import threading
from AppKit import (
    NSView, NSColor, NSBezierPath, NSPanel, NSMakeRect, NSEvent, NSScreen,
    NSRunLoop, NSDefaultRunLoopMode, NSDate, NSFloatingWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel, NSBackingStoreBuffered,
    NSShadow, NSGradient
)
from ApplicationServices import (
    AXUIElementCreateSystemWide, AXUIElementCopyAttributeValue,
    AXUIElementCopyParameterizedAttributeValue, AXValueGetValue,
    kAXFocusedUIElementAttribute, kAXSelectedTextRangeAttribute,
    kAXBoundsForRangeParameterizedAttribute, kAXPositionAttribute, kAXSizeAttribute,
    kAXValueCGPointType, kAXValueCGRectType, kAXValueCGSizeType
)

def get_focused_caret_rect() -> tuple[float, float, float, float] | None:
    """Attempt to get the screen rectangle of the active text caret or focused text box.
    Returns (x, y, w, h) in accessibility coordinates (y=0 at top of main screen), or None.
    """
    try:
        system_wide = AXUIElementCreateSystemWide()
        if not system_wide:
            return None
            
        err, focused_elem = AXUIElementCopyAttributeValue(system_wide, kAXFocusedUIElementAttribute, None)
        if err != 0 or not focused_elem:
            return None
            
        # 1. Try to get caret/selected text range bounds (finest precision)
        err, selected_range = AXUIElementCopyAttributeValue(focused_elem, kAXSelectedTextRangeAttribute, None)
        if err == 0 and selected_range:
            err, bounds_val = AXUIElementCopyParameterizedAttributeValue(
                focused_elem, kAXBoundsForRangeParameterizedAttribute, selected_range, None
            )
            if err == 0 and bounds_val:
                success, rect = AXValueGetValue(bounds_val, kAXValueCGRectType, None)
                if success and rect:
                    return rect.origin.x, rect.origin.y, rect.size.width, rect.size.height
                    
        # 2. Fallback: get the position and size of the focused text box itself
        err, pos_val = AXUIElementCopyAttributeValue(focused_elem, kAXPositionAttribute, None)
        err2, size_val = AXUIElementCopyAttributeValue(focused_elem, kAXSizeAttribute, None)
        if err == 0 and err2 == 0 and pos_val and size_val:
            success, pos = AXValueGetValue(pos_val, kAXValueCGPointType, None)
            success2, size = AXValueGetValue(size_val, kAXValueCGSizeType, None)
            if success and success2 and pos and size:
                return pos.x, pos.y, size.width, size.height
                
    except Exception:
        pass
    return None


class WaveformView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self:
            self._phase = 0.0
            self._volume = 0.15
        return self
        
    def setPhase_(self, phase):
        self._phase = phase
        self.setNeedsDisplay_(True)
        
    def setVolume_(self, volume):
        self._volume = volume
        self.setNeedsDisplay_(True)
        
    def drawRect_(self, rect):
        bounds = self.bounds()
        width = bounds.size.width
        height = bounds.size.height

        # 1. Pill background — obsidian velvet with a top→bottom subtle gradient
        pill_radius = height / 2.0
        pill_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, pill_radius, pill_radius
        )
        bg_gradient = NSGradient.alloc().initWithColors_([
            NSColor.colorWithRed_green_blue_alpha_(0.13, 0.13, 0.15, 0.95),
            NSColor.colorWithRed_green_blue_alpha_(0.05, 0.05, 0.07, 0.92),
        ])
        bg_gradient.drawInBezierPath_angle_(pill_path, 270.0)

        # 2. Inner highlight along the top edge (Apple-style beveled glass lip)
        highlight = NSBezierPath.bezierPath()
        highlight.appendBezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, pill_radius, pill_radius
        )
        highlight.setLineWidth_(1.0)
        NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.28).set()
        highlight.stroke()

        # Subtle inner top-half highlight to mimic light catching the rim
        top_highlight = NSBezierPath.bezierPath()
        top_highlight.moveToPoint_((pill_radius + 1.0, 1.0))
        top_highlight.lineToPoint_((width - pill_radius - 1.0, 1.0))
        top_highlight.setLineWidth_(0.6)
        NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.35).set()
        top_highlight.stroke()

        # 3. Equalizer bars — 5, slender (2.5px) with airy 4.5px gap, bell-curve weighted
        bar_width = 2.5
        gap = 4.5
        num_bars = 5
        total_bars_width = num_bars * bar_width + (num_bars - 1) * gap
        start_x = (width - total_bars_width) / 2.0

        for i in range(num_bars):
            # Bell-curve: center bars (i=2) reach max height
            dist_from_center = abs(i - (num_bars - 1) / 2.0)
            weight = 1.0 - (dist_from_center / ((num_bars - 1) / 2.0 + 1.0)) * 0.28

            # Idle breathing — gentle even when silent
            idle = math.sin(self._phase * 1.6 + i * 0.9) * 0.30 + 0.70
            idle_height = 5.0 + 3.0 * idle * weight

            # Voice amplitude boost with per-bar phase variance
            voice_factor = (0.35 + 0.65 * math.sin(self._phase * 2.8 + i * 1.2))
            voice_height = 11.0 * self._volume * voice_factor * weight

            bar_height = min(height - 8.0, idle_height + voice_height)
            bx = start_x + i * (bar_width + gap)
            by = (height - bar_height) / 2.0
            bar_rect = NSMakeRect(bx, by, bar_width, bar_height)
            bar_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bar_rect, bar_width / 2.0, bar_width / 2.0
            )

            # Per-bar gradient: bright top → deep base (gives the "lit cylinder" feel)
            t = i / (num_bars - 1) if num_bars > 1 else 0.5
            # Hue: cyan → indigo → pink across the spectrum
            r_top, g_top, b_top = 0.18 + t * 0.82, 0.85 - t * 0.45, 1.0
            r_bot, g_bot, b_bot = 0.45 + t * 0.45, 0.40 - t * 0.20, 0.95
            bar_gradient = NSGradient.alloc().initWithColors_([
                NSColor.colorWithRed_green_blue_alpha_(r_top, g_top, b_top, 1.0),
                NSColor.colorWithRed_green_blue_alpha_(r_bot, g_bot, b_bot, 0.85),
            ])

            # Soft outer glow on the bar (sells the "this is alive" feel)
            glow = NSShadow.alloc().init()
            glow.setShadowBlurRadius_(4.0)
            glow.setShadowOffset_((0.0, 0.0))
            glow.setShadowColor_(
                NSColor.colorWithRed_green_blue_alpha_(r_top, g_top, b_top, 0.55)
            )
            glow.set()

            bar_gradient.drawInBezierPath_angle_(bar_path, 270.0)


class WaveformOverlayUI:
    """Floating click-through voice equalizer following the text input caret."""
    
    def __init__(self):
        self._panel = None
        self._view = None
        self._phase = 0.0
        self._target_volume = 0.15
        self._current_volume = 0.15
        self._lock = threading.Lock()
        
        # Sleek Wispr Flow / Dynamic Island proportions (90x28 pill)
        self._panel_width = 90.0
        self._panel_height = 28.0
        
    def status(self, message: str) -> None:
        print(f"\r{message}", end="", flush=True)
        
    def on_partial(self, text: str) -> None:
        print(f"\r{text}", end="", flush=True)
        
    def set_volume(self, rms: float) -> None:
        with self._lock:
            # Map typical speech RMS energy (~0.01 to 0.15) to visible scale [0.15, 1.0]
            self._target_volume = max(0.15, min(1.0, rms * 7.5))
            
    def tick(self) -> None:
        """Called periodically on the main thread during recording to update position & animation."""
        if self._panel is None:
            self._panel, self._view = self._build_panel()
            
        # Advance animation phase smoothly
        self._phase += 0.16
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
            
        # Smooth volume transitions via a low-pass filter
        with self._lock:
            self._current_volume = 0.7 * self._current_volume + 0.3 * self._target_volume
            
        self._view.setPhase_(self._phase)
        self._view.setVolume_(self._current_volume)
        
        # Fetch screen dimensions
        screen_frame = NSScreen.screens()[0].frame()
        screen_width = screen_frame.size.width
        screen_height = screen_frame.size.height
        
        # Detect active focused element or text caret bounds
        rect = get_focused_caret_rect()
        if rect is not None:
            cx, cy_ax, cw, ch = rect
            # Convert Accessibility top-left origin coordinates to AppKit bottom-left origin coordinates
            cy = screen_height - cy_ax
            
            # Position centered horizontally right below the caret/textbox
            if cw < 6.0:  # fine caret line
                x = cx - (self._panel_width / 2.0)
            else:  # text selection or box
                x = cx + (cw - self._panel_width) / 2.0
            y = cy - ch - self._panel_height - 8.0
        else:
            # Fallback: anchor right next to the mouse cursor pointer
            try:
                mouse_pos = NSEvent.mouseLocation()
                x = mouse_pos.x + 12.0
                y = mouse_pos.y - self._panel_height - 12.0
            except Exception:
                x = (screen_width - self._panel_width) / 2.0
                y = 55.0
            
        # Bound coordinates to screen boundaries
        x = max(10.0, min(screen_width - self._panel_width - 10.0, x))
        y = max(10.0, min(screen_height - self._panel_height - 10.0, y))
        
        new_origin = NSMakeRect(x, y, self._panel_width, self._panel_height)
        self._panel.setFrame_display_(new_origin, True)
        
        # Pump the Cocoa event loop briefly
        NSRunLoop.currentRunLoop().runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.005)
        )
        
    def close(self) -> None:
        """Hide and release the floating panel when done."""
        if self._panel is not None:
            self._panel.orderOut_(None)
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.01)
            )
            self._panel = None
            self._view = None
            self._target_volume = 0.15
            self._current_volume = 0.15
            print() # complete the console line
            
    def _build_panel(self):
        screen_frame = NSScreen.screens()[0].frame()
        try:
            mouse_pos = NSEvent.mouseLocation()
            x = mouse_pos.x + 12.0
            y = mouse_pos.y - self._panel_height - 12.0
        except Exception:
            x = (screen_frame.size.width - self._panel_width) / 2.0
            y = 55.0
            
        rect = NSMakeRect(x, y, self._panel_width, self._panel_height)
        
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

        # Configured soft drop shadow (3-4× better than setHasShadow_(True))
        panel_shadow = NSShadow.alloc().init()
        panel_shadow.setShadowBlurRadius_(22.0)
        panel_shadow.setShadowOffset_((0.0, -3.0))
        panel_shadow.setShadowColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.55)
        )
        panel.setShadow_(panel_shadow)

        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setFloatingPanel_(True)
        
        view = WaveformView.alloc().initWithFrame_(NSMakeRect(0, 0, self._panel_width, self._panel_height))
        panel.contentView().addSubview_(view)
        panel.orderFrontRegardless()
        
        return panel, view
