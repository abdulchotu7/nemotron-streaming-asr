"""Caret sensing and panel placement for the dictation overlay.

Two lookup adapters sit behind plain-tuple interfaces:

* :func:`get_focused_caret_rect` — macOS Accessibility lookup of the active
  text caret / focused box, in AX coordinates (y=0 at the top). ``None`` when
  nothing is focused or the subsystem raises.
* :func:`get_mouse_point` — cursor position fallback. ``None`` when unavailable.

:func:`place_panel` is pure: given those tuples plus screen/panel sizes it
returns the AppKit-space ``(x, y)`` origin, owning the AX→AppKit flip,
caret-vs-box anchoring, and screen clamping. No AppKit import needed —
unit-testable with plain tuples.
"""

from __future__ import annotations


def get_focused_caret_rect() -> tuple[float, float, float, float] | None:
    """Attempt to get the screen rectangle of the active text caret or focused text box.
    Returns (x, y, w, h) in accessibility coordinates (y=0 at top of main screen), or None.
    """
    try:
        from ApplicationServices import (
            AXUIElementCreateSystemWide, AXUIElementCopyAttributeValue,
            AXUIElementCopyParameterizedAttributeValue, AXValueGetValue,
            kAXFocusedUIElementAttribute, kAXSelectedTextRangeAttribute,
            kAXBoundsForRangeParameterizedAttribute, kAXPositionAttribute, kAXSizeAttribute,
            kAXValueCGPointType, kAXValueCGRectType, kAXValueCGSizeType
        )
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


def get_mouse_point() -> tuple[float, float] | None:
    """Current mouse cursor position in AppKit coordinates, or None.

    The AppKit import stays inside the function so this module (and the pure
    :func:`place_panel`) remains importable without a GUI session.
    """
    try:
        from AppKit import NSEvent

        pos = NSEvent.mouseLocation()
        return pos.x, pos.y
    except Exception:
        return None


def place_panel(
    caret_rect: tuple[float, float, float, float] | None,
    mouse_xy: tuple[float, float] | None,
    screen_wh: tuple[float, float],
    panel_wh: tuple[float, float],
    gap: float = 6.0,
    margin: float = 10.0,
) -> tuple[float, float]:
    """Return the AppKit-space ``(x, y)`` panel origin.

    * caret known → centered horizontally under the caret (fine caret line)
      or under the selection/box, ``gap`` below it (AX y flipped to AppKit);
    * else mouse known → just offset from the cursor;
    * else → centered horizontally near the bottom of the screen.
    The result is always clamped inside the screen by ``margin``.
    """
    panel_width, panel_height = panel_wh
    screen_width, screen_height = screen_wh
    if caret_rect is not None:
        cx, cy_ax, cw, ch = caret_rect
        # Convert Accessibility top-left origin to AppKit bottom-left origin.
        cy = screen_height - cy_ax
        if cw < 6.0:  # fine caret line
            x = cx - (panel_width / 2.0)
        else:  # text selection or box
            x = cx + (cw - panel_width) / 2.0
        y = cy - ch - panel_height - gap
    elif mouse_xy is not None:
        mx, my = mouse_xy
        x = mx + 12.0
        y = my - panel_height - 12.0
    else:
        x = (screen_width - panel_width) / 2.0
        y = 55.0

    x = max(margin, min(screen_width - panel_width - margin, x))
    y = max(margin, min(screen_height - panel_height - margin, y))
    return x, y
