"""Insert text at the current keyboard cursor using native macOS APIs.

Approach (clipboard injection, per the spec): save the user's clipboard,
write the transcript, post a synthetic ⌘V via ``CGEventPost``, then restore
the clipboard. Requires macOS **Accessibility** permission for the app running
it (as with any synthetic keyboard input).

The text-level clipboard content is preserved and restored; non-text
pasteboard items are not (documented limitation of the clipboard-injection
backend).
"""

from __future__ import annotations

import time
from typing import Optional


class TextInsertionService:
    """Pastes text at the cursor while preserving the user's clipboard."""

    def __init__(self, paste_delay_s: float = 0.08):
        # Allow the focused app to process the ⌘V before we restore the
        # clipboard.
        self.paste_delay_s = paste_delay_s

    def insert(self, text: Optional[str]) -> None:
        """Insert ``text`` at the current cursor (no-op when empty)."""
        if not text:
            return
        snapshot = self._snapshot_clipboard()
        try:
            self._set_clipboard(text)
            self._post_command_v()
            time.sleep(self.paste_delay_s)
        finally:
            self._restore_clipboard(snapshot)

    @staticmethod
    def can_post_events() -> bool:
        """True if this process may post synthetic keyboard events (macOS
        Accessibility permission). Returns True if the API is unavailable."""
        try:
            from Quartz import CGPreflightPostEventAccess

            return bool(CGPreflightPostEventAccess())
        except Exception:
            return True

    # ------------------------------------------------------------ clipboard
    @staticmethod
    def _snapshot_clipboard() -> Optional[str]:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        return NSPasteboard.generalPasteboard().stringForType_(NSPasteboardTypeString)

    @staticmethod
    def _set_clipboard(text: str) -> None:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, NSPasteboardTypeString)

    @staticmethod
    def _restore_clipboard(snapshot: Optional[str]) -> None:
        from AppKit import NSPasteboard, NSPasteboardTypeString

        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        if snapshot is not None:
            pb.setString_forType_(snapshot, NSPasteboardTypeString)

    # --------------------------------------------------------------- paste
    @staticmethod
    def _post_command_v() -> None:
        from Quartz import (
            CGEventCreateKeyboardEvent,
            CGEventPost,
            CGEventSetFlags,
            kCGHIDEventTap,
            kCGEventFlagMaskCommand,
        )

        for key_down in (True, False):
            event = CGEventCreateKeyboardEvent(None, 9, key_down)  # 9 == 'v'
            CGEventSetFlags(event, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, event)
