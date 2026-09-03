"""WhisperFlow-style dictation around the streaming ASR engine (tap-to-toggle).

Layering::

    GlobalHotkey ──tap (toggle)──▶ DictationApp (recording worker)
                                         │
                                         ▼
    MicrophoneRecorder ──20 ms PCM──▶ NemotronStreamingSession (black box)
                                         │
                                         ▼
                                 LiveTranscriptController ──▶ UI
                                         │
                                         ▼
                                 TextInsertionService (paste at cursor)

User workflow: tap the hotkey (right **Option ⌥**) to start recording, speak,
then tap it again to stop — the final transcript is pasted at the cursor. A
fresh session is created per recording, so no state leaks between recordings.

Usage:
    python -m nemotron_streaming_asr.apps.dictation
    nemotron-dictation --lookahead 3 --no-insert
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from nemotron_streaming_asr import NemotronStreamingSession

from .display import RecordingDisplay
from .hotkey import GlobalHotkey, PynputGlobalHotkey
from .microphone import MicrophoneRecorder
from .text_insertion import TextInsertionService
from .transcript import LiveTranscriptController
from .utterance import Utterance


def build_display():
    """Construct the richest display available (overlay, else console).

    The overlay needs macOS AppKit; anywhere it cannot be built (CI, Linux,
    missing permissions plumbing) this silently falls back to ``ConsoleUI``.
    """
    try:
        from .wave_overlay import WaveformOverlayUI

        return WaveformOverlayUI()
    except Exception:
        return ConsoleUI()


class ConsoleUI(RecordingDisplay):
    """Minimal console UI: only the newest cumulative transcript is shown.

    Each partial update overwrites the current console line, matching the
    WhisperFlow-style live display. Swap this out for a floating overlay later
    without touching the rest of the app.
    """

    def __init__(self, stream=None):
        self._stream = stream or sys.stdout
        self._line_len = 0

    def status(self, message: str) -> None:
        """Print a status message on its own line."""
        prefix = "\n" if self._line_len else ""
        self._write(prefix + message + "\n")
        self._line_len = 0

    def on_partial(self, text: str) -> None:
        """Replace the current line with the newest cumulative transcript."""
        self._write("\r" + text + " " * max(0, self._line_len - len(text)))
        self._line_len = len(text)

    def _write(self, s: str) -> None:
        self._stream.write(s)
        self._stream.flush()


class DictationApp:
    """Coordinates hotkey, microphone, ASR session, transcript and insertion.

    One ``NemotronStreamingSession`` is created per recording (hotkey tap)
    and destroyed after the final text is inserted (stop tap + finish),
    so no streaming state leaks between recordings.
    """

    def __init__(
        self,
        model,
        language: str = "en-US",
        att_context_size=None,
        *,
        ui: RecordingDisplay,
        recorder,
        insertion: TextInsertionService,
        hotkey: GlobalHotkey,
        insert: bool = True,
    ):
        self.model = model
        self.language = language
        self.att_context_size = att_context_size
        self.insert = insert

        self._ui = ui
        self._recorder = recorder
        self._insertion = insertion

        self.transcript = LiveTranscriptController(on_update=self._ui.on_partial)
        self._hotkey = hotkey
        self._hotkey.on_press = self.start_recording
        self._hotkey.on_release = self.stop_recording

        self._session: Optional[NemotronStreamingSession] = None
        self._utterance: Optional[Utterance] = None

    @classmethod
    def build_default(
        cls,
        model,
        language: str = "en-US",
        att_context_size=None,
        insert: bool = True,
    ) -> "DictationApp":
        """Wire a production app: overlay (or console), mic, inserter, hotkey."""
        hotkey = PynputGlobalHotkey(key="alt_r")  # right Option
        return cls(
            model,
            language=language,
            att_context_size=att_context_size,
            ui=build_display(),
            recorder=MicrophoneRecorder(),
            insertion=TextInsertionService(),
            hotkey=hotkey,
            insert=insert,
        )

    # ------------------------------------------------------------- lifecycle
    @property
    def _is_recording(self) -> bool:
        return self._utterance is not None and self._utterance.is_running

    def pump_display(self) -> None:
        """Single main-thread display step: hide when idle, then tick.

        UI teardown is deferred here (never in ``stop_recording``) because
        AppKit must only be touched on the main thread — ``stop_recording``
        runs on the hotkey listener thread.
        """
        if not self._is_recording:
            self._ui.hide()
        self._ui.tick()

    def run(self) -> None:
        """Start the hotkey listener and block until interrupted."""
        self._ui.status(
            "Ready. Tap ⌥ (right Option) to start recording; tap again to stop and insert."
        )
        if self.insert and not TextInsertionService.can_post_events():
            self._ui.status(
                "⚠ Paste needs Accessibility permission: System Settings → "
                "Privacy & Security → Accessibility → enable for this terminal/app."
            )
        self._hotkey.start()
        try:
            while True:
                self.pump_display()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_recording()
            if self._utterance is not None:
                self._utterance.join(timeout=5)
            self._ui.hide()  # main thread here: safe to tear down directly
            self._hotkey.stop()

    def start_recording(self) -> None:
        """Hotkey pressed/tapped: create a fresh session and start capturing."""
        prev = self._utterance
        if prev is not None and prev.is_running:
            # A stop tap may have just landed and the previous utterance could
            # still be draining/finalizing. Wait for it so this tap is not
            # swallowed (otherwise a quick stop->start needs a fourth tap).
            prev.join(timeout=5.0)
            if prev.is_running:  # still busy -> drop this tap
                return
        self._session = NemotronStreamingSession(
            self.model,
            language=self.language,
            att_context_size=self.att_context_size,
        )
        self._utterance = Utterance(
            self._session,
            self._recorder,
            self.transcript,
            self._ui,
            self._insertion,
            insert=self.insert,
        )
        self._ui.show()
        self._utterance.start()

    def stop_recording(self) -> None:
        """Hotkey released/tapped again: signal the worker to drain and finalize.

        Only signals; the display is hidden later on the main thread by
        :meth:`pump_display` (AppKit is main-thread-only).
        """
        if self._utterance is not None:
            self._utterance.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tap-to-toggle dictation: tap the hotkey to start recording, "
        "tap again to stop — the transcript is pasted at your cursor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="prompt language (e.g. en-US); use 'auto' to detect the "
        "language from the audio and switch the prompt live",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=13,
        choices=[0, 1, 3, 6, 13],
        help="chunk-latency operating point; (lookahead+1)*80 ms per chunk "
        "(default 13 -> 1.12 s, lowest latency 0 -> 80 ms)",
    )
    parser.add_argument(
        "--no-insert",
        action="store_true",
        help="do not paste the transcript; only print it",
    )
    args = parser.parse_args()

    from mlx_audio.stt import load

    print("Loading model ...", flush=True)
    model = load(args.model)
    model.eval()

    app = DictationApp.build_default(
        model,
        language=args.language,
        att_context_size=[56, args.lookahead],
        insert=not args.no_insert,
    )
    app.run()


if __name__ == "__main__":
    main()
