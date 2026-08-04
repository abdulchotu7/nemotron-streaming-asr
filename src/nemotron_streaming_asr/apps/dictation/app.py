"""WhisperFlow-style hold-to-talk dictation around the streaming ASR engine.

Layering::

    GlobalHotkey ──press/release──▶ DictationApp (recording worker)
                                         │
                                         ▼
    MicrophoneRecorder ──20 ms PCM──▶ NemotronStreamingSession (black box)
                                         │
                                         ▼
                                 LiveTranscriptController ──▶ UI
                                         │
                                         ▼
                                 TextInsertionService (paste at cursor)

User workflow: hold ⌘⌥ (Cmd+Option), speak, release -> the final transcript is
pasted at the cursor. A fresh session is created per recording, so no state
leaks between recordings.

Usage:
    python -m nemotron_streaming_asr.apps.dictation
    nemotron-dictation --lookahead 3 --no-insert
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Optional

from nemotron_streaming_asr import NemotronStreamingSession

from .hotkey import GlobalHotkey, PynputGlobalHotkey
from .microphone import MicrophoneRecorder
from .text_insertion import TextInsertionService
from .transcript import LiveTranscriptController


class ConsoleUI:
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

    One ``NemotronStreamingSession`` is created per recording (hotkey press)
    and destroyed after the final text is inserted (hotkey release + finish),
    so no streaming state leaks between recordings.
    """

    def __init__(
        self,
        model,
        language: str = "en-US",
        att_context_size=None,
        hotkey: Optional[GlobalHotkey] = None,
        recorder=None,
        insert: bool = True,
        ui=None,
    ):
        self.model = model
        self.language = language
        self.att_context_size = att_context_size
        self.insert = insert

        self._ui = ui or ConsoleUI()
        self._recorder = recorder or MicrophoneRecorder()
        self._insertion = TextInsertionService()

        self.transcript = LiveTranscriptController(on_update=self._ui.on_partial)
        self._hotkey = hotkey or PynputGlobalHotkey(modifiers=("cmd", "option"))
        self._hotkey.on_press = self.start_recording
        self._hotkey.on_release = self.stop_recording

        self._recording = False
        self._stop_event = threading.Event()
        self._session: Optional[NemotronStreamingSession] = None
        self._worker: Optional[threading.Thread] = None

    # ------------------------------------------------------------- lifecycle
    def run(self) -> None:
        """Start the hotkey listener and block until interrupted."""
        self._ui.status("Ready. Hold ⌘⌥ (Cmd+Option) and speak; release to insert.")
        if self.insert and not TextInsertionService.can_post_events():
            self._ui.status(
                "⚠ Paste needs Accessibility permission: System Settings → "
                "Privacy & Security → Accessibility → enable for this terminal/app."
            )
        self._hotkey.start()
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_recording()
            if self._worker is not None:
                self._worker.join(timeout=5)
            self._hotkey.stop()

    def start_recording(self) -> None:
        """Hotkey pressed: create a fresh session and start capturing."""
        if self._recording:
            return
        self._recording = True
        self._stop_event.clear()
        self._session = NemotronStreamingSession(
            self.model,
            language=self.language,
            att_context_size=self.att_context_size,
        )
        self.transcript.clear()
        self._recorder.start()
        self._worker = threading.Thread(
            target=self._recording_loop, name="dictation-recording", daemon=True
        )
        self._worker.start()

    def stop_recording(self) -> None:
        """Hotkey released: signal the worker to drain and finalize."""
        if self._recording:
            self._stop_event.set()

    # -------------------------------------------------------- recording loop
    def _recording_loop(self) -> None:
        self._ui.status("Listening...")
        try:
            while not self._stop_event.is_set():
                block = self._recorder.poll(timeout=0.02)
                if block is None:
                    continue
                self._feed_and_step(block)

            # Stop capturing, then process every block already received.
            self._recorder.stop()
            for block in self._recorder.drain():
                self._feed_and_step(block)

            # End of utterance: flush the trailing partial chunk.
            for result in self._session.finish():
                self.transcript.update(result)
        except Exception as e:  # keep the app alive on any pipeline error
            self._ui.status(f"[dictation] error: {e!r}")
        finally:
            self._recorder.stop()
            self._session = None
            self._recording = False

        final_text = self.transcript.current_text
        if final_text:
            self._ui.status(f"✓ {final_text}")
            if self.insert:
                self._insertion.insert(final_text)
        else:
            self._ui.status("(no speech detected)")
        self._ui.status("Ready for next recording.")

    def _feed_and_step(self, block) -> None:
        self._session.feed(block)
        for result in self._session.step():
            self.transcript.update(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hold-to-talk dictation: hold ⌘⌥, speak, release — the "
        "transcript is pasted at your cursor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
    )
    parser.add_argument("--language", default="en-US")
    parser.add_argument(
        "--lookahead",
        type=int,
        default=13,
        choices=[0, 1, 3, 6, 13],
        help="chunk-latency operating point; (lookahead+1)*80 ms per chunk "
        "(default 13 -> 1.12 s, lowest latency 0 -> 80 ms)",
    )
    parser.add_argument(
        "--hotkey",
        default="cmd+option",
        help="modifiers held to record, '+' separated (e.g. cmd+option, cmd+shift)",
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

    hotkey = PynputGlobalHotkey(modifiers=tuple(args.hotkey.split("+")))
    app = DictationApp(
        model,
        language=args.language,
        att_context_size=[56, args.lookahead],
        hotkey=hotkey,
        insert=not args.no_insert,
    )
    app.run()


if __name__ == "__main__":
    main()
