"""One spoken utterance: worker thread, drain, finish, paste.

``Utterance`` owns the choreography of a single recording — capture loop,
drain-after-stop, trailing flush, transcript snapshot, and paste — behind
``start()`` / ``stop()`` / ``join()``. The app keeps hotkey wiring, session
construction, and the display; the thread/drain/finalize races live here,
fixed once.

Sync semantics (unchanged from the app's old inline loop):

* ``start()`` returns immediately; the worker runs as a daemon thread.
* ``stop()`` only signals; it never blocks and never touches UI.
* ``join()`` blocks for the worker and returns the final text.
"""

from __future__ import annotations

import threading


class Utterance:
    """Runs one recording from first block to pasted text."""

    def __init__(self, session, recorder, transcript, display, insertion,
                 insert: bool = True):
        self._session = session
        self._recorder = recorder
        self._transcript = transcript
        self._display = display
        self._insertion = insertion
        self._insert = insert
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        # Set synchronously in start(), cleared by the worker when it ends.
        # Thread aliveness alone can't drive this: between spawn and first
        # scheduling the worker reads as dead, which would wrongly idle-hide
        # the display and stick it hidden for the whole recording.
        self._running = False
        self._final_text = ""

    # ------------------------------------------------------------- control
    def start(self) -> None:
        """Clear the transcript, start capturing, spawn the worker."""
        self._running = True
        self._stop_event.clear()
        self._transcript.clear()
        self._recorder.start()
        self._worker = threading.Thread(
            target=self._run, name="dictation-recording", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        """Signal the worker to drain and finalize (never blocks)."""
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> str:
        """Wait for the worker and return the final transcript text."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
        return self._final_text

    @property
    def is_running(self) -> bool:
        """True from start() until the worker ends (synchronous flag)."""
        return self._running

    # -------------------------------------------------------- recording loop
    def _run(self) -> None:
        self._display.status("Listening...")
        try:
            while not self._stop_event.is_set():
                block = self._recorder.poll(timeout=0.02)
                if block is None:
                    continue
                self._display.set_level(block)
                self._feed_and_step(block)

            # Stop capturing, then process every block already received.
            self._recorder.stop()
            for block in self._recorder.drain():
                self._feed_and_step(block)

            # End of utterance: flush the trailing partial chunk.
            for result in self._session.finish():
                self._transcript.update(result)
        except Exception as e:  # keep the app alive on any pipeline error
            self._display.status(f"[dictation] error: {e!r}")
        finally:
            self._recorder.stop()
            self._running = False

        final_text = self._transcript.current_text
        self._final_text = final_text
        if final_text:
            self._display.status(f"✓ {final_text}")
            if self._insert:
                self._insertion.insert(final_text)
        else:
            self._display.status("(no speech detected)")
        self._display.status("Ready for next recording.")

    def _feed_and_step(self, block) -> None:
        self._session.feed(block)
        for result in self._session.step():
            self._transcript.update(result)
