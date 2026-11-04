"""Background job runner with cooperative (safe) cancellation.

Why a thread?
-------------
Streamlit re-executes the whole script on every widget interaction, and a
click made while a script is still running *aborts* that script at its next
``st.*`` call.  An optimisation executed inline therefore cannot be
cancelled safely: the click kills it and every local result is lost.

Here the optimisation runs in a worker thread that never touches Streamlit.
The UI only reads the job's progress and sets a cancel flag; the worker
notices the flag at its next ``cancel_callback()`` check, finishes the
current step cleanly and returns whatever it has computed so far.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np


class BackgroundJob:
    # If the browser tab is closed the UI stops calling ``touch()``; the
    # worker then stops itself instead of running orphaned.
    HEARTBEAT_TIMEOUT_S = 120.0

    def __init__(self, kind: str, label: str = "",
                 context: Optional[dict] = None):
        self.kind = kind                    # "run" | "sweep"
        self.label = label
        self.context: dict = context or {}  # read-only data for finalisation
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.error_trace: str = ""
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._progress: dict = {}
        self._last_seen = time.time()
        self._thread: Optional[threading.Thread] = None

    # ---------------- lifecycle ----------------
    def start(self, target: Callable[[Callable, Callable], Any]) -> None:
        """Run ``target(progress_cb, cancel_cb)`` in a daemon thread."""
        def _main():
            try:
                self.result = target(self.update_progress, self.should_stop)
            except BaseException as exc:      # noqa: BLE001
                self.error = exc
                self.error_trace = traceback.format_exc()
            finally:
                self.finished_at = time.time()

        self._thread = threading.Thread(
            target=_main, name=f"wds-{self.kind}-job", daemon=True
        )
        self._thread.start()

    # ---------------- cancellation ----------------
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def should_stop(self) -> bool:
        """Passed to the algorithms as ``cancel_callback``."""
        if self._cancel.is_set():
            return True
        return (time.time() - self._last_seen) > self.HEARTBEAT_TIMEOUT_S

    # ---------------- UI side ----------------
    def touch(self) -> None:
        self._last_seen = time.time()

    @property
    def is_done(self) -> bool:
        return self.finished_at is not None

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at

    # ---------------- progress (worker -> UI) ----------------
    def update_progress(self, info: dict) -> None:
        snap = dict(info)
        conv = snap.get("Convergence")
        if conv is not None:
            snap["Convergence"] = np.array(conv, copy=True)
        with self._lock:
            self._progress.update(snap)

    def get_progress(self) -> dict:
        with self._lock:
            return dict(self._progress)
