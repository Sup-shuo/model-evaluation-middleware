from __future__ import annotations

import signal
from contextlib import contextmanager

from model_evaluation.core.errors import OrchestrationInterruptedError


@contextmanager
def orchestration_signal_guard():
    """Convert SIGTERM into a Python exception so Orchestrator cleanup runs.

    SIGINT keeps KeyboardInterrupt semantics.  Handlers are installed only in
    the main thread; Python raises ValueError otherwise, in which case the
    caller still retains normal exception cleanup behavior.
    """
    previous: dict[int, object] = {}
    interrupted = False

    def _once(exc: BaseException):
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        raise exc

    def _term(signum, _frame):
        _once(OrchestrationInterruptedError(f"orchestration interrupted by signal {signum}"))

    def _int(signum, _frame):
        _once(KeyboardInterrupt())

    try:
        for sig, handler in ((signal.SIGTERM, _term), (signal.SIGINT, _int)):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
    except ValueError:
        previous.clear()
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
