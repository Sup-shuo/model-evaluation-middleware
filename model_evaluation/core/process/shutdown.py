from __future__ import annotations

import os
import signal
import subprocess
from typing import TYPE_CHECKING

from model_evaluation.core.errors import ProcessError
from model_evaluation.core.process.procfs import (
    proc_group_members as _proc_group_members,
    proc_group_snapshot as _proc_group_snapshot,
    proc_pgid as _proc_pgid,
)

if TYPE_CHECKING:
    from model_evaluation.core.process.manager import ProcessHandle


def group_alive(pgid: int | None) -> bool:
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A group that exists but cannot be signalled is still alive.
        return True
    live, zombies, complete = _proc_group_snapshot(pgid)
    if live:
        return True
    if zombies and complete:
        # Docker/container PID 1 may leave killed descendants as zombies.
        # They are terminated even though the numeric PGID still exists.
        return False
    # No observable member is ambiguous (for example non-Linux procfs or a
    # concurrent procfs race), so retain the conservative kernel answer.
    return True


def direct_stop(proc: subprocess.Popen, *, grace_seconds: float, kill_seconds: float) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=kill_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ProcessError(
            f"process did not terminate: pid={proc.pid}"
        ) from exc

def stop_with_report(
    manager,
    handle: ProcessHandle,
    *,
    graceful_signal: str = "SIGTERM",
    grace_seconds: float = 10.0,
    kill_seconds: float = 3.0,
) -> dict:
    """Bounded, idempotent cleanup for one Core-owned process group.

    Graceful shutdown is leader-first so the inference framework can run
    its own worker/communicator/runtime teardown.  Only after that bounded
    opportunity may Core use a verified owned process group as a SIGKILL
    fallback.  A bare/reused PGID is never trusted, and cleanup never
    inspects accelerator memory or unrelated device processes.

    Cleanup-internal errors are contained in the returned report; this
    method is intentionally safe to call from an exception handler.
    """
    pgid = handle.pgid or _proc_pgid(handle.pid)
    report = {
        "status": "clean",
        "pid": handle.pid,
        "pgid": pgid,
        "ownership": {"initial": "unknown", "fallback": "not_needed"},
        "graceful": {
            "attempted": False,
            "target": "leader",
            "signal": graceful_signal,
            "timeout_seconds": float(grace_seconds),
            "result": "not_needed",
        },
        "fallback": {
            "sigkill_attempted": False,
            "target": "process_group",
            "timeout_seconds": float(kill_seconds),
            "result": "not_needed",
        },
        "owned_process_group_remaining": False,
    }

    try:
        initial_status = manager._owned_group_status(handle, pgid)
    except BaseException as exc:
        manager._record_cleanup_error(report, "initial_ownership_check", exc)
        initial_status = "ambiguous"
    report["ownership"]["initial"] = initial_status

    if initial_status == "gone":
        report["graceful"]["result"] = "already_exited"
        return manager._finalize_clean_handle(handle, report)
    if initial_status != "verified":
        report["status"] = "incomplete"
        report["graceful"]["result"] = "ownership_ambiguous"
        report["owned_process_group_remaining"] = True
        return report

    # Capture concrete member identities while the leader still proves the
    # group.  These allow a safe fallback if the leader exits before its
    # workers do.
    try:
        known_members = _proc_group_members(pgid)
    except BaseException as exc:
        manager._record_cleanup_error(report, "capture_group_members", exc)
        known_members = {}
    if handle.start_ticks is not None:
        known_members.setdefault(handle.pid, handle.start_ticks)

    try:
        sig = manager._signal_number(graceful_signal)
    except BaseException as exc:
        report["graceful"]["result"] = "error"
        report["graceful"]["error"] = f"{type(exc).__name__}: {exc}"
        manager._record_cleanup_error(report, "resolve_graceful_signal", exc)
        sig = signal.SIGTERM

    report["graceful"]["attempted"] = True
    try:
        # Leader-first: let the framework coordinate its own shutdown.
        if handle.process.poll() is None:
            handle.process.send_signal(sig)
        else:
            report["graceful"]["result"] = "already_exited"
    except ProcessLookupError:
        report["graceful"]["result"] = "already_exited"
    except BaseException as exc:
        report["graceful"]["result"] = "error"
        report["graceful"]["error"] = f"{type(exc).__name__}: {exc}"
        manager._record_cleanup_error(report, "graceful_signal", exc)

    try:
        graceful_gone = manager._wait_owned_group_gone(handle, pgid, grace_seconds)
    except BaseException as exc:
        manager._record_cleanup_error(report, "graceful_wait", exc)
        graceful_gone = False
    if graceful_gone:
        if report["graceful"]["result"] not in {"already_exited", "error"}:
            report["graceful"]["result"] = "success"
        return manager._finalize_clean_handle(handle, report)

    if report["graceful"]["result"] not in {"error", "already_exited"}:
        report["graceful"]["result"] = "timeout"

    try:
        fallback_ownership = manager._owned_group_status(handle, pgid, known_members)
    except BaseException as exc:
        manager._record_cleanup_error(report, "fallback_ownership_check", exc)
        fallback_ownership = "ambiguous"
    report["ownership"]["fallback"] = fallback_ownership
    if fallback_ownership == "gone":
        return manager._finalize_clean_handle(handle, report)
    if fallback_ownership != "verified":
        report["status"] = "incomplete"
        report["fallback"]["result"] = "ownership_ambiguous"
        report["owned_process_group_remaining"] = True
        return report

    # Framework shutdown did not complete.  Fall back only to a process
    # group whose ownership is still evidenced by PID/start-ticks.
    report["fallback"]["sigkill_attempted"] = True
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        elif handle.process.poll() is None:
            handle.process.kill()
    except ProcessLookupError:
        pass
    except BaseException as exc:
        report["fallback"]["result"] = "error"
        report["fallback"]["error"] = f"{type(exc).__name__}: {exc}"
        manager._record_cleanup_error(report, "fallback_sigkill", exc)

    try:
        fallback_gone = manager._wait_owned_group_gone(handle, pgid, kill_seconds)
    except BaseException as exc:
        manager._record_cleanup_error(report, "fallback_wait", exc)
        fallback_gone = False
    if fallback_gone:
        if report["fallback"]["result"] != "error":
            report["fallback"]["result"] = "success"
        return manager._finalize_clean_handle(handle, report)

    report["status"] = "incomplete"
    report["owned_process_group_remaining"] = True
    try:
        final_ownership = manager._owned_group_status(handle, pgid, known_members)
    except BaseException as exc:
        manager._record_cleanup_error(report, "final_ownership_check", exc)
        final_ownership = "ambiguous"
    report["ownership"]["final"] = final_ownership
    if report["fallback"]["result"] != "error":
        report["fallback"]["result"] = "ownership_ambiguous" if final_ownership == "ambiguous" else "timeout"
    # Keep the ownership record for conservative stale recovery.  Do not
    # close output handles while an owned/ambiguous group may still write.
    return report
