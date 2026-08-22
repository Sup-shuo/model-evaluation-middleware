from __future__ import annotations

import shutil
from pathlib import Path

from model_evaluation.core.errors import CleanupCriticalError, ModelEvalError, ProcessError
from model_evaluation.core.files import atomic_json
from model_evaluation.core.results import iso_now, plan_timezone


def _cleanup_report(mode: str | None, failure: BaseException | None) -> dict:
    status = None
    if failure is not None:
        details = getattr(failure, "details", None)
        if isinstance(details, dict):
            status = details.get("cleanup_status")
        status = status or getattr(failure, "_model_eval_cleanup_status", None)
    return {
        "schema_version": "1.0",
        "status": "incomplete" if status == "incomplete" else "clean",
        "backend": {
            "status": "not_owned" if mode in {"attached", "external"} else "not_started"
        },
    }


def _stop_backend(orchestrator, backend_handle, backend_shutdown: dict | None) -> dict:
    shutdown = backend_shutdown or {
        "strategy": "signal",
        "signal": "SIGTERM",
        "timeout_seconds": 10.0,
    }
    try:
        return orchestrator.pm.stop_with_report(
            backend_handle,
            graceful_signal=str(shutdown.get("signal") or "SIGTERM"),
            grace_seconds=float(shutdown.get("timeout_seconds") or 10.0),
            kill_seconds=3.0,
        )
    except BaseException as exc:
        return {
            "status": "incomplete",
            "pid": backend_handle.pid,
            "pgid": backend_handle.pgid,
            "owned_process_group_remaining": True,
            "secondary_errors": [orchestrator._error_record(exc)],
            "_internal_error": exc,
        }


def finalize_run(
    orchestrator,
    *,
    run_dir: Path,
    run_id: str,
    plan: dict,
    mode: str | None,
    started_at: str,
    failure: BaseException | None,
    failure_stage: str | None,
    backend_handle,
    backend_shutdown: dict | None,
    evaluator_returncode: int | None,
) -> tuple[BaseException | None, str | None]:
    """Close a run exactly once and return the final primary failure."""
    orchestrator._status_best_effort(run_dir, "CLEANING")
    cleanup = _cleanup_report(mode, failure)

    if backend_handle is not None:
        cleanup["backend"] = _stop_backend(
            orchestrator, backend_handle, backend_shutdown
        )
        internal_error = cleanup["backend"].pop("_internal_error", None)
        if internal_error is not None:
            orchestrator._append_core_error(run_dir, "CLEANING", internal_error)
        if cleanup["backend"].get("status") != "clean":
            cleanup["status"] = "incomplete"
            cleanup_error = CleanupCriticalError(
                "owned backend process group remained or ownership became ambiguous "
                f"after bounded cleanup: pid={backend_handle.pid} pgid={backend_handle.pgid}"
            )
            cleanup.setdefault("secondary_errors", []).append(
                orchestrator._error_record(cleanup_error)
            )
            if failure is None:
                failure = cleanup_error
                failure_stage = "CLEANING"
                orchestrator._append_core_error(run_dir, "CLEANING", cleanup_error)
            else:
                try:
                    setattr(failure, "_model_eval_cleanup_status", "incomplete")
                except Exception:
                    pass
                if isinstance(failure, ModelEvalError):
                    failure.details.setdefault("cleanup_status", "incomplete")

    secondary_cleanup = (
        getattr(failure, "_model_eval_cleanup_error", None)
        if failure is not None
        else None
    )
    if secondary_cleanup is not None:
        cleanup.setdefault("secondary_errors", []).append(
            orchestrator._error_record(secondary_cleanup)
        )

    try:
        if failure is not None:
            failure_product = orchestrator._failure_record(
                run_dir,
                stage=failure_stage or "UNKNOWN",
                failure=failure,
                cleanup=cleanup,
                backend_handle=backend_handle,
                evaluator_returncode=evaluator_returncode,
            )
            orchestrator.schemas.validate("failure", failure_product)
            atomic_json(run_dir / "failure.json", failure_product)

        terminal = {
            "schema_version": "1.0",
            "run_id": run_id,
            "outcome": "success" if failure is None else "failed",
            "started_at": started_at,
            "finished_at": iso_now(plan),
            "timezone": getattr(plan_timezone(plan), "key", "Asia/Shanghai"),
            "cleanup": orchestrator._redact_diagnostic(cleanup),
        }
        if orchestrator._warning_events:
            terminal["warnings"] = orchestrator._redact_diagnostic(
                orchestrator._warning_events
            )
        if failure is not None:
            terminal["error"] = orchestrator._error_record(failure)
        orchestrator.schemas.validate("terminal", terminal)
        atomic_json(run_dir / "terminal.json", terminal)
    except BaseException as exc:
        orchestrator._append_core_error(run_dir, "FINALIZING", exc)
        details = {}
        if failure is not None:
            details["original_error"] = orchestrator._error_record(failure)
        failure = ProcessError(
            f"could not publish final result product: {exc}",
            details=details,
        )
        failure_stage = "FINALIZING"
        return failure, failure_stage

    # Commit the internal terminal state only after the complete public terminal
    # product has been validated and atomically published.
    if failure is None:
        orchestrator._status_best_effort(run_dir, "SUCCEEDED")
        shutil.rmtree(run_dir / ".run", ignore_errors=True)
    else:
        orchestrator._status_best_effort(
            run_dir,
            "FINALIZED",
            outcome="failed",
            error=orchestrator._error_record(failure),
            failure_stage=failure_stage or "UNKNOWN",
        )
    return failure, failure_stage
