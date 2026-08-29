from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from model_evaluation.core.config.parsing import load_json_strict
from model_evaluation.core.errors import (
    ConfigError,
    OrchestrationInterruptedError,
    ProcessError,
)
from model_evaluation.core.files import atomic_json
from model_evaluation.core.identifiers import stable_id
from model_evaluation.core.matrix_planner import verify_matrix_plan
from model_evaluation.core.process.signals import orchestration_signal_guard


@dataclass
class BatchExecution:
    batch_dir: Path | None = None
    status_path: Path | None = None
    status: dict[str, dict] = field(default_factory=dict)
    hard_stop: bool = False
    interrupted_exc: BaseException | None = None


def _prepare_batch(executor, matrix_plan: dict, resume_dir) -> Path:
    if not resume_dir:
        batch_dir = executor._allocate_batch_dir(matrix_plan)
        atomic_json(batch_dir / "matrix_plan.json", matrix_plan)
        return batch_dir

    batch_dir = executor._confined_batch_dir(resume_dir)
    plan_path = batch_dir / "matrix_plan.json"
    if not plan_path.is_file():
        raise ConfigError(
            "resume directory is missing matrix_plan.json: "
            f"{batch_dir}"
        )
    saved = load_json_strict(plan_path.read_text(encoding="utf-8"))
    verify_matrix_plan(saved, app=executor.app)
    if stable_id(saved, length=64) != stable_id(matrix_plan, length=64):
        raise ConfigError("resume directory contains a different matrix plan")
    return batch_dir


def _reuse_success(executor, old: dict | None, plan: dict) -> tuple[bool, str | None]:
    if not old or old.get("status") != "success":
        return False, None
    valid, reason, _ = executor._validate_success_record(old, plan)
    return valid, reason


def _history(old: dict | None) -> list[dict]:
    if not old:
        return []
    keys = (
        "status",
        "attempts",
        "started_at",
        "finished_at",
        "started_utc",
        "finished_utc",
        "run_dir",
        "error",
    )
    previous = {
        key: copy.deepcopy(old.get(key))
        for key in keys
        if old.get(key) is not None
    }
    return [*(old.get("history") or []), previous]


def _running_record(
    executor,
    matrix_plan: dict,
    plan: dict,
    *,
    index: int,
    old: dict | None,
    stale_reason: str | None,
) -> dict:
    model_id, model_label, model_ref = executor._model_identity(plan)
    run_spec = plan["run_spec"]
    record = {
        "index": index,
        "plan_id": plan["plan_id"],
        "model": run_spec["model"],
        "model_id": model_id,
        "model_label": model_label,
        "model_ref": model_ref,
        "platform": run_spec["platform"],
        "deployment": run_spec["deployment"],
        "benchmark": run_spec["benchmark"],
        "evaluation": run_spec["evaluation"],
        "status": "running",
        "attempts": int((old or {}).get("attempts", 0)) + 1,
        "started_at": executor._batch_now(matrix_plan),
    }
    if old:
        record["history"] = _history(old)
    if stale_reason:
        record["resume_validation_warning"] = stale_reason
    return record


def _result_paths(run_path: Path) -> tuple[Path, Path]:
    result_path = run_path / "result.json"
    terminal_path = run_path / "terminal.json"
    if result_path.is_file():
        return result_path, terminal_path
    return run_path / "canonical_result.json", run_path / "terminal_record.json"


def _warning_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        terminal = load_json_strict(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    warnings = terminal.get("warnings")
    if isinstance(warnings, list):
        return len(warnings)
    return int(terminal.get("warnings_count", 0) or 0)


def _run_plan(executor, matrix_plan: dict, plan: dict, record: dict) -> None:
    orchestrator = executor.app.orchestrator(
        results_root=executor.results_root,
        cache_root=executor.cache_root,
        secrets=executor.secrets_map,
    )
    run_path = Path(orchestrator.execute(plan))
    result_path, terminal_path = _result_paths(run_path)
    if not result_path.is_file():
        raise ProcessError(f"run completed without result.json: {run_path}")
    record.update(
        {
            "status": "success",
            "run_dir": str(run_path),
            "result_path": str(result_path),
            "warnings_count": _warning_count(terminal_path),
            "finished_at": executor._batch_now(matrix_plan),
        }
    )


def _mark_interrupted(
    executor,
    matrix_plan: dict,
    record: dict,
    exc: BaseException,
    *,
    message: str,
) -> None:
    record.update(
        {
            "status": "interrupted",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc) or message,
            },
            "finished_at": executor._batch_now(matrix_plan),
        }
    )


def _failed_run_dir(executor, exc: BaseException) -> Path | None:
    value = (getattr(exc, "details", {}) or {}).get("run_dir")
    if not value:
        return None
    try:
        path = executor._safe_confined_path(
            value,
            executor.results_root,
            label="failed run_dir",
            require_dir=True,
        )
    except Exception:
        return None
    return path if path != executor.results_root else None


def _failure_path(run_dir: Path) -> Path | None:
    legacy = run_dir / "error.json"
    current = run_dir / "failure.json"
    if legacy.is_file():
        return legacy
    return current if current.is_file() else None


def _apply_failure_product(record: dict, run_dir: Path) -> None:
    record["run_dir"] = str(run_dir)
    path = _failure_path(run_dir)
    if path is None:
        return
    try:
        failure = load_json_strict(path.read_text(encoding="utf-8"))
    except Exception:
        return
    primary = failure.get("error") or failure.get("primary_error")
    if isinstance(primary, dict):
        record["error"] = primary
    record["failure_path"] = str(path)
    cleanup = failure.get("cleanup") or {}
    if isinstance(cleanup, dict):
        record["cleanup_status"] = cleanup.get("status") or (
            cleanup.get("backend") or {}
        ).get("status")


def _fallback_cleanup_status(exc: BaseException) -> str | None:
    details = getattr(exc, "details", None)
    status = details.get("cleanup_status") if isinstance(details, dict) else None
    return status or getattr(exc, "_model_eval_cleanup_status", None)


def _record_failure(
    executor,
    matrix_plan: dict,
    record: dict,
    exc: BaseException,
) -> bool:
    record.update(
        {
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "finished_at": executor._batch_now(matrix_plan),
        }
    )
    failed_dir = _failed_run_dir(executor, exc)
    if failed_dir is not None:
        _apply_failure_product(record, failed_dir)
    cleanup_status = record.get("cleanup_status") or _fallback_cleanup_status(exc)
    if cleanup_status is not None:
        record["cleanup_status"] = cleanup_status
    return executor._failure_requires_hard_stop(exc, cleanup_status)


def _write_status(executor, matrix_plan: dict, state: BatchExecution) -> None:
    executor._write_status(
        state.status_path,
        matrix_plan["matrix_id"],
        state.status,
    )


def _execute_children(
    executor,
    matrix_plan: dict,
    state: BatchExecution,
    *,
    keep_going: bool,
) -> None:
    for index, plan in enumerate(matrix_plan["plans"], 1):
        plan_id = plan["plan_id"]
        old = state.status.get(plan_id)
        reusable, stale_reason = _reuse_success(executor, old, plan)
        if reusable:
            continue
        record = _running_record(
            executor,
            matrix_plan,
            plan,
            index=index,
            old=old,
            stale_reason=stale_reason,
        )
        state.status[plan_id] = record
        _write_status(executor, matrix_plan, state)
        try:
            _run_plan(executor, matrix_plan, plan, record)
        except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
            _mark_interrupted(
                executor,
                matrix_plan,
                record,
                exc,
                message="user interrupt",
            )
            state.hard_stop = True
            state.interrupted_exc = exc
            _write_status(executor, matrix_plan, state)
            break
        except Exception as exc:
            state.hard_stop = _record_failure(
                executor,
                matrix_plan,
                record,
                exc,
            )
            if state.hard_stop or not keep_going:
                _write_status(executor, matrix_plan, state)
                break
        _write_status(executor, matrix_plan, state)


def _record_outer_interrupt(
    executor,
    matrix_plan: dict,
    state: BatchExecution,
    exc: BaseException,
) -> None:
    state.hard_stop = True
    state.interrupted_exc = exc
    for record in state.status.values():
        if record.get("status") == "running":
            _mark_interrupted(
                executor,
                matrix_plan,
                record,
                exc,
                message="matrix interrupted",
            )
    if state.status_path is not None:
        _write_status(executor, matrix_plan, state)


def _finalize(
    executor,
    matrix_plan: dict,
    state: BatchExecution,
    *,
    planned_ids: set[str],
    keep_going: bool,
) -> tuple[Path, dict]:
    if state.batch_dir is None:
        if state.interrupted_exc is not None:
            raise state.interrupted_exc
        raise ProcessError("matrix execution did not establish a batch directory")

    has_failure = any(
        state.status.get(plan_id, {}).get("status") == "failed"
        for plan_id in planned_ids
    )
    if state.hard_stop or (has_failure and not keep_going):
        executor._fill_not_run_records(matrix_plan, state.status)
    try:
        summary = executor._finalize_batch(
            state.batch_dir,
            matrix_plan,
            state.status,
            hard_stop=state.hard_stop,
            keep_going=keep_going,
            interrupted=state.interrupted_exc is not None,
        )
    except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
        state.hard_stop = True
        state.interrupted_exc = state.interrupted_exc or exc
        executor._fill_not_run_records(matrix_plan, state.status)
        summary = executor._finalize_batch(
            state.batch_dir,
            matrix_plan,
            state.status,
            hard_stop=True,
            keep_going=keep_going,
            interrupted=True,
        )
    if state.interrupted_exc is not None:
        raise state.interrupted_exc
    return state.batch_dir, summary


def execute_matrix(
    executor,
    matrix_plan: dict,
    *,
    continue_on_error: bool | None = None,
    resume_dir: str | Path | None = None,
) -> tuple[Path, dict]:
    verify_matrix_plan(matrix_plan, app=executor.app)
    configured = bool(
        (matrix_plan["matrix_spec"].get("execution") or {}).get(
            "continue_on_error",
            False,
        )
    )
    keep_going = (
        configured if continue_on_error is None else bool(continue_on_error)
    )
    planned_ids = {plan["plan_id"] for plan in matrix_plan["plans"]}
    claim = {
        "kind": "other",
        "id": f"matrix:{matrix_plan['matrix_id']}",
        "exclusive": True,
    }
    state = BatchExecution()
    with orchestration_signal_guard(), executor.resources.acquire([claim]):
        try:
            state.batch_dir = _prepare_batch(executor, matrix_plan, resume_dir)
            state.status_path = state.batch_dir / "batch_status.json"
            state.status = executor._load_status(
                state.status_path,
                matrix_plan["matrix_id"],
                planned_ids,
            )
            _execute_children(
                executor,
                matrix_plan,
                state,
                keep_going=keep_going,
            )
        except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
            _record_outer_interrupt(executor, matrix_plan, state, exc)
        return _finalize(
            executor,
            matrix_plan,
            state,
            planned_ids=planned_ids,
            keep_going=keep_going,
        )


__all__ = ["execute_matrix"]
