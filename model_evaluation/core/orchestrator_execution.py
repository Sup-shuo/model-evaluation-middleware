from __future__ import annotations

import copy
from pathlib import Path

from model_evaluation.core.errors import (
    CompatibilityError,
    ModelEvalError,
    StaleProcessError,
)
from model_evaluation.core.execution_plan import validate_execution_plan
from model_evaluation.core.files import atomic_json
from model_evaluation.core.orchestration_pipeline import (
    evaluate_and_publish,
    preflight_evaluator,
    prepare_dataset_and_task,
    prepare_platform,
    start_service,
)
from model_evaluation.core.process.signals import orchestration_signal_guard
from model_evaluation.core.results import allocate_run_dir, iso_now
from model_evaluation.core.run_finalization import finalize_run
from model_evaluation.core.run_lifecycle import RunContext


def _reject_incompatible_plan(plan: dict) -> None:
    if plan["compatibility"]["status"] != "incompatible":
        return
    raise CompatibilityError(
        "; ".join(plan["compatibility"].get("reasons") or ["plan is incompatible"]),
        details={"diagnostics": plan["compatibility"].get("diagnostics") or []},
    )


def _recover_stale_processes(orchestrator) -> list[dict]:
    stale = orchestrator.pm.recover_stale_managed()
    unresolved = [
        record
        for record in stale
        if record.get("status")
        in {
            "identity_mismatch",
            "cleanup_failed",
            "invalid",
            "orphaned_group_ambiguous",
        }
    ]
    if unresolved:
        raise StaleProcessError(
            f"unresolved stale managed-process ownership records: {unresolved}"
        )
    return stale


def _run_pipeline(orchestrator, context: RunContext) -> None:
    prepare_platform(orchestrator, context)
    prepare_dataset_and_task(orchestrator, context)
    preflight_evaluator(orchestrator, context)
    start_service(orchestrator, context)
    evaluate_and_publish(orchestrator, context)


def _record_pipeline_failure(
    orchestrator,
    context: RunContext,
    exc: BaseException,
) -> str:
    failure_stage = orchestrator._current_state(context.run_dir) or "INITIALIZING"
    orchestrator._append_core_error(context.run_dir, failure_stage, exc)
    try:
        orchestrator._status(
            context.run_dir,
            "FAILED",
            error=orchestrator._error_record(exc),
            failure_stage=failure_stage,
        )
    except Exception:
        pass
    return failure_stage


def _execute_allocated_run(
    orchestrator,
    plan: dict,
    stale: list[dict],
) -> Path:
    started_at = iso_now(plan)
    run_dir = allocate_run_dir(orchestrator.results_root, plan)
    context = RunContext(
        plan=plan,
        run_dir=run_dir,
        started_at=started_at,
        stale_recovery=copy.deepcopy(stale),
    )
    failure: BaseException | None = None
    failure_stage: str | None = None
    orchestrator._warning_events = [
        copy.deepcopy(item) for item in (plan.get("warnings") or [])
    ]
    try:
        orchestrator._persist_initial(run_dir, plan, started_at=started_at)
        if stale:
            atomic_json(
                run_dir / ".run" / "diagnostics" / "stale_process_recovery.json",
                stale,
            )
        orchestrator._status(run_dir, "CREATED")
        orchestrator._status(run_dir, "PLANNED")
        _run_pipeline(orchestrator, context)
    except BaseException as exc:
        failure = exc
        failure_stage = _record_pipeline_failure(orchestrator, context, exc)
    finally:
        failure, failure_stage = finalize_run(
            orchestrator,
            run_dir=run_dir,
            run_id=context.run_id,
            plan=plan,
            mode=context.mode,
            started_at=started_at,
            failure=failure,
            failure_stage=failure_stage,
            backend_handle=context.backend_handle,
            backend_shutdown=context.backend_shutdown,
            evaluator_returncode=context.evaluator_returncode,
        )
    if failure is not None:
        if isinstance(failure, ModelEvalError):
            failure.details.setdefault("run_dir", str(run_dir))
        raise failure
    return run_dir


def execute_plan(orchestrator, plan: dict) -> Path:
    validate_execution_plan(plan, orchestrator.schemas)
    _reject_incompatible_plan(plan)
    non_global_claims = [
        claim for claim in plan["resources"] if claim["kind"] != "run_lock"
    ]
    with orchestration_signal_guard(), orchestrator.resources.run_lock():
        orchestrator._active_plan = plan
        stale = _recover_stale_processes(orchestrator)
        with orchestrator.resources.acquire(non_global_claims):
            return _execute_allocated_run(orchestrator, plan, stale)
