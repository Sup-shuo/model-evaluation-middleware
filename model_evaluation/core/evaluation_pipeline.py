from __future__ import annotations

import time
from pathlib import Path

from model_evaluation.core.compatibility import (
    evaluate,
    facts_from_environment,
    facts_from_service,
    merge_fact_sets,
)
from model_evaluation.core.errors import CompatibilityError, ModelEvalError, ProcessError
from model_evaluation.core.performance import build_performance
from model_evaluation.core.results import iso_now, plan_timezone, publish_result
from model_evaluation.core.run_lifecycle import RunContext


def _check_runtime_compatibility(orchestrator, context: RunContext) -> None:
    facts = merge_fact_sets(
        facts_from_service(context.service),
        facts_from_environment(
            context.resolved_platform["evaluation_environment"],
            "evaluation_environment",
        ),
    )
    report = evaluate(context.evaluator_requirements, facts)
    if report.compatible:
        return
    raise CompatibilityError(
        "; ".join(report.reasons),
        details={"diagnostics": report.diagnostics},
    )


def _verify_bound_inputs(orchestrator, context: RunContext) -> None:
    orchestrator._verify_task_artifacts(
        context.task,
        context.run_dir / ".run" / "task",
    )
    verified = orchestrator._invoke(
        context.dataset_client,
        "verify",
        {"artifact": context.dataset, "benchmark": context.benchmark},
        context={"offline": True, "final_verification": True},
        timeout=30,
    )
    if not verified["valid"]:
        raise ModelEvalError(
            f"dataset final verification failed: {verified.get('details') or {}}"
        )
    final_artifact = verified.get("artifact") or context.dataset
    orchestrator._verify_dataset_identity(
        context.plan["resolved"].get("dataset_resolution") or {},
        context.benchmark,
        final_artifact,
    )
    if final_artifact.get("fingerprint") != context.dataset.get("fingerprint"):
        raise CompatibilityError(
            "dataset artifact fingerprint changed after task binding"
        )


def _plan_evaluation(
    orchestrator,
    context: RunContext,
) -> tuple[dict, Path, dict]:
    plan = orchestrator._invoke(
        context.evaluator,
        "plan_evaluate",
        {
            "service": context.service,
            "task": context.task,
            "evaluation": context.evaluation,
            "cache_root": str(orchestrator.cache_root),
            "output_root": str(context.run_dir / ".run" / "framework_output"),
            "workspace": str(context.run_dir),
            "log_path": str(context.run_dir / "logs" / "evaluation.log"),
            "network_policy": "offline" if context.offline else "online",
        },
        context={
            "workspace": str(context.run_dir),
            "cache_root": str(orchestrator.cache_root),
            "offline": context.offline,
        },
        timeout=30,
    )
    raw_root = orchestrator._confined_path(
        plan["raw_result_root"],
        context.run_dir / ".run" / "framework_output",
        label="evaluator raw_result_root",
    )
    plan["raw_result_root"] = str(raw_root)
    process = plan["process"]
    process.setdefault("metadata", {}).update(
        {"role": "evaluator", "run_id": context.run_id}
    )
    process, _ = orchestrator.prepare_process_for_environment(
        process,
        platform_spec=context.platform,
        resolved_platform=context.resolved_platform,
        role="evaluator",
        context={"workspace": str(context.run_dir), "offline": context.offline},
        timeout=5,
    )
    return plan, raw_root, process


def _run_evaluator(orchestrator, context: RunContext, process: dict) -> dict:
    orchestrator._status(context.run_dir, "EVALUATING")
    started_at = iso_now(context.plan)
    clock = time.monotonic()
    completed = orchestrator.pm.run(process)
    timing = {
        "started_at": started_at,
        "finished_at": iso_now(context.plan),
        "duration_seconds": time.monotonic() - clock,
    }
    context.evaluator_returncode = completed.returncode
    if completed.returncode != 0:
        raise ProcessError(f"evaluator exited with rc={completed.returncode}")
    return timing


def _result_model(context: RunContext) -> str:
    return str(
        context.model.get("experiment_id")
        or (context.model.get("metadata") or {}).get("experiment_id")
        or context.model["id"]
    )


def _normalize_result(
    orchestrator,
    context: RunContext,
    evaluation_plan: dict,
    result_model: str,
) -> dict:
    orchestrator._status(context.run_dir, "NORMALIZING")
    return orchestrator._invoke(
        context.evaluator,
        "normalize",
        {
            "raw_result_root": evaluation_plan["raw_result_root"],
            "task": context.task,
            "run_metadata": {
                "run_id": context.run_id,
                "model": result_model,
                "benchmark": context.benchmark["id"],
            },
        },
        context={"workspace": str(context.run_dir)},
        timeout=20,
    )


def _validate_result(
    orchestrator,
    context: RunContext,
    result: dict,
    raw_root: Path,
    result_model: str,
) -> str:
    expected_framework = context.evaluation["framework"]["adapter"]
    identity = (
        result.get("run_id"),
        result.get("model"),
        result.get("benchmark"),
        result.get("framework"),
    )
    expected = (
        context.run_id,
        result_model,
        context.benchmark["id"],
        expected_framework,
    )
    if identity != expected:
        raise CompatibilityError(
            "CanonicalResult identity disagrees with the executing run/evaluator"
        )

    task_metrics = context.task.get("metrics") or {}
    required_metrics = context.benchmark.get("metrics") or []
    if task_metrics.get("namespace") == "canonical":
        missing = [
            name for name in required_metrics if name not in (result.get("metrics") or {})
        ]
        if missing:
            raise CompatibilityError(
                f"CanonicalResult is missing BenchmarkSpec metrics: {missing}"
            )
    orchestrator._verify_canonical_raw_result(result, raw_root)
    return expected_framework


def _publish(
    orchestrator,
    context: RunContext,
    result: dict,
    raw_root: Path,
    result_model: str,
    framework: str,
    timing: dict,
) -> None:
    token_usage = result.pop("token_usage", None)
    performance = build_performance(
        run_id=context.run_id,
        model=result_model,
        benchmark=context.benchmark["id"],
        framework=framework,
        evaluation_started_at=timing["started_at"],
        evaluation_finished_at=timing["finished_at"],
        evaluation_duration_seconds=timing["duration_seconds"],
        token_usage=token_usage,
    )
    result["metadata"] = {
        **(result.get("metadata") or {}),
        "started_at": context.started_at,
        "finished_at": iso_now(context.plan),
        "timezone": getattr(plan_timezone(context.plan), "key", "Asia/Shanghai"),
    }
    publish_result(
        context.run_dir,
        raw_root,
        result,
        performance=performance,
        schemas=orchestrator.schemas,
    )


def evaluate_and_publish(orchestrator, context: RunContext) -> None:
    _check_runtime_compatibility(orchestrator, context)
    _verify_bound_inputs(orchestrator, context)
    evaluation_plan, raw_root, process = _plan_evaluation(orchestrator, context)
    timing = _run_evaluator(orchestrator, context, process)
    result_model = _result_model(context)
    result = _normalize_result(
        orchestrator,
        context,
        evaluation_plan,
        result_model,
    )
    framework = _validate_result(
        orchestrator,
        context,
        result,
        raw_root,
        result_model,
    )
    _publish(
        orchestrator,
        context,
        result,
        raw_root,
        result_model,
        framework,
        timing,
    )


__all__ = ["evaluate_and_publish"]
