from __future__ import annotations

import copy

from model_evaluation.core.compatibility import (
    evaluate,
    facts_from_environment,
    facts_from_service,
    merge_fact_sets,
)
from model_evaluation.core.errors import CompatibilityError, ModelEvalError, ProcessError
from model_evaluation.core.files import atomic_json
from model_evaluation.core.results import iso_now, plan_timezone, publish_result
from model_evaluation.core.run_lifecycle import RunContext


def prepare_platform(orchestrator, context: RunContext) -> None:
    platform = context.platform
    resolved = orchestrator._revalidate_platform(context.plan, context.run_dir)
    context.resolved_platform = resolved
    versions = orchestrator._runtime_versions_base(context.plan, platform, resolved)
    orchestrator._refresh_environment_versions(platform, resolved, versions)
    context.runtime_versions = versions
    orchestrator._save_runtime_versions(context.run_dir, versions)
    orchestrator._status(context.run_dir, "PLATFORM_READY")


def prepare_dataset_and_task(orchestrator, context: RunContext) -> None:
    benchmark = context.benchmark
    evaluation = context.evaluation
    dataset_client = orchestrator.registry.get(
        "dataset", benchmark["dataset"]["provider"]
    )
    dataset = orchestrator._invoke(
        dataset_client,
        "prepare",
        {"benchmark": benchmark, "cache_root": str(orchestrator.cache_root)},
        context={
            "cache_root": str(orchestrator.cache_root),
            "workspace": str(context.run_dir / ".run" / "dataset"),
            "offline": context.offline,
            "network_policy": "offline" if context.offline else "online",
        },
        timeout=float(
            (context.plan["run_spec"].get("overrides") or {}).get(
                "dataset_timeout_seconds", 600
            )
        ),
    )
    verified = orchestrator._invoke(
        dataset_client,
        "verify",
        {"artifact": dataset, "benchmark": benchmark},
        context={"offline": True},
        timeout=30,
    )
    if not verified["valid"]:
        raise ModelEvalError(
            f"dataset verification failed: {verified.get('details') or {}}"
        )
    dataset = verified.get("artifact") or dataset
    orchestrator._verify_dataset_identity(
        context.plan["resolved"].get("dataset_resolution") or {},
        benchmark,
        dataset,
    )
    context.dataset_client = dataset_client
    context.dataset = dataset
    orchestrator._status(context.run_dir, "DATA_READY")

    binding = orchestrator.registry.get(
        "binding", context.plan["resolved"]["binding_adapter"]
    )
    task_input = {
        "benchmark": benchmark,
        "dataset_artifact": dataset,
        "staging_root": str(context.run_dir / ".run" / "task"),
        "evaluation": evaluation,
    }
    task = orchestrator._invoke(
        binding,
        "build_task",
        task_input,
        context={
            "workspace": str(context.run_dir / ".run" / "task"),
            "offline": context.offline,
        },
        timeout=60,
    )
    fingerprint = orchestrator._invoke(
        binding,
        "protocol_fingerprint",
        {
            "benchmark": benchmark,
            "dataset_artifact": dataset,
            "evaluation": evaluation,
        },
        context={
            "workspace": str(context.run_dir / ".run" / "task"),
            "offline": context.offline,
        },
        timeout=30,
    )
    if fingerprint["protocol_fingerprint"] != task["protocol_fingerprint"]:
        raise CompatibilityError(
            "binding protocol_fingerprint disagrees with FrameworkTaskArtifact: "
            f"{fingerprint['protocol_fingerprint']} != {task['protocol_fingerprint']}"
        )
    orchestrator._verify_task_artifacts(task, context.run_dir / ".run" / "task")
    context.task = task
    orchestrator._status(context.run_dir, "TASK_READY")


def _preflight_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def preflight_evaluator(orchestrator, context: RunContext) -> None:
    orchestrator._status(context.run_dir, "EVALUATOR_PREFLIGHT")
    evaluator = orchestrator.registry.get(
        "evaluator", context.evaluation["framework"]["adapter"]
    )
    context.evaluator = evaluator
    context.runtime_versions["evaluator"] = {
        "adapter": context.evaluation["framework"]["adapter"],
        "adapter_version": evaluator.identity.version,
    }
    orchestrator._save_runtime_versions(context.run_dir, context.runtime_versions)
    requirements = orchestrator._invoke(
        evaluator,
        "requirements",
        {"evaluation": context.evaluation, "task": context.task},
        context={
            "workspace": str(context.run_dir),
            "offline": context.offline,
            "preflight": True,
        },
        timeout=30,
    )
    context.evaluator_requirements = requirements
    local_requirements = {
        "schema_version": "1.0",
        "requirements": [
            row
            for row in requirements.get("requirements", [])
            if str(row.get("path", "")).startswith("evaluation_environment.")
        ],
    }
    orchestrator.schemas.validate("requirement_set", local_requirements)
    local_report = evaluate(
        local_requirements,
        facts_from_environment(
            context.resolved_platform["evaluation_environment"],
            "evaluation_environment",
        ),
    )
    if not local_report.compatible:
        raise CompatibilityError(
            "evaluator preflight failed: " + "; ".join(local_report.reasons),
            details={"diagnostics": local_report.diagnostics},
        )

    if "plan_preflight" not in (evaluator.identity.manifest.get("operations") or []):
        return
    preflight = orchestrator._invoke(
        evaluator,
        "plan_preflight",
        {
            "evaluation": context.evaluation,
            "task": context.task,
            "cache_root": str(orchestrator.cache_root),
        },
        context={
            "workspace": str(context.run_dir),
            "cache_root": str(orchestrator.cache_root),
            "offline": context.offline,
            "preflight": True,
        },
        timeout=30,
    )
    process = copy.deepcopy(preflight["process"])
    process.setdefault("metadata", {}).update(
        {"role": "evaluator_preflight", "run_id": context.run_id}
    )
    wrapped, _ = orchestrator.prepare_process_for_environment(
        process,
        platform_spec=context.platform,
        resolved_platform=context.resolved_platform,
        role="evaluator",
        context={
            "workspace": str(context.run_dir),
            "offline": context.offline,
            "preflight": True,
        },
        timeout=5,
    )
    completed = orchestrator.pm.run(wrapped)
    record = {
        "returncode": completed.returncode,
        "stdout": _preflight_text(completed.stdout)[:8000],
        "stderr": _preflight_text(completed.stderr)[:8000],
    }
    diagnostic_path = (
        context.run_dir / ".run" / "diagnostics" / "evaluator_preflight.json"
    )
    if preflight.get("result_format") == "preflight_result":
        try:
            probe = orchestrator._preflight_json_result(
                _preflight_text(completed.stdout)
            )
            orchestrator.schemas.validate("preflight_probe_result", probe)
            record["result"] = probe
            context.runtime_versions["evaluator"]["facts"] = copy.deepcopy(
                probe.get("facts") or {}
            )
            orchestrator._save_runtime_versions(
                context.run_dir, context.runtime_versions
            )
            process_passed = completed.returncode == 0
            result_passed = probe["status"] == "passed"
            if process_passed != result_passed:
                raise ProcessError(
                    "evaluator preflight process/result status mismatch: "
                    f"returncode={completed.returncode}, result.status={probe['status']!r}"
                )
            if not result_passed:
                error = probe["error"]
                raise ProcessError(
                    f"evaluator preflight failed: {error['code']}: {error['message']}"
                )
        finally:
            atomic_json(diagnostic_path, orchestrator._redact_diagnostic(record))
    else:
        atomic_json(diagnostic_path, orchestrator._redact_diagnostic(record))
        if completed.returncode != 0:
            raise ProcessError(
                f"evaluator dependency preflight failed with rc={completed.returncode}"
            )


def start_service(orchestrator, context: RunContext) -> None:
    deployment = context.deployment
    backend = orchestrator.registry.get("backend", deployment["backend"]["adapter"])
    start_plan = orchestrator._invoke(
        backend,
        "plan_start",
        {
            "model": context.model,
            "deployment": deployment,
            "platform": context.resolved_platform,
            "endpoint": context.plan["resolved"].get("endpoint", {}),
            "log_path": str(context.run_dir / "logs" / "backend.log"),
            "network_policy": "offline" if context.offline else "online",
        },
        context={"workspace": str(context.run_dir), "offline": context.offline},
        timeout=5,
    )
    attach = start_plan["attach"]
    context.backend_shutdown = copy.deepcopy(start_plan.get("shutdown"))
    if context.mode == "managed":
        orchestrator._status(context.run_dir, "SERVICE_STARTING")
        process = copy.deepcopy(start_plan["process"])
        process.setdefault("metadata", {}).update(
            {"role": "backend", "run_id": context.run_id}
        )
        resolved = context.resolved_platform
        process, _ = orchestrator.prepare_process_for_environment(
            process,
            platform_spec=context.platform,
            resolved_platform=resolved,
            role="backend",
            base_patches=(
                ("device", resolved.get("device_env_patch")),
                ("runtime", resolved.get("runtime_env_patch")),
            ),
            context={"workspace": str(context.run_dir), "offline": context.offline},
            timeout=5,
        )
        if "plan_preflight" in (backend.identity.manifest.get("operations") or []):
            preflight_plan = orchestrator._invoke(
                backend,
                "plan_preflight",
                {
                    "model": context.model,
                    "deployment": deployment,
                    "platform": resolved,
                    "network_policy": "offline" if context.offline else "online",
                },
                context={
                    "workspace": str(context.run_dir),
                    "offline": context.offline,
                    "preflight": True,
                },
                timeout=5,
            )
            report = orchestrator.run_backend_preflight(
                preflight_plan,
                platform_spec=context.platform,
                resolved_platform=resolved,
                raise_on_failure=False,
            )
            backend_runtime = {
                "adapter": deployment["backend"]["adapter"],
                "adapter_version": backend.identity.version,
                "probes": [],
            }
            for row in report.get("probes") or []:
                compact = {"id": row.get("id"), "status": row.get("status")}
                version = orchestrator._version_text(row.get("stdout"))
                if version:
                    compact["version"] = version
                facts = (row.get("result") or {}).get("facts")
                if isinstance(facts, dict):
                    compact["facts"] = copy.deepcopy(facts)
                backend_runtime["probes"].append(compact)
            context.runtime_versions["backend"] = backend_runtime
            orchestrator._save_runtime_versions(
                context.run_dir, context.runtime_versions
            )
            if report.get("status") != "passed":
                raise orchestrator._backend_preflight_error(report)
        else:
            dependency = orchestrator._run_backend_dependency_probe(
                context.run_dir,
                probe_spec=start_plan.get("dependency_probe"),
                platform_spec=context.platform,
                resolved_platform=resolved,
            )
            context.runtime_versions["backend"] = {
                "adapter": deployment["backend"]["adapter"],
                "adapter_version": backend.identity.version,
                "version": orchestrator._version_text((dependency or {}).get("stdout")),
            }
        orchestrator._save_runtime_versions(context.run_dir, context.runtime_versions)
        for claim in context.plan["resources"]:
            if claim["kind"] == "port":
                orchestrator.resources.check_port(
                    str(claim.get("host") or "127.0.0.1"), int(claim["id"])
                )
        context.backend_handle = orchestrator.pm.start(process)

    auth_ref = (attach.get("auth") or {}).get("secret_ref")
    auth_value = orchestrator.pm.secrets.resolve(auth_ref) if auth_ref else None
    ready = float(
        (start_plan.get("readiness") or {}).get(
            "timeout_seconds", 30 if context.mode != "managed" else 900
        )
    )
    context.service = orchestrator._probe_service_until_ready(
        backend,
        attach,
        auth_value,
        ready,
        context.backend_handle,
    )
    orchestrator._status(context.run_dir, "SERVICE_READY")
    if context.mode != "managed":
        context.runtime_versions["backend"] = {
            "adapter": deployment["backend"]["adapter"],
            "adapter_version": backend.identity.version,
            "management": context.mode,
        }
    orchestrator._save_runtime_versions(context.run_dir, context.runtime_versions)


def evaluate_and_publish(orchestrator, context: RunContext) -> None:
    facts = merge_fact_sets(
        facts_from_service(context.service),
        facts_from_environment(
            context.resolved_platform["evaluation_environment"],
            "evaluation_environment",
        ),
    )
    report = evaluate(context.evaluator_requirements, facts)
    if not report.compatible:
        raise CompatibilityError(
            "; ".join(report.reasons),
            details={"diagnostics": report.diagnostics},
        )

    orchestrator._verify_task_artifacts(
        context.task, context.run_dir / ".run" / "task"
    )
    final_verified = orchestrator._invoke(
        context.dataset_client,
        "verify",
        {"artifact": context.dataset, "benchmark": context.benchmark},
        context={"offline": True, "final_verification": True},
        timeout=30,
    )
    if not final_verified["valid"]:
        raise ModelEvalError(
            f"dataset final verification failed: {final_verified.get('details') or {}}"
        )
    final_artifact = final_verified.get("artifact") or context.dataset
    orchestrator._verify_dataset_identity(
        context.plan["resolved"].get("dataset_resolution") or {},
        context.benchmark,
        final_artifact,
    )
    if final_artifact.get("fingerprint") != context.dataset.get("fingerprint"):
        raise CompatibilityError(
            "dataset artifact fingerprint changed after task binding"
        )

    evaluation_plan = orchestrator._invoke(
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
        evaluation_plan["raw_result_root"],
        context.run_dir / ".run" / "framework_output",
        label="evaluator raw_result_root",
    )
    evaluation_plan["raw_result_root"] = str(raw_root)
    process = evaluation_plan["process"]
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
    orchestrator._status(context.run_dir, "EVALUATING")
    completed = orchestrator.pm.run(process)
    context.evaluator_returncode = completed.returncode
    if completed.returncode != 0:
        raise ProcessError(f"evaluator exited with rc={completed.returncode}")

    orchestrator._status(context.run_dir, "NORMALIZING")
    result_model = str(
        context.model.get("experiment_id")
        or (context.model.get("metadata") or {}).get("experiment_id")
        or context.model["id"]
    )
    result = orchestrator._invoke(
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
    expected_framework = context.evaluation["framework"]["adapter"]
    if (
        result.get("run_id") != context.run_id
        or result.get("model") != result_model
        or result.get("benchmark") != context.benchmark["id"]
        or result.get("framework") != expected_framework
    ):
        raise CompatibilityError(
            "CanonicalResult identity disagrees with the executing run/evaluator"
        )
    task_metrics = context.task.get("metrics") or {}
    if task_metrics.get("namespace") == "canonical":
        missing = [
            name
            for name in (context.benchmark.get("metrics") or [])
            if name not in (result.get("metrics") or {})
        ]
        if missing:
            raise CompatibilityError(
                f"CanonicalResult is missing BenchmarkSpec metrics: {missing}"
            )
    orchestrator._verify_canonical_raw_result(result, raw_root)
    result["metadata"] = {
        **(result.get("metadata") or {}),
        "started_at": context.started_at,
        "finished_at": iso_now(context.plan),
        "timezone": getattr(plan_timezone(context.plan), "key", "Asia/Shanghai"),
    }
    publish_result(context.run_dir, raw_root, result, schemas=orchestrator.schemas)


__all__ = [
    "evaluate_and_publish",
    "prepare_dataset_and_task",
    "prepare_platform",
    "preflight_evaluator",
    "start_service",
]
