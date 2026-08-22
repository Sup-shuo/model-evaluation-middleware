from __future__ import annotations

import copy
import shutil
import tempfile
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.files import atomic_json
from model_evaluation.core.identifiers import stable_id


EXPORT_STRATEGIES = ("round_robin", "resource_balanced")


def _resolved_spec(plan: dict, name: str) -> dict:
    return copy.deepcopy(
        ((((plan.get("resolved") or {}).get("specs") or {}).get(name)) or {})
    )


def _logical_requirements(plan: dict) -> dict:
    platform = _resolved_spec(plan, "platform")
    deployment = _resolved_spec(plan, "deployment")
    resolved_platform = (plan.get("resolved") or {}).get("platform") or {}
    device = platform.get("device") or {}
    runtime = platform.get("runtime") or {}
    selected_devices = device.get("devices")
    device_claims = [claim for claim in (plan.get("resources") or []) if claim.get("kind") == "device"]
    if isinstance(selected_devices, list):
        device_count = len(selected_devices)
    else:
        device_count = len(device_claims)

    claim_counts = Counter(str(claim.get("kind")) for claim in (plan.get("resources") or []))
    exclusive_counts = Counter(
        str(claim.get("kind"))
        for claim in (plan.get("resources") or [])
        if claim.get("exclusive", True)
    )
    claim_summary = [
        {
            "kind": kind,
            "count": count,
            "exclusive_count": exclusive_counts.get(kind, 0),
        }
        for kind, count in sorted(claim_counts.items())
    ]
    accelerator = {"count": device_count}
    adapter = device.get("adapter") or device.get("vendor")
    if adapter:
        accelerator["type"] = str(adapter)
    runtime_family = runtime.get("adapter") or runtime.get("family")
    if runtime_family:
        accelerator["runtime_family"] = str(runtime_family)
    environments = {}
    for role, key in (
        ("backend", "backend_environment"),
        ("evaluator", "evaluation_environment"),
    ):
        descriptor = resolved_platform.get(key)
        if isinstance(descriptor, dict):
            environments[role] = {
                "provider": str(descriptor.get("provider") or "unspecified"),
                "capability_id": "env-" + stable_id(descriptor, length=24),
            }
    if "evaluator" not in environments:
        descriptor = {"provider": "unspecified", "role": "evaluator"}
        environments["evaluator"] = {
            "provider": "unspecified",
            "capability_id": "env-" + stable_id(descriptor, length=24),
        }
    return {
        "accelerator": accelerator,
        "backend": {
            "adapter": str(((deployment.get("backend") or {}).get("adapter")) or "unspecified"),
            "management_mode": str(((deployment.get("management") or {}).get("mode")) or "unspecified"),
        },
        "environments": environments,
        "claims": claim_summary,
    }


def _intent(plan: dict) -> dict:
    run = plan.get("run_spec") or {}
    model = _resolved_spec(plan, "model")
    benchmark = _resolved_spec(plan, "benchmark")
    evaluation = _resolved_spec(plan, "evaluation")
    deployment = _resolved_spec(plan, "deployment")
    return {
        "model": str(model.get("experiment_id") or run.get("model") or "unspecified"),
        "benchmark": str(benchmark.get("id") or run.get("benchmark") or "unspecified"),
        "backend": str(((deployment.get("backend") or {}).get("adapter")) or run.get("deployment") or "unspecified"),
        "evaluator": str(((evaluation.get("framework") or {}).get("adapter")) or run.get("evaluation") or "unspecified"),
    }


def build_scheduler_jobs(matrix_plan: dict) -> list[dict]:
    matrix_id = str(matrix_plan.get("matrix_id") or "")
    jobs = []
    for index, plan in enumerate(matrix_plan.get("plans") or [], 1):
        requirements = _logical_requirements(plan)
        accelerator_count = int(requirements["accelerator"]["count"])
        weight = max(1, accelerator_count)
        plan_id = str(plan["plan_id"])
        jobs.append(
            {
                "schema_version": "1.1",
                "job_id": "job-" + stable_id(
                    {"source_matrix_id": matrix_id, "plan_id": plan_id}, length=24
                ),
                "source_matrix_id": matrix_id,
                "plan_id": plan_id,
                "plan_path": f"plans/{index:06d}-{plan_id}.json",
                "intent": _intent(plan),
                "requirements": requirements,
                "resource_weight": weight,
                "compatibility": str((plan.get("compatibility") or {}).get("status") or "unknown"),
            }
        )
    return jobs


CompatibilityKey = tuple[str, str, str, str, str, str, str]


def _compatibility_key(job: dict) -> CompatibilityKey:
    requirements = job["requirements"]
    accelerator = requirements["accelerator"]
    backend = requirements["backend"]
    environments = requirements.get("environments") or {}
    backend_environment = environments.get("backend") or {}
    evaluator_environment = environments.get("evaluator") or {}
    return (
        str(accelerator.get("type") or "unspecified"),
        str(accelerator.get("runtime_family") or "unspecified"),
        str(backend.get("adapter") or "unspecified"),
        str(backend.get("management_mode") or "unspecified"),
        str((job.get("intent") or {}).get("evaluator") or "unspecified"),
        str(backend_environment.get("capability_id") or "not-required"),
        str(evaluator_environment.get("capability_id") or "unspecified"),
    )


def _group_shard_counts(
    groups: dict[CompatibilityKey, list[tuple[int, dict]]],
    shards: int,
) -> dict[CompatibilityKey, int]:
    if shards < len(groups):
        raise ConfigError(
            "matrix export requires at least one shard per execution compatibility "
            f"group: shards={shards}, compatibility_groups={len(groups)}"
        )
    counts = {key: 1 for key in groups}
    remaining = shards - len(groups)
    while remaining:
        eligible = [key for key, rows in groups.items() if counts[key] < len(rows)]
        if not eligible:
            raise ConfigError("matrix export cannot allocate the requested shard count")
        selected = min(
            eligible,
            key=lambda key: (
                -ceil(
                    sum(int(job["resource_weight"]) for _, job in groups[key])
                    / counts[key]
                ),
                -sum(int(job["resource_weight"]) for _, job in groups[key]),
                key,
            ),
        )
        counts[selected] += 1
        remaining -= 1
    return counts


def _assign_compatible_group(
    rows: list[tuple[int, dict]],
    shards: int,
    strategy: str,
) -> list[list[tuple[int, dict]]]:
    assignments: list[list[tuple[int, dict]]] = [[] for _ in range(shards)]
    if strategy == "round_robin":
        for position, row in enumerate(rows):
            assignments[position % shards].append(row)
        return assignments

    loads = [0 for _ in range(shards)]
    ordered = sorted(
        rows,
        key=lambda item: (-int(item[1]["resource_weight"]), item[0]),
    )
    for index, job in ordered:
        shard_index = min(range(shards), key=lambda current: (loads[current], current))
        assignments[shard_index].append((index, job))
        loads[shard_index] += int(job["resource_weight"])
    for rows in assignments:
        rows.sort(key=lambda item: item[0])
    return assignments


def _assign_jobs(jobs: list[dict], shards: int, strategy: str) -> list[list[tuple[int, dict]]]:
    groups: dict[CompatibilityKey, list[tuple[int, dict]]] = defaultdict(list)
    for index, job in enumerate(jobs, 1):
        groups[_compatibility_key(job)].append((index, job))

    counts = _group_shard_counts(groups, shards)
    assignments: list[list[tuple[int, dict]]] = []
    for key in sorted(groups):
        assignments.extend(
            _assign_compatible_group(groups[key], counts[key], strategy)
        )
    return assignments


def _shard_requirements(rows: list[tuple[int, dict]]) -> dict:
    accelerators = [row[1]["requirements"]["accelerator"] for row in rows]
    key = _compatibility_key(rows[0][1])
    return {
        "resource_weight": sum(int(row[1]["resource_weight"]) for row in rows),
        "max_accelerator_count": max((int(item.get("count", 0)) for item in accelerators), default=0),
        "accelerator_types": sorted({str(item["type"]) for item in accelerators if item.get("type")}),
        "runtime_families": sorted(
            {str(item["runtime_family"]) for item in accelerators if item.get("runtime_family")}
        ),
        "execution_compatibility": {
            "accelerator_type": key[0],
            "runtime_family": key[1],
            "backend_adapter": key[2],
            "management_mode": key[3],
            "evaluator_adapter": key[4],
            "backend_environment": key[5],
            "evaluator_environment": key[6],
        },
    }


def build_execution_export(
    matrix_plan: dict,
    *,
    shards: int,
    strategy: str = "round_robin",
) -> tuple[dict, list[dict]]:
    plans = matrix_plan.get("plans") or []
    if not isinstance(shards, int) or isinstance(shards, bool) or shards < 1:
        raise ConfigError("matrix export shards must be a positive integer")
    if shards > len(plans):
        raise ConfigError(
            f"matrix export shards={shards} exceeds plan count={len(plans)}"
        )
    if strategy not in EXPORT_STRATEGIES:
        raise ConfigError(f"unsupported matrix export strategy: {strategy!r}")

    jobs = build_scheduler_jobs(matrix_plan)
    export_id = "matrix-export-" + stable_id(
        {
            "source_matrix_id": matrix_plan.get("matrix_id"),
            "shards": shards,
            "strategy": strategy,
            "jobs": [(job["job_id"], job["resource_weight"]) for job in jobs],
        },
        length=24,
    )
    assignments = _assign_jobs(jobs, shards, strategy)

    shard_documents = []
    shard_rows = []
    for shard_index, rows in enumerate(assignments, 1):
        path = f"shards/shard-{shard_index:04d}.json"
        requirements = _shard_requirements(rows)
        plan_rows = [
            {
                "index": index,
                "plan_id": job["plan_id"],
                "path": job["plan_path"],
                "job_id": job["job_id"],
                "job_path": f"jobs/{index:06d}-{job['job_id']}.json",
                "resource_weight": job["resource_weight"],
            }
            for index, job in rows
        ]
        document = {
            "schema_version": "1.2",
            "export_id": export_id,
            "source_matrix_id": matrix_plan["matrix_id"],
            "strategy": strategy,
            "shard_index": shard_index,
            "shard_count": shards,
            "requirements": requirements,
            "plans": plan_rows,
        }
        shard_documents.append(document)
        shard_rows.append(
            {
                "index": shard_index,
                "path": path,
                "plan_count": len(rows),
                "plan_ids": [job["plan_id"] for _, job in rows],
                "requirements": requirements,
            }
        )

    manifest = {
        "schema_version": "1.2",
        "export_id": export_id,
        "source_matrix_id": matrix_plan["matrix_id"],
        "strategy": strategy,
        "plan_count": len(plans),
        "job_count": len(jobs),
        "shard_count": shards,
        "shards": shard_rows,
    }
    return manifest, shard_documents


def export_execution_plans(
    matrix_plan: dict,
    output_dir: str | Path,
    *,
    shards: int,
    schemas,
    strategy: str = "round_robin",
) -> dict:
    """Export exact plans plus scheduler-neutral logical job descriptors.

    Job descriptors contain no physical device IDs. The exact plans remain in
    the bundle for compatible workers and are still executable with
    ``eval-manager run-plan``. This function never submits work.
    """

    manifest, shard_documents = build_execution_export(
        matrix_plan, shards=shards, strategy=strategy
    )
    jobs = build_scheduler_jobs(matrix_plan)
    schemas.validate("matrix_execution_export", manifest)
    for document in shard_documents:
        schemas.validate("matrix_execution_shard", document)
    for job in jobs:
        schemas.validate("matrix_scheduler_job", job)

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ConfigError(f"matrix export output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "plans").mkdir()
        (temporary / "jobs").mkdir()
        (temporary / "shards").mkdir()
        for index, plan in enumerate(matrix_plan["plans"], 1):
            plan_id = str(plan["plan_id"])
            atomic_json(temporary / "plans" / f"{index:06d}-{plan_id}.json", plan)
        for index, job in enumerate(jobs, 1):
            atomic_json(temporary / "jobs" / f"{index:06d}-{job['job_id']}.json", job)
        for document in shard_documents:
            atomic_json(
                temporary / "shards" / f"shard-{document['shard_index']:04d}.json",
                document,
            )
        atomic_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "output_dir": str(destination)}
