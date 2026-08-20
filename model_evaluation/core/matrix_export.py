from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.files import atomic_json
from model_evaluation.core.identifiers import stable_id


def build_execution_export(matrix_plan: dict, *, shards: int) -> tuple[dict, list[dict]]:
    plans = matrix_plan.get("plans") or []
    if not isinstance(shards, int) or isinstance(shards, bool) or shards < 1:
        raise ConfigError("matrix export shards must be a positive integer")
    if shards > len(plans):
        raise ConfigError(
            f"matrix export shards={shards} exceeds plan count={len(plans)}"
        )

    export_id = "matrix-export-" + stable_id(
        {
            "source_matrix_id": matrix_plan.get("matrix_id"),
            "shards": shards,
            "plan_ids": [plan.get("plan_id") for plan in plans],
        },
        length=24,
    )
    assignments: list[list[dict]] = [[] for _ in range(shards)]
    for index, plan in enumerate(plans, 1):
        plan_id = str(plan["plan_id"])
        assignments[(index - 1) % shards].append(
            {
                "index": index,
                "plan_id": plan_id,
                "path": f"plans/{index:06d}-{plan_id}.json",
            }
        )

    shard_documents = []
    shard_rows = []
    for index, rows in enumerate(assignments, 1):
        path = f"shards/shard-{index:04d}.json"
        document = {
            "schema_version": "1.0",
            "export_id": export_id,
            "source_matrix_id": matrix_plan["matrix_id"],
            "shard_index": index,
            "shard_count": shards,
            "plans": copy.deepcopy(rows),
        }
        shard_documents.append(document)
        shard_rows.append(
            {
                "index": index,
                "path": path,
                "plan_count": len(rows),
                "plan_ids": [row["plan_id"] for row in rows],
            }
        )

    manifest = {
        "schema_version": "1.0",
        "export_id": export_id,
        "source_matrix_id": matrix_plan["matrix_id"],
        "strategy": "round_robin",
        "plan_count": len(plans),
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
) -> dict:
    """Export exact child plans for consumption by an external scheduler.

    The bundle is scheduler-neutral.  Each plan remains executable with the
    existing ``eval-manager run-plan`` command; this function does not submit
    work or implement distributed scheduling.
    """

    manifest, shard_documents = build_execution_export(matrix_plan, shards=shards)
    schemas.validate("matrix_execution_export", manifest)
    for document in shard_documents:
        schemas.validate("matrix_execution_shard", document)

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ConfigError(f"matrix export output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        (temporary / "plans").mkdir()
        (temporary / "shards").mkdir()
        for index, plan in enumerate(matrix_plan["plans"], 1):
            plan_id = str(plan["plan_id"])
            atomic_json(temporary / "plans" / f"{index:06d}-{plan_id}.json", plan)
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
