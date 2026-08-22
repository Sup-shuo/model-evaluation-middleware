from __future__ import annotations

import copy
from typing import Any, Callable


def compile_matrix_spec(
    *,
    system_name: str,
    platform_key: str,
    backend_profile_id: str,
    evaluator_profile_id: str,
    platform_id: str,
    deployment_id: str,
    evaluation_id: str,
    model_ids: list[str],
    benchmark_ids: list[str],
    evaluation: dict[str, Any],
    per_model_overrides: dict[str, dict[str, Any]],
    slug: Callable[[str, str], str],
    smoke: bool = False,
) -> dict[str, Any]:
    """Compile the user selections into the stable MatrixSpec protocol."""

    global_overrides: dict[str, Any] = {}
    if "offline" in evaluation:
        global_overrides["offline"] = bool(evaluation["offline"])
    if "dataset_timeout_seconds" in evaluation:
        global_overrides["dataset_timeout_seconds"] = evaluation[
            "dataset_timeout_seconds"
        ]
    tags = list(evaluation.get("tags") or ["user-config"])
    if smoke and "smoke" not in tags:
        tags.append("smoke")
    matrix: dict[str, Any] = {
        "schema_version": "1.0",
        "id": slug(
            f"{system_name}:{platform_key}:{backend_profile_id}:"
            f"{evaluator_profile_id}:evaluation",
            "user-matrix",
        ),
        "models": list(model_ids),
        "platforms": [platform_id],
        "deployments": [deployment_id],
        "benchmarks": list(benchmark_ids),
        "evaluations": [evaluation_id],
        "execution": {
            "mode": "serial",
            **copy.deepcopy(evaluation.get("execution") or {}),
        },
        "tags": tags,
    }
    if global_overrides:
        matrix["overrides"] = global_overrides
    if per_model_overrides:
        matrix["per_model_overrides"] = copy.deepcopy(per_model_overrides)
    return matrix


__all__ = ["compile_matrix_spec"]
