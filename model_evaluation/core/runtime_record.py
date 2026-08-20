from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.files import atomic_json


def runtime_versions_base(plan: dict, resolved_platform: dict) -> dict:
    """Build the compact environment record saved with a normal run result."""
    value = {
        "schema_version": "1.0",
        "adapters": copy.deepcopy(plan.get("adapters") or []),
        "device": copy.deepcopy(resolved_platform.get("device")),
        "runtime": copy.deepcopy(resolved_platform.get("runtime")),
        "environments": {
            "backend": copy.deepcopy(resolved_platform.get("backend_environment")),
            "evaluator": copy.deepcopy(resolved_platform.get("evaluation_environment")),
        },
    }
    if ((plan.get("resolved") or {}).get("management_mode")) != "managed":
        value.pop("device", None)
        value.pop("runtime", None)
        value["environments"].pop("backend", None)
    return value


def refresh_environment_versions(
    *,
    active_plan: dict,
    platform: dict,
    resolved_platform: dict,
    value: dict,
    registry,
    invoke: Callable,
) -> None:
    management = (active_plan.get("resolved") or {}).get("management_mode")
    for role, key in (
        ("backend", "backend_environment"),
        ("evaluator", "evaluation_environment"),
    ):
        if role == "backend" and management != "managed":
            continue
        selection = platform.get(key)
        if not isinstance(selection, dict):
            continue
        client = registry.get("environment", selection["provider"])
        try:
            snapshot = invoke(
                client,
                "snapshot",
                {
                    "profile": selection.get("profile"),
                    "parameters": adapter_parameters(platform, key),
                },
                context={"timeout_seconds": 5, "runtime_version_record": True},
                timeout=6,
                stage="runtime_version",
            )
        except Exception as exc:
            value.setdefault("warnings", []).append(
                f"{role} environment version probe failed: {type(exc).__name__}: {exc}"
            )
            continue
        value.setdefault("environments", {})[role] = {
            **copy.deepcopy(resolved_platform.get(key) or {}),
            **copy.deepcopy(snapshot),
        }


def version_text(value: object) -> str | None:
    text = (
        value.decode("utf-8", "replace")
        if isinstance(value, (bytes, bytearray))
        else str(value or "")
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:500] if lines else None


def save_runtime_versions(run_dir: Path, value: dict, *, redact: Callable[[object], object]) -> None:
    atomic_json(run_dir / "config" / "runtime_versions.json", redact(value))
