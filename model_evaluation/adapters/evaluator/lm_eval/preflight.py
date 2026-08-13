#!/usr/bin/env python3
"""Task-aware lm-eval preflight executed in the selected evaluator environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def _failure_code(exc: BaseException) -> str:
    chain = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 8:
        chain.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    message = "\n".join(chain).lower()
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "EVALUATOR_DEPENDENCY_MISSING"
    if isinstance(exc, KeyError) and ("registered task" in message or "valid yaml" in message):
        return "EVALUATION_TASK_UNAVAILABLE"
    offline_markers = (
        "offlinemodeisenabled",
        "offline mode",
        "couldn't reach",
        "cannot reach",
        "not found in cached",
        "not found in the cache",
        "local_files_only",
    )
    if any(marker in message for marker in offline_markers):
        return "EVALUATION_DATA_UNAVAILABLE"
    return "EVALUATOR_TASK_PREFLIGHT_FAILED"


def _run(payload: dict) -> dict:
    framework_root = Path(str(payload["framework_root"])).resolve()
    sys.path.insert(0, str(framework_root))
    import lm_eval

    facts: dict[str, object] = {
        "framework": "lm_eval",
        "framework_file": getattr(lm_eval, "__file__", None) or "lm_eval",
        "framework_version": getattr(lm_eval, "__version__", None),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "cache_root": payload.get("cache_root"),
    }
    if not facts["framework_version"]:
        for distribution in ("lm_eval", "lm-eval"):
            try:
                facts["framework_version"] = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    versions: dict[str, str] = {}
    for distribution in ("lm_eval", "lm-eval", "torch", "transformers", "datasets", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    facts["packages"] = versions
    task_id = payload.get("task_id")
    if not task_id:
        facts["scope"] = "dependency"
        return facts

    from lm_eval.tasks import TaskManager

    task_root = payload.get("task_root")
    manager = TaskManager(include_path=task_root if task_root else None)
    loaded = manager.load([str(task_id)])
    tasks = loaded.get("tasks") or {}
    groups = loaded.get("groups") or {}
    if not tasks:
        raise RuntimeError(f"lm-eval task/group {task_id!r} resolved without any leaf tasks")
    facts.update(
        {
            "scope": "task_and_data",
            "task_id": str(task_id),
            "task_root": str(Path(task_root).resolve()) if task_root else None,
            "leaf_task_count": len(tasks),
            "group_count": len(groups),
            "task_data_initialized": True,
            "weights_loaded": False,
        }
    )
    return facts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.payload)
        if not isinstance(payload, dict):
            raise TypeError("preflight payload must be a JSON object")
        facts = _run(payload)
        result = {"schema_version": "1.0", "status": "passed", "facts": facts}
        returncode = 0
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        result = {
            "schema_version": "1.0",
            "status": "failed",
            "error": {
                "code": _failure_code(exc),
                "message": message,
                "details": {"exception_type": type(exc).__name__},
            },
        }
        returncode = 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
