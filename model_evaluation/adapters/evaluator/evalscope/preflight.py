#!/usr/bin/env python3
"""EvalScope dependency probe executed in the selected evaluator environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path


def _distribution_version() -> str | None:
    try:
        return importlib.metadata.version("evalscope")
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_executable(value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"EvalScope executable not found: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(f"EvalScope executable not found on PATH: {value}")
    return resolved


def _run(payload: dict) -> dict:
    import evalscope

    executable = _resolve_executable(str(payload.get("executable") or "evalscope"))
    version = getattr(evalscope, "__version__", None) or _distribution_version()
    expected = payload.get("expected_version")
    if expected and version != expected:
        raise RuntimeError(
            f"EvalScope version mismatch: expected {expected}, got {version or 'unknown'}"
        )
    return {
        "framework": "evalscope",
        "framework_version": version,
        "framework_file": getattr(evalscope, "__file__", None) or "evalscope",
        "configured_executable": str(payload.get("executable") or "evalscope"),
        "resolved_executable": executable,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "task_id": payload.get("task_id"),
        "scope": "dependency",
        "weights_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.payload)
        if not isinstance(payload, dict):
            raise TypeError("preflight payload must be a JSON object")
        result = {"schema_version": "1.0", "status": "passed", "facts": _run(payload)}
        returncode = 0
    except Exception as exc:
        code = (
            "EVALUATOR_DEPENDENCY_MISSING"
            if isinstance(exc, (ModuleNotFoundError, ImportError, FileNotFoundError))
            else "EVALUATOR_VERSION_MISMATCH"
            if "version mismatch" in str(exc).lower()
            else "EVALUATOR_PREFLIGHT_FAILED"
        )
        result = {
            "schema_version": "1.0",
            "status": "failed",
            "error": {
                "code": code,
                "message": str(exc) or type(exc).__name__,
                "details": {"exception_type": type(exc).__name__},
            },
        }
        returncode = 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
