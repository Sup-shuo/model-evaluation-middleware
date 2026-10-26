"""MUSA toolkit and driver discovery without importing the inference environment."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from model_evaluation.sdk.runtime import AdapterError


def _root(inputs: dict) -> Path:
    params = inputs.get("parameters") or {}
    value = params.get("root") or os.environ.get("MUSA_HOME") or "/usr/local/musa"
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise AdapterError("CONFIG_INVALID", "MUSA root must be absolute")
    if not (path / "include" / "musa.h").is_file() or not (path / "lib").is_dir():
        raise AdapterError("DEPENDENCY_MISSING", f"MUSA toolkit incomplete: {path}")
    return path.resolve()


def probe(inputs: dict, context: dict) -> dict:
    root = _root(inputs)
    header = (root / "include" / "musa.h").read_text(encoding="utf-8")
    match = re.search(r"^#define\s+MUSA_VERSION\s+(\d+)", header, re.MULTILINE)
    if not match:
        raise AdapterError("DEPENDENCY_MISSING", "MUSA_VERSION missing from musa.h")
    version = int(match.group(1))
    executable = shutil.which("mthreads-gmi")
    if not executable:
        raise AdapterError("DEPENDENCY_MISSING", "mthreads-gmi not found")
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
            timeout=float(context.get("timeout_seconds", 5)), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError("RESOURCE_UNAVAILABLE", "MUSA driver probe timed out",
                           retryable=True) from exc
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or not versions:
        raise AdapterError("RESOURCE_UNAVAILABLE", "MUSA driver probe failed", retryable=True)
    return {
        "schema_version": "1.0", "family": "musa",
        "version": f"{version // 10000}.{version // 100 % 100}.{version % 100}",
        "driver_version": versions[0], "available": True,
        "capabilities": {"schema_version": "1.0",
                         "values": {"runtime.compatible_device_vendors": ["mthreads"]}},
        "env_patch": {},
    }


def resolve_environment(inputs: dict, context: dict) -> dict:
    root = _root(inputs)
    return {"env_patch": {
        "set": {"MUSA_HOME": str(root)},
        "prepend_path": {
            "PATH": [str(root / "bin")],
            "LD_LIBRARY_PATH": [str(root / "lib")],
        },
    }}


def snapshot(inputs: dict, context: dict) -> dict:
    return probe(inputs, context)


OPERATIONS = {"probe": probe, "resolve_environment": resolve_environment, "snapshot": snapshot}
