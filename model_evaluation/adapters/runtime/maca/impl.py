from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from model_evaluation.sdk.runtime import AdapterError


def _parameters(inputs: dict) -> dict:
    value = inputs.get("parameters") or {}
    if not isinstance(value, dict):
        raise AdapterError("CONFIG_INVALID", "runtime parameters must be an object")
    return value


def _absolute_parameter(inputs: dict, name: str) -> Path | None:
    configured = _parameters(inputs).get(name)
    if configured is None:
        return None
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        raise AdapterError(
            "CONFIG_INVALID",
            f"MACA runtime parameter {name} must be absolute",
        )
    return path


def _root(inputs: dict) -> Path | None:
    configured = _absolute_parameter(inputs, "root")
    if configured is not None:
        return configured
    discovered = os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME")
    return Path(discovered).expanduser() if discovered else None


def _driver_root(inputs: dict) -> Path | None:
    configured = _absolute_parameter(inputs, "driver_root")
    if configured is not None:
        return configured
    discovered = os.environ.get("MXDRIVER_HOME")
    return Path(discovered).expanduser() if discovered else None


def _driver_tool(inputs: dict) -> str | None:
    configured = _absolute_parameter(inputs, "driver_tool")
    if configured is not None:
        return str(configured) if configured.is_file() else None
    return shutil.which("mx-smi")


def _runtime_version(root: Path) -> str:
    for candidate in (root / "Version.txt", root / ".version.txt"):
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")[:65536]
        match = re.search(r"(?:^|\n)(?:Version|MACA)\s*:\s*([0-9]+(?:\.[0-9]+){1,3})", text)
        if match:
            return match.group(1)
    return "unknown"


def _driver_versions(executable: str, timeout: float) -> tuple[str, str]:
    try:
        process = subprocess.run(
            [executable, "--show-version", "-i", "0"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(
            "RESOURCE_UNAVAILABLE",
            "mx-smi version probe timed out",
            retryable=True,
        ) from exc
    if process.returncode:
        raise AdapterError(
            "RESOURCE_UNAVAILABLE",
            f"mx-smi version probe failed: {(process.stderr or '').strip()}",
            retryable=True,
        )
    text = process.stdout or ""
    maca = re.search(r"^\s*MACA\s*:\s*([0-9]+(?:\.[0-9]+){1,3})", text, re.MULTILINE)
    driver = re.search(r"^\s*KMD\s*:\s*([0-9]+(?:\.[0-9]+){1,3})", text, re.MULTILINE)
    return (
        maca.group(1) if maca else "unknown",
        driver.group(1) if driver else "unknown",
    )


def _validate_root(path: Path | None, label: str) -> None:
    if path is None or not path.is_dir():
        raise AdapterError("DEPENDENCY_MISSING", f"configured {label} root does not exist: {path}")


def probe(inputs: dict, context: dict) -> dict:
    root = _root(inputs)
    _validate_root(root, "MACA")
    assert root is not None
    if not any((root / name).exists() for name in ("lib", "include")):
        raise AdapterError("DEPENDENCY_MISSING", f"MACA root is incomplete: {root}")

    driver_root = _driver_root(inputs)
    if _parameters(inputs).get("driver_root") is not None:
        _validate_root(driver_root, "MetaX driver")

    executable = _driver_tool(inputs)
    if not executable:
        raise AdapterError("DEPENDENCY_MISSING", "mx-smi not found")
    detected_maca, driver_version = _driver_versions(
        executable,
        float(context.get("timeout_seconds", 2)),
    )
    runtime_version = _runtime_version(root)
    if runtime_version == "unknown":
        runtime_version = detected_maca

    return {
        "schema_version": "1.0",
        "family": "maca",
        "version": runtime_version,
        "driver_version": driver_version,
        "available": True,
        "capabilities": {
            "schema_version": "1.0",
            "values": {"runtime.compatible_device_vendors": ["metax"]},
        },
        "env_patch": {},
    }


def resolve_environment(inputs: dict, context: dict) -> dict:
    del context
    root = _root(inputs)
    _validate_root(root, "MACA")
    assert root is not None
    resolved_root = root.resolve()

    driver_root = _driver_root(inputs)
    if _parameters(inputs).get("driver_root") is not None:
        _validate_root(driver_root, "MetaX driver")

    path_entries = [
        resolved_root / "bin",
        resolved_root / "mxgpu_llvm" / "bin",
        resolved_root / "ompi" / "bin",
        resolved_root / "ucx" / "bin",
    ]
    library_entries = [
        resolved_root / "lib",
        resolved_root / "mxshmem" / "lib",
        resolved_root / "ompi" / "lib",
        resolved_root / "ucx" / "lib",
    ]
    if driver_root is not None:
        resolved_driver_root = driver_root.resolve()
        path_entries.append(resolved_driver_root / "bin")
        library_entries.append(resolved_driver_root / "lib")

    prepend_path: dict[str, list[str]] = {}
    existing_paths = [str(path) for path in path_entries if path.is_dir()]
    existing_libraries = [str(path) for path in library_entries if path.is_dir()]
    if existing_paths:
        prepend_path["PATH"] = existing_paths
    if existing_libraries:
        prepend_path["LD_LIBRARY_PATH"] = existing_libraries

    patch: dict[str, object] = {
        "set": {
            "MACA_PATH": str(resolved_root),
            "MACA_HOME": str(resolved_root),
            "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        }
    }
    if prepend_path:
        patch["prepend_path"] = prepend_path
    return {"env_patch": patch}


def snapshot(inputs: dict, context: dict) -> dict:
    return probe(inputs, context)


OPERATIONS = {
    "probe": probe,
    "resolve_environment": resolve_environment,
    "snapshot": snapshot,
}
