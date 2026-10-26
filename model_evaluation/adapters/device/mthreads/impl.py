"""Moore Threads device facts and visibility; no backend policy."""
from __future__ import annotations

import csv
import re
import shutil
import subprocess

from model_evaluation.sdk.runtime import AdapterError


def _inventory(timeout: float) -> list[dict]:
    executable = shutil.which("mthreads-gmi")
    if not executable:
        raise AdapterError("DEPENDENCY_MISSING", "mthreads-gmi not found")
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,uuid,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError("RESOURCE_UNAVAILABLE", "mthreads-gmi timed out",
                           retryable=True) from exc
    if result.returncode:
        raise AdapterError("RESOURCE_UNAVAILABLE",
                           f"mthreads-gmi failed: {result.stderr.strip()}", retryable=True)
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        if len(row) != 4:
            raise AdapterError("RESOURCE_UNAVAILABLE", "unrecognized mthreads-gmi CSV")
        ident, name, uuid, memory = (field.strip() for field in row)
        match = re.fullmatch(r"(\d+)\s*(?:\[MiB\]|MiB)?", memory)
        if not ident.isdigit() or not name or not match:
            raise AdapterError("RESOURCE_UNAVAILABLE", "invalid mthreads-gmi device fields")
        devices.append({"id": ident, "name": name, "uuid": uuid,
                        "memory_bytes": int(match.group(1)) * 1024**2})
    if not devices:
        raise AdapterError("RESOURCE_UNAVAILABLE", "no Moore Threads devices reported")
    return devices


def probe(inputs: dict, context: dict) -> dict:
    available = _inventory(float(context.get("timeout_seconds", 5)))
    requested = [str(item) for item in inputs.get("requested_devices", [])]
    requested = requested or [available[0]["id"]]
    by_id = {item["id"]: item for item in available}
    if len(set(requested)) != len(requested) or any(item not in by_id for item in requested):
        raise AdapterError("RESOURCE_UNAVAILABLE",
                           f"invalid or unavailable Moore Threads devices: {requested}")
    return {
        "schema_version": "1.0", "vendor": "mthreads", "device_type": "accelerator",
        "devices": [by_id[item] for item in requested],
        "capabilities": {"schema_version": "1.0",
                         "values": {"device.multi_device": len(requested) > 1}},
    }


def visibility(inputs: dict, context: dict) -> dict:
    devices = [str(item) for item in inputs.get("devices", [])]
    if any(not item.isdigit() for item in devices) or len(set(devices)) != len(devices):
        raise AdapterError("CONFIG_INVALID", "MUSA visibility requires unique numeric device IDs")
    return {"env_patch": {"set": {"MUSA_VISIBLE_DEVICES": ",".join(devices)}}}


def snapshot(inputs: dict, context: dict) -> dict:
    descriptor = probe(inputs, context)
    return {"vendor": descriptor["vendor"], "devices": descriptor["devices"]}


OPERATIONS = {"probe": probe, "visibility": visibility, "snapshot": snapshot}
