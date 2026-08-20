from __future__ import annotations

import re
import shutil
import subprocess

from model_evaluation.sdk.runtime import AdapterError


_DEVICE_LINE = re.compile(
    r"^GPU#(?P<id>\d+)\s+(?P<model>\S+)\s+(?P<pci>\S+)\s+"
    r"(?P<status>.*?)\s+\(UUID:\s*(?P<uuid>[^)]+)\)\s*$"
)
_MEMORY_TOTAL = re.compile(r"^\s*vis_vram total\s*:\s*(\d+)\s+KB\s*$", re.MULTILINE)


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AdapterError(
            "RESOURCE_UNAVAILABLE",
            f"MetaX device probe timed out: {argv[0]}",
            retryable=True,
        ) from exc


def _listed_devices(timeout: float = 2.0) -> list[dict[str, object]]:
    executable = shutil.which("mx-smi")
    if not executable:
        raise AdapterError("DEPENDENCY_MISSING", "mx-smi not found")
    process = _run([executable, "-L"], timeout)
    if process.returncode:
        raise AdapterError(
            "RESOURCE_UNAVAILABLE",
            f"mx-smi failed: {(process.stderr or '').strip()}",
            retryable=True,
        )

    devices: list[dict[str, object]] = []
    for line in (process.stdout or "").splitlines():
        match = _DEVICE_LINE.match(line.strip())
        if not match:
            continue
        status = match.group("status").strip()
        if status.lower() != "available":
            continue
        model = match.group("model").strip()
        devices.append(
            {
                "id": match.group("id"),
                "name": f"MetaX {model}",
                "uuid": match.group("uuid").strip(),
                "pci_bus_id": match.group("pci").strip(),
            }
        )
    if not devices:
        raise AdapterError(
            "RESOURCE_UNAVAILABLE",
            "mx-smi reported no available MetaX devices",
            retryable=True,
        )
    return devices


def _memory_bytes(device_id: str, timeout: float) -> int | None:
    executable = shutil.which("mx-smi")
    if not executable:
        return None
    process = _run([executable, "--show-memory", "-i", device_id], timeout)
    if process.returncode:
        return None
    match = _MEMORY_TOTAL.search(process.stdout or "")
    return int(match.group(1)) * 1024 if match else None


def probe(inputs: dict, context: dict) -> dict:
    timeout = float(context.get("timeout_seconds", 2))
    available = _listed_devices(timeout)
    requested = [str(item) for item in inputs.get("requested_devices", [])]
    if not requested:
        requested = [str(available[0]["id"])]

    selected = [item for item in available if item["id"] in requested]
    if {str(item["id"]) for item in selected} != set(requested):
        raise AdapterError(
            "RESOURCE_UNAVAILABLE",
            f"requested MetaX devices unavailable: {requested}",
            retryable=True,
        )
    for item in selected:
        memory_bytes = _memory_bytes(str(item["id"]), timeout)
        if memory_bytes is not None:
            item["memory_bytes"] = memory_bytes

    return {
        "schema_version": "1.0",
        "vendor": "metax",
        "device_type": "accelerator",
        "devices": selected,
        "capabilities": {
            "schema_version": "1.0",
            "values": {"device.multi_device": len(selected) > 1},
        },
    }


def visibility(inputs: dict, context: dict) -> dict:
    del context
    devices = [str(item) for item in inputs.get("devices", [])]
    patch = {"set": {"MACA_VISIBLE_DEVICES": ",".join(devices)}} if devices else {}
    return {"env_patch": patch}


def snapshot(inputs: dict, context: dict) -> dict:
    descriptor = probe(inputs, context)
    return {"vendor": "metax", "devices": descriptor["devices"]}


OPERATIONS = {
    "probe": probe,
    "visibility": visibility,
    "snapshot": snapshot,
}
