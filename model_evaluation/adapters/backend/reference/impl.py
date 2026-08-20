from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlsplit

from model_evaluation.sdk.runtime import AdapterError


def requirements(inputs, _context):
    deployment = inputs.get("deployment") or {}
    mode = (deployment.get("management") or {}).get("mode")
    if mode != "managed":
        raise AdapterError(
            "CONFIG_INVALID",
            "reference backend supports managed demo mode only",
        )
    return {
        "schema_version": "1.0",
        "requirements": [
            {
                "path": "runtime.available",
                "op": "equals",
                "value": True,
                "message": "reference backend requires an available CPU runtime",
            },
            {
                "path": "backend_environment.python",
                "op": "equals",
                "value": True,
                "message": "reference backend requires a Python environment",
            },
        ],
    }


def plan_start(inputs, _context):
    model = inputs["model"]
    deployment = inputs["deployment"]
    parameters = deployment.get("parameters") or {}
    endpoint = inputs.get("endpoint") or {}
    host = str(endpoint.get("host") or "127.0.0.1")
    port = int(endpoint.get("port") or parameters.get("port") or 39091)
    runner = Path(__file__).with_name("runner.py").resolve()
    log_path = str(inputs.get("log_path") or "reference-backend.log")
    process = {
        "schema_version": "1.0",
        "argv": ["python", str(runner), "--serve", "--host", host, "--port", str(port)],
        "cwd": str(runner.parent),
        "env_patch": {},
        "stdin": {"mode": "null"},
        "stdout": {"mode": "file", "path": log_path},
        "stderr": {"mode": "merge_stdout"},
        "metadata": {"purpose": "hardware-free-reference-demo"},
    }
    dependency_probe = {
        "schema_version": "1.0",
        "argv": ["python", str(runner), "--version"],
        "cwd": str(runner.parent),
        "env_patch": {},
        "stdin": {"mode": "null"},
        "stdout": {"mode": "capture"},
        "stderr": {"mode": "capture"},
        "timeout_seconds": 3,
        "metadata": {"purpose": "reference-backend-version"},
    }
    return {
        "schema_version": "1.1",
        "process": process,
        "dependency_probe": dependency_probe,
        "shutdown": {
            "strategy": "signal",
            "signal": "SIGTERM",
            "timeout_seconds": float(parameters.get("shutdown_timeout_seconds", 3)),
        },
        "readiness": {
            "timeout_seconds": float(parameters.get("ready_timeout_seconds", 5)),
        },
        "attach": {
            "base_url": f"http://{host}:{port}/reference/v1",
            "model_id": str(model["id"]),
            "ownership": "managed",
            "auth": {"mode": "none"},
        },
    }


def probe_service(inputs, context):
    attach = inputs.get("attach") or {}
    parsed = urlsplit(str(attach.get("base_url") or ""))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        raise AdapterError("CONFIG_INVALID", "reference backend URL has no port")
    timeout = max(0.05, min(float(context.get("timeout_seconds", 1)), 0.5))
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        raise AdapterError(
            "SERVICE_NOT_READY",
            f"reference backend is not listening on {host}:{port}: {exc}",
            retryable=True,
        ) from exc
    return {
        "schema_version": "1.0",
        "service_type": "reference",
        "ownership": "managed",
        "model": {"id": str(attach["model_id"])},
        "protocols": {"reference": {"url": str(attach["base_url"])}},
        "capabilities": {
            "schema_version": "1.0",
            "values": {"service.reference": True},
        },
        "auth": {"mode": "none"},
        "metadata": {"purpose": "hardware-free-reference-demo"},
    }


def snapshot(_inputs, _context):
    return {
        "backend": "reference",
        "version": "1.0.0",
        "purpose": "hardware-free-reference-demo",
    }


OPERATIONS = {
    "requirements": requirements,
    "plan_start": plan_start,
    "probe_service": probe_service,
    "snapshot": snapshot,
}
