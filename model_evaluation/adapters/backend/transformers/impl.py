from __future__ import annotations

import json
from pathlib import Path

from model_evaluation.sdk.native_backend import probe_native_service
from model_evaluation.sdk.runtime import AdapterError


def _managed(inputs: dict) -> tuple[dict, dict, dict, dict]:
    model = inputs.get("model") or {}
    deployment = inputs.get("deployment") or {}
    management = deployment.get("management") or {}
    if management.get("mode") != "managed":
        raise AdapterError(
            "CONFIG_INVALID",
            "Transformers Backend supports managed deployments only; "
            "use generic_openai for an existing service",
        )
    parameters = deployment.get("parameters") or {}
    model_location = deployment.get("model_location") or {}
    return model, deployment, parameters, model_location


def _model_path(model_location: dict) -> Path:
    raw = model_location.get("local_path")
    if not raw:
        raise AdapterError(
            "CONFIG_INVALID",
            "managed Transformers Backend requires model_location.local_path",
        )
    path = Path(str(raw))
    if not path.is_absolute():
        raise AdapterError(
            "CONFIG_INVALID",
            "managed Transformers model_location.local_path must be absolute",
        )
    if not path.is_dir():
        raise AdapterError(
            "DEPENDENCY_MISSING",
            f"managed model directory not found: {path}",
        )
    return path


def _environment(parameters: dict, context: dict, inputs: dict) -> dict:
    patch = {
        "set": {
            "PYTHONHASHSEED": str(parameters.get("pythonhashseed", 1234)),
            "TOKENIZERS_PARALLELISM": str(
                parameters.get("tokenizers_parallelism", False)
            ).lower(),
        }
    }
    cache_root = parameters.get("cache_root")
    if cache_root:
        hf_root = Path(str(cache_root)) / "huggingface"
        patch["set"].update(
            {
                "HF_HOME": str(hf_root),
                "HF_DATASETS_CACHE": str(hf_root / "datasets"),
                "HF_HUB_CACHE": str(hf_root / "hub"),
            }
        )
    offline = bool(
        context.get("offline") or inputs.get("network_policy") == "offline"
    )
    if offline:
        patch["set"].update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return patch


def _payload(inputs: dict, context: dict) -> dict:
    model, _deployment, parameters, model_location = _managed(inputs)
    path = _model_path(model_location)
    tokenizer = model.get("tokenizer") or {}
    tokenizer_ref = (
        tokenizer.get("ref") if isinstance(tokenizer, dict) else None
    )
    tokenizer_path = (
        model_location.get("tokenizer_path") or tokenizer_ref or str(path)
    )
    max_model_len = int(
        parameters.get(
            "max_model_len",
            model.get("context_length", 4096),
        )
    )
    return {
        "allow_cpu_offload": bool(parameters.get("allow_cpu_offload", False)),
        "device": str(parameters.get("device") or "auto"),
        "device_map": parameters.get("device_map"),
        "dtype": str(parameters.get("dtype") or "auto"),
        "max_model_len": max_model_len,
        "model_id": str(parameters.get("api_model_name") or model.get("id")),
        "model_path": str(path),
        "offline": bool(
            context.get("offline") or inputs.get("network_policy") == "offline"
        ),
        "tokenizer_path": str(tokenizer_path),
        "trust_remote_code": bool(
            parameters.get(
                "trust_remote_code",
                model.get("trust_remote_code", False),
            )
        ),
    }


def requirements(inputs: dict, context: dict) -> dict:
    _model, _deployment, _parameters, model_location = _managed(inputs)
    _model_path(model_location)
    return {
        "schema_version": "1.0",
        "requirements": [
            {
                "path": "runtime.available",
                "op": "equals",
                "value": True,
                "message": (
                    "managed Transformers Backend requires an available "
                    "local runtime"
                ),
            }
        ],
    }


def _probe_process(
    *,
    mode: str,
    payload: dict,
    env_patch: dict,
    timeout_seconds: float,
) -> dict:
    return {
        "schema_version": "1.0",
        "argv": [
            "python",
            "-B",
            str(Path(__file__).with_name("preflight.py").resolve()),
            "--mode",
            mode,
            "--payload",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        ],
        "env_patch": env_patch,
        "stdin": {"mode": "null"},
        "stdout": {"mode": "capture"},
        "stderr": {"mode": "capture"},
        "timeout_seconds": timeout_seconds,
        "metadata": {},
    }


def plan_preflight(inputs: dict, context: dict) -> dict:
    _model, _deployment, parameters, _location = _managed(inputs)
    payload = _payload(inputs, context)
    env_patch = _environment(parameters, context, inputs)
    dependency_timeout = float(
        parameters.get("dependency_probe_timeout_seconds", 45.0)
    )
    model_timeout = float(
        parameters.get("model_probe_timeout_seconds", 120.0)
    )
    return {
        "schema_version": "1.0",
        "probes": [
            {
                "id": "backend.import",
                "phase": "backend_dependency",
                "required": True,
                "description": "Import torch and Transformers",
                "result_format": "text",
                "process": _probe_process(
                    mode="dependency",
                    payload=payload,
                    env_patch=env_patch,
                    timeout_seconds=dependency_timeout,
                ),
            },
            {
                "id": "model.config",
                "phase": "model_compatibility",
                "required": True,
                "description": (
                    "Resolve AutoConfig and AutoModelForCausalLM without "
                    "loading weights"
                ),
                "result_format": "preflight_result",
                "process": _probe_process(
                    mode="model",
                    payload=payload,
                    env_patch=env_patch,
                    timeout_seconds=model_timeout,
                ),
            },
        ],
    }


def plan_start(inputs: dict, context: dict) -> dict:
    model, _deployment, parameters, _location = _managed(inputs)
    payload = _payload(inputs, context)
    endpoint = inputs.get("endpoint") or {}
    host = str(endpoint.get("host") or "127.0.0.1")
    port = int(endpoint.get("port") or parameters.get("port") or 8091)
    max_concurrent = int(parameters.get("max_concurrent", 1))
    max_request_bytes = int(parameters.get("max_request_bytes", 16777216))
    payload.update(
        {
            "host": host,
            "max_concurrent": max_concurrent,
            "max_request_bytes": max_request_bytes,
            "port": port,
        }
    )
    env_patch = _environment(parameters, context, inputs)
    runner = str(Path(__file__).with_name("runner.py").resolve())
    process = {
        "schema_version": "1.0",
        "argv": [
            "python",
            "-B",
            runner,
            "--payload",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        ],
        "env_patch": env_patch,
        "stdin": {"mode": "null"},
        "stdout": {
            "mode": "file",
            "path": str(inputs.get("log_path") or "transformers.log"),
        },
        "stderr": {"mode": "merge_stdout"},
        "metadata": {},
    }
    workspace = context.get("workspace")
    if workspace and Path(str(workspace)).is_absolute():
        process["cwd"] = str(workspace)
    dependency_probe = _probe_process(
        mode="dependency",
        payload=payload,
        env_patch=env_patch,
        timeout_seconds=float(
            parameters.get("dependency_probe_timeout_seconds", 45.0)
        ),
    )
    return {
        "schema_version": "1.1",
        "process": process,
        "dependency_probe": dependency_probe,
        "shutdown": {
            "strategy": "signal",
            "signal": "SIGTERM",
            "timeout_seconds": float(
                parameters.get("shutdown_timeout_seconds", 15.0)
            ),
        },
        "readiness": {
            "timeout_seconds": float(
                parameters.get("ready_timeout_seconds", 900)
            )
        },
        "attach": {
            "base_url": f"http://{host}:{port}/v1",
            "model_id": payload["model_id"],
            "ownership": "managed",
            "auth": {"mode": "none"},
            "context_length": payload["max_model_len"],
            "max_concurrent": max_concurrent,
        },
    }


def probe_service(inputs: dict, context: dict) -> dict:
    attach = inputs.get("attach") or {}
    model_id = str(attach.get("model_id") or "")
    if not model_id:
        raise AdapterError(
            "CONFIG_INVALID",
            "probe_service requires attach.model_id",
        )
    return probe_native_service(
        base_url=str(attach.get("base_url") or ""),
        model_id=model_id,
        ownership=str(attach.get("ownership") or "managed"),
        timeout=float(context.get("timeout_seconds", 20.0)),
        max_concurrent=int(attach.get("max_concurrent", 1)),
    )


def snapshot(inputs: dict, context: dict) -> dict:
    _ = context
    parameters = (inputs.get("deployment") or {}).get("parameters") or {}
    return {
        "backend": "transformers",
        "execution_engine": "python_native_service",
        "configured_device": str(parameters.get("device") or "auto"),
        "configured_dtype": str(parameters.get("dtype") or "auto"),
        "version_source": "selected_environment_probe",
    }


OPERATIONS = {
    "requirements": requirements,
    "plan_preflight": plan_preflight,
    "plan_start": plan_start,
    "probe_service": probe_service,
    "snapshot": snapshot,
}
