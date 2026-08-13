#!/usr/bin/env python3
"""vLLM preflight helpers executed inside the selected backend environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path


_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "half",
    "fp32": "float32",
    "float32": "float32",
}


class PreflightFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _engine_config(payload: dict):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    model_path = str(payload["model_path"])
    declared_quantization = str(payload.get("quantization") or "").strip().lower()
    configured_dtype = str(payload.get("dtype") or "auto").strip().lower()
    if configured_dtype == "auto" and declared_quantization in _DTYPES:
        configured_dtype = _DTYPES[declared_quantization]

    candidates = {
        "model": model_path,
        "tokenizer": str(payload.get("tokenizer") or model_path),
        "trust_remote_code": bool(payload.get("trust_remote_code", False)),
        "max_model_len": int(payload["max_model_len"]),
        "tensor_parallel_size": int(payload.get("tensor_parallel_size", 1)),
        "dtype": configured_dtype,
        "gpu_memory_utilization": float(payload.get("gpu_memory_utilization", 0.8)),
        "max_num_seqs": int(payload.get("max_num_seqs", 16)),
        "generation_config": str(payload.get("generation_config") or "vllm"),
    }
    supported = inspect.signature(EngineArgs).parameters
    language_model_only = bool(payload.get("language_model_only", False))
    if language_model_only:
        if "language_model_only" not in supported:
            raise PreflightFailure(
                "BACKEND_API_INCOMPATIBLE",
                "installed vLLM does not support language_model_only",
                details={"unsupported_fields": ["language_model_only"]},
            )
        candidates["language_model_only"] = True
    required = {"model", "tokenizer", "max_model_len", "tensor_parallel_size", "dtype"}
    unsupported = sorted(key for key in candidates if key in required and key not in supported)
    if unsupported:
        raise PreflightFailure(
            "BACKEND_API_INCOMPATIBLE",
            f"installed vLLM EngineArgs lacks required preflight fields: {unsupported}",
            details={"unsupported_fields": unsupported},
        )
    kwargs = {key: value for key, value in candidates.items() if key in supported}
    config = EngineArgs(**kwargs).create_engine_config()
    model = config.model_config
    resolved_quantization = getattr(model, "quantization", None)
    if declared_quantization and declared_quantization not in _DTYPES and declared_quantization not in {
        "auto", "none", "unquantized"
    } and not resolved_quantization:
        raise PreflightFailure(
            "MODEL_METADATA_INCONSISTENT",
            f"model catalog declares quantization={declared_quantization!r}, but vLLM resolved an unquantized model; "
            "check the active model config.json and quantization metadata",
            details={
                "declared_quantization": declared_quantization,
                "resolved_quantization": resolved_quantization,
                "model_config": str(Path(model_path).resolve() / "config.json"),
            },
        )
    hf_config = getattr(model, "hf_config", None)
    versions: dict[str, str] = {}
    for distribution in ("vllm", "torch", "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "backend": "vllm",
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": versions,
        "platform": getattr(current_platform, "device_name", None) or current_platform.__class__.__name__,
        "model_path": str(Path(model_path).resolve()),
        "architectures": list(getattr(model, "architectures", None) or []),
        "model_type": getattr(hf_config, "model_type", None),
        "quantization": resolved_quantization,
        "declared_quantization": declared_quantization or None,
        "dtype": str(getattr(model, "dtype", None)),
        "max_model_len": int(getattr(model, "max_model_len")),
        "tensor_parallel_size": int(payload.get("tensor_parallel_size", 1)),
        "language_model_only": language_model_only,
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
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        result = {
            "schema_version": "1.0",
            "status": "failed",
            "error": {"code": "PREFLIGHT_INPUT_INVALID", "message": message},
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        raise SystemExit(2)
    try:
        result = {
            "schema_version": "1.0",
            "status": "passed",
            "facts": _engine_config(payload),
        }
        returncode = 0
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        code = exc.code if isinstance(exc, PreflightFailure) else "MODEL_CONFIG_INCOMPATIBLE"
        error = {"code": code, "message": message}
        if isinstance(exc, PreflightFailure) and exc.details:
            error["details"] = exc.details
        result = {
            "schema_version": "1.0",
            "status": "failed",
            "error": error,
        }
        returncode = 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
