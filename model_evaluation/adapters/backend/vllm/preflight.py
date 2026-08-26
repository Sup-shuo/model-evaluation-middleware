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


def _normalized_quantization(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _group_sizes(value: object) -> list[int]:
    found: set[int] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key == "group_size" and isinstance(item, (int, str)):
                    try:
                        size = int(item)
                    except ValueError:
                        continue
                    if size > 0:
                        found.add(size)
                else:
                    visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return sorted(found)


def _local_quantization_metadata(
    model_path: str,
) -> tuple[str | None, list[int], Path]:
    config_path = Path(model_path).resolve() / "config.json"
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, [], config_path
    if not isinstance(document, dict):
        return None, [], config_path
    quantization = document.get("quantization_config")
    if not isinstance(quantization, dict):
        return None, [], config_path
    method = _normalized_quantization(
        quantization.get("quant_method") or quantization.get("format")
    )
    return method or None, _group_sizes(quantization), config_path


def _enforce_quantization_constraints(
    payload: dict,
    *,
    quantization: object,
    group_sizes: list[int],
    config_path: Path,
) -> None:
    constraints = payload.get("quantization_constraints")
    if not isinstance(constraints, dict):
        return
    method = _normalized_quantization(quantization)
    rule = next(
        (
            value
            for key, value in constraints.items()
            if _normalized_quantization(key) == method and isinstance(value, dict)
        ),
        None,
    )
    if not rule:
        return
    supported = sorted(
        {
            int(value)
            for value in rule.get("supported_group_sizes", [])
            if isinstance(value, int) and value > 0
        }
    )
    unsupported = sorted(set(group_sizes) - set(supported))
    if not supported or not unsupported:
        return

    reason = str(
        rule.get("reason")
        or "the selected device/backend combination does not support this quantization layout"
    )
    recommendation = str(rule.get("recommendation") or "").strip()
    message = (
        f"{reason}; model declares group_size={unsupported}, "
        f"supported_group_sizes={supported}"
    )
    if recommendation:
        message += f"; recommendation: {recommendation}"
    raise PreflightFailure(
        "MODEL_QUANTIZATION_INCOMPATIBLE",
        message,
        details={
            "device_vendor": payload.get("device_vendor"),
            "quantization": method,
            "model_group_sizes": group_sizes,
            "supported_group_sizes": supported,
            "model_config": str(config_path),
            "recommendation": recommendation or None,
        },
    )


def _engine_config(payload: dict):
    model_path = str(payload["model_path"])
    declared_quantization = str(payload.get("quantization") or "").strip().lower()
    config_quantization, config_group_sizes, config_path = (
        _local_quantization_metadata(model_path)
    )
    _enforce_quantization_constraints(
        payload,
        quantization=config_quantization or declared_quantization,
        group_sizes=config_group_sizes,
        config_path=config_path,
    )

    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
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
    _enforce_quantization_constraints(
        payload,
        quantization=(
            resolved_quantization or config_quantization or declared_quantization
        ),
        group_sizes=config_group_sizes,
        config_path=config_path,
    )
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
        "quantization_group_sizes": config_group_sizes,
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
