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


_UNRESOLVED = object()

_METAX_COMPRESSED_TENSORS_RULE = {
    "device_vendor": "metax",
    "runtime_family": "maca",
    "runtime_version": "3.8.0.23",
    "backend_version": "0.23.0",
    "plugin_distribution": "vllm-metax",
    "plugin_version": (
        "0.23.0+ge3d464.d20260730.maca3.8.0.103.torch2.10.0"
    ),
    "quantization": "compressed-tensors",
    "supported_group_sizes": [64, 128],
    "capability_source": "backend.vllm.preflight.validated-stack",
    "reason": (
        "the validated MetaX/MACA/vLLM-MetaX compressed-tensors weight kernel "
        "supports group_size 64 or 128"
    ),
    "recommendation": (
        "use a manually converted BF16 artifact or a compatible quantized checkpoint"
    ),
}


def _distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("vllm", "vllm-metax", "torch", "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _weight_group_sizes(value: object) -> list[int]:
    if not isinstance(value, dict):
        return []
    config_groups = value.get("config_groups")
    if not isinstance(config_groups, dict):
        return []

    found: set[int] = set()
    for group in config_groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if not isinstance(weights, dict):
            continue
        group_size = weights.get("group_size")
        if not isinstance(group_size, (int, str)):
            continue
        try:
            size = int(group_size)
        except ValueError:
            continue
        if size > 0:
            found.add(size)
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
    return method or None, _weight_group_sizes(quantization), config_path


def _catalog_quantization(value: object) -> str | None:
    method = _normalized_quantization(value)
    if not method or method in _DTYPES or method in {"auto", "none", "unquantized"}:
        return None
    return method


def _validate_quantization_methods(
    *,
    declared: object,
    configured: object,
    resolved: object = _UNRESOLVED,
    config_path: Path,
) -> str | None:
    declared_method = _catalog_quantization(declared)
    configured_method = _catalog_quantization(configured)
    if (
        declared_method
        and configured_method
        and declared_method != configured_method
    ):
        raise PreflightFailure(
            "MODEL_METADATA_INCONSISTENT",
            (
                f"model catalog declares quantization={declared_method!r}, but "
                f"config.json declares {configured_method!r}"
            ),
            details={
                "declared_quantization": declared_method,
                "configured_quantization": configured_method,
                "resolved_quantization": None,
                "model_config": str(config_path),
            },
        )

    expected = configured_method or declared_method
    if resolved is _UNRESOLVED:
        return expected

    resolved_method = _catalog_quantization(resolved)
    if expected and resolved_method != expected:
        rendered = resolved_method or "unquantized"
        raise PreflightFailure(
            "MODEL_METADATA_INCONSISTENT",
            (
                f"model metadata declares quantization={expected!r}, but vLLM "
                f"resolved {rendered!r}"
            ),
            details={
                "declared_quantization": declared_method,
                "configured_quantization": configured_method,
                "resolved_quantization": resolved_method,
                "model_config": str(config_path),
            },
        )
    return expected or resolved_method


def _validated_quantization_rule(
    payload: dict,
    packages: dict[str, str],
    quantization: object,
) -> dict | None:
    rule = _METAX_COMPRESSED_TENSORS_RULE
    if _normalized_quantization(quantization) != rule["quantization"]:
        return None
    actual = {
        "device_vendor": _normalized_quantization(payload.get("device_vendor")),
        "runtime_family": _normalized_quantization(payload.get("runtime_family")),
        "runtime_version": str(payload.get("runtime_version") or ""),
        "backend_version": packages.get("vllm", ""),
        "plugin_version": packages.get(rule["plugin_distribution"], ""),
    }
    expected = {
        "device_vendor": rule["device_vendor"],
        "runtime_family": rule["runtime_family"],
        "runtime_version": rule["runtime_version"],
        "backend_version": rule["backend_version"],
        "plugin_version": rule["plugin_version"],
    }
    if actual != expected:
        return None
    return dict(rule)


def _enforce_quantization_rule(
    rule: dict | None,
    *,
    quantization: object,
    group_sizes: list[int],
    config_path: Path,
) -> None:
    if not rule:
        return
    method = _normalized_quantization(quantization)
    if method != _normalized_quantization(rule.get("quantization")):
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

    recommendation = str(rule.get("recommendation") or "").strip()
    message = (
        f"{rule['reason']}; model declares weight group_size={unsupported}, "
        f"supported_group_sizes={supported}"
    )
    if recommendation:
        message += f"; recommendation: {recommendation}"
    raise PreflightFailure(
        "MODEL_QUANTIZATION_INCOMPATIBLE",
        message,
        details={
            "device_vendor": rule["device_vendor"],
            "runtime_family": rule["runtime_family"],
            "validated_runtime_version": rule["runtime_version"],
            "validated_backend_version": rule["backend_version"],
            "validated_plugin_version": rule["plugin_version"],
            "capability_source": rule["capability_source"],
            "quantization": method,
            "model_weight_group_sizes": group_sizes,
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
    expected_quantization = _validate_quantization_methods(
        declared=declared_quantization,
        configured=config_quantization,
        config_path=config_path,
    )
    versions = _distribution_versions()
    quantization_rule = _validated_quantization_rule(
        payload,
        versions,
        expected_quantization,
    )
    _enforce_quantization_rule(
        quantization_rule,
        quantization=expected_quantization,
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
    effective_quantization = _validate_quantization_methods(
        declared=declared_quantization,
        configured=config_quantization,
        resolved=resolved_quantization,
        config_path=config_path,
    )
    hf_config = getattr(model, "hf_config", None)
    quantization_rule = _validated_quantization_rule(
        payload,
        versions,
        effective_quantization,
    )
    _enforce_quantization_rule(
        quantization_rule,
        quantization=effective_quantization,
        group_sizes=config_group_sizes,
        config_path=config_path,
    )
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
