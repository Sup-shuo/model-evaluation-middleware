#!/usr/bin/env python3
"""Transformers compatibility checks run inside the selected Backend Environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path


def _versions() -> dict[str, str]:
    values = {}
    for distribution in ("torch", "transformers", "accelerate"):
        try:
            values[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return values


def _dependency() -> None:
    import torch
    import transformers

    print(
        json.dumps(
            {
                "python": sys.executable,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _validate_declared_architecture(
    model_class: str,
    architectures: list[str],
) -> str:
    resolved_class = model_class.rsplit(".", 1)[-1]
    if architectures and resolved_class not in architectures:
        raise RuntimeError(
            f"checkpoint declares architectures={architectures}, but "
            f"AutoModelForCausalLM resolves {resolved_class}; "
            "use a model-family Backend Adapter"
        )
    return model_class


def _model(payload: dict) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(str(payload["model_path"])).resolve()
    tokenizer_path = Path(str(payload["tokenizer_path"])).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer path not found: {tokenizer_path}")
    trust_remote_code = bool(payload.get("trust_remote_code", False))
    offline = bool(payload.get("offline", False))
    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=offline,
    )
    architectures = list(getattr(config, "architectures", None) or [])
    mapping = AutoModelForCausalLM._model_mapping
    auto_map = getattr(config, "auto_map", None) or {}
    remote_class = (
        auto_map.get("AutoModelForCausalLM")
        if isinstance(auto_map, dict)
        else None
    )
    local_class = mapping[type(config)] if type(config) in mapping else None
    if local_class is None and not remote_class:
        raise RuntimeError(
            "AutoModelForCausalLM does not support config class "
            f"{type(config).__name__}; use a model-family Backend Adapter"
        )
    model_class = _validate_declared_architecture(
        str(remote_class) if remote_class else str(local_class.__name__),
        architectures,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=trust_remote_code,
        local_files_only=offline,
    )
    from_pretrained_parameters = inspect.signature(
        AutoModelForCausalLM.from_pretrained
    ).parameters
    return {
        "architectures": architectures,
        "backend": "transformers",
        "chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "config_class": type(config).__name__,
        "context_length": int(payload.get("max_model_len", 4096)),
        "device": str(payload.get("device") or "auto"),
        "device_map": payload.get("device_map"),
        "dtype": str(payload.get("dtype") or "auto"),
        "model_class": model_class,
        "model_path": str(model_path),
        "model_type": getattr(config, "model_type", None),
        "packages": _versions(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "supports_dtype_keyword": "dtype" in from_pretrained_parameters,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_path": str(tokenizer_path),
        "trust_remote_code": trust_remote_code,
        "weights_loaded": False,
    }


def _result(payload: dict) -> dict:
    try:
        facts = _model(payload)
    except Exception as exc:
        return {
            "schema_version": "1.0",
            "status": "failed",
            "error": {
                "code": "MODEL_CONFIG_INCOMPATIBLE",
                "message": str(exc) or type(exc).__name__,
            },
        }
    return {
        "schema_version": "1.0",
        "status": "passed",
        "facts": facts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dependency", "model"), required=True)
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.payload)
        if not isinstance(payload, dict):
            raise TypeError("preflight payload must be a JSON object")
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "failed",
                    "error": {
                        "code": "PREFLIGHT_INPUT_INVALID",
                        "message": str(exc) or type(exc).__name__,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
    if args.mode == "dependency":
        _dependency()
        return
    result = _result(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
