#!/usr/bin/env python3
"""TinyLLaVA Factory checks run inside the selected Backend Environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


class FamilyPreflightError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _versions() -> dict[str, str]:
    versions = {}
    for distribution in (
        "torch",
        "transformers",
        "accelerate",
        "pillow",
        "requests",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _dependency() -> None:
    import PIL
    import requests
    import torch
    import transformers

    print(
        json.dumps(
            {
                "Pillow": PIL.__version__,
                "python": sys.executable,
                "requests": requests.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _read_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FamilyPreflightError(
            "MODEL_CONFIG_INVALID",
            f"cannot read TinyLLaVA config.json: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise FamilyPreflightError(
            "MODEL_CONFIG_INVALID",
            "TinyLLaVA config.json must contain an object",
        )
    return value


def _model(payload: dict) -> dict:
    model_path = Path(str(payload["model_path"])).resolve()
    if not model_path.is_dir():
        raise FamilyPreflightError(
            "DEPENDENCY_MISSING",
            f"model directory not found: {model_path}",
        )
    raw = _read_config(model_path)
    if str(raw.get("model_type") or "").lower() != "tinyllava":
        raise FamilyPreflightError(
            "MODEL_FAMILY_MISMATCH",
            "TinyLLaVA Backend requires model_type='tinyllava'",
        )
    auto_map = raw.get("auto_map")
    if not isinstance(auto_map, dict) or not auto_map.get(
        "AutoModelForCausalLM"
    ):
        raise FamilyPreflightError(
            "MODEL_FAMILY_RUNTIME_UNAVAILABLE",
            "checkpoint does not embed the TinyLLaVA Factory "
            "AutoModelForCausalLM implementation; use a modern Factory "
            "checkpoint with auto_map or a separately supported legacy runtime",
        )
    if not bool(payload.get("trust_remote_code", False)):
        raise FamilyPreflightError(
            "CONFIG_INVALID",
            "TinyLLaVA Factory requires trust_remote_code=true",
        )

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=bool(payload.get("offline", False)),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=bool(payload.get("offline", False)),
        use_fast=False,
    )
    return {
        "architectures": list(getattr(config, "architectures", None) or []),
        "backend": "tinyllava",
        "config_class": type(config).__name__,
        "context_length": int(payload.get("max_model_len", 4096)),
        "device": str(payload.get("device") or "auto"),
        "device_map": payload.get("device_map"),
        "dtype": str(payload.get("dtype") or "auto"),
        "family_api": "AutoModelForCausalLM.model.chat",
        "model_path": str(model_path),
        "model_type": getattr(config, "model_type", None),
        "packages": _versions(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "tokenizer_class": type(tokenizer).__name__,
        "trust_remote_code": True,
        "weights_loaded": False,
    }


def _result(payload: dict) -> dict:
    try:
        return {
            "schema_version": "1.0",
            "status": "passed",
            "facts": _model(payload),
        }
    except Exception as exc:
        code = (
            exc.code
            if isinstance(exc, FamilyPreflightError)
            else "MODEL_CONFIG_INCOMPATIBLE"
        )
        return {
            "schema_version": "1.0",
            "status": "failed",
            "error": {
                "code": code,
                "message": str(exc) or type(exc).__name__,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dependency", "model"), required=True)
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.payload)
        if not isinstance(payload, dict):
            raise TypeError("preflight payload must be an object")
    except Exception as exc:
        result = {
            "schema_version": "1.0",
            "status": "failed",
            "error": {
                "code": "PREFLIGHT_INPUT_INVALID",
                "message": str(exc) or type(exc).__name__,
            },
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        raise SystemExit(2)
    if args.mode == "dependency":
        _dependency()
        return
    result = _result(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
