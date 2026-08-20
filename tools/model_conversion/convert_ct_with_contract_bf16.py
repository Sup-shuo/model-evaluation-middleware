#!/usr/bin/env python3
"""Convert compressed-tensors W4A16 weights to dense BF16 by contract.

This variant deliberately does not import a model implementation from
Transformers.  A small JSON contract exported from an independently loaded and
verified derivative defines the exact output config, parameter keys and shapes.
The source is read-only and the output is published under a new directory only
after a complete second source-equivalence pass.
"""

from __future__ import annotations

import argparse
import builtins
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _conversion_find_spec(name: str, *args: Any, **kwargs: Any):
    if name.split(".", 1)[0] == "torch_mlu":
        return None
    return _ORIGINAL_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _conversion_find_spec
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
_ORIGINAL_IMPORT = builtins.__import__


def _conversion_import(name: str, *args: Any, **kwargs: Any):
    if name.split(".", 1)[0] == "torch_mlu":
        raise ImportError("torch_mlu is disabled for this CPU-only conversion")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


builtins.__import__ = _conversion_import

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from compressed_tensors.compressors.pack_quantized.base import (
        PackedQuantizationCompressor,
    )

    _CT_API = "new"
except ImportError:
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        PackedQuantizationCompressor,
    )

    _CT_API = "legacy"
from compressed_tensors.quantization import QuantizationScheme


AUX_SUFFIXES = (
    ".weight_packed",
    ".weight_scale",
    ".weight_shape",
    ".weight_zero_point",
    ".weight_g_idx",
)


def log(stage: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {stage}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_tensor_index(root: Path) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise RuntimeError(f"duplicate tensor key: {key}")
                locations[key] = shard
    if not locations:
        raise RuntimeError(f"no safetensors under {root}")
    return locations


def read_tensor(locations: dict[str, Path], key: str) -> torch.Tensor:
    with safe_open(locations[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def decompress_weight(module_state: dict[str, torch.Tensor], scheme: QuantizationScheme) -> torch.Tensor:
    if _CT_API == "new":
        return PackedQuantizationCompressor.decompress(module_state, scheme)["weight"]
    return PackedQuantizationCompressor().decompress_weight(module_state, scheme.weights)


def source_value(
    locations: dict[str, Path],
    packed_prefixes: set[str],
    scheme: QuantizationScheme,
    key: str,
) -> tuple[torch.Tensor, str]:
    peer = {
        "model.language_model.embed_tokens.weight": "lm_head.weight",
        "lm_head.weight": "model.language_model.embed_tokens.weight",
    }
    source_key = key if key in locations else peer.get(key, key)
    if source_key in locations and not source_key.endswith(AUX_SUFFIXES):
        return read_tensor(locations, source_key).to(torch.bfloat16), "direct"
    prefix = key.removesuffix("weight")
    if prefix not in packed_prefixes:
        raise RuntimeError(f"no source mapping for {key}")
    names = [
        prefix + suffix
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_shape",
            "weight_zero_point",
            "weight_g_idx",
        )
        if prefix + suffix in locations
    ]
    state = {name.removeprefix(prefix): read_tensor(locations, name) for name in names}
    return decompress_weight(state, scheme).to(torch.bfloat16), "decompressed"


def copy_assets(source: Path, destination: Path) -> None:
    skipped = {"config.json", "model.safetensors.index.json", "recipe.yaml"}
    for entry in source.iterdir():
        if not entry.is_file() or entry.name in skipped:
            continue
        if entry.name.endswith(".safetensors") or ".safetensors." in entry.name:
            continue
        if entry.name.endswith((".part", ".tmp", ".lock", ".incomplete")):
            continue
        shutil.copy2(entry, destination / entry.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=4_000_000_000)
    args = parser.parse_args()

    source = args.source.resolve()
    contract_path = args.contract.resolve()
    temporary = args.temporary.absolute()
    final = args.final.absolute()
    if not source.is_dir() or not contract_path.is_file():
        raise RuntimeError("source directory or contract file is missing")
    if temporary.exists() or final.exists():
        raise RuntimeError(f"temporary/final target already exists: {temporary} / {final}")
    if temporary.parent != final.parent:
        raise RuntimeError("temporary and final must share a parent")

    source_config = load_json(source / "config.json")
    if source_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise RuntimeError(f"unsupported architecture: {source_config.get('architectures')}")
    quant = source_config.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors":
        raise RuntimeError("source is not compressed-tensors")
    if quant.get("format") != "pack-quantized":
        raise RuntimeError("source is not pack-quantized")
    groups = quant.get("config_groups") or {}
    if len(groups) != 1:
        raise RuntimeError(f"expected one config group, found {len(groups)}")
    scheme_raw = next(iter(groups.values()))
    scheme = (
        QuantizationScheme.model_validate(scheme_raw)
        if hasattr(QuantizationScheme, "model_validate")
        else QuantizationScheme.parse_obj(scheme_raw)
    )
    weights = scheme.weights
    if weights is None or weights.num_bits != 4 or weights.group_size != 32:
        raise RuntimeError(f"unsupported weight scheme: {weights}")

    contract = load_json(contract_path)
    if contract.get("schema_version") != "1.0":
        raise RuntimeError(f"unsupported contract schema_version: {contract.get('schema_version')!r}")
    target_config = contract.get("config")
    tensor_contract = contract.get("tensors")
    if not isinstance(target_config, dict) or not isinstance(tensor_contract, dict):
        raise RuntimeError("contract config/tensors are missing")
    if target_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise RuntimeError("contract architecture mismatch")
    if target_config.get("quantization_config") is not None:
        raise RuntimeError("contract target still has quantization metadata")
    target_shapes: dict[str, tuple[int, ...]] = {}
    for key, descriptor in tensor_contract.items():
        if not isinstance(descriptor, dict) or descriptor.get("dtype") != "BF16":
            raise RuntimeError(f"bad tensor contract for {key}")
        shape = descriptor.get("shape")
        if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise RuntimeError(f"bad tensor shape for {key}")
        target_shapes[str(key)] = tuple(shape)

    locations = build_tensor_index(source)
    packed_prefixes = {
        key.removesuffix("weight_packed")
        for key in locations
        if key.endswith(".weight_packed")
    }
    if not packed_prefixes:
        raise RuntimeError("source has no packed weights")
    for prefix in packed_prefixes:
        required = [prefix + "weight_scale", prefix + "weight_shape"]
        if not weights.symmetric:
            required.append(prefix + "weight_zero_point")
        missing = [key for key in required if key not in locations]
        if missing:
            raise RuntimeError(f"packed module is incomplete: {missing}")

    expected_parameters = sum(math.prod(shape) for shape in target_shapes.values())
    log(
        "contract_ok",
        source_keys=len(locations),
        packed_modules=len(packed_prefixes),
        target_keys=len(target_shapes),
        parameters=expected_parameters,
        ct_api=_CT_API,
    )

    os.mkdir(temporary)
    published = False
    try:
        started = time.time()
        copy_assets(source, temporary)
        output_map: dict[str, str] = {}
        shard_state: dict[str, torch.Tensor] = {}
        shard_bytes = 0
        shard_number = 0
        direct_count = 0
        decompressed_count = 0

        def flush_shard() -> None:
            nonlocal shard_state, shard_bytes, shard_number
            if not shard_state:
                return
            shard_number += 1
            name = f"model-{shard_number:05d}.safetensors"
            save_file(shard_state, temporary / name, metadata={"format": "pt"})
            output_map.update({key: name for key in shard_state})
            shard_state = {}
            shard_bytes = 0
            gc.collect()

        for position, key in enumerate(sorted(target_shapes), 1):
            value, kind = source_value(locations, packed_prefixes, scheme, key)
            if tuple(value.shape) != target_shapes[key]:
                raise RuntimeError(
                    f"target shape mismatch for {key}: {tuple(value.shape)} != {target_shapes[key]}"
                )
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"bad derived tensor: {key}")
            value_bytes = value.numel() * value.element_size()
            if shard_state and shard_bytes + value_bytes > args.max_shard_bytes:
                flush_shard()
            shard_state[key] = value.contiguous()
            shard_bytes += value_bytes
            direct_count += kind == "direct"
            decompressed_count += kind == "decompressed"
            if position % 100 == 0:
                log("build_progress", built=position, total=len(target_shapes))
        flush_shard()

        write_json(temporary / "config.json", target_config)
        write_json(
            temporary / "model.safetensors.index.json",
            {
                "metadata": {
                    "total_size": sum(
                        math.prod(shape) * 2 for shape in target_shapes.values()
                    )
                },
                "weight_map": output_map,
            },
        )
        output_locations = build_tensor_index(temporary)
        if set(output_locations) != set(target_shapes) or set(output_map) != set(target_shapes):
            raise RuntimeError("serialized target key set mismatch")
        wrong = {
            key for key, shard in output_locations.items() if output_map.get(key) != shard.name
        }
        if wrong:
            raise RuntimeError(f"serialized index mismatch: {sorted(wrong)[:20]}")

        checked_direct = 0
        checked_decompressed = 0
        for position, key in enumerate(sorted(target_shapes), 1):
            derived = read_tensor(output_locations, key)
            reference, kind = source_value(locations, packed_prefixes, scheme, key)
            if not torch.equal(reference, derived):
                delta = float((reference.float() - derived.float()).abs().max())
                raise RuntimeError(f"source equivalence mismatch for {key}: max_abs={delta}")
            checked_direct += kind == "direct"
            checked_decompressed += kind == "decompressed"
            del reference, derived
            if position % 100 == 0:
                log("equality_progress", checked=position, total=len(target_shapes))
        if (direct_count, decompressed_count) != (checked_direct, checked_decompressed):
            raise RuntimeError("build/equality category counts disagree")
        log(
            "full_source_equivalence_ok",
            direct=checked_direct,
            decompressed=checked_decompressed,
            shards=shard_number,
        )

        write_json(
            temporary / "DERIVATION.json",
            {
                "schema_version": "1.0",
                "source_ref": args.source_ref,
                "source_path_at_conversion": str(source),
                "source_format": (
                    "compressed-tensors pack-quantized W4A16 group32 "
                    + ("symmetric" if weights.symmetric else "asymmetric")
                ),
                "derived_format": "dense BF16 safetensors",
                "method": (
                    "independently verified Qwen3.5 target key/shape contract + global source "
                    "key index + installed compressed-tensors official weight decompressor + "
                    "complete second source-equivalence pass"
                ),
                "parameter_count": expected_parameters,
                "state_key_count": len(target_shapes),
                "contract_file": contract_path.name,
                "compressed_tensors": __import__("compressed_tensors").__version__,
            },
        )
        for entry in temporary.iterdir():
            if entry.is_file():
                with entry.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.rename(temporary, final)
        published = True
        log("publish_ok", final=final, elapsed_seconds=round(time.time() - started, 1))
        return 0
    except BaseException:
        if not published and temporary.exists():
            log("conversion_failed_retaining_exact_temp", temporary=temporary)
        raise


if __name__ == "__main__":
    sys.exit(main())
