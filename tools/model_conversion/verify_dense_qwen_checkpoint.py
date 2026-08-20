#!/usr/bin/env python3
"""Independently verify a dense Qwen checkpoint derived from CT weights."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

# Verification is CPU-only.  Keep optional accelerator modules out of
# dependency feature probes without changing the installed environment.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _verification_find_spec(name, *args, **kwargs):
    if name.split(".", 1)[0] == "torch_mlu":
        return None
    return _ORIGINAL_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _verification_find_spec

import torch
from accelerate import init_empty_weights
from compressed_tensors.compressors.pack_quantized.base import (
    PackedQuantizationCompressor,
)
from compressed_tensors.quantization import QuantizationScheme
from safetensors import safe_open
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)


AUX_SUFFIXES = (
    ".weight_packed",
    ".weight_scale",
    ".weight_shape",
    ".weight_zero_point",
    ".weight_g_idx",
)

ARCHITECTURE_CLASSES = {
    "Qwen3_5ForConditionalGeneration": (
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
    ),
    "Qwen3VLForConditionalGeneration": (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
    ),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_index(root: Path):
    locations = {}
    information = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise RuntimeError(f"duplicate key: {key}")
                view = handle.get_slice(key)
                locations[key] = shard
                information[key] = (tuple(view.get_shape()), str(view.get_dtype()))
    if not locations:
        raise RuntimeError(f"no safetensors under {root}")
    return locations, information


def read_tensor(locations, key):
    with safe_open(locations[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def find_keys(value, names, path=""):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in names:
                found.append(child_path)
            found.extend(find_keys(child, names, child_path))
    elif isinstance(value, list):
        for position, child in enumerate(value):
            found.extend(find_keys(child, names, f"{path}[{position}]"))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--expected-parameters", type=int, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    derived = args.derived.resolve()
    source_config_json = load_json(source / "config.json")
    derived_config_json = load_json(derived / "config.json")
    source_architectures = source_config_json.get("architectures") or []
    derived_architectures = derived_config_json.get("architectures") or []
    if source_architectures != derived_architectures or len(source_architectures) != 1:
        raise RuntimeError(
            f"architecture mismatch: source={source_architectures}, "
            f"derived={derived_architectures}"
        )
    architecture = source_architectures[0]
    if architecture not in ARCHITECTURE_CLASSES:
        raise RuntimeError(f"unsupported architecture: {architecture}")
    config_class, model_class = ARCHITECTURE_CLASSES[architecture]
    config_json = load_json(derived / "config.json")
    forbidden = find_keys(
        config_json,
        {"quantization_config", "compression_config", "_name_or_path"},
    )
    if forbidden:
        raise RuntimeError(f"forbidden config metadata: {forbidden}")
    dtype_fields = {
        "dtype": config_json.get("dtype"),
        "torch_dtype": config_json.get("torch_dtype"),
        "text_dtype": (config_json.get("text_config") or {}).get("dtype"),
        "vision_dtype": (config_json.get("vision_config") or {}).get("dtype"),
    }
    if any(value != "bfloat16" for value in dtype_fields.values()):
        raise RuntimeError(f"bad dtype fields: {dtype_fields}")

    output_locations, output_info = tensor_index(derived)
    index_json = load_json(derived / "model.safetensors.index.json")
    weight_map = index_json.get("weight_map") or {}
    if set(weight_map) != set(output_locations):
        raise RuntimeError("index key set does not match shard key set")
    wrong_shards = [
        key for key, shard in output_locations.items() if weight_map[key] != shard.name
    ]
    if wrong_shards:
        raise RuntimeError(f"index shard mapping mismatch: {wrong_shards[:20]}")

    clean_config = config_class.from_pretrained(derived, local_files_only=True)
    if getattr(clean_config, "quantization_config", None) is not None:
        raise RuntimeError("Transformers config retained quantization metadata")
    with init_empty_weights():
        meta_model = model_class(clean_config)
    target_shapes = {
        key: tuple(value.shape) for key, value in meta_model.state_dict().items()
    }
    del meta_model
    if set(target_shapes) != set(output_locations):
        raise RuntimeError(
            f"target/output key mismatch: missing={len(set(target_shapes)-set(output_locations))} "
            f"unexpected={len(set(output_locations)-set(target_shapes))}"
        )
    metadata_errors = [
        key
        for key, (shape, dtype) in output_info.items()
        if shape != target_shapes[key] or dtype != "BF16"
    ]
    if metadata_errors:
        raise RuntimeError(f"output shape/dtype errors: {metadata_errors[:20]}")

    quantization = source_config_json.get("quantization_config") or {}
    groups = quantization.get("config_groups") or {}
    if quantization.get("quant_method") != "compressed-tensors" or len(groups) != 1:
        raise RuntimeError("unsupported source quantization config")
    scheme = QuantizationScheme.model_validate(next(iter(groups.values())))
    source_locations, _ = tensor_index(source)
    packed_prefixes = {
        key.removesuffix("weight_packed")
        for key in source_locations
        if key.endswith(".weight_packed")
    }
    tied_peer = {
        "model.language_model.embed_tokens.weight": "lm_head.weight",
        "lm_head.weight": "model.language_model.embed_tokens.weight",
    }
    direct = 0
    decompressed = 0
    for position, key in enumerate(sorted(target_shapes), 1):
        output = read_tensor(output_locations, key)
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"non-finite output: {key}")
        source_key = key
        if source_key not in source_locations and key in tied_peer:
            source_key = tied_peer[key]
        if source_key in source_locations and not source_key.endswith(AUX_SUFFIXES):
            reference = read_tensor(source_locations, source_key).to(torch.bfloat16)
            direct += 1
        else:
            prefix = key.removesuffix("weight")
            if prefix not in packed_prefixes:
                raise RuntimeError(f"no source mapping: {key}")
            module_state = {}
            for suffix in (
                "weight_packed",
                "weight_scale",
                "weight_shape",
                "weight_zero_point",
                "weight_g_idx",
            ):
                source_name = prefix + suffix
                if source_name in source_locations:
                    module_state[suffix] = read_tensor(source_locations, source_name)
            reference = PackedQuantizationCompressor.decompress(
                module_state, scheme
            )["weight"].to(torch.bfloat16)
            decompressed += 1
        if not torch.equal(output, reference):
            delta = float((output.float() - reference.float()).abs().max())
            raise RuntimeError(f"value mismatch {key}: max_abs={delta}")
        del output, reference
        if position % 100 == 0:
            print(f"independent_equivalence {position}/{len(target_shapes)}", flush=True)

    reloaded, loading_info = model_class.from_pretrained(
        derived,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    load_errors = {
        field: loading_info.get(field) or []
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(load_errors.values()):
        raise RuntimeError(f"clean reload errors: {load_errors}")
    parameter_count = sum(parameter.numel() for parameter in reloaded.parameters())
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in reloaded.parameters()})
    if parameter_count != args.expected_parameters or parameter_dtypes != ["torch.bfloat16"]:
        raise RuntimeError(
            f"reload mismatch: parameters={parameter_count}, dtypes={parameter_dtypes}"
        )
    del reloaded
    gc.collect()
    report = {
        "architecture": architecture,
        "config_quantization_metadata": False,
        "dtype_fields": dtype_fields,
        "stored_keys": len(output_locations),
        "shards": len(set(output_locations.values())),
        "direct_equal": direct,
        "decompressed_equal": decompressed,
        "all_finite": True,
        "reload_errors": {field: len(value) for field, value in load_errors.items()},
        "reload_parameter_count": parameter_count,
        "reload_parameter_dtypes": parameter_dtypes,
    }
    print("INDEPENDENT_VERIFICATION_OK")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
