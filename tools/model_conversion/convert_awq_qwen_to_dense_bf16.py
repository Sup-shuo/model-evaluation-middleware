#!/usr/bin/env python3
"""Derive a dense BF16 Qwen checkpoint from an AWQ GEMM checkpoint.

The source is read-only. Output is built in a caller-selected temporary
directory, fully compared with the source AWQ tensors, reloaded through
Transformers, and only then renamed to the final directory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _conversion_find_spec(name: str, *args: Any, **kwargs: Any):
    # Keep this CPU conversion independent of an installed accelerator plugin.
    if name.split(".", 1)[0] in {"torch_mlu"}:
        return None
    return _ORIGINAL_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _conversion_find_spec

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration


ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
AWQ_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")
AWQ_ORDER = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32)


def log(stage: str, **fields: Any) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[{now}] {stage}" + (f" {suffix}" if suffix else ""), flush=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        raise RuntimeError(f"no safetensors found under {root}")
    return locations


def read_tensor(locations: dict[str, Path], key: str) -> torch.Tensor:
    with safe_open(locations[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def recursive_remove_quant_metadata(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        value.pop("compression_config", None)
        for child in value.values():
            recursive_remove_quant_metadata(child)
    elif isinstance(value, list):
        for child in value:
            recursive_remove_quant_metadata(child)


def copy_non_weight_assets(source: Path, temporary: Path) -> None:
    skipped = {"config.json", "model.safetensors.index.json"}
    for entry in source.iterdir():
        if not entry.is_file() or entry.name in skipped:
            continue
        if entry.name.endswith(".safetensors") or ".safetensors." in entry.name:
            continue
        if entry.name.endswith((".part", ".tmp", ".lock", ".incomplete")):
            continue
        shutil.copy2(entry, temporary / entry.name)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int32:
        raise RuntimeError(f"AWQ packed tensor is not int32: {packed.dtype}")
    shifts = (AWQ_ORDER * 4).view(1, 1, 8)
    return ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(
        packed.shape[0], packed.shape[1] * 8
    )


def dequantize_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    row_chunk: int = 256,
) -> torch.Tensor:
    """CPU equivalent of vLLM's AutoAWQ GEMM dequantization."""
    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise RuntimeError("AWQ tensors must be rank two")
    rows, packed_columns = qweight.shape
    columns = packed_columns * 8
    if rows % group_size:
        raise RuntimeError(f"AWQ input rows {rows} not divisible by group {group_size}")
    expected_groups = rows // group_size
    if tuple(qzeros.shape) != (expected_groups, packed_columns):
        raise RuntimeError(f"bad qzeros shape: {tuple(qzeros.shape)}")
    if tuple(scales.shape) != (expected_groups, columns):
        raise RuntimeError(f"bad scales shape: {tuple(scales.shape)}")
    if not bool(torch.isfinite(scales).all()):
        raise RuntimeError("AWQ scales contain non-finite values")

    output = torch.empty((rows, columns), dtype=torch.bfloat16)
    unpacked_zeros = unpack_int4(qzeros)
    for start in range(0, rows, row_chunk):
        end = min(rows, start + row_chunk)
        group_indices = torch.arange(start, end, dtype=torch.long) // group_size
        weights = unpack_int4(qweight[start:end])
        zeros = unpacked_zeros.index_select(0, group_indices)
        row_scales = scales.index_select(0, group_indices)
        dense = (weights - zeros) * row_scales
        output[start:end].copy_(dense.to(torch.bfloat16))
        del weights, zeros, row_scales, dense
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("dequantized AWQ weight contains non-finite values")
    return output


def validate_saved_index(root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    index = load_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("saved checkpoint has no weight_map")
    locations = build_tensor_index(root)
    missing = set(weight_map) - set(locations)
    extra = set(locations) - set(weight_map)
    wrong = {key for key, path in locations.items() if weight_map.get(key) != path.name}
    if missing or extra or wrong:
        raise RuntimeError(
            f"bad saved index: missing={len(missing)} extra={len(extra)} wrong={len(wrong)}"
        )
    return locations, weight_map


def source_value(
    locations: dict[str, Path], key: str, packed_prefixes: set[str], group_size: int
) -> tuple[torch.Tensor, str]:
    peer = {
        "model.language_model.embed_tokens.weight": "lm_head.weight",
        "lm_head.weight": "model.language_model.embed_tokens.weight",
    }
    source_key = key if key in locations else peer.get(key, key)
    if source_key in locations and not source_key.endswith(AWQ_SUFFIXES):
        return read_tensor(locations, source_key).to(torch.bfloat16), "direct"
    prefix = key.removesuffix("weight")
    if prefix not in packed_prefixes:
        raise RuntimeError(f"no source mapping for target tensor: {key}")
    return (
        dequantize_awq(
            read_tensor(locations, prefix + "qweight"),
            read_tensor(locations, prefix + "qzeros"),
            read_tensor(locations, prefix + "scales"),
            group_size=group_size,
        ).transpose(0, 1).contiguous(),
        "dequantized",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    temporary = args.temporary.absolute()
    final = args.final.absolute()
    if not source.is_dir():
        raise RuntimeError(f"source is not a directory: {source}")
    if temporary.parent != final.parent:
        raise RuntimeError("temporary and final must share a parent")
    if not args.contract_only and (temporary.exists() or final.exists()):
        raise RuntimeError(f"temporary/final already exists: {temporary} / {final}")

    started = time.time()
    source_config = load_json(source / "config.json")
    if source_config.get("architectures") != [ARCHITECTURE]:
        raise RuntimeError(f"unsupported architecture: {source_config.get('architectures')}")
    quant = source_config.get("quantization_config") or {}
    expected = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "version": "gemm",
        "zero_point": True,
    }
    for field, value in expected.items():
        if quant.get(field) != value:
            raise RuntimeError(f"unexpected AWQ {field}: {quant.get(field)}")
    group_size = int(quant["group_size"])

    locations = build_tensor_index(source)
    packed_prefixes = {
        key.removesuffix("qweight") for key in locations if key.endswith(".qweight")
    }
    if not packed_prefixes:
        raise RuntimeError("source has no AWQ qweight tensors")
    for prefix in packed_prefixes:
        absent = [prefix + suffix for suffix in ("qzeros", "scales") if prefix + suffix not in locations]
        if absent:
            raise RuntimeError(f"AWQ module is incomplete: {absent}")

    clean_config = Qwen3_5Config.from_pretrained(source, local_files_only=True)
    if hasattr(clean_config, "quantization_config"):
        delattr(clean_config, "quantization_config")
    clean_config.dtype = torch.bfloat16
    clean_config._name_or_path = ""
    if getattr(clean_config, "text_config", None) is not None:
        clean_config.text_config.dtype = torch.bfloat16
    if getattr(clean_config, "vision_config", None) is not None:
        clean_config.vision_config.dtype = torch.bfloat16
    with init_empty_weights():
        target_model = Qwen3_5ForConditionalGeneration(clean_config)
    target_state = target_model.state_dict()
    target_shapes = {key: tuple(value.shape) for key, value in target_state.items()}
    target_parameters = sum(value.numel() for value in target_model.parameters())
    del target_state, target_model
    gc.collect()

    mapped_direct = mapped_packed = 0
    for key in target_shapes:
        peer = {
            "model.language_model.embed_tokens.weight": "lm_head.weight",
            "lm_head.weight": "model.language_model.embed_tokens.weight",
        }
        source_key = key if key in locations else peer.get(key, key)
        if source_key in locations and not source_key.endswith(AWQ_SUFFIXES):
            mapped_direct += 1
        elif key.endswith("weight") and key.removesuffix("weight") in packed_prefixes:
            mapped_packed += 1
        else:
            raise RuntimeError(f"unmapped target key: {key}")
    log(
        "contract_ok",
        source_keys=len(locations),
        target_keys=len(target_shapes),
        direct=mapped_direct,
        packed=mapped_packed,
        parameters=target_parameters,
    )
    if args.contract_only:
        # Validate one early/middle/late module through the same CPU formula.
        samples = sorted(packed_prefixes)
        for prefix in (samples[0], samples[len(samples) // 2], samples[-1]):
            value = dequantize_awq(
                read_tensor(locations, prefix + "qweight"),
                read_tensor(locations, prefix + "qzeros"),
                read_tensor(locations, prefix + "scales"),
                group_size=group_size,
            )
            log("sample_dequant_ok", prefix=prefix, shape=tuple(value.shape))
            del value
        return 0

    os.mkdir(temporary)
    published = False
    try:
        filtered: dict[str, torch.Tensor] = {}
        direct_count = packed_count = 0
        for position, key in enumerate(sorted(target_shapes), 1):
            value, kind = source_value(locations, key, packed_prefixes, group_size)
            if tuple(value.shape) != target_shapes[key]:
                raise RuntimeError(f"shape mismatch for {key}: {tuple(value.shape)}")
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"invalid target tensor: {key}")
            filtered[key] = value
            direct_count += kind == "direct"
            packed_count += kind == "dequantized"
            if position % 100 == 0:
                log("source_build_progress", built=position, total=len(target_shapes))

        copy_non_weight_assets(source, temporary)
        clean_save_config = copy.deepcopy(clean_config)
        with init_empty_weights():
            clean_model = Qwen3_5ForConditionalGeneration(clean_save_config)
        result = clean_model.load_state_dict(filtered, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"assignment failed: {result}")
        log("save_started", max_shard_size=args.max_shard_size)
        clean_model.save_pretrained(
            temporary, safe_serialization=True, max_shard_size=args.max_shard_size
        )
        del filtered, clean_model
        gc.collect()

        config_path = temporary / "config.json"
        output_config = load_json(config_path)
        recursive_remove_quant_metadata(output_config)
        output_config.pop("_name_or_path", None)
        output_config["dtype"] = "bfloat16"
        output_config["torch_dtype"] = "bfloat16"
        for section in ("text_config", "vision_config"):
            if isinstance(output_config.get(section), dict):
                output_config[section]["dtype"] = "bfloat16"
                output_config[section].pop("torch_dtype", None)
        write_json(config_path, output_config)

        output_locations, _ = validate_saved_index(temporary)
        if set(output_locations) != set(target_shapes):
            raise RuntimeError(
                f"serialized key mismatch missing={len(set(target_shapes)-set(output_locations))} "
                f"extra={len(set(output_locations)-set(target_shapes))}"
            )
        log(
            "save_index_ok",
            stored_keys=len(output_locations),
            shards=len({path.name for path in output_locations.values()}),
        )

        checked_direct = checked_packed = 0
        for position, key in enumerate(sorted(target_shapes), 1):
            derived = read_tensor(output_locations, key)
            reference, kind = source_value(locations, key, packed_prefixes, group_size)
            if not torch.equal(reference, derived):
                delta = float((reference.float() - derived.float()).abs().max())
                raise RuntimeError(f"source equivalence mismatch {key}: {delta}")
            checked_direct += kind == "direct"
            checked_packed += kind == "dequantized"
            del reference, derived
            if position % 100 == 0:
                log("equivalence_progress", checked=position, total=len(target_shapes))
        log("full_source_equivalence_ok", direct=checked_direct, dequantized=checked_packed)

        reloaded, reload_info = Qwen3_5ForConditionalGeneration.from_pretrained(
            temporary,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if reload_info.get(field):
                raise RuntimeError(f"clean reload {field}: {reload_info[field][:20]}")
        if getattr(reloaded.config, "quantization_config", None) is not None:
            raise RuntimeError("clean reload retained quantization metadata")
        if any(parameter.dtype != torch.bfloat16 for parameter in reloaded.parameters()):
            raise RuntimeError("clean reload has non-BF16 parameters")
        reloaded_parameters = sum(parameter.numel() for parameter in reloaded.parameters())
        if reloaded_parameters != target_parameters:
            raise RuntimeError(
                f"parameter mismatch target={target_parameters} reload={reloaded_parameters}"
            )
        log("clean_reload_ok", parameters=reloaded_parameters)
        del reloaded
        gc.collect()

        write_json(
            temporary / "DERIVATION.json",
            {
                "schema_version": "1.0",
                "source_ref": args.source_ref,
                "source_path_at_conversion": str(source),
                "source_format": "AWQ GEMM W4A16 group128 zero-point",
                "derived_format": "dense BF16 safetensors",
                "method": (
                    "vLLM AutoAWQ nibble order [0,4,1,5,2,6,3,7] with "
                    "(weight-zero)*scale CPU dequantization, exact Qwen3.5 "
                    "meta-model key/shape contract, full source-equivalence check"
                ),
                "parameter_count": target_parameters,
                "state_key_count": len(output_locations),
            },
        )

        # File fsync works on the object-backed mount. Directory fsync does not
        # have portable semantics there and can block indefinitely, so publish
        # using a same-parent rename after all files have been closed.
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
