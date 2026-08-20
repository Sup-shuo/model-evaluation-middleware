#!/usr/bin/env python3
"""Safely derive a dense BF16 Qwen checkpoint from compressed-tensors.

The source is never modified.  All output is written to an exact temporary
directory, verified there, and atomically renamed to the final directory only
after a clean Transformers reload and source-equivalence validation.
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

# This utility is a CPU-only model conversion.  The MLU vLLM environment also
# exposes torch_mlu, but importing that optional backend in a CPU conversion
# process can load accelerator libraries unnecessarily.  Hide only that
# optional module from dependency feature probes; no installed file is moved
# or changed.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _conversion_find_spec(name: str, *args: Any, **kwargs: Any):
    if name.split(".", 1)[0] == "torch_mlu":
        return None
    return _ORIGINAL_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _conversion_find_spec

import torch
from accelerate import init_empty_weights
try:
    from compressed_tensors.compressors.pack_quantized.base import (
        PackedQuantizationCompressor,
    )

    _CT_DECOMPRESS_API = "state_dict_classmethod"
except ImportError:
    # compressed-tensors 0.13, used by the current MLU environment, exposes
    # the same official weight unpack/dequantize implementation under its old
    # module path and as an instance method.  Supporting both layouts avoids
    # mutating the machine environment solely for an offline conversion.
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        PackedQuantizationCompressor,
    )

    _CT_DECOMPRESS_API = "weight_instance"
from compressed_tensors.quantization import QuantizationScheme
from safetensors import safe_open
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)
from transformers.utils.quantization_config import CompressedTensorsConfig


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


def decompress_weight(module_state: dict[str, torch.Tensor], scheme: QuantizationScheme) -> torch.Tensor:
    if _CT_DECOMPRESS_API == "state_dict_classmethod":
        return PackedQuantizationCompressor.decompress(module_state, scheme)["weight"]
    return PackedQuantizationCompressor().decompress_weight(module_state, scheme.weights)


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


def build_tensor_index(root: Path) -> tuple[dict[str, Path], dict[str, tuple[tuple[int, ...], str]]]:
    locations: dict[str, Path] = {}
    information: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise RuntimeError(f"duplicate tensor key: {key}")
                view = handle.get_slice(key)
                locations[key] = shard
                information[key] = (tuple(view.get_shape()), str(view.get_dtype()))
    if not locations:
        raise RuntimeError(f"no safetensors found under {root}")
    return locations, information


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
    # recipe.yaml describes how the compressed checkpoint was quantized.  It
    # remains relevant through DERIVATION.json but must not masquerade as an
    # active recipe inside the dense derivative.
    skip_names = {"config.json", "model.safetensors.index.json", "recipe.yaml"}
    for entry in source.iterdir():
        if not entry.is_file() or entry.name in skip_names:
            continue
        if entry.name.endswith(".safetensors") or ".safetensors." in entry.name:
            continue
        shutil.copy2(entry, temporary / entry.name)


def validate_index(root: Path) -> tuple[dict[str, Path], dict[str, str], dict[str, str]]:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("model.safetensors.index.json is missing")
    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("weight_map is missing from safetensors index")
    locations, _ = build_tensor_index(root)
    missing = set(weight_map) - set(locations)
    extra = set(locations) - set(weight_map)
    wrong = {key for key in locations if weight_map.get(key) != locations[key].name}
    if missing or extra or wrong:
        raise RuntimeError(
            f"bad index: missing={len(missing)} extra={len(extra)} wrong_shard={len(wrong)}"
        )
    alias_metadata = {
        key: value
        for key, value in (index.get("metadata") or {}).items()
        if isinstance(value, str)
    }
    return locations, weight_map, alias_metadata


def resolve_tied_aliases(
    locations: dict[str, Path],
    alias_metadata: dict[str, str],
    target_keys: set[str],
    tie_word_embeddings: bool,
) -> dict[str, str]:
    """Resolve both Transformers safetensors alias metadata orientations."""
    aliases: dict[str, str] = {}
    for left, right in alias_metadata.items():
        if left in target_keys and left not in locations and right in locations:
            aliases[left] = right
        if right in target_keys and right not in locations and left in locations:
            aliases[right] = left
    # Transformers may omit the alias metadata while still de-duplicating the
    # canonical tied embedding tensor.  The model config makes this one alias
    # relationship explicit, so accept either storage orientation.
    if tie_word_embeddings:
        embedding = "model.language_model.embed_tokens.weight"
        output = "lm_head.weight"
        if embedding in target_keys and output in target_keys:
            if embedding not in locations and output in locations:
                aliases[embedding] = output
            if output not in locations and embedding in locations:
                aliases[output] = embedding
    return aliases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    source = args.source.resolve()
    temporary = args.temporary.absolute()
    final = args.final.absolute()
    if not source.is_dir():
        raise RuntimeError(f"source is not a directory: {source}")
    if temporary.exists() or final.exists():
        raise RuntimeError(f"temporary/final target already exists: {temporary} / {final}")
    if temporary.parent != final.parent:
        raise RuntimeError("temporary and final must share a parent for atomic rename")

    os.mkdir(temporary)
    published = False
    try:
        started = time.time()
        source_config_json = load_json(source / "config.json")
        architectures = source_config_json.get("architectures") or []
        if len(architectures) != 1 or architectures[0] not in ARCHITECTURE_CLASSES:
            raise RuntimeError(f"unsupported source architecture: {architectures}")
        architecture = architectures[0]
        config_class, model_class = ARCHITECTURE_CLASSES[architecture]
        raw_quantization = source_config_json.get("quantization_config")
        if not isinstance(raw_quantization, dict):
            raise RuntimeError("source has no quantization_config")
        if raw_quantization.get("quant_method") != "compressed-tensors":
            raise RuntimeError("source is not a compressed-tensors checkpoint")
        if raw_quantization.get("format") != "pack-quantized":
            raise RuntimeError("source is not pack-quantized")
        groups = raw_quantization.get("config_groups") or {}
        if len(groups) != 1:
            raise RuntimeError(f"expected one quantization group, found {len(groups)}")
        scheme = QuantizationScheme.model_validate(next(iter(groups.values())))
        weights = scheme.weights
        if weights is None or weights.num_bits != 4 or weights.group_size != 32:
            raise RuntimeError(f"unexpected source weight scheme: {weights}")

        source_locations, source_information = build_tensor_index(source)
        packed_prefixes = {
            key.removesuffix("weight_packed"): key
            for key in source_locations
            if key.endswith(".weight_packed")
        }
        if not packed_prefixes:
            raise RuntimeError("source has no packed weights")
        for prefix, packed_key in packed_prefixes.items():
            companions = [prefix + "weight_scale", prefix + "weight_shape"]
            if not weights.symmetric:
                companions.append(prefix + "weight_zero_point")
            absent = [key for key in companions if key not in source_locations]
            if absent:
                raise RuntimeError(f"missing packed companions for {packed_key}: {absent}")
        log(
            "source_preflight_ok",
            tensors=len(source_locations),
            packed_modules=len(packed_prefixes),
            symmetric=weights.symmetric,
        )

        # Build the unquantized target contract from the correct architecture.
        clean_config = config_class.from_pretrained(source, local_files_only=True)
        if hasattr(clean_config, "quantization_config"):
            delattr(clean_config, "quantization_config")
        clean_config.dtype = torch.bfloat16
        clean_config._name_or_path = ""
        if getattr(clean_config, "text_config", None) is not None:
            clean_config.text_config.dtype = torch.bfloat16
        if getattr(clean_config, "vision_config", None) is not None:
            clean_config.vision_config.dtype = torch.bfloat16
        with init_empty_weights():
            target_model = model_class(clean_config)
        target_state = target_model.state_dict()
        target_shapes = {key: tuple(value.shape) for key, value in target_state.items()}
        target_parameter_count = sum(
            __import__("math").prod(shape) for shape in target_shapes.values()
        )
        tied_embedding_keys = {
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        }
        if clean_config.tie_word_embeddings and tied_embedding_keys <= set(target_shapes):
            target_parameter_count -= __import__("math").prod(
                target_shapes["lm_head.weight"]
            )
        del target_state, target_model
        gc.collect()
        log(
            "target_contract_ready",
            architecture=architecture,
            keys=len(target_shapes),
            parameters=target_parameter_count,
        )

        load_quantization = CompressedTensorsConfig.from_dict(
            raw_quantization, run_compressed=False
        )
        log("source_model_load_started")
        model, loading_info = model_class.from_pretrained(
            source,
            local_files_only=True,
            quantization_config=load_quantization,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            values = loading_info.get(field) or []
            if values:
                raise RuntimeError(f"source load {field}: {values[:20]}")
        runtime_quantization = getattr(model.config, "quantization_config", None)
        if getattr(runtime_quantization, "run_compressed", None) is not False:
            raise RuntimeError("source did not load with run_compressed=false")
        log("source_model_load_ok", elapsed_seconds=round(time.time() - started, 1))

        loaded_state = model.state_dict()
        missing = sorted(set(target_shapes) - set(loaded_state))
        bad_shapes = sorted(
            key
            for key in set(target_shapes) & set(loaded_state)
            if tuple(loaded_state[key].shape) != target_shapes[key]
        )
        bad_dtypes = sorted(
            key
            for key in target_shapes
            if key in loaded_state and loaded_state[key].dtype != torch.bfloat16
        )
        if missing or bad_shapes or bad_dtypes:
            raise RuntimeError(
                "target filter failed: "
                f"missing={missing[:20]} bad_shapes={bad_shapes[:20]} "
                f"bad_dtypes={bad_dtypes[:20]}"
            )
        # Build the published weights directly from the raw source tensors.
        # In particular, asymmetric checkpoints may store FP16 scales.  Asking
        # Transformers to instantiate a BF16 model can cast those scales before
        # decompression and introduce an avoidable extra rounding step.  The raw
        # official decompressor followed by one BF16 cast is source-faithful.
        filtered_state: dict[str, torch.Tensor] = {}
        direct_builds = 0
        decompressed_builds = 0
        tied_peer = {
            "model.language_model.embed_tokens.weight": "lm_head.weight",
            "lm_head.weight": "model.language_model.embed_tokens.weight",
        }
        for position, key in enumerate(sorted(target_shapes), 1):
            source_key = key
            if source_key not in source_locations and key in tied_peer:
                source_key = tied_peer[key]
            if source_key in source_locations and not source_key.endswith(AUX_SUFFIXES):
                value = read_tensor(source_locations, source_key).to(torch.bfloat16)
                direct_builds += 1
            else:
                prefix = key.removesuffix("weight")
                if prefix not in packed_prefixes:
                    raise RuntimeError(f"no source mapping for target tensor: {key}")
                companion_names = [
                    prefix + suffix
                    for suffix in (
                        "weight_packed",
                        "weight_scale",
                        "weight_shape",
                        "weight_zero_point",
                        "weight_g_idx",
                    )
                    if prefix + suffix in source_locations
                ]
                module_state = {
                    name.removeprefix(prefix): read_tensor(source_locations, name)
                    for name in companion_names
                }
                value = decompress_weight(module_state, scheme).to(torch.bfloat16)
                decompressed_builds += 1
            if tuple(value.shape) != target_shapes[key]:
                raise RuntimeError(
                    f"source-built target shape mismatch for {key}: "
                    f"{tuple(value.shape)} != {target_shapes[key]}"
                )
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"bad source-built target tensor: {key}")
            filtered_state[key] = value
            if position % 100 == 0:
                log("source_build_progress", built=position, total=len(target_shapes))
        log(
            "exact_filter_ok",
            target_keys=len(filtered_state),
            dropped_source_state_keys=len(loaded_state) - len(filtered_state),
            missing=0,
            bad_shapes=0,
            bad_dtypes=0,
            direct_builds=direct_builds,
            decompressed_builds=decompressed_builds,
        )

        copy_non_weight_assets(source, temporary)

        # Do not call save_pretrained on the object constructed by the
        # compressed-tensors quantizer.  Transformers attaches private loading
        # conversion mappings to that instance and later tries to reverse them
        # during save.  A fresh, unquantized model owns the exact same target
        # contract but has no such source-checkpoint state.
        clean_save_config = copy.deepcopy(clean_config)
        if hasattr(clean_save_config, "quantization_config"):
            delattr(clean_save_config, "quantization_config")
        clean_save_config.dtype = torch.bfloat16
        clean_save_config._name_or_path = ""
        if getattr(clean_save_config, "text_config", None) is not None:
            clean_save_config.text_config.dtype = torch.bfloat16
        if getattr(clean_save_config, "vision_config", None) is not None:
            clean_save_config.vision_config.dtype = torch.bfloat16
        clean_save_config.use_cache = True
        with init_empty_weights():
            clean_model = model_class(clean_save_config)
        assign_info = clean_model.load_state_dict(filtered_state, strict=True, assign=True)
        if assign_info.missing_keys or assign_info.unexpected_keys:
            raise RuntimeError(
                f"clean model assignment failed: missing={assign_info.missing_keys[:20]} "
                f"unexpected={assign_info.unexpected_keys[:20]}"
            )
        del model, loaded_state
        gc.collect()
        log("save_started", max_shard_size=args.max_shard_size)
        clean_model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        del filtered_state, clean_model
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

        output_locations, _, alias_metadata = validate_index(temporary)
        aliases = resolve_tied_aliases(
            output_locations,
            alias_metadata,
            set(target_shapes),
            clean_config.tie_word_embeddings,
        )
        effective_output_keys = set(output_locations) | set(aliases)
        if effective_output_keys != set(target_shapes):
            raise RuntimeError(
                "serialized target key mismatch: "
                f"missing={sorted(set(target_shapes)-effective_output_keys)[:20]} "
                f"unexpected={sorted(effective_output_keys-set(target_shapes))[:20]}"
            )
        log(
            "save_index_ok",
            stored_keys=len(output_locations),
            tied_aliases=len(aliases),
            shards=len({path.name for path in output_locations.values()}),
        )

        # Full tensor-level validation against the source.  Direct tensors are
        # cast to BF16 exactly as from_pretrained does; packed tensors use the
        # official compressed-tensors decompressor before the BF16 cast.
        direct_checks = 0
        decompressed_checks = 0
        for position, key in enumerate(sorted(target_shapes), 1):
            stored_key = key if key in output_locations else aliases[key]
            derived = read_tensor(output_locations, stored_key)
            if tuple(derived.shape) != target_shapes[key]:
                raise RuntimeError(f"derived shape mismatch for {key}")
            if derived.dtype != torch.bfloat16:
                raise RuntimeError(f"derived dtype mismatch for {key}: {derived.dtype}")
            if not bool(torch.isfinite(derived).all()):
                raise RuntimeError(f"derived tensor is non-finite: {key}")

            source_key = key
            tied_peer = {
                "model.language_model.embed_tokens.weight": "lm_head.weight",
                "lm_head.weight": "model.language_model.embed_tokens.weight",
            }
            if source_key not in source_locations and key in tied_peer:
                source_key = tied_peer[key]
            if source_key in source_locations and not source_key.endswith(AUX_SUFFIXES):
                reference = read_tensor(source_locations, source_key).to(torch.bfloat16)
                direct_checks += 1
            else:
                prefix = key.removesuffix("weight")
                if prefix not in packed_prefixes:
                    raise RuntimeError(f"no source mapping for derived tensor: {key}")
                companion_names = [
                    prefix + suffix
                    for suffix in (
                        "weight_packed",
                        "weight_scale",
                        "weight_shape",
                        "weight_zero_point",
                        "weight_g_idx",
                    )
                    if prefix + suffix in source_locations
                ]
                module_state = {
                    name.removeprefix(prefix): read_tensor(source_locations, name)
                    for name in companion_names
                }
                reference = decompress_weight(module_state, scheme).to(torch.bfloat16)
                decompressed_checks += 1
            if reference.shape != derived.shape:
                raise RuntimeError(f"source reference shape mismatch for {key}")
            if not torch.equal(reference, derived):
                delta = float((reference.float() - derived.float()).abs().max())
                raise RuntimeError(f"source equivalence mismatch for {key}: max_abs={delta}")
            del reference, derived
            if position % 100 == 0:
                log("equivalence_progress", checked=position, total=len(target_shapes))
        log(
            "full_source_equivalence_ok",
            direct=direct_checks,
            decompressed=decompressed_checks,
        )

        log("clean_reload_started")
        reloaded, reload_info = model_class.from_pretrained(
            temporary,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            values = reload_info.get(field) or []
            if values:
                raise RuntimeError(f"clean reload {field}: {values[:20]}")
        if getattr(reloaded.config, "quantization_config", None) is not None:
            raise RuntimeError("clean reload retained quantization metadata")
        if any(parameter.dtype != torch.bfloat16 for parameter in reloaded.parameters()):
            raise RuntimeError("clean reload has non-BF16 parameters")
        reloaded_parameter_count = sum(parameter.numel() for parameter in reloaded.parameters())
        if reloaded_parameter_count != target_parameter_count:
            raise RuntimeError(
                f"parameter count mismatch: target={target_parameter_count} "
                f"reload={reloaded_parameter_count}"
            )
        log("clean_reload_ok", parameters=reloaded_parameter_count)
        del reloaded
        gc.collect()

        derivation = {
            "schema_version": "1.0",
            "source_ref": args.source_ref,
            "source_path_at_conversion": str(source),
            "source_format": (
                "compressed-tensors pack-quantized W4A16 group32 "
                + ("symmetric" if weights.symmetric else "asymmetric")
            ),
            "derived_format": "dense BF16 safetensors",
            "transformers": __import__("transformers").__version__,
            "compressed_tensors": __import__("compressed_tensors").__version__,
            "method": (
                f"{architecture} + CompressedTensorsConfig"
                "(run_compressed=false) structural validation + global raw source "
                "key index + official compressed-tensors scale/zero-point "
                "decompression to BF16 + exact meta-model key/shape filter + full "
                "source-equivalence validation"
            ),
            "parameter_count": target_parameter_count,
            "state_key_count": len(output_locations),
            "tied_alias_count": len(aliases),
        }
        write_json(temporary / "DERIVATION.json", derivation)

        # Flush regular files, then publish with a same-parent atomic rename.
        # Some object-backed model stores do not implement directory fsync and
        # may block indefinitely.  Do not claim POSIX directory durability on
        # those stores; full tensor validation and closed files are the useful
        # portability boundary here.
        for entry in temporary.iterdir():
            if entry.is_file():
                with entry.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.rename(temporary, final)
        published = True
        log(
            "atomic_publish_ok",
            final=final,
            elapsed_seconds=round(time.time() - started, 1),
        )
        return 0
    except BaseException:
        if not published and temporary.exists():
            # Keep failed output for diagnosis.  This project never removes a
            # model or conversion directory without explicit user approval.
            log("conversion_failed_retaining_exact_temp", temporary=temporary)
        raise


if __name__ == "__main__":
    sys.exit(main())
