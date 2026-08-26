#!/usr/bin/env python3
"""Manually derive a dense BF16 Qwen checkpoint from mixed compressed-tensors.

The source is read-only. Output is built in an exact temporary directory,
validated against the official decompressed model state, clean-reloaded, and
only then atomically renamed to the requested final directory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from serialization_validation import validate_serialized_state
from transformers import (
    AutoConfig,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from transformers.utils.quantization_config import CompressedTensorsConfig


ARCHITECTURE_CLASSES = {
    "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGeneration,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGeneration,
}
SUPPORTED_ARCHITECTURES = set(ARCHITECTURE_CLASSES)
AUXILIARY_SUFFIXES = (
    ".weight_packed",
    ".weight_scale",
    ".weight_shape",
    ".weight_zero_point",
    ".weight_g_idx",
    ".input_global_scale",
    ".weight_global_scale",
    ".k_scale",
    ".v_scale",
)
AUXILIARY_PREFIXES = ("mtp.",)


def log(stage: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {stage}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def remove_quantization_metadata(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        value.pop("compression_config", None)
        for child in value.values():
            remove_quantization_metadata(child)
    elif isinstance(value, list):
        for child in value:
            remove_quantization_metadata(child)


def copy_non_weight_assets(source: Path, output: Path) -> None:
    skipped = {"config.json", "model.safetensors.index.json", "recipe.yaml"}
    for entry in source.iterdir():
        if not entry.is_file() or entry.name in skipped:
            continue
        if entry.name.endswith(".safetensors") or ".safetensors." in entry.name:
            continue
        shutil.copy2(entry, output / entry.name)


def validate_source_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    architectures = config.get("architectures") or []
    if len(architectures) != 1 or architectures[0] not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(f"unsupported source architecture: {architectures}")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise RuntimeError("source has no quantization_config")
    if quantization.get("quant_method") != "compressed-tensors":
        raise RuntimeError("source is not a compressed-tensors checkpoint")
    if quantization.get("format") != "mixed-precision":
        raise RuntimeError("source is not mixed-precision")
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise RuntimeError("mixed-precision source has no config_groups")
    formats = {
        group.get("format")
        for group in groups.values()
        if isinstance(group, dict)
    }
    if "nvfp4-pack-quantized" not in formats:
        raise RuntimeError(f"mixed source has no NVFP4 group: {sorted(map(str, formats))}")
    return architectures[0], quantization


def clean_config(config: Any) -> Any:
    result = copy.deepcopy(config)
    if hasattr(result, "quantization_config"):
        delattr(result, "quantization_config")
    result.dtype = torch.bfloat16
    result._name_or_path = ""
    for section in ("text_config", "vision_config"):
        child = getattr(result, section, None)
        if child is not None:
            child.dtype = torch.bfloat16
    if hasattr(result, "use_cache"):
        result.use_cache = True
    return result


def validate_loading_info(
    stage: str,
    info: dict[str, Any],
    *,
    tolerate_key_remapping: bool = False,
) -> None:
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = list(info.get(field) or [])
        if not values:
            continue
        if tolerate_key_remapping and field in {"missing_keys", "unexpected_keys"}:
            log(
                "compressed_loading_key_remapping_reported",
                source_stage=stage.replace(" ", "_"),
                field=field,
                count=len(values),
            )
            continue
        raise RuntimeError(f"{stage} {field}: {values[:20]}")


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
    if not final.parent.is_dir():
        raise RuntimeError(f"final parent does not exist: {final.parent}")

    os.mkdir(temporary)
    published = False
    started = time.time()
    try:
        source_json = read_json(source / "config.json")
        architecture, raw_quantization = validate_source_config(source_json)
        model_class = ARCHITECTURE_CLASSES[architecture]
        clean_target_config = clean_config(
            AutoConfig.from_pretrained(source, local_files_only=True)
        )
        with init_empty_weights():
            target_model = model_class(clean_target_config)
        target_state = target_model.state_dict()
        target_shapes = {key: tuple(value.shape) for key, value in target_state.items()}
        target_dtypes = {
            key: torch.bfloat16 if value.is_floating_point() else value.dtype
            for key, value in target_state.items()
        }
        target_parameters = sum(parameter.numel() for parameter in target_model.parameters())
        del target_state, target_model
        gc.collect()
        log(
            "target_contract_ready",
            architecture=architecture,
            keys=len(target_shapes),
            parameters=target_parameters,
        )

        model: Any | None = None
        loaded_state: dict[str, torch.Tensor] | None = None
        serialized_parts_resolver = None
        if architecture == "Qwen3_5MoeForConditionalGeneration":
            from qwen35_moe_dense import (
                build_qwen35_moe_dense_state,
                serialized_expert_parts,
            )

            serialized_parts_resolver = serialized_expert_parts
            previous_threads = torch.get_num_threads()
            conversion_threads = min(16, previous_threads)
            log(
                "official_decompression_started",
                layout_adapter="qwen35_moe_expert_fusion",
                torch_threads=conversion_threads,
            )
            torch.set_num_threads(conversion_threads)
            try:
                filtered_state, dropped_auxiliary_keys = (
                    build_qwen35_moe_dense_state(
                        source=source,
                        raw_quantization=raw_quantization,
                        target_shapes=target_shapes,
                        target_dtypes=target_dtypes,
                        auxiliary_suffixes=AUXILIARY_SUFFIXES,
                        auxiliary_prefixes=AUXILIARY_PREFIXES,
                        log=log,
                    )
                )
            finally:
                torch.set_num_threads(previous_threads)
            log(
                "official_decompression_ok",
                keys=len(filtered_state),
                dropped_auxiliary_keys=dropped_auxiliary_keys,
                layout_adapter="qwen35_moe_expert_fusion",
                elapsed_seconds=round(time.time() - started, 1),
            )
            derivation_method = (
                "compressed-tensors official per-projection decompression; "
                "Transformers Qwen3.5 MoE expert layout fusion; exclusion of the "
                "Transformers-ignored MTP speculative-decoding auxiliary head; "
                "clean architecture state contract; exact serialized tensor "
                "comparison; clean Transformers reload"
            )
        else:
            load_quantization = CompressedTensorsConfig.from_dict(
                raw_quantization,
                run_compressed=False,
            )
            log("official_decompression_started")
            model, loading_info = model_class.from_pretrained(
                source,
                local_files_only=True,
                quantization_config=load_quantization,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                output_loading_info=True,
            )
            validate_loading_info(
                "source load",
                loading_info,
                tolerate_key_remapping=True,
            )
            runtime_quantization = getattr(
                model.config,
                "quantization_config",
                None,
            )
            if getattr(runtime_quantization, "run_compressed", None) is not False:
                raise RuntimeError("source did not load with run_compressed=false")
            if hasattr(model, "hf_quantizer"):
                model.hf_quantizer.remove_quantization_config(model)

            loaded_state = model.state_dict()
            missing = sorted(set(target_shapes) - set(loaded_state))
            bad_shapes = sorted(
                key
                for key in set(target_shapes) & set(loaded_state)
                if tuple(loaded_state[key].shape) != target_shapes[key]
            )
            if missing or bad_shapes:
                raise RuntimeError(
                    f"decompressed state mismatch: missing={missing[:20]} "
                    f"bad_shapes={bad_shapes[:20]}"
                )
            filtered_state = {}
            for key in target_shapes:
                value = loaded_state[key]
                expected_dtype = target_dtypes[key]
                if value.is_floating_point() and expected_dtype == torch.bfloat16:
                    value = value.to(torch.bfloat16)
                if value.dtype != expected_dtype:
                    raise RuntimeError(
                        f"decompressed dtype mismatch for {key}: "
                        f"{value.dtype} != {expected_dtype}"
                    )
                if value.is_floating_point() and not bool(
                    torch.isfinite(value).all()
                ):
                    raise RuntimeError(
                        f"decompressed tensor is non-finite: {key}"
                    )
                filtered_state[key] = value
            extras = sorted(set(loaded_state) - set(target_shapes))
            invalid_extras = [
                key for key in extras if not key.endswith(AUXILIARY_SUFFIXES)
            ]
            if invalid_extras:
                raise RuntimeError(
                    f"unexpected decompressed state keys: {invalid_extras[:20]}"
                )
            log(
                "official_decompression_ok",
                keys=len(filtered_state),
                dropped_auxiliary_keys=len(extras),
                elapsed_seconds=round(time.time() - started, 1),
            )
            derivation_method = (
                "CompressedTensorsConfig(run_compressed=false) official "
                "decompression; clean architecture state contract; exact "
                "serialized tensor comparison; clean Transformers reload"
            )

        copy_non_weight_assets(source, temporary)
        with init_empty_weights():
            clean_model = model_class(clean_target_config)
        assignment = clean_model.load_state_dict(filtered_state, strict=True, assign=True)
        if assignment.missing_keys or assignment.unexpected_keys:
            raise RuntimeError(
                f"clean assignment failed: missing={assignment.missing_keys[:20]} "
                f"unexpected={assignment.unexpected_keys[:20]}"
            )
        del model, loaded_state
        gc.collect()

        log("dense_save_started", max_shard_size=args.max_shard_size)
        clean_model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        output_config_path = temporary / "config.json"
        output_config = read_json(output_config_path)
        remove_quantization_metadata(output_config)
        output_config.pop("_name_or_path", None)
        output_config["dtype"] = "bfloat16"
        output_config.pop("torch_dtype", None)
        for section in ("text_config", "vision_config"):
            if isinstance(output_config.get(section), dict):
                output_config[section]["dtype"] = "bfloat16"
                output_config[section].pop("torch_dtype", None)
        write_json(output_config_path, output_config)

        serialized_validation = validate_serialized_state(
            root=temporary,
            target_state=filtered_state,
            parts_resolver=serialized_parts_resolver,
            log=log,
        )
        log(
            "serialized_source_equivalence_ok",
            **serialized_validation,
        )
        del filtered_state, clean_model
        gc.collect()

        log("clean_reload_started")
        reloaded, reload_info = model_class.from_pretrained(
            temporary,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        validate_loading_info("clean reload", reload_info)
        if getattr(reloaded.config, "quantization_config", None) is not None:
            raise RuntimeError("clean reload retained quantization metadata")
        if any(parameter.dtype != torch.bfloat16 for parameter in reloaded.parameters()):
            raise RuntimeError("clean reload contains non-BF16 parameters")
        reloaded_parameters = sum(parameter.numel() for parameter in reloaded.parameters())
        if reloaded_parameters != target_parameters:
            raise RuntimeError(
                f"parameter count mismatch: {reloaded_parameters} != {target_parameters}"
            )
        del reloaded
        gc.collect()
        log("clean_reload_ok", parameters=reloaded_parameters)

        derivation = {
            "schema_version": "1.0",
            "source_ref": args.source_ref,
            "source_path_at_conversion": str(source),
            "source_format": "compressed-tensors mixed-precision NVFP4/FP8",
            "derived_format": "dense BF16 safetensors",
            "architecture": architecture,
            "transformers": __import__("transformers").__version__,
            "compressed_tensors": __import__("compressed_tensors").__version__,
            "method": derivation_method,
            "parameter_count": target_parameters,
            "state_key_count": len(target_shapes),
        }
        write_json(temporary / "DERIVATION.json", derivation)
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
            log("conversion_failed_retaining_exact_temp", temporary=temporary)
        raise


if __name__ == "__main__":
    sys.exit(main())
