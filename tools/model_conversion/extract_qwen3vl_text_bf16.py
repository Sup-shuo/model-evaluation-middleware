#!/usr/bin/env python3
"""Extract the text tower of a dense Qwen3-VL checkpoint as Qwen3 BF16.

This is an explicit model transformation, not an in-place repair.  The source
directory is never modified.  The derived checkpoint is published under a new
identity only after its key/shape contract, every tensor, a clean Transformers
reload, and a short text-only logits comparison have passed.
"""

from __future__ import annotations

import argparse
import builtins
import gc
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# CPU-only conversion must not load an optional accelerator extension merely
# because it is installed in the controller environment.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _conversion_find_spec(name: str, *args: Any, **kwargs: Any):
    if name.split(".", 1)[0] == "torch_mlu":
        return None
    return _ORIGINAL_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _conversion_find_spec

# Transformers 4.x performs an additional direct torch_mlu import while
# deciding whether FlashAttention is available.  The conversion intentionally
# runs on CPU, so make that optional probe return false without changing the
# installed environment.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

_ORIGINAL_IMPORT = builtins.__import__


def _conversion_import(name: str, *args: Any, **kwargs: Any):
    if name.split(".", 1)[0] == "torch_mlu":
        raise ImportError("torch_mlu is disabled for this CPU-only conversion")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


builtins.__import__ = _conversion_import

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
)


TEXT_PREFIX = "model.language_model."
TOKENIZER_ASSETS = {
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


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


def tensor_index(root: Path) -> dict[str, Path]:
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


def text_config_from_vl(source_config: dict[str, Any]) -> tuple[Qwen3VLTextConfig, Qwen3Config]:
    if source_config.get("model_type") != "qwen3_vl":
        raise RuntimeError(f"expected qwen3_vl, got {source_config.get('model_type')!r}")
    if source_config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise RuntimeError(f"unsupported source architecture: {source_config.get('architectures')!r}")
    if source_config.get("quantization_config") is not None:
        raise RuntimeError("source must already be a dense checkpoint")
    raw = source_config.get("text_config")
    if not isinstance(raw, dict) or raw.get("model_type") != "qwen3_vl_text":
        raise RuntimeError("source has no Qwen3-VL text_config")

    vl_raw = dict(raw)
    vl_parameters = inspect.signature(Qwen3VLTextConfig.__init__).parameters
    if "rope_parameters" not in vl_parameters and "rope_scaling" in vl_parameters:
        modern_rope = dict(vl_raw.pop("rope_parameters", {}) or {})
        vl_raw["rope_theta"] = modern_rope.pop("rope_theta", vl_raw.get("rope_theta", 5000000.0))
        modern_rope.setdefault("rope_type", "default")
        vl_raw["rope_scaling"] = modern_rope or None
    vl_config = Qwen3VLTextConfig(**vl_raw)
    plain = dict(raw)
    plain.pop("model_type", None)
    rope = dict(plain.get("rope_parameters") or {})
    # For text tokens Qwen3-VL supplies the same position value on all three
    # MRoPE axes.  Removing the axis-selection metadata therefore yields the
    # exact ordinary Qwen3 RoPE frequencies; the logits check below verifies
    # this on the real weights rather than relying on the argument alone.
    rope.pop("mrope_interleaved", None)
    rope.pop("mrope_section", None)
    qwen_parameters = inspect.signature(Qwen3Config.__init__).parameters
    if "rope_parameters" in qwen_parameters:
        plain["rope_parameters"] = rope
    else:
        plain.pop("rope_parameters", None)
        plain["rope_theta"] = rope.get("rope_theta", plain.get("rope_theta", 5000000.0))
        plain["rope_scaling"] = None
    plain["tie_word_embeddings"] = bool(source_config.get("tie_word_embeddings", False))
    plain["dtype"] = "bfloat16"
    qwen_config = Qwen3Config(**plain)
    return vl_config, qwen_config


def target_mapping(target_keys: set[str], source_keys: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key in target_keys:
        source_key = key if key == "lm_head.weight" else TEXT_PREFIX + key.removeprefix("model.")
        if source_key not in source_keys:
            raise RuntimeError(f"source tensor missing for {key}: {source_key}")
        mapping[key] = source_key
    language_keys = {key for key in source_keys if key.startswith(TEXT_PREFIX)} | {
        key for key in source_keys if key == "lm_head.weight"
    }
    unused = language_keys - set(mapping.values())
    if unused:
        raise RuntimeError(f"unmapped source text tensors: {sorted(unused)[:20]}")
    return mapping


def copy_text_assets(source: Path, destination: Path) -> None:
    for name in sorted(TOKENIZER_ASSETS):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def validate_saved_index(root: Path, expected_keys: set[str]) -> dict[str, Path]:
    index = load_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("derived safetensors index has no weight_map")
    locations = tensor_index(root)
    if set(locations) != expected_keys or set(weight_map) != expected_keys:
        raise RuntimeError(
            "derived key mismatch: "
            f"stored_missing={len(expected_keys-set(locations))} "
            f"stored_extra={len(set(locations)-expected_keys)} "
            f"index_missing={len(expected_keys-set(weight_map))} "
            f"index_extra={len(set(weight_map)-expected_keys)}"
        )
    wrong = {key for key, shard in locations.items() if weight_map.get(key) != shard.name}
    if wrong:
        raise RuntimeError(f"derived index points to wrong shards: {sorted(wrong)[:20]}")
    return locations


def compare_text_logits(
    *,
    vl_config: Qwen3VLTextConfig,
    source_locations: dict[str, Path],
    mapping: dict[str, str],
    derived: Path,
) -> float:
    source_text_state = {
        key.removeprefix(TEXT_PREFIX): read_tensor(source_locations, key)
        for key in source_locations
        if key.startswith(TEXT_PREFIX)
    }
    with torch.device("meta"):
        vl_text = Qwen3VLTextModel(vl_config)
    assigned = vl_text.load_state_dict(source_text_state, strict=True, assign=True)
    if assigned.missing_keys or assigned.unexpected_keys:
        raise RuntimeError(
            f"VL text assignment mismatch: {assigned.missing_keys} / {assigned.unexpected_keys}"
        )
    # ``inv_freq`` is a non-persistent buffer, so it is absent from the state
    # dict and remains a meta tensor when the reference model is constructed on
    # the meta device.  Rebuild only the stateless rotary module on CPU before
    # comparing logits.  Without this, the comparison observes unmaterialized
    # buffer data even though every persistent model tensor is correct.
    vl_text.rotary_emb = Qwen3VLTextRotaryEmbedding(vl_config)
    lm_head = read_tensor(source_locations, mapping["lm_head.weight"])
    target, loading = Qwen3ForCausalLM.from_pretrained(
        derived,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading.get(field) or []
        if values:
            raise RuntimeError(f"clean reload {field}: {values[:20]}")
    if any(parameter.dtype != torch.bfloat16 for parameter in target.parameters()):
        raise RuntimeError("clean reload contains a non-BF16 parameter")

    # Older Transformers/PyTorch CPU combinations can produce NaNs in a BF16
    # forward even though the stored tensors are finite.  Cast both sides to
    # FP32 for this architecture-equivalence check; stored checkpoint equality
    # is checked separately in BF16 above.
    vl_text.float().eval()
    target.float().eval()
    lm_head = lm_head.float()
    # Ordinary text IDs only: no image/video placeholder is present.
    token_ids = torch.tensor([[151643, 9707, 11, 498, 525, 264, 1121]], dtype=torch.long)
    with torch.inference_mode():
        reference_hidden = vl_text(input_ids=token_ids, use_cache=False).last_hidden_state
        reference_logits = F.linear(reference_hidden, lm_head)
        target_logits = target(input_ids=token_ids, use_cache=False).logits
    if not bool(torch.isfinite(reference_logits).all()) or not bool(
        torch.isfinite(target_logits).all()
    ):
        raise RuntimeError("FP32 text-only logits are non-finite")
    max_abs = float((reference_logits.float() - target_logits.float()).abs().max())
    if not torch.equal(reference_logits, target_logits):
        raise RuntimeError(f"text-only logits differ after extraction: max_abs={max_abs}")
    del source_text_state, vl_text, lm_head, target, reference_hidden, reference_logits, target_logits
    gc.collect()
    return max_abs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--resume-existing-temporary",
        action="store_true",
        help="revalidate and publish an already serialized exact temporary directory",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    temporary = args.temporary.absolute()
    final = args.final.absolute()
    if not source.is_dir():
        raise RuntimeError(f"source is not a directory: {source}")
    if final.exists():
        raise RuntimeError(f"final target already exists: {final}")
    if args.resume_existing_temporary:
        if not temporary.is_dir():
            raise RuntimeError(f"temporary directory is missing: {temporary}")
    elif temporary.exists():
        raise RuntimeError(f"temporary target already exists: {temporary}")
    if temporary.parent != final.parent:
        raise RuntimeError("temporary and final must share a parent for atomic rename")

    if not args.resume_existing_temporary:
        os.mkdir(temporary)
    published = False
    try:
        started = time.time()
        source_config = load_json(source / "config.json")
        vl_config, target_config = text_config_from_vl(source_config)
        source_locations = tensor_index(source)

        with torch.device("meta"):
            target_meta = Qwen3ForCausalLM(target_config)
        target_shapes = {key: tuple(value.shape) for key, value in target_meta.state_dict().items()}
        mapping = target_mapping(set(target_shapes), set(source_locations))
        parameter_count = sum(math.prod(shape) for shape in target_shapes.values())
        del target_meta
        gc.collect()
        log(
            "contract_ok",
            target_keys=len(target_shapes),
            parameters=parameter_count,
            source_text_keys=len(mapping),
        )

        if not args.resume_existing_temporary:
            state: dict[str, torch.Tensor] = {}
            for position, key in enumerate(sorted(mapping), 1):
                value = read_tensor(source_locations, mapping[key])
                if tuple(value.shape) != target_shapes[key]:
                    raise RuntimeError(
                        f"shape mismatch for {key}: {tuple(value.shape)} != {target_shapes[key]}"
                    )
                if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"invalid BF16 source tensor: {mapping[key]}")
                state[key] = value
                if position % 100 == 0:
                    log("tensor_copy_progress", copied=position, total=len(mapping))

            target_config.architectures = ["Qwen3ForCausalLM"]
            target_config.dtype = torch.bfloat16
            target_config._name_or_path = ""
            with torch.device("meta"):
                target = Qwen3ForCausalLM(target_config)
            assigned = target.load_state_dict(state, strict=True, assign=True)
            if assigned.missing_keys or assigned.unexpected_keys:
                raise RuntimeError(
                    f"target assignment mismatch: {assigned.missing_keys} / {assigned.unexpected_keys}"
                )
            copy_text_assets(source, temporary)
            target.save_pretrained(
                temporary,
                safe_serialization=True,
                max_shard_size=args.max_shard_size,
            )
            del state, target
            gc.collect()

            output_config = load_json(temporary / "config.json")
            output_config.pop("_name_or_path", None)
            output_config.pop("quantization_config", None)
            output_config.pop("compression_config", None)
            output_config["architectures"] = ["Qwen3ForCausalLM"]
            output_config["model_type"] = "qwen3"
            output_config["dtype"] = "bfloat16"
            output_config["torch_dtype"] = "bfloat16"
            write_json(temporary / "config.json", output_config)
        else:
            log("resume_existing_temporary", temporary=temporary)

        output_locations = validate_saved_index(temporary, set(target_shapes))
        for position, key in enumerate(sorted(mapping), 1):
            original = read_tensor(source_locations, mapping[key])
            extracted = read_tensor(output_locations, key)
            if not torch.equal(original, extracted):
                delta = float((original.float() - extracted.float()).abs().max())
                raise RuntimeError(f"tensor changed during extraction: {key}, max_abs={delta}")
            if position % 100 == 0:
                log("equality_progress", checked=position, total=len(mapping))
        log("full_tensor_equality_ok", checked=len(mapping))

        logits_max_abs = compare_text_logits(
            vl_config=vl_config,
            source_locations=source_locations,
            mapping=mapping,
            derived=temporary,
        )
        log("text_logits_equality_ok", max_abs=logits_max_abs)

        write_json(
            temporary / "DERIVATION.json",
            {
                "schema_version": "1.0",
                "source_ref": args.source_ref,
                "source_path_at_conversion": str(source),
                "source_format": "Qwen3-VL dense BF16 safetensors",
                "derived_format": "Qwen3 text-only dense BF16 safetensors",
                "method": (
                    "exact model.language_model -> model key projection; visual tower omitted; "
                    "Qwen3-VL text MRoPE reduced to ordinary Qwen3 RoPE for equal text positions; "
                    "all tensors and deterministic text-only logits verified equal"
                ),
                "parameter_count": parameter_count,
                "state_key_count": len(target_shapes),
                "text_logits_max_abs": logits_max_abs,
                "transformers": __import__("transformers").__version__,
            },
        )

        # Files are closed before the same-parent rename.  Some object-backed
        # model stores do not implement directory fsync and can block forever;
        # do not pretend POSIX durability is available where it is not.
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
