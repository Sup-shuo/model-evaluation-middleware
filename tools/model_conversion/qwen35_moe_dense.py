"""Model-free Qwen3.5 MoE layout adapter for compressed-tensors checkpoints.

The source checkpoint stores one quantized tensor per expert projection while
Transformers exposes fused expert parameters. Quantization math remains owned
by compressed-tensors; this module only validates the source mapping and
assembles the decompressed tensors into the Transformers state contract.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.quantization import QuantizationConfig
from safetensors import safe_open


LogFunction = Callable[..., None]


def _tensor_locations(source: Path) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise RuntimeError(f"duplicate source tensor key: {key}")
                locations[key] = shard
    if not locations:
        raise RuntimeError(f"no source Safetensors found under {source}")
    return locations


def _schemes(raw_quantization: dict[str, Any]) -> dict[str, Any]:
    config = QuantizationConfig.model_validate(raw_quantization)
    schemes = {
        str(scheme.format): scheme for scheme in config.config_groups.values()
    }
    required = {"float-quantized", "nvfp4-pack-quantized"}
    missing = sorted(required - set(schemes))
    if missing:
        raise RuntimeError(f"Qwen3.5 MoE source is missing schemes: {missing}")
    return schemes


def _module_contract(
    module_name: str,
    locations: dict[str, Path],
    schemes: dict[str, Any],
) -> tuple[str, Any | None, type | None, tuple[str, ...]]:
    packed_key = f"{module_name}.weight_packed"
    weight_key = f"{module_name}.weight"
    scale_key = f"{module_name}.weight_scale"
    if packed_key in locations:
        scheme = schemes["nvfp4-pack-quantized"]
        compressor = BaseCompressor.get_value_from_registry(scheme.format)
        names = tuple(compressor.compression_param_names(scheme))
        return "compressed", scheme, compressor, names
    if weight_key in locations and scale_key in locations:
        scheme = schemes["float-quantized"]
        compressor = BaseCompressor.get_value_from_registry(scheme.format)
        names = tuple(compressor.compression_param_names(scheme))
        return "compressed", scheme, compressor, names
    if weight_key in locations:
        return "direct", None, None, ("weight",)
    raise RuntimeError(f"source projection is missing: {module_name}")


def _module_keys(
    module_name: str,
    locations: dict[str, Path],
    schemes: dict[str, Any],
) -> tuple[str, Any | None, type | None, tuple[str, ...], tuple[str, ...]]:
    mode, scheme, compressor, names = _module_contract(
        module_name,
        locations,
        schemes,
    )
    keys = tuple(f"{module_name}.{name}" for name in names)
    missing = [key for key in keys if key not in locations]
    if missing:
        raise RuntimeError(
            f"source projection dependencies are missing for {module_name}: {missing}"
        )
    return mode, scheme, compressor, names, keys


def _fused_expert_spec(key: str) -> tuple[str, str] | None:
    for projection in ("gate_up_proj", "down_proj"):
        suffix = f".experts.{projection}"
        if key.endswith(suffix):
            return key[: -len(f".{projection}")], projection
    return None


def _expected_source_keys(
    target_shapes: dict[str, tuple[int, ...]],
    locations: dict[str, Path],
    schemes: dict[str, Any],
) -> set[str]:
    expected: set[str] = set()
    for key, shape in target_shapes.items():
        fused = _fused_expert_spec(key)
        if fused is not None:
            prefix, projection = fused
            num_experts = shape[0]
            source_projections = (
                ("gate_proj", "up_proj")
                if projection == "gate_up_proj"
                else ("down_proj",)
            )
            for expert_index in range(num_experts):
                for source_projection in source_projections:
                    module_name = (
                        f"{prefix}.{expert_index}.{source_projection}"
                    )
                    *_, keys = _module_keys(module_name, locations, schemes)
                    expected.update(keys)
            continue

        if key.endswith(".weight"):
            module_name = key[: -len(".weight")]
            if (
                key in locations
                or f"{module_name}.weight_packed" in locations
            ):
                *_, keys = _module_keys(module_name, locations, schemes)
                expected.update(keys)
                continue
        if key not in locations:
            raise RuntimeError(f"source cannot map target tensor: {key}")
        expected.add(key)
    return expected


def _normalize(
    key: str,
    value: torch.Tensor,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    if value.is_floating_point() and expected_dtype == torch.bfloat16:
        value = value.to(torch.bfloat16)
    if tuple(value.shape) != expected_shape:
        raise RuntimeError(
            f"decompressed shape mismatch for {key}: "
            f"{tuple(value.shape)} != {expected_shape}"
        )
    if value.dtype != expected_dtype:
        raise RuntimeError(
            f"decompressed dtype mismatch for {key}: "
            f"{value.dtype} != {expected_dtype}"
        )
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"decompressed tensor is non-finite: {key}")
    return value.contiguous()


def build_qwen35_moe_dense_state(
    *,
    source: Path,
    raw_quantization: dict[str, Any],
    target_shapes: dict[str, tuple[int, ...]],
    target_dtypes: dict[str, torch.dtype],
    auxiliary_suffixes: tuple[str, ...],
    auxiliary_prefixes: tuple[str, ...],
    log: LogFunction,
) -> tuple[dict[str, torch.Tensor], int]:
    """Return the exact dense state required by Qwen3_5MoeForConditionalGeneration."""

    locations = _tensor_locations(source)
    schemes = _schemes(raw_quantization)
    expected_source_keys = _expected_source_keys(
        target_shapes,
        locations,
        schemes,
    )
    extras = sorted(set(locations) - expected_source_keys)
    invalid_extras = [
        key
        for key in extras
        if not key.endswith(auxiliary_suffixes)
        and not key.startswith(auxiliary_prefixes)
    ]
    if invalid_extras:
        raise RuntimeError(
            f"unexpected source tensor keys: {invalid_extras[:20]}"
        )
    log(
        "moe_source_contract_ready",
        target_keys=len(target_shapes),
        source_keys=len(locations),
        consumed_source_keys=len(expected_source_keys),
        dropped_auxiliary_keys=len(extras),
    )

    state: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    fused_total = sum(
        _fused_expert_spec(key) is not None for key in target_shapes
    )
    fused_done = 0

    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(locations.values()))
        }

        def load(key: str) -> torch.Tensor:
            consumed.add(key)
            return handles[locations[key]].get_tensor(key)

        def load_projection(module_name: str) -> torch.Tensor:
            mode, scheme, compressor, names, keys = _module_keys(
                module_name,
                locations,
                schemes,
            )
            if mode == "direct":
                return load(keys[0]).clone()
            assert scheme is not None and compressor is not None
            local_state = {
                name: load(key) for name, key in zip(names, keys, strict=True)
            }
            return compressor.decompress(local_state, scheme)["weight"]

        for key, expected_shape in target_shapes.items():
            expected_dtype = target_dtypes[key]
            fused = _fused_expert_spec(key)
            if fused is not None:
                prefix, projection = fused
                output = torch.empty(
                    expected_shape,
                    dtype=expected_dtype,
                    device="cpu",
                )
                for expert_index in range(expected_shape[0]):
                    if projection == "gate_up_proj":
                        gate = load_projection(
                            f"{prefix}.{expert_index}.gate_proj"
                        ).to(expected_dtype)
                        up = load_projection(
                            f"{prefix}.{expert_index}.up_proj"
                        ).to(expected_dtype)
                        value = torch.cat((gate, up), dim=0)
                        del gate, up
                    else:
                        value = load_projection(
                            f"{prefix}.{expert_index}.down_proj"
                        ).to(expected_dtype)
                    if tuple(value.shape) != expected_shape[1:]:
                        raise RuntimeError(
                            f"expert shape mismatch for {key}[{expert_index}]: "
                            f"{tuple(value.shape)} != {expected_shape[1:]}"
                        )
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(
                            f"expert tensor is non-finite: {key}[{expert_index}]"
                        )
                    output[expert_index].copy_(value)
                    del value
                state[key] = output
                fused_done += 1
                log(
                    "moe_layout_fusion_progress",
                    completed=fused_done,
                    total=fused_total,
                    key=key,
                )
                continue

            if key.endswith(".weight"):
                module_name = key[: -len(".weight")]
                if (
                    key in locations
                    or f"{module_name}.weight_packed" in locations
                ):
                    value = load_projection(module_name)
                else:
                    raise RuntimeError(f"source cannot map target tensor: {key}")
            else:
                if key not in locations:
                    raise RuntimeError(f"source cannot map target tensor: {key}")
                value = load(key).clone()
            state[key] = _normalize(
                key,
                value,
                expected_shape,
                expected_dtype,
            )

    if consumed != expected_source_keys:
        missing = sorted(expected_source_keys - consumed)
        unexpected = sorted(consumed - expected_source_keys)
        raise RuntimeError(
            f"MoE source consumption mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    return state, len(extras)


def serialized_expert_parts(
    key: str,
    reference: torch.Tensor,
    locations: dict[str, Path],
) -> list[tuple[str, torch.Tensor]] | None:
    """Resolve a fused internal expert tensor to Transformers save-format slices."""

    fused = _fused_expert_spec(key)
    if fused is None:
        return None
    prefix, projection = fused
    if reference.ndim != 3:
        raise RuntimeError(
            f"fused expert tensor must be rank 3: {key} {tuple(reference.shape)}"
        )

    parts: list[tuple[str, torch.Tensor]] = []
    if projection == "gate_up_proj":
        if reference.shape[1] % 2:
            raise RuntimeError(f"gate/up dimension is not even: {key}")
        intermediate = reference.shape[1] // 2
        for expert_index in range(reference.shape[0]):
            parts.extend(
                (
                    (
                        f"{prefix}.{expert_index}.gate_proj.weight",
                        reference[expert_index, :intermediate],
                    ),
                    (
                        f"{prefix}.{expert_index}.up_proj.weight",
                        reference[expert_index, intermediate:],
                    ),
                )
            )
    else:
        parts.extend(
            (
                f"{prefix}.{expert_index}.down_proj.weight",
                reference[expert_index],
            )
            for expert_index in range(reference.shape[0])
        )

    missing = [stored_key for stored_key, _ in parts if stored_key not in locations]
    if missing:
        return None
    return parts
