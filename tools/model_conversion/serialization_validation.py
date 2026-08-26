"""Exact serialized-state validation for manual model conversions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open


PartsResolver = Callable[
    [str, torch.Tensor, dict[str, Path]],
    list[tuple[str, torch.Tensor]] | None,
]
LogFunction = Callable[..., None]


def build_tensor_index(root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    locations: dict[str, Path] = {}
    aliases: dict[str, str] = {}
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        metadata = index.get("metadata") or {}
        if isinstance(metadata, dict):
            aliases = {
                str(key): str(value)
                for key, value in metadata.items()
                if isinstance(key, str) and isinstance(value, str)
            }
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise RuntimeError(f"duplicate serialized tensor key: {key}")
                locations[key] = shard
    if not locations:
        raise RuntimeError(f"no serialized Safetensors found under {root}")
    return locations, aliases


def _resolve_direct_key(
    key: str,
    locations: dict[str, Path],
    aliases: dict[str, str],
) -> str | None:
    if key in locations:
        return key
    peer = aliases.get(key)
    if peer in locations:
        return peer
    return next(
        (
            left
            for left, right in aliases.items()
            if right == key and left in locations
        ),
        None,
    )


def _read_tensor(locations: dict[str, Path], key: str) -> torch.Tensor:
    with safe_open(locations[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def validate_serialized_state(
    *,
    root: Path,
    target_state: dict[str, torch.Tensor],
    parts_resolver: PartsResolver | None,
    log: LogFunction,
) -> dict[str, int]:
    locations, aliases = build_tensor_index(root)
    covered: set[str] = set()
    unresolved: list[str] = []

    for position, (key, reference) in enumerate(target_state.items(), 1):
        stored_key = _resolve_direct_key(key, locations, aliases)
        if stored_key is not None:
            parts = [(stored_key, reference)]
        elif parts_resolver is not None:
            parts = parts_resolver(key, reference, locations)
        else:
            parts = None
        if parts is None:
            unresolved.append(key)
            continue

        for part_key, part_reference in parts:
            derived = _read_tensor(locations, part_key)
            if (
                derived.dtype != part_reference.dtype
                or tuple(derived.shape) != tuple(part_reference.shape)
            ):
                raise RuntimeError(
                    f"serialized contract mismatch for {key} via {part_key}"
                )
            if not torch.equal(derived, part_reference):
                maximum = float(
                    (derived.float() - part_reference.float()).abs().max()
                )
                raise RuntimeError(
                    f"serialized source-equivalence mismatch for {key} "
                    f"via {part_key}: max_abs={maximum}"
                )
            covered.add(part_key)
            del derived, part_reference

        if position % 100 == 0:
            log(
                "serialized_equivalence_progress",
                checked=position,
                total=len(target_state),
            )

    if unresolved:
        raise RuntimeError(f"serialized tensors missing: {unresolved[:20]}")
    unexpected = sorted(set(locations) - covered)
    if unexpected:
        raise RuntimeError(
            f"unexpected serialized tensor keys: {unexpected[:20]}"
        )
    return {
        "keys": len(target_state),
        "serialized_keys": len(covered),
        "shards": len({path.name for path in locations.values()}),
    }
