"""Describe component coverage by tensor names for the supported Qwen routes."""

from collections.abc import Iterable

_COMPONENTS = {
    "visual": "vision", "vision_tower": "vision", "vision_model": "vision",
    "vision_encoder": "vision", "multi_modal_projector": "vision", "mm_projector": "vision",
    "mtp": "mtp", "nextn": "mtp",
}
_QUANT_METADATA = {
    "qzeros", "scales", "g_idx", "weight_scale", "weight_scale_inv", "weight_zero_point",
    "weight_shape", "weight_global_scale", "input_scale", "input_global_scale",
    "input_zero_point", "output_scale", "output_zero_point",
}


def _slots(keys: Iterable[str]) -> dict[str, set[str]]:
    slots = {"vision": set(), "mtp": set()}
    for key in keys:
        parts = key.split(".")
        if parts[-1] in _QUANT_METADATA:
            continue
        if parts[-1] in {"qweight", "weight_packed"}:
            parts[-1] = "weight"
        for index, segment in enumerate(parts[:-1]):
            if segment in _COMPONENTS:
                slots[_COMPONENTS[segment]].add(".".join(parts[index:]))
                break
    return slots


def _status(source: set[str], output: set[str]) -> str:
    if not source:
        return "output_only" if output else "absent_in_source"
    if not output:
        return "removed"
    return "retained" if source == output else "changed"


def component_retention(source_keys: Iterable[str], output_keys: Iterable[str]) -> dict[str, str]:
    """Compare stored parameter slots, not tensor values or runtime capabilities."""
    source, output = _slots(source_keys), _slots(output_keys)
    return {name: _status(keys, output[name]) for name, keys in source.items()}
