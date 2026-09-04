from __future__ import annotations

import math
from typing import Any

from .driver import NativeDriverError


_CAPABILITY_KEYS = {
    "generation",
    "chat",
    "completion_logprobs",
    "echo",
    "tokenize",
    "detokenize",
    "media.image",
    "media.video",
    "output.grounding",
}


def validate_descriptor(value: object, *, model_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            "driver descriptor must be an object",
            status=500,
        )
    if str(value.get("model_id") or "") != model_id:
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            "driver descriptor model_id disagrees with the managed service",
            status=500,
        )
    raw = value.get("capabilities")
    if not isinstance(raw, dict):
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            "driver descriptor capabilities must be an object",
            status=500,
        )
    unknown = sorted(set(raw) - _CAPABILITY_KEYS)
    if unknown:
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            f"driver descriptor contains unknown capabilities: {unknown}",
            status=500,
        )
    capabilities = {key: bool(raw.get(key, False)) for key in _CAPABILITY_KEYS}
    if not capabilities["generation"] and not capabilities["chat"]:
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            "driver must provide generation or chat",
            status=500,
        )
    context_length = value.get("context_length")
    if context_length is not None and (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            "driver context_length must be a positive integer",
            status=500,
        )
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            "driver metadata must be an object",
            status=500,
        )
    return {
        "model_id": model_id,
        "capabilities": capabilities,
        "context_length": context_length,
        "metadata": metadata,
    }


def normalize_generation_result(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"text": value}
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise NativeDriverError(
            "DRIVER_PROTOCOL_ERROR",
            "driver result must contain string field 'text'",
            status=500,
        )
    result: dict[str, Any] = {
        "text": value["text"],
        "finish_reason": str(value.get("finish_reason") or "stop"),
    }
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens"):
        item = value.get(key)
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise NativeDriverError(
                "DRIVER_PROTOCOL_ERROR",
                f"driver result {key} must be a non-negative integer",
                status=500,
            )
        usage[key] = item
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
            "completion_tokens", 0
        )
        result["usage"] = usage
    duration = value.get("duration_seconds")
    if duration is not None:
        numeric_duration = (
            float(duration)
            if isinstance(duration, (int, float)) and not isinstance(duration, bool)
            else None
        )
        if (
            numeric_duration is None
            or not math.isfinite(numeric_duration)
            or numeric_duration < 0
        ):
            raise NativeDriverError(
                "DRIVER_PROTOCOL_ERROR",
                "duration_seconds must be finite and non-negative",
                status=500,
            )
        performance: dict[str, Any] = {"duration_seconds": numeric_duration}
        output_tokens = usage.get("completion_tokens")
        if output_tokens is not None and numeric_duration > 0:
            performance["output_tokens_per_second"] = (
                output_tokens / numeric_duration
            )
        result["performance"] = performance
    return result
