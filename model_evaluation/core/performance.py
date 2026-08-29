from __future__ import annotations

from typing import Any


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _non_negative_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def normalize_token_usage(value: object) -> dict[str, Any]:
    """Return conservative token usage without inferring missing counters."""

    raw = value if isinstance(value, dict) else {}
    source = str(raw.get("source") or "unavailable")
    values = {name: _non_negative_integer(raw.get(name)) for name in TOKEN_FIELDS}
    complete = bool(raw.get("complete")) and all(
        item is not None for item in values.values()
    )
    if complete and values["total_tokens"] != (
        values["prompt_tokens"] + values["completion_tokens"]
    ):
        complete = False
    if not complete:
        values = {name: None for name in TOKEN_FIELDS}
    return {"source": source, "complete": complete, **values}


def build_performance(
    *,
    run_id: str,
    model: str,
    benchmark: str,
    framework: str,
    evaluation_started_at: str,
    evaluation_finished_at: str,
    evaluation_duration_seconds: float,
    token_usage: object = None,
) -> dict[str, Any]:
    duration = round(max(0.0, float(evaluation_duration_seconds)), 6)
    usage = normalize_token_usage(token_usage)
    throughput = None
    if usage["complete"] and duration > 0:
        throughput = round(usage["total_tokens"] / duration, 6)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "model": model,
        "benchmark": benchmark,
        "framework": framework,
        "measurement_scope": "evaluator_process",
        "evaluation_started_at": evaluation_started_at,
        "evaluation_finished_at": evaluation_finished_at,
        "evaluation_duration_seconds": duration,
        "token_usage": usage,
        "tokens_per_second": throughput,
    }


def validate_performance_values(payload: dict[str, Any]) -> None:
    usage = normalize_token_usage(payload.get("token_usage"))
    if usage != payload.get("token_usage"):
        raise ValueError("token_usage is incomplete or internally inconsistent")
    duration = payload.get("evaluation_duration_seconds")
    expected = None
    if usage["complete"] and isinstance(duration, (int, float)) and duration > 0:
        expected = round(usage["total_tokens"] / float(duration), 6)
    if payload.get("tokens_per_second") != expected:
        raise ValueError(
            "tokens_per_second must equal total_tokens / "
            "evaluation_duration_seconds when exact usage is available"
        )


__all__ = [
    "TOKEN_FIELDS",
    "build_performance",
    "normalize_token_usage",
    "validate_performance_values",
]
