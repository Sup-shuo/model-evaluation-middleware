"""Reject recorded generation failures before accepting EvalScope metrics."""
from __future__ import annotations

from pathlib import Path

from model_evaluation.sdk.jsonutil import loads as json_loads
from model_evaluation.sdk.runtime import AdapterError


def prediction_rows(root: Path):
    """Inspect structured output, not answer text or whether an answer is correct."""
    for path in sorted((root / "predictions").rglob("*.jsonl")):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json_loads(line)
                except (ValueError, TypeError) as exc:
                    raise AdapterError(
                        "RESULT_INVALID", f"invalid EvalScope prediction {path}:{number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AdapterError(
                        "RESULT_INVALID", f"EvalScope prediction must be an object: {path}:{number}"
                    )
                output = row.get("model_output")
                if not isinstance(output, dict):
                    yield path, number, row
                    continue  # Legacy prediction layouts do not expose ModelOutput.
                reason = output.get("error")
                choices = output.get("choices") or []
                if any(
                    isinstance(choice, dict) and choice.get("stop_reason") == "model_length"
                    for choice in choices
                ):
                    reason = "model_length: input exceeds the service context budget"
                if reason:
                    raise AdapterError(
                        "RESULT_INVALID",
                        f"EvalScope recorded a generation failure at {path}:{number}: {str(reason)[:300]}",
                    )
                yield path, number, row


def validate_predictions(root: Path) -> None:
    for _ in prediction_rows(root):
        pass
