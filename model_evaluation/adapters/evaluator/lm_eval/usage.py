from __future__ import annotations

import json
import os
import threading
from pathlib import Path


_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _non_negative_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _usage_counts(payload: bytes) -> dict[str, int] | None:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    usage = document.get("usage") if isinstance(document, dict) else None
    if not isinstance(usage, dict):
        return None
    values = {name: _non_negative_integer(usage.get(name)) for name in _TOKEN_FIELDS}
    if any(value is None for value in values.values()):
        return None
    if values["total_tokens"] != (
        values["prompt_tokens"] + values["completion_tokens"]
    ):
        return None
    return values


class UsageAccumulator:
    """Persist exact OpenAI response usage without affecting proxy traffic."""

    def __init__(self, path: Path):
        self.path = path
        # ``observe`` persists a snapshot while holding the lock; the write
        # path calls ``snapshot`` again, so re-entrancy is required.
        self._lock = threading.RLock()
        self._responses = 0
        self._responses_with_usage = 0
        self._totals = {name: 0 for name in _TOKEN_FIELDS}

    def observe(self, payload: bytes) -> None:
        counts = _usage_counts(payload)
        with self._lock:
            self._responses += 1
            if counts is not None:
                self._responses_with_usage += 1
                for name in _TOKEN_FIELDS:
                    self._totals[name] += counts[name]
            try:
                self._write()
            except OSError:
                # Performance collection is optional and must never alter an
                # otherwise valid evaluator response.
                return

    def snapshot(self) -> dict:
        complete = (
            self._responses > 0
            and self._responses == self._responses_with_usage
        )
        counts = (
            dict(self._totals)
            if complete
            else {name: None for name in _TOKEN_FIELDS}
        )
        return {
            "schema_version": "1.0",
            "source": "openai_response_usage",
            "complete": complete,
            "successful_responses": self._responses,
            "responses_with_usage": self._responses_with_usage,
            **counts,
        }

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        temporary.write_text(
            json.dumps(
                self.snapshot(),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


__all__ = ["UsageAccumulator"]
