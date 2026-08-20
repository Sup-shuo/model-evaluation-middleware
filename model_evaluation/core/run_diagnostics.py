from __future__ import annotations

import copy
import traceback
from pathlib import Path
from typing import Callable

from model_evaluation.core.security import redact_text
from model_evaluation.core.serialization import json_loads_strict


def error_record(exc: BaseException, *, redact: Callable[[object], object]) -> dict:
    record = {
        "type": type(exc).__name__,
        "code": str(getattr(exc, "code", "INTERNAL_ERROR")),
        "message": str(exc),
    }
    if hasattr(exc, "retryable"):
        record["retryable"] = bool(getattr(exc, "retryable"))
    details = getattr(exc, "details", None)
    if isinstance(details, dict) and details:
        record["details"] = copy.deepcopy(details)
    return redact(record)


def current_state(run_dir: Path) -> str | None:
    try:
        value = json_loads_strict(
            (run_dir / ".run" / "status.json").read_text(encoding="utf-8")
        )
        state = value.get("state")
        return str(state) if state else None
    except Exception:
        return None


def append_core_error(
    run_dir: Path,
    stage: str,
    exc: BaseException,
    *,
    timestamp: str,
    redaction_values: tuple[str, ...],
) -> None:
    """Append a bounded diagnostic without replacing the primary failure."""
    path = run_dir / "logs" / "core_error.log"
    try:
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        rendered = redact_text(rendered, redaction_values)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] stage={stage}\n")
            handle.write(rendered)
            if rendered and not rendered.endswith("\n"):
                handle.write("\n")
            handle.write("\n")
    except Exception:
        pass


def log_tail(
    path: Path,
    *,
    redaction_values: tuple[str, ...],
    max_lines: int = 40,
    max_chars: int = 8000,
    max_bytes: int = 65536,
) -> list[str]:
    try:
        if not path.is_file():
            return []
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            take = min(size, max(1, int(max_bytes)))
            handle.seek(size - take)
            raw = handle.read(take)
        text = raw.decode("utf-8", errors="replace")
        if take < size:
            _, separator, remainder = text.partition("\n")
            if separator:
                text = remainder
        rendered = "\n".join(text.splitlines()[-max_lines:])
        rendered = redact_text(rendered[-max_chars:], redaction_values)
        return rendered.splitlines()
    except Exception:
        return []


def failure_record(
    run_dir: Path,
    *,
    stage: str,
    failure: BaseException,
    cleanup: dict,
    timestamp: str,
    error_builder: Callable[[BaseException], dict],
    redact: Callable[[object], object],
    redaction_values: tuple[str, ...],
    backend_handle=None,
    evaluator_returncode=None,
) -> dict:
    logs = {}
    for key, relative in (
        ("backend", Path("logs/backend.log")),
        ("evaluator", Path("logs/evaluation.log")),
        ("core", Path("logs/core_error.log")),
    ):
        path = run_dir / relative
        if path.is_file():
            logs[key] = {
                "path": relative.as_posix(),
                "tail": log_tail(path, redaction_values=redaction_values),
            }

    process = {}
    if backend_handle is not None:
        process["backend"] = {
            "pid": backend_handle.pid,
            "pgid": backend_handle.pgid,
            "returncode": backend_handle.poll(),
        }
    if evaluator_returncode is not None:
        process["evaluator"] = {"returncode": evaluator_returncode}

    record = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "time": timestamp,
        "stage": stage,
        "primary_error": error_builder(failure),
        "cleanup": cleanup,
        "logs": logs,
    }
    if process:
        record["process"] = process
    secondary = getattr(failure, "_model_eval_cleanup_error", None)
    if secondary is not None:
        record.setdefault("secondary_errors", []).append(
            {"stage": "PROCESS_CLEANUP", **error_builder(secondary)}
        )
    return redact(record)
