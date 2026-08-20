from __future__ import annotations

from datetime import datetime
from pathlib import Path

from model_evaluation.core.errors import ResultProductError
from model_evaluation.core.serialization import json_loads_strict


FINAL_PRODUCT_FILES = {
    "result": "result.json",
    "metrics": "metrics.json",
    "terminal": "terminal.json",
    "failure": "failure.json",
}


def _load(path: Path) -> dict:
    if not path.is_file():
        raise ResultProductError(f"result product file is missing: {path.name}")
    value = json_loads_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResultProductError(f"result product file must contain an object: {path.name}")
    return value


def _same(label: str, left: object, right: object) -> None:
    if left != right:
        raise ResultProductError(f"result product mismatch for {label}: {left!r} != {right!r}")


def _not_after(label: str, earlier: str, later: str) -> None:
    """Require two Schema-validated RFC 3339 timestamps to be ordered."""

    try:
        left = datetime.fromisoformat(earlier.replace("Z", "+00:00"))
        right = datetime.fromisoformat(later.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ResultProductError(f"invalid timestamp while checking {label}") from exc
    if left > right:
        raise ResultProductError(
            f"result product mismatch for {label}: {earlier!r} is after {later!r}"
        )


def _artifact(root: Path, relative: str, *, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ResultProductError(f"{label} must use a confined relative path: {relative!r}")
    path = root.joinpath(raw)
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise ResultProductError(f"{label} may not traverse a symlink: {relative!r}")
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ResultProductError(f"{label} escapes the run directory: {relative!r}")
    if not resolved.is_file():
        raise ResultProductError(f"{label} does not exist: {relative!r}")
    return resolved


def _effective_samples(metrics: dict) -> int | None:
    values: list[int] = []
    for detail in (metrics.get("tasks") or {}).values():
        count = (detail.get("sample_count") or {}).get("effective")
        if isinstance(count, int):
            values.append(count)
    return sum(values) if values else None


def inspect_run_product(run_dir: str | Path, schemas) -> dict:
    """Validate a finished run and return its stable, human-facing summary.

    This checks JSON Schema, cross-file identities and metrics, and that every
    public artifact reference is a real file confined to the run directory.
    It deliberately does not make provenance or tamper-resistance claims.
    """

    supplied = Path(run_dir).expanduser()
    if supplied.is_symlink():
        raise ResultProductError(f"run directory may not be a symlink: {supplied}")
    root = supplied.resolve()
    if not root.is_dir():
        raise ResultProductError(f"run directory not found: {root}")

    terminal = _load(root / FINAL_PRODUCT_FILES["terminal"])
    schemas.validate("terminal", terminal)
    _same("terminal.run_id", terminal["run_id"], root.name)

    outcome = terminal["outcome"]
    failure_path = root / FINAL_PRODUCT_FILES["failure"]
    result_path = root / FINAL_PRODUCT_FILES["result"]
    metrics_path = root / FINAL_PRODUCT_FILES["metrics"]

    if outcome == "success" and failure_path.exists():
        raise ResultProductError("successful run must not contain failure.json")
    if outcome == "failed" and not failure_path.is_file():
        raise ResultProductError("failed run must contain failure.json")
    if result_path.exists() != metrics_path.exists():
        raise ResultProductError("result.json and metrics.json must be published together")
    if outcome == "success" and not result_path.is_file():
        raise ResultProductError("successful run must contain result.json and metrics.json")

    failure = None
    if failure_path.is_file():
        failure = _load(failure_path)
        schemas.validate("failure", failure)
        _same("failure.run_id", failure["run_id"], terminal["run_id"])
        _same("failure.cleanup", failure["cleanup"], terminal["cleanup"])
        _same("failure error", failure["primary_error"], terminal["error"])
        for name, log in (failure.get("logs") or {}).items():
            _artifact(root, log["path"], label=f"failure log {name}")

    result = None
    metrics = None
    artifact_count = 0
    if result_path.is_file():
        result = _load(result_path)
        metrics = _load(metrics_path)
        schemas.validate("result", result)
        schemas.validate("metrics", metrics)
        for key in ("run_id", "model", "benchmark", "framework"):
            _same(key, result[key], metrics[key])
        _same("result.run_id", result["run_id"], terminal["run_id"])
        _same("summary metrics", result["metrics"], metrics["summary"])
        breakdowns = result.get("breakdowns") or {}
        if breakdowns:
            _same(
                "breakdown summary metrics",
                breakdowns["summary"]["metrics"],
                result["metrics"],
            )
        _same("group metrics", breakdowns.get("groups") or {}, metrics["groups"])
        _same("task metrics", breakdowns.get("tasks") or {}, metrics["tasks"])
        _artifact(root, result["raw_result"]["path"], label="raw result")
        artifact_count += 1
        for index, artifact in enumerate(result.get("sample_artifacts") or [], 1):
            _artifact(root, artifact["path"], label=f"sample artifact {index}")
            artifact_count += 1
        metadata = result.get("metadata") or {}
        for key in ("started_at", "timezone"):
            if key in metadata:
                _same(f"metadata.{key}", metadata[key], terminal[key])
        if "finished_at" in metadata:
            # result.json is published before backend cleanup and terminal.json.
            # Its completion timestamp may therefore precede the final terminal
            # timestamp by a few seconds, but it must never be later.
            _not_after(
                "metadata.finished_at",
                metadata["finished_at"],
                terminal["finished_at"],
            )

    return {
        "ok": True,
        "schema_version": "1.0",
        "run_dir": str(root),
        "run_id": terminal["run_id"],
        "outcome": outcome,
        "started_at": terminal["started_at"],
        "finished_at": terminal["finished_at"],
        "timezone": terminal["timezone"],
        "cleanup": terminal["cleanup"]["status"],
        "model": result.get("model") if result else None,
        "benchmark": result.get("benchmark") if result else None,
        "framework": result.get("framework") if result else None,
        "summary": metrics.get("summary") if metrics else None,
        "groups": len(metrics.get("groups") or {}) if metrics else 0,
        "tasks": len(metrics.get("tasks") or {}) if metrics else 0,
        "effective_samples": _effective_samples(metrics) if metrics else None,
        "artifacts": artifact_count,
        "failure_stage": failure.get("stage") if failure else None,
        "error": failure.get("primary_error") if failure else None,
    }
