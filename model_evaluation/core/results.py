from __future__ import annotations

import copy
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from model_evaluation.core.files import atomic_json
from model_evaluation.core.errors import CompatibilityError, ConfigError


DEFAULT_TIMEZONE = "Asia/Shanghai"


def plan_timezone(plan: dict) -> ZoneInfo:
    platform = (((plan.get("resolved") or {}).get("specs") or {}).get("platform") or {})
    name = str((platform.get("metadata") or {}).get("timezone") or DEFAULT_TIMEZONE)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ConfigError(f"system timezone is unavailable: {name!r}") from exc


def local_now(plan: dict) -> datetime:
    return datetime.now(plan_timezone(plan))


def iso_now(plan: dict) -> str:
    return local_now(plan).isoformat(timespec="seconds")


def _component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9.+-]+", "-", str(value)).strip(".-") or "unknown"
    return text[:80].rstrip(".-") or "unknown"


def run_id_base(plan: dict, *, when: datetime | None = None) -> str:
    run_spec = plan.get("run_spec") or {}
    model_spec = (((plan.get("resolved") or {}).get("specs") or {}).get("model") or {})
    model_identity = (
        model_spec.get("experiment_id")
        or (model_spec.get("metadata") or {}).get("experiment_id")
        or run_spec.get("model")
        or "unknown-model"
    )
    stamp = (when or local_now(plan)).strftime("%Y%m%d-%H%M%S")
    return f"{_component(model_identity)}_{_component(run_spec.get('benchmark') or 'unknown-benchmark')}_{stamp}"


def allocate_run_dir(results_root: str | Path, plan: dict) -> Path:
    root = Path(results_root).resolve()
    base = run_id_base(plan)
    for index in range(1, 10_000):
        name = base if index == 1 else f"{base}-{index}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise ConfigError(f"could not allocate a unique result directory for {base!r}")


def build_run_config(plan: dict, *, run_id: str, started_at: str) -> dict:
    specs = copy.deepcopy(((plan.get("resolved") or {}).get("specs") or {}))
    resolved_platform = copy.deepcopy((plan.get("resolved") or {}).get("platform") or {})
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "plan_id": plan.get("plan_id"),
        "started_at": started_at,
        "timezone": getattr(plan_timezone(plan), "key", DEFAULT_TIMEZONE),
        "adapters": copy.deepcopy(plan.get("adapters") or []),
        "selection": copy.deepcopy(plan.get("run_spec") or {}),
        "model": specs.get("model") or {},
        "benchmark": specs.get("benchmark") or {},
        "backend": specs.get("deployment") or {},
        "evaluator": specs.get("evaluation") or {},
        "system": specs.get("platform") or {},
        "resolved_runtime": resolved_platform,
    }


def _confined_file(value: str | Path, root: Path, *, label: str) -> Path:
    base = root.resolve()
    raw = Path(value)
    path = raw.resolve()
    if base not in path.parents or not path.is_file():
        raise CompatibilityError(f"{label} is outside the evaluator output directory: {path}")
    relative = path.relative_to(base)
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CompatibilityError(f"{label} may not traverse a symlink: {current}")
    return path


def _copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def publish_result(run_dir: str | Path, raw_root: str | Path, result: dict) -> dict:
    """Publish framework output as a compact, user-facing experiment result.

    Framework-native files remain lossless under ``raw/``.  ``result.json`` and
    ``metrics.json`` are convenient views; they are not cryptographic evidence.
    """

    root = Path(run_dir).resolve()
    evaluator_root = Path(raw_root).resolve()
    published = copy.deepcopy(result)

    raw_ref = published.get("raw_result") or {}
    source = _confined_file(raw_ref.get("path") or "", evaluator_root, label="framework result")
    suffix = source.suffix if source.suffix else ".json"
    destination = root / "raw" / f"framework_result{suffix}"
    _copy_artifact(source, destination)
    published["raw_result"] = {
        "path": destination.relative_to(root).as_posix(),
        "media_type": raw_ref.get("media_type") or "application/json",
    }

    sample_refs = []
    seen_names: set[str] = set()
    for index, artifact in enumerate(published.pop("sample_artifacts", []) or [], 1):
        sample_source = _confined_file(artifact.get("path") or "", evaluator_root, label="sample result")
        base_name = re.sub(r"[^A-Za-z0-9._+-]+", "-", sample_source.name).strip(".-") or f"samples-{index}.jsonl"
        name = base_name
        counter = 2
        while name in seen_names:
            stem, suffix = Path(base_name).stem, Path(base_name).suffix
            name = f"{stem}-{counter}{suffix}"
            counter += 1
        seen_names.add(name)
        sample_destination = root / "samples" / name
        _copy_artifact(sample_source, sample_destination)
        sample_refs.append({
            "path": sample_destination.relative_to(root).as_posix(),
            "media_type": artifact.get("media_type") or "application/x-ndjson",
        })
    if sample_refs:
        published["sample_artifacts"] = sample_refs

    breakdowns = published.get("breakdowns") or {}
    metrics = {
        "schema_version": "1.0",
        "run_id": published["run_id"],
        "model": published["model"],
        "benchmark": published["benchmark"],
        "framework": published["framework"],
        "summary": copy.deepcopy(published.get("metrics") or {}),
        "groups": copy.deepcopy(breakdowns.get("groups") or {}),
        "tasks": copy.deepcopy(breakdowns.get("tasks") or {}),
    }
    atomic_json(root / "metrics.json", metrics)
    atomic_json(root / "result.json", published)
    return published
