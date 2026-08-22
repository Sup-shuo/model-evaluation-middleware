from __future__ import annotations

import copy
import stat
from pathlib import Path
from typing import Any, Iterable

import yaml

from model_evaluation.core.config.catalog import ConfigEntry
from model_evaluation.core.errors import ConfigError
from model_evaluation.core.files import atomic_text


def _profile(value: object, fallback: str | None, *, label: str) -> str:
    selected = str(value or fallback or "").strip()
    if not selected:
        raise ConfigError(
            f"{label} cannot be inferred safely; pass the corresponding explicit migration option"
        )
    return selected


def migrate_document(
    kind: str,
    document: dict[str, Any],
    *,
    backend_profile: str | None = None,
    evaluator_profile: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Migrate one public user document without guessing framework profiles."""

    version = str(document.get("schema_version") or "")
    if version == "1.3":
        return copy.deepcopy(document), False
    if version != "1.2":
        raise ConfigError(f"unsupported {kind} schema migration: {version!r}")
    if kind not in {"system", "evaluation"}:
        raise ConfigError(f"no migration is defined for configuration kind {kind!r}")

    migrated = copy.deepcopy(document)
    migrated["schema_version"] = "1.3"
    profiles = migrated.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ConfigError(f"{kind}.profiles must be an object")

    if kind == "system":
        defaults = profiles.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise ConfigError("system.profiles.defaults must be an object")
        defaults.pop("backend", None)
        defaults.pop("evaluator", None)
        if defaults:
            profiles["defaults"] = defaults
        else:
            profiles.pop("defaults", None)
        migrated["profiles"] = profiles
        return migrated, True

    legacy_profiles = copy.deepcopy(profiles)
    selected_backend = _profile(
        legacy_profiles.pop("backend", None), backend_profile,
        label="evaluation backend profile",
    )
    selected_evaluator = _profile(
        legacy_profiles.pop("evaluator", None), evaluator_profile,
        label="evaluation evaluator profile",
    )
    if legacy_profiles:
        migrated["profiles"] = legacy_profiles
    else:
        migrated.pop("profiles", None)

    backend_parameters = migrated.get("backend") or {}
    evaluator_parameters = migrated.get("evaluator") or {}
    if not isinstance(backend_parameters, dict) or not isinstance(evaluator_parameters, dict):
        raise ConfigError("evaluation backend/evaluator parameters must be objects")
    migrated["backend"] = {"profile": selected_backend}
    migrated["evaluator"] = {"profile": selected_evaluator}
    if backend_parameters:
        migrated["backend"]["parameters"] = backend_parameters
    if evaluator_parameters:
        migrated["evaluator"]["parameters"] = evaluator_parameters
    return migrated, True


def migrate_entry(
    entry: ConfigEntry,
    *,
    write: bool,
    backend_profile: str | None = None,
    evaluator_profile: str | None = None,
) -> dict[str, Any]:
    return migrate_entries(
        [entry],
        write=write,
        backend_profile=backend_profile,
        evaluator_profile=evaluator_profile,
    )[0]


def migrate_entries(
    entries: Iterable[ConfigEntry],
    *,
    write: bool,
    backend_profile: str | None = None,
    evaluator_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Preflight a catalog and apply writes with in-process rollback.

    Each replacement is atomic. If a later replacement fails, files already
    replaced by this invocation are restored to their exact original text and
    mode before the error is returned.
    """

    prepared: list[tuple[ConfigEntry, dict[str, Any], bool, dict[str, Any]]] = []
    for entry in entries:
        if entry.error or entry.data is None:
            raise ConfigError(
                f"cannot migrate {entry.path}: {entry.error or 'invalid document'}"
            )
        migrated, changed = migrate_document(
            entry.kind,
            entry.data,
            backend_profile=backend_profile,
            evaluator_profile=evaluator_profile,
        )
        row = {
            "kind": entry.kind,
            "reference": entry.reference,
            "path": str(entry.path),
            "from": entry.schema_version,
            "to": str(migrated.get("schema_version")),
            "changed": changed,
            "written": False,
        }
        prepared.append((entry, migrated, changed, row))

    if not write:
        return [row for _, _, _, row in prepared]

    originals: dict[Path, tuple[str, int]] = {}
    written: list[Path] = []
    try:
        for entry, migrated, changed, row in prepared:
            if not changed:
                continue
            path = entry.path
            originals[path] = (
                path.read_text(encoding="utf-8"),
                stat.S_IMODE(path.stat().st_mode),
            )
            text = yaml.safe_dump(migrated, allow_unicode=True, sort_keys=False)
            atomic_text(path, text)
            # The replacement has happened before chmod. Track it immediately
            # so a chmod failure also restores the original file.
            written.append(path)
            path.chmod(originals[path][1])
            row["written"] = True
    except Exception as exc:
        rollback_errors = []
        for path in reversed(written):
            original_text, original_mode = originals[path]
            try:
                atomic_text(path, original_text)
                path.chmod(original_mode)
            except Exception as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        detail = f"; rollback failures: {rollback_errors}" if rollback_errors else ""
        raise ConfigError(f"configuration migration write failed and was rolled back: {exc}{detail}") from exc
    return [row for _, _, _, row in prepared]
