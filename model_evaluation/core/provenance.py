from __future__ import annotations

from typing import Any

from model_evaluation.core.errors import ConfigError


_MOVING_REVISIONS = {
    "main",
    "master",
    "latest",
    "head",
    "default",
    "trunk",
    "dev",
    "develop",
}
_PLACEHOLDER_MARKERS = (
    "replace-with",
    "placeholder",
    "unversioned",
    "unknown",
    "unpinned",
)


def _config(model: dict[str, Any]) -> dict[str, Any]:
    value = model.get("provenance") or {}
    if not isinstance(value, dict):
        raise ConfigError("ModelSpec.provenance must be an object")
    return value


def _stable_revision(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    revision = value.strip()
    lowered = revision.lower()
    return lowered not in _MOVING_REVISIONS and not any(
        marker in lowered for marker in _PLACEHOLDER_MARKERS
    )


def assess_model_provenance(
    model: dict[str, Any], deployment: dict[str, Any]
) -> dict[str, Any]:
    """Describe the configured model source for later reproduction.

    ``policy`` is retained as a compatibility/display hint for existing model
    catalogs.  It never makes Core hash a model directory or certify its bytes.
    """
    cfg = _config(model)
    policy = str(cfg.get("policy") or "migration").lower()
    if policy not in {"migration", "pinned"}:
        raise ConfigError("model provenance policy must be one of: migration, pinned")

    source = model.get("source") or {}
    source_type = str(source.get("type") or "other")
    source_ref = str(source.get("ref") or "")
    revision = source.get("revision")
    revision_stable = _stable_revision(revision)
    management = str((deployment.get("management") or {}).get("mode") or "managed")
    local_path = (deployment.get("model_location") or {}).get("local_path")

    return {
        "policy": policy,
        "source_type": source_type,
        "source_ref": source_ref,
        "source_revision": str(revision) if revision is not None else None,
        "revision_stable": revision_stable,
        "management_mode": management,
        "local_materialization": bool(local_path),
        "recording": (
            "remote_endpoint"
            if management in {"external", "attached"}
            else "local_path"
            if local_path
            else "declared_revision"
            if revision_stable
            else "declared_source"
        ),
    }
