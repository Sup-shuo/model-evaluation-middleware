from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import threading
from model_evaluation.core.errors import ConfigError
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.config.parsing import load_json_strict, load_yaml_strict
from model_evaluation.core.security import looks_secret_name

_NON_SECRET_SUFFIXES=("_ref","_mode","_type","_name","_id","_url","_path","_root")

def _is_secret_ref(value: object) -> bool:
    return isinstance(value,str) and value.startswith("secret://")

def reject_inline_secrets(value: Any, path: str = "<root>") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_l=str(key).lower(); child=f"{path}.{key}"
            if looks_secret_name(key_l) and not isinstance(item,(dict,list)):
                if not key_l.endswith(_NON_SECRET_SUFFIXES) and not _is_secret_ref(item):
                    raise ConfigError(f"raw secret-like field is forbidden in persisted spec at {child}; use secret://...")
            if looks_secret_name(key_l) and isinstance(item,list) and not all(_is_secret_ref(x) for x in item):
                raise ConfigError(f"raw secret-like list is forbidden in persisted spec at {child}; use secret://...")
            reject_inline_secrets(item,child)
    elif isinstance(value, list):
        for idx,item in enumerate(value): reject_inline_secrets(item,f"{path}[{idx}]")

# Backward-compatible internal alias.
_reject_inline_secrets=reject_inline_secrets


def _apply_spec_defaults(kind: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Protocol v1.1 intentionally applies no implementation defaults.

    Environment providers and device identities are Adapter-owned choices and must
    therefore be explicit in persisted specs (or omitted when the role is unused).
    """
    return obj

SPEC_DIRS = {
    "model": ("models", "model_spec"),
    "platform": ("platforms", "platform_profile"),
    "deployment": ("deployments", "deployment_profile"),
    "benchmark": ("benchmarks", "benchmark_spec"),
    "evaluation": ("evaluations", "evaluation_profile"),
    "run": ("runs", "run_spec"),
}

class SpecRepository:
    def __init__(self, root: str | Path, schemas: SchemaStore):
        self.root = Path(root).resolve()
        self.schemas = schemas
        self._overlays: dict[tuple[str, str], dict[str, Any]] = {}
        self._overlay_lock = threading.RLock()

    def register(self, kind: str, obj: dict[str, Any], *, replace: bool = True) -> dict[str, Any]:
        """Register an in-memory generated spec without touching the shipped spec tree.

        User-facing configuration is translated into these overlays; Planner and
        Orchestrator continue to consume the same internal Spec contracts.
        """
        value = self._validated_overlay(kind, obj)
        spec_id = value["id"]
        key=(kind,spec_id)
        with self._overlay_lock:
            if key in self._overlays and not replace:
                raise ConfigError(f"generated {kind} spec already registered: {spec_id}")
            overlays = dict(self._overlays)
            overlays[key] = copy.deepcopy(value)
            self._overlays = overlays
        return copy.deepcopy(value)

    def clear_overlays(self) -> None:
        self.replace_overlays({})

    def overlay_snapshot(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Return a detached point-in-time copy of all generated specs."""
        with self._overlay_lock:
            return copy.deepcopy(self._overlays)

    def replace_overlays(self, overlays: dict[tuple[str, str], dict[str, Any]]) -> None:
        """Validate and publish a complete overlay set in one atomic swap.

        Callers may prepare a user configuration in a private repository and
        publish it only after every generated spec has passed validation.  A
        concurrent resolver consequently observes either the old complete set
        or the new complete set, never an incrementally populated mixture.
        """
        if not isinstance(overlays, dict):
            raise ConfigError("generated spec overlays must be a mapping")
        normalized: dict[tuple[str, str], dict[str, Any]] = {}
        for raw_key, raw_value in overlays.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ConfigError(f"invalid generated spec overlay key: {raw_key!r}")
            kind, spec_id = raw_key
            registered = self._validated_overlay(str(kind), raw_value)
            if registered["id"] != spec_id:
                raise ConfigError(
                    f"generated {kind} overlay key {spec_id!r} disagrees with id {registered['id']!r}"
                )
            key = (str(kind), str(spec_id))
            if key in normalized:
                raise ConfigError(f"duplicate generated spec overlay: {kind}/{spec_id}")
            normalized[key] = registered
        with self._overlay_lock:
            self._overlays = normalized

    def fork(self, *, include_overlays: bool = True) -> "SpecRepository":
        """Create an independent repository sharing only read-only roots/schemas."""
        repository = SpecRepository(self.root, self.schemas)
        if include_overlays:
            repository.replace_overlays(self.overlay_snapshot())
        return repository

    def _validated_overlay(self, kind: str, obj: dict[str, Any]) -> dict[str, Any]:
        if kind not in SPEC_DIRS:
            raise ConfigError(f"unknown spec kind: {kind}")
        if kind == "run":
            raise ConfigError("run specs are not registered as overlays")
        if not isinstance(obj, dict):
            raise ConfigError(f"generated {kind} spec must be an object")
        value = _apply_spec_defaults(kind, copy.deepcopy(obj))
        self.schemas.validate(SPEC_DIRS[kind][1], value)
        spec_id = value.get("id")
        if not isinstance(spec_id, str) or not spec_id:
            raise ConfigError(f"generated {kind} spec requires a non-empty id")
        return value

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                obj = load_json_strict(text)
            elif path.suffix.lower() in {".yaml", ".yml"}:
                obj = load_yaml_strict(text)
            else:
                raise ConfigError(f"unsupported spec format: {path}")
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"failed to parse spec {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ConfigError(f"spec must be an object: {path}")
        _reject_inline_secrets(obj,str(path))
        return obj

    def load_path(self, kind: str, path: str | Path) -> dict:
        if kind not in SPEC_DIRS:
            raise ConfigError(f"unknown spec kind: {kind}")
        p = Path(path).resolve()
        obj = _apply_spec_defaults(kind, self._read(p))
        self.schemas.validate(SPEC_DIRS[kind][1], obj)
        return obj

    def resolve(self, kind: str, spec_id: str) -> dict:
        if kind not in SPEC_DIRS:
            raise ConfigError(f"unknown spec kind: {kind}")
        dirname, schema = SPEC_DIRS[kind]
        with self._overlay_lock:
            overlay=copy.deepcopy(self._overlays.get((kind,spec_id)))
        if overlay is not None:
            return overlay
        base = (self.root / dirname).resolve()
        if not spec_id or any(part in {"", ".", ".."} for part in Path(spec_id).parts) or Path(spec_id).is_absolute():
            raise ConfigError(f"invalid/path-escaping {kind} spec id: {spec_id!r}")
        matches: list[Path] = []
        for ext in (".yaml", ".yml", ".json"):
            p = (base / f"{spec_id}{ext}").resolve()
            if base not in p.parents:
                raise ConfigError(f"{kind} spec id escapes repository: {spec_id!r}")
            if p.is_file():
                matches.append(p)
        if len(matches) != 1:
            raise ConfigError(f"expected exactly one {kind} spec for {spec_id!r}, found {len(matches)}")
        obj = _apply_spec_defaults(kind, self._read(matches[0]))
        self.schemas.validate(schema, obj)
        if obj.get("id") != spec_id and kind != "run":
            raise ConfigError(f"{kind} spec filename/reference {spec_id!r} disagrees with id {obj.get('id')!r}")
        return obj

    def resolve_run(self, run_or_path: str) -> dict:
        p = Path(run_or_path)
        if p.is_file():
            return self.load_path("run", p)
        return self.resolve("run", run_or_path)

    def resolve_bundle(self, run_spec: dict) -> dict[str, Any]:
        # Hold one read transaction across every overlay lookup.  ``resolve``
        # re-enters this RLock, so a concurrent complete-overlay publication can
        # happen before or after the bundle, never between its five components.
        with self._overlay_lock:
            self.schemas.validate("run_spec", run_spec)
            return {
                "run": run_spec,
                "model": self.resolve("model", run_spec["model"]),
                "platform": self.resolve("platform", run_spec["platform"]),
                "deployment": self.resolve("deployment", run_spec["deployment"]),
                "benchmark": self.resolve("benchmark", run_spec["benchmark"]),
                "evaluation": self.resolve("evaluation", run_spec["evaluation"]),
            }
