from __future__ import annotations

import os
from copy import deepcopy
from collections.abc import Callable, Iterable

from model_evaluation.core.errors import ConfigError

EMPTY_PATCH = {"set": {}, "unset": [], "prepend_path": {}, "append_path": {}}

class EnvPatchMerger:
    """Merge environment patches with explicit scalar ownership.

    Scalar set/unset mutations are exclusive. PATH-like prepend/append
    mutations are intentionally composable across layers and preserve the
    Orchestrator's deterministic add order. A variable cannot mix scalar and
    path-list ownership.
    """
    def __init__(self):
        self._patch = deepcopy(EMPTY_PATCH)
        self._scalar_owners: dict[str, tuple[str, object]] = {}
        self._path_owners: dict[str, list[tuple[str, str]]] = {}

    def add(self, owner: str, patch: dict | None) -> None:
        patch = patch or {}
        for key, value in (patch.get("set") or {}).items():
            self._claim_scalar(owner, key, ("set", str(value)))
            self._patch["set"][key] = str(value)
        for key in patch.get("unset") or []:
            self._claim_scalar(owner, key, ("unset", None))
            if key not in self._patch["unset"]:
                self._patch["unset"].append(key)
        for action in ("prepend_path", "append_path"):
            for key, values in (patch.get(action) or {}).items():
                self._claim_path(owner, key, action)
                self._patch[action].setdefault(key, []).extend(str(v) for v in values)

    def _claim_scalar(self, owner: str, key: str, value: object) -> None:
        if key in self._path_owners:
            raise ConfigError(f"environment ownership conflict for {key}: path mutation vs scalar owner {owner}")
        prev = self._scalar_owners.get(key)
        if prev and prev != (owner, value):
            raise ConfigError(f"environment ownership conflict for {key}: {prev[0]} vs {owner}")
        self._scalar_owners[key] = (owner, value)

    def _claim_path(self, owner: str, key: str, action: str) -> None:
        if key in self._scalar_owners:
            raise ConfigError(f"environment ownership conflict for {key}: scalar mutation vs path owner {owner}")
        self._path_owners.setdefault(key, []).append((owner, action))

    def result(self) -> dict:
        return deepcopy(self._patch)


def apply_env_patch(base: dict[str, str] | None, patch: dict | None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    patch = patch or {}
    for key in patch.get("unset") or []:
        env.pop(key, None)
    for key, value in (patch.get("set") or {}).items():
        env[key] = str(value)
    for key, values in (patch.get("prepend_path") or {}).items():
        existing = env.get(key, "")
        parts = [str(v) for v in values]
        if existing: parts.append(existing)
        env[key] = os.pathsep.join(parts)
    for key, values in (patch.get("append_path") or {}).items():
        existing = env.get(key, "")
        parts = ([existing] if existing else []) + [str(v) for v in values]
        env[key] = os.pathsep.join(parts)
    return env


def derive_env_patch_additions(before: dict | None, after: dict | None) -> dict:
    """Return only mutations added by a wrapper, rejecting rewrites of existing owners."""
    before = before or {}; after = after or {}; out = deepcopy(EMPTY_PATCH)
    bset, aset = before.get("set") or {}, after.get("set") or {}
    for key, value in bset.items():
        if aset.get(key) != value: raise ConfigError(f"environment provider rewrote already-owned set variable {key}")
    out["set"] = {k: v for k, v in aset.items() if k not in bset}
    bunset, aunset = set(before.get("unset") or []), set(after.get("unset") or [])
    if not bunset.issubset(aunset): raise ConfigError("environment provider removed an already-owned unset mutation")
    out["unset"] = sorted(aunset - bunset)
    for action in ("prepend_path", "append_path"):
        bmap, amap = before.get(action) or {}, after.get(action) or {}
        for key, old_values in bmap.items():
            new_values = amap.get(key)
            if new_values is None: raise ConfigError(f"environment provider removed already-owned {action} mutation for {key}")
            old_values, new_values = list(old_values), list(new_values)
            if action == "prepend_path":
                if old_values and new_values[-len(old_values):] != old_values: raise ConfigError(f"environment provider rewrote already-owned prepend_path for {key}")
                additions = new_values[:-len(old_values)] if old_values else new_values
            else:
                if new_values[:len(old_values)] != old_values: raise ConfigError(f"environment provider rewrote already-owned append_path for {key}")
                additions = new_values[len(old_values):]
            if additions: out[action][key] = additions
        for key, vals in amap.items():
            if key not in bmap: out[action][key] = list(vals)
    return out


def prepare_process_for_environment(
    process: dict,
    *,
    base_patches: Iterable[tuple[str, dict | None]] = (),
    process_owner: str,
    wrap: Callable[[dict], dict],
) -> dict:
    """Compose Core-owned patches, then let the selected Environment wrap once.

    The wrapper is allowed to *add* environment mutations (for example placing
    ``venv/bin`` ahead of an existing runtime PATH) but may not rewrite or
    remove mutations already owned by preceding layers.  Crucially, the
    wrapper's resulting ProcessSpec is returned as-is after that validation;
    Core must not decompose and re-merge it, because doing so can invert PATH
    precedence and make ``doctor`` disagree with ``run``.
    """
    prepared = deepcopy(process)
    merger = EnvPatchMerger()
    for owner, patch in base_patches:
        merger.add(owner, patch)
    merger.add(process_owner, prepared.get("env_patch"))
    prepared["env_patch"] = merger.result()
    before = deepcopy(prepared.get("env_patch") or {})
    wrapped = deepcopy(wrap(prepared))
    if not isinstance(wrapped, dict):
        raise ConfigError("environment wrapper returned a non-object ProcessSpec")
    # Validation-only: do not reconstruct the patch after this point.
    derive_env_patch_additions(before, wrapped.get("env_patch"))
    return wrapped
