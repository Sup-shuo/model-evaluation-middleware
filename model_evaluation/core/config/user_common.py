from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import Any

from model_evaluation.core.config.adapter_config import (
    adapter_user_config as _adapter_user_config,
    validate_adapter_user_parameters as _validate_adapter_user_parameters,
)
from model_evaluation.core.config.documents import load_yaml_document
from model_evaluation.core.errors import ConfigError

_SAFE = re.compile(r"[^A-Za-z0-9._@+-]+")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return load_yaml_document(path, reject_secrets=True)


def _slug(text: str, prefix: str) -> str:
    base = _SAFE.sub("-", text.strip()).strip("-._") or prefix
    base = base[:48]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{base}-{digest}"


def _absolute(value: str, label: str) -> str:
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        raise ConfigError(f"{label} 必须是绝对路径: {value}")
    return str(p.resolve())


def _project_path(value: object, label: str, project_root: str | Path) -> str:
    """Resolve a generated-output path, keeping relative values in the project.

    Absolute paths remain an explicit machine-level override.  Relative paths
    are confined below the project root so ``results`` has a portable default
    without allowing a seemingly local value to escape through ``..``.
    """
    base = Path(project_root).resolve()
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return str(raw.resolve())
    resolved = (base / raw).resolve()
    if resolved != base and base not in resolved.parents:
        raise ConfigError(f"{label} 相对路径不能越过项目根目录: {value}")
    return str(resolved)


def _selected_profile(system: dict[str, Any], evaluation: dict[str, Any], kind: str) -> tuple[str, dict[str, Any]]:
    profiles = system["profiles"]
    table = profiles.get(kind) or {}
    if not table:
        raise ConfigError(f"system.profiles.{kind} 未登记任何 profile")
    if kind in {"backend", "evaluator"}:
        selection = evaluation.get(kind) or {}
        selected_raw = selection.get("profile") if isinstance(selection, dict) else None
        label = f"evaluation.{kind}.profile"
    else:
        selected_raw = (evaluation.get("profiles") or {}).get(kind)
        selected_raw = selected_raw or (profiles.get("defaults") or {}).get(kind)
        label = f"evaluation.profiles.{kind}"
    if selected_raw is None:
        if kind == "hardware" and len(table) == 1:
            selected_raw = next(iter(table))
        else:
            available = ", ".join(sorted(table))
            raise ConfigError(
                f"{label} 必须显式选择已登记的 profile；可选: {available}"
            )
    selected = str(selected_raw).strip()
    if selected not in table:
        available = ", ".join(sorted(table))
        raise ConfigError(f"{label}={selected!r} 不存在；可选: {available}")
    return selected, copy.deepcopy(table[selected])


def _merge_ergonomic_parameter(params: dict[str, Any], key: str, value: object, *, label: str, absolute: bool = False) -> None:
    if value is None:
        return
    normalized = _absolute(str(value), label) if absolute else copy.deepcopy(value)
    if key in params and params[key] != normalized:
        raise ConfigError(f"{label} 与 parameters.{key} 同时配置且值不一致")
    params[key] = normalized


def _environment(app, value: object, *, system: dict[str, Any], label: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Normalize an EnvironmentProvider or a named system environment profile.

    A string first resolves against ``system.profiles.environment``.  If no
    named profile exists, the string keeps the alpha20 shorthand semantics and
    is treated as an EnvironmentProvider name.  This keeps simple ``current``
    configs working while allowing reusable isolated environments such as
    ``backend-py312`` or ``evaluator-py311``.
    """
    env_profiles = ((system.get("profiles") or {}).get("environment") or {})
    resolved_value = copy.deepcopy(value)
    if isinstance(value, str) and value in env_profiles:
        resolved_value = copy.deepcopy(env_profiles[value])
        label = f"{label} -> system.profiles.environment.{value}"

    explicit_profile = False
    if resolved_value is None:
        raise ConfigError(f"{label} 必须显式选择 EnvironmentProvider 或命名 Environment profile")
    elif isinstance(resolved_value, str):
        provider = resolved_value.strip()
        if not provider:
            raise ConfigError(f"{label} 不能为空")
        profile = provider
        explicit_profile = False
        params = {}
    elif isinstance(resolved_value, dict):
        provider = str(resolved_value.get("type") or "").strip()
        if not provider:
            raise ConfigError(f"{label}.type 不能为空")
        if resolved_value.get("profile") is not None and resolved_value.get("name") is not None:
            raise ConfigError(f"{label} 不能同时配置 profile 和 name；请只保留 profile")
        raw_profile = resolved_value.get("profile") if resolved_value.get("profile") is not None else resolved_value.get("name")
        explicit_profile = raw_profile is not None
        profile = str(raw_profile).strip() if explicit_profile else provider
        if not profile:
            raise ConfigError(f"{label}.profile/name 不能为空")
        params = copy.deepcopy(resolved_value.get("parameters") or {})
        if not isinstance(params, dict):
            raise ConfigError(f"{label}.parameters 必须是 object")
        _merge_ergonomic_parameter(params, "executable", resolved_value.get("executable"), label=f"{label}.executable", absolute=True)
    else:
        raise ConfigError(f"{label} 必须是命名 Environment profile、EnvironmentProvider 名称或 object")

    client = _validate_adapter_user_parameters(app, "environment", provider, params, f"{label}.parameters")
    user_cfg = _adapter_user_config(client)
    if bool(user_cfg.get("profile_required")) and not explicit_profile:
        raise ConfigError(f"{label} 选择的 environment adapter {provider!r} 要求显式 profile/name")
    return {"provider": provider, "profile": profile}, params


def _environment_selection(app, value: object, *, system: dict[str, Any], label: str) -> dict[str, Any]:
    env, params = _environment(app, value, system=system, label=label)
    out: dict[str, Any] = copy.deepcopy(env)
    if params:
        out["parameters"] = params
    return out


def _backend_mode(backend: dict[str, Any], client) -> str:
    explicit = backend.get("mode")
    if explicit:
        return str(explicit)
    value = _adapter_user_config(client).get("default_management_mode")
    if not value:
        raise ConfigError(
            f"backend adapter {client.identity.name!r} 未声明默认 management mode；"
            "请在 system.yaml backend profile 中显式配置 mode"
        )
    return str(value)


def _runtime_families(backend: dict[str, Any], *, mode: str, backend_profile_id: str) -> list[str] | None:
    """Resolve only explicit Core-owned runtime compatibility; never infer from inventory."""
    if mode != "managed":
        return None
    compatibility = backend.get("compatibility") or {}
    if not isinstance(compatibility, dict):
        raise ConfigError(f"backend {backend_profile_id!r} compatibility 必须是 object")
    families = compatibility.get("runtime_families")
    if not families:
        raise ConfigError(
            f"managed backend {backend_profile_id!r} 必须显式声明 compatibility.runtime_families；"
            "Runtime compatibility 不再从 system inventory 或 selected runtime 推断"
        )
    return [str(x) for x in families]


def _validate_runtime_compatibility(runtime_type: str, families: list[str] | None, *, backend_profile_id: str) -> None:
    if families is not None and runtime_type not in families:
        raise ConfigError(
            f"Profile compatibility 失败：backend {backend_profile_id!r} 允许 compatibility.runtime_families={families}，"
            f"但所选 hardware runtime 是 {runtime_type!r}"
        )

def _backend_default_parameters(client) -> dict[str, Any]:
    value = _adapter_user_config(client).get("default_parameters") or {}
    if not isinstance(value, dict):
        raise ConfigError(f"backend adapter {client.identity.name!r} user_config.default_parameters 必须是 object")
    return copy.deepcopy(value)
