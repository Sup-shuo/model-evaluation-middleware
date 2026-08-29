from __future__ import annotations

import copy
from typing import Any

from model_evaluation.core.config.adapter_config import (
    adapter_user_config as _adapter_user_config,
)
from model_evaluation.core.config.loader import SpecRepository
from model_evaluation.core.config.merge import deep_merge as _deep_merge
from model_evaluation.core.config.user_common import (
    _absolute,
    _backend_default_parameters,
)
from model_evaluation.core.errors import ConfigError


def compile_deployment(
    deployment_id: str,
    backend: dict[str, Any],
    system: dict[str, Any],
    backend_client,
    *,
    mode: str,
    runtime_families: list[str] | None,
) -> dict[str, Any]:
    backend_type = str(backend["type"])
    params = _deep_merge(_backend_default_parameters(backend_client), copy.deepcopy(backend.get("parameters") or {}))
    if backend.get("executable"):
        # Backend executables are workload commands.  Keep an unqualified
        # command unqualified so the selected EnvironmentProvider can
        # resolve it inside Conda/venv/current rather than freezing it to
        # the controller process environment.  Explicit absolute paths
        # remain explicit and isolated environment adapters validate that
        # they belong to the selected environment.
        params["executable"] = str(backend["executable"]).strip()

    deployment: dict[str, Any] = {
        "schema_version": "1.1",
        "id": deployment_id,
        "backend": {"adapter": backend_type},
        "management": {"mode": mode},
    }
    if runtime_families is not None:
        deployment["compatibility"] = {"runtime_families": list(runtime_families)}

    location_cfg = _adapter_user_config(backend_client).get("model_location") or {}
    if location_cfg:
        if not isinstance(location_cfg, dict):
            raise ConfigError(f"backend adapter {backend_type!r} user_config.model_location 必须是 object")
        modes = [str(x) for x in (location_cfg.get("modes") or [])]
        if mode in modes:
            location: dict[str, Any] = {}
            root_value = backend.get("model_root") or (system.get("models") or {}).get("root")
            if root_value is not None:
                location["root"] = _absolute(str(root_value), "models.root/backend.model_root")
            elif bool(location_cfg.get("requires_root")):
                raise ConfigError(
                    f"backend profile {backend_type!r} 的 {mode} 模式需要本地模型根目录；"
                    "请配置 system.models.root 或 backend.model_root"
                )
            path_template = location_cfg.get("path_template")
            if path_template:
                location["path_template"] = str(path_template)
            if location:
                deployment["model_location"] = location

    if params:
        deployment["parameters"] = params
    if backend.get("endpoint"):
        deployment["endpoint"] = copy.deepcopy(backend["endpoint"])
    if mode in {"external", "attached"} and not deployment.get("endpoint"):
        raise ConfigError("external/attached backend profile 必须配置 endpoint")
    return deployment

def compile_evaluation(
    evaluation_id: str,
    evaluator: dict[str, Any],
    evaluation: dict[str, Any],
    evaluator_client,
    *,
    specs: SpecRepository,
    smoke: bool = False,
) -> dict[str, Any]:
    evaluator_type = str(evaluator["type"])
    preset = str(evaluator.get("preset") or f"{evaluator_type}_current")
    try:
        base = specs.resolve("evaluation", preset)
    except Exception as exc:
        raise ConfigError(
            f"找不到评测框架 preset {preset!r}; 新评测框架请先提供内部 EvaluationProfile/Binding"
        ) from exc
    preset_adapter = str((base.get("framework") or {}).get("adapter") or "")
    if preset_adapter != evaluator_type:
        raise ConfigError(
            f"evaluator profile type/preset 不一致：type={evaluator_type!r}，"
            f"preset {preset!r} 实际 framework.adapter={preset_adapter!r}"
        )

    out = copy.deepcopy(base)
    out["id"] = evaluation_id
    params = copy.deepcopy(out.get("parameters") or {})
    user_cfg = _adapter_user_config(evaluator_client)
    root_parameter = user_cfg.get("root_parameter")
    if root_parameter:
        root_value = evaluator.get("root")
        if root_value is None and bool(user_cfg.get("root_required")):
            raise ConfigError(
                f"evaluator adapter {evaluator_type!r} 要求 system evaluator profile 配置 root"
            )
        if root_value is not None:
            params[str(root_parameter)] = _absolute(
                str(root_value), f"profiles.evaluator.{evaluator_type}.root"
            )
    if evaluator.get("parameters"):
        params = _deep_merge(params, evaluator["parameters"])
    evaluator_parameters = (evaluation.get("evaluator") or {}).get("parameters") or {}
    if evaluator_parameters:
        params = _deep_merge(params, evaluator_parameters)
    if params:
        out["parameters"] = params
    if smoke:
        out["execution"] = {"mode": "smoke", "sample_limit": 1}
    return out
