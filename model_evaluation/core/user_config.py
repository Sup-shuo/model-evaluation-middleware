from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_evaluation.core.config.adapter_config import (
    adapter_user_config as _adapter_user_config,
    validate_adapter_user_parameters as _validate_adapter_user_parameters,
)
from model_evaluation.core.config.documents import load_yaml_document
from model_evaluation.core.config.loader import SpecRepository
from model_evaluation.core.config.merge import deep_merge as _deep_merge
from model_evaluation.core.config.matrix_compiler import compile_matrix_spec
from model_evaluation.core.config.model_compiler import compile_models
from model_evaluation.core.config.model_catalog import resolve_model_entries
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


@dataclass(frozen=True)
class UserConfigBundle:
    system: dict[str, Any]
    evaluation: dict[str, Any]
    matrix_spec: dict[str, Any]
    cache_root: str
    results_root: str
    generated: dict[str, Any]
    specs: SpecRepository


class UserConfigResolver:
    """把两份用户 YAML 转换为既有内部 Specs。

    system.yaml 只登记这台机器已经部署好的 profile；evaluation.yaml 选择本次
    使用的 hardware/backend/evaluator profile，并提供运行参数。具体 Adapter 的
    默认值、参数空间和少量用户层映射策略由 Adapter manifest 自己声明。
    """

    def __init__(self, app):
        self.app = app

    def load(
        self,
        system_path: str | Path,
        evaluation_path: str | Path,
        *,
        smoke: bool = False,
    ) -> UserConfigBundle:
        system_file = Path(system_path).expanduser().resolve()
        evaluation_file = Path(evaluation_path).expanduser().resolve()
        system = _load_yaml(system_file)
        evaluation = _load_yaml(evaluation_file)
        self.app.matrix_schemas.validate("user_system", system)
        self.app.matrix_schemas.validate("user_evaluation", evaluation)
        model_entries, model_catalog_root, model_catalog_enabled = resolve_model_entries(
            self.app, evaluation, evaluation_file
        )
        # Resolve into a private repository.  The shared Application repository
        # remains untouched until the entire user configuration has succeeded,
        # so a late validation error cannot publish a half-populated catalog.
        specs = self.app.specs.fork(include_overlays=False)

        system_name = str(system["system"]["name"])
        backend_profile_id, backend = _selected_profile(system, evaluation, "backend")
        evaluator_profile_id, evaluator = _selected_profile(system, evaluation, "evaluator")
        backend_type = str(backend["type"])
        evaluator_type = str(evaluator["type"])

        backend_profile_parameters = copy.deepcopy(backend.get("parameters") or {})
        if backend.get("executable") is not None:
            executable = str(backend.get("executable") or "").strip()
            if not executable:
                raise ConfigError(f"system.profiles.backend.{backend_profile_id}.executable 不能为空")
            if "executable" in backend_profile_parameters and backend_profile_parameters["executable"] != executable:
                raise ConfigError(
                    f"system.profiles.backend.{backend_profile_id}.executable 与 parameters.executable 同时配置且值不一致"
                )
            backend_profile_parameters["executable"] = executable
        backend_client = _validate_adapter_user_parameters(
            self.app, "backend", backend_type, backend_profile_parameters,
            f"system.profiles.backend.{backend_profile_id}.parameters",
        )
        backend_run_parameters = copy.deepcopy((evaluation.get("backend") or {}).get("parameters") or {})
        _validate_adapter_user_parameters(
            self.app, "backend", backend_type, backend_run_parameters, "evaluation.backend.parameters",
        )
        evaluator_client = _validate_adapter_user_parameters(
            self.app, "evaluator", evaluator_type, evaluator.get("parameters"),
            f"system.profiles.evaluator.{evaluator_profile_id}.parameters",
        )
        evaluator_run_parameters = copy.deepcopy((evaluation.get("evaluator") or {}).get("parameters") or {})
        _validate_adapter_user_parameters(
            self.app, "evaluator", evaluator_type, evaluator_run_parameters, "evaluation.evaluator.parameters",
        )

        mode = _backend_mode(backend, backend_client)
        hardware_profile_id: str | None = None
        hardware: dict[str, Any] | None = None
        runtime: dict[str, Any] | None = None
        device_type: str | None = None
        runtime_type: str | None = None
        device_params: dict[str, Any] = {}
        runtime_params: dict[str, Any] = {}
        runtime_families = _runtime_families(backend, mode=mode, backend_profile_id=backend_profile_id)

        resources = evaluation.get("resources") or {}
        devices: list[str] | None = None
        if mode == "managed":
            hardware_profile_id, hardware = _selected_profile(system, evaluation, "hardware")
            runtime = hardware["runtime"]
            device_type = str(hardware["type"])
            runtime_type = str(runtime["type"])

            device_params = copy.deepcopy(hardware.get("parameters") or {})
            if not isinstance(device_params, dict):
                raise ConfigError(f"system.profiles.hardware.{hardware_profile_id}.parameters 必须是 object")
            _validate_adapter_user_parameters(
                self.app, "device", device_type, device_params,
                f"system.profiles.hardware.{hardware_profile_id}.parameters",
            )

            runtime_params = copy.deepcopy(runtime.get("parameters") or {})
            if not isinstance(runtime_params, dict):
                raise ConfigError(f"system.profiles.hardware.{hardware_profile_id}.runtime.parameters 必须是 object")
            _merge_ergonomic_parameter(
                runtime_params, "root", runtime.get("root"),
                label=f"system.profiles.hardware.{hardware_profile_id}.runtime.root", absolute=True,
            )
            _validate_adapter_user_parameters(
                self.app, "runtime", runtime_type, runtime_params,
                f"system.profiles.hardware.{hardware_profile_id}.runtime.parameters",
            )
            _validate_runtime_compatibility(runtime_type, runtime_families, backend_profile_id=backend_profile_id)

            # Device identity is normally machine-owned.  An evaluation may still
            # override it for a one-off run, but otherwise the selected Hardware
            # Profile supplies the stable per-machine selection.
            selected_devices = resources.get("devices")
            if selected_devices is None:
                selected_devices = hardware.get("devices")
            if selected_devices is not None:
                devices = [str(x) for x in selected_devices or []]
        else:
            if (evaluation.get("profiles") or {}).get("hardware") is not None:
                raise ConfigError("external/attached backend 不使用本地 Hardware profile；请删除 evaluation.profiles.hardware")
            if resources.get("devices") is not None:
                raise ConfigError("external/attached backend 不使用本地 resources.devices")
            if backend.get("environment") is not None:
                raise ConfigError("external/attached backend 不启动本地 Backend，因此不能配置 backend.environment")

        environment_overrides = evaluation.get("environments") or {}
        evaluator_env_value = environment_overrides.get("evaluator", evaluator.get("environment"))
        eval_env, eval_env_params = _environment(
            self.app, evaluator_env_value, system=system,
            label="evaluation.environments.evaluator" if "evaluator" in environment_overrides else f"system.profiles.evaluator.{evaluator_profile_id}.environment",
        )

        backend_env: dict[str, str] | None = None
        backend_env_params: dict[str, Any] = {}
        if mode == "managed":
            backend_env_value = environment_overrides.get("backend", backend.get("environment"))
            backend_env, backend_env_params = _environment(
                self.app, backend_env_value, system=system,
                label="evaluation.environments.backend" if "backend" in environment_overrides else f"system.profiles.backend.{backend_profile_id}.environment",
            )
        elif "backend" in environment_overrides:
            raise ConfigError("external/attached backend 不启动本地 Backend，因此不能配置 evaluation.environments.backend")

        platform_key = hardware_profile_id if hardware_profile_id is not None else "evaluation-only"
        platform_id = _slug(f"{system_name}:{platform_key}", "user-platform")
        deployment_id = _slug(f"{backend_profile_id}:{backend_type}", "user-backend")
        evaluation_id = _slug(f"{evaluator_profile_id}:{evaluator_type}", "user-evaluator")

        cache_root = _project_path(
            system["paths"]["cache"],
            "paths.cache",
            self.app.project_root,
        )
        results_root = _project_path(
            system["paths"].get("results", "results"),
            "paths.results",
            self.app.project_root,
        )

        metadata: dict[str, Any] = {"user_profiles": {}}
        if hardware_profile_id is not None:
            metadata["user_profiles"]["hardware"] = hardware_profile_id
        if system.get("metadata"):
            metadata = _deep_merge(metadata, system["metadata"])

        eval_env_sel: dict[str, Any] = copy.deepcopy(eval_env)
        if eval_env_params:
            eval_env_sel["parameters"] = eval_env_params
        platform: dict[str, Any] = {
            "schema_version": "1.1",
            "id": platform_id,
            "evaluation_environment": eval_env_sel,
            "metadata": metadata,
        }
        if mode == "managed":
            assert hardware is not None and runtime is not None and device_type is not None and runtime_type is not None and backend_env is not None
            device_sel: dict[str, Any] = {"adapter": device_type}
            if devices is not None:
                device_sel["devices"] = devices
            if device_params:
                device_sel["parameters"] = device_params
            runtime_sel: dict[str, Any] = {"adapter": runtime_type}
            if runtime.get("profile"):
                runtime_sel["profile"] = str(runtime["profile"])
            if runtime_params:
                runtime_sel["parameters"] = runtime_params
            backend_env_sel: dict[str, Any] = copy.deepcopy(backend_env)
            if backend_env_params:
                backend_env_sel["parameters"] = backend_env_params
            platform.update({
                "device": device_sel,
                "runtime": runtime_sel,
                "backend_environment": backend_env_sel,
            })
        specs.register("platform", platform)

        deployment = self._deployment(
            deployment_id, backend, system, backend_client,
            mode=mode, runtime_families=runtime_families,
        )
        specs.register("deployment", deployment)

        evaluation_spec = self._evaluation(
            evaluation_id,
            evaluator,
            evaluation,
            evaluator_client,
            specs=specs,
            smoke=smoke,
        )
        specs.register("evaluation", evaluation_spec)

        compiled_models = compile_models(
            app=self.app,
            specs=specs,
            model_entries=model_entries,
            deployment=deployment,
            backend_type=backend_type,
            backend_client=backend_client,
            mode=mode,
            available_devices=devices,
            backend_profile_parameters=backend_profile_parameters,
            backend_run_parameters=backend_run_parameters,
            slug=_slug,
            environment_selection=lambda value, label: _environment_selection(
                self.app,
                value,
                system=system,
                label=label,
            ),
        )

        benchmark_ids = [str(x) for x in evaluation["benchmarks"]]
        for benchmark_id in benchmark_ids:
            benchmark = specs.resolve("benchmark", benchmark_id)
            dataset = benchmark.get("dataset") or {}
            _validate_adapter_user_parameters(
                self.app,
                "dataset",
                str(dataset.get("provider") or ""),
                dataset.get("parameters") or {},
                f"benchmark[{benchmark_id}].dataset.parameters",
            )

        matrix = compile_matrix_spec(
            system_name=system_name,
            platform_key=platform_key,
            backend_profile_id=backend_profile_id,
            evaluator_profile_id=evaluator_profile_id,
            platform_id=platform_id,
            deployment_id=deployment_id,
            evaluation_id=evaluation_id,
            model_ids=compiled_models.ids,
            benchmark_ids=benchmark_ids,
            evaluation=evaluation,
            per_model_overrides=compiled_models.overrides,
            slug=_slug,
            smoke=smoke,
        )
        self.app.matrix_schemas.validate("matrix_spec", matrix)

        selected_profiles = {"backend": backend_profile_id, "evaluator": evaluator_profile_id}
        if hardware_profile_id is not None:
            selected_profiles["hardware"] = hardware_profile_id
        bundle = UserConfigBundle(
            system=system,
            evaluation=evaluation,
            matrix_spec=matrix,
            cache_root=cache_root,
            results_root=results_root,
            generated={
                "platform_id": platform_id,
                "deployment_id": deployment_id,
                "evaluation_id": evaluation_id,
                "model_ids": compiled_models.names,
                "model_catalog": {
                    "enabled": model_catalog_enabled,
                    "root": str(model_catalog_root),
                },
                "selected_profiles": selected_profiles,
                "run_mode": "smoke" if smoke else "standard",
            },
            specs=specs,
        )
        # Keep the historical app.specs lookup API working, but publish only a
        # complete validated snapshot.  Planning from ``bundle.specs`` remains
        # independent if a later load replaces this compatibility view.
        self.app.specs.replace_overlays(specs.overlay_snapshot())
        return bundle

    def _deployment(
        self,
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

    def _evaluation(
        self,
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
