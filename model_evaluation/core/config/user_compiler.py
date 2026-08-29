from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_evaluation.core.config.adapter_config import (
    validate_adapter_user_parameters as _validate_adapter_user_parameters,
)
from model_evaluation.core.config.loader import SpecRepository
from model_evaluation.core.config.matrix_compiler import compile_matrix_spec
from model_evaluation.core.config.merge import deep_merge as _deep_merge
from model_evaluation.core.config.model_catalog import resolve_model_entries
from model_evaluation.core.config.model_compiler import compile_models
from model_evaluation.core.config.user_common import (
    _backend_mode,
    _environment,
    _environment_selection,
    _load_yaml,
    _merge_ergonomic_parameter,
    _project_path,
    _runtime_families,
    _selected_profile,
    _slug,
    _validate_runtime_compatibility,
)
from model_evaluation.core.config.user_profiles import (
    compile_deployment,
    compile_evaluation,
)
from model_evaluation.core.errors import ConfigError


@dataclass(frozen=True)
class UserConfigBundle:
    system: dict[str, Any]
    evaluation: dict[str, Any]
    matrix_spec: dict[str, Any]
    cache_root: str
    results_root: str
    generated: dict[str, Any]
    specs: SpecRepository


def compile_user_config(
    app,
    system_path: str | Path,
    evaluation_path: str | Path,
    *,
    smoke: bool = False,
) -> UserConfigBundle:
    system_file = Path(system_path).expanduser().resolve()
    evaluation_file = Path(evaluation_path).expanduser().resolve()
    system = _load_yaml(system_file)
    evaluation = _load_yaml(evaluation_file)
    app.matrix_schemas.validate("user_system", system)
    app.matrix_schemas.validate("user_evaluation", evaluation)
    model_entries, model_catalog_root, model_catalog_enabled = resolve_model_entries(
        app, evaluation, evaluation_file
    )
    # Resolve into a private repository.  The shared Application repository
    # remains untouched until the entire user configuration has succeeded,
    # so a late validation error cannot publish a half-populated catalog.
    specs = app.specs.fork(include_overlays=False)

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
        app, "backend", backend_type, backend_profile_parameters,
        f"system.profiles.backend.{backend_profile_id}.parameters",
    )
    backend_run_parameters = copy.deepcopy((evaluation.get("backend") or {}).get("parameters") or {})
    _validate_adapter_user_parameters(
        app, "backend", backend_type, backend_run_parameters, "evaluation.backend.parameters",
    )
    evaluator_client = _validate_adapter_user_parameters(
        app, "evaluator", evaluator_type, evaluator.get("parameters"),
        f"system.profiles.evaluator.{evaluator_profile_id}.parameters",
    )
    evaluator_run_parameters = copy.deepcopy((evaluation.get("evaluator") or {}).get("parameters") or {})
    _validate_adapter_user_parameters(
        app, "evaluator", evaluator_type, evaluator_run_parameters, "evaluation.evaluator.parameters",
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
            app, "device", device_type, device_params,
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
            app, "runtime", runtime_type, runtime_params,
            f"system.profiles.hardware.{hardware_profile_id}.runtime.parameters",
        )
        _validate_runtime_compatibility(runtime_type, runtime_families, backend_profile_id=backend_profile_id)

        # Physical device identity is machine-owned. Evaluation may select
        # and reorder only devices exposed by the selected Hardware profile.
        system_devices = hardware.get("devices")
        selected_devices = resources.get("devices")
        if selected_devices is not None:
            if system_devices is None:
                raise ConfigError(
                    "evaluation.resources.devices 要求 "
                    f"system.profiles.hardware.{hardware_profile_id}.devices "
                    "显式声明设备池"
                )
            available = [str(value) for value in system_devices]
            available_set = set(available)
            devices = [str(value) for value in selected_devices]
            unavailable = [value for value in devices if value not in available_set]
            if unavailable:
                raise ConfigError(
                    "evaluation.resources.devices 必须只选择 "
                    f"system.profiles.hardware.{hardware_profile_id}.devices 中的设备；"
                    f"未声明设备={unavailable}，System 设备池={available}"
                )
        elif system_devices is not None:
            devices = [str(value) for value in system_devices]
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
        app, evaluator_env_value, system=system,
        label="evaluation.environments.evaluator" if "evaluator" in environment_overrides else f"system.profiles.evaluator.{evaluator_profile_id}.environment",
    )

    backend_env: dict[str, str] | None = None
    backend_env_params: dict[str, Any] = {}
    if mode == "managed":
        backend_env_value = environment_overrides.get("backend", backend.get("environment"))
        backend_env, backend_env_params = _environment(
            app, backend_env_value, system=system,
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
        app.project_root,
    )
    results_root = _project_path(
        system["paths"].get("results", "results"),
        "paths.results",
        app.project_root,
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

    deployment = compile_deployment(
        deployment_id, backend, system, backend_client,
        mode=mode, runtime_families=runtime_families,
    )
    specs.register("deployment", deployment)

    evaluation_spec = compile_evaluation(
        evaluation_id,
        evaluator,
        evaluation,
        evaluator_client,
        specs=specs,
        smoke=smoke,
    )
    specs.register("evaluation", evaluation_spec)

    compiled_models = compile_models(
        app=app,
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
            app,
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
            app,
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
    app.matrix_schemas.validate("matrix_spec", matrix)

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
    app.specs.replace_overlays(specs.overlay_snapshot())
    return bundle
