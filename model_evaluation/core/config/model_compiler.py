from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from model_evaluation.core.config.adapter_config import (
    adapter_user_config,
    validate_adapter_user_parameters,
)
from model_evaluation.core.config.merge import deep_merge
from model_evaluation.core.errors import ConfigError


@dataclass(frozen=True)
class CompiledModels:
    ids: list[str]
    overrides: dict[str, dict[str, Any]]
    names: dict[str, str]


def _model_devices(
    resources: object,
    *,
    available_devices: list[str] | None,
    mode: str,
    label: str,
) -> list[str] | None:
    value = resources or {}
    if not isinstance(value, dict):
        raise ConfigError(f"{label}.resources 必须是 object")
    count = value.get("device_count")
    if count is None:
        return copy.deepcopy(available_devices)
    if mode != "managed":
        raise ConfigError(f"{label}.resources.device_count 仅适用于 managed backend")
    if available_devices is None:
        raise ConfigError(
            f"{label}.resources.device_count 需要 System hardware profile 显式提供 devices 设备池"
        )
    count = int(count)
    if count > len(available_devices):
        raise ConfigError(
            f"{label}.resources.device_count={count} 超过可用设备池 {available_devices}"
        )
    return copy.deepcopy(available_devices[:count])


def _backend_parameters(app, row: dict[str, Any], *, backend_type: str, label: str) -> dict[str, Any]:
    if "backends" in row:
        value = copy.deepcopy((row.get("backends") or {}).get(backend_type) or {})
        parameter_label = f"{label}.backends.{backend_type}"
        if "backend" in row:
            value = deep_merge(value, copy.deepcopy(row.get("backend") or {}))
            parameter_label = f"{label}.overrides.backend"
    else:
        value = copy.deepcopy(row.get("backend") or {})
        parameter_label = f"{label}.backend"
    if value:
        validate_adapter_user_parameters(
            app,
            "backend",
            backend_type,
            value,
            parameter_label,
        )
    return value


def _derived_backend_parameters(
    client,
    *,
    profile_parameters: dict[str, Any],
    run_parameters: dict[str, Any],
    devices: list[str] | None,
    mode: str,
) -> dict[str, Any]:
    output = copy.deepcopy(run_parameters)
    if mode != "managed":
        return output
    rules = adapter_user_config(client).get("derived_parameters") or {}
    if not isinstance(rules, dict):
        raise ConfigError(
            f"backend adapter {client.identity.name!r} user_config.derived_parameters 必须是 object"
        )
    for key, rule in rules.items():
        if key in profile_parameters or key in output:
            continue
        if not isinstance(rule, dict):
            raise ConfigError(
                f"backend adapter {client.identity.name!r} derived parameter {key!r} 规则必须是 object"
            )
        if rule.get("source") != "selected_device_count":
            raise ConfigError(
                f"backend adapter {client.identity.name!r} derived parameter {key!r} "
                f"使用未知 source: {rule.get('source')!r}"
            )
        if devices is None:
            continue
        value = len(devices)
        if rule.get("minimum") is not None:
            value = max(int(rule["minimum"]), value)
        output[str(key)] = value
    return output


def compile_models(
    *,
    app,
    specs,
    model_entries: list[dict[str, Any]],
    deployment: dict[str, Any],
    backend_type: str,
    backend_client,
    mode: str,
    available_devices: list[str] | None,
    backend_profile_parameters: dict[str, Any],
    backend_run_parameters: dict[str, Any],
    slug: Callable[[str, str], str],
    environment_selection: Callable[[object, str], dict[str, Any]],
) -> CompiledModels:
    ids: list[str] = []
    overrides: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    seen_ids: set[str] = set()

    for row in model_entries:
        run_resources = copy.deepcopy(row.pop("_run_resources", None) or {})
        source_value = row.get("source")
        if source_value is not None and not isinstance(source_value, dict):
            raise ConfigError("model source 必须是 object")
        ref = str(
            (source_value or {}).get("ref") or row.get("ref") or row.get("name") or ""
        ).strip()
        if not ref:
            raise ConfigError("模型配置需要 source.ref（或 legacy ref/name）")
        experiment_id = str(row.get("id") or ref).strip()
        label = str(row.get("label") or row.get("name") or experiment_id).strip()
        if not experiment_id or not label:
            raise ConfigError("evaluation.models id/label cannot be empty")
        if experiment_id in seen_ids:
            raise ConfigError(f"evaluation.models id 重复: {experiment_id}")
        seen_ids.add(experiment_id)
        model_id = slug(experiment_id, "user-model")
        default_source_type = "local" if deployment.get("model_location") else "other"
        source = {
            "type": str((source_value or {}).get("type") or row.get("source_type") or default_source_type),
            "ref": ref,
        }
        revision = (source_value or {}).get("revision", row.get("revision"))
        if revision is not None:
            source["revision"] = str(revision)
        model: dict[str, Any] = {
            "schema_version": "1.0",
            "id": model_id,
            "source": source,
            "provenance": copy.deepcopy(row.get("provenance") or {"policy": "migration"}),
            "experiment_id": experiment_id,
            "label": label,
            "metadata": copy.deepcopy(row.get("metadata") or {}),
        }
        for key in ("architecture", "quantization", "format", "chat_template"):
            if row.get(key) is not None:
                model[key] = str(row[key])
        if row.get("context_length") is not None:
            model["context_length"] = int(row["context_length"])
        if row.get("trust_remote_code") is not None:
            model["trust_remote_code"] = bool(row["trust_remote_code"])
        if row.get("tokenizer") is not None:
            tokenizer = row["tokenizer"]
            model["tokenizer"] = (
                {"ref": tokenizer}
                if isinstance(tokenizer, str)
                else copy.deepcopy(tokenizer)
            )
        specs.register("model", model)
        ids.append(model_id)
        names[model_id] = label

        patch: dict[str, Any] = {}
        model_parameters = _backend_parameters(
            app,
            row,
            backend_type=backend_type,
            label=f"evaluation.models[{experiment_id}]",
        )
        selected_devices = _model_devices(
            run_resources,
            available_devices=available_devices,
            mode=mode,
            label=f"evaluation.models[{experiment_id}]",
        )
        if run_resources:
            patch["platform"] = {"device": {"devices": selected_devices}}
        parameters = _derived_backend_parameters(
            backend_client,
            profile_parameters=backend_profile_parameters,
            run_parameters=deep_merge(backend_run_parameters, model_parameters),
            devices=selected_devices,
            mode=mode,
        )
        if parameters:
            patch["deployment"] = {"parameters": parameters}
        if row.get("model_location"):
            patch.setdefault("deployment", {})["model_location"] = copy.deepcopy(
                row["model_location"]
            )
        if mode in {"external", "attached"}:
            patch.setdefault("deployment", {}).setdefault("endpoint", {})["model_id"] = ref

        model_environments = row.get("environments") or {}
        if "backend" in model_environments:
            if mode != "managed":
                raise ConfigError(
                    f"evaluation.models[{experiment_id}].environments.backend 仅适用于 managed backend"
                )
            patch.setdefault("platform", {})["backend_environment"] = environment_selection(
                model_environments["backend"],
                f"evaluation.models[{experiment_id}].environments.backend",
            )
        if "evaluator" in model_environments:
            patch.setdefault("platform", {})["evaluation_environment"] = environment_selection(
                model_environments["evaluator"],
                f"evaluation.models[{experiment_id}].environments.evaluator",
            )
        if patch:
            overrides[model_id] = patch

    return CompiledModels(ids=ids, overrides=overrides, names=names)


__all__ = ["CompiledModels", "compile_models"]
