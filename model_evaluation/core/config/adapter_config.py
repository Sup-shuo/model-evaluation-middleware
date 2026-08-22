from __future__ import annotations

import copy
from typing import Any

from jsonschema import Draft202012Validator

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.core.schema.formats import contract_format_checker


def adapter_user_config(client) -> dict[str, Any]:
    value = client.identity.manifest.get("user_config") or {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"adapter {client.identity.kind}/{client.identity.name} user_config 必须是 object"
        )
    if value and value.get("schema_version") != "1.0":
        raise ConfigError(
            f"adapter {client.identity.kind}/{client.identity.name} "
            f"user_config.schema_version 不受支持: {value.get('schema_version')!r}"
        )
    return copy.deepcopy(value)


def validate_adapter_user_parameters(
    app,
    kind: str,
    adapter_name: str,
    value: object,
    label: str,
):
    """Validate Adapter-owned user parameters and resolve Adapter identity."""

    client = app.registry.get(kind, adapter_name)
    if value in (None, {}):
        return client
    if not isinstance(value, dict):
        raise ConfigError(f"{label} 必须是 object")
    user_config = adapter_user_config(client)
    schema_name = user_config.get("parameters_schema")
    if not schema_name:
        raise ConfigError(
            f"{kind} adapter {adapter_name!r} 未声明用户参数 schema，"
            f"不能安全接受 {label} 覆盖"
        )
    adapter_root = client.identity.path.parent.resolve()
    schema_path = (adapter_root / str(schema_name)).resolve()
    try:
        schema_path.relative_to(adapter_root)
    except ValueError as exc:
        raise ConfigError(
            f"adapter {kind}/{adapter_name} 的 user_config.parameters_schema 越界"
        ) from exc
    if not schema_path.is_file():
        raise ConfigError(
            f"adapter {kind}/{adapter_name} 缺少用户参数 schema: {schema_path.name}"
        )
    try:
        schema = json_loads_strict(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(
            schema,
            format_checker=contract_format_checker(),
        )
        errors = sorted(
            validator.iter_errors(value),
            key=lambda error: list(error.absolute_path),
        )
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"无法加载 {kind}/{adapter_name} 用户参数 schema: {exc}"
        ) from exc
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ConfigError(
            f"{label} 参数不合法（{kind}/{adapter_name}，{path}）：{error.message}"
        )
    return client


__all__ = ["adapter_user_config", "validate_adapter_user_parameters"]
