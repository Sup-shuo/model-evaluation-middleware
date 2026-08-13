from __future__ import annotations

import copy
from typing import Any

from model_evaluation.core.errors import ConfigError


def adapter_parameters(platform: dict[str, Any], component: str) -> dict[str, Any]:
    """Return opaque parameters owned by one selected Platform component.

    PlatformProfile v1.1 stores functional parameters beside the component they
    configure. metadata is intentionally non-functional.
    """
    value = platform.get(component)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"PlatformProfile.{component} must be an object")
    params = value.get("parameters") or {}
    if not isinstance(params, dict):
        raise ConfigError(f"PlatformProfile.{component}.parameters must be an object")
    return copy.deepcopy(params)
