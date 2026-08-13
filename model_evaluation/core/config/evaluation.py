from __future__ import annotations

import copy
from typing import Any

from model_evaluation.core.errors import ConfigError


def resolve_evaluation_profile(evaluation: dict[str, Any], platform: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the evaluator-owned profile unchanged apart from a defensive copy.

    Earlier alpha releases supported Core-owned parameter templates hidden under
    ``metadata.middleware``.  That feature had no real production use and made
    metadata affect execution, so alpha21 deliberately removes it.  Evaluator
    parameters remain fully adapter-owned.
    """
    effective = copy.deepcopy(evaluation)
    params = copy.deepcopy(effective.get("parameters") or {})
    if not isinstance(params, dict):
        raise ConfigError("EvaluationProfile.parameters must be an object")
    metadata = effective.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ConfigError("EvaluationProfile.metadata must be an object")
    middleware = metadata.get("middleware") or {}
    if isinstance(middleware, dict) and "parameter_templates" in middleware:
        raise ConfigError("EvaluationProfile.metadata.middleware.parameter_templates was removed in alpha21; use explicit evaluator parameters")
    effective["parameters"] = params
    return effective, {"parameter_templates_removed": True}
