from __future__ import annotations

import copy
import math
from itertools import product

from model_evaluation.core.config.overrides import validate_run_overrides
from model_evaluation.core.errors import ConfigError
from model_evaluation.core.identifiers import stable_id
from model_evaluation.core.planner import Planner
from model_evaluation.core.execution_plan import validate_execution_plan


AXES = ("model", "platform", "deployment", "benchmark", "evaluation")
AXIS_KEYS = {
    "model": "models",
    "platform": "platforms",
    "deployment": "deployments",
    "benchmark": "benchmarks",
    "evaluation": "evaluations",
}
_DEFAULT_MAX_COMBINATIONS = 100_000


def _excluded(combo: dict[str, str], rules: list[dict]) -> bool:
    return any(
        all(combo.get(key) == value for key, value in rule.items())
        for rule in rules
    )


def _merge_dict(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def finalize_matrix_plan(obj: dict) -> None:
    obj["matrix_id"] = "matrix-" + stable_id(
        obj,
        length=24,
        exclude_keys={"matrix_id"},
    )


def verify_matrix_plan(obj: dict, *, app) -> None:
    app.matrix_schemas.validate("matrix_plan", obj)
    expected_id = "matrix-" + stable_id(
        obj,
        length=24,
        exclude_keys={"matrix_id"},
    )
    if obj.get("matrix_id") != expected_id:
        raise ConfigError("matrix_id does not match normalized matrix plan")
    app.matrix_schemas.validate("matrix_spec", obj.get("matrix_spec"))
    plans = obj.get("plans") or []
    if not plans:
        raise ConfigError("matrix plan contains no child plans")
    plan_ids = []
    for plan in plans:
        validate_execution_plan(plan, app.schemas)
        plan_ids.append(plan["plan_id"])
    if len(plan_ids) != len(set(plan_ids)):
        raise ConfigError("matrix plan contains duplicate child plan_id values")
    expected_runs = MatrixPlanner(app).expand(obj["matrix_spec"])
    actual_runs = [plan.get("run_spec") for plan in plans]
    if actual_runs != expected_runs:
        raise ConfigError(
            "matrix child plans do not exactly match deterministic MatrixSpec expansion"
        )
    summary = obj.get("summary") or {}
    incompatible = sum(
        1
        for plan in plans
        if plan.get("compatibility", {}).get("status") == "incompatible"
    )
    if summary.get("runs") != len(plans) or summary.get("incompatible") != incompatible:
        raise ConfigError("matrix summary does not match child plans")


class MatrixPlanner:
    def __init__(self, app):
        self.app = app

    def _planner(self, specs=None):
        if specs is None or specs is self.app.specs:
            return self.app.planner
        return Planner(
            project_root=self.app.root,
            schemas=self.app.schemas,
            specs=specs,
            registry=self.app.registry,
        )

    def expand(self, spec: dict) -> list[dict]:
        self.app.matrix_schemas.validate("matrix_spec", spec)
        values = [spec[AXIS_KEYS[axis]] for axis in AXES]
        excludes = spec.get("exclude") or []
        execution = spec.get("execution") or {}
        max_runs = int(execution.get("max_runs", 10_000))
        max_combinations = int(
            execution.get("max_combinations", _DEFAULT_MAX_COMBINATIONS)
        )
        total = math.prod(len(value) for value in values)
        if total > max_combinations:
            raise ConfigError(
                f"matrix Cartesian product has {total} combinations, exceeding "
                f"max_combinations={max_combinations}; split the matrix or raise the "
                "explicit safety bound"
            )
        unknown_models = set(spec.get("per_model_overrides") or {}) - set(
            spec["models"]
        )
        if unknown_models:
            raise ConfigError(
                "per_model_overrides references models outside matrix axis: "
                f"{sorted(unknown_models)}"
            )
        axis_values = {axis: set(spec[AXIS_KEYS[axis]]) for axis in AXES}
        for rule in excludes:
            for axis, value in rule.items():
                if value not in axis_values[axis]:
                    raise ConfigError(
                        f"exclude rule references value outside {axis} axis: {value!r}"
                    )
        runs = []
        seen = set()
        for combination in product(*values):
            combo = dict(zip(AXES, combination))
            if _excluded(combo, excludes):
                continue
            run = {"schema_version": "1.0", **combo}
            overrides = copy.deepcopy(spec.get("overrides") or {})
            model_patch = (spec.get("per_model_overrides") or {}).get(
                combo["model"]
            ) or {}
            if model_patch:
                overrides = _merge_dict(overrides, model_patch)
            if overrides:
                run["overrides"] = overrides
            tags = list(spec.get("tags") or [])
            if tags:
                run["tags"] = tags
            validate_run_overrides(run)
            key = stable_id(run, length=64)
            if key in seen:
                continue
            seen.add(key)
            runs.append(run)
            if len(runs) > max_runs:
                raise ConfigError(
                    f"matrix expands to more than max_runs={max_runs}; narrow "
                    "axes/exclusions or raise the explicit limit"
                )
        if not runs:
            raise ConfigError("matrix expands to zero runs")
        return runs

    def build(self, spec: dict, *, specs=None) -> dict:
        runs = self.expand(spec)
        cache = {}
        planner = self._planner(specs)
        plans = [planner.build(run, cache=cache) for run in runs]
        incompatible = sum(
            1
            for plan in plans
            if plan["compatibility"]["status"] == "incompatible"
        )
        obj = {
            "schema_version": "1.0",
            "matrix_id": "matrix-pending",
            "matrix_spec": copy.deepcopy(spec),
            "plans": plans,
            "summary": {
                "runs": len(plans),
                "incompatible": incompatible,
                "planning_cache_entries": len(cache),
            },
        }
        finalize_matrix_plan(obj)
        self.app.matrix_schemas.validate("matrix_plan", obj)
        return obj
