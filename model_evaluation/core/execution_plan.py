from __future__ import annotations

from typing import Any

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.identifiers import stable_id


EXECUTION_STAGES = (
    "PLATFORM_READY",
    "DATA_READY",
    "TASK_READY",
    "EVALUATOR_PREFLIGHT",
    "SERVICE_STARTING",
    "SERVICE_READY",
    "EVALUATING",
    "NORMALIZING",
    "CLEANING",
)


def _adapter_identities(resolved: dict[str, Any]) -> set[tuple[str, str]]:
    specs = resolved["specs"]
    platform = specs["platform"]
    deployment = specs["deployment"]
    benchmark = specs["benchmark"]
    evaluation = specs["evaluation"]
    mode = resolved["management_mode"]
    expected = {
        ("environment", str(platform["evaluation_environment"]["provider"])),
        ("backend", str(deployment["backend"]["adapter"])),
        ("dataset", str(benchmark["dataset"]["provider"])),
        ("binding", str(resolved["binding_adapter"])),
        ("evaluator", str(evaluation["framework"]["adapter"])),
    }
    if mode == "managed":
        expected.update(
            {
                ("device", str(platform["device"]["adapter"])),
                ("runtime", str(platform["runtime"]["adapter"])),
                ("environment", str(platform["backend_environment"]["provider"])),
            }
        )
    return expected


def validate_execution_plan(plan: dict[str, Any], schemas) -> None:
    """Validate the frozen Core-to-Orchestrator execution contract.

    JSON Schema closes the data shape.  These checks bind duplicated identities
    that intentionally remain easy for humans and external schedulers to read.
    """

    schemas.validate("execution_plan", plan)
    expected_id = "plan-" + stable_id(plan, length=24, exclude_keys={"plan_id"})
    if plan.get("plan_id") != expected_id:
        raise ConfigError("plan_id does not match normalized execution plan")

    if tuple(plan.get("stages") or ()) != EXECUTION_STAGES:
        raise ConfigError("execution plan stages do not match the Core lifecycle")

    resolved = plan["resolved"]
    specs = resolved["specs"]
    if specs["run"] != plan["run_spec"]:
        raise ConfigError("execution plan run_spec disagrees with resolved.specs.run")
    if resolved["management_mode"] != specs["deployment"]["management"]["mode"]:
        raise ConfigError(
            "execution plan management_mode disagrees with DeploymentProfile"
        )

    actual_adapters = [
        (str(row["kind"]), str(row["name"])) for row in plan["adapters"]
    ]
    if len(actual_adapters) != len(set(actual_adapters)):
        raise ConfigError("execution plan contains duplicate Adapter identities")
    expected_adapters = _adapter_identities(resolved)
    if set(actual_adapters) != expected_adapters:
        raise ConfigError(
            "execution plan Adapter identities disagree with resolved specifications: "
            f"expected={sorted(expected_adapters)}, actual={sorted(actual_adapters)}"
        )

    run_locks = [
        claim
        for claim in plan["resources"]
        if claim.get("kind") == "run_lock"
        and claim.get("id") == "global-orchestrator"
        and claim.get("exclusive", True)
    ]
    if len(run_locks) != 1:
        raise ConfigError(
            "execution plan must contain exactly one exclusive global run lock"
        )


__all__ = ["EXECUTION_STAGES", "validate_execution_plan"]
