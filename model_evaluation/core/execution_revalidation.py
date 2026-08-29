from __future__ import annotations

import copy
from pathlib import Path

from model_evaluation.core.compatibility import (
    device_runtime_compatibility,
    evaluate,
    facts_from_device,
    facts_from_environment,
    facts_from_runtime,
    merge_fact_sets,
)
from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.errors import CompatibilityError
from model_evaluation.core.files import atomic_json


def _verify_environment_identity(
    label: str,
    current: dict,
    expected: dict,
) -> None:
    for key in ("provider", "identity", "python", "executable_root"):
        expected_value = expected.get(key)
        if expected_value is not None and current.get(key) != expected_value:
            raise CompatibilityError(
                f"{label} {key} changed after planning: "
                f"expected {expected_value!r}, got {current.get(key)!r}"
            )


def _resolve_environment(orchestrator, platform: dict, key: str) -> dict:
    environment = platform[key]
    client = orchestrator.registry.get("environment", environment["provider"])
    return orchestrator._invoke(
        client,
        "resolve",
        {
            "profile": environment["profile"],
            "parameters": adapter_parameters(platform, key),
        },
        context={"timeout_seconds": 3, "execution_revalidation": True},
        timeout=4,
    )


def _revalidate_managed_platform(
    orchestrator,
    *,
    platform: dict,
    expected: dict,
) -> tuple[dict, dict, dict]:
    device_client = orchestrator.registry.get(
        "device", platform["device"]["adapter"]
    )
    device_params = adapter_parameters(platform, "device")
    runtime_params = adapter_parameters(platform, "runtime")
    current_device = orchestrator._invoke(
        device_client,
        "probe",
        {
            "requested_devices": platform["device"].get("devices", []),
            "parameters": device_params,
        },
        context={"timeout_seconds": 3, "execution_revalidation": True},
        timeout=4,
    )
    wanted = {device["id"] for device in expected["device"]["devices"]}
    got = {device["id"] for device in current_device["devices"]}
    if current_device["vendor"] != expected["device"]["vendor"] or got != wanted:
        raise CompatibilityError(
            "device facts changed after planning: expected vendor/devices="
            f'{expected["device"]["vendor"]}/{sorted(wanted)}, '
            f'got={current_device["vendor"]}/{sorted(got)}'
        )

    device_patch = orchestrator._invoke(
        device_client,
        "visibility",
        {
            "devices": [device["id"] for device in current_device["devices"]],
            "descriptor": current_device,
            "parameters": device_params,
        },
        context={"timeout_seconds": 3, "execution_revalidation": True},
        timeout=4,
    )["env_patch"]
    if device_patch != expected.get("device_env_patch", {}):
        raise CompatibilityError("device visibility EnvPatch changed after planning")

    runtime_client = orchestrator.registry.get(
        "runtime", platform["runtime"]["adapter"]
    )
    runtime = orchestrator._invoke(
        runtime_client,
        "probe",
        {
            "profile": platform["runtime"].get("profile"),
            "parameters": runtime_params,
        },
        context={"timeout_seconds": 3, "execution_revalidation": True},
        timeout=4,
    )
    if runtime["family"] != expected["runtime"]["family"] or not runtime["available"]:
        raise CompatibilityError(
            "runtime facts changed after planning: expected "
            f'{expected["runtime"]["family"]}, got={runtime.get("family")} '
            f'available={runtime.get("available")}'
        )
    expected_version = expected["runtime"].get("version")
    if (
        expected_version not in (None, "unknown")
        and runtime.get("version") != expected_version
    ):
        raise CompatibilityError(
            "runtime version changed after planning: expected "
            f'{expected_version}, got={runtime.get("version")}'
        )

    pair = device_runtime_compatibility(current_device, runtime)
    pair_report = {
        "compatible": pair.compatible,
        "reasons": pair.reasons,
        "optional_misses": pair.optional_misses,
        "diagnostics": pair.diagnostics,
    }
    if not pair.compatible:
        raise CompatibilityError(
            "; ".join(pair.reasons),
            details={"diagnostics": pair.diagnostics},
        )

    runtime_patch = orchestrator._invoke(
        runtime_client,
        "resolve_environment",
        {
            "descriptor": runtime,
            "profile": platform["runtime"].get("profile"),
            "parameters": runtime_params,
        },
        context={"timeout_seconds": 3, "execution_revalidation": True},
        timeout=4,
    )["env_patch"]
    if runtime_patch != expected.get("runtime_env_patch", {}):
        raise CompatibilityError("runtime EnvPatch changed after planning")

    backend_environment = _resolve_environment(
        orchestrator, platform, "backend_environment"
    )
    _verify_environment_identity(
        "backend environment",
        backend_environment,
        expected["backend_environment"],
    )
    fresh = {
        "device": current_device,
        "runtime": runtime,
        "backend_environment": backend_environment,
        "device_env_patch": device_patch,
        "runtime_env_patch": runtime_patch,
    }
    facts = merge_fact_sets(
        facts_from_device(current_device),
        facts_from_runtime(runtime),
        facts_from_environment(backend_environment, "backend_environment"),
    )
    return fresh, facts, pair_report


def _evaluate_requirements(
    plan: dict,
    facts: dict,
    run_dir: Path,
    reports: dict[str, dict],
) -> None:
    for name, key in (
        ("deployment_compatibility", "deployment_compatibility_requirements"),
        ("backend", "backend_requirements"),
        ("binding", "binding_requirements"),
    ):
        requirements = plan["resolved"].get(key)
        if not requirements:
            continue
        report = evaluate(requirements, facts)
        reports[name] = {
            "compatible": report.compatible,
            "reasons": report.reasons,
            "optional_misses": report.optional_misses,
            "diagnostics": report.diagnostics,
        }
        if report.compatible:
            continue
        atomic_json(
            run_dir / ".run" / "diagnostics" / "execution_preflight_compatibility.json",
            {"compatible": False, "reports": reports},
        )
        raise CompatibilityError(
            f"{name} requirements changed after planning: "
            + "; ".join(report.reasons),
            details={"diagnostics": report.diagnostics},
        )


def revalidate_platform(orchestrator, plan: dict, run_dir: Path) -> dict:
    """Re-probe the platform without changing the frozen execution plan."""

    specs = plan["resolved"]["specs"]
    deployment = specs["deployment"]
    platform = specs["platform"]
    expected = plan["resolved"]["platform"]

    evaluation_environment = _resolve_environment(
        orchestrator, platform, "evaluation_environment"
    )
    _verify_environment_identity(
        "evaluation environment",
        evaluation_environment,
        expected["evaluation_environment"],
    )

    fresh = copy.deepcopy(expected)
    fresh["evaluation_environment"] = evaluation_environment
    facts = merge_fact_sets(
        facts_from_environment(evaluation_environment, "evaluation_environment")
    )
    reports: dict[str, dict] = {}

    if deployment["management"]["mode"] == "managed":
        managed, managed_facts, pair_report = _revalidate_managed_platform(
            orchestrator,
            platform=platform,
            expected=expected,
        )
        fresh.update(managed)
        facts = merge_fact_sets(facts, managed_facts)
        reports["device_runtime_pair"] = pair_report

    _evaluate_requirements(plan, facts, run_dir, reports)
    atomic_json(
        run_dir / ".run" / "diagnostics" / "execution_preflight_compatibility.json",
        {"compatible": True, "reports": reports},
    )
    return fresh
