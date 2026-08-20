from __future__ import annotations

import copy
import json
from pathlib import Path

from model_evaluation.commands.render import render_doctor
from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.security import redact_diagnostic


EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS = 30


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def _prepare_process(
    orchestrator,
    process: dict,
    *,
    platform: dict,
    resolved_platform: dict,
    role: str,
    base_patches=(),
) -> tuple[dict, list]:
    return orchestrator.prepare_process_for_environment(
        process,
        platform_spec=platform,
        resolved_platform=resolved_platform,
        role=role,
        base_patches=tuple(base_patches),
        context={"doctor": True, "offline": True, "preflight": True},
        timeout=5,
    )


def _evaluator_check(app, orchestrator, evaluator: dict, cache_root: str, platform: dict, resolved_platform: dict) -> tuple[dict, list]:
    client = app.registry.get("evaluator", evaluator["framework"]["adapter"])
    warnings: list = []
    try:
        operations = client.identity.manifest.get("operations") or []
        if "plan_preflight" not in operations:
            snapshot = client.invoke(
                "snapshot",
                {"evaluation": evaluator},
                context={"doctor": True, "offline": True},
                timeout=EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS,
            )
            return (
                {
                    "status": "ok" if not snapshot.get("probe_error") else "failed",
                    "snapshot": snapshot,
                },
                warnings,
            )

        planned = client.invoke(
            "plan_preflight",
            {"evaluation": evaluator, "cache_root": cache_root},
            context={
                "doctor": True,
                "cache_root": cache_root,
                "offline": True,
                "preflight": True,
            },
            timeout=EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS,
        )
        process = copy.deepcopy(planned["process"])
        process.setdefault("metadata", {})["role"] = "doctor_evaluator_preflight"
        process, warnings = _prepare_process(
            orchestrator,
            process,
            platform=platform,
            resolved_platform=resolved_platform,
            role="evaluator",
        )
        completed = orchestrator.pm.run(process)
        check = {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": _text(completed.stdout)[:4000],
            "stderr": _text(completed.stderr)[:4000],
        }
        if planned.get("result_format") == "preflight_result":
            result = orchestrator._preflight_json_result(_text(completed.stdout))
            app.schemas.validate("preflight_probe_result", result)
            check["result"] = result
            if (completed.returncode == 0) != (result["status"] == "passed"):
                check["status"] = "failed"
                check["error"] = "preflight process/result status mismatch"
        return check, warnings
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, warnings


def _backend_check(app, orchestrator, child: dict, cache_root: str, platform: dict, resolved_platform: dict, model: dict, deployment: dict) -> tuple[dict, list]:
    if deployment["management"]["mode"] != "managed":
        return {"status": "deferred_external"}, []
    warnings: list = []
    backend = app.registry.get("backend", deployment["backend"]["adapter"])
    try:
        offline = bool((child["run_spec"].get("overrides") or {}).get("offline", False))
        operations = backend.identity.manifest.get("operations") or []
        if "plan_preflight" in operations:
            planned = backend.invoke(
                "plan_preflight",
                {
                    "model": model,
                    "deployment": deployment,
                    "platform": resolved_platform,
                    "network_policy": "offline" if offline else "online",
                },
                context={"doctor": True, "offline": offline, "preflight": True},
                timeout=5,
            )
            report = orchestrator.run_backend_preflight(
                planned,
                platform_spec=platform,
                resolved_platform=resolved_platform,
                raise_on_failure=False,
            )
            return {
                "status": "ok" if report["status"] == "passed" else "failed",
                "preflight": report,
            }, warnings

        start = backend.invoke(
            "plan_start",
            {
                "model": model,
                "deployment": deployment,
                "platform": resolved_platform,
                "endpoint": child["resolved"].get("endpoint", {}),
                "log_path": str(Path(cache_root) / "doctor-backend.log"),
                "network_policy": "offline" if offline else "online",
            },
            context={"doctor": True, "offline": offline},
            timeout=5,
        )
        probe = copy.deepcopy(start["dependency_probe"])
        original_argv = list(probe.get("argv") or [])
        probe.setdefault("metadata", {})["role"] = "doctor_backend_dependency_probe"
        probe, warnings = _prepare_process(
            orchestrator,
            probe,
            platform=platform,
            resolved_platform=resolved_platform,
            role="backend",
            base_patches=(
                ("device", resolved_platform.get("device_env_patch")),
                ("runtime", resolved_platform.get("runtime_env_patch")),
            ),
        )
        completed = orchestrator.pm.run(probe)
        return {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "argv": original_argv,
            "wrapped_argv": probe.get("argv"),
            "stdout": _text(completed.stdout)[:4000],
            "stderr": _text(completed.stderr)[:4000],
        }, warnings
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, warnings


def _doctor_row(app, orchestrator, child: dict, cache_root: str) -> dict:
    specs = child["resolved"]["specs"]
    platform = specs["platform"]
    resolved_platform = child["resolved"].get("platform") or {}
    model = specs["model"]
    metadata = model.get("metadata") or {}
    model_id = model.get("experiment_id") or metadata.get("experiment_id") or model["id"]
    row = {
        "model_id": model_id,
        "model_label": model.get("label") or metadata.get("label") or model_id,
        "model_ref": (model.get("source") or {}).get("ref"),
        "benchmark": specs["benchmark"]["id"],
        "compatibility": child["compatibility"]["status"],
        "reasons": child["compatibility"].get("reasons") or [],
        "backend_environment": (resolved_platform.get("backend_environment") or {}).get("identity"),
        "evaluation_environment": (resolved_platform.get("evaluation_environment") or {}).get("identity"),
        "runtime": (resolved_platform.get("runtime") or {}).get("family"),
        "devices": [
            device.get("id")
            for device in ((resolved_platform.get("device") or {}).get("devices") or [])
        ],
        "checks": {},
        "deferred": ["backend service readiness", "service capability compatibility"],
        "warnings": copy.deepcopy(child.get("warnings") or []),
    }
    evaluator, evaluator_warnings = _evaluator_check(
        app,
        orchestrator,
        specs["evaluation"],
        cache_root,
        platform,
        resolved_platform,
    )
    backend, backend_warnings = _backend_check(
        app,
        orchestrator,
        child,
        cache_root,
        platform,
        resolved_platform,
        model,
        specs["deployment"],
    )
    row["checks"]["evaluator_environment"] = evaluator
    row["checks"]["backend_environment"] = backend
    row["warnings"].extend(evaluator_warnings)
    row["warnings"].extend(backend_warnings)
    return row


def run_doctor(app, *, system_config: str | None, evaluation_config: str | None, output_format: str) -> bool:
    plan, bundle = app.user_matrix_plan(system_config, evaluation_config)
    orchestrator = app.orchestrator(
        results_root=bundle.results_root,
        cache_root=bundle.cache_root,
    )
    rows = [
        _doctor_row(app, orchestrator, child, bundle.cache_root)
        for child in plan["plans"]
    ]
    ok = all(
        row["compatibility"] != "incompatible"
        and all(check.get("status") != "failed" for check in row["checks"].values())
        for row in rows
    )
    payload = {
        "ok": ok,
        "scope": "local-workload-preflight-no-service-start",
        "system": bundle.system["system"]["name"],
        "selected_profiles": bundle.generated.get("selected_profiles", {}),
        "runs": rows,
    }
    redacted = redact_diagnostic(payload, orchestrator.pm.secrets.redaction_values())
    if output_format == "json":
        print(json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_doctor(redacted), end="")
    return ok
