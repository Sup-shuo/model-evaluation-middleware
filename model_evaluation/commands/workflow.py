from __future__ import annotations

import copy
import json
from typing import Any

from model_evaluation.commands.doctor import build_doctor_report
from model_evaluation.commands.render import render_check, render_explanation
from model_evaluation.core.resources import ResourceManager
from model_evaluation.core.security import redact_diagnostic


PHASES = ("validation", "planning", "doctor", "resources")


def _error(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "code": str(getattr(exc, "code", None) or "CHECK_FAILED"),
        "message": str(exc) or type(exc).__name__,
    }


def _skipped(reason: str) -> dict:
    return {"status": "skipped", "details": {"reason": reason}}


def _plan_preview(plan: dict) -> list[dict]:
    rows = []
    for child in plan.get("plans") or []:
        run = child.get("run_spec") or {}
        rows.append(
            {
                "plan_id": child.get("plan_id"),
                "model": run.get("model"),
                "platform": run.get("platform"),
                "backend": run.get("deployment"),
                "benchmark": run.get("benchmark"),
                "evaluation": run.get("evaluation"),
                "compatibility": (child.get("compatibility") or {}).get("status"),
                "reasons": copy.deepcopy(
                    (child.get("compatibility") or {}).get("reasons") or []
                ),
            }
        )
    return rows


def _resource_report(plan: dict) -> dict:
    """Inspect point-in-time resources without acquiring framework leases."""

    runs = []
    ok = True
    for child in plan.get("plans") or []:
        claims = []
        for claim in child.get("resources") or []:
            row = {
                "kind": claim.get("kind"),
                "id": str(claim.get("id")),
                "status": "declared",
            }
            if claim.get("kind") == "port":
                host = str(claim.get("host") or "127.0.0.1")
                try:
                    ResourceManager.check_port(host, int(claim["id"]))
                    row["status"] = "available"
                except Exception as exc:
                    row["status"] = "unavailable"
                    row["error"] = _error(exc)
                    ok = False
            claims.append(row)
        runs.append({"plan_id": child.get("plan_id"), "claims": claims})
    return {
        "ok": ok,
        "mode": "point-in-time-read-only",
        "notes": [
            "device and cache claims are declared but not acquired",
            "resource availability may change after this check",
        ],
        "runs": runs,
    }


def _explanations(report: dict) -> list[dict]:
    explanations: list[dict[str, Any]] = []
    phases = report.get("phases") or {}
    for name in PHASES:
        phase = phases.get(name) or {}
        error = phase.get("error") or {}
        if phase.get("status") == "failed" and error:
            explanations.append(
                {
                    "phase": name,
                    "code": error.get("code") or f"{name.upper()}_FAILED",
                    "message": error.get("message") or f"{name} failed",
                }
            )
    planning = (phases.get("planning") or {}).get("details") or {}
    for row in planning.get("preview") or []:
        if row.get("compatibility") == "incompatible":
            reasons = row.get("reasons") or ["planned combination is incompatible"]
            for reason in reasons:
                explanations.append(
                    {
                        "phase": "planning",
                        "code": "INCOMPATIBLE_PLAN",
                        "message": f"{row.get('model')}: {reason}",
                    }
                )
    doctor = (phases.get("doctor") or {}).get("details") or {}
    for run in doctor.get("runs") or []:
        for name, check in sorted((run.get("checks") or {}).items()):
            if check.get("status") == "failed":
                explanations.append(
                    {
                        "phase": "doctor",
                        "code": "DOCTOR_CHECK_FAILED",
                        "message": (
                            f"{run.get('model_id') or '-'} / {name}: "
                            f"{check.get('error') or 'preflight failed'}"
                        ),
                    }
                )
    resources = (phases.get("resources") or {}).get("details") or {}
    for run in resources.get("runs") or []:
        for claim in run.get("claims") or []:
            if claim.get("status") == "unavailable":
                error = claim.get("error") or {}
                explanations.append(
                    {
                        "phase": "resources",
                        "code": error.get("code") or "RESOURCE_UNAVAILABLE",
                        "message": error.get("message") or (
                            f"{claim.get('kind')} {claim.get('id')} is unavailable"
                        ),
                    }
                )
    return explanations


def build_check_report(
    app,
    *,
    system_config: str | None,
    evaluation_config: str | None,
    smoke: bool = False,
) -> dict:
    report = {
        "schema_version": "1.0",
        "ok": False,
        "scope": "validate-doctor-plan-resource-preview-no-service-start",
        "system": None,
        "phases": {name: _skipped("an earlier phase did not complete") for name in PHASES},
        "explanations": [],
    }
    try:
        bundle = app.load_user_config(
            system_config,
            evaluation_config,
            smoke=smoke,
        )
        report["system"] = bundle.system["system"]["name"]
        report["phases"]["validation"] = {
            "status": "ok",
            "details": {
                "models": list(bundle.generated["model_ids"].values()),
                "benchmarks": copy.deepcopy(bundle.evaluation["benchmarks"]),
                "selected_profiles": copy.deepcopy(
                    bundle.generated.get("selected_profiles", {})
                ),
            },
        }
    except Exception as exc:
        report["phases"]["validation"] = {"status": "failed", "error": _error(exc)}
        report["explanations"] = _explanations(report)
        app.schemas.validate("diagnostic_report", report)
        return report

    try:
        plan = app.build_user_matrix_plan(bundle)
        incompatible = int((plan.get("summary") or {}).get("incompatible", 0))
        report["phases"]["planning"] = {
            "status": "failed" if incompatible else "ok",
            "details": {
                "matrix_id": plan["matrix_id"],
                "runs": len(plan.get("plans") or []),
                "incompatible": incompatible,
                "preview": _plan_preview(plan),
            },
        }
    except Exception as exc:
        report["phases"]["planning"] = {"status": "failed", "error": _error(exc)}
        report["explanations"] = _explanations(report)
        app.schemas.validate("diagnostic_report", report)
        return report

    try:
        doctor = build_doctor_report(app, plan=plan, bundle=bundle)
        report["phases"]["doctor"] = {
            "status": "ok" if doctor["ok"] else "failed",
            "details": doctor,
        }
    except Exception as exc:
        report["phases"]["doctor"] = {"status": "failed", "error": _error(exc)}

    try:
        resources = _resource_report(plan)
        report["phases"]["resources"] = {
            "status": "ok" if resources["ok"] else "failed",
            "details": resources,
        }
    except Exception as exc:
        report["phases"]["resources"] = {"status": "failed", "error": _error(exc)}

    report["ok"] = all(
        phase.get("status") == "ok" for phase in report["phases"].values()
    )
    report["explanations"] = _explanations(report)
    redacted = redact_diagnostic(report, [])
    app.schemas.validate("diagnostic_report", redacted)
    return redacted


def run_check(
    app,
    *,
    system_config: str | None,
    evaluation_config: str | None,
    output_format: str,
    explain: bool = False,
    smoke: bool = False,
) -> bool:
    report = build_check_report(
        app,
        system_config=system_config,
        evaluation_config=evaluation_config,
        smoke=smoke,
    )
    if output_format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif explain:
        print(render_explanation(report), end="")
    else:
        print(render_check(report), end="")
    return bool(report["ok"])
