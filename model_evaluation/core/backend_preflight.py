from __future__ import annotations

import copy
import time
from pathlib import Path

from model_evaluation.core.errors import ProcessError
from model_evaluation.core.files import atomic_json


def preflight_error(report: dict) -> ProcessError:
    failures = [
        f"{row['id']}={row['status']}: "
        f"{str(row.get('error') or row.get('stderr') or row.get('stdout') or 'probe failed')[-2000:]}"
        for row in (report.get("probes") or [])
        if row.get("required") and row.get("status") != "passed"
    ]
    return ProcessError("backend preflight failed: " + "; ".join(failures))


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


def _blocked_probe(declared: dict) -> dict:
    return {
        "id": str(declared["id"]),
        "phase": str(declared["phase"]),
        "required": bool(declared["required"]),
        "status": "blocked",
        "duration_ms": 0,
        "error": "blocked by a required backend_dependency probe failure",
    }


def _execute_probe(orchestrator, declared: dict, platform_spec: dict, resolved_platform: dict) -> dict:
    started = time.monotonic()
    process = copy.deepcopy(declared["process"])
    process.setdefault("metadata", {}).update(
        {
            "role": "backend_preflight",
            "probe_id": str(declared["id"]),
            "probe_phase": str(declared["phase"]),
        }
    )
    row = {
        "id": str(declared["id"]),
        "phase": str(declared["phase"]),
        "required": bool(declared["required"]),
        "status": "failed",
        "duration_ms": 0,
        "argv": list(process.get("argv") or []),
    }
    try:
        wrapped, _warnings = orchestrator.prepare_process_for_environment(
            process,
            platform_spec=platform_spec,
            resolved_platform=resolved_platform,
            role="backend",
            base_patches=(
                ("device", resolved_platform.get("device_env_patch")),
                ("runtime", resolved_platform.get("runtime_env_patch")),
            ),
            context={"diagnostic": True, "preflight": True},
            timeout=5,
        )
        row["wrapped_argv"] = list(wrapped.get("argv") or [])
        completed = orchestrator.pm.run(wrapped)
        stdout = _text(completed.stdout)
        row.update(
            {
                "returncode": int(completed.returncode),
                "stdout": stdout[-8000:],
                "stderr": _text(completed.stderr)[-8000:],
            }
        )
        if declared["result_format"] == "preflight_result":
            result = orchestrator._preflight_json_result(stdout)
            orchestrator.schemas.validate("preflight_probe_result", result)
            row["result"] = result
            process_passed = completed.returncode == 0
            result_passed = result["status"] == "passed"
            if process_passed != result_passed:
                row["error"] = (
                    "preflight process/result status mismatch: "
                    f"returncode={completed.returncode}, result.status={result['status']!r}"
                )
            elif result_passed:
                row["status"] = "passed"
            else:
                error = result["error"]
                row["error"] = f"{error['code']}: {error['message']}"
        elif completed.returncode == 0:
            row["status"] = "passed"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["duration_ms"] = max(0, int((time.monotonic() - started) * 1000))
    return row


def run_backend_preflight(
    orchestrator,
    preflight_plan: dict,
    *,
    platform_spec: dict,
    resolved_platform: dict,
    report_path: str | Path | None = None,
    raise_on_failure: bool = True,
) -> dict:
    """Execute an Adapter-declared preflight plan with portable Core semantics."""
    orchestrator.schemas.validate("backend_preflight_plan", preflight_plan)
    rows: list[dict] = []
    dependency_blocked = False
    for declared in preflight_plan["probes"]:
        if dependency_blocked and declared["phase"] == "model_compatibility":
            row = _blocked_probe(declared)
        else:
            row = _execute_probe(orchestrator, declared, platform_spec, resolved_platform)
        rows.append(row)
        dependency_blocked = dependency_blocked or (
            row["required"]
            and row["phase"] == "backend_dependency"
            and row["status"] != "passed"
        )

    failed = any(row["required"] and row["status"] != "passed" for row in rows)
    report = {
        "schema_version": "1.0",
        "status": "failed" if failed else "passed",
        "probes": rows,
        "environment": str(
            (resolved_platform.get("backend_environment") or {}).get("identity") or ""
        ),
        "executable_root": str(
            (resolved_platform.get("backend_environment") or {}).get("executable_root") or ""
        ),
    }
    orchestrator.schemas.validate("preflight_report", report)
    if report_path is not None:
        atomic_json(report_path, orchestrator._redact_diagnostic(report))
    if failed and raise_on_failure:
        raise preflight_error(report)
    return report
