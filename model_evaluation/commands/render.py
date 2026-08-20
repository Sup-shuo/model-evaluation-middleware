from __future__ import annotations


def render_doctor(payload: dict) -> str:
    lines = [
        f"Doctor: {'PASS' if payload.get('ok') else 'FAIL'}",
        f"System: {payload.get('system') or '-'}",
        f"Scope: {payload.get('scope') or '-'}",
    ]
    for index, run in enumerate(payload.get("runs") or [], 1):
        lines.extend(
            [
                "",
                f"[{index}] {run.get('model_label') or run.get('model_id') or '-'}",
                f"  model: {run.get('model_id') or '-'}",
                f"  benchmark: {run.get('benchmark') or '-'}",
                f"  compatibility: {run.get('compatibility') or '-'}",
                f"  devices: {', '.join(str(value) for value in (run.get('devices') or [])) or '-'}",
                f"  runtime: {run.get('runtime') or '-'}",
            ]
        )
        for name, check in sorted((run.get("checks") or {}).items()):
            lines.append(f"  {name}: {check.get('status') or 'unknown'}")
            if check.get("error"):
                lines.append(f"    error: {check['error']}")
        for reason in run.get("reasons") or []:
            lines.append(f"  reason: {reason}")
        for warning in run.get("warnings") or []:
            lines.append(f"  warning: {warning}")
    return "\n".join(lines) + "\n"


def render_inspection(report: dict) -> str:
    lines = [
        f"Result: {'VALID' if report.get('ok') else 'INVALID'}",
        f"Run: {report.get('run_id') or '-'}",
        f"Outcome: {report.get('outcome') or '-'}",
        f"Model: {report.get('model') or '-'}",
        f"Benchmark: {report.get('benchmark') or '-'}",
        f"Framework: {report.get('framework') or '-'}",
        f"Started: {report.get('started_at') or '-'}",
        f"Finished: {report.get('finished_at') or '-'}",
        f"Cleanup: {report.get('cleanup') or '-'}",
        f"Tasks: {report.get('tasks', 0)}",
        f"Effective samples: {report.get('effective_samples') if report.get('effective_samples') is not None else '-'}",
    ]
    summary = report.get("summary") or {}
    if summary:
        lines.append("Metrics:")
        for name, entry in sorted(summary.items()):
            value = entry.get("value") if isinstance(entry, dict) else entry
            lines.append(f"  {name}: {value}")
    if report.get("error"):
        error = report["error"]
        lines.append(f"Error: {error.get('code')}: {error.get('message')}")
    return "\n".join(lines) + "\n"


def render_check(report: dict) -> str:
    lines = [
        f"Check: {'PASS' if report.get('ok') else 'FAIL'}",
        f"System: {report.get('system') or '-'}",
        f"Scope: {report.get('scope') or '-'}",
    ]
    labels = {
        "validation": "Validate configuration",
        "planning": "Preview execution plan",
        "doctor": "Doctor local environments",
        "resources": "Inspect point-in-time resources",
    }
    for name in ("validation", "planning", "doctor", "resources"):
        phase = (report.get("phases") or {}).get(name) or {}
        lines.append(f"[{str(phase.get('status') or 'unknown').upper()}] {labels[name]}")
        if phase.get("error"):
            lines.append(f"  {phase['error'].get('message') or '-'}")
    planning = ((report.get("phases") or {}).get("planning") or {}).get("details") or {}
    if planning.get("runs") is not None:
        lines.append(
            f"Plan: {planning.get('runs')} run(s), "
            f"{planning.get('incompatible', 0)} incompatible"
        )
    return "\n".join(lines) + "\n"


def render_explanation(report: dict) -> str:
    lines = [
        f"Explain: {'RUNNABLE' if report.get('ok') else 'NOT RUNNABLE'}",
        f"System: {report.get('system') or '-'}",
    ]
    explanations = report.get("explanations") or []
    if not explanations:
        lines.append("No blocking issue was found by validation, doctor, planning, or resource preview.")
    else:
        for index, item in enumerate(explanations, 1):
            lines.append(
                f"{index}. [{item.get('phase')}] {item.get('code')}: {item.get('message')}"
            )
    lines.append(
        "Resource estimates are reported only when the selected Backend provides them; unknown values are not inferred."
    )
    return "\n".join(lines) + "\n"
