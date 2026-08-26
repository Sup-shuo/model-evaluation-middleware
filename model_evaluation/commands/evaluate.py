from __future__ import annotations

import json
from pathlib import Path

from model_evaluation.commands.render import render_check, render_explanation
from model_evaluation.commands.workflow import build_check_report
from model_evaluation.core.errors import ResultProductError
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.result_report import write_run_report


def _load_batch_runs(batch_dir: Path) -> list[dict]:
    path = batch_dir / "runs.json"
    try:
        rows = json_loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultProductError(f"cannot read batch runs: {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise ResultProductError(f"batch runs must be a JSON array: {path}")
    return rows


def _write_success_reports(
    rows: list[dict],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    reports = []
    warnings = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "success":
            continue
        run_dir = row.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            raise ResultProductError("successful batch row is missing run_dir")
        try:
            reports.append(write_run_report(run_dir))
        except (OSError, ValueError, ResultProductError) as exc:
            warnings.append(
                {
                    "code": "RESULT_PROJECTION_FAILED",
                    "run_dir": run_dir,
                    "message": str(exc) or type(exc).__name__,
                }
            )
    return reports, warnings


def _render_completion(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "",
        f"Automatic evaluation: {'PASS' if payload['ok'] else 'FAIL'}",
        f"Batch: {payload['batch_dir']}",
        (
            "Runs: "
            f"{summary.get('success', 0)} success, "
            f"{summary.get('failed', 0)} failed, "
            f"{summary.get('not_run', 0)} not run"
        ),
    ]
    for row in payload["runs"]:
        run_dir = row.get("run_dir")
        if run_dir:
            lines.append(f"  [{row.get('status', 'unknown')}] {run_dir}")
    for report in payload["rendered_reports"]:
        lines.append(f"  TXT: {report['text']}")
        lines.append(f"  SVG: {report['svg']}")
    for warning in payload["projection_warnings"]:
        lines.append(
            f"  WARNING [{warning['code']}]: {warning['run_dir']}: "
            f"{warning['message']}"
        )
    successful = [
        row.get("run_dir")
        for row in payload["runs"]
        if row.get("status") == "success" and row.get("run_dir")
    ]
    if successful:
        lines.append(f"Inspect: eval-manager inspect {successful[0]}")
    return "\n".join(lines) + "\n"


def run_automatic_evaluation(
    app,
    *,
    system_config: str | None,
    evaluation_config: str | None,
    results_root: str | None,
    cache_root: str | None,
    smoke: bool,
    output_format: str,
) -> int:
    """Preflight and execute the existing managed evaluation workflow."""

    check = build_check_report(
        app,
        system_config=system_config,
        evaluation_config=evaluation_config,
        smoke=smoke,
    )
    if not check["ok"]:
        if output_format == "json":
            print(json.dumps(
                {
                    "ok": False,
                    "mode": "automatic-evaluation",
                    "stage": "preflight",
                    "check": check,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ))
        else:
            print(render_check(check), end="")
            print(render_explanation(check), end="")
            print("No inference service was started.")
        return 2

    if output_format == "human":
        print(render_check(check), end="")
        print("Preflight passed. Starting the managed evaluation workflow...")

    plan, bundle = app.user_matrix_plan(
        system_config,
        evaluation_config,
        smoke=smoke,
    )
    executor = app.matrix_executor(
        results_root=results_root or bundle.results_root,
        cache_root=cache_root or bundle.cache_root,
    )
    batch_dir, summary = executor.execute(plan)
    rows = _load_batch_runs(batch_dir)
    rendered_reports, projection_warnings = _write_success_reports(rows)
    ok = not summary.get("failed") and not summary.get("not_run")
    payload = {
        "ok": bool(ok),
        "mode": "automatic-evaluation",
        "smoke": bool(smoke),
        "batch_dir": str(batch_dir),
        "summary": summary,
        "runs": rows,
        "rendered_reports": rendered_reports,
        "projection_warnings": projection_warnings,
    }
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_render_completion(payload), end="")
    return 0 if ok else 3
