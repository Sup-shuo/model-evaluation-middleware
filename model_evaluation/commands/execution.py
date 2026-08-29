from __future__ import annotations

import argparse
from pathlib import Path

from model_evaluation import package_root
from model_evaluation.commands.output import dump
from model_evaluation.core.app import Application
from model_evaluation.core.errors import ConfigError, ResultProductError
from model_evaluation.core.files import atomic_json
from model_evaluation.core.result_product import inspect_run_product
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.result_report import write_run_report


def _matrix_executor(app: Application, args: argparse.Namespace, plan: dict):
    user_paths = (plan.get("summary") or {}).get("user_config") or {}
    return app.matrix_executor(
        results_root=args.results_root or user_paths.get("results_root"),
        cache_root=args.cache_root or user_paths.get("cache_root"),
    )


def _render_run_summary(path: str | Path) -> dict[str, str]:
    try:
        return write_run_report(path)
    except (OSError, ValueError) as exc:
        raise ResultProductError(
            f"cannot render result summary for {path}: {exc}"
        ) from exc


def _render_batch_summaries(
    batch_dir: Path,
    *,
    render_run=None,
) -> list[dict[str, str]]:
    renderer = render_run or _render_run_summary
    runs_path = batch_dir / "runs.json"
    try:
        rows = json_loads_strict(runs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultProductError(
            f"cannot read batch runs: {runs_path}: {exc}"
        ) from exc
    if not isinstance(rows, list):
        raise ResultProductError(f"batch runs must be a JSON array: {runs_path}")
    reports = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "success":
            continue
        run_dir = row.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            raise ResultProductError("successful batch row is missing run_dir")
        reports.append(renderer(run_dir))
    return reports


def _emit_batch(
    path: Path,
    summary: dict,
    rendered_reports: list[dict[str, str]] | None = None,
) -> None:
    payload = {"batch_dir": str(path), "summary": summary}
    if rendered_reports is not None:
        payload["rendered_reports"] = rendered_reports
    dump(payload)
    if summary["failed"] or summary["not_run"]:
        raise SystemExit(3)


def _handle_plan_or_run(args: argparse.Namespace, app: Application) -> bool:
    if args.cmd == "demo":
        example_root = package_root() / "examples" / "mock"
        plan, bundle = app.user_matrix_plan(
            example_root / "system.yaml",
            example_root / "evaluation.yaml",
        )
        executor = app.matrix_executor(
            results_root=args.results_root or bundle.results_root,
            cache_root=args.cache_root or bundle.cache_root,
        )
        batch_dir, summary = executor.execute(plan)
        runs = json_loads_strict(
            (batch_dir / "runs.json").read_text(encoding="utf-8")
        )
        successful = [row for row in runs if row.get("status") == "success"]
        if summary.get("failed") or summary.get("not_run") or len(successful) != 1:
            _emit_batch(batch_dir, summary)
        run_dir = Path(str(successful[0]["run_dir"]))
        report = inspect_run_product(run_dir, app.schemas)
        rendered = _render_run_summary(run_dir) if args.render_summary else None
        dump(
            {
                "ok": True,
                "demo": "reference",
                "batch_dir": str(batch_dir),
                "report": report,
                **({"rendered_report": rendered} if rendered else {}),
            }
        )
        return True

    if args.cmd == "plan":
        if args.run:
            if args.smoke:
                raise ConfigError("--smoke 不能修改已有 RunSpec；请从 System/Evaluation 配置重新生成计划")
            plan = app.plan(args.run)
        else:
            plan, _ = app.user_matrix_plan(
                args.system_config,
                args.evaluation_config,
                smoke=args.smoke,
            )
        if args.output:
            atomic_json(args.output, plan)
        dump(plan)
        return True

    if args.cmd == "run":
        if args.run:
            if args.smoke:
                raise ConfigError("--smoke 不能修改已有 RunSpec；请从 System/Evaluation 配置重新生成计划")
            plan = app.plan(args.run)
            orchestrator = app.orchestrator(
                results_root=args.results_root,
                cache_root=args.cache_root,
            )
            run_dir = orchestrator.execute(plan)
            if args.render_summary:
                dump(
                    {
                        "run_dir": str(run_dir),
                        "rendered_report": _render_run_summary(run_dir),
                    }
                )
            else:
                print(run_dir)
            return True
        plan, bundle = app.user_matrix_plan(
            args.system_config,
            args.evaluation_config,
            smoke=args.smoke,
        )
        executor = app.matrix_executor(
            results_root=args.results_root or bundle.results_root,
            cache_root=args.cache_root or bundle.cache_root,
        )
        path, summary = executor.execute(plan)
        reports = _render_batch_summaries(path) if args.render_summary else None
        _emit_batch(path, summary, reports)
        return True

    if args.cmd == "run-plan":
        raw = json_loads_strict(Path(args.plan).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "matrix_id" in raw:
            plan = app.load_matrix_plan(args.plan)
            executor = _matrix_executor(app, args, plan)
            path, summary = executor.execute(
                plan,
                continue_on_error=args.continue_on_error,
                resume_dir=args.resume_dir,
            )
            reports = _render_batch_summaries(path) if args.render_summary else None
            _emit_batch(path, summary, reports)
            return True
        plan = app.load_plan(args.plan)
        orchestrator = app.orchestrator(
            results_root=args.results_root,
            cache_root=args.cache_root,
        )
        run_dir = orchestrator.execute(plan)
        if args.render_summary:
            dump(
                {
                    "run_dir": str(run_dir),
                    "rendered_report": _render_run_summary(run_dir),
                }
            )
        else:
            print(run_dir)
        return True
    return False


def _handle_matrix(args: argparse.Namespace, app: Application) -> bool:
    if args.cmd == "matrix-validate":
        dump({"ok": True, "matrix": app.matrices.load(args.matrix)})
        return True
    if args.cmd == "matrix-expand":
        runs = app.matrix_expand(args.matrix)
        payload = {"runs": runs, "count": len(runs)}
        if args.output:
            atomic_json(args.output, payload)
        dump(payload)
        return True
    if args.cmd == "matrix-plan":
        plan = app.matrix_plan(args.matrix)
        if args.output:
            atomic_json(args.output, plan)
        dump(plan)
        return True
    if args.cmd == "matrix-export":
        plan = app.load_matrix_plan(args.plan)
        dump(
            app.export_matrix_plan(
                plan,
                args.output,
                shards=args.shards,
                strategy=args.strategy,
            )
        )
        return True
    if args.cmd == "matrix-run":
        plan = app.matrix_plan(args.matrix)
        executor = app.matrix_executor(
            results_root=args.results_root,
            cache_root=args.cache_root,
        )
        path, summary = executor.execute(
            plan,
            continue_on_error=args.continue_on_error,
        )
        reports = _render_batch_summaries(path) if args.render_summary else None
        _emit_batch(path, summary, reports)
        return True
    if args.cmd == "run-matrix-plan":
        plan = app.load_matrix_plan(args.plan)
        executor = _matrix_executor(app, args, plan)
        path, summary = executor.execute(
            plan,
            continue_on_error=args.continue_on_error,
            resume_dir=args.resume_dir,
        )
        reports = _render_batch_summaries(path) if args.render_summary else None
        _emit_batch(path, summary, reports)
        return True
    return False
