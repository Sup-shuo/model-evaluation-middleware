from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from model_evaluation import package_root
from model_evaluation.commands.adapters import check_adapter_root
from model_evaluation.commands.doctor import (
    EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS,
    run_doctor,
)
from model_evaluation.commands.render import render_inspection
from model_evaluation.commands.workflow import run_check
from model_evaluation.core.app import Application
from model_evaluation.core.config.deployment import resolve_deployment_profile
from model_evaluation.core.config.evaluation import resolve_evaluation_profile
from model_evaluation.core.errors import ModelEvalError
from model_evaluation.core.files import atomic_json
from model_evaluation.core.result_product import inspect_run_product
from model_evaluation.core.security import redact_diagnostic
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.environment_snapshot import (
    controller_environment_snapshot,
    requirements_lock_text,
)
from model_evaluation.onboarding import HARDWARE_TEMPLATES, initialize_project


def dump(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def doctor_dump(obj, orchestrator) -> None:
    """Backward-compatible JSON renderer used by integrations and tests."""
    dump(redact_diagnostic(obj, orchestrator.pm.secrets.redaction_values()))


def add_user_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--system-config",
        default=None,
        help=(
            "机器配置路径或 config/systems/ 下的 ID；默认 "
            "MODEL_EVAL_SYSTEM_CONFIG 或 config/system.yaml"
        ),
    )
    parser.add_argument(
        "--evaluation-config",
        default=None,
        help=(
            "评测配置路径或 config/evaluations/ 下的 ID；默认 "
            "MODEL_EVAL_EVALUATION_CONFIG 或 config/evaluation.yaml"
        ),
    )


def _project_root() -> Path:
    configured = os.environ.get("MODEL_EVAL_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def _add_execution_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root")
    parser.add_argument("--cache-root")


def _add_resume_args(parser: argparse.ArgumentParser) -> None:
    _add_execution_overrides(parser)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--resume-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-manager",
        description="模型评测适配中间层 v4.1",
    )
    commands = parser.add_subparsers(dest="cmd", required=True)

    init_parser = commands.add_parser(
        "init",
        help="创建最小 System/Model/Evaluation 工程骨架；不覆盖现有文件",
    )
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument(
        "--hardware",
        choices=sorted(HARDWARE_TEMPLATES),
        default="nvidia",
    )

    snapshot = commands.add_parser(
        "environment-snapshot",
        help="导出运行 Core 的 Python 环境和可选 requirements lock",
    )
    snapshot.add_argument("-o", "--output")
    snapshot.add_argument("--requirements-lock")

    commands.add_parser("schema-check")
    commands.add_parser("adapters")

    demo = commands.add_parser(
        "demo",
        help="运行无需 GPU/NPU 的 reference E2E，并输出最终 JSON 报告",
    )
    demo.add_argument("--results-root")
    demo.add_argument("--cache-root")

    adapter_check = commands.add_parser(
        "adapter-check",
        help="验证一个外部 Adapter root 的目录、Manifest 与协议版本",
    )
    adapter_check.add_argument("root")

    result_check = commands.add_parser(
        "result-check",
        help="验证最终结果 Schema、跨文件一致性与产物路径",
    )
    result_check.add_argument("run_dir")

    inspect_parser = commands.add_parser(
        "inspect",
        help="验证并展示一个完成的结果目录",
    )
    inspect_parser.add_argument("run_dir")
    inspect_parser.add_argument("--format", choices=("human", "json"), default="human")

    validate = commands.add_parser(
        "validate",
        help="不带参数时验证 config/system.yaml + config/evaluation.yaml",
    )
    validate.add_argument("run", nargs="?")
    add_user_config_args(validate)

    doctor = commands.add_parser(
        "doctor",
        help="检查当前机器、所选环境与评测框架是否具备本地运行条件（不启动模型服务）",
    )
    add_user_config_args(doctor)
    doctor.add_argument("--format", choices=("human", "json"), default="human")

    check = commands.add_parser(
        "check",
        help="组合 validate、doctor、plan preview 与只读资源检查（不启动模型服务）",
    )
    add_user_config_args(check)
    check.add_argument("--format", choices=("human", "json"), default="human")

    explain = commands.add_parser(
        "explain",
        help="解释所选模型、后端与机器组合为何可运行或被阻止",
    )
    add_user_config_args(explain)
    explain.add_argument("--format", choices=("human", "json"), default="human")

    plan = commands.add_parser(
        "plan",
        help="不带 RunSpec 时从两份用户配置生成批量计划",
    )
    plan.add_argument("run", nargs="?")
    plan.add_argument("-o", "--output")
    add_user_config_args(plan)

    run = commands.add_parser(
        "run",
        help="不带 RunSpec 时直接运行 config/evaluation.yaml",
    )
    run.add_argument("run", nargs="?")
    _add_execution_overrides(run)
    add_user_config_args(run)

    run_plan = commands.add_parser("run-plan")
    run_plan.add_argument("plan")
    _add_resume_args(run_plan)

    matrix_validate = commands.add_parser("matrix-validate")
    matrix_validate.add_argument("matrix")

    matrix_expand = commands.add_parser("matrix-expand")
    matrix_expand.add_argument("matrix")
    matrix_expand.add_argument("-o", "--output")

    matrix_plan = commands.add_parser("matrix-plan")
    matrix_plan.add_argument("matrix")
    matrix_plan.add_argument("-o", "--output")

    matrix_export = commands.add_parser(
        "matrix-export",
        help="把已保存的 Matrix plan 分片为调度器无关的 execution-plan bundle",
    )
    matrix_export.add_argument("plan")
    matrix_export.add_argument("-o", "--output", required=True)
    matrix_export.add_argument("--shards", type=int, default=1)

    matrix_run = commands.add_parser("matrix-run")
    matrix_run.add_argument("matrix")
    _add_execution_overrides(matrix_run)
    matrix_run.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    saved_matrix = commands.add_parser("run-matrix-plan")
    saved_matrix.add_argument("plan")
    _add_resume_args(saved_matrix)
    return parser


def _handle_bootstrap(args: argparse.Namespace) -> bool:
    if args.cmd == "init":
        created = initialize_project(args.path, hardware=args.hardware)
        dump(
            {
                "ok": True,
                "root": str(Path(args.path).expanduser().resolve()),
                "created": [str(path) for path in created],
            }
        )
        return True

    if args.cmd == "environment-snapshot":
        snapshot = controller_environment_snapshot()
        if args.output:
            atomic_json(args.output, snapshot)
        if args.requirements_lock:
            from model_evaluation.core.files import atomic_text

            atomic_text(args.requirements_lock, requirements_lock_text(snapshot))
        dump(snapshot)
        return True
    return False


def _handle_introspection(args: argparse.Namespace, app: Application) -> bool:
    if args.cmd == "schema-check":
        dump({"ok": True, "schemas": app.schemas.validate_all_schemas()})
        return True
    if args.cmd == "adapters":
        app.registry.discover()
        dump([identity.manifest for identity in app.registry.identities()])
        return True
    if args.cmd == "adapter-check":
        dump(check_adapter_root(args.root, app.schemas))
        return True
    if args.cmd in {"result-check", "inspect"}:
        report = inspect_run_product(args.run_dir, app.schemas)
        if args.cmd == "result-check" or args.format == "json":
            dump(report)
        else:
            print(render_inspection(report), end="")
        return True
    return False


def _handle_validate(args: argparse.Namespace, app: Application) -> None:
    if args.run:
        run = app.specs.resolve_run(args.run)
        bundle = app.specs.resolve_bundle(run)
        _, deployment_resolution = resolve_deployment_profile(
            bundle["deployment"],
            bundle["model"],
            bundle["platform"],
            (run.get("overrides") or {}).get("deployment"),
        )
        _, evaluation_resolution = resolve_evaluation_profile(
            bundle["evaluation"],
            bundle["platform"],
        )
        dump(
            {
                "ok": True,
                "mode": "internal-run-spec",
                "run": run,
                "resolved_ids": {
                    key: value.get("id") if isinstance(value, dict) else None
                    for key, value in bundle.items()
                },
                "deployment_resolution": deployment_resolution,
                "evaluation_resolution": evaluation_resolution,
            }
        )
        return

    bundle = app.load_user_config(args.system_config, args.evaluation_config)
    dump(
        {
            "ok": True,
            "mode": "user-config",
            "system": bundle.system["system"]["name"],
            "profiles": bundle.generated.get("selected_profiles", {}),
            "models": list(bundle.generated["model_ids"].values()),
            "benchmarks": bundle.evaluation["benchmarks"],
            "cache_root": bundle.cache_root,
            "results_root": bundle.results_root,
            "generated": bundle.generated,
        }
    )


def _matrix_executor(app: Application, args: argparse.Namespace, plan: dict):
    user_paths = (plan.get("summary") or {}).get("user_config") or {}
    return app.matrix_executor(
        results_root=args.results_root or user_paths.get("results_root"),
        cache_root=args.cache_root or user_paths.get("cache_root"),
    )


def _emit_batch(path: Path, summary: dict) -> None:
    dump({"batch_dir": str(path), "summary": summary})
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
        dump(
            {
                "ok": True,
                "demo": "reference",
                "batch_dir": str(batch_dir),
                "report": report,
            }
        )
        return True

    if args.cmd == "plan":
        if args.run:
            plan = app.plan(args.run)
        else:
            plan, _ = app.user_matrix_plan(
                args.system_config,
                args.evaluation_config,
            )
        if args.output:
            atomic_json(args.output, plan)
        dump(plan)
        return True

    if args.cmd == "run":
        if args.run:
            plan = app.plan(args.run)
            orchestrator = app.orchestrator(
                results_root=args.results_root,
                cache_root=args.cache_root,
            )
            print(orchestrator.execute(plan))
            return True
        plan, bundle = app.user_matrix_plan(
            args.system_config,
            args.evaluation_config,
        )
        executor = app.matrix_executor(
            results_root=args.results_root or bundle.results_root,
            cache_root=args.cache_root or bundle.cache_root,
        )
        path, summary = executor.execute(plan)
        _emit_batch(path, summary)
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
            _emit_batch(path, summary)
            return True
        plan = app.load_plan(args.plan)
        orchestrator = app.orchestrator(
            results_root=args.results_root,
            cache_root=args.cache_root,
        )
        print(orchestrator.execute(plan))
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
        dump(app.export_matrix_plan(plan, args.output, shards=args.shards))
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
        _emit_batch(path, summary)
        return True
    if args.cmd == "run-matrix-plan":
        plan = app.load_matrix_plan(args.plan)
        executor = _matrix_executor(app, args, plan)
        path, summary = executor.execute(
            plan,
            continue_on_error=args.continue_on_error,
            resume_dir=args.resume_dir,
        )
        _emit_batch(path, summary)
        return True
    return False


def _main() -> None:
    args = build_parser().parse_args()
    if _handle_bootstrap(args):
        return

    app = Application(package_root(), project_root=_project_root())
    if _handle_introspection(args, app):
        return
    if args.cmd == "validate":
        _handle_validate(args, app)
        return
    if args.cmd == "doctor":
        ok = run_doctor(
            app,
            system_config=args.system_config,
            evaluation_config=args.evaluation_config,
            output_format=args.format,
        )
        raise SystemExit(0 if ok else 2)
    if args.cmd in {"check", "explain"}:
        ok = run_check(
            app,
            system_config=args.system_config,
            evaluation_config=args.evaluation_config,
            output_format=args.format,
            explain=args.cmd == "explain",
        )
        raise SystemExit(0 if ok else 2)
    if _handle_plan_or_run(args, app):
        return
    if _handle_matrix(args, app):
        return
    raise AssertionError(f"unhandled command: {args.cmd}")


def main() -> None:
    try:
        _main()
    except ModelEvalError as exc:
        print(f"{getattr(exc, 'code', 'MODEL_EVAL_ERROR')}: {exc}", file=sys.stderr)
        details = getattr(exc, "details", {}) or {}
        if details.get("run_dir"):
            print(f"run_dir: {details['run_dir']}", file=sys.stderr)
        raise SystemExit(2) from exc
