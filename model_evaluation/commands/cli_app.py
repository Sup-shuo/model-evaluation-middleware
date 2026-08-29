from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from model_evaluation import package_root
from model_evaluation.commands.adapters import check_adapter_root
from model_evaluation.commands.evaluate import run_automatic_evaluation
from model_evaluation.commands.doctor import (
    EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS,
    run_doctor,
)
from model_evaluation.commands.render import render_inspection
from model_evaluation.commands.workflow import run_check
from model_evaluation.commands.configuration import handle_config_command
from model_evaluation.commands.parser import build_parser
from model_evaluation.core.app import Application
from model_evaluation.core.config.deployment import resolve_deployment_profile
from model_evaluation.core.config.evaluation import resolve_evaluation_profile
from model_evaluation.core.errors import ConfigError, ModelEvalError
from model_evaluation.core.files import atomic_json
from model_evaluation.core.result_product import inspect_run_product
from model_evaluation.core.batch_product import inspect_batch_product
from model_evaluation.environment_snapshot import (
    controller_environment_snapshot,
    requirements_lock_text,
)
from model_evaluation.onboarding import HARDWARE_TEMPLATES, initialize_project


from model_evaluation.commands.execution import (
    _handle_matrix,
    _handle_plan_or_run,
    _render_batch_summaries as _execution_render_batch_summaries,
    _render_run_summary as _execution_render_run_summary,
)
from model_evaluation.commands.output import doctor_dump, dump


def _render_run_summary(path: str | Path) -> dict[str, str]:
    return _execution_render_run_summary(path)


def _render_batch_summaries(batch_dir: Path) -> list[dict[str, str]]:
    return _execution_render_batch_summaries(
        batch_dir,
        render_run=_render_run_summary,
    )


def _project_root() -> Path:
    configured = os.environ.get("MODEL_EVAL_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def _prompt_value(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except EOFError as exc:
        raise ConfigError("guided init requires an interactive terminal") from exc
    selected = value or (default or "")
    if not selected:
        raise ConfigError(f"guided init requires a value for {label}")
    if "REPLACE_WITH" in selected:
        raise ConfigError(f"guided init does not accept placeholder values for {label}")
    return selected


def _parse_devices(value: str) -> tuple[int | str, ...]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ConfigError("device IDs must be a non-empty comma-separated list")
    devices: tuple[int | str, ...] = tuple(
        int(item) if item.isdecimal() else item for item in items
    )
    if len(set(devices)) != len(devices):
        raise ConfigError("device IDs must be unique")
    return devices


def _guided_init_values(args: argparse.Namespace) -> dict[str, object]:
    choices = ", ".join(sorted(HARDWARE_TEMPLATES))
    hardware = _prompt_value(f"Hardware ({choices})", args.hardware)
    if hardware not in HARDWARE_TEMPLATES:
        raise ConfigError(f"unsupported onboarding hardware template: {hardware}")
    template = HARDWARE_TEMPLATES[hardware]
    print(f"Validation status: {template.validation_status}")
    print(
        "Template stack: vLLM + lm-evaluation-harness; backend and evaluator "
        "use the current Controller Python environment. Edit System profiles "
        "when isolated environments are required."
    )
    devices = (
        ()
        if hardware == "cpu"
        else _parse_devices(_prompt_value("Device IDs (comma-separated)", args.devices))
    )
    return {
        "hardware": hardware,
        "devices": devices,
        "model_id": _prompt_value("Model ID", args.model_id),
        "models_root": _prompt_value("Models root", args.models_root),
        "model_ref": _prompt_value("Model path relative to models root"),
        "evaluator_root": _prompt_value("lm-evaluation-harness checkout"),
        "cache_root": _prompt_value("Dataset/cache directory", "cache"),
        "results_root": _prompt_value("Results directory", args.results_root),
        "benchmark": _prompt_value("Benchmark", args.benchmark),
        "backend_executable": _prompt_value(
            "Inference executable", args.backend_executable
        ),
    }


def _handle_bootstrap(args: argparse.Namespace) -> bool:
    if args.cmd == "init":
        values = (
            _guided_init_values(args)
            if args.guided
            else {
                "hardware": args.hardware,
                "devices": () if args.hardware == "cpu" else _parse_devices(args.devices),
                "model_id": args.model_id,
                "model_ref": args.model_ref,
                "models_root": args.models_root,
                "evaluator_root": args.lm_eval_root,
                "cache_root": args.cache_root,
                "results_root": args.results_root,
                "benchmark": args.benchmark,
                "backend_executable": args.backend_executable,
            }
        )
        created = initialize_project(args.path, **values)
        project_root = Path(args.path).expanduser().resolve()
        dump(
            {
                "ok": True,
                "root": str(project_root),
                "created": [str(path) for path in created],
                "next_steps": [
                    f"cd {project_root}",
                    "eval-manager evaluate --smoke",
                    "eval-manager evaluate",
                ],
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
        product_root = Path(args.run_dir).expanduser()
        if (product_root / "summary.json").is_file() and (
            product_root / "runs.json"
        ).is_file():
            report = inspect_batch_product(
                product_root,
                matrix_schemas=app.matrix_schemas,
                run_schemas=app.schemas,
            )
        else:
            report = inspect_run_product(product_root, app.schemas)
        if args.cmd == "result-check" or args.format == "json":
            dump(report)
        else:
            print(render_inspection(report), end="")
        return True
    return False


def _handle_validate(args: argparse.Namespace, app: Application) -> None:
    if args.run:
        if args.smoke:
            raise ConfigError("--smoke 不能修改已有 RunSpec；请从 System/Evaluation 配置重新生成计划")
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

    bundle = app.load_user_config(
        args.system_config,
        args.evaluation_config,
        smoke=args.smoke,
    )
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


def _main() -> None:
    args = build_parser().parse_args()
    if _handle_bootstrap(args):
        return

    app = Application(package_root(), project_root=_project_root())
    if handle_config_command(args, app):
        return
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
            smoke=args.smoke,
        )
        raise SystemExit(0 if ok else 2)
    if args.cmd in {"check", "explain"}:
        ok = run_check(
            app,
            system_config=args.system_config,
            evaluation_config=args.evaluation_config,
            output_format=args.format,
            explain=args.cmd == "explain",
            smoke=args.smoke,
        )
        raise SystemExit(0 if ok else 2)
    if args.cmd == "evaluate":
        raise SystemExit(
            run_automatic_evaluation(
                app,
                system_config=args.system_config,
                evaluation_config=args.evaluation_config,
                results_root=args.results_root,
                cache_root=args.cache_root,
                smoke=args.smoke,
                output_format=args.format,
            )
        )
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
