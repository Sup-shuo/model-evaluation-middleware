from __future__ import annotations

import argparse

from model_evaluation.commands.configuration import add_config_commands
from model_evaluation.onboarding import HARDWARE_TEMPLATES


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
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="临时启用标准 Smoke 模式：每个任务只评测 1 个样本，不修改 YAML",
    )


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


def _add_render_summary_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--render-summary",
        action="store_true",
        help=(
            "为每个成功 run 生成 result-summary.txt 和 "
            "result-summary.svg；正式 JSON 结果保持不变"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-manager",
        description="模型评测适配中间层 v4.1",
    )
    commands = parser.add_subparsers(dest="cmd", required=True)

    add_config_commands(commands)

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
    init_parser.add_argument(
        "--guided",
        action="store_true",
        help="交互填写最小 vLLM + lm-eval 配置；不安装环境或下载模型",
    )
    init_parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated device IDs; unused by the CPU template",
    )
    init_parser.add_argument("--model-id", default="example-model")
    init_parser.add_argument(
        "--model-ref",
        default="REPLACE_WITH_MODEL_RELATIVE_PATH",
        help="相对于 models root 的本地模型路径",
    )
    init_parser.add_argument("--models-root", default="/REPLACE_WITH_MODEL_ROOT")
    init_parser.add_argument(
        "--lm-eval-root",
        default="/REPLACE_WITH_LM_EVALUATION_HARNESS",
    )
    init_parser.add_argument("--cache-root", default="/REPLACE_WITH_CACHE_ROOT")
    init_parser.add_argument("--results-root", default="results")
    init_parser.add_argument("--benchmark", default="bbh")
    init_parser.add_argument("--backend-executable", default="vllm")

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
    _add_render_summary_arg(demo)

    adapter_check = commands.add_parser(
        "adapter-check",
        help="验证一个外部 Adapter root 的目录、Manifest 与协议版本",
    )
    adapter_check.add_argument("root")

    result_check = commands.add_parser(
        "result-check",
        help="验证最终结果 Schema、跨文件一致性与产物路径",
    )
    result_check.add_argument("run_dir", help="单次 run 或 Matrix batch 结果目录")

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

    evaluate = commands.add_parser(
        "evaluate",
        help="小白入口：预检通过后自动启动推理、执行评测、保存并展示结果",
    )
    add_user_config_args(evaluate)
    _add_execution_overrides(evaluate)
    evaluate.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
    )

    run = commands.add_parser(
        "run",
        help="不带 RunSpec 时直接运行 config/evaluation.yaml",
    )
    run.add_argument("run", nargs="?")
    _add_execution_overrides(run)
    add_user_config_args(run)
    _add_render_summary_arg(run)

    run_plan = commands.add_parser("run-plan")
    run_plan.add_argument("plan")
    _add_resume_args(run_plan)
    _add_render_summary_arg(run_plan)

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
    matrix_export.add_argument(
        "--strategy",
        choices=("resource_balanced", "round_robin"),
        default="resource_balanced",
        help="按逻辑加速卡需求均衡分片，或保留简单轮询",
    )

    matrix_run = commands.add_parser("matrix-run")
    matrix_run.add_argument("matrix")
    _add_execution_overrides(matrix_run)
    matrix_run.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    _add_render_summary_arg(matrix_run)

    saved_matrix = commands.add_parser("run-matrix-plan")
    saved_matrix.add_argument("plan")
    _add_resume_args(saved_matrix)
    _add_render_summary_arg(saved_matrix)
    return parser
