"""Human-readable views over a completed run product.

The text and SVG files produced here are derived presentation artifacts.  The
JSON result product remains the stable machine-readable interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

from model_evaluation.core.files import atomic_text


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required result file is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read result file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def metric_value(entry: object) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get("value"), (int, float)):
        return "N/A"
    return f"{float(entry['value']):.6f}"


def first_metric(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metrics = row.get("metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return "N/A", {}
    name = sorted(metrics)[0]
    value = metrics[name]
    return name, value if isinstance(value, dict) else {}


def short_task(value: str) -> str:
    for prefix in ("leaderboard_bbh_local_", "leaderboard_bbh_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def environment_lines(
    run_dir: Path,
    run_config: dict[str, Any],
    versions: dict[str, Any],
) -> list[str]:
    device = versions.get("device") or {}
    devices = device.get("devices") or []
    runtime = versions.get("runtime") or {}
    backend = versions.get("backend") or {}
    evaluator = versions.get("evaluator") or {}
    evaluator_facts = evaluator.get("facts") or {}
    backend_parameters = ((run_config.get("backend") or {}).get("parameters") or {})
    model_location = ((run_config.get("backend") or {}).get("model_location") or {})
    cache_root = evaluator_facts.get("cache_root") or "N/A"
    port = backend_parameters.get("port", 8091)
    project_root = run_dir.parent.parent
    device_names = ", ".join(
        str(item.get("name") or item.get("id")) for item in devices
    ) or "N/A"
    backend_version = next(
        (
            str(item.get("version"))
            for item in backend.get("probes") or []
            if item.get("id") == "backend.import" and item.get("version")
        ),
        str(backend.get("adapter_version") or "N/A"),
    )
    return [
        "Environment",
        f"  Hardware       : {device.get('vendor', 'N/A')} / {device_names}",
        f"  GPU count      : {len(devices)}",
        f"  Runtime        : {runtime.get('family', 'N/A')} {runtime.get('version', 'N/A')}",
        f"  Driver         : {runtime.get('driver_version', 'N/A')}",
        f"  Backend        : {backend.get('adapter', 'N/A')} {backend_version}",
        f"  Executable     : {backend_parameters.get('executable', 'vllm')}",
        f"  Completions API: http://127.0.0.1:{port}/v1/completions",
        f"  Evaluator      : {evaluator_facts.get('framework', evaluator.get('adapter', 'N/A'))} "
        f"{evaluator_facts.get('framework_version', evaluator.get('adapter_version', 'N/A'))}",
        f"  Transformers   : {(evaluator_facts.get('packages') or {}).get('transformers', 'N/A')}",
        f"  Project        : {project_root}",
        f"  Harness        : {evaluator_facts.get('framework_file', 'N/A')}",
        f"  Dataset cache  : {cache_root}",
        f"  Model path     : {model_location.get('local_path', 'N/A')}",
    ]


def render_lines(run_dir: Path) -> list[str]:
    run_dir = run_dir.resolve()
    metrics = load_object(run_dir / "metrics.json")
    result = load_object(run_dir / "result.json")
    terminal = load_object(run_dir / "terminal.json")
    run_config = load_object(run_dir / "config" / "run_config.json")
    versions = load_object(run_dir / "config" / "runtime_versions.json")

    if terminal.get("outcome") != "success":
        raise ValueError(f"summary rendering requires a successful run: {run_dir}")

    lines = [
        f"Run:       {result.get('run_id', run_dir.name)}",
        f"Outcome:   {terminal.get('outcome', 'unknown')}",
        f"Model:     {result.get('model', metrics.get('model', 'N/A'))}",
        f"Benchmark: {result.get('benchmark', metrics.get('benchmark', 'N/A'))}",
        f"Framework: {result.get('framework', metrics.get('framework', 'N/A'))}",
        f"Started:   {terminal.get('started_at', 'N/A')}",
        f"Finished:  {terminal.get('finished_at', 'N/A')}",
        "",
        *environment_lines(run_dir, run_config, versions),
        "",
    ]

    tasks = metrics.get("tasks") or {}
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for task_id, raw_row in sorted(tasks.items()):
        row = raw_row if isinstance(raw_row, dict) else {}
        metric_name, metric = first_metric(row)
        sample_count = row.get("sample_count") or {}
        stderr = metric.get("stderr")
        rows.append(
            (
                short_task(str(task_id)),
                str(row.get("version", "N/A")),
                str(row.get("num_fewshot", "N/A")),
                metric_name,
                metric_value(metric),
                "N/A" if not isinstance(stderr, (int, float)) else f"{float(stderr):.6f}",
                str(sample_count.get("effective", "N/A")),
            )
        )

    headers = ("Task", "Version", "n-shot", "Metric", "Value", "Stderr", "Samples")
    widths = [len(value) for value in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def table_row(values: tuple[str, ...]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(values, widths)
        ) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines.extend([separator, table_row(headers), separator])
    lines.extend(table_row(row) for row in rows)
    lines.append(separator)

    summary = metrics.get("summary") or {}
    summary_name = sorted(summary)[0] if summary else "N/A"
    summary_metric = summary.get(summary_name) if summary else {}
    effective = sum(
        int(((row if isinstance(row, dict) else {}).get("sample_count") or {}).get("effective") or 0)
        for row in tasks.values()
    )
    lines.extend(
        [
            f"Summary: {summary_name}={metric_value(summary_metric)}  tasks={len(rows)}  samples={effective}",
            "Source: result.json + metrics.json + terminal.json + config/*.json",
            "Note: this view is for human verification; native JSON remains the authoritative saved output.",
        ]
    )
    return lines


def svg_text(lines: list[str]) -> str:
    char_width = 8.2
    line_height = 19
    width = max(960, int(max((len(line) for line in lines), default=1) * char_width + 48))
    height = max(240, len(lines) * line_height + 48)
    text = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        '<style>text{font-family:Menlo,Consolas,monospace;font-size:14px;white-space:pre}</style>',
    ]
    for index, line in enumerate(lines):
        color = "#86efac" if line.startswith("Summary:") else "#e5e7eb"
        text.append(
            f'<text x="24" y="{32 + index * line_height}" fill="{color}">{escape(line)}</text>'
        )
    text.append("</svg>")
    return "\n".join(text) + "\n"


def write_svg(lines: list[str], path: Path) -> None:
    atomic_text(path, svg_text(lines))


def write_run_report(run_dir: str | Path) -> dict[str, str]:
    run = Path(run_dir).resolve()
    lines = render_lines(run)
    rendered = "\n".join(lines) + "\n"
    text_path = run / "result-summary.txt"
    svg_path = run / "result-summary.svg"
    atomic_text(text_path, rendered)
    write_svg(lines, svg_path)
    return {"run_dir": str(run), "text": str(text_path), "svg": str(svg_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--svg", type=Path, help="also save a terminal-like SVG snapshot")
    parser.add_argument("--text", type=Path, help="also save the rendered plain-text report")
    args = parser.parse_args(argv)
    lines = render_lines(args.run_dir)
    output = "\n".join(lines) + "\n"
    sys.stdout.write(output)
    if args.text:
        atomic_text(args.text, output)
    if args.svg:
        write_svg(lines, args.svg)
    return 0


__all__ = ["render_lines", "svg_text", "write_run_report", "write_svg"]
