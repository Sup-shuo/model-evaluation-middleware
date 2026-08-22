from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from model_evaluation.core.errors import ResultProductError
from model_evaluation.core.result_product import inspect_run_product
from model_evaluation.core.result_relocation import load_result_relocation
from model_evaluation.core.serialization import json_loads_strict


BATCH_PRODUCT_FILES = (
    "summary.json",
    "runs.json",
    "metrics.tsv",
    "group_metrics.tsv",
    "task_metrics.tsv",
)


def _load(path: Path):
    if not path.is_file():
        raise ResultProductError(f"batch product file is missing: {path.name}")
    try:
        return json_loads_strict(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResultProductError(f"cannot parse batch product file {path.name}: {exc}") from exc


def _tsv(values) -> str:
    cells = []
    for value in values:
        text = str("" if value is None else value)
        cells.append(text.replace("\t", " ").replace("\r", " ").replace("\n", " "))
    return "\t".join(cells)


def _breakdown(result: dict, name: str) -> dict:
    direct = result.get(name)
    if isinstance(direct, dict):
        return direct
    nested = (result.get("breakdowns") or {}).get(name)
    return nested if isinstance(nested, dict) else {}


def _append_breakdown_metrics(
    lines: list[str], *, kind: str, row: dict, result: dict
) -> None:
    for item_id, detail in sorted(_breakdown(result, kind).items()):
        if not isinstance(detail, dict):
            continue
        sample = detail.get("sample_count") or {}
        subtasks = detail.get("subtasks") or []
        config = detail.get("config")
        config_json = (
            json.dumps(
                config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if isinstance(config, dict)
            else ""
        )
        for namespace, key in (
            ("framework_native", "metrics"),
            ("canonical", "canonical_metrics"),
        ):
            table = detail.get(key) or {}
            if not isinstance(table, dict):
                continue
            for metric, entry in sorted(table.items()):
                metric_entry = entry if isinstance(entry, dict) else {"value": entry}
                lines.append(
                    _tsv(
                        (
                            row.get("model_id", ""),
                            row.get("model_label", ""),
                            row.get("model_ref", ""),
                            result.get("benchmark", ""),
                            result.get("framework", ""),
                            item_id,
                            detail.get("label", ""),
                            namespace,
                            metric,
                            metric_entry.get("value", ""),
                            metric_entry.get("stderr", ""),
                            metric_entry.get("higher_is_better", ""),
                            sample.get("original", ""),
                            sample.get("effective", ""),
                            detail.get("num_fewshot", ""),
                            detail.get("version", ""),
                            ",".join(str(value) for value in subtasks),
                            config_json,
                            row.get("run_dir", ""),
                        )
                    )
                )


def render_batch_tables(products: list[tuple[dict, dict]]) -> dict[str, str]:
    metric_lines = [
        "model_id\tmodel_label\tmodel_ref\tbenchmark\tframework\tmetric\t"
        "value\tstderr\thigher_is_better\trun_dir"
    ]
    detail_header = (
        "model_id\tmodel_label\tmodel_ref\tbenchmark\tframework\t{kind}_id\t"
        "{kind}_label\tmetric_namespace\tmetric\tvalue\tstderr\t"
        "higher_is_better\tsample_original\tsample_effective\tnum_fewshot\t"
        "version\tsubtasks\tconfig_json\trun_dir"
    )
    group_lines = [detail_header.format(kind="group")]
    task_lines = [detail_header.format(kind="task")]
    for row, result in products:
        for metric, entry in sorted((result.get("metrics") or {}).items()):
            metric_entry = entry if isinstance(entry, dict) else {"value": entry}
            metric_lines.append(
                _tsv(
                    (
                        row.get("model_id", ""),
                        row.get("model_label", ""),
                        row.get("model_ref", ""),
                        result.get("benchmark", ""),
                        result.get("framework", ""),
                        metric,
                        metric_entry.get("value", ""),
                        metric_entry.get("stderr", ""),
                        metric_entry.get("higher_is_better", ""),
                        row.get("run_dir", ""),
                    )
                )
            )
        _append_breakdown_metrics(group_lines, kind="groups", row=row, result=result)
        _append_breakdown_metrics(task_lines, kind="tasks", row=row, result=result)
    return {
        "metrics.tsv": "\n".join(metric_lines) + "\n",
        "group_metrics.tsv": "\n".join(group_lines) + "\n",
        "task_metrics.tsv": "\n".join(task_lines) + "\n",
    }


def _require_table(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ResultProductError(f"batch product file is missing: {path.name}")
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResultProductError(f"cannot read batch table {path.name}: {exc}") from exc
    if actual != expected:
        raise ResultProductError(
            f"batch table contents disagree with referenced run products: {path.name}"
        )


def inspect_batch_product(
    batch_dir: str | Path,
    *,
    matrix_schemas,
    run_schemas=None,
) -> dict:
    """Validate one public Matrix result and its referenced successful runs."""

    supplied = Path(batch_dir).expanduser()
    if supplied.is_symlink():
        raise ResultProductError(f"batch directory may not be a symlink: {supplied}")
    root = supplied.resolve()
    if not root.is_dir():
        raise ResultProductError(f"batch directory not found: {root}")

    summary = _load(root / "summary.json")
    runs = _load(root / "runs.json")
    matrix_schemas.validate("matrix_batch_summary", summary)
    matrix_schemas.validate("matrix_batch_runs", runs)
    if summary["batch_id"] != root.name:
        raise ResultProductError("batch summary batch_id disagrees with directory name")

    statuses = Counter(str(row["status"]) for row in runs)
    expected_counts = {
        "planned": len(runs),
        "success": statuses["success"],
        "failed": statuses["failed"],
        "interrupted": statuses["interrupted"],
        "not_run": statuses["not_run"],
    }
    for name, value in expected_counts.items():
        if summary[name] != value:
            raise ResultProductError(
                f"batch summary {name} disagrees with runs.json: "
                f"{summary[name]} != {value}"
            )
    expected_outcome = (
        "interrupted"
        if statuses["interrupted"]
        else "success"
        if statuses["success"] == len(runs)
        else "failed"
    )
    if summary["outcome"] != expected_outcome:
        raise ResultProductError(
            "batch summary outcome disagrees with runs.json: "
            f"{summary['outcome']!r} != {expected_outcome!r}"
        )
    indices = [int(row["index"]) for row in runs]
    if sorted(indices) != list(range(1, len(runs) + 1)):
        raise ResultProductError("batch run indices must be contiguous and one-based")
    plan_ids = [str(row["plan_id"]) for row in runs]
    if len(plan_ids) != len(set(plan_ids)):
        raise ResultProductError("batch runs contain duplicate plan_id values")

    checked_runs = 0
    products = []
    results_root = root.parent.parent if root.parent.name == "_batches" else root.parent
    relocation = load_result_relocation(results_root)
    for row in runs:
        if row["status"] != "success":
            continue
        relocated = relocation.relocate(str(row["run_dir"]), label="batch run_dir")
        if run_schemas is not None:
            report = inspect_run_product(relocated, run_schemas)
            if report["outcome"] != "success":
                raise ResultProductError(
                    f"batch success row references a non-success run: {relocated}"
                )
            if report["plan_id"] != row["plan_id"]:
                raise ResultProductError(
                    "batch success row plan_id disagrees with run_config.json"
                )
            checked_runs += 1
        result = _load(relocated / "result.json")
        products.append((row, result))

    expected_tables = render_batch_tables(products)
    for name, expected in expected_tables.items():
        _require_table(root / name, expected)

    return {
        "ok": True,
        "schema_version": "1.0",
        "product": "matrix_batch",
        "batch_dir": str(root),
        **summary,
        "checked_runs": checked_runs,
    }


__all__ = ["BATCH_PRODUCT_FILES", "inspect_batch_product", "render_batch_tables"]
