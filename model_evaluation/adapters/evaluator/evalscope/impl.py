from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from model_evaluation.sdk.jsonutil import loads as json_loads
from model_evaluation.sdk.manifest import load_manifest
from model_evaluation.sdk.runtime import AdapterError


def requirements(i, c):
    return {
        "schema_version": "1.0",
        "requirements": [
            {
                "path": "service.protocol.openai_chat",
                "op": "equals",
                "value": True,
                "message": "EvalScope openai_api requires an OpenAI-compatible Chat Completions endpoint",
            },
            {
                "path": "service.generation",
                "op": "equals",
                "value": True,
                "message": "EvalScope requires generation capability",
            },
            {
                "path": "evaluation_environment.python",
                "op": "equals",
                "value": True,
                "message": "EvalScope requires a Python-capable evaluation environment",
            },
        ],
    }


def _parameters(evaluation: dict) -> dict:
    value = (evaluation or {}).get("parameters") or {}
    if not isinstance(value, dict):
        raise AdapterError("CONFIG_INVALID", "EvalScope parameters must be an object")
    return value


def _execution_env(cache_root, offline: bool) -> dict:
    env: dict[str, dict] = {"set": {"PYTHONHASHSEED": "0"}}
    if cache_root:
        root = Path(str(cache_root))
        if not root.is_absolute():
            raise AdapterError("CONFIG_INVALID", "EvalScope cache_root must be absolute")
        env["set"]["EVALSCOPE_CACHE"] = str((root / "evalscope").resolve())
        env["set"]["MODELSCOPE_CACHE"] = str((root / "modelscope").resolve())
    if offline:
        env["set"].update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    return env


def plan_preflight(i, c):
    evaluation = i["evaluation"]
    params = _parameters(evaluation)
    payload = {
        "executable": str(params.get("executable") or "evalscope"),
        "expected_version": params.get("expected_version"),
        "task_id": (i.get("task") or {}).get("task_id"),
    }
    process = {
        "schema_version": "1.0",
        "argv": [
            "python",
            str(Path(__file__).with_name("preflight.py").resolve()),
            "--payload",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        ],
        "env_patch": _execution_env(i.get("cache_root") or c.get("cache_root"), bool(c.get("offline", False))),
        "stdin": {"mode": "null"},
        "stdout": {"mode": "capture"},
        "stderr": {"mode": "capture"},
        "timeout_seconds": float(params.get("preflight_timeout_seconds", 60.0)),
        "metadata": {"role": "evaluator_preflight", "framework": "evalscope"},
    }
    return {"process": process, "result_format": "preflight_result"}


def _api_root(chat_url: str) -> str:
    parsed = urlsplit(str(chat_url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AdapterError("CONFIG_INVALID", "EvalScope requires an HTTP(S) Chat Completions URL")
    suffix = "/chat/completions"
    path = parsed.path.rstrip("/")
    if not path.endswith(suffix):
        raise AdapterError(
            "COMPATIBILITY_ERROR",
            f"OpenAI Chat Completions URL must end with {suffix}: {chat_url}",
        )
    root_path = path[: -len(suffix)] or "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, root_path, "", ""))


def _generation_config(params: dict) -> dict:
    return {
        "temperature": float(params.get("temperature", 0.0)),
        "top_p": float(params.get("top_p", 1.0)),
        "top_k": int(params.get("top_k", 1)),
        "max_tokens": int(params.get("max_tokens", 512)),
        "timeout": float(params.get("request_timeout", 300.0)),
        "retries": int(params.get("max_retries", 1)),
        "seed": int(params.get("seed", 42)),
    }


def plan_evaluate(i, c):
    service = i["service"]
    task = i["task"]
    evaluation = i["evaluation"]
    params = _parameters(evaluation)
    auth = service.get("auth") or {"mode": "none"}
    if auth.get("mode") != "none":
        raise AdapterError(
            "COMPATIBILITY_ERROR",
            "EvalScope Adapter 1.0 supports local unauthenticated OpenAI-compatible services",
        )
    protocols = service.get("protocols") or {}
    chat_url = (protocols.get("openai_chat") or {}).get("url")
    if not chat_url:
        raise AdapterError("COMPATIBILITY_ERROR", "service does not expose openai_chat")
    api_root = _api_root(str(chat_url))
    model_id = str((service.get("model") or {}).get("id") or "")
    if not model_id:
        raise AdapterError("CONFIG_INVALID", "service model id is missing")
    task_id = str(task.get("task_id") or "")
    if not task_id:
        raise AdapterError("CONFIG_INVALID", "EvalScope task_id is missing")

    output = Path(i["output_root"]).resolve()
    executable = str(params.get("executable") or "evalscope")
    argv = [
        executable,
        "eval",
        "--model",
        model_id,
        "--model-id",
        model_id,
        "--api-url",
        api_root,
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--datasets",
        task_id,
        "--work-dir",
        str(output),
        "--no-timestamp",
        "--seed",
        str(int(params.get("seed", 42))),
        "--eval-batch-size",
        str(int(params.get("eval_batch_size", 8))),
        "--generation-config",
        json.dumps(_generation_config(params), separators=(",", ":"), sort_keys=True),
    ]
    if params.get("limit") is not None:
        argv += ["--limit", str(params["limit"])]
    dataset_args = (task.get("metadata") or {}).get("evalscope_dataset_args")
    if dataset_args:
        argv += [
            "--dataset-args",
            json.dumps({task_id: dataset_args}, separators=(",", ":"), sort_keys=True),
        ]
    dataset_hub = params.get("dataset_hub")
    if dataset_hub:
        argv += ["--dataset-hub", str(dataset_hub)]
    if bool(params.get("ignore_errors", False)):
        argv.append("--ignore-errors")
    argv.append("--collect-perf" if bool(params.get("collect_perf", False)) else "--no-collect-perf")

    process = {
        "schema_version": "1.0",
        "argv": argv,
        "cwd": str(output),
        "env_patch": _execution_env(
            i.get("cache_root") or c.get("cache_root"),
            i.get("network_policy") == "offline" or bool(c.get("offline", False)),
        ),
        "stdin": {"mode": "null"},
        "stdout": {"mode": "file", "path": str(Path(i.get("log_path") or output / "evaluation.log"))},
        "stderr": {"mode": "merge_stdout"},
        "timeout_seconds": float(params.get("evaluation_timeout_seconds", 86400.0)),
        "metadata": {
            "framework": "evalscope",
            "eval_type": "openai_api",
            "task_id": task_id,
        },
    }
    return {"process": process, "raw_result_root": str(output)}


def _find_report(root: Path, task_id: str) -> tuple[Path, object]:
    report_root = root / "reports"
    files = sorted(path for path in report_root.rglob("*.json") if path.is_file() and not path.is_symlink())
    preferred = [path for path in files if path.stem == task_id]
    candidates = preferred or files
    if len(candidates) != 1:
        raise AdapterError(
            "RESULT_INVALID",
            f"expected one EvalScope report for {task_id!r}, found {[str(path) for path in candidates]}",
        )
    path = candidates[0]
    try:
        return path, json_loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdapterError("RESULT_INVALID", f"cannot read EvalScope report {path}: {exc}") from exc


def _category_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return "/".join(str(item) for item in value if item not in (None, "")) or None
    return str(value)


def _current_report_rows(value: dict) -> tuple[list[dict], list[dict]]:
    """Flatten EvalScope's current nested Report JSON without losing subset rows."""
    metrics = value.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise AdapterError("RESULT_INVALID", "EvalScope report metrics must be a non-empty list")
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    dataset_name = value.get("dataset_name") or value.get("dataset")
    for metric in metrics:
        if not isinstance(metric, dict):
            raise AdapterError("RESULT_INVALID", "EvalScope report metric must be an object")
        name = metric.get("name") or metric.get("metric_name") or metric.get("metric")
        row = {
            "dataset_name": dataset_name,
            "metric_name": name,
            "num": metric.get("num"),
            "score": metric.get("score"),
        }
        summary_rows.append(row)
        categories = metric.get("categories") or []
        if not isinstance(categories, list):
            raise AdapterError("RESULT_INVALID", "EvalScope metric categories must be a list")
        for category in categories:
            if not isinstance(category, dict):
                raise AdapterError("RESULT_INVALID", "EvalScope metric category must be an object")
            category_name = _category_text(category.get("name"))
            subsets = category.get("subsets") or []
            if not isinstance(subsets, list):
                raise AdapterError("RESULT_INVALID", "EvalScope metric subsets must be a list")
            if not subsets:
                detail_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "category_name": category_name,
                        "metric_name": name,
                        "num": category.get("num"),
                        "score": category.get("score"),
                    }
                )
            for subset in subsets:
                if not isinstance(subset, dict):
                    raise AdapterError("RESULT_INVALID", "EvalScope metric subset must be an object")
                detail_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "category_name": category_name,
                        "subset_name": _category_text(subset.get("name")),
                        "metric_name": name,
                        "num": subset.get("num"),
                        "score": subset.get("score"),
                    }
                )
    return summary_rows, detail_rows or summary_rows


def _report_rows(value: object) -> tuple[list[dict], list[dict]]:
    if isinstance(value, dict) and isinstance(value.get("metrics"), list):
        return _current_report_rows(value)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and isinstance(value.get("results"), list):
        rows = value["results"]
    elif isinstance(value, dict) and isinstance(value.get("reports"), list):
        rows = value["reports"]
    elif isinstance(value, dict) and "score" in value:
        rows = [value]
    else:
        raise AdapterError("RESULT_INVALID", "unsupported EvalScope report JSON structure")
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise AdapterError("RESULT_INVALID", "EvalScope report must contain metric row objects")
    return rows, rows


def _metric_entry(row: dict) -> tuple[str, dict]:
    name = row.get("metric_name") or row.get("metric")
    score = row.get("score")
    if not isinstance(name, str) or not name:
        raise AdapterError("RESULT_INVALID", "EvalScope report row is missing metric_name")
    if not isinstance(score, (int, float, str, bool)) and score is not None:
        raise AdapterError("RESULT_INVALID", f"EvalScope metric {name!r} has unsupported score")
    return name, {"value": score}


def _row_id(row: dict, index: int) -> str:
    values = [
        row.get("dataset_name") or row.get("dataset"),
        row.get("category_name") or row.get("category"),
        row.get("subset_name") or row.get("subset"),
    ]
    parts = [_category_text(value) for value in values]
    parts = [value for value in parts if value]
    return "/".join(parts) if parts else f"row-{index}"


def _sample_count(row: dict) -> dict | None:
    value = row.get("num") if "num" in row else row.get("sample_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return {"original": value, "effective": value}
    return None


def _sample_artifacts(root: Path) -> list[dict]:
    artifacts = []
    for directory in (root / "predictions", root / "reviews"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.jsonl")):
            if path.is_file() and not path.is_symlink():
                artifacts.append({"path": str(path.resolve()), "media_type": "application/x-ndjson"})
    return artifacts


def normalize(i, c):
    root = Path(i["raw_result_root"]).resolve()
    task = i["task"]
    task_id = str(task["task_id"])
    report_path, report = _find_report(root, task_id)
    rows, detail_rows = _report_rows(report)
    native_occurrences: dict[str, list[dict]] = {}
    tasks = {}
    for row in rows:
        metric_name, entry = _metric_entry(row)
        native_occurrences.setdefault(metric_name, []).append(entry)
    for index, row in enumerate(detail_rows, 1):
        metric_name, entry = _metric_entry(row)
        row_id = _row_id(row, index)
        detail = tasks.setdefault(row_id, {"metrics": {}})
        if metric_name in detail["metrics"]:
            raise AdapterError(
                "RESULT_INVALID",
                f"duplicate EvalScope metric {metric_name!r} for report row {row_id!r}",
            )
        detail["metrics"][metric_name] = entry
        count = _sample_count(row)
        if count:
            previous = detail.get("sample_count")
            if previous is not None and previous != count:
                raise AdapterError(
                    "RESULT_INVALID",
                    f"inconsistent EvalScope sample count for report row {row_id!r}",
                )
            detail["sample_count"] = count
        config = {
            key: row[key]
            for key in ("dataset_name", "category_name", "subset_name")
            if key in row
        }
        previous_config = detail.get("config")
        if previous_config is not None and previous_config != config:
            raise AdapterError(
                "RESULT_INVALID",
                f"inconsistent EvalScope identity fields for report row {row_id!r}",
            )
        detail["config"] = config

    metric_contract = task.get("metrics") or {}
    namespace = str(metric_contract.get("namespace") or "framework_native")
    mapping = metric_contract.get("mapping") or {}
    required = list(metric_contract.get("required") or [])
    native_summary = {}
    for name, entries in native_occurrences.items():
        if len(entries) == 1:
            native_summary[name] = entries[0]
        else:
            for index, entry in enumerate(entries, 1):
                native_summary[f"{name}#{index}"] = entry

    if namespace == "canonical":
        metrics = {}
        for native_name, canonical_name in mapping.items():
            entries = native_occurrences.get(native_name) or []
            if len(entries) > 1:
                raise AdapterError(
                    "RESULT_INVALID",
                    f"EvalScope metric {native_name!r} is ambiguous across {len(entries)} report rows",
                )
            if entries:
                if canonical_name in metrics:
                    raise AdapterError(
                        "RESULT_INVALID",
                        f"multiple EvalScope metrics map to canonical metric {canonical_name!r}",
                    )
                metrics[canonical_name] = entries[0]
        missing = [name for name in required if name not in metrics]
        if missing:
            raise AdapterError("RESULT_INVALID", f"missing required canonical metrics: {missing}")
    elif namespace == "framework_native":
        metrics = native_summary
    else:
        raise AdapterError("RESULT_INVALID", f"unsupported metric namespace: {namespace}")
    if not metrics:
        raise AdapterError("RESULT_INVALID", "EvalScope report contains no usable summary metrics")

    summary = {
        "id": task_id,
        "kind": "task",
        "metric_namespace": namespace,
        "metrics": metrics,
    }
    if namespace == "canonical":
        summary["native_metrics"] = native_summary
    run = i["run_metadata"]
    result = {
        "schema_version": "1.0",
        "run_id": run["run_id"],
        "model": run["model"],
        "benchmark": run["benchmark"],
        "framework": "evalscope",
        "metrics": metrics,
        "raw_result": {"path": str(report_path), "media_type": "application/json"},
        "breakdowns": {"summary": summary, "groups": {}, "tasks": tasks},
        "metadata": {
            "task_id": task_id,
            "protocol_fingerprint": task["protocol_fingerprint"],
            "metric_namespace": namespace,
        },
    }
    artifacts = _sample_artifacts(root)
    if artifacts:
        result["sample_artifacts"] = artifacts
    return result


def snapshot(i, c):
    params = _parameters(i.get("evaluation") or {})
    return {
        "framework": "evalscope",
        "adapter_version": str(load_manifest(Path(__file__).with_name("manifest.json"))["version"]),
        "configured_executable": str(params.get("executable") or "evalscope"),
        "expected_version": params.get("expected_version"),
        "version_source": "selected_environment_preflight",
    }


OPERATIONS = {
    "requirements": requirements,
    "plan_preflight": plan_preflight,
    "plan_evaluate": plan_evaluate,
    "normalize": normalize,
    "snapshot": snapshot,
}
