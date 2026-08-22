from __future__ import annotations

import hashlib
import json

from model_evaluation.sdk.runtime import AdapterError


def requirements(i, c):
    return {"schema_version": "1.0", "requirements": []}


def _metric_contract(benchmark: dict, evaluation: dict) -> dict:
    required = list(benchmark.get("metrics") or [])
    params = (evaluation or {}).get("parameters") or {}
    maps = params.get("metric_maps") or {}
    if not isinstance(maps, dict):
        raise AdapterError("CONFIG_INVALID", "EvalScope parameters.metric_maps must be an object")
    mapping = maps.get(benchmark["id"])
    if mapping is None:
        return {"namespace": "framework_native", "required": required}
    if not isinstance(mapping, dict) or not mapping or not all(
        isinstance(source, str)
        and source
        and isinstance(target, str)
        and target
        for source, target in mapping.items()
    ):
        raise AdapterError(
            "CONFIG_INVALID",
            f"EvalScope metric map for {benchmark['id']} must be a non-empty-string mapping",
        )
    if len(set(mapping.values())) != len(mapping):
        raise AdapterError(
            "CONFIG_INVALID",
            f"EvalScope metric map for {benchmark['id']} maps multiple native metrics to one canonical metric",
        )
    missing = [name for name in required if name not in mapping.values()]
    if missing:
        raise AdapterError(
            "CONFIG_INVALID",
            f"EvalScope metric map for {benchmark['id']} does not provide required metrics: {missing}",
        )
    return {"namespace": "canonical", "mapping": dict(mapping), "required": required}


def _basis(benchmark: dict, dataset: dict, evaluation: dict) -> dict:
    protocol = benchmark.get("protocol") or {}
    task_id = str(protocol.get("task") or benchmark["id"])
    dataset_args = protocol.get("evalscope_dataset_args") or {}
    if not isinstance(dataset_args, dict):
        raise AdapterError("CONFIG_INVALID", "benchmark.protocol.evalscope_dataset_args must be an object")
    dataset_args = dict(dataset_args)
    fewshot = protocol.get("fewshot")
    if fewshot is not None:
        if isinstance(fewshot, bool) or not isinstance(fewshot, int) or fewshot < 0:
            raise AdapterError("CONFIG_INVALID", "benchmark.protocol.fewshot must be a non-negative integer")
        configured = dataset_args.get("few_shot_num")
        if configured is not None and configured != fewshot:
            raise AdapterError(
                "CONFIG_INVALID",
                "benchmark.protocol.fewshot conflicts with evalscope_dataset_args.few_shot_num",
            )
        dataset_args["few_shot_num"] = fewshot
    split = protocol.get("split")
    if split is not None:
        if not isinstance(split, str) or not split:
            raise AdapterError("CONFIG_INVALID", "benchmark.protocol.split must be a non-empty string")
        configured = dataset_args.get("eval_split")
        if configured is not None and configured != split:
            raise AdapterError(
                "CONFIG_INVALID",
                "benchmark.protocol.split conflicts with evalscope_dataset_args.eval_split",
            )
        dataset_args["eval_split"] = split
    return {
        "binding": "evalscope/v1-native",
        "benchmark_id": benchmark["id"],
        "task_id": task_id,
        "protocol": protocol,
        "dataset_fingerprint": dataset.get("fingerprint"),
        "metric_contract": _metric_contract(benchmark, evaluation),
        "dataset_args": dataset_args,
    }


def _fingerprint(benchmark: dict, dataset: dict, evaluation: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            _basis(benchmark, dataset, evaluation),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def build_task(i, c):
    benchmark = i["benchmark"]
    dataset = i["dataset_artifact"]
    evaluation = i.get("evaluation") or {}
    if (dataset.get("materialization") or {}).get("kind") != "virtual":
        raise AdapterError(
            "COMPATIBILITY_ERROR",
            "generic EvalScope binding expects a virtual dataset managed by EvalScope",
        )
    basis = _basis(benchmark, dataset, evaluation)
    protocol = benchmark.get("protocol") or {}
    execution = {
        "inference": list(protocol.get("inference") or ["generation"]),
        "num_fewshot": protocol.get("fewshot"),
    }
    metadata = {
        "dataset_fingerprint": dataset.get("fingerprint"),
        "evalscope_dataset_args": basis["dataset_args"],
        "dataset_management": "evalscope-native",
    }
    return {
        "schema_version": "1.0",
        "framework": "evalscope",
        "benchmark_id": benchmark["id"],
        "task_id": basis["task_id"],
        "protocol_fingerprint": _fingerprint(benchmark, dataset, evaluation),
        "execution": execution,
        "metrics": basis["metric_contract"],
        "metadata": metadata,
    }


def protocol_fingerprint(i, c):
    return {
        "protocol_fingerprint": _fingerprint(
            i["benchmark"], i["dataset_artifact"], i.get("evaluation") or {}
        )
    }


OPERATIONS = {
    "requirements": requirements,
    "build_task": build_task,
    "protocol_fingerprint": protocol_fingerprint,
}
