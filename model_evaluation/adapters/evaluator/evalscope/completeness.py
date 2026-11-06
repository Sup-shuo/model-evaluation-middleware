"""Opt-in dataset completeness checks for structured EvalScope artifacts."""
import math
from pathlib import Path
if __package__:
    from .predictions import prediction_rows
else:
    from predictions import prediction_rows
from model_evaluation.sdk.jsonutil import loads
from model_evaluation.sdk.runtime import AdapterError


def _invalid(message):
    raise AdapterError("RESULT_INVALID", "EvalScope completeness: " + message)


def _image(messages):
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and isinstance(part.get("image"), str) and part["image"]:
                return True
            if part.get("type") == "image_url":
                url = part.get("image_url")
                if isinstance(url, dict) and isinstance(url.get("url"), str) and url["url"]:
                    return True
    return False


def validate_completeness(root: Path, task: dict, rows: list, detail_rows: list):
    policy = (task.get("metadata") or {}).get("evalscope_result_validation")
    if policy is None:
        return  # Existing task artifacts and other benchmarks retain their contract.
    if not isinstance(policy, dict):
        raise AdapterError("CONFIG_INVALID", "result validation must be an object")
    counts = policy.get("subset_counts")
    if not isinstance(counts, dict) or not counts or any(
        not isinstance(k, str) or not k or type(v) is not int or v <= 0
        for k, v in counts.items()
    ):
        raise AdapterError("CONFIG_INVALID", "result validation requires positive subset_counts")
    counts = dict(counts)
    selected = ((task.get("metadata") or {}).get("evalscope_dataset_args") or {}).get("subset_list")
    if selected:
        if not isinstance(selected, list) or len(set(selected)) != len(selected) or set(selected) - counts.keys():
            raise AdapterError("CONFIG_INVALID", "invalid completeness subset_list")
        counts = {key: counts[key] for key in selected}
    limit = policy.get("sample_limit")
    if limit is not None:
        if type(limit) is int and limit > 0:
            counts = {key: min(value, limit) for key, value in counts.items()}
        elif type(limit) is float and 0 < limit <= 1:
            counts = {key: int(value * limit) for key, value in counts.items()}
        else:
            raise AdapterError("CONFIG_INVALID", "completeness sample_limit must be a positive integer or fraction in (0, 1]")
    reported = {}
    for row in detail_rows:
        subset = row.get("subset_name") or row.get("subset")
        if subset:
            number = row.get("num", row.get("sample_count"))
            if type(number) is not int or number != counts.get(subset):
                _invalid("unexpected report subset/count: " + str((subset, number)))
            reported[subset] = number
    if reported != counts:
        _invalid("report subset counts do not match requested evaluation")
    for row in rows:
        number = row.get("num", row.get("sample_count"))
        if number is not None and (type(number) is not int or number != sum(counts.values())):
            _invalid("summary sample count does not match requested evaluation")
    identities = {}
    for kind in ("predictions", "reviews"):
        seen = set()
        observed = dict.fromkeys(counts, 0)
        for path, number, row in _rows(root, kind):
            score = row.get("sample_score")
            metadata = row.get("metadata") if kind == "predictions" else (
                score.get("sample_metadata") if isinstance(score, dict) else None)
            if not isinstance(metadata, dict):
                _invalid(f"missing {kind} sample metadata")
            category, index = metadata.get("category"), metadata.get("index")
            if category not in counts or type(index) is not int or index < 0:
                _invalid(f"invalid {kind} sample identity")
            identity = (category, index)
            if identity in seen:
                _invalid(f"duplicate {kind} sample identity: {identity}")
            seen.add(identity)
            observed[category] += 1
            if policy.get("require_images") and not _image(row.get("messages")):
                _invalid(f"missing image-bearing {kind} input")
            if kind == "predictions":
                output = row.get("model_output")
                if not isinstance(output, dict) or not isinstance(output.get("choices"), list) or not output["choices"] or not all(isinstance(choice, dict) for choice in output["choices"]):
                    _invalid("missing structured model output/choices")
                # Generation errors and context overflow are checked by validate_predictions.
            elif not isinstance(score, dict) or not isinstance(score.get("score"), dict):
                _invalid("missing structured review score")
            else:
                values = score["score"].get("value")
                if not isinstance(values, dict) or not values or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in values.values()
                ):
                    _invalid("missing numeric review score values")
        if observed != counts:
            _invalid(f"{kind} subset counts do not match: {observed}")
        identities[kind] = seen
    if identities["predictions"] != identities["reviews"]:
        _invalid("prediction/review sample identities do not match")


def _rows(root, kind):
    if kind == "predictions":
        yield from prediction_rows(root)
        return
    for path in sorted((root / kind).rglob("*.jsonl")):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = loads(line)
                except (ValueError, TypeError) as exc:
                    _invalid(f"invalid {kind} JSON at {path}:{number}: {exc}")
                if not isinstance(row, dict):
                    _invalid(f"non-object {kind} row at {path}:{number}")
                yield path, number, row
