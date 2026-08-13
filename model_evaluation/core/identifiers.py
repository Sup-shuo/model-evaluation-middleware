from __future__ import annotations

import copy
import hashlib

from model_evaluation.core.serialization import json_dumps_strict


def stable_id(value: object, *, length: int = 24, exclude_keys: set[str] | None = None) -> str:
    """Return a deterministic short identifier for config/result correlation.

    This is deliberately not an integrity or authenticity claim.  It only
    gives equivalent normalized inputs the same convenient identifier.
    """
    normalized = copy.deepcopy(value)
    if exclude_keys and isinstance(normalized, dict):
        for key in exclude_keys:
            normalized.pop(key, None)
    payload = json_dumps_strict(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
