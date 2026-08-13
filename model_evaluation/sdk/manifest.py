from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from model_evaluation.sdk.jsonutil import loads as json_loads


@lru_cache(maxsize=64)
def load_manifest(path: str | Path) -> dict:
    """Load an adapter's authoritative manifest.json.

    Adapter identity/version/operations must have exactly one source of truth.
    The shell entrypoint and the Python invoke path both consume manifest.json;
    implementation modules therefore do not duplicate MANIFEST constants.
    """
    p = Path(path).resolve()
    obj = json_loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"adapter manifest must be a JSON object: {p}")
    return obj
