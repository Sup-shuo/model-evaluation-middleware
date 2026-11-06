"""Pre-start check for explicitly declared language-only model artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from model_evaluation.core.errors import CompatibilityError


class ArtifactModalityMismatch(CompatibilityError):
    code = "ARTIFACT_MODALITY_MISMATCH"


def _metadata(path: Path) -> dict:
    # Only read small JSON sidecars, never open weights or import model code.
    try:
        if not path.is_file():
            return {}
        with path.open("rb") as stream:
            raw = stream.read(2**20 + 1)
        value = json.loads(raw) if len(raw) <= 2**20 else None
    except (OSError, ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def _language_marker(specs: dict) -> str | None:
    transform = (specs["model"].get("metadata") or {}).get("transformation")
    if isinstance(transform, dict) and transform.get("visual_capability") == "removed":
        return "model.metadata.transformation.visual_capability=removed"
    local = (specs["deployment"].get("model_location") or {}).get("local_path")
    if not local or not Path(local).is_absolute():
        return None
    root = Path(local)
    if _metadata(root / "config.json").get("language_model_only") is True:
        return "config.json: language_model_only=true"
    mode = _metadata(root / "SOURCE_GGUF.json").get("export_mode")
    if mode in ("language-only", "text-llm"):
        return f"SOURCE_GGUF.json: export_mode={mode}"
    derivation = _metadata(root / "DERIVATION.json")
    retention = derivation.get("component_retention")
    if isinstance(retention, dict) and retention.get("vision") == "removed":
        return "DERIVATION.json: component_retention.vision=removed"
    fmt = derivation.get("derived_format")
    if fmt == "Qwen3 text-only dense BF16 safetensors":
        return f"DERIVATION.json: derived_format={fmt}"
    return None


def modality_skip(plan: dict) -> dict | None:
    resolved = plan.get("resolved") or {}
    specs = resolved.get("specs") or {}
    benchmark = specs.get("benchmark") or {}
    if (benchmark.get("metadata") or {}).get("input_modality") != "image_text":
        return None
    if resolved.get("management_mode") != "managed":
        return None
    marker = _language_marker(specs)
    if not marker:
        return None
    return {"code": ArtifactModalityMismatch.code, "marker": marker,
            "message": f"{benchmark['id']} requires image input; the selected model "
                       f"artifact declares no vision support ({marker})."}
