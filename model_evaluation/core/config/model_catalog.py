from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from model_evaluation.core.config.documents import (
    load_yaml_document,
    yaml_document_paths,
)
from model_evaluation.core.config.merge import deep_merge
from model_evaluation.core.errors import ConfigError


def _catalog_root(evaluation_path: Path, configured: object) -> tuple[Path, bool]:
    if configured is not None:
        raw = Path(str(configured)).expanduser()
        root = raw if raw.is_absolute() else evaluation_path.parent / raw
        root = root.resolve()
        if not root.is_dir():
            raise ConfigError(f"evaluation.model_catalog 目录不存在: {root}")
        return root, True

    candidates = [evaluation_path.parent / "models"]
    if evaluation_path.parent.name == "evaluations" or "evaluations" in evaluation_path.parts:
        config_root = next(
            (parent for parent in evaluation_path.parents if parent.name == "config"),
            evaluation_path.parent,
        )
        candidates.insert(0, config_root / "models")
    for candidate in candidates:
        root = candidate.resolve()
        if root.is_dir() and yaml_document_paths(root):
            return root, True
    return candidates[0].resolve(), False


def _load_model_catalog(app, evaluation_path: Path, configured: object) -> tuple[dict[str, dict[str, Any]], Path, bool]:
    root, enabled = _catalog_root(evaluation_path, configured)
    if not enabled:
        return {}, root, False
    catalog: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    paths = yaml_document_paths(root)
    if configured is not None and not paths:
        raise ConfigError(f"evaluation.model_catalog 中没有 YAML 模型配置: {root}")
    for path in paths:
        resolved = path.resolve()
        if root != resolved.parent and root not in resolved.parents:
            raise ConfigError(f"模型配置越过 catalog 根目录: {path}")
        if path.is_symlink():
            raise ConfigError(f"模型 catalog 不接受符号链接文件: {path}")
        row = load_yaml_document(path, reject_secrets=True)
        app.matrix_schemas.validate("user_model", row)
        model_id = str(row["id"]).strip()
        if model_id in catalog:
            raise ConfigError(
                f"模型 catalog id 重复: {model_id!r}: "
                f"{sources[model_id].relative_to(root)} 与 {path.relative_to(root)}"
            )
        catalog[model_id] = copy.deepcopy(row)
        sources[model_id] = path
    return catalog, root, True


def resolve_model_entries(app, evaluation: dict[str, Any], evaluation_path: Path) -> tuple[list[dict[str, Any]], Path, bool]:
    catalog, root, catalog_enabled = _load_model_catalog(
        app, evaluation_path, evaluation.get("model_catalog")
    )
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(evaluation["models"]):
        if isinstance(item, str):
            if catalog_enabled:
                if item not in catalog:
                    available = ", ".join(sorted(catalog)) or "<empty>"
                    raise ConfigError(
                        f"evaluation.models[{index}] 引用了不存在的模型 catalog id {item!r}; 可选: {available}"
                    )
                row = copy.deepcopy(catalog[item])
            else:
                row = {"ref": item}
        else:
            raw = copy.deepcopy(item)
            run_resources = raw.pop("resources", None)
            catalog_id = raw.get("id")
            is_catalog_reference = bool(
                catalog_enabled
                and catalog_id in catalog
                and set(raw).issubset({"id", "overrides"})
            )
            if is_catalog_reference:
                overrides = raw.get("overrides") or {}
                identity_fields = {
                    "id", "label", "source", "ref", "name", "source_type", "revision",
                    "architecture", "quantization", "format", "provenance", "metadata",
                }
                changed_identity = sorted(identity_fields.intersection(overrides))
                if changed_identity:
                    raise ConfigError(
                        f"evaluation.models[{index}].overrides 不能修改模型身份字段: {changed_identity}; "
                        "请新建一份 model catalog 配置"
                    )
                row = deep_merge(catalog[str(catalog_id)], overrides)
                row["id"] = str(catalog_id)
            else:
                row = raw
            if run_resources is not None:
                row["_run_resources"] = run_resources
        row.pop("schema_version", None)
        rows.append(row)
    return rows, root, catalog_enabled
