from __future__ import annotations

from pathlib import Path
from typing import Any

from model_evaluation.core.config.loader import reject_inline_secrets
from model_evaluation.core.config.parsing import load_yaml_strict
from model_evaluation.core.errors import ConfigError


def is_platform_metadata(path: Path) -> bool:
    """Return whether a relative path is generated host metadata."""

    return any(part == ".DS_Store" or part.startswith("._") for part in path.parts)


def yaml_document_paths(root: str | Path) -> list[Path]:
    """Discover YAML documents below a root with one metadata policy."""

    catalog_root = Path(root)
    if not catalog_root.is_dir():
        return []
    return sorted(
        path
        for path in {*catalog_root.rglob("*.yaml"), *catalog_root.rglob("*.yml")}
        if path.is_file()
        and not is_platform_metadata(path.relative_to(catalog_root))
    )


def load_yaml_document(
    path: str | Path,
    *,
    reject_secrets: bool = False,
) -> dict[str, Any]:
    """Load one user YAML object with consistent errors and secret handling."""

    document_path = Path(path).expanduser().resolve()
    if not document_path.is_file():
        raise ConfigError(f"用户配置文件不存在: {document_path}")
    try:
        value = load_yaml_strict(document_path.read_text(encoding="utf-8"))
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"无法读取用户配置 {document_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"用户配置必须是 YAML object: {document_path}")
    if reject_secrets:
        reject_inline_secrets(value, str(document_path))
    return value
