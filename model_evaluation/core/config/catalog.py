from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from model_evaluation.core.config.documents import (
    load_yaml_document,
    yaml_document_paths,
)


CONFIG_KINDS = ("system", "model", "evaluation")
_CATALOG_DIRS = {
    "system": "systems",
    "model": "models",
    "evaluation": "evaluations",
}
_ROOT_FILES = {
    "system": "system.yaml",
    "evaluation": "evaluation.yaml",
}


def iter_config_paths(project_root: str | Path, kind: str | None = None) -> Iterable[tuple[str, Path]]:
    base = Path(project_root).resolve() / "config"
    kinds = (kind,) if kind else CONFIG_KINDS
    for current in kinds:
        if current not in CONFIG_KINDS:
            raise ValueError(f"unknown configuration kind: {current!r}")
        root_file = _ROOT_FILES.get(current)
        if root_file and (base / root_file).is_file():
            yield current, (base / root_file).absolute()
        yield from (
            (current, path.absolute())
            for path in yaml_document_paths(base / _CATALOG_DIRS[current])
        )


def _relative_reference(project_root: Path, kind: str, path: Path) -> str:
    config_root = project_root / "config"
    root_file = _ROOT_FILES.get(kind)
    if root_file and path == (config_root / root_file).absolute():
        return "default"
    catalog_root = (config_root / _CATALOG_DIRS[kind]).absolute()
    relative = path.relative_to(catalog_root)
    return relative.with_suffix("").as_posix()


@dataclass(frozen=True)
class ConfigEntry:
    kind: str
    reference: str
    path: Path
    schema_version: str | None
    data: dict[str, Any] | None
    error: str | None = None


def scan_config_catalog(project_root: str | Path, kind: str | None = None) -> list[ConfigEntry]:
    root = Path(project_root).resolve()
    entries: list[ConfigEntry] = []
    for current, path in iter_config_paths(root, kind):
        reference = _relative_reference(root, current, path)
        if path.is_symlink():
            entries.append(ConfigEntry(current, reference, path, None, None, "symbolic links are not accepted"))
            continue
        try:
            value = load_yaml_document(path)
            if current == "model" and value.get("id"):
                reference = str(value["id"])
            entries.append(
                ConfigEntry(
                    current,
                    reference,
                    path,
                    str(value.get("schema_version")) if value.get("schema_version") is not None else None,
                    value,
                )
            )
        except Exception as exc:
            entries.append(ConfigEntry(current, reference, path, None, None, str(exc)))
    return entries


def resolve_config_reference(
    project_root: str | Path,
    value: str | Path,
    *,
    catalog_dir: str,
) -> Path:
    """Resolve a System/Evaluation ID, including a confined nested ID.

    Existing explicit files retain precedence. An extensionless value such as
    ``team/smoke`` resolves below ``config/evaluations`` or ``config/systems``.
    """

    project = Path(project_root).resolve()
    raw = Path(value).expanduser()
    if raw.is_file() or raw.is_absolute() or raw.suffix:
        return raw
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError(f"invalid/path-escaping user config id: {value!r}")
    root = (project / "config" / catalog_dir).resolve()
    candidates = [(root / raw).with_suffix(extension).resolve() for extension in (".yaml", ".yml")]
    confined = [path for path in candidates if path == root or root in path.parents]
    matches = [path for path in confined if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous user config id {str(value)!r} in config/{catalog_dir}")
    return raw
