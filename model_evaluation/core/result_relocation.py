from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.serialization import json_loads_strict


_MAP_NAME = "RELOCATION.json"


def _absolute_normal_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts:
        raise ConfigError(f"{label} must be a normalized absolute path: {value!r}")
    if path == Path(path.anchor):
        raise ConfigError(f"{label} may not be a filesystem root")
    return path


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@dataclass(frozen=True)
class ResultRelocationMap:
    """Resolve persisted result paths after a byte-preserving tree move."""

    current_root: Path
    old_roots: tuple[Path, ...] = ()

    def relocate(self, value: str, *, label: str) -> Path:
        path = _absolute_normal_path(value, label)
        if _is_within(path, self.current_root):
            return path
        matches = [root for root in self.old_roots if _is_within(path, root)]
        if len(matches) != 1:
            return path
        return self.current_root / path.relative_to(matches[0])


def load_result_relocation(results_root: str | Path) -> ResultRelocationMap:
    current = Path(results_root).resolve()
    path = current / _MAP_NAME
    if not path.exists():
        return ResultRelocationMap(current)
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"result relocation map is missing/unsafe: {path}")
    try:
        obj = json_loads_strict(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"failed to parse result relocation map {path}: {exc}") from exc
    if not isinstance(obj, dict) or set(obj) != {"schema_version", "mappings"}:
        raise ConfigError("RELOCATION.json must contain only schema_version and mappings")
    if obj.get("schema_version") != "1.0":
        raise ConfigError("unsupported result relocation schema_version")
    mappings = obj.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ConfigError("RELOCATION.json mappings must be a non-empty array")

    old_roots: list[Path] = []
    for index, row in enumerate(mappings):
        label = f"RELOCATION.json mappings[{index}]"
        if not isinstance(row, dict) or set(row) != {"old_root", "new_root"}:
            raise ConfigError(f"{label} must contain only old_root and new_root")
        old = _absolute_normal_path(row.get("old_root"), f"{label}.old_root")
        new = _absolute_normal_path(row.get("new_root"), f"{label}.new_root")
        if new != current:
            raise ConfigError(f"{label}.new_root must equal current results root: {current}")
        if old == current:
            raise ConfigError(f"{label} creates a relocation chain/self-map")
        for prior in old_roots:
            if _is_within(old, prior) or _is_within(prior, old):
                raise ConfigError(f"{label}.old_root duplicates or overlaps another mapping")
        old_roots.append(old)
    return ResultRelocationMap(current, tuple(old_roots))
