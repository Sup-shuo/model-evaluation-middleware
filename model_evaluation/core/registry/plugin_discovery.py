from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from model_evaluation.core.errors import AdapterProtocolError, ConfigError


ENTRY_POINT_GROUP = "model_evaluation.adapters"
_MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


@dataclass(frozen=True)
class AdapterCandidate:
    kind: str
    name: str
    entry: Path
    source: str


def adapter_search_roots(builtin_root: Path) -> list[tuple[Path, str]]:
    """Return the built-in root plus explicitly configured development roots."""
    roots = [(builtin_root.resolve(), "builtin")]
    configured = os.environ.get("MODEL_EVAL_ADAPTER_PATHS", "")
    for raw in configured.split(os.pathsep):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ConfigError("MODEL_EVAL_ADAPTER_PATHS entries must be absolute")
        resolved = path.resolve()
        if not resolved.is_dir():
            raise ConfigError(f"adapter search root is not a directory: {resolved}")
        roots.append((resolved, f"path:{resolved}"))
    return roots


def filesystem_candidates(
    roots: list[tuple[Path, str]], valid_kinds: set[str]
) -> list[AdapterCandidate]:
    found: list[AdapterCandidate] = []
    for root, source in roots:
        for kind_dir in sorted(root.iterdir() if root.is_dir() else []):
            if not kind_dir.is_dir() or kind_dir.name not in valid_kinds:
                continue
            for implementation in sorted(path for path in kind_dir.iterdir() if path.is_dir()):
                entry = implementation / "adapter"
                if entry.is_file() and os.access(entry, os.X_OK):
                    found.append(
                        AdapterCandidate(
                            kind=kind_dir.name,
                            name=implementation.name,
                            entry=entry.resolve(),
                            source=source,
                        )
                    )
    return found


def _selected_entry_points():
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group=ENTRY_POINT_GROUP))
    return list(discovered.get(ENTRY_POINT_GROUP, ()))


def _parse_entry_point_name(value: str, valid_kinds: set[str]) -> tuple[str, str]:
    if "." not in value:
        raise AdapterProtocolError(
            f"adapter entry point name must be '<kind>.<name>': {value!r}"
        )
    kind, name = value.split(".", 1)
    if kind not in valid_kinds or not name or "/" in name or "\\" in name:
        raise AdapterProtocolError(f"invalid adapter entry point name: {value!r}")
    return kind, name


def installed_plugin_candidates(valid_kinds: set[str]) -> list[AdapterCandidate]:
    """Resolve installed Adapter entry points without importing plugin code.

    An installed distribution declares, for example::

        [project.entry-points."model_evaluation.adapters"]
        "device.example" = "example_adapter.adapters.device.example"

    The value names a package-data directory containing the executable
    ``adapter`` launcher.  We locate it through distribution metadata rather
    than ``EntryPoint.load()``, so discovery never imports third-party code
    into the Core process.  The launcher still executes through the normal
    JSON-over-stdio isolation boundary.
    """
    candidates: list[AdapterCandidate] = []
    for entry_point in sorted(_selected_entry_points(), key=lambda item: item.name):
        kind, name = _parse_entry_point_name(entry_point.name, valid_kinds)
        module = entry_point.module
        if entry_point.attr or entry_point.extras or not _MODULE_NAME.fullmatch(module):
            raise AdapterProtocolError(
                f"adapter entry point {entry_point.name!r} must reference a module directory "
                "without an attribute or extras"
            )
        distribution = entry_point.dist
        if distribution is None:
            raise AdapterProtocolError(
                f"adapter entry point {entry_point.name!r} has no owning distribution"
            )
        module_dir = Path(*module.split("."))
        entry = Path(distribution.locate_file(module_dir / "adapter")).resolve()
        if not entry.is_file() or not os.access(entry, os.X_OK):
            raise AdapterProtocolError(
                f"installed adapter entry point {entry_point.name!r} does not resolve to an "
                f"executable launcher: {entry}"
            )
        dist_name = distribution.metadata.get("Name") or "unknown-distribution"
        candidates.append(
            AdapterCandidate(
                kind=kind,
                name=name,
                entry=entry,
                source=f"entry-point:{dist_name}",
            )
        )
    return candidates


def index_candidates(candidates: list[AdapterCandidate]) -> dict[tuple[str, str], AdapterCandidate]:
    indexed: dict[tuple[str, str], AdapterCandidate] = {}
    for candidate in candidates:
        key = (candidate.kind, candidate.name)
        previous = indexed.get(key)
        if previous is not None:
            raise ConfigError(
                f"duplicate adapter {candidate.kind}/{candidate.name}: "
                f"{previous.source} ({previous.entry}) and {candidate.source} ({candidate.entry})"
            )
        indexed[key] = candidate
    return indexed
