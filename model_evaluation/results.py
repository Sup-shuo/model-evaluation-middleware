"""Stable, read-only access to a completed result product.

This module consumes the same frozen JSON files as ``eval-manager inspect``.
It does not reinterpret framework scores and does not make provenance or
tamper-resistance claims.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_evaluation import package_root
from model_evaluation.core.result_product import inspect_run_product
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.serialization import json_loads_strict


def _load_optional(path: Path) -> dict | None:
    if not path.is_file():
        return None
    value = json_loads_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"result product file must contain an object: {path}")
    return value


@dataclass(frozen=True)
class Artifact:
    kind: str
    path: Path
    media_type: str | None = None


class MetricsView:
    def __init__(self, payload: dict | None):
        self._payload = copy.deepcopy(payload or {})

    def summary(self) -> dict:
        return copy.deepcopy(self._payload.get("summary") or {})

    def groups(self) -> dict:
        return copy.deepcopy(self._payload.get("groups") or {})

    def tasks(self) -> dict:
        return copy.deepcopy(self._payload.get("tasks") or {})


class RunProduct:
    def __init__(
        self,
        *,
        root: Path,
        report: dict,
        result: dict | None,
        metrics: dict | None,
        terminal: dict,
        failure: dict | None,
        run_config: dict | None,
        runtime_versions: dict | None,
    ):
        self.root = root
        self._report = copy.deepcopy(report)
        self._result = copy.deepcopy(result)
        self._terminal = copy.deepcopy(terminal)
        self._failure = copy.deepcopy(failure)
        self._run_config = copy.deepcopy(run_config)
        self._runtime_versions = copy.deepcopy(runtime_versions)
        self.metrics = MetricsView(metrics)

    @property
    def run_id(self) -> str:
        return str(self._terminal["run_id"])

    @property
    def outcome(self) -> str:
        return str(self._terminal["outcome"])

    def summary(self) -> dict:
        return copy.deepcopy(self._report)

    def result(self) -> dict | None:
        return copy.deepcopy(self._result)

    def terminal(self) -> dict:
        return copy.deepcopy(self._terminal)

    def failure(self) -> dict | None:
        return copy.deepcopy(self._failure)

    def runtime(self) -> dict[str, Any]:
        return {
            "run_config": copy.deepcopy(self._run_config),
            "runtime_versions": copy.deepcopy(self._runtime_versions),
        }

    def artifacts(self) -> tuple[Artifact, ...]:
        rows: list[Artifact] = []
        if self._result:
            raw = self._result.get("raw_result") or {}
            if raw.get("path"):
                rows.append(
                    Artifact(
                        "raw",
                        (self.root / raw["path"]).resolve(),
                        raw.get("media_type"),
                    )
                )
            for item in self._result.get("sample_artifacts") or []:
                rows.append(
                    Artifact(
                        "sample",
                        (self.root / item["path"]).resolve(),
                        item.get("media_type"),
                    )
                )
        if self._failure:
            for name, item in sorted((self._failure.get("logs") or {}).items()):
                rows.append(
                    Artifact(
                        f"log:{name}",
                        (self.root / item["path"]).resolve(),
                        "text/plain",
                    )
                )
        return tuple(rows)


def load_run(path: str | Path, *, schemas: SchemaStore | None = None) -> RunProduct:
    """Validate and load one finished run directory.

    Returned mappings are defensive copies.  Artifact paths have already been
    checked by the public result-product validator and remain confined to the
    run directory.
    """

    supplied = Path(path).expanduser()
    schema_store = schemas or SchemaStore(package_root() / "schemas")
    report = inspect_run_product(supplied, schema_store)
    root = Path(report["run_dir"])
    return RunProduct(
        root=root,
        report=report,
        result=_load_optional(root / "result.json"),
        metrics=_load_optional(root / "metrics.json"),
        terminal=_load_optional(root / "terminal.json") or {},
        failure=_load_optional(root / "failure.json"),
        run_config=_load_optional(root / "config" / "run_config.json"),
        runtime_versions=_load_optional(root / "config" / "runtime_versions.json"),
    )


__all__ = ["Artifact", "MetricsView", "RunProduct", "load_run"]
