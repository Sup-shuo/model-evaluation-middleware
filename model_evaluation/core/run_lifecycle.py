from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_evaluation.core.errors import ProcessError


ACTIVE_STATES = (
    "CREATED",
    "PLANNED",
    "PLATFORM_READY",
    "DATA_READY",
    "TASK_READY",
    "EVALUATOR_PREFLIGHT",
    "SERVICE_STARTING",
    "SERVICE_READY",
    "EVALUATING",
    "NORMALIZING",
)
TERMINAL_STATES = ("SUCCEEDED", "FINALIZED")

_NEXT = {
    None: {"CREATED", "FAILED", "CLEANING"},
    "CREATED": {"PLANNED"},
    "PLANNED": {"PLATFORM_READY"},
    "PLATFORM_READY": {"DATA_READY"},
    "DATA_READY": {"TASK_READY"},
    "TASK_READY": {"EVALUATOR_PREFLIGHT"},
    "EVALUATOR_PREFLIGHT": {"SERVICE_STARTING", "SERVICE_READY"},
    "SERVICE_STARTING": {"SERVICE_READY"},
    "SERVICE_READY": {"EVALUATING"},
    "EVALUATING": {"NORMALIZING"},
    "NORMALIZING": {"CLEANING"},
    "FAILED": {"CLEANING"},
    "CLEANING": {"SUCCEEDED", "FINALIZED"},
    "SUCCEEDED": set(),
    "FINALIZED": set(),
}


def validate_state_transition(previous: str | None, current: str) -> None:
    """Validate the single fixed Core lifecycle.

    Failure and cleanup are recovery edges, not configurable workflow stages.
    A run may enter them from any active state so that diagnostics remain
    writable even when an earlier status write was interrupted.
    """

    if current == "FAILED" and previous in {None, *ACTIVE_STATES}:
        return
    if current == "CLEANING" and previous in {None, *ACTIVE_STATES, "FAILED"}:
        return
    if current not in _NEXT.get(previous, set()):
        raise ProcessError(
            f"invalid run lifecycle transition: {previous or '<none>'} -> {current}"
        )


@dataclass
class RunContext:
    plan: dict[str, Any]
    run_dir: Path
    started_at: str
    stale_recovery: list[dict[str, Any]] = field(default_factory=list)
    backend_handle: Any = None
    backend_shutdown: dict[str, Any] | None = None
    evaluator_returncode: int | None = None
    resolved_platform: dict[str, Any] | None = None
    runtime_versions: dict[str, Any] = field(default_factory=dict)
    dataset_client: Any = None
    dataset: dict[str, Any] | None = None
    task: dict[str, Any] | None = None
    evaluator: Any = None
    evaluator_requirements: dict[str, Any] | None = None
    service: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def specs(self) -> dict[str, Any]:
        return self.plan["resolved"]["specs"]

    @property
    def platform(self) -> dict[str, Any]:
        return self.specs["platform"]

    @property
    def deployment(self) -> dict[str, Any]:
        return self.specs["deployment"]

    @property
    def benchmark(self) -> dict[str, Any]:
        return self.specs["benchmark"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.specs["evaluation"]

    @property
    def model(self) -> dict[str, Any]:
        return self.specs["model"]

    @property
    def mode(self) -> str:
        return str(self.deployment["management"]["mode"])

    @property
    def offline(self) -> bool:
        return bool((self.plan["run_spec"].get("overrides") or {}).get("offline", False))


__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "RunContext",
    "validate_state_transition",
]
