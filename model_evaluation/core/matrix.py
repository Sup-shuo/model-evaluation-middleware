from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from model_evaluation.core.config.parsing import load_json_strict
from model_evaluation.core.errors import (
    CleanupCriticalError,
    ConfigError,
    OrchestrationInterruptedError,
    StaleProcessError,
)
from model_evaluation.core.matrix_config import MatrixRepository, MatrixSchemas
from model_evaluation.core.matrix_execution import execute_matrix
from model_evaluation.core.matrix_product import MatrixProductMixin
from model_evaluation.core.matrix_planner import (
    MatrixPlanner,
    finalize_matrix_plan,
    verify_matrix_plan,
)
from model_evaluation.core.resources import ResourceManager
from model_evaluation.core.result_relocation import ResultRelocationMap, load_result_relocation

class MatrixExecutor(MatrixProductMixin):
    @staticmethod
    def _failure_requires_hard_stop(exc: BaseException, cleanup_status: str | None) -> bool:
        # Never continue a batch when Core-owned process cleanup is incomplete,
        # even if a more useful primary error (OOM/backend failure/etc.) is kept.
        return (
            cleanup_status == "incomplete"
            or isinstance(
                exc,
                (
                    CleanupCriticalError,
                    OrchestrationInterruptedError,
                    StaleProcessError,
                ),
            )
        )

    def __init__(
        self,
        app,
        *,
        results_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        secrets_map: dict[str, str] | None = None,
    ):
        self.app = app
        project_root = Path(getattr(app, "project_root", app.root))
        self.results_root = Path(results_root or project_root / "results").resolve()
        self.cache_root = Path(cache_root or project_root / "cache").resolve()
        self.secrets_map = secrets_map
        self.result_relocation = load_result_relocation(self.results_root)
        self.resources = ResourceManager(app.host_runtime_root / "resources")

    @staticmethod
    def _batch_timezone(matrix_plan: dict) -> ZoneInfo:
        names={
            str((((plan.get('resolved') or {}).get('specs') or {}).get('platform') or {}).get('metadata',{}).get('timezone'))
            for plan in (matrix_plan.get('plans') or [])
            if ((((plan.get('resolved') or {}).get('specs') or {}).get('platform') or {}).get('metadata',{}).get('timezone'))
        }
        if len(names) > 1:
            raise ConfigError(f'matrix child plans declare multiple result timezones: {sorted(names)}')
        name=next(iter(names),'Asia/Shanghai')
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError as exc:
            raise ConfigError(f'system timezone is unavailable: {name!r}') from exc

    @classmethod
    def _batch_id(cls, matrix_plan: dict, *, when: datetime|None=None) -> str:
        instant = when or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        return instant.astimezone(cls._batch_timezone(matrix_plan)).strftime(
            "%y%m%d-%H%M"
        )

    @classmethod
    def _batch_now(cls, matrix_plan: dict) -> str:
        return datetime.now(cls._batch_timezone(matrix_plan)).isoformat(timespec='seconds')

    def _allocate_batch_dir(self, matrix_plan: dict, *, when: datetime|None=None) -> Path:
        root = self.results_root / "_batches"
        root.mkdir(parents=True, exist_ok=True)
        base = self._batch_id(matrix_plan, when=when)
        for index in range(1, 10_000):
            name = base if index == 1 else f"{base}-{index}"
            candidate = root / name
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise ConfigError(f'could not allocate a unique batch directory for {base!r}')

    @staticmethod
    def _safe_confined_path(
        value: str | Path,
        root: str | Path,
        *,
        label: str,
        require_file: bool = False,
        require_dir: bool = False,
    ) -> Path:
        base = Path(root).resolve()
        raw = Path(value)
        lexical = raw.absolute()
        for candidate in (lexical, *lexical.parents):
            if candidate.is_symlink():
                raise ValueError(f"{label} may not traverse a symlink: {candidate}")
            if candidate == base:
                break
        path = raw.resolve()
        if path != base and base not in path.parents:
            raise ValueError(f"{label} escapes root: root={base} path={path}")
        if require_file and not path.is_file():
            raise ValueError(f"{label} is not a regular file: {path}")
        if require_dir and not path.is_dir():
            raise ValueError(f"{label} is not a directory: {path}")
        return path

    def _confined_batch_dir(self, value: str|Path) -> Path:
        base = (self.results_root / "_batches").resolve()
        path = self._safe_confined_path(
            value,
            base,
            label="resume directory",
            require_dir=True,
        )
        if path == base:
            raise ConfigError(
                f"resume directory must be a child of {base}: {path}"
            )
        return path

    def _stored_path(
        self,
        value: str | Path,
        root: str | Path,
        *,
        label: str,
        require_file: bool = False,
        require_dir: bool = False,
    ) -> Path:
        relocation = getattr(
            self,
            "result_relocation",
            ResultRelocationMap(Path(self.results_root).resolve()),
        )
        relocated = relocation.relocate(str(value), label=label)
        return self._safe_confined_path(
            relocated,
            root,
            label=label,
            require_file=require_file,
            require_dir=require_dir,
        )

    @staticmethod
    def _load_status(
        path: Path,
        matrix_id: str,
        planned_ids: set[str],
    ) -> dict[str, dict]:
        if not path.is_file():
            return {}
        try:
            previous = load_json_strict(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigError(f"failed to parse batch status {path}: {exc}") from exc
        if previous.get("matrix_id") != matrix_id:
            raise ConfigError("batch status belongs to a different matrix plan")
        status = {}
        for row in previous.get("runs", []):
            if not isinstance(row, dict) or not row.get("plan_id"):
                raise ConfigError("batch status contains invalid run record")
            plan_id = str(row["plan_id"])
            if plan_id not in planned_ids:
                raise ConfigError(
                    f"batch status references unknown plan_id: {plan_id}"
                )
            if plan_id in status:
                raise ConfigError(
                    f"batch status contains duplicate plan_id: {plan_id}"
                )
            status[plan_id] = row
        return status

    def execute(
        self,
        matrix_plan: dict,
        *,
        continue_on_error: bool | None = None,
        resume_dir: str | Path | None = None,
    ) -> tuple[Path, dict]:
        return execute_matrix(
            self,
            matrix_plan,
            continue_on_error=continue_on_error,
            resume_dir=resume_dir,
        )
