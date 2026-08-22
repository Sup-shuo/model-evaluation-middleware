from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from model_evaluation.core.files import atomic_json
from model_evaluation.core.config.parsing import load_json_strict
from model_evaluation.core.process.signals import orchestration_signal_guard
from model_evaluation.core.errors import (
    CleanupCriticalError,
    ConfigError,
    OrchestrationInterruptedError,
    ProcessError,
    StaleProcessError,
)
from model_evaluation.core.identifiers import stable_id
from model_evaluation.core.matrix_config import MatrixRepository, MatrixSchemas
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
        verify_matrix_plan(matrix_plan, app=self.app)
        execution = matrix_plan["matrix_spec"].get("execution") or {}
        configured = bool(execution.get("continue_on_error", False))
        keep_going = (
            configured if continue_on_error is None else bool(continue_on_error)
        )
        planned_ids = {plan["plan_id"] for plan in matrix_plan["plans"]}
        batch_claim = {
            "kind": "other",
            "id": f"matrix:{matrix_plan['matrix_id']}",
            "exclusive": True,
        }
        with orchestration_signal_guard(), self.resources.acquire([batch_claim]):
            batch_dir: Path | None = None
            status_path: Path | None = None
            status: dict[str, dict] = {}
            hard_stop = False
            interrupted_exc: BaseException | None = None
            try:
                if resume_dir:
                    batch_dir = self._confined_batch_dir(resume_dir)
                    plan_path = batch_dir / "matrix_plan.json"
                    if not plan_path.is_file():
                        raise ConfigError(
                            "resume directory is missing matrix_plan.json: "
                            f"{batch_dir}"
                        )
                    saved = load_json_strict(plan_path.read_text(encoding="utf-8"))
                    verify_matrix_plan(saved, app=self.app)
                    if stable_id(saved, length=64) != stable_id(
                        matrix_plan,
                        length=64,
                    ):
                        raise ConfigError(
                            "resume directory contains a different matrix plan"
                        )
                else:
                    batch_dir = self._allocate_batch_dir(matrix_plan)
                    atomic_json(batch_dir / "matrix_plan.json", matrix_plan)
                status_path = batch_dir / "batch_status.json"
                status = self._load_status(
                    status_path,
                    matrix_plan["matrix_id"],
                    planned_ids,
                )
                for index, plan in enumerate(matrix_plan["plans"], 1):
                    plan_id = plan["plan_id"]
                    old = status.get(plan_id)
                    stale_reason = None
                    if old and old.get("status") == "success":
                        ok, stale_reason, _ = self._validate_success_record(old, plan)
                        if ok:
                            continue
                    model_id, model_label, model_ref = self._model_identity(plan)
                    run_spec = plan["run_spec"]
                    rec = {
                        "index": index,
                        "plan_id": plan_id,
                        "model": run_spec["model"],
                        "model_id": model_id,
                        "model_label": model_label,
                        "model_ref": model_ref,
                        "platform": run_spec["platform"],
                        "deployment": run_spec["deployment"],
                        "benchmark": run_spec["benchmark"],
                        "evaluation": run_spec["evaluation"],
                        "status": "running",
                        "attempts": int((old or {}).get("attempts", 0)) + 1,
                        "started_at": self._batch_now(matrix_plan),
                    }
                    if old:
                        history = list(old.get("history") or [])
                        historical_keys = (
                            "status",
                            "attempts",
                            "started_at",
                            "finished_at",
                            "started_utc",
                            "finished_utc",
                            "run_dir",
                            "error",
                        )
                        history.append(
                            {
                                key: copy.deepcopy(old.get(key))
                                for key in historical_keys
                                if old.get(key) is not None
                            }
                        )
                        rec["history"] = history
                    if stale_reason:
                        rec["resume_validation_warning"] = stale_reason
                    status[plan_id] = rec
                    self._write_status(
                        status_path,
                        matrix_plan["matrix_id"],
                        status,
                    )
                    try:
                        orchestrator = self.app.orchestrator(
                            results_root=self.results_root,
                            cache_root=self.cache_root,
                            secrets=self.secrets_map,
                        )
                        path = orchestrator.execute(plan)
                        run_path = Path(path)
                        result_path = run_path / "result.json"
                        terminal_path = run_path / "terminal.json"
                        if not result_path.is_file():
                            # Runs produced before the lightweight product layout
                            # remain resumable and aggregatable.
                            result_path = run_path / "canonical_result.json"
                            terminal_path = run_path / "terminal_record.json"
                        if not result_path.is_file():
                            raise ProcessError(
                                f"run completed without result.json: {path}"
                            )
                        warnings_count = 0
                        if terminal_path.is_file():
                            try:
                                terminal_obj = load_json_strict(
                                    terminal_path.read_text(encoding="utf-8")
                                )
                                warning_rows = terminal_obj.get("warnings")
                                warnings_count = (
                                    len(warning_rows)
                                    if isinstance(warning_rows, list)
                                    else int(
                                        terminal_obj.get("warnings_count", 0) or 0
                                    )
                                )
                            except Exception:
                                warnings_count = 0
                        rec.update(
                            {
                                "status": "success",
                                "run_dir": str(path),
                                "result_path": str(result_path),
                                "warnings_count": warnings_count,
                                "finished_at": self._batch_now(matrix_plan),
                            }
                        )
                    except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
                        rec.update(
                            {
                                "status": "interrupted",
                                "error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc) or "user interrupt",
                                },
                                "finished_at": self._batch_now(matrix_plan),
                            }
                        )
                        hard_stop = True
                        interrupted_exc = exc
                        self._write_status(
                            status_path,
                            matrix_plan["matrix_id"],
                            status,
                        )
                        break
                    except Exception as exc:
                        message = str(exc)
                        rec.update(
                            {
                                "status": "failed",
                                "error": {
                                    "type": type(exc).__name__,
                                    "message": message,
                                },
                                "finished_at": self._batch_now(matrix_plan),
                            }
                        )
                        failed_dir = (getattr(exc, "details", {}) or {}).get(
                            "run_dir"
                        )
                        if failed_dir:
                            try:
                                failed_run_dir = self._safe_confined_path(
                                    failed_dir,
                                    self.results_root,
                                    label="failed run_dir",
                                    require_dir=True,
                                )
                                if failed_run_dir != self.results_root:
                                    rec["run_dir"] = str(failed_run_dir)
                                    failure_path = failed_run_dir / "error.json"
                                    if not failure_path.is_file():
                                        failure_path = failed_run_dir / "failure.json"
                                    if failure_path.is_file():
                                        failure_obj = load_json_strict(
                                            failure_path.read_text(encoding="utf-8")
                                        )
                                        primary = failure_obj.get(
                                            "error"
                                        ) or failure_obj.get("primary_error")
                                        if isinstance(primary, dict):
                                            rec["error"] = primary
                                        rec["failure_path"] = str(failure_path)
                                        cleanup = failure_obj.get("cleanup") or {}
                                        if isinstance(cleanup, dict):
                                            rec["cleanup_status"] = cleanup.get(
                                                "status"
                                            ) or (cleanup.get("backend") or {}).get(
                                                "status"
                                            )
                            except Exception:
                                pass
                        cleanup_status = rec.get("cleanup_status")
                        if cleanup_status is None:
                            details = getattr(exc, "details", None)
                            if isinstance(details, dict):
                                cleanup_status = details.get("cleanup_status")
                            cleanup_status = cleanup_status or getattr(
                                exc,
                                "_model_eval_cleanup_status",
                                None,
                            )
                            if cleanup_status is not None:
                                rec["cleanup_status"] = cleanup_status
                        hard_stop = self._failure_requires_hard_stop(
                            exc,
                            cleanup_status,
                        )
                        if hard_stop or not keep_going:
                            self._write_status(
                                status_path,
                                matrix_plan["matrix_id"],
                                status,
                            )
                            break
                    self._write_status(
                        status_path,
                        matrix_plan["matrix_id"],
                        status,
                    )
            except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
                hard_stop = True
                interrupted_exc = exc
                for row in status.values():
                    if row.get("status") == "running":
                        row.update(
                            {
                                "status": "interrupted",
                                "error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc) or "matrix interrupted",
                                },
                                "finished_at": self._batch_now(matrix_plan),
                            }
                        )
                if status_path is not None:
                    self._write_status(
                        status_path,
                        matrix_plan["matrix_id"],
                        status,
                    )

            if batch_dir is None:
                if interrupted_exc is not None:
                    raise interrupted_exc
                raise ProcessError(
                    "matrix execution did not establish a batch directory"
                )
            has_failure = any(
                status.get(plan_id, {}).get("status") == "failed"
                for plan_id in planned_ids
            )
            if hard_stop or (has_failure and not keep_going):
                self._fill_not_run_records(matrix_plan, status)
            try:
                summary = self._finalize_batch(
                    batch_dir,
                    matrix_plan,
                    status,
                    hard_stop=hard_stop,
                    keep_going=keep_going,
                    interrupted=interrupted_exc is not None,
                )
            except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
                # If the first termination signal arrives during batch product
                # finalization, record the interruption and retry finalization
                # once so resume state is not silently abandoned.
                hard_stop = True
                interrupted_exc = interrupted_exc or exc
                self._fill_not_run_records(matrix_plan, status)
                summary = self._finalize_batch(
                    batch_dir,
                    matrix_plan,
                    status,
                    hard_stop=True,
                    keep_going=keep_going,
                    interrupted=True,
                )
            if interrupted_exc is not None:
                raise interrupted_exc
            return batch_dir, summary
