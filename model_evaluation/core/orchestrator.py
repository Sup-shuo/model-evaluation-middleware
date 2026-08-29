from __future__ import annotations

import copy
import time
from pathlib import Path

from model_evaluation.core.files import atomic_json
from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.errors import (
    AdapterExecutionError,
    CompatibilityError,
    ProcessError,
)
from model_evaluation.core.process.env import prepare_process_for_environment
from model_evaluation.core.process.manager import ProcessManager
from model_evaluation.core.resources import ResourceManager
from model_evaluation.core.results import build_run_config, iso_now
from model_evaluation.core.runtime_record import (
    refresh_environment_versions,
    runtime_versions_base,
    save_runtime_versions,
    version_text,
)
from model_evaluation.core.run_diagnostics import (
    append_core_error,
    current_state,
    error_record,
    failure_record,
    log_tail,
)
from model_evaluation.core.backend_preflight import (
    preflight_error,
    run_backend_preflight as execute_backend_preflight,
)
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.security import redact_diagnostic
from model_evaluation.core.run_lifecycle import validate_state_transition
from model_evaluation.core.execution_revalidation import revalidate_platform
from model_evaluation.core.orchestrator_execution import execute_plan

class Orchestrator:
    def __init__(
        self,
        *,
        project_root: str | Path,
        schemas: SchemaStore,
        registry,
        process_manager: ProcessManager,
        resource_manager: ResourceManager,
        results_root: str | Path,
        cache_root: str | Path,
    ):
        self.project_root = Path(project_root).resolve()
        self.schemas = schemas
        self.registry = registry
        self.pm = process_manager
        self.resources = resource_manager
        self.results_root = Path(results_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self._warning_events: list[dict] = []
        self._active_plan: dict = {}

    def _invoke(
        self,
        client,
        operation: str,
        input_obj: dict,
        *,
        context: dict | None = None,
        timeout: float | None = None,
        stage: str = "execution",
    ) -> dict:
        out = client.invoke(
            operation,
            input_obj,
            context=context,
            timeout=timeout,
        )
        identity = client.identity
        for message in client.last_warnings:
            self._warning_events.append(
                {
                    "stage": stage,
                    "adapter": f"{identity.kind}/{identity.name}",
                    "operation": operation,
                    "message": str(message),
                }
            )
        return out

    @staticmethod
    def _run_id(plan: dict) -> str:
        """Compatibility helper for callers that only need the base name.

        Actual allocation is atomic and appends ``-2``, ``-3`` ... only when
        two runs start in the same Beijing-time second.
        """
        from model_evaluation.core.results import run_id_base
        return run_id_base(plan)

    def _status(self, run_dir: Path, state: str, **extra) -> None:
        validate_state_transition(current_state(run_dir), state)
        record={'state':state,'time':iso_now(self._active_plan),**extra}
        atomic_json(run_dir/'.run'/'status.json',record)

    def _status_best_effort(self, run_dir: Path, state: str, **extra) -> None:
        try:
            self._status(run_dir, state, **extra)
        except BaseException as exc:
            self._append_core_error(run_dir, state, exc)

    def _persist_initial(self, run_dir: Path, plan: dict, *, started_at: str) -> None:
        for subdirectory in (
            "config",
            "logs",
            ".run/framework_output",
            ".run/task",
            ".run/dataset",
        ):
            (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)
        run_config = build_run_config(
            plan,
            run_id=run_dir.name,
            started_at=started_at,
        )
        self.schemas.validate("run_config", run_config)
        atomic_json(run_dir / "config" / "run_config.json", run_config)

    def _probe_service_until_ready(self, client, attach: dict, auth_value: str | None, timeout_seconds: float, process_handle=None) -> dict:
        deadline = time.monotonic() + timeout_seconds
        last = None
        while time.monotonic() < deadline:
            if process_handle is not None and process_handle.poll() is not None:
                raise ProcessError(f"backend process exited before readiness: rc={process_handle.poll()}")
            try:
                input_obj = {"attach": attach}
                if auth_value is not None:
                    input_obj["auth_value"] = auth_value
                remaining = max(0.5, deadline - time.monotonic())
                attempt_budget = min(20.0, remaining)
                out = self._invoke(
                    client,
                    "probe_service",
                    input_obj,
                    context={
                        "timeout_seconds": attempt_budget,
                        "secure_execution": auth_value is not None,
                    },
                    timeout=attempt_budget + 2.0,
                )
                return out
            except AdapterExecutionError as exc:
                last = exc
                if not exc.retryable:
                    raise
                time.sleep(0.5)
        raise ProcessError(f"service readiness timeout after {timeout_seconds}s: {last}")

    def _runtime_versions_base(self, plan: dict, platform: dict, resolved_platform: dict) -> dict:
        # ``platform`` remains in this compatibility method's signature for
        # callers/tests from earlier releases; the record uses resolved facts.
        del platform
        return runtime_versions_base(plan, resolved_platform)

    def _refresh_environment_versions(self, platform: dict, resolved_platform: dict, value: dict) -> None:
        refresh_environment_versions(
            active_plan=self._active_plan,
            platform=platform,
            resolved_platform=resolved_platform,
            value=value,
            registry=self.registry,
            invoke=self._invoke,
        )

    @staticmethod
    def _version_text(value: object) -> str | None:
        return version_text(value)

    def _save_runtime_versions(self, run_dir: Path, value: dict) -> None:
        save_runtime_versions(run_dir, value, redact=self._redact_diagnostic)

    def prepare_process_for_environment(
        self,
        process: dict,
        *,
        platform_spec: dict,
        resolved_platform: dict,
        role: str,
        base_patches: tuple[tuple[str, dict | None], ...] = (),
        context: dict | None = None,
        timeout: float = 5.0,
    ) -> tuple[dict, list[dict]]:
        """Prepare one ProcessSpec using the exact same environment semantics everywhere.

        ``role`` selects the machine-scoped Environment binding.  Callers may
        provide preceding Device/Runtime patches, but the selected Environment
        always gets the final wrapping step so its executable/PATH semantics
        are authoritative.
        """
        if role not in {"backend", "evaluator"}:
            raise CompatibilityError(f"unsupported execution environment role: {role}")
        key = "backend_environment" if role == "backend" else "evaluation_environment"
        env_sel = platform_spec[key]
        env_desc = resolved_platform[key]
        env_client = self.registry.get("environment", env_sel["provider"])
        before_warnings = len(self._warning_events)

        def _wrap(prepared: dict) -> dict:
            return self._invoke(
                env_client,
                "wrap_process",
                {
                    "process": prepared,
                    "environment": env_desc,
                    "parameters": adapter_parameters(platform_spec, key),
                },
                context=context or {},
                timeout=timeout,
            )["process"]

        wrapped = prepare_process_for_environment(
            process,
            base_patches=base_patches,
            process_owner=role,
            wrap=_wrap,
        )
        return wrapped, copy.deepcopy(self._warning_events[before_warnings:])

    def _run_backend_dependency_probe(
        self,
        run_dir: Path,
        *,
        probe_spec: dict | None,
        platform_spec: dict,
        resolved_platform: dict,
    ) -> dict | None:
        if probe_spec is None:
            return None
        probe = copy.deepcopy(probe_spec)
        probe.setdefault("metadata", {}).setdefault("role", "backend_dependency_probe")
        wrapped, _ = self.prepare_process_for_environment(
            probe,
            platform_spec=platform_spec,
            resolved_platform=resolved_platform,
            role="backend",
            base_patches=(
                ("device", resolved_platform.get("device_env_patch")),
                ("runtime", resolved_platform.get("runtime_env_patch")),
            ),
            context={"diagnostic": True, "preflight": True, "timeout_seconds": 4},
            timeout=5,
        )
        cp = self.pm.run(wrapped)

        def dec(value):
            if value is None:
                return ""
            return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)

        record = {
            "argv": probe.get("argv"),
            "wrapped_argv": wrapped.get("argv"),
            "returncode": cp.returncode,
            "stdout": dec(cp.stdout)[:4000],
            "stderr": dec(cp.stderr)[:4000],
            "environment": (resolved_platform.get("backend_environment") or {}).get("identity"),
            "executable_root": (resolved_platform.get("backend_environment") or {}).get("executable_root"),
        }
        atomic_json(run_dir / ".run" / "diagnostics" / "backend_dependency_probe.json", self._redact_diagnostic(record))
        if cp.returncode != 0:
            raise ProcessError(
                "backend dependency probe failed inside selected environment "
                f"{record['environment']!r}: rc={cp.returncode}: {record['stderr'] or record['stdout']}"
            )
        return record

    @staticmethod
    def _preflight_json_result(stdout: str) -> dict:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                value = json_loads_strict(line)
            except Exception:
                continue
            if isinstance(value, dict):
                return value
        raise ProcessError("preflight probe did not emit a JSON object on its last JSON line")

    def run_backend_preflight(
        self,
        preflight_plan: dict,
        *,
        platform_spec: dict,
        resolved_platform: dict,
        report_path: str | Path | None = None,
        raise_on_failure: bool = True,
    ) -> dict:
        return execute_backend_preflight(
            self,
            preflight_plan,
            platform_spec=platform_spec,
            resolved_platform=resolved_platform,
            report_path=report_path,
            raise_on_failure=raise_on_failure,
        )

    @staticmethod
    def _backend_preflight_error(report: dict) -> ProcessError:
        return preflight_error(report)


    @staticmethod
    def _confined_path(value: str | Path, root: str | Path, *, label: str, reject_symlink: bool=False) -> Path:
        base = Path(root).resolve()
        raw = Path(value)
        if reject_symlink:
            lexical=raw.absolute()
            for candidate in (lexical,*lexical.parents):
                if candidate.is_symlink():
                    raise CompatibilityError(f"{label} may not traverse a symlink: {candidate}")
                if candidate == base:
                    break
        path=raw.resolve()
        if path != base and base not in path.parents:
            raise CompatibilityError(f"{label} escapes its Core-owned root: root={base} path={path}")
        return path

    @staticmethod
    def _verify_canonical_raw_result(result: dict, raw_root: Path) -> None:
        ref = result.get('raw_result') or {}
        path_value = ref.get('path')
        if not path_value:
            raise CompatibilityError('CanonicalResult.raw_result.path is missing')
        path = Orchestrator._confined_path(
            path_value,
            raw_root,
            label='CanonicalResult.raw_result',
            reject_symlink=True,
        )
        if not path.is_file():
            raise CompatibilityError(f'CanonicalResult raw result is missing/unsafe: {path}')

    @staticmethod
    def _verify_task_artifacts(task: dict, staging_root: Path) -> None:
        staging=Path(staging_root).resolve()
        strict=bool((task.get('provenance') or {}).get('strict'))
        task_root_value=task.get('task_root')
        task_root=None
        if task_root_value:
            task_root=Orchestrator._confined_path(task_root_value,staging,label='FrameworkTaskArtifact.task_root',reject_symlink=True)
            if not task_root.is_dir():
                raise CompatibilityError(f'FrameworkTaskArtifact.task_root is missing/unsafe: {task_root}')
        elif strict:
            raise CompatibilityError('strict FrameworkTaskArtifact requires a Core-confined task_root')
        for artifact in task.get('artifacts') or []:
            raw_path=Path(artifact['path'])
            if raw_path.is_symlink():
                raise CompatibilityError(f'framework task protocol artifact is a symlink: {raw_path}')
            path=raw_path.resolve()
            if task_root is not None:
                path=Orchestrator._confined_path(raw_path,task_root,label='FrameworkTaskArtifact.artifact',reject_symlink=True)
            elif strict:
                raise CompatibilityError('strict FrameworkTaskArtifact artifact is not confined to task_root')
            if not path.is_file():
                raise CompatibilityError(f'framework task protocol artifact disappeared/unsafe: {path}')

    @staticmethod
    def _verify_dataset_identity(dataset_resolution: dict, benchmark: dict, artifact: dict) -> None:
        expected_id=(dataset_resolution or {}).get('dataset_id')
        if expected_id is not None and artifact.get('dataset_id') != expected_id:
            raise CompatibilityError(f'dataset artifact id changed after planning: expected {expected_id!r}, got {artifact.get("dataset_id")!r}')
        declared_revision=((benchmark.get('dataset') or {}).get('revision'))
        if declared_revision is not None and artifact.get('revision') != declared_revision:
            raise CompatibilityError(f'dataset artifact revision disagrees with BenchmarkSpec: expected {declared_revision!r}, got {artifact.get("revision")!r}')

    def _revalidate_platform(self, plan: dict, run_dir: Path) -> dict:
        return revalidate_platform(self, plan, run_dir)

    def _redact_diagnostic(self, value):
        return redact_diagnostic(value, self.pm.secrets.redaction_values())

    def _error_record(self, exc: BaseException) -> dict:
        return error_record(exc, redact=self._redact_diagnostic)

    @staticmethod
    def _current_state(run_dir: Path) -> str | None:
        return current_state(run_dir)

    def _append_core_error(self, run_dir: Path, stage: str, exc: BaseException) -> None:
        append_core_error(
            run_dir,
            stage,
            exc,
            timestamp=iso_now(self._active_plan),
            redaction_values=self.pm.secrets.redaction_values(),
        )

    def _log_tail(self, path: Path, *, max_lines: int = 40, max_chars: int = 8000, max_bytes: int = 65536) -> list[str]:
        return log_tail(
            path,
            max_lines=max_lines,
            max_chars=max_chars,
            max_bytes=max_bytes,
            redaction_values=self.pm.secrets.redaction_values(),
        )

    def _failure_record(self, run_dir: Path, *, stage: str, failure: BaseException, cleanup: dict, backend_handle=None, evaluator_returncode=None) -> dict:
        return failure_record(
            run_dir,
            stage=stage,
            failure=failure,
            cleanup=cleanup,
            timestamp=iso_now(self._active_plan),
            error_builder=self._error_record,
            redact=self._redact_diagnostic,
            redaction_values=self.pm.secrets.redaction_values(),
            backend_handle=backend_handle,
            evaluator_returncode=evaluator_returncode,
        )

    def execute(self, plan: dict) -> Path:
        return execute_plan(self, plan)
