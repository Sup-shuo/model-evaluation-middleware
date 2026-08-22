from __future__ import annotations

import copy
import time
from pathlib import Path

from model_evaluation.core.files import atomic_json
from model_evaluation.core.compatibility import (
    device_runtime_compatibility,
    evaluate,
    facts_from_device,
    facts_from_runtime,
    facts_from_environment,
    merge_fact_sets,
)
from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.errors import (
    AdapterExecutionError,
    CompatibilityError,
    ModelEvalError,
    ProcessError,
    StaleProcessError,
)
from model_evaluation.core.process.env import prepare_process_for_environment
from model_evaluation.core.process.manager import ProcessManager
from model_evaluation.core.process.signals import orchestration_signal_guard
from model_evaluation.core.resources import ResourceManager
from model_evaluation.core.results import allocate_run_dir, build_run_config, iso_now
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
from model_evaluation.core.run_finalization import finalize_run
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.security import redact_diagnostic
from model_evaluation.core.execution_plan import validate_execution_plan
from model_evaluation.core.run_lifecycle import RunContext, validate_state_transition
from model_evaluation.core.orchestration_pipeline import (
    evaluate_and_publish,
    prepare_dataset_and_task,
    prepare_platform,
    preflight_evaluator,
    start_service,
)

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

    @staticmethod
    def _verify_environment_identity(label: str, current: dict, expected: dict) -> None:
        for key in ('provider','identity','python','executable_root'):
            expected_value=expected.get(key)
            if expected_value is not None and current.get(key) != expected_value:
                raise CompatibilityError(f'{label} {key} changed after planning: expected {expected_value!r}, got {current.get(key)!r}')

    def _revalidate_platform(self, plan: dict, run_dir: Path) -> dict:
        """Re-probe the execution platform and re-evaluate all pre-service requirements.

        Identity checks protect the frozen plan, while requirement re-evaluation
        catches capability drift that does not necessarily change identity/version.
        The returned descriptors are the fresh execution-time descriptors and are
        used for process wrapping instead of stale planning snapshots.
        """
        specs = plan['resolved']['specs']
        deployment = specs['deployment']
        platform = specs['platform']
        expected = plan['resolved']['platform']
        eval_client=self.registry.get('environment',platform['evaluation_environment']['provider'])
        current_eval=self._invoke(eval_client,'resolve',{'profile':platform['evaluation_environment']['profile'],'parameters':adapter_parameters(platform,'evaluation_environment')},context={'timeout_seconds':3,'execution_revalidation':True},timeout=4)
        self._verify_environment_identity('evaluation environment',current_eval,expected['evaluation_environment'])

        fresh=copy.deepcopy(expected)
        fresh['evaluation_environment']=current_eval
        facts=merge_fact_sets(facts_from_environment(current_eval,'evaluation_environment'))
        reports: dict[str,dict] = {}

        if deployment['management']['mode'] == 'managed':
            dev_client=self.registry.get('device',platform['device']['adapter'])
            device_params = adapter_parameters(platform, 'device')
            runtime_params = adapter_parameters(platform, 'runtime')
            current=self._invoke(dev_client,'probe',{'requested_devices':platform['device'].get('devices',[]),'parameters':device_params},context={'timeout_seconds':3,'execution_revalidation':True},timeout=4)
            wanted = {device['id'] for device in expected['device']['devices']}
            got = {device['id'] for device in current['devices']}
            if current['vendor'] != expected['device']['vendor'] or got != wanted:
                raise CompatibilityError(f'device facts changed after planning: expected vendor/devices={expected["device"]["vendor"]}/{sorted(wanted)}, got={current["vendor"]}/{sorted(got)}')
            current_visibility=self._invoke(dev_client,'visibility',{'devices':[d['id'] for d in current['devices']],'descriptor':current,'parameters':device_params},context={'timeout_seconds':3,'execution_revalidation':True},timeout=4)['env_patch']
            if current_visibility != expected.get('device_env_patch',{}):
                raise CompatibilityError('device visibility EnvPatch changed after planning')

            rt_client=self.registry.get('runtime',platform['runtime']['adapter'])
            runtime=self._invoke(rt_client,'probe',{'profile':platform['runtime'].get('profile'),'parameters':runtime_params},context={'timeout_seconds':3,'execution_revalidation':True},timeout=4)
            if runtime['family'] != expected['runtime']['family'] or not runtime['available']:
                raise CompatibilityError(f'runtime facts changed after planning: expected {expected["runtime"]["family"]}, got {runtime.get("family")} available={runtime.get("available")}')
            if expected['runtime'].get('version') not in (None,'unknown') and runtime.get('version') != expected['runtime'].get('version'):
                raise CompatibilityError(f'runtime version changed after planning: expected {expected["runtime"].get("version")}, got {runtime.get("version")}')
            pair=device_runtime_compatibility(current,runtime)
            reports['device_runtime_pair']={'compatible':pair.compatible,'reasons':pair.reasons,'optional_misses':pair.optional_misses,'diagnostics':pair.diagnostics}
            if not pair.compatible:
                raise CompatibilityError('; '.join(pair.reasons), details={'diagnostics': pair.diagnostics})
            current_runtime_patch=self._invoke(rt_client,'resolve_environment',{'descriptor':runtime,'profile':platform['runtime'].get('profile'),'parameters':runtime_params},context={'timeout_seconds':3,'execution_revalidation':True},timeout=4)['env_patch']
            if current_runtime_patch != expected.get('runtime_env_patch',{}):
                raise CompatibilityError('runtime EnvPatch changed after planning')

            be_client=self.registry.get('environment',platform['backend_environment']['provider'])
            current_be=self._invoke(be_client,'resolve',{'profile':platform['backend_environment']['profile'],'parameters':adapter_parameters(platform,'backend_environment')},context={'timeout_seconds':3,'execution_revalidation':True},timeout=4)
            self._verify_environment_identity('backend environment',current_be,expected['backend_environment'])

            fresh.update({'device':current,'runtime':runtime,'backend_environment':current_be,'device_env_patch':current_visibility,'runtime_env_patch':current_runtime_patch})
            facts=merge_fact_sets(facts,facts_from_device(current),facts_from_runtime(runtime),facts_from_environment(current_be,'backend_environment'))

        for name,key in (
            ('deployment_compatibility','deployment_compatibility_requirements'),
            ('backend','backend_requirements'),
            ('binding','binding_requirements'),
        ):
            req=plan['resolved'].get(key)
            if not req:
                continue
            report=evaluate(req,facts)
            reports[name]={'compatible':report.compatible,'reasons':report.reasons,'optional_misses':report.optional_misses,'diagnostics':report.diagnostics}
            if not report.compatible:
                atomic_json(run_dir/'.run'/'diagnostics'/'execution_preflight_compatibility.json',{'compatible':False,'reports':reports})
                raise CompatibilityError(
                    f'{name} requirements changed after planning: ' + '; '.join(report.reasons),
                    details={'diagnostics': report.diagnostics},
                )

        atomic_json(run_dir/'.run'/'diagnostics'/'execution_preflight_compatibility.json',{'compatible':True,'reports':reports})
        return fresh

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
        validate_execution_plan(plan,self.schemas)
        if plan['compatibility']['status']=='incompatible':
            raise CompatibilityError(
                '; '.join(plan['compatibility'].get('reasons') or ['plan is incompatible']),
                details={'diagnostics': plan['compatibility'].get('diagnostics') or []},
            )
        non_global_claims=[c for c in plan['resources'] if c['kind']!='run_lock']
        with orchestration_signal_guard(), self.resources.run_lock():
            self._active_plan=plan
            stale=self.pm.recover_stale_managed()
            unresolved=[x for x in stale if x.get('status') in {'identity_mismatch','cleanup_failed','invalid','orphaned_group_ambiguous'}]
            if unresolved:
                raise StaleProcessError(f"unresolved stale managed-process ownership records: {unresolved}")
            with self.resources.acquire(non_global_claims):
                started_at=iso_now(plan)
                run_dir = allocate_run_dir(self.results_root, plan)
                context = RunContext(
                    plan=plan,
                    run_dir=run_dir,
                    started_at=started_at,
                    stale_recovery=copy.deepcopy(stale),
                )
                failure: BaseException | None = None
                failure_stage: str | None = None
                self._warning_events=[copy.deepcopy(x) for x in (plan.get('warnings') or [])]
                try:
                    self._persist_initial(run_dir,plan,started_at=started_at)
                    if stale:
                        atomic_json(
                            run_dir / '.run' / 'diagnostics' / 'stale_process_recovery.json',
                            stale,
                        )
                    self._status(run_dir, 'CREATED')
                    self._status(run_dir, 'PLANNED')
                    prepare_platform(self, context)
                    prepare_dataset_and_task(self, context)
                    preflight_evaluator(self, context)
                    start_service(self, context)
                    evaluate_and_publish(self, context)
                except BaseException as exc:
                    failure=exc
                    failure_stage=self._current_state(run_dir) or 'INITIALIZING'
                    self._append_core_error(run_dir,failure_stage,exc)
                    try: self._status(run_dir,'FAILED',error=self._error_record(exc),failure_stage=failure_stage)
                    except Exception: pass
                finally:
                    failure, failure_stage = finalize_run(
                        self,
                        run_dir=run_dir,
                        run_id=context.run_id,
                        plan=plan,
                        mode=context.mode,
                        started_at=started_at,
                        failure=failure,
                        failure_stage=failure_stage,
                        backend_handle=context.backend_handle,
                        backend_shutdown=context.backend_shutdown,
                        evaluator_returncode=context.evaluator_returncode,
                    )
                if failure is not None:
                    if isinstance(failure, ModelEvalError):
                        failure.details.setdefault('run_dir',str(run_dir))
                    raise failure
        return run_dir
