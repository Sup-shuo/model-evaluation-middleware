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
    facts_from_service,
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
from model_evaluation.core.results import allocate_run_dir, build_run_config, iso_now, plan_timezone, publish_result
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
        atomic_json(
            run_dir / "config" / "run_config.json",
            build_run_config(
                plan,
                run_id=run_dir.name,
                started_at=started_at,
            ),
        )

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
            reports['device_runtime_pair']={'compatible':pair.compatible,'reasons':pair.reasons,'optional_misses':pair.optional_misses}
            if not pair.compatible:
                raise CompatibilityError('; '.join(pair.reasons))
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
            reports[name]={'compatible':report.compatible,'reasons':report.reasons,'optional_misses':report.optional_misses}
            if not report.compatible:
                atomic_json(run_dir/'.run'/'diagnostics'/'execution_preflight_compatibility.json',{'compatible':False,'reports':reports})
                raise CompatibilityError(f'{name} requirements changed after planning: ' + '; '.join(report.reasons))

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
        self.schemas.validate('execution_plan',plan)
        if plan['compatibility']['status']=='incompatible':
            raise CompatibilityError('; '.join(plan['compatibility'].get('reasons') or ['plan is incompatible']))
        self._active_plan=plan
        non_global_claims=[c for c in plan['resources'] if c['kind']!='run_lock']
        with orchestration_signal_guard(), self.resources.run_lock():
            stale=self.pm.recover_stale_managed()
            unresolved=[x for x in stale if x.get('status') in {'identity_mismatch','cleanup_failed','invalid','orphaned_group_ambiguous'}]
            if unresolved:
                raise StaleProcessError(f"unresolved stale managed-process ownership records: {unresolved}")
            with self.resources.acquire(non_global_claims):
                started_at=iso_now(plan)
                run_dir = allocate_run_dir(self.results_root, plan)
                run_id = run_dir.name
                backend_handle = None
                failure: BaseException | None = None
                failure_stage: str | None = None
                backend_shutdown: dict | None = None
                mode: str | None = None
                evaluator_returncode: int | None = None
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
                    specs = plan['resolved']['specs']
                    platform = specs['platform']
                    deployment = specs['deployment']
                    benchmark = specs['benchmark']
                    evaluation = specs['evaluation']
                    model = specs['model']
                    mode = deployment['management']['mode']
                    offline=bool((plan['run_spec'].get('overrides') or {}).get('offline',False))
                    pdesc=self._revalidate_platform(plan,run_dir)
                    runtime_versions=self._runtime_versions_base(plan,platform,pdesc)
                    self._refresh_environment_versions(platform,pdesc,runtime_versions)
                    self._save_runtime_versions(run_dir,runtime_versions)
                    self._status(run_dir,'PLATFORM_READY')
                    dataset_client=self.registry.get('dataset',benchmark['dataset']['provider'])
                    dataset=self._invoke(dataset_client,'prepare',{'benchmark':benchmark,'cache_root':str(self.cache_root)},context={'cache_root':str(self.cache_root),'workspace':str(run_dir/'.run'/'dataset'),'offline':offline,'network_policy':'offline' if offline else 'online'},timeout=float((plan['run_spec'].get('overrides') or {}).get('dataset_timeout_seconds',600)))
                    verified=self._invoke(dataset_client,'verify',{'artifact':dataset,'benchmark':benchmark},context={'offline':True},timeout=30)
                    if not verified['valid']:
                        details=verified.get('details') or {}
                        raise ModelEvalError(f'dataset verification failed: {details}')
                    dataset=verified.get('artifact') or dataset
                    self._verify_dataset_identity(plan['resolved'].get('dataset_resolution') or {},benchmark,dataset)
                    self._status(run_dir,'DATA_READY')
                    binding=self.registry.get('binding',plan['resolved']['binding_adapter'])
                    task_input={'benchmark':benchmark,'dataset_artifact':dataset,'staging_root':str(run_dir/'.run'/'task'),'evaluation':evaluation}
                    task=self._invoke(binding,'build_task',task_input,context={'workspace':str(run_dir/'.run'/'task'),'offline':offline},timeout=60)
                    fp=self._invoke(binding,'protocol_fingerprint',{'benchmark':benchmark,'dataset_artifact':dataset,'evaluation':evaluation},context={'workspace':str(run_dir/'.run'/'task'),'offline':offline},timeout=30)
                    if fp['protocol_fingerprint'] != task['protocol_fingerprint']:
                        raise CompatibilityError(f"binding protocol_fingerprint disagrees with FrameworkTaskArtifact: {fp['protocol_fingerprint']} != {task['protocol_fingerprint']}")
                    self._verify_task_artifacts(task,run_dir/'.run'/'task')
                    self._status(run_dir, 'TASK_READY')
                    self._status(run_dir, 'EVALUATOR_PREFLIGHT')
                    evaluator=self.registry.get('evaluator',evaluation['framework']['adapter'])
                    runtime_versions['evaluator']={
                        'adapter':evaluation['framework']['adapter'],
                        'adapter_version':evaluator.identity.version,
                    }
                    self._save_runtime_versions(run_dir,runtime_versions)
                    # Framework source probes may each spend up to 10 seconds on
                    # shared/NFS worktrees.  The Adapter RPC budget must be larger
                    # than the Adapter's own bounded probes, otherwise Core would
                    # terminate an otherwise healthy check first.
                    req=self._invoke(evaluator,'requirements',{'evaluation':evaluation,'task':task},context={'workspace':str(run_dir),'offline':offline,'preflight':True},timeout=30)
                    local_req={'schema_version':'1.0','requirements':[r for r in req.get('requirements',[]) if str(r.get('path','')).startswith('evaluation_environment.')]}
                    self.schemas.validate('requirement_set',local_req)
                    local_report=evaluate(local_req,facts_from_environment(pdesc['evaluation_environment'],'evaluation_environment'))
                    if not local_report.compatible:
                        raise CompatibilityError(
                            'evaluator preflight failed: ' + '; '.join(local_report.reasons)
                        )
                    # Evaluators may provide a cheap process-level preflight that is
                    # executed inside the selected evaluation environment. This is
                    # deliberately before Backend startup so missing Python packages
                    # or framework dependencies fail without loading the model.
                    if 'plan_preflight' in (evaluator.identity.manifest.get('operations') or []):
                        preflight=self._invoke(evaluator,'plan_preflight',{'evaluation':evaluation,'task':task,'cache_root':str(self.cache_root)},context={'workspace':str(run_dir),'cache_root':str(self.cache_root),'offline':offline,'preflight':True},timeout=30)
                        pre_proc = copy.deepcopy(preflight['process'])
                        pre_proc.setdefault('metadata', {}).update(
                            {'role': 'evaluator_preflight', 'run_id': run_id}
                        )
                        pre_wrapped,_=self.prepare_process_for_environment(
                            pre_proc,platform_spec=platform,resolved_platform=pdesc,role='evaluator',
                            context={'workspace':str(run_dir),'offline':offline,'preflight':True},timeout=5,
                        )
                        pre_cp=self.pm.run(pre_wrapped)
                        def _preflight_text(value):
                            if value is None:
                                return ''
                            return value.decode('utf-8','replace') if isinstance(value,(bytes,bytearray)) else str(value)
                        preflight_record={
                                'returncode':pre_cp.returncode,
                                'stdout':_preflight_text(pre_cp.stdout)[:8000],
                                'stderr':_preflight_text(pre_cp.stderr)[:8000],
                            }
                        if preflight.get('result_format')=='preflight_result':
                            try:
                                probe_result=self._preflight_json_result(_preflight_text(pre_cp.stdout))
                                self.schemas.validate('preflight_probe_result',probe_result)
                                preflight_record['result']=probe_result
                                runtime_versions['evaluator']['facts']=copy.deepcopy(probe_result.get('facts') or {})
                                self._save_runtime_versions(run_dir,runtime_versions)
                                process_passed = pre_cp.returncode == 0
                                result_passed = probe_result['status'] == 'passed'
                                if process_passed != result_passed:
                                    raise ProcessError(
                                        'evaluator preflight process/result status mismatch: '
                                        f"returncode={pre_cp.returncode}, result.status={probe_result['status']!r}"
                                    )
                                if not result_passed:
                                    error=probe_result['error']
                                    raise ProcessError(f"evaluator preflight failed: {error['code']}: {error['message']}")
                            finally:
                                atomic_json(run_dir/'.run'/'diagnostics'/'evaluator_preflight.json',self._redact_diagnostic(preflight_record))
                        else:
                            atomic_json(run_dir/'.run'/'diagnostics'/'evaluator_preflight.json',self._redact_diagnostic(preflight_record))
                            if pre_cp.returncode != 0:
                                raise ProcessError(f'evaluator dependency preflight failed with rc={pre_cp.returncode}')
                    backend=self.registry.get('backend',deployment['backend']['adapter'])
                    start_input={'model':model,'deployment':deployment,'platform':pdesc,'endpoint':plan['resolved'].get('endpoint',{}),'log_path':str(run_dir/'logs'/'backend.log'),'network_policy':'offline' if offline else 'online'}
                    start_plan=self._invoke(backend,'plan_start',start_input,context={'workspace':str(run_dir), 'offline':offline},timeout=5)
                    attach = start_plan['attach']
                    backend_shutdown = copy.deepcopy(start_plan.get('shutdown'))
                    if mode=='managed':
                        self._status(run_dir,'SERVICE_STARTING')
                        proc = copy.deepcopy(start_plan['process'])
                        proc.setdefault('metadata', {}).update(
                            {'role': 'backend', 'run_id': run_id}
                        )
                        p=pdesc
                        proc,_=self.prepare_process_for_environment(
                            proc,platform_spec=platform,resolved_platform=p,role='backend',
                            base_patches=(("device",p.get('device_env_patch')),("runtime",p.get('runtime_env_patch'))),
                            context={'workspace':str(run_dir),'offline':offline},timeout=5,
                        )
                        if 'plan_preflight' in (backend.identity.manifest.get('operations') or []):
                            preflight_input={
                                'model':model,'deployment':deployment,'platform':pdesc,
                                'network_policy':'offline' if offline else 'online',
                            }
                            preflight_plan=self._invoke(
                                backend,'plan_preflight',preflight_input,
                                context={'workspace':str(run_dir),'offline':offline,'preflight':True},timeout=5,
                            )
                            backend_report=self.run_backend_preflight(
                                preflight_plan,platform_spec=platform,resolved_platform=p,
                                raise_on_failure=False,
                            )
                            backend_runtime={'adapter':deployment['backend']['adapter'],'adapter_version':backend.identity.version,'probes':[]}
                            for row in backend_report.get('probes') or []:
                                compact={'id':row.get('id'),'status':row.get('status')}
                                version=self._version_text(row.get('stdout'))
                                if version: compact['version']=version
                                facts=((row.get('result') or {}).get('facts'))
                                if isinstance(facts,dict): compact['facts']=copy.deepcopy(facts)
                                backend_runtime['probes'].append(compact)
                            runtime_versions['backend']=backend_runtime
                            self._save_runtime_versions(run_dir,runtime_versions)
                            if backend_report.get('status') != 'passed':
                                raise self._backend_preflight_error(backend_report)
                        else:
                            dependency_record=self._run_backend_dependency_probe(
                                run_dir,probe_spec=start_plan.get('dependency_probe'),platform_spec=platform,resolved_platform=p,
                            )
                            runtime_versions['backend']={
                                'adapter':deployment['backend']['adapter'],
                                'adapter_version':backend.identity.version,
                                'version':self._version_text((dependency_record or {}).get('stdout')),
                            }
                        self._save_runtime_versions(run_dir,runtime_versions)
                        # Re-check immediately before releasing the external OS port race window to the backend.
                        for claim in plan['resources']:
                            if claim['kind']=='port': self.resources.check_port(str(claim.get('host') or '127.0.0.1'),int(claim['id']))
                        backend_handle=self.pm.start(proc)
                    auth_ref = (attach.get('auth') or {}).get('secret_ref')
                    auth_value = self.pm.secrets.resolve(auth_ref) if auth_ref else None
                    ready=float((start_plan.get('readiness') or {}).get('timeout_seconds',30 if mode!='managed' else 900))
                    service = self._probe_service_until_ready(
                        backend,
                        attach,
                        auth_value,
                        ready,
                        backend_handle,
                    )
                    self._status(run_dir, 'SERVICE_READY')
                    if mode!='managed':
                        runtime_versions['backend']={'adapter':deployment['backend']['adapter'],'adapter_version':backend.identity.version,'management':mode}
                    self._save_runtime_versions(run_dir,runtime_versions)
                    eval_env=pdesc['evaluation_environment']
                    eval_facts=merge_fact_sets(facts_from_service(service),facts_from_environment(eval_env,'evaluation_environment'))
                    report=evaluate(req,eval_facts)
                    if not report.compatible:
                        raise CompatibilityError('; '.join(report.reasons))
                    self._verify_task_artifacts(task,run_dir/'.run'/'task')
                    # Strict datasets are verified again immediately before evaluator planning/execution.
                    final_verified=self._invoke(dataset_client,'verify',{'artifact':dataset,'benchmark':benchmark},context={'offline':True,'final_verification':True},timeout=30)
                    if not final_verified['valid']:
                        raise ModelEvalError(f'dataset final verification failed: {final_verified.get("details") or {}}')
                    final_artifact=final_verified.get('artifact') or dataset
                    self._verify_dataset_identity(plan['resolved'].get('dataset_resolution') or {},benchmark,final_artifact)
                    if final_artifact.get('fingerprint') != dataset.get('fingerprint'):
                        raise CompatibilityError('dataset artifact fingerprint changed after task binding')
                    ep=self._invoke(evaluator,'plan_evaluate',{'service':service,'task':task,'evaluation':evaluation,'cache_root':str(self.cache_root),'output_root':str(run_dir/'.run'/'framework_output'),'workspace':str(run_dir),'log_path':str(run_dir/'logs'/'evaluation.log'),'network_policy':'offline' if offline else 'online'},context={'workspace':str(run_dir),'cache_root':str(self.cache_root),'offline':offline},timeout=30)
                    raw_result_root = self._confined_path(
                        ep['raw_result_root'],
                        run_dir / '.run' / 'framework_output',
                        label='evaluator raw_result_root',
                    )
                    ep['raw_result_root'] = str(raw_result_root)
                    eval_proc = ep['process']
                    eval_proc.setdefault('metadata', {}).update(
                        {'role': 'evaluator', 'run_id': run_id}
                    )
                    eval_proc,_=self.prepare_process_for_environment(
                        eval_proc,platform_spec=platform,resolved_platform=pdesc,role='evaluator',
                        context={'workspace':str(run_dir),'offline':offline},timeout=5,
                    )
                    self._status(run_dir, 'EVALUATING')
                    cp = self.pm.run(eval_proc)
                    evaluator_returncode = cp.returncode
                    if cp.returncode != 0:
                        raise ProcessError(f"evaluator exited with rc={cp.returncode}")
                    self._status(run_dir,'NORMALIZING')
                    result_model=str(model.get('experiment_id') or (model.get('metadata') or {}).get('experiment_id') or model['id'])
                    result=self._invoke(evaluator,'normalize',{'raw_result_root':ep['raw_result_root'],'task':task,'run_metadata':{'run_id':run_id,'model':result_model,'benchmark':benchmark['id']}},context={'workspace':str(run_dir)},timeout=20)
                    expected_framework=evaluation['framework']['adapter']
                    if result.get('run_id')!=run_id or result.get('model')!=result_model or result.get('benchmark')!=benchmark['id'] or result.get('framework')!=expected_framework:
                        raise CompatibilityError('CanonicalResult identity disagrees with the executing run/evaluator')
                    task_metrics=task.get('metrics') or {}
                    if task_metrics.get('namespace')=='canonical':
                        missing=[name for name in (benchmark.get('metrics') or []) if name not in (result.get('metrics') or {})]
                        if missing:
                            raise CompatibilityError(
                                f'CanonicalResult is missing BenchmarkSpec metrics: {missing}'
                            )
                    self._verify_canonical_raw_result(result,raw_result_root)
                    result['metadata']={**(result.get('metadata') or {}),'started_at':started_at,'finished_at':iso_now(plan),'timezone':getattr(plan_timezone(plan),'key','Asia/Shanghai')}
                    result=publish_result(run_dir,raw_result_root,result,schemas=self.schemas)
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
                        run_id=run_id,
                        plan=plan,
                        mode=mode,
                        started_at=started_at,
                        failure=failure,
                        failure_stage=failure_stage,
                        backend_handle=backend_handle,
                        backend_shutdown=backend_shutdown,
                        evaluator_returncode=evaluator_returncode,
                    )
                if failure is not None:
                    if isinstance(failure, ModelEvalError):
                        failure.details.setdefault('run_dir',str(run_dir))
                    raise failure
        return run_dir
