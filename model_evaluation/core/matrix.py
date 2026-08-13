from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jsonschema import Draft202012Validator

from model_evaluation.core.files import atomic_json, atomic_text
from model_evaluation.core.config.overrides import validate_run_overrides
from model_evaluation.core.config.loader import reject_inline_secrets
from model_evaluation.core.config.parsing import load_json_strict, load_yaml_strict
from model_evaluation.core.process.signals import orchestration_signal_guard
from model_evaluation.core.errors import ConfigError, ProcessError, CleanupCriticalError, OrchestrationInterruptedError, StaleProcessError
from model_evaluation.core.identifiers import stable_id
from model_evaluation.core.planner import Planner
from model_evaluation.core.resources import ResourceManager
from model_evaluation.core.result_relocation import ResultRelocationMap, load_result_relocation
from model_evaluation.core.schema.formats import contract_format_checker

AXES = ("model", "platform", "deployment", "benchmark", "evaluation")
AXIS_KEYS = {"model":"models","platform":"platforms","deployment":"deployments","benchmark":"benchmarks","evaluation":"evaluations"}
_DEFAULT_MAX_COMBINATIONS = 100_000


class MatrixSchemas:
    def __init__(self, root: str | Path):
        self.root=Path(root).resolve()

    def _load(self,name:str)->dict:
        p=self.root/f"{name}.schema.json"
        if not p.is_file(): raise ConfigError(f"matrix schema missing: {p}")
        return load_json_strict(p.read_text(encoding='utf-8'))

    def validate(self,name:str,obj:object)->None:
        schema=self._load(name); Draft202012Validator.check_schema(schema)
        errors=sorted(Draft202012Validator(schema,format_checker=contract_format_checker()).iter_errors(obj),key=lambda e:list(e.absolute_path))
        if errors:
            e=errors[0]; path='.'.join(map(str,e.absolute_path)) or '<root>'
            raise ConfigError(f"{name} validation failed at {path}: {e.message}")


class MatrixRepository:
    def __init__(self, root: str | Path, schemas: MatrixSchemas):
        self.root=Path(root).resolve(); self.schemas=schemas

    def load(self, value: str | Path) -> dict:
        p=Path(value); expected_id=None
        if not p.is_file():
            expected_id=str(value)
            base=self.root
            if not value or Path(value).is_absolute() or any(part in {'','.','..'} for part in Path(value).parts):
                raise ConfigError(f'invalid/path-escaping matrix spec id: {value!r}')
            matches=[(base/f"{value}{ext}").resolve() for ext in ('.yaml','.yml','.json')]
            matches=[x for x in matches if x.is_file() and (base==x.parent or base in x.parents)]
            if len(matches)!=1: raise ConfigError(f"expected exactly one matrix spec for {value!r}, found {len(matches)}")
            p=matches[0]
        p=p.resolve()
        try:
            obj=load_json_strict(p.read_text(encoding='utf-8')) if p.suffix.lower()=='.json' else load_yaml_strict(p.read_text(encoding='utf-8'))
        except Exception as exc: raise ConfigError(f"failed to parse matrix spec {p}: {exc}") from exc
        if not isinstance(obj,dict): raise ConfigError(f"matrix spec must be an object: {p}")
        reject_inline_secrets(obj,str(p))
        self.schemas.validate('matrix_spec',obj)
        if expected_id is not None and obj.get('id') != expected_id:
            raise ConfigError(f'matrix spec filename/reference {expected_id!r} disagrees with id {obj.get("id")!r}')
        return obj


def _excluded(combo: dict[str,str], rules: list[dict]) -> bool:
    return any(all(combo.get(k)==v for k,v in rule.items()) for rule in rules)


def _merge_dict(base: dict, patch: dict) -> dict:
    out=copy.deepcopy(base)
    for key,value in patch.items():
        if isinstance(value,dict) and isinstance(out.get(key),dict): out[key]=_merge_dict(out[key],value)
        else: out[key]=copy.deepcopy(value)
    return out


def finalize_matrix_plan(obj: dict) -> None:
    obj['matrix_id']='matrix-'+stable_id(obj,length=24,exclude_keys={'matrix_id'})


def _expected_runs_from_spec(spec: dict, *, app) -> list[dict]:
    return MatrixPlanner(app).expand(spec)


def verify_matrix_plan(obj: dict, *, app) -> None:
    app.matrix_schemas.validate('matrix_plan',obj)
    expected_id='matrix-'+stable_id(obj,length=24,exclude_keys={'matrix_id'})
    if obj.get('matrix_id') != expected_id:
        raise ConfigError('matrix_id does not match normalized matrix plan')
    app.matrix_schemas.validate('matrix_spec',obj.get('matrix_spec'))
    plans=obj.get('plans') or []
    if not plans: raise ConfigError('matrix plan contains no child plans')
    plan_ids=[]
    for plan in plans:
        app.schemas.validate('execution_plan',plan)
        expected_plan_id='plan-'+stable_id(plan,length=24,exclude_keys={'plan_id'})
        if plan.get('plan_id') != expected_plan_id:
            raise ConfigError('child plan_id does not match normalized execution plan')
        plan_ids.append(plan['plan_id'])
    if len(plan_ids) != len(set(plan_ids)):
        raise ConfigError('matrix plan contains duplicate child plan_id values')
    expected_runs=_expected_runs_from_spec(obj['matrix_spec'],app=app)
    actual_runs=[p.get('run_spec') for p in plans]
    if actual_runs != expected_runs:
        raise ConfigError('matrix child plans do not exactly match deterministic MatrixSpec expansion')
    summary=obj.get('summary') or {}
    incompatible=sum(1 for p in plans if p.get('compatibility',{}).get('status')=='incompatible')
    if summary.get('runs') != len(plans) or summary.get('incompatible') != incompatible:
        raise ConfigError('matrix summary does not match child plans')


class MatrixPlanner:
    def __init__(self, app): self.app=app

    def _planner(self, specs=None):
        if specs is None or specs is self.app.specs:
            return self.app.planner
        return Planner(
            project_root=self.app.root,
            schemas=self.app.schemas,
            specs=specs,
            registry=self.app.registry,
        )

    def expand(self, spec: dict) -> list[dict]:
        self.app.matrix_schemas.validate('matrix_spec',spec)
        values=[spec[AXIS_KEYS[a]] for a in AXES]; excludes=spec.get('exclude') or []
        execution=spec.get('execution') or {}; max_runs=int(execution.get('max_runs',10000)); max_combinations=int(execution.get('max_combinations',_DEFAULT_MAX_COMBINATIONS))
        total=math.prod(len(v) for v in values)
        if total > max_combinations:
            raise ConfigError(f'matrix Cartesian product has {total} combinations, exceeding max_combinations={max_combinations}; split the matrix or raise the explicit safety bound')
        unknown_models=set(spec.get('per_model_overrides') or {})-set(spec['models'])
        if unknown_models: raise ConfigError(f'per_model_overrides references models outside matrix axis: {sorted(unknown_models)}')
        axis_values={a:set(spec[AXIS_KEYS[a]]) for a in AXES}
        for rule in excludes:
            for axis,value in rule.items():
                if value not in axis_values[axis]: raise ConfigError(f'exclude rule references value outside {axis} axis: {value!r}')
        runs=[]; seen=set()
        for tup in product(*values):
            combo=dict(zip(AXES,tup))
            if _excluded(combo,excludes): continue
            run={"schema_version":"1.0",**combo}
            overrides=copy.deepcopy(spec.get('overrides') or {})
            model_patch=(spec.get('per_model_overrides') or {}).get(combo['model']) or {}
            if model_patch: overrides=_merge_dict(overrides,model_patch)
            if overrides: run['overrides']=overrides
            tags=list(spec.get('tags') or [])
            if tags: run['tags']=tags
            validate_run_overrides(run)
            key=stable_id(run,length=64)
            if key in seen: continue
            seen.add(key); runs.append(run)
            if len(runs)>max_runs:
                raise ConfigError(f'matrix expands to more than max_runs={max_runs}; narrow axes/exclusions or raise the explicit limit')
        if not runs: raise ConfigError('matrix expands to zero runs')
        return runs

    def build(self, spec: dict, *, specs=None) -> dict:
        runs=self.expand(spec); cache={}; planner=self._planner(specs); plans=[planner.build(run,cache=cache) for run in runs]
        incompatible=sum(1 for p in plans if p['compatibility']['status']=='incompatible')
        obj={"schema_version":"1.0","matrix_id":"matrix-pending","matrix_spec":copy.deepcopy(spec),"plans":plans,"summary":{"runs":len(plans),"incompatible":incompatible,"planning_cache_entries":len(cache)}}
        finalize_matrix_plan(obj); self.app.matrix_schemas.validate('matrix_plan',obj); return obj


class MatrixExecutor:
    @staticmethod
    def _failure_requires_hard_stop(exc: BaseException, cleanup_status: str | None) -> bool:
        # Never continue a batch when Core-owned process cleanup is incomplete,
        # even if a more useful primary error (OOM/backend failure/etc.) is kept.
        return (
            cleanup_status == 'incomplete'
            or isinstance(exc,(CleanupCriticalError,OrchestrationInterruptedError,StaleProcessError))
        )

    def __init__(self, app, *, results_root: str|Path|None=None, cache_root: str|Path|None=None, secrets_map: dict[str,str]|None=None):
        self.app=app; project_root=Path(getattr(app,'project_root',app.root)); self.results_root=Path(results_root or project_root/'results').resolve(); self.cache_root=Path(cache_root or project_root/'cache').resolve(); self.secrets_map=secrets_map
        self.result_relocation=load_result_relocation(self.results_root)
        self.resources=ResourceManager(app.host_runtime_root/'resources')

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
        instant=when or datetime.now(timezone.utc)
        if instant.tzinfo is None: instant=instant.replace(tzinfo=timezone.utc)
        return instant.astimezone(cls._batch_timezone(matrix_plan)).strftime('%Y%m%d-%H%M%S')

    @classmethod
    def _batch_now(cls, matrix_plan: dict) -> str:
        return datetime.now(cls._batch_timezone(matrix_plan)).isoformat(timespec='seconds')

    def _allocate_batch_dir(self, matrix_plan: dict, *, when: datetime|None=None) -> Path:
        root=self.results_root/'_batches'; root.mkdir(parents=True,exist_ok=True)
        base=self._batch_id(matrix_plan,when=when)
        for index in range(1,10_000):
            name=base if index == 1 else f'{base}-{index}'
            candidate=root/name
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise ConfigError(f'could not allocate a unique batch directory for {base!r}')

    @staticmethod
    def _safe_confined_path(value: str|Path, root: str|Path, *, label: str, require_file: bool=False, require_dir: bool=False) -> Path:
        base=Path(root).resolve(); raw=Path(value); lexical=raw.absolute()
        for candidate in (lexical,*lexical.parents):
            if candidate.is_symlink(): raise ValueError(f'{label} may not traverse a symlink: {candidate}')
            if candidate == base: break
        path=raw.resolve()
        if path != base and base not in path.parents: raise ValueError(f'{label} escapes root: root={base} path={path}')
        if require_file and not path.is_file(): raise ValueError(f'{label} is not a regular file: {path}')
        if require_dir and not path.is_dir(): raise ValueError(f'{label} is not a directory: {path}')
        return path

    def _confined_batch_dir(self, value: str|Path) -> Path:
        base=(self.results_root/'_batches').resolve()
        path=self._safe_confined_path(value,base,label='resume directory',require_dir=True)
        if path == base: raise ConfigError(f'resume directory must be a child of {base}: {path}')
        return path

    def _stored_path(self, value: str|Path, root: str|Path, *, label: str, require_file: bool=False, require_dir: bool=False) -> Path:
        relocation=getattr(self,'result_relocation',ResultRelocationMap(Path(self.results_root).resolve()))
        relocated=relocation.relocate(str(value),label=label)
        return self._safe_confined_path(relocated,root,label=label,require_file=require_file,require_dir=require_dir)

    @staticmethod
    def _load_status(path: Path, matrix_id: str, planned_ids: set[str]) -> dict[str,dict]:
        if not path.is_file(): return {}
        try: prev=load_json_strict(path.read_text(encoding='utf-8'))
        except Exception as exc: raise ConfigError(f'failed to parse batch status {path}: {exc}') from exc
        if prev.get('matrix_id') != matrix_id: raise ConfigError('batch status belongs to a different matrix plan')
        status={}
        for row in prev.get('runs',[]):
            if not isinstance(row,dict) or not row.get('plan_id'): raise ConfigError('batch status contains invalid run record')
            pid=str(row['plan_id'])
            if pid not in planned_ids: raise ConfigError(f'batch status references unknown plan_id: {pid}')
            if pid in status: raise ConfigError(f'batch status contains duplicate plan_id: {pid}')
            status[pid]=row
        return status

    def _validate_success_record(self, rec: dict, plan: dict) -> tuple[bool,str|None,dict|None]:
        try:
            raw=rec.get('run_dir')
            if not raw: raise ValueError('missing run_dir')
            run_dir=self._stored_path(raw,self.results_root,label='run_dir',require_dir=True)
            if run_dir == self.results_root: raise ValueError(f'run_dir must be a child of results root: {run_dir}')
            # A batch consumes the compact run product.  SHA manifests and raw
            # framework files are deliberately not prerequisites for comparing
            # already-completed runs.
            recorded=rec.get('result_path') or rec.get('canonical_result_path')
            recorded_name=Path(str(recorded)).name if recorded else None
            use_product=(
                recorded_name == 'result.json'
                or (recorded_name != 'canonical_result.json' and ((run_dir/'result.json').is_file() or (run_dir/'terminal.json').is_file()))
            )
            if use_product:
                result_path=run_dir/'result.json'; terminal_path=run_dir/'terminal.json'; config_path=run_dir/'config'/'run_config.json'
                kind='run product'
            else:
                result_path=run_dir/'canonical_result.json'; terminal_path=run_dir/'terminal_record.json'; config_path=run_dir/'config'/'execution_plan.json'
                kind='legacy run'
            for label,path in (('result',result_path),('terminal',terminal_path),('config',config_path)):
                if not path.is_file(): raise ValueError(f'{kind} is missing {label} file: {path.relative_to(run_dir)}')
            terminal=load_json_strict(terminal_path.read_text(encoding='utf-8'))
            outcome=str(terminal.get('outcome') or terminal.get('status') or '').lower()
            if outcome not in {'success','succeeded'}: raise ValueError(f'terminal outcome is not success: {outcome!r}')
            saved_config=load_json_strict(config_path.read_text(encoding='utf-8'))
            if saved_config.get('plan_id') != plan.get('plan_id'): raise ValueError(f'{kind} belongs to a different child plan')
            result=load_json_strict(result_path.read_text(encoding='utf-8'))
            self.app.schemas.validate('canonical_result',result)
            expected_framework=plan['resolved']['specs']['evaluation']['framework']['adapter']
            model_spec=((plan.get('resolved') or {}).get('specs') or {}).get('model') or {}
            model_meta=model_spec.get('metadata') or {}
            expected_model=model_spec.get('experiment_id') or model_meta.get('experiment_id') or model_spec.get('id') or plan['run_spec']['model']
            if result.get('run_id') != run_dir.name or result.get('model') != expected_model or result.get('benchmark') != plan['run_spec']['benchmark'] or result.get('framework') != expected_framework:
                raise ValueError('result identity disagrees with child plan/run directory')
            if recorded:
                stored=self._stored_path(recorded,run_dir,label='result_path',require_file=True)
                if stored != result_path: raise ValueError('recorded result path disagrees with relocated run directory')
            return True,None,result
        except Exception as exc:
            return False,str(exc),None

    def _write_status(self, path: Path, matrix_id: str, status: dict[str,dict]) -> None:
        atomic_json(path,{"matrix_id":matrix_id,"runs":sorted(status.values(),key=lambda x:int(x.get('index',0)))})

    @staticmethod
    def _tsv(values) -> str:
        return '\t'.join(str('' if value is None else value).replace('\t',' ').replace('\r',' ').replace('\n',' ') for value in values)

    @staticmethod
    def _public_run(row: dict) -> dict:
        keys=(
            'index','plan_id','model_id','model_label','model_ref','benchmark','platform','deployment','evaluation',
            'status','attempts','started_at','finished_at','run_dir','result_path','warnings_count','cleanup_status','error',
        )
        return {key:copy.deepcopy(row[key]) for key in keys if key in row}

    @staticmethod
    def _breakdown(result: dict, name: str) -> dict:
        direct=result.get(name)
        if isinstance(direct,dict): return direct
        nested=(result.get('breakdowns') or {}).get(name)
        return nested if isinstance(nested,dict) else {}

    def _append_breakdown_metrics(self, lines: list[str], *, kind: str, row: dict, result: dict) -> None:
        for item_id,detail in sorted(self._breakdown(result,kind).items()):
            if not isinstance(detail,dict): continue
            sample=detail.get('sample_count') or {}
            subtasks=detail.get('subtasks') or []
            config=detail.get('config')
            config_json=json.dumps(config,ensure_ascii=False,sort_keys=True,separators=(',',':')) if isinstance(config,dict) else ''
            for namespace,key in (('framework_native','metrics'),('canonical','canonical_metrics')):
                table=detail.get(key) or {}
                if not isinstance(table,dict): continue
                for metric,entry in sorted(table.items()):
                    metric_entry=entry if isinstance(entry,dict) else {'value':entry}
                    lines.append(self._tsv((
                        row.get('model_id',''),row.get('model_label',''),row.get('model_ref',''),
                        result.get('benchmark',''),result.get('framework',''),item_id,detail.get('label',''),namespace,metric,
                        metric_entry.get('value',''),metric_entry.get('stderr',''),metric_entry.get('higher_is_better',''),
                        sample.get('original',''),sample.get('effective',''),detail.get('num_fewshot',''),detail.get('version',''),
                        ','.join(str(x) for x in subtasks),config_json,row.get('run_dir',''),
                    )))

    def _finalize_batch(self, batch_dir: Path, matrix_plan: dict, status: dict[str,dict], *, hard_stop: bool, keep_going: bool, interrupted: bool=False) -> dict:
        rows=sorted(status.values(),key=lambda x:int(x.get('index',0)))
        metric_lines=['model_id\tmodel_label\tmodel_ref\tbenchmark\tframework\tmetric\tvalue\tstderr\thigher_is_better\trun_dir']
        detail_header='model_id\tmodel_label\tmodel_ref\tbenchmark\tframework\t{kind}_id\t{kind}_label\tmetric_namespace\tmetric\tvalue\tstderr\thigher_is_better\tsample_original\tsample_effective\tnum_fewshot\tversion\tsubtasks\tconfig_json\trun_dir'
        group_lines=[detail_header.format(kind='group')]; task_lines=[detail_header.format(kind='task')]
        plan_by_id={p['plan_id']:p for p in matrix_plan['plans']}
        for x in rows:
            if x.get('status')!='success': continue
            ok,err,result=self._validate_success_record(x,plan_by_id[x['plan_id']])
            if not ok or result is None:
                x['status']='failed'; x['error']={'type':'ResultValidationError','message':err or 'success result invalid'}
                continue
            for metric,entry in sorted((result.get('metrics') or {}).items()):
                metric_entry=entry if isinstance(entry,dict) else {'value':entry}
                metric_lines.append(self._tsv((x.get('model_id',''),x.get('model_label',''),x.get('model_ref',''),result.get('benchmark',''),result.get('framework',''),metric,metric_entry.get('value',''),metric_entry.get('stderr',''),metric_entry.get('higher_is_better',''),x.get('run_dir',''))))
            self._append_breakdown_metrics(group_lines,kind='groups',row=x,result=result)
            self._append_breakdown_metrics(task_lines,kind='tasks',row=x,result=result)
        rows=sorted(status.values(),key=lambda x:int(x.get('index',0)))
        counts={"planned":len(matrix_plan['plans']),"success":sum(x.get('status')=='success' for x in rows),"failed":sum(x.get('status')=='failed' for x in rows),"interrupted":sum(x.get('status')=='interrupted' for x in rows),"not_run":sum(x.get('status')=='not_run' for x in rows)}
        outcome='interrupted' if interrupted else ('success' if counts['success']==counts['planned'] else 'failed')
        summary={"batch_id":batch_dir.name,"matrix_id":matrix_plan['matrix_id'],"outcome":outcome,**counts,"warnings":sum(int(x.get('warnings_count',0) or 0) for x in rows),"hard_stop":hard_stop,"continue_on_error":keep_going}
        # batch_status.json and matrix_plan.json are resumable internal state;
        # the following five files are the lightweight user-facing product.
        self._write_status(batch_dir/'batch_status.json',matrix_plan['matrix_id'],status)
        atomic_json(batch_dir/'summary.json',summary); atomic_json(batch_dir/'runs.json',[self._public_run(x) for x in rows])
        atomic_text(batch_dir/'metrics.tsv','\n'.join(metric_lines)+'\n')
        atomic_text(batch_dir/'group_metrics.tsv','\n'.join(group_lines)+'\n')
        atomic_text(batch_dir/'task_metrics.tsv','\n'.join(task_lines)+'\n')
        return summary

    def execute(self, matrix_plan: dict, *, continue_on_error: bool|None=None, resume_dir: str|Path|None=None) -> tuple[Path,dict]:
        verify_matrix_plan(matrix_plan,app=self.app)
        configured=bool((matrix_plan['matrix_spec'].get('execution') or {}).get('continue_on_error',False)); keep_going=configured if continue_on_error is None else bool(continue_on_error)
        planned_ids={p['plan_id'] for p in matrix_plan['plans']}
        batch_claim={"kind":"other","id":f"matrix:{matrix_plan['matrix_id']}","exclusive":True}
        with orchestration_signal_guard(), self.resources.acquire([batch_claim]):
            batch_dir: Path|None=None; status_path: Path|None=None; status: dict[str,dict]={}
            hard_stop=False; interrupted_exc: BaseException|None=None
            try:
                if resume_dir:
                    batch_dir=self._confined_batch_dir(resume_dir)
                    plan_path=batch_dir/'matrix_plan.json'
                    if not plan_path.is_file(): raise ConfigError(f'resume directory is missing matrix_plan.json: {batch_dir}')
                    saved=load_json_strict(plan_path.read_text(encoding='utf-8')); verify_matrix_plan(saved,app=self.app)
                    if stable_id(saved,length=64)!=stable_id(matrix_plan,length=64): raise ConfigError('resume directory contains a different matrix plan')
                else:
                    batch_dir=self._allocate_batch_dir(matrix_plan); atomic_json(batch_dir/'matrix_plan.json',matrix_plan)
                status_path=batch_dir/'batch_status.json'; status=self._load_status(status_path,matrix_plan['matrix_id'],planned_ids)
                for idx,plan in enumerate(matrix_plan['plans'],1):
                    pid=plan['plan_id']; old=status.get(pid); stale_reason=None
                    if old and old.get('status')=='success':
                        ok,stale_reason,_=self._validate_success_record(old,plan)
                        if ok: continue
                    model_spec=((plan.get('resolved') or {}).get('specs') or {}).get('model') or {}
                    model_meta=model_spec.get('metadata') or {}
                    rec={"index":idx,"plan_id":pid,
                         "model":plan['run_spec']['model'],
                         "model_id":str(model_spec.get('experiment_id') or model_meta.get('experiment_id') or plan['run_spec']['model']),
                         "model_label":str(model_spec.get('label') or model_meta.get('label') or model_spec.get('experiment_id') or model_meta.get('experiment_id') or plan['run_spec']['model']),
                         "model_ref":str((model_spec.get('source') or {}).get('ref') or ''),
                         "platform":plan['run_spec']['platform'],"deployment":plan['run_spec']['deployment'],"benchmark":plan['run_spec']['benchmark'],"evaluation":plan['run_spec']['evaluation'],"status":"running","attempts":int((old or {}).get('attempts',0))+1,"started_at":self._batch_now(matrix_plan)}
                    if old:
                        history=list(old.get('history') or [])
                        history.append({k:copy.deepcopy(old.get(k)) for k in ('status','attempts','started_at','finished_at','started_utc','finished_utc','run_dir','error') if old.get(k) is not None})
                        rec['history']=history
                    if stale_reason: rec['resume_validation_warning']=stale_reason
                    status[pid]=rec; self._write_status(status_path,matrix_plan['matrix_id'],status)
                    try:
                        orch=self.app.orchestrator(results_root=self.results_root,cache_root=self.cache_root,secrets=self.secrets_map)
                        path=orch.execute(plan); run_path=Path(path)
                        result_path=run_path/'result.json'
                        terminal_path=run_path/'terminal.json'
                        if not result_path.is_file():
                            # Runs produced before the lightweight product layout
                            # remain resumable and aggregatable.
                            result_path=run_path/'canonical_result.json'; terminal_path=run_path/'terminal_record.json'
                        if not result_path.is_file(): raise ProcessError(f'run completed without result.json: {path}')
                        warnings_count=0
                        if terminal_path.is_file():
                            try:
                                terminal_obj=load_json_strict(terminal_path.read_text(encoding='utf-8'))
                                warning_rows=terminal_obj.get('warnings')
                                warnings_count=len(warning_rows) if isinstance(warning_rows,list) else int(terminal_obj.get('warnings_count',0) or 0)
                            except Exception:
                                warnings_count=0
                        rec.update({"status":"success","run_dir":str(path),"result_path":str(result_path),"warnings_count":warnings_count,"finished_at":self._batch_now(matrix_plan)})
                    except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
                        rec.update({"status":"interrupted","error":{"type":type(exc).__name__,"message":str(exc) or "user interrupt"},"finished_at":self._batch_now(matrix_plan)}); hard_stop=True; interrupted_exc=exc
                        self._write_status(status_path,matrix_plan['matrix_id'],status); break
                    except Exception as exc:
                        text=str(exc); rec.update({"status":"failed","error":{"type":type(exc).__name__,"message":text},"finished_at":self._batch_now(matrix_plan)})
                        failed_dir=(getattr(exc,'details',{}) or {}).get('run_dir')
                        if failed_dir:
                            try:
                                fd=self._safe_confined_path(failed_dir,self.results_root,label='failed run_dir',require_dir=True)
                                if fd != self.results_root:
                                    rec['run_dir']=str(fd)
                                    failure_path=fd/'error.json'
                                    if not failure_path.is_file(): failure_path=fd/'failure.json'
                                    if failure_path.is_file():
                                        failure_obj=load_json_strict(failure_path.read_text(encoding='utf-8'))
                                        primary=failure_obj.get('error') or failure_obj.get('primary_error')
                                        if isinstance(primary,dict): rec['error']=primary
                                        rec['failure_path']=str(failure_path)
                                        cleanup=failure_obj.get('cleanup') or {}
                                        if isinstance(cleanup,dict):
                                            rec['cleanup_status']=cleanup.get('status') or (cleanup.get('backend') or {}).get('status')
                            except Exception: pass
                        cleanup_status = rec.get('cleanup_status')
                        if cleanup_status is None:
                            details = getattr(exc, 'details', None)
                            if isinstance(details, dict):
                                cleanup_status = details.get('cleanup_status')
                            cleanup_status = cleanup_status or getattr(exc, '_model_eval_cleanup_status', None)
                            if cleanup_status is not None:
                                rec['cleanup_status'] = cleanup_status
                        hard_stop=self._failure_requires_hard_stop(exc,cleanup_status)
                        if hard_stop or not keep_going:
                            self._write_status(status_path,matrix_plan['matrix_id'],status); break
                    self._write_status(status_path,matrix_plan['matrix_id'],status)
            except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
                hard_stop=True; interrupted_exc=exc
                for row in status.values():
                    if row.get('status')=='running':
                        row.update({'status':'interrupted','error':{'type':type(exc).__name__,'message':str(exc) or 'matrix interrupted'},'finished_at':self._batch_now(matrix_plan)})
                if status_path is not None: self._write_status(status_path,matrix_plan['matrix_id'],status)

            if batch_dir is None:
                if interrupted_exc is not None: raise interrupted_exc
                raise ProcessError('matrix execution did not establish a batch directory')
            if hard_stop or (any(status.get(x,{}).get('status')=='failed' for x in planned_ids) and not keep_going):
                for idx,p in enumerate(matrix_plan['plans'],1):
                    if p['plan_id'] not in status:
                        model_spec=((p.get('resolved') or {}).get('specs') or {}).get('model') or {}; model_meta=model_spec.get('metadata') or {}
                        status[p['plan_id']]={"index":idx,"plan_id":p['plan_id'],"model":p['run_spec']['model'],"model_id":str(model_spec.get('experiment_id') or model_meta.get('experiment_id') or p['run_spec']['model']),"model_label":str(model_spec.get('label') or model_meta.get('label') or model_spec.get('experiment_id') or model_meta.get('experiment_id') or p['run_spec']['model']),"model_ref":str((model_spec.get('source') or {}).get('ref') or ''),"platform":p['run_spec']['platform'],"deployment":p['run_spec']['deployment'],"benchmark":p['run_spec']['benchmark'],"evaluation":p['run_spec']['evaluation'],"status":"not_run","attempts":0}
            try:
                summary=self._finalize_batch(batch_dir,matrix_plan,status,hard_stop=hard_stop,keep_going=keep_going,interrupted=interrupted_exc is not None)
            except (KeyboardInterrupt, OrchestrationInterruptedError) as exc:
                # If the first termination signal arrives during batch product
                # finalization, record the interruption and retry finalization
                # once so resume state is not silently abandoned.
                hard_stop=True; interrupted_exc=interrupted_exc or exc
                for idx,p in enumerate(matrix_plan['plans'],1):
                    if p['plan_id'] not in status:
                        model_spec=((p.get('resolved') or {}).get('specs') or {}).get('model') or {}; model_meta=model_spec.get('metadata') or {}
                        status[p['plan_id']]={"index":idx,"plan_id":p['plan_id'],"model":p['run_spec']['model'],"model_id":str(model_spec.get('experiment_id') or model_meta.get('experiment_id') or p['run_spec']['model']),"model_label":str(model_spec.get('label') or model_meta.get('label') or model_spec.get('experiment_id') or model_meta.get('experiment_id') or p['run_spec']['model']),"model_ref":str((model_spec.get('source') or {}).get('ref') or ''),"platform":p['run_spec']['platform'],"deployment":p['run_spec']['deployment'],"benchmark":p['run_spec']['benchmark'],"evaluation":p['run_spec']['evaluation'],"status":"not_run","attempts":0}
                summary=self._finalize_batch(batch_dir,matrix_plan,status,hard_stop=True,keep_going=keep_going,interrupted=True)
            if interrupted_exc is not None: raise interrupted_exc
            return batch_dir,summary
