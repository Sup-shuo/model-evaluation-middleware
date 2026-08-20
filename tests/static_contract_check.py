#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[1]
PACKAGE_ROOT=ROOT/'model_evaluation'
sys.path.insert(0,str(ROOT))
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.security import adapter_subprocess_env
from model_evaluation.core.serialization import json_loads_strict
from model_evaluation.core.config.parsing import load_yaml_strict
from model_evaluation.core.config.loader import _apply_spec_defaults
from model_evaluation.core.registry.adapter_registry import _validate_schema_versions, _validate_adapter_name
from model_evaluation.core.schema.formats import contract_format_checker

EXPECTED_OPS={
 'device': {'probe','visibility','snapshot'},
 'runtime': {'probe','resolve_environment','snapshot'},
 'environment': {'resolve','wrap_process','snapshot'},
 'backend': {'requirements','plan_start','probe_service','snapshot'},
 'dataset': {'resolve','prepare','verify','snapshot'},
 'binding': {'requirements','build_task','protocol_fingerprint'},
 'evaluator': {'requirements','plan_evaluate','normalize','snapshot'},
}
SPEC_SCHEMAS={'models':'model_spec','platforms':'platform_profile','deployments':'deployment_profile','benchmarks':'benchmark_spec','evaluations':'evaluation_profile','runs':'run_spec'}
MANIFEST_GLOBAL_BUDGET_SECONDS=5.0
MANIFEST_PER_ADAPTER_SECONDS=0.5


def _lint_python_adapter_sources() -> None:
    """Additional lint for built-in Python code; not the normative adapter gate."""
    for path in sorted((PACKAGE_ROOT/'adapters').glob('*/*/*.py')):
        tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[]
                if isinstance(node,ast.Import): names=[a.name for a in node.names]
                elif node.module: names=[node.module]
                if any(n=='adapters' or n.startswith('model_evaluation.adapters.') for n in names):
                    raise SystemExit(f'cross-adapter import forbidden: {path}: {names}')
            if path.name == 'impl.py' and isinstance(node,ast.Attribute) and node.attr=='Popen':
                raise SystemExit(f'adapter operation implementation may not own long-lived process: {path}')
            if path.name == 'impl.py' and isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                if any(isinstance(t,ast.Name) and t.id=='MANIFEST' for t in targets):
                    raise SystemExit(f'impl.py duplicates adapter manifest identity; manifest.json is authoritative: {path}')


def _load_protocol_manifest(entry: Path, deadline: float) -> dict:
    remaining=deadline-time.monotonic()
    if remaining <= 0:
        raise SystemExit('adapter manifest validation exceeded global bounded budget')
    timeout=min(MANIFEST_PER_ADAPTER_SECONDS,remaining)
    try:
        proc=subprocess.run(
            [str(entry),'manifest'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            timeout=timeout,check=False,env={**adapter_subprocess_env(), 'PYTHONDONTWRITEBYTECODE':'1'},
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f'adapter manifest timed out: {entry}') from exc
    if proc.returncode != 0:
        raise SystemExit(f'adapter manifest failed: {entry}: {proc.stderr[-1000:]}')
    try: obj=json_loads_strict(proc.stdout)
    except Exception as exc: raise SystemExit(f'adapter manifest is not JSON: {entry}') from exc
    if not isinstance(obj,dict): raise SystemExit(f'adapter manifest must be object: {entry}')
    return obj


def _check_adapters(schemas: SchemaStore) -> int:
    manifests=0; deadline=time.monotonic()+MANIFEST_GLOBAL_BUDGET_SECONDS
    for entry in sorted((PACKAGE_ROOT/'adapters').glob('*/*/adapter')):
        impl_dir=entry.parent
        if not entry.is_file() or not os.access(entry,os.X_OK):
            raise SystemExit(f'adapter entrypoint missing/not executable: {entry}')
        live=_load_protocol_manifest(entry,deadline)
        schemas.validate('adapter_manifest',live)
        disk_path=impl_dir/'manifest.json'
        if disk_path.is_file():
            disk=json_loads_strict(disk_path.read_text(encoding='utf-8'))
            schemas.validate('adapter_manifest',disk)
            if live != disk: raise SystemExit(f'protocol manifest differs from manifest.json: {entry}')
        expected_kind=impl_dir.parent.name; expected_name=impl_dir.name
        _validate_adapter_name(expected_name); _validate_schema_versions(live)
        if live.get('kind') != expected_kind or live.get('name') != expected_name:
            raise SystemExit(f'adapter path/manifest identity mismatch: {entry}')
        implementation=live.get('implementation') or {}
        if 'user_config' in implementation or 'user_parameters_schema' in implementation:
            raise SystemExit(f'adapter uses legacy hidden implementation user-config protocol: {entry}')
        user_cfg=live.get('user_config') or {}
        if not isinstance(user_cfg,dict): raise SystemExit(f'adapter user_config must be object: {entry}')
        if user_cfg and user_cfg.get('schema_version') != '1.0': raise SystemExit(f'adapter user_config schema_version must be 1.0: {entry}')
        user_schema=user_cfg.get('parameters_schema')
        user_validator=None
        if user_schema:
            schema_path=(impl_dir/str(user_schema)).resolve()
            try:
                schema_path.relative_to(impl_dir.resolve())
            except ValueError:
                raise SystemExit(f'adapter user parameter schema escapes adapter directory: {entry}')
            if not schema_path.is_file():
                raise SystemExit(f'adapter user parameter schema missing: {schema_path}')
            try:
                user_obj=json_loads_strict(schema_path.read_text(encoding='utf-8'))
                Draft202012Validator.check_schema(user_obj)
                user_validator=Draft202012Validator(user_obj,format_checker=contract_format_checker())
            except Exception as exc:
                raise SystemExit(f'adapter user parameter schema invalid: {schema_path}: {exc}')
        if 'default_management_mode' in user_cfg and user_cfg['default_management_mode'] not in {'managed','attached','external'}:
            raise SystemExit(f'adapter has invalid default_management_mode: {entry}')
        defaults=user_cfg.get('default_parameters') or {}
        if not isinstance(defaults,dict): raise SystemExit(f'adapter user_config.default_parameters must be object: {entry}')
        if defaults:
            if user_validator is None: raise SystemExit(f'adapter defaults require user_config.parameters_schema: {entry}')
            errors=sorted(user_validator.iter_errors(defaults),key=lambda e:list(e.absolute_path))
            if errors: raise SystemExit(f'adapter default_parameters violate user schema: {entry}: {errors[0].message}')
        derived=user_cfg.get('derived_parameters') or {}
        if not isinstance(derived,dict): raise SystemExit(f'adapter user_config.derived_parameters must be object: {entry}')
        for key,rule in derived.items():
            if not isinstance(rule,dict) or rule.get('source') not in {'selected_device_count'}:
                raise SystemExit(f'adapter derived parameter has unsupported rule: {entry}: {key}')
        location=user_cfg.get('model_location') or {}
        if location:
            if not isinstance(location,dict): raise SystemExit(f'adapter user_config.model_location must be object: {entry}')
            if not isinstance(location.get('modes') or [],list): raise SystemExit(f'adapter model_location.modes must be array: {entry}')
            if location.get('path_template') is not None and not isinstance(location.get('path_template'),str): raise SystemExit(f'adapter model_location.path_template must be string: {entry}')
        if 'profile_required' in user_cfg and not isinstance(user_cfg['profile_required'],bool): raise SystemExit(f'adapter profile_required must be boolean: {entry}')
        if 'root_required' in user_cfg and not isinstance(user_cfg['root_required'],bool): raise SystemExit(f'adapter root_required must be boolean: {entry}')
        if 'root_parameter' in user_cfg and (not isinstance(user_cfg['root_parameter'],str) or not user_cfg['root_parameter']): raise SystemExit(f'adapter root_parameter must be non-empty string: {entry}')
        if live.get('adapter_api') != '1.0':
            raise SystemExit(f'current release requires exact adapter API 1.0: {entry}')
        required=EXPECTED_OPS.get(live['kind'])
        if required is None or not required.issubset(set(live.get('operations') or [])):
            raise SystemExit(f'missing required operations: {entry}')
        if live['kind']=='backend' and 'plan_preflight' in (live.get('operations') or []):
            declared=live.get('schema_versions') or {}
            if declared.get('backend_preflight_plan')!='1.0' or declared.get('preflight_probe_result')!='1.0':
                raise SystemExit(
                    f'backend plan_preflight must declare backend_preflight_plan and preflight_probe_result 1.0: {entry}'
                )
        input_defs=(schemas.load('adapter_operation_inputs').get('$defs') or {})
        for operation in live.get('operations') or []:
            if f"{live['kind']}_{operation}" not in input_defs:
                raise SystemExit(f'adapter operation lacks formal input schema: {entry}: {operation}')
        manifests += 1
    return manifests


def main():
    schemas=SchemaStore(PACKAGE_ROOT/'schemas'); checked=schemas.validate_all_schemas(); specs=0; loaded_specs={}
    ext_checked=[]
    for path in sorted((PACKAGE_ROOT/'schemas'/'user').glob('*.schema.json')):
        schema=json_loads_strict(path.read_text(encoding='utf-8')); Draft202012Validator.check_schema(schema)
        # Keep extension-schema format policy aligned with SchemaStore.
        Draft202012Validator(schema,format_checker=contract_format_checker())
        ext_checked.append(path.name)
    _lint_python_adapter_sources()
    manifests=_check_adapters(schemas)
    for dirname,schema in SPEC_SCHEMAS.items():
        for path in sorted((PACKAGE_ROOT/'presets'/dirname).glob('*.yaml')):
            obj=load_yaml_strict(path.read_text(encoding='utf-8')); obj=_apply_spec_defaults('platform' if dirname=='platforms' else '',obj); schemas.validate(schema,obj); loaded_specs.setdefault(dirname,{})[obj.get('id') or path.stem]=obj; specs += 1
    def require_adapter(kind,name,where):
        entry=PACKAGE_ROOT/'adapters'/kind/name/'adapter'
        if not entry.is_file() or not os.access(entry,os.X_OK): raise SystemExit(f'shipped spec references missing adapter {kind}/{name}: {where}')
    for sid,obj in loaded_specs.get('platforms',{}).items():
        require_adapter('environment',obj['evaluation_environment']['provider'],f'platform {sid} evaluation_environment')
        if 'device' in obj:
            require_adapter('device',obj['device']['adapter'],f'platform {sid}')
            require_adapter('runtime',obj['runtime']['adapter'],f'platform {sid}')
            require_adapter('environment',obj['backend_environment']['provider'],f'platform {sid} backend_environment')
    for sid,obj in loaded_specs.get('deployments',{}).items():
        require_adapter('backend',obj['backend']['adapter'],f'deployment {sid}')
        if (obj.get('management') or {}).get('mode') == 'managed':
            families=((obj.get('compatibility') or {}).get('runtime_families'))
            if not families: raise SystemExit(f'managed deployment lacks explicit compatibility.runtime_families: {sid}')
        if 'runtime_families' in (obj.get('parameters') or {}): raise SystemExit(f'deployment leaks runtime compatibility into parameters: {sid}')
    for sid,obj in loaded_specs.get('benchmarks',{}).items(): require_adapter('dataset',obj['dataset']['provider'],f'benchmark {sid}')
    for sid,obj in loaded_specs.get('evaluations',{}).items():
        require_adapter('evaluator',obj['framework']['adapter'],f'evaluation {sid}')
        require_adapter('binding',obj['binding']['adapter'],f'evaluation {sid} default binding')
    for bid,b in loaded_specs.get('benchmarks',{}).items():
        for eid,e in loaded_specs.get('evaluations',{}).items():
            framework=e['framework']['adapter']
            binding=(b.get('bindings') or {}).get(framework) or e['binding']['adapter']
            require_adapter('binding',binding,f'benchmark/evaluation pair {bid} / {eid}')
    from model_evaluation.core.matrix import MatrixSchemas
    ms=MatrixSchemas(PACKAGE_ROOT/'schemas'/'user')
    user_system=load_yaml_strict((ROOT/'config'/'system.yaml').read_text(encoding='utf-8')); ms.validate('user_system',user_system)
    user_evaluation=load_yaml_strict((ROOT/'config'/'evaluation.yaml').read_text(encoding='utf-8')); ms.validate('user_evaluation',user_evaluation)
    for path in sorted((ROOT/'config'/'systems').glob('*.yaml')):
        ms.validate('user_system',load_yaml_strict(path.read_text(encoding='utf-8')))
    for path in sorted((ROOT/'config'/'evaluations').glob('*.yaml')):
        ms.validate('user_evaluation',load_yaml_strict(path.read_text(encoding='utf-8')))
    model_ids=set()
    for path in sorted((ROOT/'config'/'models').rglob('*.yaml')):
        model=load_yaml_strict(path.read_text(encoding='utf-8')); ms.validate('user_model',model)
        if model['id'] in model_ids: raise SystemExit(f'duplicate shipped user model id: {model["id"]}: {path}')
        model_ids.add(model['id'])
    for path in sorted((PACKAGE_ROOT/'presets'/'matrices').glob('*.yaml')):
        obj=load_yaml_strict(path.read_text(encoding='utf-8')); ms.validate('matrix_spec',obj); specs += 1
    core_text='\n'.join(p.read_text(encoding='utf-8') for p in (PACKAGE_ROOT/'core').rglob('*.py'))
    architecture_text=core_text
    forbidden_patterns=(
        r'CUDA(?:_|\b)',r'MLU(?:_|\b)',r'ROCM(?:_|\b)',r'NEUWARE(?:_|\b)',r'ASCEND(?:_|\b)',r'CANN(?:_|\b)',
        r'\bvllm\b',r'\bllama_cpp\b',r'\bollama\b',r'\bsglang\b',r'\blm_eval\b',r'\bopencompass\b',r'\blighteval\b',
        r'\bbbh\b',r'\bmmlu\b',r'\bhellaswag\b',
    )
    for pattern in forbidden_patterns:
        if re.search(pattern,architecture_text,re.IGNORECASE): raise SystemExit(f'core/user_config contains implementation-specific literal matching: {pattern}')
    runtime_protocol_text='\n'.join(p.read_text(encoding='utf-8') for base in (PACKAGE_ROOT/'core',PACKAGE_ROOT/'adapters') for p in base.rglob('*.py'))
    if 'metadata.version_argv' in runtime_protocol_text or "get('version_argv')" in runtime_protocol_text or 'get("version_argv")' in runtime_protocol_text:
        raise SystemExit('hidden backend dependency protocol version_argv is forbidden; use BackendPreflightPlan or legacy BackendStartPlan.dependency_probe')
    dataset_text='\n'.join(p.read_text(encoding='utf-8') for p in (PACKAGE_ROOT/'adapters'/'dataset').rglob('*.py'))
    for forbidden in ('lm_eval','opencompass','lighteval'):
        if forbidden.lower() in dataset_text.lower(): raise SystemExit(f'dataset adapter contains evaluator-specific literal: {forbidden}')
    print(f'STATIC CONTRACT CHECK OK: schemas={len(checked)} extension_schemas={len(ext_checked)} adapters={manifests} specs={specs}')

if __name__=='__main__': main()
