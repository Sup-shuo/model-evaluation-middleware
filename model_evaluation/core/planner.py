from __future__ import annotations

from pathlib import Path
from typing import Any

from model_evaluation.core.compatibility import (
    device_runtime_compatibility,
    evaluate,
    facts_from_device,
    facts_from_environment,
    facts_from_runtime,
    merge_fact_sets,
)
from model_evaluation.core.config.loader import SpecRepository
from model_evaluation.core.config.deployment import resolve_deployment_profile
from model_evaluation.core.config.evaluation import resolve_evaluation_profile
from model_evaluation.core.config.overrides import validate_run_overrides
from model_evaluation.core.config.platform import adapter_parameters
from model_evaluation.core.identifiers import stable_id
from model_evaluation.core.errors import ConfigError
from model_evaluation.core.provenance import assess_model_provenance
from model_evaluation.core.registry.adapter_registry import AdapterRegistry
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.execution_plan import EXECUTION_STAGES, validate_execution_plan

class Planner:
    def __init__(self, *, project_root: str | Path, schemas: SchemaStore, specs: SpecRepository, registry: AdapterRegistry):
        self.project_root=Path(project_root).resolve(); self.schemas=schemas; self.specs=specs; self.registry=registry

    def build(self, run_spec: dict, *, cache: dict | None = None) -> dict:
        validate_run_overrides(run_spec)
        source_bundle=self.specs.resolve_bundle(run_spec)
        bundle=dict(source_bundle)
        overrides=run_spec.get('overrides') or {}
        platform=__import__('copy').deepcopy(source_bundle['platform'])
        platform_patch=overrides.get('platform') or {}
        if platform_patch:
            def merge(base, patch):
                out=__import__('copy').deepcopy(base)
                for key,value in patch.items():
                    if isinstance(value,dict) and isinstance(out.get(key),dict): out[key]=merge(out[key],value)
                    else: out[key]=__import__('copy').deepcopy(value)
                return out
            platform=merge(platform,platform_patch)
            self.schemas.validate('platform_profile',platform)
        bundle['platform']=platform
        effective_deployment, deployment_resolution=resolve_deployment_profile(source_bundle['deployment'], source_bundle['model'], platform, overrides.get('deployment'))
        bundle['deployment']=effective_deployment
        effective_evaluation,evaluation_resolution=resolve_evaluation_profile(source_bundle['evaluation'], platform)
        bundle['evaluation']=effective_evaluation
        platform=bundle['platform']; deployment=bundle['deployment']; evaluation=bundle['evaluation']; benchmark=bundle['benchmark']; model=bundle['model']
        model_source=assess_model_provenance(model,deployment)
        plan_warnings: list[dict[str,Any]] = []

        def invoke_cached(client, operation: str, input_obj: dict, *, context: dict, timeout: float):
            ident=client.identity
            if cache is None:
                output=client.invoke(operation,input_obj,context=context,timeout=timeout)
                warnings=list(client.last_warnings)
            else:
                key=(ident.kind,ident.name,ident.version,operation,stable_id(input_obj),stable_id(context))
                if key not in cache:
                    output=client.invoke(operation,input_obj,context=context,timeout=timeout)
                    cache[key]={'output':output,'warnings':list(client.last_warnings)}
                cached=cache[key]
                if isinstance(cached,dict) and 'output' in cached and 'warnings' in cached:
                    output=cached['output']; warnings=list(cached.get('warnings') or [])
                else:  # compatibility with an in-memory cache created by an older caller
                    output=cached; warnings=[]
            for message in warnings:
                plan_warnings.append({'stage':'planning','adapter':f'{ident.kind}/{ident.name}','operation':operation,'message':str(message)})
            import copy
            return copy.deepcopy(output)
        mode=deployment['management']['mode']; framework_name=evaluation['framework']['adapter']
        binding_name=str((benchmark.get('bindings') or {}).get(framework_name) or evaluation['binding']['adapter'])
        local_platform_fields=('device','runtime','backend_environment')
        if mode == 'managed':
            missing=[name for name in local_platform_fields if name not in platform]
            if missing:
                raise ConfigError(f"managed deployment requires local Platform components: missing={missing}")
            families=((deployment.get('compatibility') or {}).get('runtime_families'))
            if not families:
                raise ConfigError(
                    f"managed deployment {deployment['id']} must declare compatibility.runtime_families; "
                    "Core never infers runtime compatibility"
                )
        elif any(name in platform for name in local_platform_fields):
            raise ConfigError(
                f"{mode} deployment must use an evaluation-only Platform; device/runtime/backend_environment are not execution dependencies"
            )
        clients={
            'evaluation_env': self.registry.get('environment',platform['evaluation_environment']['provider']),
            'backend': self.registry.get('backend',deployment['backend']['adapter']),
            'dataset': self.registry.get('dataset',benchmark['dataset']['provider']),
            'binding': self.registry.get('binding',binding_name),
            'evaluator': self.registry.get('evaluator',framework_name),
        }
        if mode == 'managed':
            clients.update({
                'device': self.registry.get('device',platform['device']['adapter']),
                'runtime': self.registry.get('runtime',platform['runtime']['adapter']),
                'backend_env': self.registry.get('environment',platform['backend_environment']['provider']),
            })
        ctx={'timeout_seconds':2,'offline':True,'planning':True}
        resolved: dict[str,Any]={
            'specs':bundle,'source_deployment_spec':source_bundle['deployment'],'deployment_resolution':deployment_resolution,'source_evaluation_spec':source_bundle['evaluation'],'evaluation_resolution':evaluation_resolution,'management_mode':mode,'binding_adapter':binding_name,
            'deferred_checks':['service_capabilities','evaluator_environment_capabilities'],
            'model_source':model_source,
        }
        facts: dict[str,Any]={}; reasons=[]; optional=[]; diagnostics=[]
        resources=[
            {'kind':'run_lock','id':'global-orchestrator','exclusive':True},
            {'kind':'workspace','id':'run-workspace','exclusive':True},
        ]
        backend_reqs=invoke_cached(clients['backend'],'requirements',{'model':model,'deployment':deployment},context=ctx,timeout=3)
        resolved['backend_requirements']=backend_reqs
        if mode=='managed':
            device_params=adapter_parameters(platform,'device'); runtime_params=adapter_parameters(platform,'runtime')
            device=invoke_cached(clients['device'],'probe',{'requested_devices':platform['device'].get('devices',[]),'parameters':device_params},context=ctx,timeout=3)
            visibility=invoke_cached(clients['device'],'visibility',{'devices':[d['id'] for d in device['devices']], 'descriptor':device,'parameters':device_params},context=ctx,timeout=3)['env_patch']
            runtime=invoke_cached(clients['runtime'],'probe',{'profile':platform['runtime'].get('profile'),'parameters':runtime_params},context=ctx,timeout=3)
            runtime_patch=invoke_cached(clients['runtime'],'resolve_environment',{'descriptor':runtime,'profile':platform['runtime'].get('profile'),'parameters':runtime_params},context=ctx,timeout=3)['env_patch']
            backend_env=invoke_cached(clients['backend_env'],'resolve',{'profile':platform['backend_environment']['profile'],'parameters':adapter_parameters(platform,'backend_environment')},context=ctx,timeout=4)
            resolved['platform']={'device':device,'runtime':runtime,'backend_environment':backend_env,'device_env_patch':visibility,'runtime_env_patch':runtime_patch}
            facts=merge_fact_sets(facts_from_device(device),facts_from_runtime(runtime),facts_from_environment(backend_env,'backend_environment'))
            pair=device_runtime_compatibility(device,runtime)
            reasons.extend(pair.reasons); optional.extend(pair.optional_misses); diagnostics.extend(pair.diagnostics)
            deployment_compatibility={
                'schema_version':'1.0',
                'requirements':[{'path':'runtime.family','op':'in','value':list(deployment['compatibility']['runtime_families']),
                                 'message':'selected runtime is not allowed by DeploymentProfile.compatibility.runtime_families'}],
            }
            self.schemas.validate('requirement_set',deployment_compatibility)
            resolved['deployment_compatibility_requirements']=deployment_compatibility
            compatibility_report=evaluate(deployment_compatibility,facts)
            reasons.extend(compatibility_report.reasons); optional.extend(compatibility_report.optional_misses); diagnostics.extend(compatibility_report.diagnostics)
            for did in [d['id'] for d in device['devices']]: resources.append({'kind':'device','id':f"{device['vendor']}:{did}",'exclusive':True,'metadata':{'vendor':device['vendor'],'device_id':str(did)}})
            params=deployment.get('parameters') or {}; endpoint=deployment.get('endpoint') or {}; port_value=endpoint.get('port') if endpoint.get('port') is not None else params.get('port')
            if port_value is None: raise ConfigError(f"managed deployment {deployment['id']} must declare endpoint.port or parameters.port; Core does not assume backend-specific ports")
            port=int(port_value); host=str(endpoint.get('host') or '127.0.0.1')
            if not (1 <= port <= 65535): raise ConfigError(f'managed deployment port must be in 1..65535: {port}')
            if not host.strip(): raise ConfigError('managed deployment host must be non-empty')
            resources.append({'kind':'port','id':str(port),'exclusive':True,'host':host})
            resolved['endpoint']={'host':host,'port':port}
        else:
            resolved['platform']={'device_probe_skipped':True,'runtime_probe_skipped':True,'reason':'backend is not locally managed'}
        evaluation_env=invoke_cached(clients['evaluation_env'],'resolve',{'profile':platform['evaluation_environment']['profile'],'parameters':adapter_parameters(platform,'evaluation_environment')},context=ctx,timeout=4)
        resolved['platform']['evaluation_environment']=evaluation_env
        facts=merge_fact_sets(facts,facts_from_environment(evaluation_env,'evaluation_environment'))
        report=evaluate(backend_reqs,facts)
        reasons.extend(report.reasons); optional.extend(report.optional_misses); diagnostics.extend(report.diagnostics)
        resolved['dataset_resolution']=invoke_cached(clients['dataset'],'resolve',{'benchmark':benchmark},context=ctx,timeout=3)
        dataset_lock_id='dataset:'+benchmark['dataset']['provider']+':'+stable_id(resolved['dataset_resolution'],length=24)
        resources.append({'kind':'cache_lock','id':dataset_lock_id,'exclusive':True,'metadata':{'provider':benchmark['dataset']['provider'],'dataset_id':resolved['dataset_resolution'].get('dataset_id'),'revision':resolved['dataset_resolution'].get('revision')}})
        bind_req=invoke_cached(clients['binding'],'requirements',{'benchmark':benchmark},context=ctx,timeout=3)
        resolved['binding_requirements']=bind_req
        bind_report=evaluate(bind_req,facts)
        reasons.extend(bind_report.reasons); optional.extend(bind_report.optional_misses); diagnostics.extend(bind_report.diagnostics)
        unique={}
        for client in clients.values():
            ident=client.identity; unique[(ident.kind,ident.name)]={'kind':ident.kind,'name':ident.name,'version':ident.version}
        adapters=[unique[k] for k in sorted(unique)]
        if reasons: status='incompatible'
        else: status='unknown'; reasons=['service/evaluator capability checks are deferred until SERVICE_READY']
        if optional: resolved['planning_optional_misses']=optional
        if diagnostics: resolved['compatibility_diagnostics']=diagnostics
        plan={'schema_version':'1.0','plan_id':'plan-pending','run_spec':run_spec,'adapters':adapters,'compatibility':{'status':status,'reasons':reasons,'diagnostics':diagnostics},'resources':resources,'stages':list(EXECUTION_STAGES),'resolved':resolved,'protocol_fingerprints':{},'warnings':plan_warnings}
        plan['plan_id']='plan-'+stable_id(plan,length=24,exclude_keys={'plan_id'})
        validate_execution_plan(plan,self.schemas)
        return plan
