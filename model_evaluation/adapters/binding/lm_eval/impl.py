from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError
from model_evaluation.sdk.gitmeta import normalize_object_id, read_git_head
def requirements(i,c): return {"schema_version":"1.0","requirements":[]}

def _metric_contract(benchmark,evaluation):
    required=list(benchmark.get('metrics') or [])
    params=(evaluation.get('parameters') or {}) if isinstance(evaluation,dict) else {}
    all_maps=params.get('metric_maps') or {}
    if not isinstance(all_maps,dict): raise AdapterError('CONFIG_INVALID','lm_eval parameters.metric_maps must be an object')
    mapping=all_maps.get(benchmark['id'])
    if mapping is None:
        return {"metric_namespace":"framework_native","required_metrics":required}
    if not isinstance(mapping,dict) or not all(isinstance(k,str) and k and isinstance(v,str) and v for k,v in mapping.items()):
        raise AdapterError('CONFIG_INVALID',f'lm_eval metric map for {benchmark["id"]} must be a non-empty-string mapping')
    values=list(mapping.values())
    if len(values)!=len(set(values)): raise AdapterError('CONFIG_INVALID',f'lm_eval metric map for {benchmark["id"]} maps multiple framework metrics to one canonical metric')
    missing=[name for name in required if name not in values]
    if missing: raise AdapterError('CONFIG_INVALID',f'lm_eval metric map for {benchmark["id"]} does not provide required canonical metrics: {missing}')
    return {"metric_namespace":"canonical","metric_map":dict(mapping),"required_metrics":required}
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _framework_revision(root,e):
    params=((e or {}).get('parameters') or {})
    policy=str(params.get('provenance_policy') or 'migration').lower()
    if policy not in {'migration','strict'}: raise AdapterError('CONFIG_INVALID',f'unsupported provenance_policy: {policy}')
    declared_raw=params.get('framework_revision')
    if policy=='strict' and not declared_raw:
        raise AdapterError('CONFIG_INVALID','strict lm-eval provenance requires parameters.framework_revision so the framework identity is frozen in the ExecutionPlan')
    declared=normalize_object_id(declared_raw) if declared_raw else None
    if declared_raw and declared is None:
        raise AdapterError('CONFIG_INVALID','parameters.framework_revision must be a 40-64 character hexadecimal Git object id')
    actual=None
    try:
        p=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=10,check=False)
        if p.returncode==0: actual=normalize_object_id(p.stdout)
    except Exception: pass
    actual=actual or read_git_head(root)
    if declared:
        if actual is None:
            raise AdapterError('COMPATIBILITY_ERROR',f'lm-eval framework revision could not be established under {root}')
        if actual != declared:
            raise AdapterError('COMPATIBILITY_ERROR',f'lm-eval framework revision mismatch: expected {declared}, got {actual}')
        return str(declared)
    return actual or 'UNPINNED'
def _protocol_sources(task,e):
    params=((e or {}).get('parameters') or {}); root_text=params.get('tool_root') or params.get('framework_root')
    if not root_text: return None,[], 'UNPINNED'
    root=Path(str(root_text)).resolve(); source=root/'lm_eval'/'tasks'/task
    if not source.is_dir(): return root,[],_framework_revision(root,e)
    files=[p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in {'.yaml','.yml','.py','.json'} and '__pycache__' not in p.parts]
    return root,sorted(files),_framework_revision(root,e)
def _basis(b,d,e):
    protocol=b.get('protocol') or {}; task=str(protocol.get('task') or b['id']); root,files,revision=_protocol_sources(task,e)
    sources={p.relative_to(root).as_posix():_sha(p) for p in files} if root else {}
    basis={"benchmark":b,"dataset_fingerprint":d.get('fingerprint'),"binding":"lm_eval/v1-native","framework_revision":revision,"framework_task_sources":sources}
    return task,root,files,revision,basis
def _fingerprint(b,d,e):
    _,_,_,_,basis=_basis(b,d,e); return hashlib.sha256(json.dumps(basis,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def build_task(i,c):
    b=i['benchmark']; d=i['dataset_artifact']; e=i.get('evaluation') or {}
    if (d.get('materialization') or {}).get('kind') != 'virtual': raise AdapterError('COMPATIBILITY_ERROR',f"generic lm_eval binding does not consume concrete dataset provider output for {b['id']}; install specialized binding lm_eval.{b['id']}")
    protocol=b.get('protocol') or {}; task,root,files,revision,basis=_basis(b,d,e); inference=protocol.get('inference') or ['generation']
    if not isinstance(inference,list): raise AdapterError('CONFIG_INVALID','benchmark.protocol.inference must be a list')
    fp=hashlib.sha256(json.dumps(basis,sort_keys=True,separators=(',',':')).encode()).hexdigest(); source_verified=bool(files and revision!='UNPINNED')
    policy=str(((e.get('parameters') or {}).get('provenance_policy') or 'migration')).lower()
    if policy not in {'migration','strict'}: raise AdapterError('CONFIG_INVALID',f'unsupported provenance_policy: {policy}')
    if policy=='strict': raise AdapterError('COMPATIBILITY_ERROR',f'strict lm-eval provenance for {b["id"]} requires a specialized binding that materializes Core-confined task artifacts')
    artifacts=[{"path":str(p),"sha256":_sha(p)} for p in files]
    metric_contract=_metric_contract(b,e)
    metrics={"namespace":metric_contract.get("metric_namespace","framework_native"),"required":list(metric_contract.get("required_metrics") or [])}
    if metric_contract.get("metric_map") is not None: metrics["mapping"]=dict(metric_contract["metric_map"])
    metadata={"dataset_fingerprint":d.get('fingerprint'),"provenance_mode":"framework_native_verified_migration" if source_verified else "migration_framework_native"}
    if not files: metadata['warning']='framework-native task source directory could not be fingerprinted; use a specialized binding or configured tool_root'
    elif revision=='UNPINNED': metadata['warning']='framework task bytes are fingerprinted but framework revision is unpinned'
    return {"schema_version":"1.0","framework":"lm_eval","benchmark_id":b['id'],"task_id":task,"artifacts":artifacts,"protocol_fingerprint":fp,
            "execution":{"inference":inference,"num_fewshot":protocol.get('fewshot')},
            "metrics":metrics,
            "provenance":{"strict":False,"framework_revision":revision,"source_fingerprinted":source_verified,"policy":policy},
            "metadata":metadata}
def fingerprint(i,c): return {"protocol_fingerprint":_fingerprint(i['benchmark'],i['dataset_artifact'],i.get('evaluation') or {})}
OPERATIONS={"requirements":requirements,"build_task":build_task,"protocol_fingerprint":fingerprint}
