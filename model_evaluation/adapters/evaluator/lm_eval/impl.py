from __future__ import annotations
import json, subprocess
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError
from model_evaluation.sdk.jsonutil import loads as json_loads
from model_evaluation.sdk.manifest import load_manifest
from model_evaluation.sdk.gitmeta import normalize_object_id, read_git_head


def _framework_source(params):
    root_value=params.get('tool_root') or params.get('framework_root')
    if not root_value: raise AdapterError('CONFIG_INVALID','lm_eval evaluation profile requires resolved parameters.tool_root')
    policy=str(params.get('provenance_policy') or 'migration').lower()
    if policy not in {'migration','strict'}: raise AdapterError('CONFIG_INVALID',f'unsupported provenance_policy: {policy}')
    declared_raw=params.get('framework_revision')
    if policy=='strict' and not declared_raw:
        raise AdapterError('CONFIG_INVALID','strict lm-eval provenance requires parameters.framework_revision so the framework identity is frozen in the ExecutionPlan')
    declared=normalize_object_id(declared_raw) if declared_raw else None
    if declared_raw and declared is None:
        raise AdapterError('CONFIG_INVALID','parameters.framework_revision must be a 40-64 character hexadecimal Git object id')
    root=Path(str(root_value)).resolve()
    if not root.is_dir() or not (root/'lm_eval').is_dir(): raise AdapterError('DEPENDENCY_MISSING',f'lm-evaluation-harness source root unavailable: {root}')
    revision=None; dirty=None
    try:
        p=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=10,check=False)
        candidate=normalize_object_id(p.stdout) if p.returncode==0 else None
        if candidate:
            revision=candidate
            q=subprocess.run(['git','-C',str(root),'-c','core.fileMode=false','status','--porcelain','--untracked-files=all'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=10,check=False)
            dirty=bool(q.stdout.strip()) if q.returncode==0 else None
    except Exception: pass
    if revision is None:
        revision = read_git_head(root)
    displayed_revision=revision or 'unversioned-source'
    if declared and revision != declared: raise AdapterError('COMPATIBILITY_ERROR',f'lm-eval framework revision mismatch: expected {declared}, got {displayed_revision}')
    if params.get('require_clean_framework',True) and dirty is not False:
        detail='dirty' if dirty is True else 'cleanliness could not be established'
        raise AdapterError('COMPATIBILITY_ERROR',f'lm-eval framework tree {detail}: {root}')
    return root,displayed_revision,dirty

def requirements(i,c):
    # Cheap local preflight: validate the configured harness source/revision before
    # a potentially expensive model server is started.
    evaluation=i.get('evaluation') or {}
    if evaluation:
        _framework_source(evaluation.get('parameters') or {})
    task=i['task']; inf=(task.get('execution') or {}).get('inference') or ['generation']
    req=[
        {"path":"service.protocol.openai_completion","op":"equals","value":True,"message":"lm_eval local-completions requires an OpenAI-compatible completions endpoint"},
        {"path":"evaluation_environment.python","op":"equals","value":True,"message":"lm_eval requires a Python-capable evaluation environment"},
    ]
    if 'generation' in inf: req.append({"path":"service.generation","op":"equals","value":True})
    if any(x in inf for x in ('loglikelihood','multiple_choice','loglikelihood_rolling')):
        req += [
            {"path":"service.completion_logprobs","op":"equals","value":True},
            {"path":"service.echo","op":"equals","value":True},
            {"path":"service.tokenizer_available","op":"equals","value":True},
        ]
    return {"schema_version":"1.0","requirements":req}

def _safe(v,label):
    s=str(v)
    if ',' in s or s!=s.strip(): raise AdapterError('CONFIG_INVALID',f'{label} cannot contain comma or edge whitespace for lm_eval model_args')
    return s

def _execution_env(i,c,params,framework_root):
    env={"set":{"TOKENIZERS_PARALLELISM":str(params.get('tokenizers_parallelism','false')).lower(),"PYTHONHASHSEED":str(int(params.get('pythonhashseed',1234)))},"prepend_path":{"PYTHONPATH":[str(framework_root)]}}
    cache_value=i.get('cache_root') or c.get('cache_root')
    if cache_value:
        cache_root=Path(str(cache_value))
        if not cache_root.is_absolute(): raise AdapterError('CONFIG_INVALID','lm_eval cache_root must be absolute')
        hf_home=cache_root/'huggingface'
        env['set'].update({"HF_HOME":str(hf_home),"HF_DATASETS_CACHE":str(hf_home/'datasets'),"HF_HUB_CACHE":str(hf_home/'hub')})
    if c.get('offline',False) or i.get('network_policy')=='offline':
        env['set'].update({"HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
    return env, str(cache_root.resolve()) if cache_value else None

def plan_preflight(i,c):
    evaluation=i['evaluation']; params=evaluation.get('parameters') or {}; root,revision,dirty=_framework_source(params)
    # Loading the selected task constructs its dataset before the model service
    # starts. In offline mode this is also a deterministic cache-readiness check.
    env,cache_root=_execution_env(i,c,params,root); task=i.get('task') or {}
    payload={"framework_root":str(root),"cache_root":cache_root}
    if task:
        payload['task_id']=task['task_id']
        if task.get('task_root'): payload['task_root']=task['task_root']
    process={"schema_version":"1.0","argv":["python",str(Path(__file__).with_name('preflight.py').resolve()),'--payload',json.dumps(payload,separators=(',',':'),sort_keys=True)],
             "cwd":str(root),"env_patch":env,
             "stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"},
             "timeout_seconds":float(params.get('preflight_timeout_seconds',180.0)),"metadata":{"role":"evaluator_preflight","scope":"task_and_data" if task else "dependency"}}
    return {"process":process,"result_format":"preflight_result"}

def plan_evaluate(i,c):
    svc=i['service']; task=i['task']; prof=i['evaluation']; params=prof.get('parameters') or {}; proto=svc.get('protocols') or {}; completion=(proto.get('openai_completion') or {}).get('url')
    if not completion: raise AdapterError('COMPATIBILITY_ERROR','service does not expose openai_completion')
    model=_safe((svc.get('model') or {})['id'],'model'); caps=(svc.get('capabilities') or {}).get('values',{}); tok=svc.get('tokenizer') or {}; limits=svc.get('limits') or {}
    tok_path=tok.get('path') if tok.get('mode')=='local' else None; remote_tok=bool(caps.get('service.tokenize') and caps.get('service.detokenize') and all((proto.get(k) or {}).get('url') for k in ('tokenizer_info','tokenize','detokenize'))); tok_backend='huggingface' if tok_path else ('remote' if remote_tok else 'none')
    auth=svc.get('auth') or {"mode":"none"}; proxy_mode=None
    if auth.get('mode')=='none': proxy_mode='strip'
    elif auth.get('mode')=='bearer': proxy_mode='inject'
    base_for_args='__MODEL_EVAL_PROXY_COMPLETIONS__' if proxy_mode else completion
    base=_safe(base_for_args,'base_url')
    seeds={
        'random':int(params.get('random_seed',0)),
        'numpy':int(params.get('numpy_random_seed',1234)),
        'torch':int(params.get('torch_random_seed',1234)),
        'fewshot':int(params.get('fewshot_random_seed',1234)),
        'request':int(params.get('request_seed',1234)),
        'pythonhash':int(params.get('pythonhashseed',1234)),
    }
    items=[f'model={model}',f'base_url={base}',f'tokenizer_backend={tok_backend}',f'tokenized_requests={str(bool(params.get("tokenized_requests",False))).lower()}',f'num_concurrent={int(params.get("num_concurrent",limits.get("recommended_concurrency",16)))}',f'max_retries={int(params.get("max_retries",3))}',f'timeout={int(params.get("request_timeout",600))}',f'seed={seeds["request"]}']
    max_len=params.get('max_length',svc.get('context_length'))
    if max_len is not None: items.append(f'max_length={int(max_len)}')
    if tok_path: items.append('tokenizer='+_safe(tok_path,'tokenizer'))
    out=Path(i['output_root']).resolve(); framework_root,framework_revision,framework_dirty=_framework_source(params)
    seed_arg=','.join(str(seeds[name]) for name in ('random','numpy','torch','fewshot'))
    harness=['python','-m','lm_eval','run','--model','local-completions','--model_args',','.join(items),'--tasks',task['task_id'],'--batch_size',str(params.get('batch_size',1)),'--seed',seed_arg,'--show_config','--output_path',str(out)]
    if task.get('task_root'): harness += ['--include_path',str(task['task_root'])]
    nf=(task.get('execution') or {}).get('num_fewshot')
    if nf is not None: harness += ['--num_fewshot',str(int(nf))]
    if params.get('limit') is not None: harness += ['--limit',str(params['limit'])]
    if params.get('log_samples',False): harness.append('--log_samples')
    env,_=_execution_env(i,c,params,framework_root)
    secret_env={}
    if proxy_mode:
        runner=['python',str(Path(__file__).with_name('runner.py')),'--completion-url',completion,'--auth-mode',proxy_mode,'--timeout',str(int(params.get('request_timeout',600)))]
        if tok_backend=='remote':
            info=(proto.get('tokenizer_info') or {}).get('url'); tokenize=(proto.get('tokenize') or {}).get('url'); detokenize=(proto.get('detokenize') or {}).get('url')
            if not info or not tokenize or not detokenize: raise AdapterError('COMPATIBILITY_ERROR','remote lm-eval tokenizer requires tokenizer_info/tokenize/detokenize service protocols')
            runner += ['--tokenizer-info-url',info,'--tokenize-url',tokenize,'--detokenize-url',detokenize]
        runner += ['--',*harness]; argv=runner
        if proxy_mode=='inject': secret_env['MODEL_EVAL_UPSTREAM_API_KEY']=auth['secret_ref']
    else:
        argv=harness
    spec={"schema_version":"1.0","argv":argv,"cwd":str(framework_root),"env_patch":env,"stdin":{"mode":"null"},"stdout":{"mode":"file","path":str(Path(i.get('log_path') or out/'evaluation.log'))},"stderr":{"mode":"merge_stdout"},"timeout_seconds":float(params.get('evaluation_timeout_seconds',86400)),"metadata":{"transport_proxy":proxy_mode or 'none',"framework_revision":framework_revision,"framework_dirty":framework_dirty,"reproducibility":{"random_seed":seeds['random'],"numpy_random_seed":seeds['numpy'],"torch_random_seed":seeds['torch'],"fewshot_random_seed":seeds['fewshot'],"request_seed":seeds['request'],"pythonhashseed":seeds['pythonhash']}}}
    if secret_env: spec['secret_env']=secret_env
    return {"process":spec,"raw_result_root":str(out)}

def _find_result(root):
    files=sorted(Path(root).rglob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for p in files:
        try:
            obj=json_loads(p.read_text(encoding='utf-8'))
            if isinstance(obj,dict) and (isinstance(obj.get('results'),dict) or isinstance(obj.get('groups'),dict)):
                return p,obj
        except Exception:
            pass
    raise AdapterError('RESULT_INVALID',f'no lm_eval result JSON found under {root}')

def _metric_key_parts(key: str):
    head, sep, filter_name = key.partition(',')
    return head, filter_name if sep else None

def _is_stderr_metric_key(key: str) -> bool:
    head,_=_metric_key_parts(key)
    return head.endswith('_stderr')

def _stderr_candidates(key: str):
    head,filter_name=_metric_key_parts(key)
    out=[]
    if filter_name is not None:
        out.append(f'{head}_stderr,{filter_name}')
    out.append(f'{head}_stderr')
    return out

_RESULT_METADATA_KEYS={
    'alias','name','sample_len','sample_count','samples','n-samples','n_samples',
    'num_fewshot','config','version','task_name','group_name'
}

def _select_result_row(obj,task_id):
    results=obj.get('results') if isinstance(obj.get('results'),dict) else {}
    groups=obj.get('groups') if isinstance(obj.get('groups'),dict) else {}
    if task_id in groups:
        return groups[task_id],'group'
    if task_id in results:
        return results[task_id],'task'
    raise AdapterError('RESULT_INVALID',f'task/group result missing: {task_id}')

def _higher_is_better(obj,task_id,source):
    maps=[]
    if source=='group':
        maps.append(obj.get('group_higher_is_better'))
    maps.append(obj.get('higher_is_better'))
    for mapping in maps:
        if isinstance(mapping,dict) and isinstance(mapping.get(task_id),dict):
            return mapping[task_id]
    return {}

def _is_metric_candidate(key,value,row,hib):
    if key in _RESULT_METADATA_KEYS or _is_stderr_metric_key(key) or isinstance(value,(dict,list)):
        return False
    if not (isinstance(value,(int,float,bool)) or value is None):
        return False
    head,filter_name=_metric_key_parts(key)
    if filter_name is not None:
        return True
    if isinstance(hib,dict) and (key in hib or head in hib):
        return True
    return any(candidate in row for candidate in _stderr_candidates(key))

def _row_metrics(row,hib):
    metrics={}
    for k,v in row.items():
        if not _is_metric_candidate(k,v,row,hib):
            continue
        entry={"value":v}; head,_=_metric_key_parts(k)
        if isinstance(hib,dict):
            hib_value=hib.get(k) if k in hib else hib.get(head)
            if hib_value is not None:
                entry['higher_is_better']=bool(hib_value)
        for stderr_key in _stderr_candidates(k):
            if stderr_key not in row:
                continue
            stderr=row[stderr_key]
            # lm-eval uses strings such as "N/A" when stderr is undefined.
            # Omit the optional field instead of turning that sentinel into a
            # misleading number or leaking framework-specific null semantics.
            if isinstance(stderr,(int,float)) and not isinstance(stderr,bool):
                entry['stderr']=float(stderr)
            break
        metrics[k]=entry
    return metrics

def _canonical_metrics(native,metric_map,*,reject_collision=False):
    mapped={}; collision=False
    for framework_name,canonical_name in metric_map.items():
        if framework_name not in native:
            continue
        if canonical_name in mapped:
            collision=True
            if reject_collision:
                raise AdapterError('RESULT_INVALID',f'multiple framework metrics map to canonical metric {canonical_name}')
            continue
        mapped[canonical_name]=native[framework_name]
    return {} if collision else mapped

def _integer(value):
    return int(value) if isinstance(value,int) and not isinstance(value,bool) and value >= 0 else None

def _sample_count(obj,item_id,row):
    for table_name in ('n-samples','n_samples'):
        table=obj.get(table_name)
        value=table.get(item_id) if isinstance(table,dict) else None
        if isinstance(value,dict):
            count={}
            for source,target in (('original','original'),('effective','effective')):
                parsed=_integer(value.get(source))
                if parsed is not None: count[target]=parsed
            if count: return count
        parsed=_integer(value)
        if parsed is not None: return {'effective':parsed}
    count={}
    original=_integer(row.get('sample_count'))
    effective=_integer(row.get('sample_len'))
    if original is not None: count['original']=original
    if effective is not None: count['effective']=effective
    return count

def _table_value(obj,names,item_id):
    for name in names:
        table=obj.get(name)
        if isinstance(table,dict) and item_id in table:
            return table[item_id]
    return None

def _breakdown_detail(obj,item_id,row,kind,metric_map):
    native=_row_metrics(row,_higher_is_better(obj,item_id,kind))
    detail={'metrics':native}
    canonical=_canonical_metrics(native,metric_map)
    if canonical: detail['canonical_metrics']=canonical
    label=next((row.get(k) for k in ('alias','name','task_name','group_name') if isinstance(row.get(k),str) and row.get(k)),None)
    configs=obj.get('configs') if isinstance(obj.get('configs'),dict) else {}
    config=configs.get(item_id)
    if label is None and isinstance(config,dict):
        label=next((config.get(k) for k in ('task_alias','group_alias','alias','task') if isinstance(config.get(k),str) and config.get(k)),None)
    if label is not None: detail['label']=label
    counts=_sample_count(obj,item_id,row)
    if counts: detail['sample_count']=counts
    fewshot=_table_value(obj,('n-shot','n_shot'),item_id)
    if fewshot is None: fewshot=row.get('num_fewshot')
    if fewshot is None and isinstance(config,dict): fewshot=config.get('num_fewshot')
    parsed_fewshot=_integer(fewshot)
    if parsed_fewshot is not None: detail['num_fewshot']=parsed_fewshot
    version=_table_value(obj,('versions',),item_id)
    if version is None: version=row.get('version')
    if isinstance(version,(str,int,float)) and not isinstance(version,bool):
        detail['version']=version
    if isinstance(config,dict): detail['config']=config
    if kind=='group':
        subtasks=(obj.get('group_subtasks') or {}).get(item_id) if isinstance(obj.get('group_subtasks'),dict) else None
        if isinstance(subtasks,list) and all(isinstance(x,str) and x for x in subtasks): detail['subtasks']=subtasks
    return detail

def _result_breakdowns(obj,task_id,source,summary_metrics,native_summary,namespace,metric_map):
    groups={}
    for item_id,row in sorted((obj.get('groups') or {}).items()):
        if isinstance(item_id,str) and item_id and isinstance(row,dict):
            groups[item_id]=_breakdown_detail(obj,item_id,row,'group',metric_map)
    tasks={}
    for item_id,row in sorted((obj.get('results') or {}).items()):
        # lm-eval repeats aggregate group rows in ``results`` as well as
        # ``groups``. Keep the public product tables disjoint so a 24-task
        # benchmark is reported as one group plus 24 tasks, not 25 tasks.
        if isinstance(item_id,str) and item_id and item_id not in groups and isinstance(row,dict):
            tasks[item_id]=_breakdown_detail(obj,item_id,row,'task',metric_map)
    summary={'id':task_id,'kind':source,'metric_namespace':namespace,'metrics':summary_metrics}
    if namespace=='canonical': summary['native_metrics']=native_summary
    return {'summary':summary,'groups':groups,'tasks':tasks}

def _sample_artifacts(root):
    artifacts=[]
    for path in sorted(Path(root).rglob('*.jsonl')):
        if path.is_file() and not path.is_symlink():
            artifacts.append({'path':str(path.resolve()),'media_type':'application/x-ndjson'})
    return artifacts

def normalize(i,c):
    p,obj=_find_result(i['raw_result_root']); task=i['task']; run=i['run_metadata']
    row,source=_select_result_row(obj,task['task_id'])
    if not isinstance(row,dict):
        raise AdapterError('RESULT_INVALID',f'task/group result missing: {task["task_id"]}')
    native_metrics=_row_metrics(row,_higher_is_better(obj,task['task_id'],source))
    if not native_metrics:
        raise AdapterError('RESULT_INVALID','no scalar metrics found in lm_eval result')
    metric_contract=task.get('metrics') or {}; namespace=str(metric_contract.get('namespace') or 'framework_native'); metric_map=metric_contract.get('mapping') or {}; required=list(metric_contract.get('required') or [])
    if namespace=='canonical':
        if not isinstance(metric_map,dict): raise AdapterError('RESULT_INVALID','canonical metric contract requires task.metrics.mapping')
        mapped=_canonical_metrics(native_metrics,metric_map,reject_collision=True)
        missing=[name for name in required if name not in mapped]
        if missing: raise AdapterError('RESULT_INVALID',f'missing required canonical metrics: {missing}')
        metrics=mapped
    elif namespace=='framework_native':
        metrics=native_metrics
    else:
        raise AdapterError('RESULT_INVALID',f'unsupported metric namespace: {namespace}')
    result={"schema_version":"1.0","run_id":run['run_id'],"model":run['model'],"benchmark":run['benchmark'],"framework":"lm_eval","metrics":metrics,"raw_result":{"path":str(p),"media_type":"application/json"},"breakdowns":_result_breakdowns(obj,task['task_id'],source,metrics,native_metrics,namespace,metric_map),"metadata":{"task_id":task['task_id'],"result_scope":source,"protocol_fingerprint":task['protocol_fingerprint'],"metric_namespace":namespace}}
    sample_artifacts=_sample_artifacts(i['raw_result_root'])
    if sample_artifacts: result['sample_artifacts']=sample_artifacts
    return result

def snapshot(i,c):
    params=(i.get('evaluation') or {}).get('parameters') or {}
    adapter_version=str(load_manifest(Path(__file__).with_name('manifest.json'))['version'])
    try:
        root,revision,dirty=_framework_source(params)
        return {"framework":"lm_eval","adapter_version":adapter_version,"framework_root":str(root),"framework_revision":revision,"framework_dirty":dirty}
    except AdapterError as exc:
        return {"framework":"lm_eval","adapter_version":adapter_version,"framework_root":params.get('tool_root') or params.get('framework_root'),"probe_error":str(exc)}
OPERATIONS={"requirements":requirements,"plan_preflight":plan_preflight,"plan_evaluate":plan_evaluate,"normalize":normalize,"snapshot":snapshot}
