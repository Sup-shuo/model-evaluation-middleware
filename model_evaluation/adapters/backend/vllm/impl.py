from __future__ import annotations
import json
from pathlib import Path
from model_evaluation.sdk.http import normalize_http_base_url, optional_json_probe, request_json
from model_evaluation.sdk.openai_service import probe_completion_capabilities
from model_evaluation.sdk.runtime import AdapterError

RESERVED={
    "--served-model-name","--host","--port","--tokenizer","--tokenizer-mode","--max-model-len","--seed",
    "--generation-config","--api-key","--tensor-parallel-size","--dtype","--gpu-memory-utilization","--max-num-seqs",
    "--trust-remote-code","--chat-template","--override-generation-config","--enable-reasoning","--reasoning-parser",
    "--language-model-only",
}

def _auth(value):
    auth=value or {"mode":"none"}; mode=auth.get('mode'); ref=auth.get('secret_ref')
    if mode=='none' and ref is not None: raise AdapterError('CONFIG_INVALID','auth.mode=none may not carry secret_ref')
    if mode=='bearer' and not isinstance(ref,str): raise AdapterError('CONFIG_INVALID','auth.mode=bearer requires secret_ref')
    if mode not in {'none','bearer'}: raise AdapterError('CONFIG_INVALID',f'unsupported auth mode: {mode}')
    return auth

def _extra(params):
    extra=params.get('extra_args') or []
    if not isinstance(extra,list) or not all(isinstance(x,str) and x for x in extra): raise AdapterError('CONFIG_INVALID','parameters.extra_args must be a string array')
    for arg in extra:
        flag=arg.split('=',1)[0]
        if flag in RESERVED: raise AdapterError('CONFIG_CONFLICT',f'extra_args may not override adapter-owned field {flag}')
    return extra

def requirements(i,c):
    dep=i.get('deployment') or {}; model=i.get('model') or {}; mode=(dep.get('management') or {}).get('mode'); params=dep.get('parameters') or {}; req=[]
    if mode=='managed':
        loc=(dep.get('model_location') or {}).get('local_path')
        if not loc or not Path(str(loc)).is_absolute(): raise AdapterError('CONFIG_INVALID','managed vLLM requires absolute model_location.local_path')
        model_path=Path(str(loc))
        if not model_path.is_dir(): raise AdapterError('DEPENDENCY_MISSING',f'managed model directory not found: {model_path}')
        tokenizer=model.get('tokenizer') or {}; tok=(dep.get('model_location') or {}).get('tokenizer_path') or (tokenizer.get('ref') if isinstance(tokenizer,dict) else None)
        if tok and Path(str(tok)).is_absolute() and not Path(str(tok)).exists(): raise AdapterError('DEPENDENCY_MISSING',f'tokenizer path unavailable: {tok}')
        _extra(params); tp=int(params.get('tensor_parallel_size',1))
        if tp < 1: raise AdapterError('CONFIG_INVALID','tensor_parallel_size must be >= 1')
        req += [
            {"path":"runtime.available","op":"equals","value":True,"message":"managed vLLM requires an available local runtime"},
            {"path":"device.count","op":"gte","value":tp,"message":f"vLLM tensor_parallel_size={tp} requires at least {tp} selected devices"},
        ]
    elif mode in {'attached','external'}:
        ep=dep.get('endpoint') or {}; normalize_http_base_url(ep.get('base_url')); _auth(ep.get('auth'))
        tok=ep.get('tokenizer_path')
        if tok and (not Path(str(tok)).is_absolute() or not Path(str(tok)).exists()): raise AdapterError('DEPENDENCY_MISSING',f'local tokenizer_path unavailable: {tok}')
    else: raise AdapterError('CONFIG_INVALID',f'unsupported vLLM management mode: {mode}')
    return {"schema_version":"1.0","requirements":req}

def plan_start(i,c):
    model=i['model']; dep=i['deployment']; mode=dep['management']['mode']; params=dep.get('parameters') or {}; endpoint=i.get('endpoint') or {}
    if mode!='managed':
        ep=dep.get('endpoint') or {}; base=normalize_http_base_url(ep.get('base_url') or endpoint.get('base_url')); auth=_auth(ep.get('auth'))
        attach={"base_url":base,"model_id":str(ep.get('model_id') or params.get('api_model_name') or model['id']),"ownership":mode,"auth":auth,"num_concurrent":int(params.get('num_concurrent',16))}
        if ep.get('context_length') is not None: attach['context_length']=int(ep['context_length'])
        if ep.get('tokenizer_path'): attach['tokenizer_path']=str(ep['tokenizer_path'])
        return {"schema_version":"1.1","attach":attach,"readiness":{"timeout_seconds":float(params.get('ready_timeout_seconds',30))}}
    loc=(dep.get('model_location') or {}).get('local_path')
    if not loc or not Path(loc).is_absolute(): raise AdapterError('CONFIG_INVALID','managed vLLM requires absolute model_location.local_path')
    host=str(endpoint.get('host') or '127.0.0.1'); port=int(endpoint.get('port') or params.get('port') or 8091); api_model=str(params.get('api_model_name') or model['id'])
    model_tokenizer=model.get('tokenizer') or {}; tokenizer_ref=model_tokenizer.get('ref') if isinstance(model_tokenizer,dict) else None
    tokenizer=str((dep.get('model_location') or {}).get('tokenizer_path') or tokenizer_ref or loc); max_len=int(params.get('max_model_len',model.get('context_length',4096))); auth=_auth((dep.get('endpoint') or {}).get('auth'))
    argv=[str(params.get('executable') or 'vllm'),'serve',loc,'--tokenizer',tokenizer,'--tokenizer-mode','auto','--served-model-name',api_model,'--host',host,'--port',str(port),'--max-model-len',str(max_len),'--seed',str(int(params.get('seed',1234))),'--generation-config',str(params.get('generation_config','vllm'))]
    managed={"tensor_parallel_size":"--tensor-parallel-size","dtype":"--dtype","gpu_memory_utilization":"--gpu-memory-utilization","max_num_seqs":"--max-num-seqs"}; defaults={"tensor_parallel_size":1,"dtype":"auto","gpu_memory_utilization":0.8,"max_num_seqs":16}
    for key,flag in managed.items(): argv += [flag,str(params.get(key,defaults[key]))]
    if params.get('trust_remote_code',model.get('trust_remote_code',False)): argv.append('--trust-remote-code')
    if params.get('language_model_only',False): argv.append('--language-model-only')
    chat_template=params.get('chat_template',model.get('chat_template'))
    if chat_template: argv += ['--chat-template',str(chat_template)]
    argv += _extra(params)
    env={"set":{"PYTHONHASHSEED":str(params.get('pythonhashseed',1234)),"TOKENIZERS_PARALLELISM":str(params.get('tokenizers_parallelism','false')).lower()}}
    cache=params.get('cache_root')
    if cache:
        hf=str(Path(cache)/'huggingface'); env['set'].update({"HF_HOME":hf,"HF_DATASETS_CACHE":str(Path(hf)/'datasets'),"HF_HUB_CACHE":str(Path(hf)/'hub')})
    offline=bool(c.get('offline',False) or i.get('network_policy')=='offline')
    if offline: env['set'].update({"HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
    # Do not default cwd to the model's parent. Model roots are frequently
    # object-backed mounts; putting one on Python's implicit sys.path can make
    # every import perform remote directory lookups. Core supplies a private
    # run workspace, while an explicit user `cwd` remains supported.
    workspace=c.get('workspace')
    default_cwd=str(workspace) if workspace and Path(str(workspace)).is_absolute() else None
    cwd=params.get('cwd') or default_cwd
    spec={"schema_version":"1.0","argv":argv,"env_patch":env,"stdin":{"mode":"null"},"stdout":{"mode":"file","path":str(i.get('log_path') or 'vllm.log')},"stderr":{"mode":"merge_stdout"},"metadata":{}}
    if cwd: spec['cwd']=str(cwd)
    if auth.get('mode')=='bearer': spec['secret_env']={"VLLM_API_KEY":auth['secret_ref']}
    probe={"schema_version":"1.0","argv":[str(params.get('executable') or 'vllm'),'--version'],"env_patch":env,"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"},"timeout_seconds":float(params.get('dependency_probe_timeout_seconds',45.0)),"metadata":{}}
    return {"schema_version":"1.1","process":spec,"dependency_probe":probe,"shutdown":{"strategy":"signal","signal":"SIGTERM","timeout_seconds":float(params.get('shutdown_timeout_seconds',15.0))},"readiness":{"timeout_seconds":float(params.get('ready_timeout_seconds',900))},"attach":{"base_url":f"http://{host}:{port}/v1","model_id":api_model,"ownership":"managed","auth":auth,"tokenizer_path":tokenizer,"context_length":max_len,"num_concurrent":int(params.get('num_concurrent',16))}}

def plan_preflight(i,c):
    model=i['model']; dep=i['deployment']; mode=dep['management']['mode']; params=dep.get('parameters') or {}
    if mode!='managed':
        raise AdapterError('CONFIG_INVALID','vLLM local preflight is only available for managed deployments')
    loc=str((dep.get('model_location') or {}).get('local_path') or '')
    if not loc or not Path(loc).is_absolute():
        raise AdapterError('CONFIG_INVALID','managed vLLM preflight requires absolute model_location.local_path')
    model_tokenizer=model.get('tokenizer') or {}; tokenizer_ref=model_tokenizer.get('ref') if isinstance(model_tokenizer,dict) else None
    tokenizer=str((dep.get('model_location') or {}).get('tokenizer_path') or tokenizer_ref or loc)
    max_len=int(params.get('max_model_len',model.get('context_length',4096)))
    timeout=float(params.get('dependency_probe_timeout_seconds',45.0))
    env={"set":{"PYTHONHASHSEED":str(params.get('pythonhashseed',1234)),"TOKENIZERS_PARALLELISM":str(params.get('tokenizers_parallelism','false')).lower()}}
    if bool(c.get('offline',False) or i.get('network_policy')=='offline'):
        env['set'].update({"HF_HUB_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
    payload={
        'model_path':loc,'tokenizer':tokenizer,'architecture':model.get('architecture'),
        'quantization':model.get('quantization'),'trust_remote_code':bool(params.get('trust_remote_code',model.get('trust_remote_code',False))),
        'max_model_len':max_len,'tensor_parallel_size':int(params.get('tensor_parallel_size',1)),
        'dtype':params.get('dtype','auto'),'gpu_memory_utilization':float(params.get('gpu_memory_utilization',0.8)),
        'max_num_seqs':int(params.get('max_num_seqs',16)),'generation_config':str(params.get('generation_config','vllm')),
        'language_model_only':bool(params.get('language_model_only',False)),
    }
    stdio={"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"},"metadata":{}}
    version={"schema_version":"1.0","argv":[str(params.get('executable') or 'vllm'),'--version'],"env_patch":env,"timeout_seconds":timeout,**stdio}
    compatibility={"schema_version":"1.0","argv":["python",str(Path(__file__).with_name('preflight.py').resolve()),'--payload',json.dumps(payload,separators=(',',':'),sort_keys=True)],"env_patch":env,"timeout_seconds":float(params.get('model_probe_timeout_seconds',120.0)),**stdio}
    return {"schema_version":"1.0","probes":[
        {"id":"backend.import","phase":"backend_dependency","required":True,"description":"Import vLLM and the selected platform plugin","result_format":"text","process":version},
        {"id":"model.config","phase":"model_compatibility","required":True,"description":"Build the backend-native model configuration without loading weights","result_format":"preflight_result","process":compatibility},
    ]}

def _model_info(payload, model):
    data=(payload.get('data') or []) if isinstance(payload,dict) else []
    items=[x for x in data if isinstance(x,dict)]; ids={str(x.get('id')) for x in items}
    if not ids: raise AdapterError('SERVICE_NOT_READY','/models did not report any model ids',retryable=True)
    if model not in ids: raise AdapterError('SERVICE_NOT_READY',f'model {model!r} not listed by service; available={sorted(ids)}',retryable=False)
    item=next((x for x in items if str(x.get('id'))==model),None)
    return item or (items[0] if len(items)==1 else {})

def probe_service(i,c):
    a=i.get('attach') or {}; base=normalize_http_base_url(a.get('base_url')); model=str(a.get('model_id') or '')
    if not model: raise AdapterError('CONFIG_INVALID','probe_service requires model_id')
    timeout=float(c.get('timeout_seconds',20)); request_timeout=max(0.25,min(3.0,timeout/8.0)); token=i.get('auth_value'); _,payload=request_json(base+'/models',bearer=token,timeout=request_timeout); info=_model_info(payload,model)
    protocols={"openai_models":{"url":base+'/models'}}; evidence={"models":"observed"}
    generation,logprobs,echo,completion_evidence=probe_completion_capabilities(base=base,model=model,bearer=token,timeout=request_timeout)
    evidence.update(completion_evidence)
    if generation or logprobs or echo: protocols['openai_completion']={"url":base+'/completions'}
    chat_ok,chat_obj,chat_err=optional_json_probe(base+'/chat/completions',method='POST',payload={"model":model,"messages":[{"role":"user","content":"Hi"}],"max_tokens":1,"temperature":0},bearer=token,timeout=request_timeout)
    chat_choice=(chat_obj.get('choices') or [{}])[0] if isinstance(chat_obj,dict) else {}; chat=bool(chat_ok and isinstance(chat_obj,dict) and isinstance(chat_obj.get('choices'),list) and chat_obj['choices'] and isinstance(chat_choice,dict) and isinstance(chat_choice.get('message'),dict)); evidence['chat']='observed' if chat else f"probe_failed:{chat_err}"
    if chat: protocols['openai_chat']={"url":base+'/chat/completions'}
    root=base[:-3] if base.endswith('/v1') else base; info_url=root+'/tokenizer_info'; tokenize_url=root+'/tokenize'; detokenize_url=root+'/detokenize'
    info_ok,info_obj,info_err=optional_json_probe(info_url,method='GET',bearer=token,timeout=request_timeout)
    tok_ok,tok_obj,tok_err=optional_json_probe(tokenize_url,method='POST',payload={"model":model,"prompt":"hello"},bearer=token,timeout=request_timeout); token_ids=[]
    if tok_ok and isinstance(tok_obj,dict): token_ids=tok_obj.get('tokens') or tok_obj.get('token_ids') or []
    tokenize=bool(info_ok and tok_ok and isinstance(token_ids,list)); evidence['tokenizer_info']='observed' if info_ok else f"probe_failed:{info_err}"; evidence['tokenize']='observed' if tokenize else f"probe_failed:{tok_err}"
    if info_ok: protocols['tokenizer_info']={"url":info_url}
    if tokenize: protocols['tokenize']={"url":tokenize_url}
    detokenize=False
    if tokenize and token_ids:
        det_ok,det_obj,det_err=optional_json_probe(detokenize_url,method='POST',payload={"model":model,"tokens":token_ids[:8]},bearer=token,timeout=request_timeout); detokenize=bool(det_ok and isinstance(det_obj,dict)); evidence['detokenize']='observed' if detokenize else f"probe_failed:{det_err}"
        if detokenize: protocols['detokenize']={"url":detokenize_url}
    else: evidence['detokenize']='not_probed_without_tokens'
    caps={"service.generation":generation,"service.chat":chat,"service.completion_logprobs":logprobs,"service.echo":echo,"service.tokenize":tokenize,"service.detokenize":detokenize}
    context=a.get('context_length')
    if context is None:
        for key in ('max_model_len','max_model_length','context_length'):
            if isinstance(info,dict) and isinstance(info.get(key),(int,float)): context=int(info[key]); evidence['context_length']='observed_models'; break
    elif a.get('ownership')=='managed': evidence['context_length']='declared_by_managed_start_config'
    else: evidence['context_length']='declared_by_endpoint_profile'
    desc={"schema_version":"1.0","service_type":"llm","ownership":a.get('ownership','managed'),"model":{"id":model},"protocols":protocols,"capabilities":{"schema_version":"1.0","values":caps},"auth":a.get('auth') or {"mode":"none"},"tokenizer":{"mode":"local","path":str(a['tokenizer_path'])} if a.get('tokenizer_path') else {"mode":"remote" if tokenize and detokenize else "none"},"limits":{"recommended_concurrency":int(a.get('num_concurrent',16))},"metadata":{"capability_evidence":evidence}}
    if context is not None: desc['context_length']=int(context)
    return desc

def snapshot(i,c):
    exe=str((i.get('deployment') or {}).get('parameters',{}).get('executable') or 'vllm')
    return {'backend':'vllm','configured_executable':exe,'version_source':'selected_environment_probe'}
OPERATIONS={"requirements":requirements,"plan_preflight":plan_preflight,"plan_start":plan_start,"probe_service":probe_service,"snapshot":snapshot}
