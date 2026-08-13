from __future__ import annotations
from pathlib import Path
from model_evaluation.sdk.http import normalize_http_base_url
from model_evaluation.sdk.openai_service import probe_openai_service
from model_evaluation.sdk.runtime import AdapterError
RESERVED={'--host','--port','--model','-m','--alias','--ctx-size','-c'}

def _extra(params):
    extra=params.get('extra_args') or []
    if not isinstance(extra,list) or not all(isinstance(x,str) and x for x in extra): raise AdapterError('CONFIG_INVALID','parameters.extra_args must be a string array')
    for arg in extra:
        if arg.split('=',1)[0] in RESERVED: raise AdapterError('CONFIG_CONFLICT',f'extra_args may not override adapter-owned field {arg}')
    return extra

def _auth(value):
    auth=value or {'mode':'none'}; mode=auth.get('mode'); ref=auth.get('secret_ref')
    if mode=='none' and ref is not None: raise AdapterError('CONFIG_INVALID','auth.mode=none may not carry secret_ref')
    if mode=='bearer' and not isinstance(ref,str): raise AdapterError('CONFIG_INVALID','auth.mode=bearer requires secret_ref')
    if mode not in {'none','bearer'}: raise AdapterError('CONFIG_INVALID',f'unsupported auth mode: {mode}')
    return auth

def requirements(i,c):
    dep=i.get('deployment') or {}; mode=(dep.get('management') or {}).get('mode'); params=dep.get('parameters') or {}; req=[]
    if mode=='managed':
        loc=(dep.get('model_location') or {}).get('local_path')
        if not loc or not Path(str(loc)).is_absolute() or not Path(str(loc)).is_file(): raise AdapterError('DEPENDENCY_MISSING',f'llama.cpp requires an existing absolute GGUF model file: {loc}')
        tok=(dep.get('model_location') or {}).get('tokenizer_path')
        if tok and (not Path(str(tok)).is_absolute() or not Path(str(tok)).is_dir()): raise AdapterError('DEPENDENCY_MISSING',f'tokenizer_path unavailable: {tok}')
        _extra(params); req.append({"path":"runtime.available","op":"equals","value":True,"message":"managed llama.cpp requires an available local runtime"})
    elif mode in {'attached','external'}:
        ep=dep.get('endpoint') or {}; normalize_http_base_url(ep.get('base_url')); _auth(ep.get('auth'))
        tok=ep.get('tokenizer_path')
        if tok and (not Path(str(tok)).is_absolute() or not Path(str(tok)).exists()): raise AdapterError('DEPENDENCY_MISSING',f'local tokenizer_path unavailable: {tok}')
    else: raise AdapterError('CONFIG_INVALID',f'unsupported llama.cpp management mode: {mode}')
    return {"schema_version":"1.0","requirements":req}

def plan_start(i,c):
    model=i['model']; dep=i['deployment']; mode=dep['management']['mode']; params=dep.get('parameters') or {}; ep=dep.get('endpoint') or {}
    if mode!='managed':
        return {"schema_version":"1.1","attach":{"base_url":normalize_http_base_url(ep.get('base_url')),"model_id":str(ep.get('model_id') or model['id']),"ownership":mode,"auth":_auth(ep.get('auth')),"context_length":ep.get('context_length'),"tokenizer_path":ep.get('tokenizer_path'),"num_concurrent":int(params.get('num_concurrent',16))},"readiness":{"timeout_seconds":float(params.get('ready_timeout_seconds',30))}}
    loc=str((dep.get('model_location') or {}).get('local_path')); tokenizer=(dep.get('model_location') or {}).get('tokenizer_path'); endpoint=i.get('endpoint') or {}; host=str(endpoint.get('host') or '127.0.0.1'); port=int(endpoint.get('port') or params.get('port') or 8091); api_model=str(params.get('api_model_name') or model['id']); ctx=int(params.get('context_length') or params.get('max_model_len') or 4096)
    argv=[str(params.get('executable') or 'llama-server'),'--model',loc,'--alias',api_model,'--host',host,'--port',str(port),'--ctx-size',str(ctx),*_extra(params)]
    spec={"schema_version":"1.0","argv":argv,"cwd":str(params.get('cwd') or Path(loc).parent),"env_patch":{},"stdin":{"mode":"null"},"stdout":{"mode":"file","path":str(i.get('log_path') or 'llama_cpp.log')},"stderr":{"mode":"merge_stdout"},"metadata":{}}
    probe={"schema_version":"1.0","argv":[str(params.get('executable') or 'llama-server'),'--version'],"env_patch":{},"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"},"timeout_seconds":8.0,"metadata":{}}
    return {"schema_version":"1.1","process":spec,"dependency_probe":probe,"shutdown":{"strategy":"signal","signal":"SIGTERM","timeout_seconds":float(params.get('shutdown_timeout_seconds',15.0))},"readiness":{"timeout_seconds":float(params.get('ready_timeout_seconds',300))},"attach":{"base_url":f'http://{host}:{port}/v1',"model_id":api_model,"ownership":"managed","auth":{"mode":"none"},"context_length":ctx,"tokenizer_path":tokenizer,"num_concurrent":int(params.get('num_concurrent',16))}}

def probe_service(i,c):
    a=i.get('attach') or {}; return probe_openai_service(base_url=a['base_url'],model=str(a['model_id']),ownership=a.get('ownership','managed'),auth=a.get('auth') or {'mode':'none'},bearer=i.get('auth_value'),timeout=float(c.get('timeout_seconds',3)),tokenizer_path=a.get('tokenizer_path'),context_length=a.get('context_length'),probe_root_tokenizer=True,num_concurrent=int(a.get('num_concurrent',16)))

def snapshot(i,c):
    exe=str((i.get('deployment') or {}).get('parameters',{}).get('executable') or 'llama-server')
    return {'backend':'llama_cpp','configured_executable':exe,'version_source':'selected_environment_probe'}
OPERATIONS={'requirements':requirements,'plan_start':plan_start,'probe_service':probe_service,'snapshot':snapshot}
