from __future__ import annotations
from pathlib import Path
from model_evaluation.sdk.http import normalize_http_base_url
from model_evaluation.sdk.openai_service import probe_openai_service
from model_evaluation.sdk.runtime import AdapterError

def _auth(value):
    auth=value or {'mode':'none'}; mode=auth.get('mode'); ref=auth.get('secret_ref')
    if mode=='none' and ref is not None: raise AdapterError('CONFIG_INVALID','auth.mode=none may not carry secret_ref')
    if mode=='bearer' and not isinstance(ref,str): raise AdapterError('CONFIG_INVALID','auth.mode=bearer requires secret_ref')
    if mode not in {'none','bearer'}: raise AdapterError('CONFIG_INVALID',f'unsupported auth mode: {mode}')
    return auth

def _model_tag(model, dep):
    params=dep.get('parameters') or {}; ep=dep.get('endpoint') or {}
    value=ep.get('model_id') or params.get('model_id') or params.get('api_model_name') or (model.get('source') or {}).get('ref') or model.get('id')
    tag=str(value or '')
    if not tag or any(x.isspace() for x in tag) or ',' in tag:
        raise AdapterError('CONFIG_INVALID','Ollama service model id/tag must be non-empty and contain no whitespace/comma')
    return tag

def requirements(i,c):
    dep=i.get('deployment') or {}; mode=(dep.get('management') or {}).get('mode'); params=dep.get('parameters') or {}; req=[]
    if params.get('extra_args'): raise AdapterError('CONFIG_INVALID','Ollama server adapter does not accept extra_args; configure Ollama through its profile/environment')
    if mode=='managed': req.append({"path":"runtime.available","op":"equals","value":True,"message":"managed Ollama requires an available local runtime"})
    elif mode in {'attached','external'}:
        ep=dep.get('endpoint') or {}; normalize_http_base_url(ep.get('base_url')); _auth(ep.get('auth'))
        tok=ep.get('tokenizer_path')
        if tok and (not Path(str(tok)).is_absolute() or not Path(str(tok)).exists()): raise AdapterError('DEPENDENCY_MISSING',f'local tokenizer_path unavailable: {tok}')
    else: raise AdapterError('CONFIG_INVALID',f'unsupported Ollama management mode: {mode}')
    _model_tag(i.get('model') or {},dep)
    return {"schema_version":"1.0","requirements":req}

def plan_start(i,c):
    model=i['model']; dep=i['deployment']; mode=dep['management']['mode']; params=dep.get('parameters') or {}; tag=_model_tag(model,dep); ep=dep.get('endpoint') or {}
    if mode!='managed':
        return {"schema_version":"1.1","attach":{"base_url":normalize_http_base_url(ep.get('base_url')),"model_id":str(ep.get('model_id') or tag),"ownership":mode,"auth":_auth(ep.get('auth')),"context_length":ep.get('context_length'),"tokenizer_path":ep.get('tokenizer_path'),"num_concurrent":int(params.get('num_concurrent',16))},"readiness":{"timeout_seconds":float(params.get('ready_timeout_seconds',30))}}
    endpoint=i.get('endpoint') or {}; host=str(endpoint.get('host') or '127.0.0.1'); port=int(endpoint.get('port') or params.get('port') or 11434); exe=str(params.get('executable') or 'ollama'); tokenizer=(dep.get('model_location') or {}).get('tokenizer_path')
    spec={"schema_version":"1.0","argv":[exe,'serve'],"cwd":str(params.get('cwd') or '.'),"env_patch":{"set":{"OLLAMA_HOST":f'{host}:{port}'}},"stdin":{"mode":"null"},"stdout":{"mode":"file","path":str(i.get('log_path') or 'ollama.log')},"stderr":{"mode":"merge_stdout"},"metadata":{}}
    probe={"schema_version":"1.0","argv":[exe,'--version'],"env_patch":{},"stdin":{"mode":"null"},"stdout":{"mode":"capture"},"stderr":{"mode":"capture"},"timeout_seconds":8.0,"metadata":{}}
    return {"schema_version":"1.1","process":spec,"dependency_probe":probe,"shutdown":{"strategy":"signal","signal":"SIGTERM","timeout_seconds":float(params.get('shutdown_timeout_seconds',15.0))},"readiness":{"timeout_seconds":float(params.get('ready_timeout_seconds',180))},"attach":{"base_url":f'http://{host}:{port}/v1',"model_id":tag,"ownership":"managed","auth":{"mode":"none"},"context_length":params.get('context_length'),"tokenizer_path":tokenizer,"num_concurrent":int(params.get('num_concurrent',16))}}

def probe_service(i,c):
    a=i.get('attach') or {}; return probe_openai_service(base_url=a['base_url'],model=str(a['model_id']),ownership=a.get('ownership','managed'),auth=a.get('auth') or {'mode':'none'},bearer=i.get('auth_value'),timeout=float(c.get('timeout_seconds',3)),tokenizer_path=a.get('tokenizer_path'),context_length=a.get('context_length'),probe_root_tokenizer=False,num_concurrent=int(a.get('num_concurrent',16)))

def snapshot(i,c):
    exe=str((i.get('deployment') or {}).get('parameters',{}).get('executable') or 'ollama')
    return {'backend':'ollama','configured_executable':exe,'version_source':'selected_environment_probe'}
OPERATIONS={'requirements':requirements,'plan_start':plan_start,'probe_service':probe_service,'snapshot':snapshot}
