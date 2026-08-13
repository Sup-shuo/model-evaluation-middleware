from __future__ import annotations
from pathlib import Path
from model_evaluation.sdk.http import normalize_http_base_url, optional_json_probe, request_json
from model_evaluation.sdk.openai_service import probe_completion_capabilities
from model_evaluation.sdk.runtime import AdapterError

def _auth(value):
    auth=value or {"mode":"none"}; mode=auth.get('mode'); ref=auth.get('secret_ref')
    if mode=='none' and ref is not None: raise AdapterError('CONFIG_INVALID','auth.mode=none may not carry secret_ref')
    if mode=='bearer' and not isinstance(ref,str): raise AdapterError('CONFIG_INVALID','auth.mode=bearer requires secret_ref')
    if mode not in {'none','bearer'}: raise AdapterError('CONFIG_INVALID',f'unsupported auth mode: {mode}')
    return auth

def requirements(i,c):
    dep=i.get('deployment') or {}; mode=(dep.get('management') or {}).get('mode')
    if mode not in {'attached','external'}: raise AdapterError('CONFIG_INVALID','generic_openai supports attached/external management only')
    ep=dep.get('endpoint') or {}; normalize_http_base_url(ep.get('base_url')); _auth(ep.get('auth'))
    tok=ep.get('tokenizer_path')
    if tok and (not Path(str(tok)).is_absolute() or not Path(str(tok)).exists()): raise AdapterError('DEPENDENCY_MISSING',f'local tokenizer_path unavailable: {tok}')
    if not (ep.get('model_id') or (i.get('model') or {}).get('source',{}).get('ref') or (i.get('model') or {}).get('id')): raise AdapterError('CONFIG_INVALID','external endpoint model id is required')
    return {"schema_version":"1.0","requirements":[]}

def plan_start(i,c):
    dep=i['deployment']; mode=dep['management']['mode']
    if mode not in {'attached','external'}: raise AdapterError('CONFIG_INVALID','generic_openai supports attached/external management only')
    ep=dep.get('endpoint') or {}; base=normalize_http_base_url(ep.get('base_url')); model=str(ep.get('model_id') or (i.get('model') or {}).get('source',{}).get('ref') or (i.get('model') or {}).get('id') or '')
    if not model: raise AdapterError('CONFIG_INVALID','external endpoint model id is required')
    attach={"base_url":base,"model_id":model,"ownership":mode,"auth":_auth(ep.get('auth')),"declared_capabilities":ep.get('capabilities') or {}}
    if ep.get('tokenizer_path'): attach['tokenizer_path']=str(ep['tokenizer_path'])
    if ep.get('context_length') is not None: attach['context_length']=int(ep['context_length'])
    if ep.get('tokenizer_info_url'): attach['tokenizer_info_url']=normalize_http_base_url(ep['tokenizer_info_url'])
    if ep.get('tokenize_url'): attach['tokenize_url']=normalize_http_base_url(ep['tokenize_url'])
    if ep.get('detokenize_url'): attach['detokenize_url']=normalize_http_base_url(ep['detokenize_url'])
    return {"schema_version":"1.1","attach":attach,"readiness":{"timeout_seconds":float((dep.get('parameters') or {}).get('ready_timeout_seconds',30))}}

def probe_service(i,c):
    a=i.get('attach') or {}; base=normalize_http_base_url(a.get('base_url')); model=str(a.get('model_id') or ''); timeout=float(c.get('timeout_seconds',20)); request_timeout=max(0.25,min(3.0,timeout/8.0)); token=i.get('auth_value')
    _,payload=request_json(base+'/models',bearer=token,timeout=request_timeout); ids={str(x.get('id')) for x in (payload.get('data') or []) if isinstance(x,dict) and x.get('id') is not None} if isinstance(payload,dict) else set()
    if not ids: raise AdapterError('SERVICE_NOT_READY','/models did not report any model ids',retryable=True)
    if model not in ids: raise AdapterError('SERVICE_NOT_READY',f'model {model!r} not listed by endpoint',retryable=False)
    declared=dict(a.get('declared_capabilities') or {}); evidence={"models":"observed"}
    generation,logprobs,echo,completion_evidence=probe_completion_capabilities(base=base,model=model,bearer=token,timeout=request_timeout)
    chat_ok,chat_obj,chat_err=optional_json_probe(base+'/chat/completions',method='POST',payload={"model":model,"messages":[{"role":"user","content":"Hi"}],"max_tokens":1,"temperature":0},bearer=token,timeout=request_timeout); chat_choice=(chat_obj.get('choices') or [{}])[0] if isinstance(chat_obj,dict) else {}; chat=bool(chat_ok and isinstance(chat_obj,dict) and isinstance(chat_obj.get('choices'),list) and chat_obj['choices'] and isinstance(chat_choice,dict) and isinstance(chat_choice.get('message'),dict))
    protocols={"openai_models":{"url":base+'/models'}}
    if generation or logprobs or echo: protocols['openai_completion']={"url":base+'/completions'}
    if chat: protocols['openai_chat']={"url":base+'/chat/completions'}
    evidence['chat']='observed' if chat else f'probe_failed:{chat_err}'
    tokenize=False; detokenize=False; token_ids=[]; info_ok=False
    if a.get('tokenizer_info_url'):
        info_ok,_,info_err=optional_json_probe(a['tokenizer_info_url'],method='GET',bearer=token,timeout=request_timeout); evidence['tokenizer_info']='observed' if info_ok else f'probe_failed:{info_err}'
        if info_ok: protocols['tokenizer_info']={"url":a['tokenizer_info_url']}
    if a.get('tokenize_url'):
        t_ok,t_obj,t_err=optional_json_probe(a['tokenize_url'],method='POST',payload={"model":model,"prompt":"hello"},bearer=token,timeout=request_timeout)
        if t_ok and isinstance(t_obj,dict): token_ids=t_obj.get('tokens') or t_obj.get('token_ids') or []
        tokenize=bool(info_ok and t_ok and isinstance(token_ids,list)); evidence['tokenize']='observed' if tokenize else f'probe_failed:{t_err}'
        if tokenize: protocols['tokenize']={"url":a['tokenize_url']}
    if a.get('detokenize_url'):
        if tokenize and token_ids:
            d_ok,d_obj,d_err=optional_json_probe(a['detokenize_url'],method='POST',payload={"model":model,"tokens":token_ids[:8]},bearer=token,timeout=request_timeout); detokenize=bool(d_ok and isinstance(d_obj,dict)); evidence['detokenize']='observed' if detokenize else f'probe_failed:{d_err}'
            if detokenize: protocols['detokenize']={"url":a['detokenize_url']}
        else: evidence['detokenize']='not_probed_without_tokens'
    evidence.update(completion_evidence)
    caps={"service.generation":generation,"service.chat":chat,"service.completion_logprobs":logprobs,"service.echo":echo,"service.tokenize":tokenize,"service.detokenize":detokenize}
    desc={"schema_version":"1.0","service_type":"llm","ownership":a.get('ownership','external'),"model":{"id":model},"protocols":protocols,"capabilities":{"schema_version":"1.0","values":caps},"auth":a.get('auth') or {"mode":"none"},"tokenizer":{"mode":"local","path":str(a["tokenizer_path"])} if a.get("tokenizer_path") else {"mode":"remote" if tokenize and detokenize else "none"},"limits":{"recommended_concurrency":int(a.get("num_concurrent",16))},"metadata":{"capability_evidence":evidence,"declared_capabilities":declared}}
    if a.get('context_length') is not None: desc['context_length']=int(a['context_length']); desc['metadata']['context_length_evidence']='declared_by_endpoint_profile'
    return desc

def snapshot(i,c): return {"backend":"generic_openai","management":"external"}
OPERATIONS={"requirements":requirements,"plan_start":plan_start,"probe_service":probe_service,"snapshot":snapshot}
