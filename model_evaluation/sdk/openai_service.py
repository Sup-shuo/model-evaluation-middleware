from __future__ import annotations
from model_evaluation.sdk.http import normalize_http_base_url, optional_json_probe, optional_json_probe_detail, request_json
from model_evaluation.sdk.runtime import AdapterError


def _listed_model(payload: object, model: str) -> dict:
    if not isinstance(payload,dict): raise AdapterError('SERVICE_NOT_READY','/models returned a non-object response',retryable=True)
    items=[x for x in (payload.get('data') or []) if isinstance(x,dict) and x.get('id') is not None]
    ids={str(x['id']) for x in items}
    if not ids: raise AdapterError('SERVICE_NOT_READY','/models did not report any model ids',retryable=True)
    if model not in ids: raise AdapterError('SERVICE_NOT_READY',f'model {model!r} not listed by service; available={sorted(ids)}',retryable=False)
    return next((x for x in items if str(x.get('id'))==model),{})


def probe_completion_capabilities(*, base: str, model: str, bearer: str|None, timeout: float) -> tuple[bool,bool,bool,dict]:
    gen_obj=None; gen_err=None
    try:
        _,gen_obj=request_json(base+'/completions',method='POST',payload={"model":model,"prompt":"Hello","max_tokens":1,"temperature":0},bearer=bearer,timeout=timeout)
        gen_ok=True
    except AdapterError as exc:
        if exc.retryable:
            raise
        gen_ok=False; gen_err=str(exc)
    gen_choice=(gen_obj.get('choices') or [{}])[0] if isinstance(gen_obj,dict) else {}
    generation=bool(gen_ok and isinstance(gen_obj,dict) and isinstance(gen_obj.get('choices'),list) and gen_obj['choices'] and isinstance(gen_choice,dict) and isinstance(gen_choice.get('text'),str))
    lik_ok,lik_obj,lik_err,lik_retryable=optional_json_probe_detail(base+'/completions',method='POST',payload={"model":model,"prompt":"Hello","max_tokens":1,"temperature":0,"logprobs":1,"echo":True},bearer=bearer,timeout=timeout)
    if not lik_ok and lik_retryable:
        raise AdapterError('SERVICE_NOT_READY',f'likelihood capability probe not ready: {lik_err}',retryable=True)
    lik_choice=(lik_obj.get('choices') or [{}])[0] if isinstance(lik_obj,dict) else {}
    logprobs=bool(lik_ok and isinstance(lik_choice,dict) and lik_choice.get('logprobs') is not None)
    echo=bool(lik_ok and isinstance(lik_choice,dict) and isinstance(lik_choice.get('text'),str) and lik_choice['text'].startswith('Hello'))
    evidence={"generation":"observed" if generation else f"probe_failed:{gen_err or 'response_shape'}","completion_logprobs":"observed" if logprobs else f"probe_failed:{lik_err}" if not lik_ok else "not_observed","echo":"observed" if echo else f"probe_failed:{lik_err}" if not lik_ok else "not_observed"}
    return generation,logprobs,echo,evidence

def probe_openai_service(*, base_url: str, model: str, ownership: str, auth: dict, bearer: str|None, timeout: float, tokenizer_path: str|None=None, context_length: int|None=None, probe_root_tokenizer: bool=False, num_concurrent: int=16) -> dict:
    request_timeout=max(0.25,min(3.0,float(timeout)/8.0))
    base=normalize_http_base_url(base_url); _,models=request_json(base+'/models',bearer=bearer,timeout=request_timeout); info=_listed_model(models,model)
    protocols={"openai_models":{"url":base+'/models'}}; evidence={"models":"observed"}
    generation,logprobs,echo,completion_evidence=probe_completion_capabilities(base=base,model=model,bearer=bearer,timeout=request_timeout)
    evidence.update(completion_evidence)
    if generation or logprobs or echo: protocols['openai_completion']={"url":base+'/completions'}
    chat_ok,chat_obj,chat_err=optional_json_probe(base+'/chat/completions',method='POST',payload={"model":model,"messages":[{"role":"user","content":"Hi"}],"max_tokens":1,"temperature":0},bearer=bearer,timeout=request_timeout)
    chat_choice=(chat_obj.get('choices') or [{}])[0] if isinstance(chat_obj,dict) else {}; chat=bool(chat_ok and isinstance(chat_obj,dict) and isinstance(chat_obj.get('choices'),list) and chat_obj['choices'] and isinstance(chat_choice,dict) and isinstance(chat_choice.get('message'),dict))
    if chat: protocols['openai_chat']={"url":base+'/chat/completions'}
    evidence['chat']='observed' if chat else f'probe_failed:{chat_err}'
    tokenize=detokenize=False
    if probe_root_tokenizer:
        root=base[:-3] if base.endswith('/v1') else base; info_url=root+'/tokenizer_info'; tokenize_url=root+'/tokenize'; detokenize_url=root+'/detokenize'
        info_ok,_,info_err=optional_json_probe(info_url,method='GET',bearer=bearer,timeout=request_timeout)
        tok_ok,tok_obj,tok_err=optional_json_probe(tokenize_url,method='POST',payload={"model":model,"prompt":"hello"},bearer=bearer,timeout=request_timeout); token_ids=[]
        if tok_ok and isinstance(tok_obj,dict): token_ids=tok_obj.get('tokens') or tok_obj.get('token_ids') or []
        tokenize=bool(info_ok and tok_ok and isinstance(token_ids,list)); evidence['tokenizer_info']='observed' if info_ok else f'probe_failed:{info_err}'; evidence['tokenize']='observed' if tokenize else f'probe_failed:{tok_err}'
        if info_ok: protocols['tokenizer_info']={"url":info_url}
        if tokenize: protocols['tokenize']={"url":tokenize_url}
        if tokenize and token_ids:
            det_ok,det_obj,det_err=optional_json_probe(detokenize_url,method='POST',payload={"model":model,"tokens":token_ids[:8]},bearer=bearer,timeout=request_timeout); detokenize=bool(det_ok and isinstance(det_obj,dict)); evidence['detokenize']='observed' if detokenize else f'probe_failed:{det_err}'
            if detokenize: protocols['detokenize']={"url":detokenize_url}
    caps={"service.generation":generation,"service.chat":chat,"service.completion_logprobs":logprobs,"service.echo":echo,"service.tokenize":tokenize,"service.detokenize":detokenize}
    if context_length is None:
        for key in ('max_model_len','max_model_length','context_length'):
            if isinstance(info.get(key),(int,float)): context_length=int(info[key]); evidence['context_length']='observed_models'; break
    desc={"schema_version":"1.0","service_type":"llm","ownership":ownership,"model":{"id":model},"protocols":protocols,"capabilities":{"schema_version":"1.0","values":caps},"auth":auth or {"mode":"none"},"tokenizer":{"mode":"local","path":str(tokenizer_path)} if tokenizer_path else {"mode":"remote" if tokenize and detokenize else "none"},"limits":{"recommended_concurrency":int(num_concurrent)},"metadata":{"capability_evidence":evidence}}
    if context_length is not None: desc['context_length']=int(context_length)
    return desc
