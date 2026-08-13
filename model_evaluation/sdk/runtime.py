from __future__ import annotations
import sys
from collections.abc import Callable
from model_evaluation.sdk.jsonutil import dumps as json_dumps, loads as json_loads, reject_non_finite

API_VERSION = '1.0'
ERROR_CODES = {
    'CONFIG_INVALID', 'CONFIG_CONFLICT', 'COMPATIBILITY_ERROR', 'RESOURCE_UNAVAILABLE',
    'DEPENDENCY_MISSING', 'SERVICE_START_FAILED', 'SERVICE_NOT_READY', 'DATASET_INVALID',
    'EVALUATION_FAILED', 'RESULT_INVALID', 'ADAPTER_INTERNAL_ERROR', 'ADAPTER_PROTOCOL_ERROR',
}

class AdapterError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, details: dict | None = None):
        super().__init__(message); self.code=code if code in ERROR_CODES else 'ADAPTER_INTERNAL_ERROR'; self.retryable=retryable; self.details=details or {}

def _write(obj: dict) -> None:
    sys.stdout.write(json_dumps(obj,ensure_ascii=False,sort_keys=True)+'\n'); sys.stdout.flush()

def run_adapter(manifest: dict, operations: dict[str, Callable[[dict, dict], dict]]) -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {'manifest','invoke'}:
        print('usage: adapter {manifest|invoke}',file=sys.stderr); raise SystemExit(2)
    if sys.argv[1]=='manifest': _write(manifest); return
    request_id='unknown'
    try:
        request=json_loads(sys.stdin.read())
        if not isinstance(request,dict): raise AdapterError('ADAPTER_PROTOCOL_ERROR','request must be a JSON object')
        request_id=str(request.get('request_id','unknown')); api=str(request.get('api_version',''))
        if api != API_VERSION: raise AdapterError('ADAPTER_PROTOCOL_ERROR',f'unsupported api version: {api}; expected exactly {API_VERSION}')
        op=request.get('operation')
        if op not in operations: raise AdapterError('ADAPTER_PROTOCOL_ERROR',f'unknown operation: {op}')
        input_obj=request.get('input')
        if not isinstance(input_obj,dict): raise AdapterError('CONFIG_INVALID','input must be an object')
        context=request.get('context') or {}
        if not isinstance(context,dict): raise AdapterError('CONFIG_INVALID','context must be an object')
        output=operations[op](input_obj,context)
        if not isinstance(output,dict): raise AdapterError('ADAPTER_INTERNAL_ERROR','operation returned non-object output')
        try: reject_non_finite(output)
        except ValueError as exc: raise AdapterError('RESULT_INVALID' if op=='normalize' else 'ADAPTER_PROTOCOL_ERROR',str(exc)) from exc
        _write({'api_version':API_VERSION,'request_id':request_id,'ok':True,'output':output,'warnings':[]})
    except AdapterError as exc:
        _write({'api_version':API_VERSION,'request_id':request_id,'ok':False,'error':{'code':exc.code,'message':str(exc),'retryable':exc.retryable,'details':exc.details},'warnings':[]}); raise SystemExit(1)
    except Exception as exc:
        print(f'adapter internal error: {exc}',file=sys.stderr)
        _write({'api_version':API_VERSION,'request_id':request_id,'ok':False,'error':{'code':'ADAPTER_INTERNAL_ERROR','message':str(exc),'retryable':False,'details':{}},'warnings':[]}); raise SystemExit(1)
