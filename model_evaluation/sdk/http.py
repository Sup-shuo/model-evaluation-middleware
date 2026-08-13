from __future__ import annotations
from urllib.parse import urlparse
import urllib.error
import urllib.request
from model_evaluation.sdk.runtime import AdapterError
from model_evaluation.sdk.jsonutil import dumps as json_dumps, loads as json_loads

_DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

_OPENER = urllib.request.build_opener(_NoRedirect())


def normalize_http_base_url(value: str) -> str:
    base=str(value or '').strip().rstrip('/')
    parsed=urlparse(base)
    if parsed.scheme not in {'http','https'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AdapterError('CONFIG_INVALID','endpoint base URL must be a credential-free http(s) URL without query/fragment')
    return base


def _read_limited(resp, max_bytes: int) -> bytes:
    length=resp.headers.get('Content-Length') if getattr(resp,'headers',None) else None
    if length:
        try:
            if int(length) > max_bytes: raise AdapterError('SERVICE_NOT_READY',f'HTTP response exceeds {max_bytes} bytes')
        except ValueError:
            pass
    raw=resp.read(max_bytes+1)
    if len(raw)>max_bytes: raise AdapterError('SERVICE_NOT_READY',f'HTTP response exceeds {max_bytes} bytes')
    return raw


def request_json(url: str, *, method: str = 'GET', payload: dict | None = None, bearer: str | None = None, timeout: float = 3.0, max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES) -> tuple[int, object]:
    data = None if payload is None else json_dumps(payload,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    headers = {'Accept': 'application/json'}
    if data is not None: headers['Content-Type'] = 'application/json'
    if bearer: headers['Authorization'] = f'Bearer {bearer}'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = _read_limited(resp,max_response_bytes); status = int(resp.status)
    except urllib.error.HTTPError as exc:
        try: raw=_read_limited(exc,max_response_bytes)
        except AdapterError: raw=b'<response too large>'
        if 300 <= int(exc.code) < 400:
            location=(exc.headers or {}).get('Location') if getattr(exc,'headers',None) else None
            raise AdapterError('SERVICE_NOT_READY', f'HTTP redirect refused for {url}: status={exc.code} location={location!r}', retryable=False) from exc
        raise AdapterError('SERVICE_NOT_READY', f'HTTP {exc.code} from {url}: {raw[:500]!r}', retryable=500 <= exc.code < 600) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AdapterError('SERVICE_NOT_READY', f'request failed for {url}: {exc}', retryable=True) from exc
    if not raw: return status, {}
    try: return status, json_loads(raw.decode('utf-8'))
    except Exception as exc: raise AdapterError('SERVICE_NOT_READY', f'non-JSON/invalid JSON response from {url}') from exc


def optional_json_probe_detail(url: str, *, method: str='GET', payload: dict | None=None, bearer: str | None=None, timeout: float=3.0) -> tuple[bool, object | None, str | None, bool]:
    try:
        _, obj=request_json(url,method=method,payload=payload,bearer=bearer,timeout=timeout)
        return True,obj,None,False
    except AdapterError as exc:
        return False,None,str(exc),bool(exc.retryable)

def optional_json_probe(url: str, *, method: str='GET', payload: dict | None=None, bearer: str | None=None, timeout: float=3.0) -> tuple[bool, object | None, str | None]:
    ok,obj,err,_=optional_json_probe_detail(url,method=method,payload=payload,bearer=bearer,timeout=timeout)
    return ok,obj,err
