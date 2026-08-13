from __future__ import annotations
import csv, hashlib, time, urllib.error, urllib.request
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError
from model_evaluation.sdk.jsonutil import loads as json_loads

SOURCE_COMMIT="9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BASE_URL=f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{SOURCE_COMMIT}/bbh"
EXPECTED=Path(__file__).with_name('expected.tsv')
MAX_FILE_BYTES=16*1024*1024
INTEGRITY_POLICIES={'basic','strict'}
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl): return None
_OPENER=urllib.request.build_opener(_NoRedirect())

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _expected():
    rows=[]
    with EXPECTED.open('r',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f,delimiter='\t')
        if set(r.fieldnames or []) != {'task','samples','sha256'}: raise AdapterError('ADAPTER_INTERNAL_ERROR','invalid bundled BBH expected manifest')
        for row in r: rows.append((row['task'],int(row['samples']),row['sha256']))
    if not rows: raise AdapterError('ADAPTER_INTERNAL_ERROR','empty bundled BBH expected manifest')
    return rows

def _integrity_policy(benchmark: dict) -> str:
    params=(benchmark.get('dataset') or {}).get('parameters') or {}
    if not isinstance(params,dict): raise AdapterError('CONFIG_INVALID','BBH dataset parameters must be an object')
    policy=params.get('integrity_policy','basic')
    if not isinstance(policy,str) or policy not in INTEGRITY_POLICIES:
        raise AdapterError('CONFIG_INVALID',f'BBH integrity_policy must be one of {sorted(INTEGRITY_POLICIES)}')
    return policy

def _validate_file(path: Path, samples: int, digest: str, *, integrity_policy: str) -> None:
    if path.is_symlink() or not path.is_file(): raise AdapterError('DATASET_INVALID',f'missing/unsafe BBH file: {path}')
    try: size=path.stat().st_size
    except OSError as exc: raise AdapterError('DATASET_INVALID',f'cannot inspect BBH file {path}: {exc}') from exc
    if size>MAX_FILE_BYTES: raise AdapterError('DATASET_INVALID',f'BBH file exceeds {MAX_FILE_BYTES} bytes: {path}')
    try: payload=json_loads(path.read_text(encoding='utf-8'))
    except Exception as exc: raise AdapterError('DATASET_INVALID',f'invalid BBH JSON {path}: {exc}') from exc
    if not isinstance(payload,dict): raise AdapterError('DATASET_INVALID',f'BBH JSON root must be an object: {path}')
    examples=payload.get('examples')
    if not isinstance(examples,list) or not examples: raise AdapterError('DATASET_INVALID',f'BBH examples must be a non-empty list: {path}')
    if integrity_policy=='strict' and len(examples)!=samples: raise AdapterError('DATASET_INVALID',f'BBH sample count mismatch: {path}')
    for idx,item in enumerate(examples):
        if not isinstance(item,dict) or not isinstance(item.get('input'),str) or not item.get('input') or not isinstance(item.get('target'),str) or not item.get('target'):
            raise AdapterError('DATASET_INVALID',f'invalid BBH example {idx}: {path}')
    if integrity_policy=='strict':
        actual=_sha(path)
        if actual!=digest: raise AdapterError('DATASET_INVALID',f'BBH SHA256 mismatch: {path}')

def _artifact(root: Path, integrity_policy: str) -> dict:
    files=[]; total=0
    for task,samples,digest in _expected():
        path=root/f'{task}.json'; _validate_file(path,samples,digest,integrity_policy=integrity_policy)
        item={'path':str(path.resolve())}
        if integrity_policy=='strict': item['sha256']=digest
        files.append(item); total+=samples
    strict=integrity_policy=='strict'
    artifact={"schema_version":"1.0","dataset_id":"bbh","revision":SOURCE_COMMIT,"root":str(root.resolve()),"files":files,"materialization":{"kind":"concrete"},"metadata":{"source":"BIG-Bench-Hard","source_commit":SOURCE_COMMIT,"task_count":len(files),"integrity_policy":integrity_policy,"content_fingerprinted":strict,"content_verified":strict,"structure_verified":True,"revision_provenance":"content-verified" if strict else "provider-declared"}}
    if integrity_policy=='strict':
        artifact['metadata']['total_samples']=total
        artifact['fingerprint']=hashlib.sha256((SOURCE_COMMIT+'\n').encode()+EXPECTED.read_bytes()).hexdigest()
        artifact['metadata']['expected_manifest_sha256']=_sha(EXPECTED)
    return artifact

def resolve(i,c):
    benchmark=i.get('benchmark') or {}; requested=benchmark.get('dataset',{}).get('revision'); policy=_integrity_policy(benchmark)
    if requested and requested!=SOURCE_COMMIT: raise AdapterError('CONFIG_INVALID',f'BBH revision must match pinned provider revision {SOURCE_COMMIT}')
    out={"dataset_id":"bbh","revision":SOURCE_COMMIT,"source":"BIG-Bench-Hard","integrity_policy":policy}
    if policy=='strict': out['expected_manifest_sha256']=_sha(EXPECTED)
    return out

def _download(task: str, path: Path, samples: int, digest: str, *, integrity_policy: str, timeout: float, retries: int) -> None:
    url=f'{BASE_URL}/{task}.json'; last=None
    for attempt in range(retries):
        tmp=path.with_suffix('.json.part'); tmp.unlink(missing_ok=True)
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'model-evaluation-middleware/4.1'})
            with _OPENER.open(req,timeout=timeout) as r, tmp.open('wb') as f:
                total=0
                while True:
                    chunk=r.read(min(1024*1024,MAX_FILE_BYTES-total+1))
                    if not chunk: break
                    total+=len(chunk)
                    if total>MAX_FILE_BYTES: raise AdapterError('DATASET_INVALID',f'BBH file exceeds {MAX_FILE_BYTES} bytes: {task}')
                    f.write(chunk)
            _validate_file(tmp,samples,digest,integrity_policy=integrity_policy); tmp.replace(path); return
        except (urllib.error.URLError,urllib.error.HTTPError,TimeoutError,OSError,AdapterError) as exc:
            last=exc; tmp.unlink(missing_ok=True)
            if attempt+1<retries: time.sleep(min(2.0,0.25*(attempt+1)))
    if isinstance(last,AdapterError): raise last
    raise AdapterError('RESOURCE_UNAVAILABLE',f'failed to download {url}: {last}',retryable=True)

def prepare(i,c):
    b=i['benchmark']; resolve({'benchmark':b},c); policy=_integrity_policy(b); root=Path(i.get('cache_root') or c.get('cache_root') or '.').resolve()/'bbh'/SOURCE_COMMIT; root.mkdir(parents=True,exist_ok=True)
    offline=bool(c.get('offline',False)); params=(b.get('dataset') or {}).get('parameters') or {}; timeout=float(params.get('download_timeout_seconds',30)); retries=int(params.get('download_retries',3))
    if timeout<=0 or retries<1: raise AdapterError('CONFIG_INVALID','BBH download_timeout_seconds must be positive and download_retries must be >= 1')
    # Cache mutation serialization is owned by Core ResourceManager.  This
    # adapter keeps only atomic per-file download/replace semantics.
    for task,samples,digest in _expected():
        path=root/f'{task}.json'
        try: _validate_file(path,samples,digest,integrity_policy=policy); continue
        except AdapterError:
            if offline: raise AdapterError('DATASET_INVALID',f'BBH cache missing/stale in offline mode: {path}')
        _download(task,path,samples,digest,integrity_policy=policy,timeout=timeout,retries=retries); _validate_file(path,samples,digest,integrity_policy=policy)
    return _artifact(root,policy)

def verify(i,c):
    try:
        policy=_integrity_policy(i['benchmark']); artifact=_artifact(Path(i['artifact']['root']),policy)
        return {"valid":True,"artifact":artifact,"details":{"problems":[]}}
    except AdapterError as exc: return {"valid":False,"artifact":i['artifact'],"details":{"problems":[str(exc)]}}
def snapshot(i,c):
    policy=_integrity_policy(i.get('benchmark') or {})
    out={"provider":"bbh_local","source_commit":SOURCE_COMMIT,"integrity_policy":policy,"artifact":i.get('artifact')}
    if policy=='strict': out['expected_manifest_sha256']=_sha(EXPECTED)
    return out
OPERATIONS={"resolve":resolve,"prepare":prepare,"verify":verify,"snapshot":snapshot}
