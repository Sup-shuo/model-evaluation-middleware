from __future__ import annotations
import hashlib, json
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError
INTEGRITY_POLICIES={'basic','strict'}
def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def _integrity_policy(benchmark):
    params=(benchmark.get('dataset') or {}).get('parameters') or {}
    if not isinstance(params,dict): raise AdapterError('CONFIG_INVALID','local_files dataset parameters must be an object')
    policy=params.get('integrity_policy','basic')
    if not isinstance(policy,str) or policy not in INTEGRITY_POLICIES:
        raise AdapterError('CONFIG_INVALID',f'local_files integrity_policy must be one of {sorted(INTEGRITY_POLICIES)}')
    return policy
def _build(b):
    p=(b.get('dataset') or {}).get('parameters') or {}; policy=_integrity_policy(b); root=Path(str(p.get('root') or '')).expanduser().resolve()
    if not root.is_dir(): raise AdapterError('DATASET_INVALID',f'local dataset root not found: {root}')
    rels=p.get('files')
    if not isinstance(rels,list) or not rels: raise AdapterError('CONFIG_INVALID','local_files requires non-empty dataset.parameters.files to avoid unbounded directory scans')
    files=[]; logical=[]
    for rel_raw in rels:
        rel=Path(str(rel_raw))
        if rel.is_absolute() or '..' in rel.parts: raise AdapterError('CONFIG_INVALID',f'invalid dataset relative path: {rel_raw}')
        path=(root/rel).resolve()
        if root not in path.parents and path!=root: raise AdapterError('CONFIG_INVALID',f'dataset file escapes root: {rel_raw}')
        if not path.is_file(): raise AdapterError('DATASET_INVALID',f'dataset file missing: {path}')
        if path.suffix.lower()=='.json':
            try: payload=json.loads(path.read_text(encoding='utf-8'))
            except Exception as exc: raise AdapterError('DATASET_INVALID',f'invalid JSON dataset file {path}: {exc}') from exc
            if not isinstance(payload,(dict,list)): raise AdapterError('DATASET_INVALID',f'JSON dataset root must be an object or array: {path}')
        item={"path":str(path)}; logical_name=rel.as_posix()
        if policy=='strict':
            digest=_sha(path); item['sha256']=digest; logical.append((logical_name,digest))
        else: logical.append(logical_name)
        files.append(item)
    strict=policy=='strict'; declared_revision=(b.get('dataset') or {}).get('revision')
    revision_provenance='user-declared' if declared_revision else ('content-derived' if strict else 'unversioned')
    artifact={"schema_version":"1.0","dataset_id":b['id'],"revision":declared_revision or 'unversioned',"root":str(root),"files":files,"materialization":{"kind":"concrete"},"metadata":{"logical_files":[x[0] if isinstance(x,tuple) else x for x in logical],"integrity_policy":policy,"content_fingerprinted":strict,"content_verified":False,"structure_verified":True,"revision_provenance":revision_provenance}}
    if policy=='strict':
        fp=hashlib.sha256(json.dumps({"revision":(b.get('dataset') or {}).get('revision'),"files":logical},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        artifact['fingerprint']=fp
        if not (b.get('dataset') or {}).get('revision'): artifact['revision']=fp
    return artifact
def resolve(i,c):
    b=i['benchmark']; p=(b.get('dataset') or {}).get('parameters') or {}; policy=_integrity_policy(b); return {"dataset_id":b['id'],"revision":(b.get('dataset') or {}).get('revision') or 'unversioned',"root":p.get('root'),"integrity_policy":policy}
def prepare(i,c): return _build(i['benchmark'])
def verify(i,c):
    try:
        artifact=_build(i['benchmark'])
        if _integrity_policy(i['benchmark'])=='strict' and i['artifact'].get('fingerprint')!=artifact.get('fingerprint'):
            raise AdapterError('DATASET_INVALID','local dataset content fingerprint changed since prepare')
        return {"valid":True,"artifact":artifact,"details":{"problems":[]}}
    except AdapterError as exc: return {"valid":False,"artifact":i['artifact'],"details":{"problems":[str(exc)]}}
def snapshot(i,c): return {"provider":"local_files","integrity_policy":_integrity_policy(i.get('benchmark') or {}),"artifact":i.get('artifact')}
OPERATIONS={"resolve":resolve,"prepare":prepare,"verify":verify,"snapshot":snapshot}
