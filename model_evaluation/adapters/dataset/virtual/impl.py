from __future__ import annotations
import hashlib, json
from pathlib import Path
def _fingerprint(benchmark): return hashlib.sha256(json.dumps(benchmark.get('dataset') or {},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def resolve(i,c):
    b=i['benchmark']; return {"dataset_id":b['id'],"revision":(b.get('dataset') or {}).get('revision') or 'framework-native','mode':'virtual'}
def prepare(i,c):
    b=i['benchmark']; root=Path(i.get('cache_root') or c.get('cache_root') or '.').resolve()/"virtual"/b['id']; root.mkdir(parents=True,exist_ok=True)
    return {"schema_version":"1.0","dataset_id":b['id'],"revision":(b.get('dataset') or {}).get('revision') or 'framework-native',"root":str(root),"files":[],"fingerprint":_fingerprint(b),"materialization":{"kind":"virtual"},"metadata":{"warning":"dataset bytes are managed outside this provider; use a concrete provider for strict dataset provenance"}}
def verify(i,c): return {"valid":True,"artifact":i['artifact']}
def snapshot(i,c): return {"provider":"virtual","artifact":i.get('artifact')}
OPERATIONS={"resolve":resolve,"prepare":prepare,"verify":verify,"snapshot":snapshot}
