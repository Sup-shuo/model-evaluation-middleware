from __future__ import annotations
import glob,re
from model_evaluation.sdk.runtime import AdapterError
def _ids():
    ids=[]
    for p in glob.glob('/dev/davinci[0-9]*'):
        m=re.search(r'(\d+)$',p)
        if m: ids.append(m.group(1))
    return sorted(set(ids),key=int)
def probe(i,c):
    avail=_ids(); req=[str(x) for x in i.get('requested_devices',[])]
    if not req and avail: req=[avail[0]]
    if not req or not set(req).issubset(set(avail)): raise AdapterError('RESOURCE_UNAVAILABLE',f'requested Ascend devices unavailable: {req or "any"}',retryable=True)
    return {"schema_version":"1.0","vendor":"huawei","device_type":"accelerator","devices":[{"id":x,"name":"Ascend NPU"} for x in req],"capabilities":{"schema_version":"1.0","values":{"device.multi_device":len(req)>1}}}
def visibility(i,c):
    ids=','.join(str(x) for x in i.get('devices',[])); return {"env_patch":{"set":{"ASCEND_RT_VISIBLE_DEVICES":ids}} if ids else {"env_patch":{}}}
def snapshot(i,c): return {"vendor":"huawei","devices":probe(i,c)['devices']}
OPERATIONS={"probe":probe,"visibility":visibility,"snapshot":snapshot}
