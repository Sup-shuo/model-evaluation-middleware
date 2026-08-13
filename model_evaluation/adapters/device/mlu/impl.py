from __future__ import annotations
import glob, os, re, shutil, subprocess
from model_evaluation.sdk.runtime import AdapterError
def _ids_from_nodes(nodes):
    # Cambricon containers expose several device-node families.  Only
    # cambricon_devN is an addressable compute device; cambricon_ipcmN,
    # cambricon_ctl and cambricon_gdr are helper/control nodes.
    ids=set()
    for node in nodes:
        m=re.fullmatch(r'/dev/cambricon_dev(\d+)', str(node))
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=int)

def _ids(timeout=2.0):
    ids=_ids_from_nodes(glob.glob('/dev/cambricon*'))
    if ids:
        return ids
    exe=shutil.which('cnmon')
    if exe:
        try:
            p=subprocess.run([exe],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
            if p.returncode==0: raise AdapterError('RESOURCE_UNAVAILABLE','cnmon is available but stable MLU device IDs could not be derived from /dev nodes; refusing to invent device 0')
        except subprocess.TimeoutExpired: pass
    raise AdapterError('RESOURCE_UNAVAILABLE','no Cambricon MLU device detected',retryable=True)
def probe(i,c):
    avail=_ids(float(c.get('timeout_seconds',2))); req=[str(x) for x in i.get('requested_devices',[])]
    if not req and avail: req=[avail[0]]
    if not set(req).issubset(set(avail)): raise AdapterError('RESOURCE_UNAVAILABLE',f'requested MLU devices unavailable: {req}')
    return {"schema_version":"1.0","vendor":"cambricon","device_type":"accelerator","devices":[{"id":x,"name":"Cambricon MLU"} for x in req],"capabilities":{"schema_version":"1.0","values":{"device.multi_device":len(req)>1}}}
def visibility(i,c):
    ids=[str(x) for x in i.get('devices',[])]; return {"env_patch":{"set":{"MLU_VISIBLE_DEVICES":','.join(ids)}} if ids else {"env_patch":{}}}
def snapshot(i,c): return {"vendor":"cambricon","devices":probe(i,c)["devices"]}
OPERATIONS={"probe":probe,"visibility":visibility,"snapshot":snapshot}
