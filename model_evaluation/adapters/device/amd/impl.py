from __future__ import annotations
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError
def _devices():
    out=[]
    for card in sorted(Path('/sys/class/drm').glob('card[0-9]*')):
        vendor=card/'device/vendor'
        try: v=vendor.read_text().strip().lower()
        except OSError: continue
        if v!='0x1002': continue
        idx=card.name.removeprefix('card'); item={"id":idx,"name":"AMD GPU"}; mem=card/'device/mem_info_vram_total'
        try: item['memory_bytes']=int(mem.read_text().strip())
        except (OSError,ValueError): pass
        out.append(item)
    return out
def probe(i,c):
    ds=_devices(); req=[str(x) for x in i.get('requested_devices',[])]
    if not req and ds: req=[str(ds[0]['id'])]
    if req: ds=[d for d in ds if d['id'] in req]
    if not ds or (req and {d['id'] for d in ds}!=set(req)): raise AdapterError('RESOURCE_UNAVAILABLE',f'requested AMD devices unavailable: {req or "any"}',retryable=True)
    return {"schema_version":"1.0","vendor":"amd","device_type":"accelerator","devices":ds,"capabilities":{"schema_version":"1.0","values":{"device.multi_device":len(ds)>1}}}
def visibility(i,c):
    ids=','.join(str(x) for x in i.get('devices',[])); return {"env_patch":{"set":{"ROCR_VISIBLE_DEVICES":ids,"HIP_VISIBLE_DEVICES":ids}} if ids else {"env_patch":{}}}
def snapshot(i,c): return {"vendor":"amd","devices":probe(i,c)['devices']}
OPERATIONS={"probe":probe,"visibility":visibility,"snapshot":snapshot}
