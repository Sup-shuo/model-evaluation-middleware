from __future__ import annotations
import csv, io, shutil, subprocess
from model_evaluation.sdk.runtime import AdapterError
def _rows(timeout=2.0):
    exe=shutil.which("nvidia-smi")
    if not exe: raise AdapterError("DEPENDENCY_MISSING","nvidia-smi not found")
    try: p=subprocess.run([exe,"--query-gpu=index,name,memory.total,uuid","--format=csv,noheader,nounits"],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
    except subprocess.TimeoutExpired as e: raise AdapterError("RESOURCE_UNAVAILABLE","nvidia-smi timed out",retryable=True) from e
    if p.returncode: raise AdapterError("RESOURCE_UNAVAILABLE",f"nvidia-smi failed: {p.stderr.strip()}",retryable=True)
    return [r for r in csv.reader(io.StringIO(p.stdout)) if r]
def probe(i,c):
    requested=[str(x) for x in i.get("requested_devices",[])]; rows=_rows(float(c.get("timeout_seconds",2)))
    if not requested and rows: requested=[str(rows[0][0]).strip()]
    devices=[]
    for r in rows:
        idx,name,mem,uuid=(x.strip() for x in r[:4])
        if requested and idx not in requested: continue
        devices.append({"id":idx,"name":name,"memory_bytes":int(float(mem)*1024*1024),"uuid":uuid})
    if requested and {d["id"] for d in devices} != set(requested): raise AdapterError("RESOURCE_UNAVAILABLE",f"requested NVIDIA devices unavailable: {requested}")
    return {"schema_version":"1.0","vendor":"nvidia","device_type":"accelerator","devices":devices,"capabilities":{"schema_version":"1.0","values":{"device.multi_device":len(devices)>1}}}
def visibility(i,c):
    ids=[str(x) for x in i.get("devices",[])]; return {"env_patch":{"set":{"CUDA_VISIBLE_DEVICES":",".join(ids)}} if ids else {"env_patch":{}}}
def snapshot(i,c): return {"vendor":"nvidia","devices":probe(i,c)["devices"]}
OPERATIONS={"probe":probe,"visibility":visibility,"snapshot":snapshot}
