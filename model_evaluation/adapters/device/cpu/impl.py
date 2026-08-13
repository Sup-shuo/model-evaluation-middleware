from __future__ import annotations
import os, platform
def probe(i,c):
    return {"schema_version":"1.0","vendor":"generic","device_type":"cpu","devices":[{"id":"cpu","name":platform.processor() or "CPU"}],"capabilities":{"schema_version":"1.0","values":{"device.multi_device":False,"device.cpu_count":os.cpu_count() or 1}}}
def visibility(i,c): return {"env_patch":{}}
def snapshot(i,c): return {"platform":platform.platform(),"processor":platform.processor(),"cpu_count":os.cpu_count()}
OPERATIONS={"probe":probe,"visibility":visibility,"snapshot":snapshot}
