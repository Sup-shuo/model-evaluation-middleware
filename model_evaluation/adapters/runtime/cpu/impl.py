from __future__ import annotations
import platform
def _desc():
    return {"schema_version":"1.0","family":"cpu","version":platform.release(),"available":True,"capabilities":{"schema_version":"1.0","values":{"runtime.compatible_device_vendors":["generic"]}},"env_patch":{}}
def probe(i,c): return _desc()
def resolve_environment(i,c): return {"env_patch":{}}
def snapshot(i,c): return _desc()
OPERATIONS={"probe":probe,"resolve_environment":resolve_environment,"snapshot":snapshot}
