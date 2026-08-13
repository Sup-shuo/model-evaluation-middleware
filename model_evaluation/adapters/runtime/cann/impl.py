from __future__ import annotations
import os, re
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError

def _params(i):
    value=i.get('parameters') or {}
    if not isinstance(value,dict): raise AdapterError('CONFIG_INVALID','runtime parameters must be an object')
    return value

def _root(i):
    configured=_params(i).get('root')
    if configured is not None:
        p=Path(str(configured)).expanduser()
        if not p.is_absolute(): raise AdapterError('CONFIG_INVALID','CANN runtime parameter root must be absolute')
        return p
    env=os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
    return Path(env).expanduser() if env else None

def _version(root:Path):
    for f in (root/'version.cfg',root/'version.info',root/'VERSION'):
        if f.is_file():
            m=re.search(r'([0-9]+(?:\.[0-9]+){1,2})',f.read_text(encoding='utf-8',errors='ignore')[:4096])
            if m:return m.group(1)
    return 'unknown'

def probe(i,c):
    root=_root(i)
    if root is None or not root.is_dir():
        if _params(i).get('root') is not None: raise AdapterError('DEPENDENCY_MISSING',f'configured CANN root does not exist: {_params(i).get("root")}')
        raise AdapterError('DEPENDENCY_MISSING','CANN runtime not detected')
    if not any((root/p).exists() for p in ('lib64','lib64/stub','compiler','runtime')): raise AdapterError('DEPENDENCY_MISSING',f'CANN root is incomplete: {root}')
    return {"schema_version":"1.0","family":"cann","version":_version(root),"available":True,"capabilities":{"schema_version":"1.0","values":{"runtime.compatible_device_vendors":["huawei"]}},"env_patch":{}}

def resolve_environment(i,c):
    root=_root(i); patch={}
    if root and root.is_dir():
        patch={"set":{"ASCEND_HOME_PATH":str(root.resolve())}}
        libs=[str((root/p).resolve()) for p in ('lib64','lib64/plugin/opskernel') if (root/p).is_dir()]
        if libs: patch['prepend_path']={'LD_LIBRARY_PATH':libs}
    elif root is not None and _params(i).get('root') is not None:
        raise AdapterError('DEPENDENCY_MISSING',f'configured CANN root does not exist: {_params(i).get("root")}')
    return {"env_patch":patch}

def snapshot(i,c): return probe(i,c)
OPERATIONS={"probe":probe,"resolve_environment":resolve_environment,"snapshot":snapshot}
