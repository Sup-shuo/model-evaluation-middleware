from __future__ import annotations
import os, re, shutil
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
        if not p.is_absolute(): raise AdapterError('CONFIG_INVALID','Neuware runtime parameter root must be absolute')
        return p
    discovered=os.environ.get('NEUWARE_HOME')
    return Path(discovered).expanduser() if discovered else None

def _version(root:Path):
    for f in (root/'version.txt',root/'VERSION',root/'include/cnrt.h'):
        if f.is_file():
            m=re.search(r'([0-9]+(?:\.[0-9]+){1,2})',f.read_text(encoding='utf-8',errors='ignore')[:65536])
            if m:return m.group(1)
    return 'unknown'

def probe(i,c):
    root=_root(i); cnmon=shutil.which('cnmon')
    if _params(i).get('root') is not None and (root is None or not root.is_dir()): raise AdapterError('DEPENDENCY_MISSING',f'configured Neuware root does not exist: {root}')
    if (root is None or not root.is_dir()) and not cnmon: raise AdapterError('DEPENDENCY_MISSING','Neuware runtime not detected')
    if root and root.is_dir() and not any((root/p).exists() for p in ('lib64','lib','include')): raise AdapterError('DEPENDENCY_MISSING',f'Neuware root is incomplete: {root}')
    return {"schema_version":"1.0","family":"neuware","version":_version(root) if root and root.is_dir() else 'unknown',"available":True,"capabilities":{"schema_version":"1.0","values":{"runtime.compatible_device_vendors":["cambricon"]}},"env_patch":{}}

def resolve_environment(i,c):
    root=_root(i); patch={}
    if root and root.is_dir():
        patch={"set":{"NEUWARE_HOME":str(root.resolve())}}
        libs=[str((root/p).resolve()) for p in ('lib64','lib') if (root/p).is_dir()]
        if libs: patch['prepend_path']={'LD_LIBRARY_PATH':libs}
    elif root is not None and _params(i).get('root') is not None:
        raise AdapterError('DEPENDENCY_MISSING',f'configured Neuware root does not exist: {root}')
    return {"env_patch":patch}

def snapshot(i,c): return probe(i,c)
OPERATIONS={"probe":probe,"resolve_environment":resolve_environment,"snapshot":snapshot}
