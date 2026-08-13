from __future__ import annotations
import os, re, shutil, subprocess
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
        if not p.is_absolute(): raise AdapterError('CONFIG_INVALID','ROCm runtime parameter root must be absolute')
        return p
    discovered=os.environ.get('ROCM_PATH')
    return Path(discovered).expanduser() if discovered else None

def _rocminfo(i):
    configured=_params(i).get('probe_tool')
    if configured:
        p=Path(str(configured)).expanduser()
        if not p.is_absolute(): raise AdapterError('CONFIG_INVALID','ROCm runtime parameter probe_tool must be absolute')
        return str(p) if p.is_file() else None
    return shutil.which('rocminfo')

def _version(root:Path):
    for f in (root/'.info/version',root/'.info/version-dev',root/'lib/.info/version'):
        if f.is_file():
            m=re.search(r'([0-9]+(?:\.[0-9]+){1,2})',f.read_text(encoding='utf-8',errors='ignore')[:1024])
            if m:return m.group(1)
    return 'unknown'

def probe(i,c):
    root=_root(i); rocminfo=_rocminfo(i)
    if _params(i).get('root') is not None and (root is None or not root.is_dir()): raise AdapterError('DEPENDENCY_MISSING',f'configured ROCm root does not exist: {root}')
    if (root is None or not root.is_dir()) and not rocminfo and not Path('/dev/kfd').exists(): raise AdapterError('DEPENDENCY_MISSING','ROCm runtime not detected')
    if rocminfo:
        try:p=subprocess.run([rocminfo],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=float(c.get('timeout_seconds',2)),check=False)
        except subprocess.TimeoutExpired as e: raise AdapterError('RESOURCE_UNAVAILABLE','rocminfo timed out',retryable=True) from e
        if p.returncode and not Path('/dev/kfd').exists(): raise AdapterError('RESOURCE_UNAVAILABLE',f'rocminfo failed: {(p.stderr or b"").decode("utf-8","ignore")[:500]}',retryable=True)
    return {"schema_version":"1.0","family":"rocm","version":_version(root) if root and root.is_dir() else 'unknown',"available":True,"capabilities":{"schema_version":"1.0","values":{"runtime.compatible_device_vendors":["amd"]}},"env_patch":{}}

def resolve_environment(i,c):
    root=_root(i); patch={}
    if root and root.is_dir():
        patch={"set":{"ROCM_PATH":str(root.resolve())}}
        paths=[str((root/'bin').resolve())] if (root/'bin').is_dir() else []
        if paths: patch['prepend_path']={'PATH':paths}
        libs=[str((root/x).resolve()) for x in ('lib','lib64') if (root/x).is_dir()]
        if libs: patch.setdefault('prepend_path',{})['LD_LIBRARY_PATH']=libs
    elif root is not None and _params(i).get('root') is not None:
        raise AdapterError('DEPENDENCY_MISSING',f'configured ROCm root does not exist: {root}')
    return {"env_patch":patch}

def snapshot(i,c): return probe(i,c)
OPERATIONS={"probe":probe,"resolve_environment":resolve_environment,"snapshot":snapshot}
