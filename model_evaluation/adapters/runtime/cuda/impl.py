from __future__ import annotations
import os, re, shutil, subprocess
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError

def _params(i):
    value=i.get('parameters') or {}
    if not isinstance(value,dict): raise AdapterError('CONFIG_INVALID','runtime parameters must be an object')
    return value

def _root(i):
    params=_params(i); configured=params.get('root')
    if configured is not None:
        p=Path(str(configured)).expanduser()
        if not p.is_absolute(): raise AdapterError('CONFIG_INVALID','CUDA runtime parameter root must be absolute')
        return p
    discovered=os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    return Path(discovered).expanduser() if discovered else None

def _run(argv,timeout):
    try: return subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
    except subprocess.TimeoutExpired as e: raise AdapterError('RESOURCE_UNAVAILABLE',f'runtime probe timed out: {argv[0]}',retryable=True) from e

def _tool(i,key,default):
    configured=_params(i).get(key)
    if configured:
        p=Path(str(configured)).expanduser()
        if not p.is_absolute(): raise AdapterError('CONFIG_INVALID',f'CUDA runtime parameter {key} must be an absolute executable path')
        return str(p) if p.is_file() else None
    return shutil.which(default)

def _toolkit_version(timeout,i):
    nvcc=_tool(i,'nvcc','nvcc')
    if nvcc:
        p=_run([nvcc,'--version'],timeout)
        if p.returncode==0:
            m=re.search(r'release\s+([0-9]+(?:\.[0-9]+)+)',p.stdout or '')
            if m: return m.group(1)
    root=_root(i)
    if root:
        for name in ('version.json','version.txt'):
            f=root/name
            if f.is_file():
                text=f.read_text(encoding='utf-8',errors='ignore')[:4096]
                m=re.search(r'([0-9]+(?:\.[0-9]+){1,2})',text)
                if m: return m.group(1)
    return 'unknown'

def _driver(timeout,i):
    smi=_tool(i,'driver_tool','nvidia-smi')
    if not smi: raise AdapterError('DEPENDENCY_MISSING','nvidia-smi not found; CUDA driver availability cannot be established')
    p=_run([smi,'--query-gpu=driver_version','--format=csv,noheader'],timeout)
    if p.returncode: raise AdapterError('RESOURCE_UNAVAILABLE',f'nvidia-smi failed: {(p.stderr or "").strip()}',retryable=True)
    return next((x.strip() for x in (p.stdout or '').splitlines() if x.strip()),'unknown')

def probe(i,c):
    timeout=float(c.get('timeout_seconds',2)); driver=_driver(timeout,i); version=_toolkit_version(timeout,i)
    return {"schema_version":"1.0","family":"cuda","version":version,"driver_version":driver,"available":True,"capabilities":{"schema_version":"1.0","values":{"runtime.compatible_device_vendors":["nvidia"]}},"env_patch":{}}

def resolve_environment(i,c):
    root=_root(i); patch={}
    if root and root.is_dir():
        patch={"set":{"CUDA_HOME":str(root.resolve())}}
        bin_dir=root/'bin'
        if bin_dir.is_dir(): patch['prepend_path']={'PATH':[str(bin_dir.resolve())]}
        libs=[str((root/x).resolve()) for x in ('lib64','lib') if (root/x).is_dir()]
        if libs: patch.setdefault('prepend_path',{})['LD_LIBRARY_PATH']=libs
    elif root is not None and _params(i).get('root') is not None:
        raise AdapterError('DEPENDENCY_MISSING',f'configured CUDA root does not exist: {root}')
    return {"env_patch":patch}

def snapshot(i,c): return probe(i,c)
OPERATIONS={"probe":probe,"resolve_environment":resolve_environment,"snapshot":snapshot}
