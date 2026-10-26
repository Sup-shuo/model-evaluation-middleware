from __future__ import annotations
import os, platform, re
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
        root=root.resolve()
        # Managed processes inherit only generic environment variables. Rebuild
        # CANN's runtime hints from the selected installation, not the caller's
        # shell or another CANN installation.
        patch={"set":{
            "ASCEND_HOME_PATH":str(root),
            "ASCEND_AICPU_PATH":str(root),
            "ASCEND_TOOLKIT_HOME":str(root),
        }}
        for name, relative in (("ASCEND_OPP_PATH","opp"),("TOOLCHAIN_HOME","toolkit")):
            if (root/relative).is_dir():
                patch["set"][name]=str((root/relative).resolve())
        paths={
            "LD_LIBRARY_PATH":(
                "lib64", "lib64/plugin/opskernel", "lib64/plugin/nnengine",
                f"opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/{platform.machine()}",
            ),
            "PYTHONPATH":("python/site-packages","opp/built-in/op_impl/ai_core/tbe"),
            "PATH":("bin","tools/ccec_compiler/bin"),
        }
        for name, relatives in paths.items():
            values=list(dict.fromkeys(str((root/p).resolve()) for p in relatives if (root/p).is_dir()))
            if values:
                patch.setdefault("prepend_path",{})[name]=values
    elif root is not None and _params(i).get('root') is not None:
        raise AdapterError('DEPENDENCY_MISSING',f'configured CANN root does not exist: {_params(i).get("root")}')
    return {"env_patch":patch}

def snapshot(i,c): return probe(i,c)
OPERATIONS={"probe":probe,"resolve_environment":resolve_environment,"snapshot":snapshot}
