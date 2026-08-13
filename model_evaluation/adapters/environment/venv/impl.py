from __future__ import annotations

import json, os, subprocess
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError


def _root(input_obj: dict) -> Path:
    raw=str(input_obj.get('profile') or ((input_obj.get('environment') or {}).get('identity')) or '').strip()
    if not raw:
        raise AdapterError('CONFIG_INVALID','venv profile must be an absolute virtual-environment path')
    root=Path(raw).expanduser()
    if not root.is_absolute():
        raise AdapterError('CONFIG_INVALID',f'venv profile must be an absolute path: {root}')
    root=root.resolve()
    bindir=root/('Scripts' if os.name=='nt' else 'bin')
    py=bindir/('python.exe' if os.name=='nt' else 'python')
    if not root.is_dir() or not py.is_file():
        raise AdapterError('DEPENDENCY_MISSING',f'python virtual environment unavailable: {root}')
    return root


def _assert_absolute_executable_belongs_to_environment(process: dict, bindir: Path) -> None:
    argv=list(process.get('argv') or [])
    if not argv:
        return
    exe=Path(str(argv[0])).expanduser()
    if not exe.is_absolute():
        return
    lexical_exe=exe.absolute()
    lexical_root=bindir.absolute()
    if lexical_exe != lexical_root and lexical_root not in lexical_exe.parents:
        raise AdapterError(
            'CONFIG_CONFLICT',
            f'absolute executable {exe} is outside selected venv executable_root {bindir}; '
            'use an unqualified command name to resolve it inside the selected environment, or point it into that environment',
        )


def resolve(i,c):
    root=_root(i)
    bindir=root/('Scripts' if os.name=='nt' else 'bin')
    py=bindir/('python.exe' if os.name=='nt' else 'python')
    return {
        "schema_version":"1.0","provider":"venv","identity":str(root),"python":str(py),
        "executable_root":str(bindir),
        "capabilities":{"schema_version":"1.0","values":{"environment.python":True}},
        "metadata":{},
    }


def wrap(i,c):
    process=dict(i['process'])
    env=dict(process.get('env_patch') or {})
    resolved=i.get('environment') or {}
    root=_root({'environment':resolved})
    bindir=root/('Scripts' if os.name=='nt' else 'bin')
    _assert_absolute_executable_belongs_to_environment(process,bindir)
    set_values=dict(env.get('set') or {})
    set_values['VIRTUAL_ENV']=str(root)
    prepend=dict(env.get('prepend_path') or {})
    existing=list(prepend.get('PATH') or [])
    prepend['PATH']=[str(bindir),*existing]
    env['set']=set_values
    env['prepend_path']=prepend
    process['env_patch']=env
    argv=list(process.get('argv') or [])
    if argv and argv[0] in {'python','python3'}:
        argv[0]=str(bindir/('python.exe' if os.name=='nt' else 'python'))
        process['argv']=argv
    return {"process":process}


def snapshot(i,c):
    d=resolve(i,c)
    out={"provider":"venv","identity":d['identity'],"python":d['python'],"executable_root":d['executable_root']}
    try:
        p=subprocess.run([d['python'],'-c','import json,platform;print(json.dumps({"python_implementation":platform.python_implementation(),"python_version":platform.python_version()}))'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=float(c.get('timeout_seconds',3)),check=False)
        if p.returncode==0: out.update(json.loads((p.stdout or '').strip().splitlines()[-1]))
    except Exception: pass
    return out


OPERATIONS={"resolve":resolve,"wrap_process":wrap,"snapshot":snapshot}
