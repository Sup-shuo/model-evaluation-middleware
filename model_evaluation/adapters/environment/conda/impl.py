from __future__ import annotations
import json, os, shutil, subprocess
from pathlib import Path
from model_evaluation.sdk.runtime import AdapterError


def _conda(input_obj=None):
    params=(input_obj or {}).get('parameters') or {}
    candidate=params.get('executable')
    if candidate:
        p=Path(str(candidate)).expanduser()
        if not p.is_absolute() or not p.is_file() or not os.access(p,os.X_OK):
            raise AdapterError('DEPENDENCY_MISSING',f'configured conda executable unavailable: {p}')
        return str(p.resolve())
    exe=shutil.which('conda')
    if not exe:
        raise AdapterError('DEPENDENCY_MISSING','conda executable not found')
    return exe


def _selector(profile: str) -> tuple[str,str]:
    raw=str(profile or '').strip()
    if not raw:
        raise AdapterError('CONFIG_INVALID','conda profile/environment name or absolute prefix is required')
    p=Path(raw).expanduser()
    if p.is_absolute():
        prefix=p.resolve()
        if not prefix.is_dir():
            raise AdapterError('DEPENDENCY_MISSING',f'conda environment prefix unavailable: {prefix}')
        return '-p',str(prefix)
    return '-n',raw


def _assert_absolute_executable_belongs_to_environment(process: dict, environment: dict) -> None:
    argv=list(process.get('argv') or [])
    if not argv:
        return
    exe=Path(str(argv[0])).expanduser()
    if not exe.is_absolute():
        return
    root_value=environment.get('executable_root')
    if not root_value:
        return
    root=Path(str(root_value)).expanduser()
    try:
        lexical_exe=exe.absolute()
        lexical_root=root.absolute()
    except OSError:
        return
    if lexical_exe != lexical_root and lexical_root not in lexical_exe.parents:
        raise AdapterError(
            'CONFIG_CONFLICT',
            f'absolute executable {exe} is outside selected conda environment executable_root {root}; '
            'use an unqualified command name to resolve it inside the selected environment, or point it into that environment',
        )


def resolve(i,c):
    flag,profile=_selector(str(i.get('profile') or ''))
    exe=_conda(i)
    if flag == '-p':
        prefix=Path(profile)
        bindir=prefix/('Scripts' if os.name=='nt' else 'bin')
        py=bindir/('python.exe' if os.name=='nt' else 'python')
        if not py.is_file() or not os.access(py,os.X_OK):
            raise AdapterError('DEPENDENCY_MISSING',f'conda environment Python unavailable: {py}')
        return {
            "schema_version":"1.0","provider":"conda","identity":profile,"python":str(py),
            "executable_root":str(bindir),
            "capabilities":{"schema_version":"1.0","values":{"environment.python":True}},
            "metadata":{"selection_mode":"prefix","probe_mode":"direct_prefix"},
        }
    try:
        p=subprocess.run(
            [exe,'run',flag,profile,'python','-c','import sys;print(sys.executable)'],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            timeout=float(c.get('timeout_seconds',3)),check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise AdapterError('RESOURCE_UNAVAILABLE',f'conda environment probe timed out: {profile}',retryable=True) from e
    if p.returncode:
        raise AdapterError('DEPENDENCY_MISSING',f'conda environment unavailable: {profile}: {p.stderr.strip()}')
    lines=(p.stdout or '').strip().splitlines()
    if not lines:
        raise AdapterError('DEPENDENCY_MISSING',f'conda environment did not report a Python executable: {profile}')
    py=lines[-1]
    return {
        "schema_version":"1.0","provider":"conda","identity":profile,"python":py,
        "executable_root":str(Path(py).expanduser().absolute().parent),
        "capabilities":{"schema_version":"1.0","values":{"environment.python":True}},
        "metadata":{"selection_mode":"name","probe_mode":"conda_run"},
    }


def wrap(i,c):
    process=dict(i['process'])
    environment=i.get('environment') or {}
    identity=str(environment.get('identity') or i.get('profile') or '')
    flag,profile=_selector(identity)
    _assert_absolute_executable_belongs_to_environment(process,environment)
    process['argv']=[_conda(i),'run','--no-capture-output',flag,profile,*process['argv']]
    return {"process":process}


def snapshot(i,c):
    d=resolve(i,c)
    out={
        "provider":"conda","identity":d['identity'],"python":d.get('python'),
        "executable_root":d.get('executable_root'),"selection_mode":(d.get('metadata') or {}).get('selection_mode'),
        "conda_executable":_conda(i),
    }
    try:
        flag,profile=_selector(d['identity'])
        p=subprocess.run([_conda(i),'run','--no-capture-output',flag,profile,'python','-c','import json,platform;print(json.dumps({"python_implementation":platform.python_implementation(),"python_version":platform.python_version()}))'],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=float(c.get('timeout_seconds',5)),check=False)
        if p.returncode==0: out.update(json.loads((p.stdout or '').strip().splitlines()[-1]))
    except Exception: pass
    return out


OPERATIONS={"resolve":resolve,"wrap_process":wrap,"snapshot":snapshot}
