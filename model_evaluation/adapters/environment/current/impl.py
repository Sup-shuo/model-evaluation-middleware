from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

def _controller_python(context):
    raw=str((context or {}).get("controller_python") or sys.executable)
    return str(Path(raw).resolve())

def resolve(i,c):
    python=_controller_python(c)
    return {"schema_version":"1.0","provider":"current","identity":i.get('profile') or 'current',"python":python,"executable_root":str(Path(python).parent),"capabilities":{"schema_version":"1.0","values":{"environment.python":True}}}

def wrap(i,c):
    process=dict(i["process"]); argv=list(process.get("argv") or [])
    resolved=str((i.get("environment") or {}).get("python") or _controller_python(c))
    if argv and argv[0] in {"python","python3"}: argv[0]=resolved
    process["argv"]=argv
    return {"process":process}

def snapshot(i,c):
    python=_controller_python(c)
    out={"provider":"current","python":python,"path":os.environ.get('PATH','')}
    try:
        p=subprocess.run(
            [python,'-c','import json,platform;print(json.dumps({"python_implementation":platform.python_implementation(),"python_version":platform.python_version()}))'],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            timeout=float((c or {}).get('timeout_seconds',3)),check=False,
        )
        if p.returncode==0:
            out.update(json.loads((p.stdout or '').strip().splitlines()[-1]))
    except Exception:
        pass
    return out

OPERATIONS={"resolve":resolve,"wrap_process":wrap,"snapshot":snapshot}
