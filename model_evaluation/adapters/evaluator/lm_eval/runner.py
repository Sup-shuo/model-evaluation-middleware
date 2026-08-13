#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path

PLACEHOLDER='__MODEL_EVAL_PROXY_COMPLETIONS__'

def _bind_python_to_current_interpreter(cmd):
    cmd=list(cmd)
    if cmd and cmd[0] in {'python','python3'}:
        cmd[0]=sys.executable
    return cmd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--completion-url',required=True); ap.add_argument('--tokenizer-info-url'); ap.add_argument('--tokenize-url'); ap.add_argument('--detokenize-url'); ap.add_argument('--auth-mode',choices=['strip','inject'],required=True); ap.add_argument('--timeout',type=float,default=600.0); ap.add_argument('command',nargs=argparse.REMAINDER); ns=ap.parse_args()
    cmd=list(ns.command); cmd=cmd[1:] if cmd and cmd[0]=='--' else cmd
    if not cmd: raise SystemExit('missing evaluator command')
    ready=Path(tempfile.gettempdir())/f'model-eval-lmproxy-{os.getpid()}.json'; ready.unlink(missing_ok=True)
    proxy=[sys.executable,str(Path(__file__).with_name('transport_proxy.py')),'--completion-url',ns.completion_url,'--auth-mode',ns.auth_mode,'--ready-file',str(ready),'--parent-pid',str(os.getpid()),'--timeout',str(ns.timeout)]
    for flag,val in (('--tokenizer-info-url',ns.tokenizer_info_url),('--tokenize-url',ns.tokenize_url),('--detokenize-url',ns.detokenize_url)):
        if val: proxy += [flag,val]
    p=subprocess.Popen(proxy,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=sys.stderr)
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and not ready.is_file():
            if p.poll() is not None: raise SystemExit(f'transport proxy exited early rc={p.returncode}')
            time.sleep(.05)
        if not ready.is_file(): raise SystemExit('transport proxy readiness timeout')
        port=int(json.loads(ready.read_text())['port']); base=f'http://127.0.0.1:{port}/v1/completions'; cmd=[x.replace(PLACEHOLDER,base) for x in cmd]
        cmd=_bind_python_to_current_interpreter(cmd)
        env=os.environ.copy(); env.pop('MODEL_EVAL_UPSTREAM_API_KEY',None); env['OPENAI_API_KEY']='local-transport-proxy'
        child=subprocess.Popen(cmd,env=env)
        return child.wait()
    finally:
        ready.unlink(missing_ok=True)
        if p.poll() is None:
            p.terminate()
            try: p.wait(timeout=1)
            except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=1)
if __name__=='__main__': raise SystemExit(main())
