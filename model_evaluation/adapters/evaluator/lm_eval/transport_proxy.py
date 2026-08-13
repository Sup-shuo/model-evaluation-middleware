#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, threading, time, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOP={"connection","keep-alive","proxy-authenticate","proxy-authorization","te","trailers","transfer-encoding","upgrade","host","content-length"}
SECRET_ENV='MODEL_EVAL_UPSTREAM_API_KEY'
DEFAULT_MAX_BODY=64*1024*1024

class Server(ThreadingHTTPServer): daemon_threads=True
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl): return None
OPENER=urllib.request.build_opener(NoRedirect())

def alive(pid):
    try: os.kill(pid,0); return True
    except OSError: return False

def read_limited(resp,limit):
    length=getattr(resp,'headers',{}).get('Content-Length') if getattr(resp,'headers',None) else None
    if length:
        try:
            if int(length)>limit: raise ValueError('response too large')
        except (ValueError,TypeError):
            if str(length).isdigit() and int(length)>limit: raise
    data=resp.read(limit+1)
    if len(data)>limit: raise ValueError('response too large')
    return data

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--completion-url',required=True); ap.add_argument('--tokenizer-info-url'); ap.add_argument('--tokenize-url'); ap.add_argument('--detokenize-url'); ap.add_argument('--auth-mode',choices=['strip','inject'],required=True); ap.add_argument('--ready-file',required=True); ap.add_argument('--parent-pid',type=int,required=True); ap.add_argument('--timeout',type=float,default=600.0); ap.add_argument('--max-body-bytes',type=int,default=DEFAULT_MAX_BODY); ns=ap.parse_args()
    if ns.max_body_bytes <= 0: raise SystemExit('--max-body-bytes must be positive')
    key=os.environ.get(SECRET_ENV,'')
    if ns.auth_mode=='inject' and not key: raise SystemExit(f'{SECRET_ENV} required')
    routes={('POST','/v1/completions'):ns.completion_url,('GET','/tokenizer_info'):ns.tokenizer_info_url,('POST','/tokenize'):ns.tokenize_url,('POST','/detokenize'):ns.detokenize_url}
    for target in [x for x in routes.values() if x]:
        p=urllib.parse.urlsplit(target)
        if p.scheme not in {'http','https'} or not p.netloc or p.username or p.password or p.query or p.fragment: raise SystemExit(f'invalid upstream URL: {target}')
    class H(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,fmt,*args): pass
        def _send(self,status,payload,headers=None):
            self.send_response(status)
            for k,v in (headers or {}).items():
                if k.lower() not in HOP and k.lower() not in {'content-length','location'}: self.send_header(k,v)
            self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload); self.wfile.flush()
        def _go(self,method):
            route=urllib.parse.urlsplit(self.path).path.rstrip('/') or '/'; target=routes.get((method,route))
            if not target: self._send(404,b'{"error":"unsupported proxy route"}',{'Content-Type':'application/json'}); return
            try: n=int(self.headers.get('Content-Length','0') or '0')
            except ValueError: self._send(400,b'{"error":"invalid content length"}',{'Content-Type':'application/json'}); return
            if n<0 or n>ns.max_body_bytes: self._send(413,b'{"error":"request too large"}',{'Content-Type':'application/json'}); return
            body=self.rfile.read(n) if n else None
            headers={k:v for k,v in self.headers.items() if k.lower() not in HOP and k.lower()!='authorization'}
            if ns.auth_mode=='inject': headers['Authorization']=f'Bearer {key}'
            req=urllib.request.Request(target,data=body,headers=headers,method=method)
            try:
                with OPENER.open(req,timeout=ns.timeout) as resp:
                    status=resp.status; rh=resp.headers; payload=read_limited(resp,ns.max_body_bytes)
            except urllib.error.HTTPError as exc:
                if 300 <= int(exc.code) < 400:
                    payload=b'{"error":"upstream redirect refused"}'; status=502; rh={'Content-Type':'application/json'}
                else:
                    status=exc.code; rh=exc.headers
                    try: payload=read_limited(exc,ns.max_body_bytes)
                    except ValueError: payload=b'{"error":"upstream response too large"}'; status=502; rh={'Content-Type':'application/json'}
            except ValueError as exc: status=502; rh={'Content-Type':'application/json'}; payload=json.dumps({'error':str(exc)},allow_nan=False).encode()
            except Exception as exc: status=502; rh={'Content-Type':'application/json'}; payload=json.dumps({'error':str(exc)},allow_nan=False).encode()
            self._send(status,payload,rh)
        def do_GET(self): self._go('GET')
        def do_POST(self): self._go('POST')
    server=Server(('127.0.0.1',0),H)
    def watch_parent():
        while alive(ns.parent_pid): time.sleep(.25)
        server.shutdown()
    threading.Thread(target=watch_parent,daemon=True).start()
    ready=Path(ns.ready_file); ready.parent.mkdir(parents=True,exist_ok=True); tmp=ready.with_name(ready.name+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps({'port':int(server.server_address[1])},allow_nan=False)+'\n'); tmp.replace(ready)
    try: server.serve_forever(poll_interval=.2)
    finally: ready.unlink(missing_ok=True); server.server_close()
if __name__=='__main__': main()
