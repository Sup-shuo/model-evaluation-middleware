#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from .usage import UsageAccumulator
except ImportError:
    from usage import UsageAccumulator


HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
SECRET_ENV = "MODEL_EVAL_UPSTREAM_API_KEY"
DEFAULT_MAX_BODY = 64 * 1024 * 1024


class Server(ThreadingHTTPServer):
    daemon_threads = True


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(NoRedirect())


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_limited(response, limit: int) -> bytes:
    headers = getattr(response, "headers", None)
    length = headers.get("Content-Length") if headers else None
    if length and str(length).isdigit() and int(length) > limit:
        raise ValueError("response too large")
    data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("response too large")
    return data


def _forward(request, *, timeout: float, limit: int):
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, response.headers, read_limited(response, limit)
    except urllib.error.HTTPError as exc:
        if 300 <= int(exc.code) < 400:
            return 502, {"Content-Type": "application/json"}, (
                b'{"error":"upstream redirect refused"}'
            )
        try:
            return exc.code, exc.headers, read_limited(exc, limit)
        except ValueError:
            return 502, {"Content-Type": "application/json"}, (
                b'{"error":"upstream response too large"}'
            )
    except Exception as exc:
        payload = json.dumps({"error": str(exc)}, allow_nan=False).encode()
        return 502, {"Content-Type": "application/json"}, payload


def _validated_routes(namespace) -> dict[tuple[str, str], str | None]:
    routes = {
        ("POST", "/v1/completions"): namespace.completion_url,
        ("GET", "/tokenizer_info"): namespace.tokenizer_info_url,
        ("POST", "/tokenize"): namespace.tokenize_url,
        ("POST", "/detokenize"): namespace.detokenize_url,
    }
    for target in (value for value in routes.values() if value):
        parsed = urllib.parse.urlsplit(target)
        invalid = (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        )
        if invalid:
            raise SystemExit(f"invalid upstream URL: {target}")
    return routes


def _handler(namespace, routes, api_key: str, usage: UsageAccumulator | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return None

        def _send(self, status: int, payload: bytes, headers=None) -> None:
            self.send_response(status)
            for key, value in (headers or {}).items():
                lowered = key.lower()
                if lowered not in HOP_HEADERS and lowered not in {
                    "content-length",
                    "location",
                }:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

        def _content_length(self) -> int | None:
            try:
                value = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                self._send(
                    400,
                    b'{"error":"invalid content length"}',
                    {"Content-Type": "application/json"},
                )
                return None
            if value < 0 or value > namespace.max_body_bytes:
                self._send(
                    413,
                    b'{"error":"request too large"}',
                    {"Content-Type": "application/json"},
                )
                return None
            return value

        def _go(self, method: str) -> None:
            route = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
            target = routes.get((method, route))
            if not target:
                self._send(
                    404,
                    b'{"error":"unsupported proxy route"}',
                    {"Content-Type": "application/json"},
                )
                return
            length = self._content_length()
            if length is None:
                return
            body = self.rfile.read(length) if length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_HEADERS
                and key.lower() != "authorization"
            }
            if namespace.auth_mode == "inject":
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(
                target,
                data=body,
                headers=headers,
                method=method,
            )
            status, response_headers, payload = _forward(
                request,
                timeout=namespace.timeout,
                limit=namespace.max_body_bytes,
            )
            is_completion = method == "POST" and route == "/v1/completions"
            if usage is not None and is_completion and 200 <= status < 300:
                usage.observe(payload)
            self._send(status, payload, response_headers)

        def do_GET(self):
            self._go("GET")

        def do_POST(self):
            self._go("POST")

    return Handler


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion-url", required=True)
    parser.add_argument("--tokenizer-info-url")
    parser.add_argument("--tokenize-url")
    parser.add_argument("--detokenize-url")
    parser.add_argument("--auth-mode", choices=["strip", "inject"], required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--usage-file")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY)
    return parser.parse_args()


def main() -> None:
    namespace = _parse_args()
    if namespace.max_body_bytes <= 0:
        raise SystemExit("--max-body-bytes must be positive")
    api_key = os.environ.get(SECRET_ENV, "")
    if namespace.auth_mode == "inject" and not api_key:
        raise SystemExit(f"{SECRET_ENV} required")
    routes = _validated_routes(namespace)
    usage = (
        UsageAccumulator(Path(namespace.usage_file))
        if namespace.usage_file
        else None
    )
    server = Server(
        ("127.0.0.1", 0),
        _handler(namespace, routes, api_key, usage),
    )

    def watch_parent() -> None:
        while alive(namespace.parent_pid):
            time.sleep(0.25)
        server.shutdown()

    threading.Thread(target=watch_parent, daemon=True).start()
    ready = Path(namespace.ready_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready.with_name(ready.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps({"port": int(server.server_address[1])}, allow_nan=False) + "\n"
    )
    temporary.replace(ready)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        ready.unlink(missing_ok=True)
        server.server_close()


if __name__ == "__main__":
    main()
