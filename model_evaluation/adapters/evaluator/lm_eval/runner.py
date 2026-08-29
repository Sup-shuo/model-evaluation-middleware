#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PLACEHOLDER = "__MODEL_EVAL_PROXY_COMPLETIONS__"


def _bind_python_to_current_interpreter(command):
    prepared = list(command)
    if prepared and prepared[0] in {"python", "python3"}:
        prepared[0] = sys.executable
    return prepared


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion-url", required=True)
    parser.add_argument("--tokenizer-info-url")
    parser.add_argument("--tokenize-url")
    parser.add_argument("--detokenize-url")
    parser.add_argument("--auth-mode", choices=["strip", "inject"], required=True)
    parser.add_argument("--usage-file")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _proxy_command(namespace, ready: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("transport_proxy.py")),
        "--completion-url",
        namespace.completion_url,
        "--auth-mode",
        namespace.auth_mode,
        "--ready-file",
        str(ready),
        "--parent-pid",
        str(os.getpid()),
        "--timeout",
        str(namespace.timeout),
    ]
    optional = (
        ("--tokenizer-info-url", namespace.tokenizer_info_url),
        ("--tokenize-url", namespace.tokenize_url),
        ("--detokenize-url", namespace.detokenize_url),
        ("--usage-file", namespace.usage_file),
    )
    for flag, value in optional:
        if value:
            command.extend((flag, value))
    return command


def _wait_for_proxy(proxy, ready: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ready.is_file():
        if proxy.poll() is not None:
            raise SystemExit(f"transport proxy exited early rc={proxy.returncode}")
        time.sleep(0.05)
    if not ready.is_file():
        raise SystemExit("transport proxy readiness timeout")
    return int(json.loads(ready.read_text())["port"])


def _stop_proxy(proxy) -> None:
    if proxy.poll() is not None:
        return
    proxy.terminate()
    try:
        proxy.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proxy.kill()
        proxy.wait(timeout=1)


def main() -> int:
    namespace = _parse_args()
    command = list(namespace.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing evaluator command")

    ready = Path(tempfile.gettempdir()) / f"model-eval-lmproxy-{os.getpid()}.json"
    ready.unlink(missing_ok=True)
    proxy = subprocess.Popen(
        _proxy_command(namespace, ready),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
    )
    try:
        port = _wait_for_proxy(proxy, ready)
        base_url = f"http://127.0.0.1:{port}/v1/completions"
        command = [item.replace(PLACEHOLDER, base_url) for item in command]
        command = _bind_python_to_current_interpreter(command)
        environment = os.environ.copy()
        environment.pop("MODEL_EVAL_UPSTREAM_API_KEY", None)
        environment["OPENAI_API_KEY"] = "local-transport-proxy"
        child = subprocess.Popen(command, env=environment)
        return child.wait()
    finally:
        ready.unlink(missing_ok=True)
        _stop_proxy(proxy)


if __name__ == "__main__":
    raise SystemExit(main())
