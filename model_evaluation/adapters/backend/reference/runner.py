#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import socket
import threading


def serve(host: str, port: int) -> None:
    stopped = threading.Event()

    def stop(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(4)
        listener.settimeout(0.2)
        print(f"reference backend listening on {host}:{port}", flush=True)
        while not stopped.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                connection.sendall(b"reference-backend\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39091)
    args = parser.parse_args()
    if args.version:
        print("reference-backend 1.0.0")
        return
    if not args.serve:
        parser.error("--serve is required")
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
