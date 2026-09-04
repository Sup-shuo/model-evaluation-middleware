#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from driver import TransformersCausalLMDriver
from model_evaluation.sdk.native_backend import serve_native_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload)
    if not isinstance(payload, dict):
        raise TypeError("runner payload must be a JSON object")
    driver = TransformersCausalLMDriver(payload)
    serve_native_backend(
        driver,
        host=str(payload["host"]),
        port=int(payload["port"]),
        model_id=str(payload["model_id"]),
        max_request_bytes=int(payload["max_request_bytes"]),
        max_concurrent=int(payload["max_concurrent"]),
    )


if __name__ == "__main__":
    main()
