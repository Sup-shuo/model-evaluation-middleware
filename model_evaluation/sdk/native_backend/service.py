from __future__ import annotations

import json
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .driver import NativeDriver, NativeDriverError
from .protocol import normalize_generation_result, validate_descriptor


def _reject_constant(token: str):
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _strict_loads(value: bytes) -> object:
    return json.loads(value.decode("utf-8"), parse_constant=_reject_constant)


class NativeApplication:
    def __init__(
        self,
        driver: NativeDriver,
        *,
        model_id: str,
        max_request_bytes: int,
        max_concurrent: int,
    ):
        for name, value in (
            ("max_request_bytes", max_request_bytes),
            ("max_concurrent", max_concurrent),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.driver = driver
        self.model_id = model_id
        self.max_request_bytes = max_request_bytes
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._lock = threading.Lock()
        self._state = "loading"
        self._descriptor: dict[str, Any] | None = None
        self._error: dict[str, Any] | None = None

    def load(self) -> None:
        try:
            self.driver.load()
            descriptor = validate_descriptor(
                self.driver.describe(),
                model_id=self.model_id,
            )
        except Exception as exc:
            code = (
                exc.code if isinstance(exc, NativeDriverError) else "MODEL_LOAD_FAILED"
            )
            with self._lock:
                self._state = "failed"
                self._error = {
                    "code": code,
                    "message": str(exc) or type(exc).__name__,
                }
            return
        with self._lock:
            self._descriptor = descriptor
            self._state = "ready"

    def health(self) -> dict[str, Any]:
        with self._lock:
            value: dict[str, Any] = {
                "schema_version": "1.0",
                "state": self._state,
                "model_id": self.model_id,
            }
            if self._error:
                value["error"] = dict(self._error)
            return value

    def descriptor(self) -> dict[str, Any]:
        with self._lock:
            if self._state != "ready" or self._descriptor is None:
                raise NativeDriverError(
                    "SERVICE_NOT_READY",
                    f"native backend state is {self._state}",
                    status=503,
                )
            return dict(self._descriptor)

    def invoke(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        descriptor = self.descriptor()
        capabilities = descriptor["capabilities"]
        if payload.get("stream"):
            raise NativeDriverError(
                "OPERATION_UNSUPPORTED",
                "streaming is not supported",
            )
        if payload.get("model") != self.model_id:
            raise NativeDriverError(
                "MODEL_NOT_FOUND",
                f"requested model {payload.get('model')!r} is not loaded",
                status=404,
            )
        if not self._slots.acquire(timeout=30):
            raise NativeDriverError(
                "RESOURCE_BUSY",
                "native backend concurrency limit reached",
                status=429,
            )
        try:
            parameters = {
                key: value
                for key, value in payload.items()
                if key not in {"model", "prompt", "messages", "stream"}
            }
            if operation == "completion":
                if not capabilities["generation"]:
                    raise NativeDriverError(
                        "OPERATION_UNSUPPORTED",
                        "driver does not provide completions",
                    )
                prompt = payload.get("prompt")
                if not isinstance(prompt, str):
                    raise NativeDriverError(
                        "REQUEST_INVALID",
                        "prompt must be a string",
                    )
                value = self.driver.completion(prompt, parameters)
            elif operation == "chat":
                if not capabilities["chat"]:
                    raise NativeDriverError(
                        "OPERATION_UNSUPPORTED",
                        "driver does not provide chat",
                    )
                messages = payload.get("messages")
                if not isinstance(messages, list) or not all(
                    isinstance(item, dict) for item in messages
                ):
                    raise NativeDriverError(
                        "REQUEST_INVALID",
                        "messages must be an array of objects",
                    )
                value = self.driver.chat(messages, parameters)
            else:
                raise NativeDriverError(
                    "OPERATION_UNSUPPORTED",
                    f"unknown operation: {operation}",
                    status=404,
                )
            return normalize_generation_result(value)
        finally:
            self._slots.release()

    def close(self) -> None:
        self.driver.close()


class NativeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, application: NativeApplication):
        self.application = application
        super().__init__(address, NativeHandler)


class NativeHandler(BaseHTTPRequestHandler):
    server: NativeServer

    def log_message(self, fmt: str, *args) -> None:
        print(f"native-backend http: {fmt % args}", flush=True)

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception) -> None:
        if isinstance(exc, NativeDriverError):
            status = exc.status
            code = exc.code
        else:
            status = 500
            code = "NATIVE_BACKEND_INTERNAL_ERROR"
        self._write(
            status,
            {
                "error": {
                    "code": code,
                    "message": str(exc) or type(exc).__name__,
                }
            },
        )

    def _payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError as exc:
            raise NativeDriverError(
                "REQUEST_INVALID",
                "invalid Content-Length",
            ) from exc
        if length < 1 or length > self.server.application.max_request_bytes:
            raise NativeDriverError(
                "REQUEST_INVALID",
                "request body is empty or exceeds configured size limit",
                status=413,
            )
        try:
            value = _strict_loads(self.rfile.read(length))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise NativeDriverError(
                "REQUEST_INVALID",
                f"request body is not valid JSON: {exc}",
            ) from exc
        if not isinstance(value, dict):
            raise NativeDriverError(
                "REQUEST_INVALID",
                "request body must be an object",
            )
        return value

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path == "/health":
                self._write(200, self.server.application.health())
                return
            if path == "/v1/models":
                descriptor = self.server.application.descriptor()
                model = {
                    "id": descriptor["model_id"],
                    "object": "model",
                    "owned_by": "model-evaluation-native-backend",
                    "capabilities": descriptor["capabilities"],
                    "metadata": descriptor["metadata"],
                }
                if descriptor.get("context_length") is not None:
                    model["context_length"] = descriptor["context_length"]
                self._write(200, {"object": "list", "data": [model]})
                return
            raise NativeDriverError(
                "NOT_FOUND",
                f"unknown endpoint: {path}",
                status=404,
            )
        except Exception as exc:
            self._error(exc)

    def _generation_response(
        self,
        *,
        chat: bool,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if chat:
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
            response: dict[str, Any] = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.server.application.model_id,
                "choices": [choice],
            }
        else:
            choice = {
                "index": 0,
                "text": result["text"],
                "finish_reason": result["finish_reason"],
            }
            response = {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.server.application.model_id,
                "choices": [choice],
            }
        for key in ("usage", "performance"):
            if key in result:
                response[key] = result[key]
        return response

    def _tokenize(self, payload: dict[str, Any]) -> None:
        descriptor = self.server.application.descriptor()
        if not descriptor["capabilities"]["tokenize"]:
            raise NativeDriverError(
                "OPERATION_UNSUPPORTED",
                "tokenization is unavailable",
            )
        text = payload.get("prompt")
        if not isinstance(text, str):
            raise NativeDriverError(
                "REQUEST_INVALID",
                "prompt must be a string",
            )
        tokens = self.server.application.driver.tokenize(text)
        self._write(200, {"tokens": tokens})

    def _detokenize(self, payload: dict[str, Any]) -> None:
        descriptor = self.server.application.descriptor()
        if not descriptor["capabilities"]["detokenize"]:
            raise NativeDriverError(
                "OPERATION_UNSUPPORTED",
                "detokenization is unavailable",
            )
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in tokens
        ):
            raise NativeDriverError(
                "REQUEST_INVALID",
                "tokens must be an integer array",
            )
        text = self.server.application.driver.detokenize(tokens)
        self._write(200, {"text": text})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            payload = self._payload()
            if path == "/v1/completions":
                result = self.server.application.invoke("completion", payload)
                self._write(
                    200,
                    self._generation_response(chat=False, result=result),
                )
                return
            if path == "/v1/chat/completions":
                result = self.server.application.invoke("chat", payload)
                self._write(
                    200,
                    self._generation_response(chat=True, result=result),
                )
                return
            if path == "/tokenize":
                self._tokenize(payload)
                return
            if path == "/detokenize":
                self._detokenize(payload)
                return
            raise NativeDriverError(
                "NOT_FOUND",
                f"unknown endpoint: {path}",
                status=404,
            )
        except Exception as exc:
            self._error(exc)


def serve_native_backend(
    driver: NativeDriver,
    *,
    host: str,
    port: int,
    model_id: str,
    max_request_bytes: int = 16 * 1024 * 1024,
    max_concurrent: int = 1,
) -> None:
    """Serve one model driver until SIGINT/SIGTERM."""

    application = NativeApplication(
        driver,
        model_id=model_id,
        max_request_bytes=max_request_bytes,
        max_concurrent=max_concurrent,
    )
    server = NativeServer((host, port), application)
    server.timeout = 0.25
    stopped = threading.Event()

    def stop(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    loader = threading.Thread(
        target=application.load,
        name="native-model-loader",
        daemon=True,
    )
    loader.start()
    print(f"native backend listening on {host}:{port} for {model_id}", flush=True)
    try:
        while not stopped.is_set():
            server.handle_request()
    finally:
        server.server_close()
        if loader.is_alive():
            print(
                "native backend stopped while model loading was still active",
                flush=True,
            )
        else:
            application.close()
