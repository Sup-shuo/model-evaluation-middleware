from __future__ import annotations

from typing import Any

from model_evaluation.sdk.http import normalize_http_base_url, request_json
from model_evaluation.sdk.runtime import AdapterError


_SERVICE_CAPABILITIES = {
    "generation": "service.generation",
    "chat": "service.chat",
    "completion_logprobs": "service.completion_logprobs",
    "echo": "service.echo",
    "tokenize": "service.tokenize",
    "detokenize": "service.detokenize",
    "media.image": "service.media.image",
    "media.video": "service.media.video",
    "output.grounding": "service.output.grounding",
}


def _model_descriptor(payload: object, model_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AdapterError(
            "SERVICE_NOT_READY",
            "/models returned a non-object response",
            retryable=True,
        )
    models = [
        item
        for item in (payload.get("data") or [])
        if isinstance(item, dict) and item.get("id") is not None
    ]
    found = next(
        (item for item in models if str(item.get("id")) == model_id),
        None,
    )
    if found is None:
        available = sorted(str(item.get("id")) for item in models)
        raise AdapterError(
            "SERVICE_NOT_READY",
            f"model {model_id!r} not listed by native service; available={available}",
            retryable=not bool(models),
        )
    return found


def _health(payload: object, model_id: str) -> None:
    if not isinstance(payload, dict):
        raise AdapterError(
            "SERVICE_NOT_READY",
            "/health returned a non-object response",
            retryable=True,
        )
    if str(payload.get("model_id") or "") != model_id:
        raise AdapterError(
            "SERVICE_NOT_READY",
            "native service health response names a different model",
            retryable=False,
        )
    state = str(payload.get("state") or "")
    if state == "ready":
        return
    if state == "failed":
        error = payload.get("error") or {}
        code = str(error.get("code") or "MODEL_LOAD_FAILED")
        message = str(error.get("message") or "native model loading failed")
        raise AdapterError(code, message, retryable=False)
    raise AdapterError(
        "SERVICE_NOT_READY",
        f"native service state is {state or 'unknown'}",
        retryable=True,
    )


def probe_native_service(
    *,
    base_url: str,
    model_id: str,
    ownership: str,
    timeout: float,
    max_concurrent: int,
) -> dict[str, Any]:
    """Read the driver-reported contract without triggering inference."""

    base = normalize_http_base_url(base_url)
    request_timeout = max(0.25, min(3.0, float(timeout) / 8.0))
    health_url = base[:-3] + "/health" if base.endswith("/v1") else base + "/health"
    _, health = request_json(health_url, timeout=request_timeout)
    _health(health, model_id)
    _, payload = request_json(base + "/models", timeout=request_timeout)
    model = _model_descriptor(payload, model_id)
    reported = model.get("capabilities")
    if not isinstance(reported, dict):
        raise AdapterError(
            "SERVICE_NOT_READY",
            "native service model descriptor has no capability object",
            retryable=False,
        )
    capabilities = {
        public: bool(reported.get(driver_key, False))
        for driver_key, public in _SERVICE_CAPABILITIES.items()
    }
    protocols: dict[str, dict[str, str]] = {
        "openai_models": {"url": base + "/models"}
    }
    if capabilities["service.generation"]:
        protocols["openai_completion"] = {"url": base + "/completions"}
    if capabilities["service.chat"]:
        protocols["openai_chat"] = {"url": base + "/chat/completions"}
    root = base[:-3] if base.endswith("/v1") else base
    if capabilities["service.tokenize"]:
        protocols["tokenize"] = {"url": root + "/tokenize"}
    if capabilities["service.detokenize"]:
        protocols["detokenize"] = {"url": root + "/detokenize"}
    descriptor: dict[str, Any] = {
        "schema_version": "1.0",
        "service_type": "llm-native",
        "ownership": ownership,
        "model": {"id": model_id},
        "protocols": protocols,
        "capabilities": {"schema_version": "1.0", "values": capabilities},
        "auth": {"mode": "none"},
        "tokenizer": {
            "mode": (
                "remote"
                if capabilities["service.tokenize"]
                and capabilities["service.detokenize"]
                else "none"
            )
        },
        "limits": {"recommended_concurrency": max(1, int(max_concurrent))},
        "metadata": {
            "capability_evidence": "reported_by_loaded_native_driver",
            "driver": model.get("metadata") or {},
        },
    }
    context_length = model.get("context_length")
    if isinstance(context_length, int) and not isinstance(context_length, bool):
        descriptor["context_length"] = context_length
    return descriptor
