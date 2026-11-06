"""Opt-in OpenAI Chat Completions image transport preflight."""
from model_evaluation.sdk.http import request_json
from model_evaluation.sdk.runtime import AdapterError

# Small, self-contained RGB PNG; no external URLs or dataset dependency.
_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeUlEQVR4nO3PQQkAMAzAwEqpf1WTMhF7HINABFzm7H7dcEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFj12kKwEPu0W5YAAAAABJRU5ErkJggg=="


def require_chat_image(url: str, model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is the image? Reply with one word."},
            {"type": "image_url", "image_url": {"url": _IMAGE}},
        ]}],
        "max_tokens": 8,
        "temperature": 0,
    }
    try:
        _, reply = request_json(url, method="POST", payload=payload, timeout=20)
    except AdapterError as exc:
        raise AdapterError("COMPATIBILITY_ERROR", f"Image input preflight failed: {exc}") from exc
    choices = reply.get("choices") if isinstance(reply, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict) or not any(
        isinstance(message.get(k), str) and message[k].strip()
        for k in ("content", "reasoning", "reasoning_content")
    ):
        raise AdapterError("COMPATIBILITY_ERROR", "Image input preflight returned no generated text")
