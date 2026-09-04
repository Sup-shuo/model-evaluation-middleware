from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from model_evaluation.sdk.native_backend import NativeDriver, NativeDriverError


_DTYPE_NAMES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}

_GENERATION_PARAMETERS = {
    "max_new_tokens",
    "max_tokens",
    "num_beams",
    "seed",
    "temperature",
    "top_p",
}


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeDriverError(
            "REQUEST_INVALID",
            f"{name} must be an integer >= {minimum}",
        )
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeDriverError(
            "REQUEST_INVALID",
            f"{name} must be numeric",
        )
    numeric = float(value)
    if (
        not math.isfinite(numeric)
        or numeric < minimum
        or (maximum is not None and numeric > maximum)
    ):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise NativeDriverError(
            "REQUEST_INVALID",
            f"{name} must be finite and >= {minimum}{suffix}",
        )
    return numeric


class TinyLlavaDriver(NativeDriver):
    """Driver for modern TinyLLaVA Factory checkpoints with model.chat."""

    def __init__(self, configuration: dict[str, Any]):
        self.configuration = dict(configuration)
        self.torch = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self._descriptor: dict[str, Any] | None = None

    def _resolve_device(self, torch) -> str:
        configured = str(self.configuration.get("device") or "auto")
        if configured != "auto":
            return configured
        if torch.cuda.is_available():
            return "cuda:0"
        for namespace, name in (("mlu", "mlu:0"), ("npu", "npu:0")):
            module = getattr(torch, namespace, None)
            available = getattr(module, "is_available", None)
            if callable(available) and available():
                return name
        return "cpu"

    def _from_pretrained(self, factory, path: str, kwargs: dict[str, Any]):
        signature = inspect.signature(factory.from_pretrained).parameters
        if "dtype" not in signature and "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return factory.from_pretrained(path, **kwargs)

    def _check_offload(self) -> None:
        if bool(self.configuration.get("allow_cpu_offload", False)):
            return
        device_map = getattr(self.model, "hf_device_map", None)
        if not isinstance(device_map, dict):
            return
        offloaded = sorted(
            {
                str(value)
                for value in device_map.values()
                if str(value).lower() in {"cpu", "disk"}
            }
        )
        if offloaded:
            raise NativeDriverError(
                "CPU_OFFLOAD_FORBIDDEN",
                f"TinyLLaVA resolved CPU/disk offload targets: {offloaded}",
                status=500,
            )

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise NativeDriverError(
                "DEPENDENCY_MISSING",
                f"TinyLLaVA dependency import failed: {exc}",
                status=500,
            ) from exc
        self.torch = torch
        path = str(self.configuration["model_path"])
        offline = bool(self.configuration.get("offline", False))
        kwargs: dict[str, Any] = {
            "local_files_only": offline,
            "trust_remote_code": True,
        }
        dtype = str(self.configuration.get("dtype") or "auto")
        if dtype != "auto":
            kwargs["dtype"] = getattr(torch, _DTYPE_NAMES[dtype])
        if self.configuration.get("device_map") is not None:
            kwargs["device_map"] = self.configuration["device_map"]
        try:
            self.model = self._from_pretrained(
                AutoModelForCausalLM,
                path,
                kwargs,
            )
            config = self.model.config
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=True,
                local_files_only=offline,
                use_fast=False,
                model_max_length=getattr(
                    config,
                    "tokenizer_model_max_length",
                    int(self.configuration.get("max_model_len") or 4096),
                ),
                padding_side=getattr(
                    config,
                    "tokenizer_padding_side",
                    "right",
                ),
            )
            if not callable(getattr(self.model, "chat", None)):
                raise NativeDriverError(
                    "MODEL_FAMILY_API_MISSING",
                    "loaded TinyLLaVA model does not provide model.chat",
                    status=500,
                )
            self.device = self._resolve_device(torch)
            if self.configuration.get("device_map") is None:
                self.model.to(self.device)
            self._check_offload()
            self.model.eval()
        except NativeDriverError:
            raise
        except Exception as exc:
            raise NativeDriverError(
                "MODEL_LOAD_FAILED",
                f"TinyLLaVA model loading failed: {exc}",
                status=500,
            ) from exc
        self._descriptor = {
            "model_id": str(self.configuration["model_id"]),
            "capabilities": {
                "generation": False,
                "chat": True,
                "completion_logprobs": False,
                "echo": False,
                "tokenize": True,
                "detokenize": True,
                "media.image": True,
                "media.video": False,
                "output.grounding": False,
            },
            "context_length": int(
                self.configuration.get("max_model_len") or 4096
            ),
            "metadata": {
                "backend": "tinyllava",
                "chat_mode": "single_turn_image",
                "device": str(self.device),
                "family_api": "model.chat",
                "model_class": type(self.model).__name__,
            },
        }

    def describe(self) -> dict[str, Any]:
        if self._descriptor is None:
            raise NativeDriverError(
                "SERVICE_NOT_READY",
                "TinyLLaVA model is not loaded",
                status=503,
            )
        return dict(self._descriptor)

    @staticmethod
    def _text_and_image(
        messages: list[dict[str, Any]],
    ) -> tuple[str, str]:
        system_parts: list[str] = []
        user_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "")
            if role == "system":
                content = message.get("content")
                if not isinstance(content, str):
                    raise NativeDriverError(
                        "REQUEST_INVALID",
                        "TinyLLaVA system content must be text",
                    )
                system_parts.append(content)
            elif role == "user":
                user_messages.append(message)
            else:
                raise NativeDriverError(
                    "OPERATION_UNSUPPORTED",
                    "TinyLLaVA Adapter supports one system message and "
                    "one user turn only",
                )
        if len(system_parts) > 1:
            raise NativeDriverError(
                "REQUEST_INVALID",
                "TinyLLaVA Adapter accepts at most one system message",
            )
        if len(user_messages) != 1:
            raise NativeDriverError(
                "REQUEST_INVALID",
                "TinyLLaVA Adapter requires exactly one user message",
            )
        content = user_messages[0].get("content")
        if not isinstance(content, list):
            raise NativeDriverError(
                "REQUEST_INVALID",
                "TinyLLaVA user content must contain text and image_url parts",
            )
        text_parts: list[str] = []
        images: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise NativeDriverError(
                    "REQUEST_INVALID",
                    "TinyLLaVA content parts must be objects",
                )
            part_type = part.get("type")
            if (
                part_type == "text"
                and isinstance(part.get("text"), str)
                and part["text"]
            ):
                text_parts.append(part["text"])
                continue
            if part_type == "image_url":
                image_url = part.get("image_url")
                url = (
                    image_url.get("url")
                    if isinstance(image_url, dict)
                    else image_url
                )
                if isinstance(url, str) and url:
                    images.append(url)
                    continue
            raise NativeDriverError(
                "REQUEST_INVALID",
                f"unsupported TinyLLaVA content part: {part_type!r}",
            )
        if len(images) != 1 or not text_parts:
            raise NativeDriverError(
                "REQUEST_INVALID",
                "TinyLLaVA requires one image and at least one text part",
            )
        prompt_parts = system_parts + text_parts
        return "\n".join(prompt_parts), images[0]

    def _validate_image(self, image: str) -> str:
        parsed = urlparse(image)
        if parsed.scheme in {"http", "https"}:
            if bool(self.configuration.get("offline", False)):
                raise NativeDriverError(
                    "NETWORK_POLICY_VIOLATION",
                    "remote TinyLLaVA image URL is forbidden in offline mode",
                )
            return image
        path = Path(image)
        if not path.is_absolute() or not path.is_file():
            raise NativeDriverError(
                "REQUEST_INVALID",
                "TinyLLaVA local image must be an existing absolute file",
            )
        return str(path)

    def _parameters(self, raw: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(raw) - _GENERATION_PARAMETERS)
        if unknown:
            raise NativeDriverError(
                "REQUEST_INVALID",
                f"unsupported TinyLLaVA generation parameters: {unknown}",
            )
        values = dict(raw)
        max_tokens = values.pop("max_tokens", None)
        if max_tokens is not None and "max_new_tokens" in values:
            raise NativeDriverError(
                "REQUEST_INVALID",
                "use either max_tokens or max_new_tokens, not both",
            )
        requested_max = values.get(
            "max_new_tokens",
            512 if max_tokens is None else max_tokens,
        )
        values["max_new_tokens"] = _integer(
            requested_max,
            name="max_tokens",
            minimum=1,
        )
        values["num_beams"] = _integer(
            values.get("num_beams", 1),
            name="num_beams",
            minimum=1,
        )
        values["temperature"] = _number(
            values.get("temperature", 0),
            name="temperature",
            minimum=0.0,
        )
        if "top_p" in values:
            values["top_p"] = _number(
                values["top_p"],
                name="top_p",
                minimum=0.0,
                maximum=1.0,
            )
        seed = values.pop("seed", None)
        if seed is not None:
            self.torch.manual_seed(_integer(seed, name="seed", minimum=0))
        return values

    def chat(
        self,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        prompt, image = self._text_and_image(messages)
        image = self._validate_image(image)
        kwargs = self._parameters(parameters)
        try:
            result = self.model.chat(
                prompt=prompt,
                image=image,
                tokenizer=self.tokenizer,
                **kwargs,
            )
        except Exception as exc:
            raise NativeDriverError(
                "MODEL_INFERENCE_FAILED",
                f"TinyLLaVA model.chat failed: {exc}",
                status=500,
            ) from exc
        if not (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], str)
        ):
            raise NativeDriverError(
                "MODEL_FAMILY_API_INVALID",
                "TinyLLaVA model.chat returned an unexpected result",
                status=500,
            )
        text, duration = result
        prompt_tokens = len(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )
        completion_tokens = len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )
        return {
            "text": text,
            "finish_reason": "stop",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_seconds": float(duration),
        }

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def detokenize(self, tokens: list[int]) -> str:
        return str(
            self.tokenizer.decode(tokens, skip_special_tokens=False)
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.torch is None:
            return
        for namespace in ("cuda", "mlu", "npu"):
            module = getattr(self.torch, namespace, None)
            empty_cache = getattr(module, "empty_cache", None)
            if callable(empty_cache):
                try:
                    empty_cache()
                except Exception:
                    pass
