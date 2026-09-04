from __future__ import annotations

import inspect
import math
import time
from typing import Any

from model_evaluation.sdk.native_backend import NativeDriver, NativeDriverError


_DTYPE_NAMES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}

_GENERATION_PARAMETERS = {
    "do_sample",
    "max_new_tokens",
    "max_tokens",
    "min_new_tokens",
    "repetition_penalty",
    "seed",
    "stop",
    "temperature",
    "top_k",
    "top_p",
}


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeDriverError(
            "REQUEST_INVALID",
            f"{name} must be an integer >= {minimum}",
        )
    return value


def _number(value: object, *, name: str, minimum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise NativeDriverError(
            "REQUEST_INVALID",
            f"{name} must be numeric and >= {minimum}",
        )
    return float(value)


def _stop_sequences(value: object) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(
        isinstance(item, str) and item for item in values
    ):
        raise NativeDriverError(
            "REQUEST_INVALID",
            "stop must be a non-empty string or string array",
        )
    return values


class TransformersCausalLMDriver(NativeDriver):
    """Standard AutoModelForCausalLM driver without model-family branches."""

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
            if module is not None and callable(getattr(module, "is_available", None)):
                if module.is_available():
                    return name
        return "cpu"

    def _model_kwargs(self, torch) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "trust_remote_code": bool(
                self.configuration.get("trust_remote_code", False)
            ),
            "local_files_only": bool(self.configuration.get("offline", False)),
        }
        dtype_name = str(self.configuration.get("dtype") or "auto")
        if dtype_name != "auto":
            parameters["dtype"] = getattr(torch, _DTYPE_NAMES[dtype_name])
        device_map = self.configuration.get("device_map")
        if device_map is not None:
            parameters["device_map"] = device_map
        return parameters

    @staticmethod
    def _call_from_pretrained(factory, path: str, parameters: dict[str, Any]):
        signature = inspect.signature(factory.from_pretrained).parameters
        if "dtype" not in signature and "dtype" in parameters:
            parameters["torch_dtype"] = parameters.pop("dtype")
        return factory.from_pretrained(path, **parameters)

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
                "resolved Transformers device_map uses "
                f"{offloaded}; set allow_cpu_offload=true explicitly",
                status=500,
            )

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise NativeDriverError(
                "DEPENDENCY_MISSING",
                f"Transformers Backend dependency import failed: {exc}",
                status=500,
            ) from exc
        self.torch = torch
        model_path = str(self.configuration["model_path"])
        tokenizer_path = str(self.configuration["tokenizer_path"])
        trust_remote_code = bool(
            self.configuration.get("trust_remote_code", False)
        )
        offline = bool(self.configuration.get("offline", False))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=trust_remote_code,
                local_files_only=offline,
            )
            self.model = self._call_from_pretrained(
                AutoModelForCausalLM,
                model_path,
                self._model_kwargs(torch),
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
                f"Transformers model loading failed: {exc}",
                status=500,
            ) from exc
        config = getattr(self.model, "config", None)
        configured_context = int(
            self.configuration.get("max_model_len") or 4096
        )
        model_context = None
        for name in (
            "max_position_embeddings",
            "n_positions",
            "seq_length",
        ):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                model_context = value
                break
        context_length = min(
            configured_context,
            model_context or configured_context,
        )
        chat = bool(getattr(self.tokenizer, "chat_template", None))
        self._descriptor = {
            "model_id": str(self.configuration["model_id"]),
            "capabilities": {
                "generation": True,
                "chat": chat,
                "completion_logprobs": False,
                "echo": False,
                "tokenize": True,
                "detokenize": True,
                "media.image": False,
                "media.video": False,
                "output.grounding": False,
            },
            "context_length": context_length,
            "metadata": {
                "backend": "transformers",
                "device": str(self.device),
                "model_class": type(self.model).__name__,
                "tokenizer_class": type(self.tokenizer).__name__,
            },
        }

    def describe(self) -> dict[str, Any]:
        if self._descriptor is None:
            raise NativeDriverError(
                "SERVICE_NOT_READY",
                "Transformers model is not loaded",
                status=503,
            )
        return dict(self._descriptor)

    def _input_device(self):
        embeddings = self.model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None and getattr(device, "type", None) != "meta":
            return device
        for parameter in self.model.parameters():
            if getattr(parameter.device, "type", None) != "meta":
                return parameter.device
        raise NativeDriverError(
            "MODEL_DEVICE_UNAVAILABLE",
            "could not resolve a concrete model input device",
            status=500,
        )

    def _inputs(self, text: str):
        encoded = self.tokenizer(text, return_tensors="pt")
        target = self._input_device()
        return {key: value.to(target) for key, value in encoded.items()}

    def _generation_kwargs(
        self,
        parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        unknown = sorted(set(parameters) - _GENERATION_PARAMETERS)
        if unknown:
            raise NativeDriverError(
                "REQUEST_INVALID",
                f"unsupported generation parameters: {unknown}",
            )
        kwargs = dict(parameters)
        stops = _stop_sequences(kwargs.pop("stop", None))
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_new_tokens" in kwargs:
            raise NativeDriverError(
                "REQUEST_INVALID",
                "use either max_tokens or max_new_tokens, not both",
            )
        requested_max = kwargs.get(
            "max_new_tokens",
            16 if max_tokens is None else max_tokens,
        )
        kwargs["max_new_tokens"] = _integer(
            requested_max,
            name="max_tokens",
            minimum=1,
        )
        if "min_new_tokens" in kwargs:
            kwargs["min_new_tokens"] = _integer(
                kwargs["min_new_tokens"],
                name="min_new_tokens",
                minimum=0,
            )
        if "top_k" in kwargs:
            kwargs["top_k"] = _integer(
                kwargs["top_k"],
                name="top_k",
                minimum=0,
            )
        if "do_sample" in kwargs and not isinstance(kwargs["do_sample"], bool):
            raise NativeDriverError(
                "REQUEST_INVALID",
                "do_sample must be boolean",
            )
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self.torch.manual_seed(_integer(seed, name="seed", minimum=0))
        for name, minimum in (
            ("temperature", 0.0),
            ("top_p", 0.0),
            ("repetition_penalty", 0.0),
        ):
            if name in kwargs:
                kwargs[name] = _number(
                    kwargs[name],
                    name=name,
                    minimum=minimum,
                )
        if kwargs.get("temperature") == 0:
            kwargs["do_sample"] = False
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)
        return kwargs, stops

    def completion(
        self,
        prompt: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        inputs = self._inputs(prompt)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        kwargs, stops = self._generation_kwargs(parameters)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        duration = time.perf_counter() - started
        generated = output[0, prompt_tokens:]
        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
        stop_positions = [
            text.find(stop)
            for stop in stops
            if text.find(stop) >= 0
        ]
        stopped = bool(stop_positions)
        if stopped:
            text = text[: min(stop_positions)]
        completion_tokens = len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )
        finish_reason = "stop"
        if not stopped and int(generated.shape[-1]) >= kwargs["max_new_tokens"]:
            finish_reason = "length"
        return {
            "text": text,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration,
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        for message in messages:
            if not isinstance(message.get("content"), str):
                raise NativeDriverError(
                    "REQUEST_INVALID",
                    "generic Transformers chat accepts text content only; "
                    "use a model-family Backend Adapter for multimodal input",
                )
        if not getattr(self.tokenizer, "chat_template", None):
            raise NativeDriverError(
                "OPERATION_UNSUPPORTED",
                "tokenizer has no chat template",
            )
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.completion(prompt, parameters)

    def tokenize(self, text: str) -> list[int]:
        return list(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def detokenize(self, tokens: list[int]) -> str:
        return str(
            self.tokenizer.decode(
                tokens,
                skip_special_tokens=False,
            )
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
