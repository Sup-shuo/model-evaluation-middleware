from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_evaluation.adapters.backend.tinyllava import impl as tinyllava_impl
from model_evaluation.adapters.backend.tinyllava import preflight as tinyllava_preflight
from model_evaluation.adapters.backend.tinyllava.driver import TinyLlavaDriver
from model_evaluation.adapters.backend.transformers import impl as transformers_impl
from model_evaluation.adapters.backend.transformers import preflight as transformers_preflight
from model_evaluation.adapters.backend.transformers.driver import (
    TransformersCausalLMDriver,
)
from model_evaluation.core.registry.operation_contracts import validate_operation_output
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.sdk.http import request_json
from model_evaluation.sdk.native_backend import (
    NativeDriver,
    NativeDriverError,
    probe_native_service,
)
from model_evaluation.sdk.native_backend.protocol import (
    normalize_generation_result,
    validate_descriptor,
)
from model_evaluation.sdk.native_backend.service import NativeApplication, NativeServer


class _FakeDriver(NativeDriver):
    def load(self) -> None:
        pass

    def describe(self) -> dict:
        return {
            "model_id": "fake/model",
            "capabilities": {
                "generation": True,
                "chat": True,
                "tokenize": True,
                "detokenize": True,
            },
            "context_length": 2048,
            "metadata": {"driver": "fake"},
        }

    def completion(self, prompt: str, parameters: dict) -> dict:
        return {
            "text": prompt.upper(),
            "prompt_tokens": len(prompt),
            "completion_tokens": len(prompt),
            "duration_seconds": 0.5,
        }

    def chat(self, messages: list[dict], parameters: dict) -> dict:
        return {
            "text": str(messages[-1]["content"]),
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }

    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def detokenize(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


class NativeBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schemas = SchemaStore(ROOT / "model_evaluation" / "schemas")

    def test_native_service_and_probe_publish_truthful_contract(self):
        application = NativeApplication(
            _FakeDriver(),
            model_id="fake/model",
            max_request_bytes=4096,
            max_concurrent=1,
        )
        application.load()
        server = NativeServer(("127.0.0.1", 0), application)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            base = f"http://{host}:{port}/v1"
            descriptor = probe_native_service(
                base_url=base,
                model_id="fake/model",
                ownership="managed",
                timeout=2,
                max_concurrent=1,
            )
            validate_operation_output(
                self.schemas,
                "backend",
                "probe_service",
                descriptor,
                input_obj={
                    "attach": {
                        "model_id": "fake/model",
                        "ownership": "managed",
                        "auth": {"mode": "none"},
                    }
                },
            )
            values = descriptor["capabilities"]["values"]
            self.assertTrue(values["service.generation"])
            self.assertTrue(values["service.chat"])
            self.assertFalse(values["service.completion_logprobs"])
            self.assertIn("openai_completion", descriptor["protocols"])
            _, response = request_json(
                base + "/completions",
                method="POST",
                payload={
                    "model": "fake/model",
                    "prompt": "hello",
                    "max_tokens": 1,
                },
                timeout=2,
            )
            self.assertEqual(response["choices"][0]["text"], "HELLO")
            self.assertEqual(response["usage"]["total_tokens"], 10)
            self.assertEqual(
                response["performance"]["output_tokens_per_second"],
                10.0,
            )
        finally:
            server.shutdown()
            server.server_close()
            application.close()
            thread.join(timeout=2)

    def test_native_descriptor_rejects_unknown_capability(self):
        with self.assertRaisesRegex(NativeDriverError, "unknown capabilities"):
            validate_descriptor(
                {
                    "model_id": "m",
                    "capabilities": {
                        "generation": True,
                        "made_up": True,
                    },
                },
                model_id="m",
            )

    def test_native_protocol_rejects_non_finite_duration(self):
        with self.assertRaisesRegex(NativeDriverError, "finite"):
            normalize_generation_result(
                {"text": "ok", "duration_seconds": float("nan")}
            )

    def test_native_application_rejects_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "max_concurrent"):
            NativeApplication(
                _FakeDriver(),
                model_id="fake/model",
                max_request_bytes=4096,
                max_concurrent=0,
            )

    def _inputs(self, model_path: Path, adapter: str) -> dict:
        return {
            "model": {
                "schema_version": "1.0",
                "id": "example/model",
                "source": {"type": "local", "ref": str(model_path)},
                "context_length": 4096,
            },
            "deployment": {
                "schema_version": "1.1",
                "id": f"{adapter}-managed",
                "backend": {"adapter": adapter},
                "management": {"mode": "managed"},
                "model_location": {"local_path": str(model_path)},
                "parameters": {
                    "port": 18091,
                    "dependency_probe_timeout_seconds": 11,
                    "model_probe_timeout_seconds": 22,
                },
            },
            "platform": {},
            "endpoint": {"host": "127.0.0.1", "port": 18091},
            "log_path": str(model_path / "backend.log"),
            "network_policy": "offline",
        }

    def _assert_plans(self, impl, adapter: str, model_path: Path) -> None:
        inputs = self._inputs(model_path, adapter)
        preflight = impl.plan_preflight(inputs, {"offline": True})
        start = impl.plan_start(
            inputs,
            {"offline": True, "workspace": str(model_path.parent)},
        )
        validate_operation_output(
            self.schemas,
            "backend",
            "plan_preflight",
            preflight,
            input_obj=inputs,
        )
        validate_operation_output(
            self.schemas,
            "backend",
            "plan_start",
            start,
            input_obj=inputs,
        )
        self.assertEqual(
            [probe["phase"] for probe in preflight["probes"]],
            ["backend_dependency", "model_compatibility"],
        )
        self.assertEqual(
            [probe["process"]["timeout_seconds"] for probe in preflight["probes"]],
            [11.0, 22.0],
        )
        self.assertEqual(start["process"]["argv"][:2], ["python", "-B"])
        self.assertEqual(start["attach"]["model_id"], "example/model")
        self.assertEqual(start["attach"]["ownership"], "managed")
        self.assertEqual(start["shutdown"]["signal"], "SIGTERM")
        payload = json.loads(start["process"]["argv"][-1])
        self.assertTrue(payload["offline"])
        self.assertEqual(payload["model_path"], str(model_path))

    def test_transformers_adapter_plans_validate(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model"
            model_path.mkdir()
            self._assert_plans(transformers_impl, "transformers", model_path)

    def test_tinyllava_adapter_plans_validate(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model"
            model_path.mkdir()
            self._assert_plans(tinyllava_impl, "tinyllava", model_path)

    def test_transformers_generation_parameters_follow_openai_boundary(self):
        driver = TransformersCausalLMDriver({})
        kwargs, stops = driver._generation_kwargs(
            {
                "max_tokens": 3,
                "temperature": 0,
                "stop": ["END"],
            }
        )
        self.assertEqual(kwargs, {"max_new_tokens": 3, "do_sample": False})
        self.assertEqual(stops, ["END"])
        with self.assertRaisesRegex(NativeDriverError, "integer"):
            driver._generation_kwargs({"max_tokens": True})

    def test_transformers_rejects_model_family_architecture_mismatch(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "use a model-family Backend Adapter",
        ):
            transformers_preflight._validate_declared_architecture(
                "Qwen2ForCausalLM",
                ["LlavaQwenForCausalLM"],
            )
        self.assertEqual(
            transformers_preflight._validate_declared_architecture(
                "transformers_modules.model.CustomForCausalLM",
                ["CustomForCausalLM"],
            ),
            "transformers_modules.model.CustomForCausalLM",
        )

    def test_tinyllava_legacy_checkpoint_fails_before_dependency_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "tinyllava",
                        "architectures": ["TinyLlavaLlamaForCausalLM"],
                    }
                ),
                encoding="utf-8",
            )
            result = tinyllava_preflight._result(
                {
                    "model_path": str(model_path),
                    "trust_remote_code": True,
                }
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["error"]["code"],
            "MODEL_FAMILY_RUNTIME_UNAVAILABLE",
        )

    def test_tinyllava_request_boundary_is_single_turn_image(self):
        prompt, image = TinyLlavaDriver._text_and_image(
            [
                {"role": "system", "content": "Answer briefly."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "/tmp/a.png"}},
                        {"type": "text", "text": "What is shown?"},
                    ],
                },
            ]
        )
        self.assertEqual(prompt, "Answer briefly.\nWhat is shown?")
        self.assertEqual(image, "/tmp/a.png")
        with self.assertRaisesRegex(NativeDriverError, "exactly one user"):
            TinyLlavaDriver._text_and_image([])
        with self.assertRaisesRegex(NativeDriverError, "at most one system"):
            TinyLlavaDriver._text_and_image(
                [
                    {"role": "system", "content": "first"},
                    {"role": "system", "content": "second"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "question"},
                            {"type": "image_url", "image_url": "/tmp/a.png"},
                        ],
                    },
                ]
            )

    def test_tinyllava_generation_parameters_are_strictly_typed(self):
        driver = TinyLlavaDriver({})
        self.assertEqual(
            driver._parameters(
                {"max_tokens": 8, "num_beams": 2, "temperature": 0}
            ),
            {"max_new_tokens": 8, "num_beams": 2, "temperature": 0.0},
        )
        for invalid in (
            {"max_tokens": 0},
            {"max_tokens": True},
            {"num_beams": 0},
            {"temperature": float("inf")},
            {"top_p": 1.1},
        ):
            with self.subTest(parameters=invalid):
                with self.assertRaises(NativeDriverError):
                    driver._parameters(invalid)


if __name__ == "__main__":
    unittest.main()
