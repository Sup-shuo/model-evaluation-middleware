from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from model_evaluation.adapters.evaluator.lm_eval import transport_proxy
from model_evaluation.adapters.evaluator.lm_eval.usage import UsageAccumulator
from model_evaluation.core.performance import (
    build_performance,
    normalize_token_usage,
    validate_performance_values,
)
from model_evaluation.core.schema.validator import SchemaStore


ROOT = Path(__file__).resolve().parents[1]


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.schemas = SchemaStore(ROOT / "model_evaluation" / "schemas")

    def test_exact_usage_builds_end_to_end_evaluator_throughput(self):
        performance = build_performance(
            run_id="run",
            model="model",
            benchmark="bbh",
            framework="lm_eval",
            evaluation_started_at="2026-08-29T12:00:00+08:00",
            evaluation_finished_at="2026-08-29T12:00:10+08:00",
            evaluation_duration_seconds=10.0,
            token_usage={
                "source": "openai_response_usage",
                "complete": True,
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
            },
        )
        self.schemas.validate("performance", performance)
        validate_performance_values(performance)
        self.assertEqual(performance["tokens_per_second"], 10.0)

    def test_missing_or_inconsistent_usage_is_never_inferred(self):
        missing = normalize_token_usage(None)
        self.assertFalse(missing["complete"])
        self.assertIsNone(missing["total_tokens"])

        inconsistent = normalize_token_usage(
            {
                "source": "openai_response_usage",
                "complete": True,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 99,
            }
        )
        self.assertFalse(inconsistent["complete"])
        self.assertIsNone(inconsistent["prompt_tokens"])
        self.assertIsNone(inconsistent["completion_tokens"])
        self.assertIsNone(inconsistent["total_tokens"])

    def test_usage_accumulator_sums_only_complete_response_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "token_usage.json"
            accumulator = UsageAccumulator(path)
            accumulator.observe(
                json.dumps(
                    {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        }
                    }
                ).encode()
            )
            accumulator.observe(
                json.dumps(
                    {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 3,
                            "total_tokens": 23,
                        }
                    }
                ).encode()
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(saved["complete"])
            self.assertEqual(saved["successful_responses"], 2)
            self.assertEqual(saved["prompt_tokens"], 30)
            self.assertEqual(saved["completion_tokens"], 5)
            self.assertEqual(saved["total_tokens"], 35)

            accumulator.observe(b'{"choices":[]}')
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(saved["complete"])
            self.assertIsNone(saved["prompt_tokens"])
            self.assertIsNone(saved["completion_tokens"])
            self.assertIsNone(saved["total_tokens"])

    def test_transport_proxy_preserves_response_and_records_usage(self):
        response_body = json.dumps(
            {
                "choices": [{"text": "ok"}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 2,
                    "total_tokens": 13,
                },
            }
        ).encode()

        class UpstreamHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return None

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = threading.Thread(
            target=upstream.serve_forever,
            daemon=True,
        )
        upstream_thread.start()
        with tempfile.TemporaryDirectory() as temp:
            usage_path = Path(temp) / "token_usage.json"
            namespace = SimpleNamespace(
                completion_url=(
                    f"http://127.0.0.1:{upstream.server_address[1]}/v1/completions"
                ),
                tokenizer_info_url=None,
                tokenize_url=None,
                detokenize_url=None,
                auth_mode="strip",
                timeout=2.0,
                max_body_bytes=1024 * 1024,
            )
            routes = transport_proxy._validated_routes(namespace)
            proxy = transport_proxy.Server(
                ("127.0.0.1", 0),
                transport_proxy._handler(
                    namespace,
                    routes,
                    "",
                    UsageAccumulator(usage_path),
                ),
            )
            proxy_thread = threading.Thread(
                target=proxy.serve_forever,
                daemon=True,
            )
            proxy_thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.server_address[1]}/v1/completions",
                    data=b'{"model":"m","prompt":"hello"}',
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    self.assertEqual(response.read(), response_body)
                saved = json.loads(usage_path.read_text(encoding="utf-8"))
                self.assertTrue(saved["complete"])
                self.assertEqual(saved["prompt_tokens"], 11)
                self.assertEqual(saved["completion_tokens"], 2)
                self.assertEqual(saved["total_tokens"], 13)
            finally:
                proxy.shutdown()
                proxy.server_close()
                upstream.shutdown()
                upstream.server_close()

    def test_usage_write_failure_does_not_change_proxy_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            blocked_parent = Path(temp) / "not-a-directory"
            blocked_parent.write_text("blocked", encoding="utf-8")
            accumulator = UsageAccumulator(blocked_parent / "token_usage.json")
            accumulator.observe(
                b'{"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
            )
            self.assertTrue(accumulator.snapshot()["complete"])


if __name__ == "__main__":
    unittest.main()
