from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from model_evaluation.adapters.binding.evalscope import impl as binding
from model_evaluation.adapters.evaluator.evalscope import impl as evaluator
from model_evaluation.core.schema.validator import SchemaStore


class EvalScopeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas = SchemaStore(Path(__file__).resolve().parents[1] / "model_evaluation" / "schemas")

    def _task(self) -> dict:
        benchmark = {
            "schema_version": "1.0",
            "id": "fixture_eval",
            "dataset": {"provider": "virtual", "revision": "evalscope-native"},
            "protocol": {
                "task": "fixture_eval",
                "fewshot": 0,
                "inference": ["generation"],
                "evalscope_dataset_args": {"subset_list": ["main"]},
            },
            "metrics": ["accuracy"],
        }
        dataset = {
            "fingerprint": "f" * 64,
            "materialization": {"kind": "virtual"},
        }
        evaluation = {
            "parameters": {
                "metric_maps": {"fixture_eval": {"AverageAccuracy": "accuracy"}}
            }
        }
        task = binding.build_task(
            {"benchmark": benchmark, "dataset_artifact": dataset, "evaluation": evaluation},
            {},
        )
        fingerprint = binding.protocol_fingerprint(
            {"benchmark": benchmark, "dataset_artifact": dataset, "evaluation": evaluation},
            {},
        )
        self.assertEqual(task["protocol_fingerprint"], fingerprint["protocol_fingerprint"])
        self.schemas.validate("framework_task_artifact", task)
        return task

    def test_binding_builds_framework_task_and_metric_contract(self) -> None:
        task = self._task()
        self.assertEqual(task["framework"], "evalscope")
        self.assertEqual(task["task_id"], "fixture_eval")
        self.assertEqual(task["metrics"]["namespace"], "canonical")
        self.assertEqual(task["metrics"]["mapping"], {"AverageAccuracy": "accuracy"})
        self.assertEqual(
            task["metadata"]["evalscope_dataset_args"], {"subset_list": ["main"]}
        )

    def test_requirements_use_chat_service_and_python_environment(self) -> None:
        output = evaluator.requirements({"task": self._task()}, {})
        self.schemas.validate("requirement_set", output)
        paths = {item["path"] for item in output["requirements"]}
        self.assertEqual(
            paths,
            {
                "service.protocol.openai_chat",
                "service.generation",
                "evaluation_environment.python",
            },
        )

    def test_plan_preflight_is_package_based_and_has_no_source_root(self) -> None:
        plan = evaluator.plan_preflight(
            {
                "evaluation": {
                    "parameters": {
                        "executable": "/opt/eval/bin/evalscope",
                        "expected_version": "1.5.2",
                    }
                },
                "task": self._task(),
                "cache_root": "/cache",
            },
            {"offline": True},
        )
        self.assertEqual(plan["result_format"], "preflight_result")
        payload = json.loads(plan["process"]["argv"][-1])
        self.assertEqual(payload["expected_version"], "1.5.2")
        self.assertNotIn("tool_root", payload)
        self.assertEqual(plan["process"]["env_patch"]["set"]["HF_HUB_OFFLINE"], "1")
        self.schemas.validate("process_spec", plan["process"])

    def test_plan_evaluate_uses_existing_openai_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = evaluator.plan_evaluate(
                {
                    "service": {
                        "model": {"id": "served-model"},
                        "protocols": {
                            "openai_chat": {
                                "url": "http://127.0.0.1:8091/v1/chat/completions"
                            }
                        },
                        "auth": {"mode": "none"},
                    },
                    "task": self._task(),
                    "evaluation": {
                        "parameters": {
                            "executable": "evalscope",
                            "eval_batch_size": 4,
                            "limit": 5,
                            "seed": 7,
                        }
                    },
                    "cache_root": str(root / "cache"),
                    "output_root": str(root / "output"),
                    "log_path": str(root / "evaluation.log"),
                    "network_policy": "offline",
                },
                {},
            )
            argv = plan["process"]["argv"]
            self.assertEqual(argv[:2], ["evalscope", "eval"])
            self.assertEqual(argv[argv.index("--api-url") + 1], "http://127.0.0.1:8091/v1")
            self.assertEqual(argv[argv.index("--datasets") + 1], "fixture_eval")
            self.assertEqual(argv[argv.index("--limit") + 1], "5")
            self.assertIn("--no-timestamp", argv)
            self.assertNotIn("vllm", argv)
            self.assertEqual(plan["raw_result_root"], str((root / "output").resolve()))
            self.schemas.validate("process_spec", plan["process"])

    def test_plan_evaluate_rejects_secret_bearing_service_in_v1(self) -> None:
        with self.assertRaisesRegex(Exception, "unauthenticated"):
            evaluator.plan_evaluate(
                {
                    "service": {
                        "model": {"id": "m"},
                        "protocols": {
                            "openai_chat": {
                                "url": "http://127.0.0.1:8091/v1/chat/completions"
                            }
                        },
                        "auth": {"mode": "bearer", "secret_ref": "env:API_KEY"},
                    },
                    "task": self._task(),
                    "evaluation": {},
                    "output_root": "/tmp/output",
                },
                {},
            )

    def test_normalize_maps_report_and_sample_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "reports" / "served-model" / "fixture_eval.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps(
                    {
                        "name": "served-model@fixture_eval",
                        "model_name": "served-model",
                        "dataset_name": "fixture_eval",
                        "score": 0.6,
                        "metrics": [
                        {
                            "name": "AverageAccuracy",
                            "num": 5,
                            "score": 0.6,
                            "categories": [
                                {
                                    "name": ["default"],
                                    "num": 5,
                                    "score": 0.6,
                                    "subsets": [
                                        {"name": "main", "num": 5, "score": 0.6}
                                    ],
                                }
                            ],
                        }
                        ],
                        "analysis": "N/A",
                    }
                ),
                encoding="utf-8",
            )
            prediction = root / "predictions" / "served-model" / "fixture_eval.jsonl"
            prediction.parent.mkdir(parents=True)
            prediction.write_text('{"sample":1}\n', encoding="utf-8")
            review = root / "reviews" / "served-model" / "fixture_eval.jsonl"
            review.parent.mkdir(parents=True)
            review.write_text('{"reviewed":true}\n', encoding="utf-8")

            result = evaluator.normalize(
                {
                    "raw_result_root": str(root),
                    "task": self._task(),
                    "run_metadata": {
                        "run_id": "run-evalscope",
                        "model": "catalog-model",
                        "benchmark": "fixture_eval",
                    },
                },
                {},
            )
            self.assertEqual(result["framework"], "evalscope")
            self.assertEqual(result["metrics"], {"accuracy": {"value": 0.6}})
            self.assertEqual(len(result["breakdowns"]["tasks"]), 1)
            detail = next(iter(result["breakdowns"]["tasks"].values()))
            self.assertEqual(detail["sample_count"], {"original": 5, "effective": 5})
            self.assertEqual(len(result["sample_artifacts"]), 2)
            self.assertEqual(Path(result["raw_result"]["path"]), report.resolve())
            self.schemas.validate("canonical_result", result)

    def test_normalize_rejects_unknown_report_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "reports" / "fixture_eval.json"
            report.parent.mkdir(parents=True)
            report.write_text('{"unexpected":true}', encoding="utf-8")
            with self.assertRaisesRegex(Exception, "unsupported EvalScope report"):
                evaluator.normalize(
                    {
                        "raw_result_root": str(root),
                        "task": self._task(),
                        "run_metadata": {
                            "run_id": "run-evalscope",
                            "model": "catalog-model",
                            "benchmark": "fixture_eval",
                        },
                    },
                    {},
                )

    def test_checked_in_evalscope_presets_are_schema_valid(self) -> None:
        package_root = Path(__file__).resolve().parents[1] / "model_evaluation"
        benchmark = yaml.safe_load(
            (package_root / "presets" / "benchmarks" / "evalscope_gsm8k_smoke.yaml").read_text()
        )
        evaluation = yaml.safe_load(
            (package_root / "presets" / "evaluations" / "evalscope_current.yaml").read_text()
        )
        self.schemas.validate("benchmark_spec", benchmark)
        self.schemas.validate("evaluation_profile", evaluation)
        self.assertEqual(benchmark["bindings"], {"evalscope": "evalscope"})
        self.assertEqual(
            evaluation["parameters"]["metric_maps"]["evalscope_gsm8k_smoke"],
            {"AverageAccuracy": "accuracy"},
        )


if __name__ == "__main__":
    unittest.main()
