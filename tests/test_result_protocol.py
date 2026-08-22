from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_evaluation.core.errors import ResultProductError, SchemaValidationError
from model_evaluation.core.result_product import inspect_run_product
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.results import load_run


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_reproduction_records(
    run: Path,
    *,
    model: str = "model",
    benchmark: str = "bbh",
    framework: str = "lm_eval",
    started_at: str = "2026-08-15T12:00:00+08:00",
    include_runtime: bool = True,
) -> None:
    adapters = [
        {"kind": "backend", "name": "reference", "version": "1.0.0"},
        {"kind": "evaluator", "name": framework, "version": "1.0.0"},
    ]
    selection = {
        "schema_version": "1.0",
        "model": model,
        "platform": "test-platform",
        "deployment": "test-backend",
        "benchmark": benchmark,
        "evaluation": "test-evaluation",
    }
    evaluation_environment = {
        "schema_version": "1.0",
        "provider": "current",
        "identity": "test-controller",
        "python": "/usr/bin/python3",
        "capabilities": {"schema_version": "1.0", "values": {}},
    }
    run_config = {
        "schema_version": "1.0",
        "run_id": run.name,
        "plan_id": "plan-" + "a" * 24,
        "started_at": started_at,
        "timezone": "Asia/Shanghai",
        "adapters": adapters,
        "selection": selection,
        "model": {
            "schema_version": "1.0",
            "id": model,
            "source": {"type": "local", "ref": f"/models/{model}"},
            "experiment_id": model,
        },
        "benchmark": {
            "schema_version": "1.0",
            "id": benchmark,
            "dataset": {"provider": "virtual"},
            "protocol": {},
            "metrics": ["accuracy"],
        },
        "backend": {
            "schema_version": "1.1",
            "id": "test-backend",
            "backend": {"adapter": "reference"},
            "management": {"mode": "external"},
        },
        "evaluator": {
            "schema_version": "1.1",
            "id": "test-evaluation",
            "framework": {"adapter": framework},
            "binding": {"adapter": "reference_eval"},
        },
        "system": {
            "schema_version": "1.1",
            "id": "test-platform",
            "evaluation_environment": {
                "provider": "current",
                "profile": "current",
            },
        },
        "resolved_runtime": {
            "device_probe_skipped": True,
            "runtime_probe_skipped": True,
            "reason": "backend is not locally managed",
            "evaluation_environment": evaluation_environment,
        },
    }
    _write(run / "config" / "run_config.json", run_config)
    if include_runtime:
        _write(
            run / "config" / "runtime_versions.json",
            {
                "schema_version": "1.0",
                "adapters": adapters,
                "environments": {"evaluator": evaluation_environment},
                "backend": {
                    "adapter": "reference",
                    "adapter_version": "1.0.0",
                    "management": "external",
                },
                "evaluator": {
                    "adapter": framework,
                    "adapter_version": "1.0.0",
                },
            },
        )


class ResultProtocolTests(unittest.TestCase):
    def setUp(self):
        self.schemas = SchemaStore(ROOT / "model_evaluation" / "schemas")

    def _success(self, root: Path) -> Path:
        run = root / "model_bbh_20260815-120000"
        (run / "raw").mkdir(parents=True)
        (run / "samples").mkdir()
        (run / "raw" / "framework_result.json").write_text("{}\n", encoding="utf-8")
        (run / "samples" / "task.jsonl").write_text("{}\n", encoding="utf-8")
        metric = {"accuracy": {"value": 0.5, "higher_is_better": True}}
        detail = {
            "metrics": {"acc_norm,none": {"value": 0.5}},
            "sample_count": {"original": 10, "effective": 10},
        }
        result = {
            "schema_version": "1.0",
            "run_id": run.name,
            "model": "model",
            "benchmark": "bbh",
            "framework": "lm_eval",
            "metrics": metric,
            "raw_result": {"path": "raw/framework_result.json", "media_type": "application/json"},
            "breakdowns": {
                "summary": {
                    "id": "bbh",
                    "kind": "group",
                    "metric_namespace": "canonical",
                    "metrics": metric,
                },
                "groups": {"bbh": detail},
                "tasks": {"task": detail},
            },
            "sample_artifacts": [
                {"path": "samples/task.jsonl", "media_type": "application/x-ndjson"}
            ],
            "metadata": {
                "started_at": "2026-08-15T12:00:00+08:00",
                "finished_at": "2026-08-15T12:01:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        }
        metrics = {
            "schema_version": "1.0",
            "run_id": run.name,
            "model": "model",
            "benchmark": "bbh",
            "framework": "lm_eval",
            "summary": metric,
            "groups": {"bbh": detail},
            "tasks": {"task": detail},
        }
        terminal = {
            "schema_version": "1.0",
            "run_id": run.name,
            "outcome": "success",
            "started_at": "2026-08-15T12:00:00+08:00",
            "finished_at": "2026-08-15T12:01:00+08:00",
            "timezone": "Asia/Shanghai",
            "cleanup": {"status": "clean", "backend": {"status": "clean"}},
        }
        _write(run / "result.json", result)
        _write(run / "metrics.json", metrics)
        _write(run / "terminal.json", terminal)
        _write_reproduction_records(run)
        return run

    def test_success_product_schema_and_consistency(self):
        with tempfile.TemporaryDirectory() as td:
            report = inspect_run_product(self._success(Path(td)), self.schemas)
            self.assertEqual(report["outcome"], "success")
            self.assertEqual(report["tasks"], 1)
            self.assertEqual(report["effective_samples"], 10)
            self.assertEqual(report["artifacts"], 2)
            self.assertTrue(report["runtime_recorded"])

    def test_success_product_requires_run_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            (run / "config" / "run_config.json").unlink()
            with self.assertRaisesRegex(ResultProductError, "run_config.json"):
                inspect_run_product(run, self.schemas)

    def test_success_product_requires_runtime_versions(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            (run / "config" / "runtime_versions.json").unlink()
            with self.assertRaisesRegex(ResultProductError, "runtime_versions.json"):
                inspect_run_product(run, self.schemas)

    def test_result_identity_must_match_run_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            config_path = run / "config" / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["model"]["experiment_id"] = "different-model"
            _write(config_path, config)
            with self.assertRaisesRegex(ResultProductError, "result.model"):
                inspect_run_product(run, self.schemas)

    def test_checked_in_sanitized_result_example_is_a_valid_product(self):
        run = (
            ROOT
            / "examples"
            / "result_example"
            / "cpu_example-model_reference_example-benchmark_260101-1200"
        )
        report = inspect_run_product(run, self.schemas)
        self.assertEqual(report["outcome"], "success")
        self.assertEqual(report["model"], "example-model")
        self.assertEqual(report["summary"]["accuracy"]["value"], 0.75)
        self.assertEqual(report["tasks"], 2)
        self.assertEqual(report["effective_samples"], 4)
        self.assertEqual(report["artifacts"], 3)

    def test_python_result_sdk_exposes_stable_read_only_views(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            runtime_path = run / "config" / "runtime_versions.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["backend"]["version"] = "1.0"
            _write(runtime_path, runtime)
            loaded = load_run(run, schemas=self.schemas)
            self.assertEqual(loaded.run_id, run.name)
            self.assertEqual(loaded.outcome, "success")
            self.assertEqual(loaded.metrics.summary()["accuracy"]["value"], 0.5)
            self.assertEqual(set(loaded.metrics.tasks()), {"task"})
            self.assertEqual(
                loaded.runtime()["runtime_versions"]["backend"]["version"],
                "1.0",
            )
            artifacts = loaded.artifacts()
            self.assertEqual([artifact.kind for artifact in artifacts], ["raw", "sample"])
            self.assertTrue(all(artifact.path.is_file() for artifact in artifacts))
            summary = loaded.metrics.summary()
            summary["accuracy"]["value"] = 0.0
            self.assertEqual(loaded.metrics.summary()["accuracy"]["value"], 0.5)

    def test_python_result_sdk_preserves_run_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = self._success(root)
            link = root / "linked-run"
            link.symlink_to(run, target_is_directory=True)
            with self.assertRaisesRegex(ResultProductError, "may not be a symlink"):
                load_run(link, schemas=self.schemas)

    def test_result_completion_may_precede_terminal_finalization(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            result["metadata"]["finished_at"] = "2026-08-15T12:00:55+08:00"
            _write(run / "result.json", result)
            report = inspect_run_product(run, self.schemas)
            self.assertEqual(report["outcome"], "success")

    def test_result_completion_after_terminal_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            result["metadata"]["finished_at"] = "2026-08-15T12:01:01+08:00"
            _write(run / "result.json", result)
            with self.assertRaisesRegex(ResultProductError, "metadata.finished_at"):
                inspect_run_product(run, self.schemas)

    def test_cross_file_metric_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            metrics["summary"]["accuracy"]["value"] = 0.1
            _write(run / "metrics.json", metrics)
            with self.assertRaisesRegex(ResultProductError, "summary metrics"):
                inspect_run_product(run, self.schemas)

    def test_breakdown_summary_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            result["breakdowns"]["summary"]["metrics"]["accuracy"]["value"] = 0.1
            _write(run / "result.json", result)
            with self.assertRaisesRegex(ResultProductError, "breakdown summary metrics"):
                inspect_run_product(run, self.schemas)

    def test_success_requires_clean_cleanup_and_no_error(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            terminal = json.loads((run / "terminal.json").read_text(encoding="utf-8"))
            terminal["cleanup"]["status"] = "incomplete"
            _write(run / "terminal.json", terminal)
            with self.assertRaises(SchemaValidationError):
                inspect_run_product(run, self.schemas)

            terminal["cleanup"]["status"] = "clean"
            terminal["error"] = {
                "type": "ProcessError",
                "code": "PROCESS_ERROR",
                "message": "contradiction",
            }
            _write(run / "terminal.json", terminal)
            with self.assertRaises(SchemaValidationError):
                inspect_run_product(run, self.schemas)

    def test_artifact_escape_is_rejected_by_schema(self):
        with tempfile.TemporaryDirectory() as td:
            run = self._success(Path(td))
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            result["raw_result"]["path"] = "../outside.json"
            _write(run / "result.json", result)
            with self.assertRaises(SchemaValidationError):
                inspect_run_product(run, self.schemas)

    def test_failed_product_requires_matching_failure(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "failed_bbh_20260815-120000"
            (run / "logs").mkdir(parents=True)
            (run / "logs" / "core_error.log").write_text("failed\n", encoding="utf-8")
            error = {"type": "ProcessError", "code": "PROCESS_ERROR", "message": "failed"}
            cleanup = {"status": "clean", "backend": {"status": "not_started"}}
            terminal = {
                "schema_version": "1.0",
                "run_id": run.name,
                "outcome": "failed",
                "started_at": "2026-08-15T12:00:00+08:00",
                "finished_at": "2026-08-15T12:00:01+08:00",
                "timezone": "Asia/Shanghai",
                "cleanup": cleanup,
                "error": error,
            }
            failure = {
                "schema_version": "1.0",
                "run_id": run.name,
                "time": "2026-08-15T12:00:01+08:00",
                "stage": "EVALUATING",
                "primary_error": error,
                "cleanup": cleanup,
                "logs": {"core": {"path": "logs/core_error.log", "tail": ["failed"]}},
            }
            _write(run / "terminal.json", terminal)
            _write(run / "failure.json", failure)
            _write_reproduction_records(run, include_runtime=False)
            report = inspect_run_product(run, self.schemas)
            self.assertEqual(report["outcome"], "failed")
            self.assertEqual(report["failure_stage"], "EVALUATING")

            failure["primary_error"]["message"] = "different failure"
            _write(run / "failure.json", failure)
            with self.assertRaisesRegex(ResultProductError, "failure error"):
                inspect_run_product(run, self.schemas)


if __name__ == "__main__":
    unittest.main()
