from __future__ import annotations

import json
import tempfile
import unittest
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_evaluation.core.matrix import MatrixExecutor
from model_evaluation.core.result_relocation import ResultRelocationMap, load_result_relocation


class _CanonicalSchemas:
    def validate(self, name: str, obj: object) -> None:
        if name != "canonical_result":
            raise AssertionError(f"unexpected schema: {name}")
        if not isinstance(obj, dict) or obj.get("schema_version") != "1.0":
            raise ValueError("invalid result")


def _plan(*, plan_id: str = "plan-1", timezone_name: str | None = None) -> dict:
    metadata = {} if timezone_name is None else {"timezone": timezone_name}
    return {
        "plan_id": plan_id,
        "run_spec": {
            "model": "model-catalog-entry",
            "benchmark": "bbh",
            "platform": "system",
            "deployment": "vllm",
            "evaluation": "lm-eval",
        },
        "resolved": {
            "specs": {
                "model": {"id": "model-runtime-id", "experiment_id": "model-catalog-entry"},
                "platform": {"metadata": metadata},
                "evaluation": {"framework": {"adapter": "lm_eval"}},
            }
        },
    }


def _result(run_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "model": "model-catalog-entry",
        "benchmark": "bbh",
        "framework": "lm_eval",
        "metrics": {
            "accuracy": {"value": 0.75, "stderr": 0.02, "higher_is_better": True}
        },
        # Neither this referenced file nor a SHA manifest is required by Batch.
        "raw_result": {"path": "raw/framework_result.json"},
        "breakdowns": {
            "summary": {
                "id": "bbh",
                "kind": "group",
                "metric_namespace": "canonical",
                "metrics": {"accuracy": {"value": 0.75}},
            },
            "groups": {
                "bbh": {
                    "label": "BBH",
                    "metrics": {"acc_norm,none": {"value": 0.75, "stderr": 0.02}},
                    "canonical_metrics": {"accuracy": {"value": 0.75}},
                    "subtasks": ["boolean_expressions"],
                }
            },
            "tasks": {
                "boolean_expressions": {
                    "label": "Boolean Expressions",
                    "metrics": {"acc_norm,none": {"value": 1.0}},
                    "canonical_metrics": {"accuracy": {"value": 1.0}},
                    "sample_count": {"original": 250, "effective": 1},
                    "num_fewshot": 3,
                    "version": "1.0",
                    "config": {"dataset_path": "/cache/bbh", "tag": "a\tb"},
                }
            },
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _executor(results_root: Path) -> MatrixExecutor:
    executor = object.__new__(MatrixExecutor)
    executor.results_root = results_root.resolve()
    executor.result_relocation = ResultRelocationMap(executor.results_root)
    executor.app = SimpleNamespace(schemas=_CanonicalSchemas())
    return executor


class BatchProductTests(unittest.TestCase):
    def test_new_run_product_is_consumed_without_sha_or_raw_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            run_dir = root / "run-new"
            _write_json(run_dir / "result.json", _result(run_dir.name))
            _write_json(run_dir / "terminal.json", {"outcome": "success", "warnings": []})
            _write_json(run_dir / "config" / "run_config.json", {"plan_id": "plan-1"})
            (run_dir / "SHA256SUMS").write_text("intentionally stale\n", encoding="utf-8")

            ok, reason, result = _executor(root)._validate_success_record(
                {"run_dir": str(run_dir), "result_path": str(run_dir / "result.json")},
                _plan(),
            )

        self.assertTrue(ok, reason)
        self.assertEqual(result["metrics"]["accuracy"]["value"], 0.75)

    def test_legacy_run_product_supports_relocation_without_sha(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve()
            old_root = base / "old-results"
            current = base / "current-results"
            current.mkdir()
            run_dir = current / "run-legacy"
            old_run = old_root / run_dir.name
            _write_json(run_dir / "canonical_result.json", _result(run_dir.name))
            _write_json(run_dir / "terminal_record.json", {"outcome": "success"})
            _write_json(run_dir / "config" / "execution_plan.json", {"plan_id": "plan-1"})
            _write_json(
                current / "RELOCATION.json",
                {"schema_version": "1.0", "mappings": [{"old_root": str(old_root), "new_root": str(current)}]},
            )
            executor = _executor(current)
            executor.result_relocation = load_result_relocation(current)

            ok, reason, result = executor._validate_success_record(
                {"run_dir": str(old_run), "canonical_result_path": str(old_run / "canonical_result.json")},
                _plan(),
            )

        self.assertTrue(ok, reason)
        self.assertEqual(result["run_id"], "run-legacy")

    def test_batch_id_uses_system_timezone_and_collision_suffix(self) -> None:
        instant = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(MatrixExecutor._batch_id({"plans": [_plan()]}, when=instant), "20260101-080000")
        matrix = {"plans": [_plan(timezone_name="America/Los_Angeles")]}
        self.assertEqual(MatrixExecutor._batch_id(matrix, when=instant), "20251231-160000")

        with tempfile.TemporaryDirectory() as td:
            executor = _executor(Path(td))
            first = Path(td) / "_batches" / "20251231-160000"
            first.mkdir(parents=True)
            allocated = executor._allocate_batch_dir(matrix, when=instant)
            self.assertEqual(allocated.name, "20251231-160000-2")

    def test_finalize_writes_only_lightweight_public_product_and_detail_tables(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            run_dir = root / "run-new"
            batch_dir = root / "_batches" / "20260101-080000"
            batch_dir.mkdir(parents=True)
            _write_json(run_dir / "result.json", _result(run_dir.name))
            _write_json(run_dir / "terminal.json", {"outcome": "success"})
            _write_json(run_dir / "config" / "run_config.json", {"plan_id": "plan-1"})
            plan = _plan()
            matrix_plan = {"matrix_id": "matrix-1", "plans": [plan]}
            status = {
                "plan-1": {
                    "index": 1,
                    "plan_id": "plan-1",
                    "model_id": "model-experiment-id",
                    "model_label": "Model Label",
                    "model_ref": "Org/Model",
                    "benchmark": "bbh",
                    "platform": "system",
                    "deployment": "vllm",
                    "evaluation": "lm-eval",
                    "status": "success",
                    "attempts": 1,
                    "run_dir": str(run_dir),
                    "result_path": str(run_dir / "result.json"),
                }
            }

            summary = _executor(root)._finalize_batch(
                batch_dir, matrix_plan, status, hard_stop=False, keep_going=False
            )

            public = {"summary.json", "runs.json", "metrics.tsv", "group_metrics.tsv", "task_metrics.tsv"}
            self.assertTrue(all((batch_dir / name).is_file() for name in public))
            self.assertFalse((batch_dir / "SHA256SUMS").exists())
            self.assertFalse((batch_dir / "batch_terminal_record.json").exists())
            self.assertFalse((batch_dir / "canonical_results.json").exists())
            runs = json.loads((batch_dir / "runs.json").read_text(encoding="utf-8"))
            groups = (batch_dir / "group_metrics.tsv").read_text(encoding="utf-8")
            tasks = (batch_dir / "task_metrics.tsv").read_text(encoding="utf-8")

        self.assertEqual(summary["outcome"], "success")
        self.assertEqual(runs[0]["result_path"], str(run_dir / "result.json"))
        self.assertIn("framework_native\tacc_norm,none\t0.75", groups)
        self.assertIn("canonical\taccuracy\t0.75", groups)
        self.assertIn("boolean_expressions\tBoolean Expressions", tasks)
        self.assertIn("\t250\t1\t3\t1.0\t", tasks)


if __name__ == "__main__":
    unittest.main()
