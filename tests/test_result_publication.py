from __future__ import annotations

import json
import tempfile
import unittest
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_evaluation.core.results import allocate_run_dir, publish_result, run_id_base


class ResultPublicationTests(unittest.TestCase):
    @staticmethod
    def _plan() -> dict:
        return {
            "run_spec": {
                "model": "internal-model",
                "platform": "internal-platform",
                "deployment": "internal-deployment",
                "benchmark": "bbh",
            },
            "resolved": {
                "specs": {
                    "model": {"experiment_id": "qwen35-08b-base"},
                    "platform": {
                        "device": {"adapter": "nvidia"},
                        "metadata": {
                            "timezone": "Asia/Shanghai",
                            "result_platform": "a100",
                        },
                    },
                    "deployment": {"backend": {"adapter": "vllm"}},
                }
            },
        }

    def test_short_beijing_run_name_and_same_second_suffix(self):
        when = datetime(2026, 8, 13, 2, 19, 20, tzinfo=ZoneInfo("Asia/Shanghai"))
        base = run_id_base(self._plan(), when=when)
        self.assertEqual(base, "a100_qwen35-08b-base_vllm_bbh_260813-0219")
        with tempfile.TemporaryDirectory() as td:
            with patch("model_evaluation.core.results.run_id_base", return_value=base):
                first = allocate_run_dir(td, self._plan())
                second = allocate_run_dir(td, self._plan())
            self.assertEqual(first.name, base)
            self.assertEqual(second.name, f"{base}-2")

    def test_publish_result_keeps_full_metrics_raw_and_optional_samples(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "run"; raw_root = run / ".run" / "framework_output"
            raw_root.mkdir(parents=True)
            raw = raw_root / "result.json"; raw.write_text('{"results":{"task":{}}}\n', encoding="utf-8")
            sample = raw_root / "task.jsonl"; sample.write_text('{"doc_id":1}\n', encoding="utf-8")
            result = {
                "schema_version": "1.0", "run_id": "run", "model": "m", "benchmark": "bbh",
                "framework": "lm_eval", "metrics": {"accuracy": {"value": 0.5}},
                "raw_result": {"path": str(raw), "media_type": "application/json"},
                "breakdowns": {
                    "summary": {"id": "bbh", "kind": "group", "metric_namespace": "canonical", "metrics": {"accuracy": {"value": 0.5}}},
                    "groups": {"bbh": {"metrics": {"acc_norm,none": {"value": 0.5}}}},
                    "tasks": {"task": {"metrics": {"acc_norm,none": {"value": 1.0}}, "sample_count": {"effective": 1}}},
                },
                "sample_artifacts": [{"path": str(sample), "media_type": "application/x-ndjson"}],
            }
            published = publish_result(run, raw_root, result)
            metrics = json.loads((run / "metrics.json").read_text())
            saved = json.loads((run / "result.json").read_text())

            self.assertEqual(metrics["tasks"]["task"]["sample_count"]["effective"], 1)
            self.assertEqual(metrics["framework"], "lm_eval")
            self.assertEqual(saved["raw_result"]["path"], "raw/framework_result.json")
            self.assertNotIn("sha256", saved["raw_result"])
            self.assertEqual(published["sample_artifacts"][0]["path"], "samples/task.jsonl")
            self.assertTrue((run / "raw" / "framework_result.json").is_file())
            self.assertEqual((run / "raw" / "framework_result.json").read_bytes(), raw.read_bytes())
            self.assertTrue((run / "samples" / "task.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
