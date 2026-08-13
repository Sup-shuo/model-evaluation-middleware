from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_evaluation.adapters.evaluator.lm_eval import impl as lm_eval


class LmEvalResultProductTests(unittest.TestCase):
    def test_group_result_expands_groups_tasks_counts_configs_and_versions(self):
        raw = {
            "results": {
                # Real lm-eval output repeats group aggregates in results.
                # The product view must not expose that row as a task.
                "leaderboard_bbh": {
                    "alias": "BBH",
                    "acc_norm,none": 0.5,
                    "acc_norm_stderr,none": 0.1,
                },
                "bbh_boolean": {
                    "alias": "Boolean Expressions",
                    "acc_norm,none": 1.0,
                    "acc_norm_stderr,none": "N/A",
                },
                "bbh_date": {
                    "alias": "Date Understanding",
                    "acc_norm,none": 0.0,
                    "acc_norm_stderr,none": 0.0,
                },
            },
            "groups": {
                "leaderboard_bbh": {
                    "alias": "BBH",
                    "acc_norm,none": 0.5,
                    "acc_norm_stderr,none": 0.1,
                }
            },
            "group_subtasks": {
                "leaderboard_bbh": ["bbh_boolean", "bbh_date"]
            },
            "n-samples": {
                "bbh_boolean": {"original": 250, "effective": 1},
                "bbh_date": {"original": 250, "effective": 1},
            },
            "n-shot": {"bbh_boolean": 3, "bbh_date": 3},
            "versions": {"bbh_boolean": "1.0", "bbh_date": "1.0"},
            "configs": {
                "bbh_boolean": {"task": "bbh_boolean", "dataset_path": "/cache/bbh"},
                "bbh_date": {"task": "bbh_date", "dataset_path": "/cache/bbh"},
            },
            "higher_is_better": {
                "leaderboard_bbh": {"acc_norm": True},
                "bbh_boolean": {"acc_norm": True},
                "bbh_date": {"acc_norm": True},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            Path(td, "result.json").write_text(json.dumps(raw), encoding="utf-8")
            Path(td, "samples_bbh_boolean.jsonl").write_text('{"doc_id":0}\n', encoding="utf-8")
            result = lm_eval.normalize(
                {
                    "raw_result_root": td,
                    "task": {
                        "task_id": "leaderboard_bbh",
                        "protocol_fingerprint": "p",
                        "metrics": {
                            "namespace": "canonical",
                            "required": ["accuracy"],
                            "mapping": {"acc_norm,none": "accuracy"},
                        },
                    },
                    "run_metadata": {"run_id": "run", "model": "model", "benchmark": "bbh"},
                },
                {},
            )

        self.assertEqual(result["metrics"]["accuracy"]["value"], 0.5)
        self.assertEqual(result["breakdowns"]["summary"]["kind"], "group")
        self.assertEqual(result["breakdowns"]["summary"]["native_metrics"]["acc_norm,none"]["value"], 0.5)
        self.assertEqual(set(result["breakdowns"]["tasks"]), {"bbh_boolean", "bbh_date"})
        self.assertNotIn("leaderboard_bbh", result["breakdowns"]["tasks"])
        boolean = result["breakdowns"]["tasks"]["bbh_boolean"]
        self.assertEqual(boolean["canonical_metrics"]["accuracy"]["value"], 1.0)
        self.assertEqual(boolean["sample_count"], {"original": 250, "effective": 1})
        self.assertEqual(boolean["num_fewshot"], 3)
        self.assertEqual(boolean["version"], "1.0")
        self.assertEqual(boolean["config"]["dataset_path"], "/cache/bbh")
        self.assertNotIn("stderr", boolean["metrics"]["acc_norm,none"])
        group = result["breakdowns"]["groups"]["leaderboard_bbh"]
        self.assertEqual(group["subtasks"], ["bbh_boolean", "bbh_date"])
        self.assertEqual(result["sample_artifacts"][0]["media_type"], "application/x-ndjson")
        self.assertNotIn("sha256", result["sample_artifacts"][0])

    def test_single_task_result_has_stable_empty_group_table(self):
        raw = {
            "results": {"task": {"acc,none": 0.75, "acc_stderr,none": 0.0}},
            "n-samples": {"task": {"original": 100, "effective": 100}},
            "higher_is_better": {"task": {"acc": True}},
        }
        with tempfile.TemporaryDirectory() as td:
            Path(td, "result.json").write_text(json.dumps(raw), encoding="utf-8")
            result = lm_eval.normalize(
                {
                    "raw_result_root": td,
                    "task": {"task_id": "task", "protocol_fingerprint": "p"},
                    "run_metadata": {"run_id": "run", "model": "model", "benchmark": "task"},
                },
                {},
            )
        self.assertEqual(result["breakdowns"]["summary"]["kind"], "task")
        self.assertEqual(result["breakdowns"]["groups"], {})
        self.assertEqual(result["breakdowns"]["tasks"]["task"]["sample_count"]["effective"], 100)
        self.assertNotIn("sample_artifacts", result)


if __name__ == "__main__":
    unittest.main()
