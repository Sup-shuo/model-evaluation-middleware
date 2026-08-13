from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("print_result", ROOT / "scripts" / "print_result.py")
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ResultReportTests(unittest.TestCase):
    def test_saved_result_prints_tasks_environment_and_svg(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "results" / "model_bbh_20260813-120000"
            (run / "config").mkdir(parents=True)
            values = {
                "metrics.json": {
                    "model": "model",
                    "benchmark": "bbh",
                    "framework": "lm_eval",
                    "summary": {"accuracy": {"value": 0.5}},
                    "groups": {},
                    "tasks": {
                        "leaderboard_bbh_local_boolean_expressions": {
                            "version": 1,
                            "num_fewshot": 3,
                            "metrics": {"acc_norm,none": {"value": 1.0, "stderr": 0.1}},
                            "sample_count": {"effective": 250},
                        }
                    },
                },
                "result.json": {
                    "run_id": run.name,
                    "model": "model",
                    "benchmark": "bbh",
                    "framework": "lm_eval",
                },
                "terminal.json": {
                    "outcome": "success",
                    "started_at": "2026-08-13T12:00:00+08:00",
                    "finished_at": "2026-08-13T12:30:00+08:00",
                },
            }
            for name, value in values.items():
                (run / name).write_text(json.dumps(value), encoding="utf-8")
            (run / "config" / "run_config.json").write_text(
                json.dumps(
                    {
                        "backend": {
                            "parameters": {"port": 8091, "executable": "/usr/local/bin/vllm"},
                            "model_location": {"local_path": "/data/model"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run / "config" / "runtime_versions.json").write_text(
                json.dumps(
                    {
                        "device": {"vendor": "nvidia", "devices": [{"id": "0", "name": "A100"}]},
                        "runtime": {"family": "cuda", "version": "13.0", "driver_version": "550.90.07"},
                        "backend": {"adapter": "vllm", "adapter_version": "1.0"},
                        "evaluator": {
                            "adapter": "lm_eval",
                            "facts": {
                                "framework": "lm_eval",
                                "framework_version": "0.4.13",
                                "cache_root": "/cache",
                                "packages": {"transformers": "5.14.1"},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            lines = REPORT.render_lines(run)
            text = "\n".join(lines)
            self.assertIn("Outcome:   success", text)
            self.assertIn("boolean_expressions", text)
            self.assertIn("samples=250", text)
            self.assertIn("http://127.0.0.1:8091/v1/completions", text)
            svg = Path(temp) / "summary.svg"
            REPORT.write_svg(lines, svg)
            self.assertIn("<svg", svg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
