from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model_evaluation import result_report as REPORT


ROOT = Path(__file__).resolve().parents[1]


class ResultReportTests(unittest.TestCase):
    def make_run(self, root: Path, name: str = "model_bbh_20260813-120000") -> Path:
        run = root / "results" / name
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
                        "metrics": {
                            "acc_norm,none": {"value": 1.0, "stderr": 0.1}
                        },
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
        for filename, value in values.items():
            (run / filename).write_text(json.dumps(value), encoding="utf-8")
        (run / "config" / "run_config.json").write_text(
            json.dumps(
                {
                    "backend": {
                        "parameters": {
                            "port": 8091,
                            "executable": "/usr/local/bin/vllm",
                        },
                        "model_location": {"local_path": "/data/model"},
                    }
                }
            ),
            encoding="utf-8",
        )
        (run / "config" / "runtime_versions.json").write_text(
            json.dumps(
                {
                    "device": {
                        "vendor": "nvidia",
                        "devices": [{"id": "0", "name": "A100"}],
                    },
                    "runtime": {
                        "family": "cuda",
                        "version": "13.0",
                        "driver_version": "550.90.07",
                    },
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
        return run

    def test_saved_result_prints_tasks_environment_and_svg(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(Path(temp))
            lines = REPORT.render_lines(run)
            text = "\n".join(lines)
            self.assertIn("Outcome:   success", text)
            self.assertIn("boolean_expressions", text)
            self.assertIn("samples=250", text)
            self.assertIn("http://127.0.0.1:8091/v1/completions", text)
            svg = Path(temp) / "summary.svg"
            REPORT.write_svg(lines, svg)
            self.assertIn("<svg", svg.read_text(encoding="utf-8"))

    def test_write_run_report_saves_default_text_and_svg(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(Path(temp))
            report = REPORT.write_run_report(run)
            self.assertEqual(Path(report["run_dir"]), run.resolve())
            self.assertTrue((run / "result-summary.txt").is_file())
            self.assertTrue((run / "result-summary.svg").is_file())

    def test_source_tree_script_runs_without_installing_the_package(self):
        with tempfile.TemporaryDirectory() as temp:
            run = self.make_run(Path(temp))
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "print_result.py"), str(run)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("Summary: accuracy=0.500000", process.stdout)

if __name__ == "__main__":
    unittest.main()
