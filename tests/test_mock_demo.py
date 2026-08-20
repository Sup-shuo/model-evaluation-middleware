from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MockDemoTests(unittest.TestCase):
    def test_demo_runs_real_reference_pipeline_without_accelerator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            environment = dict(os.environ)
            environment["MODEL_EVAL_PROJECT_ROOT"] = str(ROOT)
            environment["MODEL_EVAL_RUNTIME_ROOT"] = str(root / "runtime")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "eval-manager"),
                    "demo",
                    "--results-root",
                    str(root / "results"),
                    "--cache-root",
                    str(root / "cache"),
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            report = payload["report"]
            run_dir = Path(report["run_dir"])

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["demo"], "reference")
            self.assertEqual(report["outcome"], "success")
            self.assertEqual(report["cleanup"], "clean")
            self.assertEqual(report["model"], "mock-model")
            self.assertEqual(report["benchmark"], "mock_demo")
            self.assertEqual(report["framework"], "reference_eval")
            self.assertEqual(report["summary"]["contract_ok"]["value"], 1)

            run_config = json.loads(
                (run_dir / "config" / "run_config.json").read_text(encoding="utf-8")
            )
            adapters = {
                (row["kind"], row["name"])
                for row in run_config["adapters"]
            }
            self.assertIn(("device", "cpu"), adapters)
            self.assertIn(("runtime", "cpu"), adapters)
            self.assertIn(("backend", "reference"), adapters)
            self.assertIn(("dataset", "virtual"), adapters)
            self.assertIn(("binding", "reference_eval"), adapters)
            self.assertIn(("evaluator", "reference_eval"), adapters)
            self.assertFalse((run_dir / ".run").exists())

    def test_mock_cache_and_results_are_project_relative(self) -> None:
        sys.path.insert(0, str(ROOT))
        from model_evaluation import package_root
        from model_evaluation.core.app import Application

        app = Application(package_root(), project_root=ROOT)
        example_root = package_root() / "examples" / "mock"
        bundle = app.load_user_config(
            example_root / "system.yaml",
            example_root / "evaluation.yaml",
        )
        self.assertEqual(bundle.cache_root, str((ROOT / "cache").resolve()))
        self.assertEqual(bundle.results_root, str((ROOT / "results").resolve()))


if __name__ == "__main__":
    unittest.main()
