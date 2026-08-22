from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_evaluation.commands.cli_app import _render_batch_summaries, build_parser


class CliResultReportTests(unittest.TestCase):
    def test_render_summary_flag_is_available_on_all_execution_commands(self):
        parser = build_parser()
        cases = (
            ["demo", "--render-summary"],
            ["run", "--render-summary"],
            ["run-plan", "plan.json", "--render-summary"],
            ["matrix-run", "matrix.yaml", "--render-summary"],
            ["run-matrix-plan", "plan.json", "--render-summary"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertTrue(parser.parse_args(argv).render_summary)

    def test_smoke_flag_is_available_on_user_config_commands(self):
        parser = build_parser()
        for command in ("validate", "doctor", "check", "explain", "plan", "run"):
            with self.subTest(command=command):
                self.assertTrue(parser.parse_args([command, "--smoke"]).smoke)

    def test_batch_renderer_only_processes_successful_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            success = root / "results" / "success"
            failed = root / "results" / "failed"
            batch = root / "results" / "_batches" / "batch"
            batch.mkdir(parents=True)
            (batch / "runs.json").write_text(
                json.dumps(
                    [
                        {"status": "success", "run_dir": str(success)},
                        {"status": "failed", "run_dir": str(failed)},
                    ]
                ),
                encoding="utf-8",
            )
            expected = {
                "run_dir": str(success),
                "text": str(success / "result-summary.txt"),
                "svg": str(success / "result-summary.svg"),
            }
            with patch(
                "model_evaluation.commands.cli_app._render_run_summary",
                return_value=expected,
            ) as render:
                reports = _render_batch_summaries(batch)
            self.assertEqual(reports, [expected])
            render.assert_called_once_with(str(success))


if __name__ == "__main__":
    unittest.main()
