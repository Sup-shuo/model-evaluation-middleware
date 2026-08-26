from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_evaluation.commands.cli_app import _render_batch_summaries, build_parser
from model_evaluation.commands.evaluate import (
    _write_success_reports,
    run_automatic_evaluation,
)
from model_evaluation.core.errors import ConfigError


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

    def test_automatic_evaluate_command_is_beginner_focused(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "--smoke", "--format", "json"])
        self.assertEqual(args.cmd, "evaluate")
        self.assertTrue(args.smoke)
        self.assertEqual(args.format, "json")

    def test_guided_init_options_are_available(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "init",
                "project",
                "--guided",
                "--hardware",
                "mlu",
                "--model-id",
                "org/model",
            ]
        )
        self.assertTrue(args.guided)
        self.assertEqual(args.hardware, "mlu")
        self.assertEqual(args.model_id, "org/model")
        self.assertEqual(args.devices, "0")

    def test_guided_prompt_rejects_placeholder_default(self):
        from model_evaluation.commands.cli_app import _prompt_value
        with patch("builtins.input", return_value=""):
            with self.assertRaises(ConfigError):
                _prompt_value("Models root", "/REPLACE_WITH_MODEL_ROOT")

    def test_automatic_projection_failure_is_a_warning(self):
        rows = [{"status": "success", "run_dir": "/results/run-1"}]
        with patch(
            "model_evaluation.commands.evaluate.write_run_report",
            side_effect=OSError("projection disk unavailable"),
        ):
            reports, warnings = _write_success_reports(rows)
        self.assertEqual(reports, [])
        self.assertEqual(warnings[0]["code"], "RESULT_PROJECTION_FAILED")
        self.assertEqual(warnings[0]["run_dir"], "/results/run-1")

    def test_automatic_projection_warning_keeps_success_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp) / "batch"
            batch.mkdir()
            (batch / "runs.json").write_text(
                json.dumps([{"status": "success", "run_dir": "/results/run-1"}]),
                encoding="utf-8",
            )

            class Executor:
                def execute(self, plan):
                    return batch, {"success": 1, "failed": 0, "not_run": 0}

            class App:
                def user_matrix_plan(self, *args, **kwargs):
                    return object(), SimpleNamespace(
                        results_root=Path(temp) / "results",
                        cache_root=Path(temp) / "cache",
                    )

                def matrix_executor(self, **kwargs):
                    return Executor()

            output = io.StringIO()
            with patch(
                "model_evaluation.commands.evaluate.build_check_report",
                return_value={"ok": True},
            ), patch(
                "model_evaluation.commands.evaluate.write_run_report",
                side_effect=OSError("projection disk unavailable"),
            ), redirect_stdout(output):
                code = run_automatic_evaluation(
                    App(),
                    system_config=None,
                    evaluation_config=None,
                    results_root=None,
                    cache_root=None,
                    smoke=False,
                    output_format="json",
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(
                payload["projection_warnings"][0]["code"],
                "RESULT_PROJECTION_FAILED",
            )

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
