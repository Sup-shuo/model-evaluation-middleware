from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_evaluation.adapters.evaluator.evalscope import impl
from model_evaluation.adapters.evaluator.evalscope.predictions import validate_predictions
from model_evaluation.sdk.runtime import AdapterError


class EvalScopePredictionErrorTests(unittest.TestCase):
    def check_output(self, output):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "predictions" / "model" / "fixture.jsonl"
            prediction.parent.mkdir(parents=True)
            prediction.write_text(json.dumps({"model_output": output}) + "\n")
            validate_predictions(root)

    def test_context_error_is_not_a_wrong_answer(self):
        with self.assertRaisesRegex(AdapterError, "model_length"):
            self.check_output({"choices": [{"stop_reason": "model_length"}], "error": None})

    def test_explicit_generation_error_is_rejected(self):
        with self.assertRaisesRegex(AdapterError, "connection failed"):
            self.check_output({"choices": [], "error": "connection failed"})

    def test_normal_token_limit_is_a_valid_generation(self):
        self.check_output({"choices": [{"stop_reason": "max_tokens"}], "error": None})

    def test_incorrect_or_empty_answer_is_not_an_infrastructure_error(self):
        for text in ("Input length exceeds model context", "", "ANSWER: B"):
            with self.subTest(text=text):
                self.check_output({"choices": [{"stop_reason": "stop", "message": {"content": text}}]})

    def test_legacy_predictions_and_missing_predictions_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validate_predictions(root)
            path = root / "predictions" / "fixture.jsonl"
            path.parent.mkdir()
            path.write_text('{"sample": 1}\n\n')
            validate_predictions(root)

    def test_malformed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions" / "fixture.jsonl"
            path.parent.mkdir()
            path.write_text("not json\n")
            with self.assertRaisesRegex(AdapterError, "invalid EvalScope prediction"):
                validate_predictions(root)

    def test_normalization_does_not_publish_metrics_with_generation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "reports" / "fixture.json"
            report.parent.mkdir()
            report.write_text(json.dumps({"metric_name": "mean_acc", "score": 0.9, "num": 1}))
            prediction = root / "predictions" / "fixture.jsonl"
            prediction.parent.mkdir()
            prediction.write_text(json.dumps({"model_output": {
                "choices": [{"stop_reason": "model_length"}]}}) + "\n")
            with self.assertRaisesRegex(AdapterError, "generation failure"):
                impl.normalize({"raw_result_root": str(root), "task": {"task_id": "fixture"}}, {})
