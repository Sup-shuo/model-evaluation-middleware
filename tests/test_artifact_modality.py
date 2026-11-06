import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from model_evaluation.commands.doctor import _doctor_row
from model_evaluation.core.artifact_modality import modality_skip
from model_evaluation.core.matrix_execution import BatchExecution, _execute_children
from model_evaluation.core.matrix_product import MatrixProductMixin
from model_evaluation.core.orchestrator_execution import execute_plan


class ModalityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.specs = {
            "model": {"id": "model"}, "platform": {}, "evaluation": {},
            "benchmark": {"id": "mmstar", "metadata": {"input_modality": "image_text"}},
            "deployment": {"model_location": {"local_path": tmp.name}},
        }
        self.plan = {"plan_id": "a", "compatibility": {"status": "compatible"},
                     "run_spec": dict.fromkeys(("model","platform","deployment","benchmark","evaluation"), "id"),
                     "resolved": {"management_mode": "managed", "specs": self.specs}}

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def test_explicit_sources(self):
        for filename, data in [
            ("config.json", {"language_model_only": True}),
            ("SOURCE_GGUF.json", {"export_mode": "text-llm"}),
            ("SOURCE_GGUF.json", {"export_mode": "language-only"}),
            ("DERIVATION.json", {"component_retention": {"vision": "removed"}}),
            ("DERIVATION.json", {"derived_format": "Qwen3 text-only dense BF16 safetensors"}),
        ]:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                self.specs["deployment"]["model_location"]["local_path"] = directory
                (Path(directory) / filename).write_text(json.dumps(data))
                self.assertIn(filename, modality_skip(self.plan)["marker"])
        self.specs["model"]["metadata"] = {"transformation": {"visual_capability": "removed"}}
        self.assertIn("visual_capability", modality_skip(self.plan)["marker"])

    def test_unknown_continues(self):
        self.assertIsNone(modality_skip(self.plan))
        self.write("config.json", {"language_model_only": "true"})
        self.write("SOURCE_GGUF.json", {"export_mode": "full"})
        self.specs["deployment"]["parameters"] = {"language_model_only": True}
        self.assertIsNone(modality_skip(self.plan))
        (self.root / "SOURCE_GGUF.json").write_text("{")
        self.assertIsNone(modality_skip(self.plan))

    def test_text_and_external_do_not_inspect_files(self):
        self.write("config.json", {"language_model_only": True})
        with patch("model_evaluation.core.artifact_modality._metadata", side_effect=AssertionError):
            self.plan["resolved"]["management_mode"] = "external"
            self.assertIsNone(modality_skip(self.plan))
            self.plan["resolved"]["management_mode"] = "managed"
            self.specs["benchmark"] = {"id": "bbh"}
            self.assertIsNone(modality_skip(self.plan))

    def test_doctor_no_probe(self):
        self.write("SOURCE_GGUF.json", {"export_mode": "language-only"})
        row = _doctor_row(None, None, self.plan, str(self.root))
        self.assertEqual(row["checks"]["artifact_modality"]["status"], "skipped")

    def test_batch_records_and_continues_without_start(self):
        self.write("config.json", {"language_model_only": True})
        executor = Mock()
        executor._model_identity = MatrixProductMixin._model_identity
        orch = executor.app.orchestrator.return_value
        def execute(plan):
            if plan["plan_id"] == "b":
                return self.root
            return execute_plan(orch, plan)
        orch.execute.side_effect = execute
        self.write("result.json", {})
        batch = {"matrix_id": "test", "plans": [self.plan, dict(self.plan, plan_id="b")]}
        state = BatchExecution()
        with patch("model_evaluation.core.orchestrator_execution.validate_execution_plan"):
            _execute_children(executor, batch, state, keep_going=False)
        row = MatrixProductMixin._public_run(state.status["a"])
        self.assertEqual((row["status"], row["attempts"]), ("not_run", 0))
        self.assertEqual(row["error"]["code"], "ARTIFACT_MODALITY_MISMATCH")
        self.assertNotIn("run_dir", row)
        self.assertEqual(state.status["b"]["status"], "success")
        orch.resources.run_lock.assert_not_called()
