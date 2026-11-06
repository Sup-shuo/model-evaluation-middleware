"""Public synthetic catalog regression tests; no model weights or devices."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_evaluation.core.config.model_catalog import resolve_model_entries
from model_evaluation.core.errors import ConfigError


class ModalitySelectionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "native": {"id": "native", "source": {"type": "hf", "ref": "example/native"},
                       "metadata": {"artifacts_by_modality": {"text": "text", "image_text": "vision"}}},
            "text": {"id": "text", "source": {"type": "local", "ref": "example/text"},
                     "metadata": {"transformation": {"visual_capability": "removed"}}},
            "vision": {"id": "vision", "source": {"type": "local", "ref": "example/vision"},
                       "backends": {"vllm": {"dtype": "bfloat16"}}},
        }
        self.app = SimpleNamespace(specs=SimpleNamespace(resolve=lambda kind, value: {
            "metadata": {"input_modality": "image_text" if value == "images" else "text"}
        }))

    def resolve(self, models, benchmarks=None):
        with patch("model_evaluation.core.config.model_catalog._load_model_catalog",
                   return_value=(copy.deepcopy(self.catalog), Path("/catalog"), True)):
            return resolve_model_entries(self.app, {"models": models,
                "benchmarks": benchmarks or ["images"]}, Path("/evaluation.yaml"))[0]

    def test_default_and_explicit_text_unchanged(self):
        self.assertEqual(self.resolve(["native"])[0]["id"], "native")
        self.assertEqual(self.resolve(["text"])[0]["id"], "text")
        self.assertEqual(self.resolve([{"id": "native", "select_artifact_by_modality": False}])[0]["id"], "native")

    def test_selected_identity_source_and_loading(self):
        row = self.resolve([{"id": "native", "select_artifact_by_modality": True,
                             "resources": {"device_count": 2},
                             "overrides": {"backend": {"max_model_len": 1024}}}])[0]
        self.assertEqual(row["id"], "vision")
        self.assertEqual(row["source"]["ref"], "example/vision")
        self.assertEqual(row["backends"]["vllm"]["dtype"], "bfloat16")
        self.assertEqual(row["_run_resources"], {"device_count": 2})
        self.assertEqual(row["metadata"]["artifact_selection"]["requested_model_id"], "native")
        self.assertNotIn("artifact_selection", self.catalog["vision"].get("metadata", {}))
        self.assertEqual(self.resolve([{"id": "native", "select_artifact_by_modality": True}], ["bbh"])[0]["id"], "text")

    def test_bad_or_missing_mapping_and_mixed_benchmarks(self):
        selection = [{"id": "native", "select_artifact_by_modality": True}]
        for mapping in (None, {}, {"image_text": "missing"}, {"image_text": "native"},
                        {"image_text": 42}, {"text": "text"}):
            with self.subTest(mapping=mapping):
                self.catalog["native"]["metadata"]["artifacts_by_modality"] = mapping
                with self.assertRaises(ConfigError):
                    self.resolve(selection)
        with self.assertRaisesRegex(ConfigError, "one input_modality"):
            self.resolve(selection, ["images", "bbh"])
        with self.assertRaisesRegex(ConfigError, "existing model catalog"):
            self.resolve([{"id": "missing", "select_artifact_by_modality": True}])

    def test_identity_overrides_still_rejected(self):
        with self.assertRaisesRegex(ConfigError, "身份"):
            self.resolve([{"id": "native", "select_artifact_by_modality": True,
                           "overrides": {"source": {"ref": "wrong"}}}])


    def test_compiler_uses_target_identity(self):
        from model_evaluation.core.config.model_compiler import compile_models
        from unittest.mock import Mock
        row = self.resolve([{"id": "native", "select_artifact_by_modality": True}])[0]
        specs = Mock()
        with patch("model_evaluation.core.config.model_compiler._backend_parameters", return_value={}), patch(
                "model_evaluation.core.config.model_compiler._derived_backend_parameters", return_value={}):
            result = compile_models(app=self.app, specs=specs, model_entries=[row],
                deployment={}, backend_type="vllm", backend_client=None, mode="managed",
                available_devices=None, backend_profile_parameters={}, backend_run_parameters={},
                slug=lambda value, prefix: prefix + "-" + value,
                environment_selection=lambda value, label: value)
        self.assertEqual(result.ids, ["user-model-vision"])
        model = specs.register.call_args.args[1]
        self.assertEqual(model["experiment_id"], "vision")
        self.assertEqual(model["source"]["ref"], "example/vision")

    def test_catalog_selection_schema_accepts_only_boolean_flag(self):
        import json
        from jsonschema import Draft202012Validator
        schema = json.loads((Path(__file__).parents[1] /
            "model_evaluation/schemas/user/user_evaluation.schema.json").read_text())
        schema = {"$defs": schema["$defs"], "$ref": "#/$defs/catalogReference"}
        for flag in (True, False):
            Draft202012Validator(schema).validate({"id": "native", "select_artifact_by_modality": flag})
        self.assertTrue(list(Draft202012Validator(schema).iter_errors(
            {"id": "native", "select_artifact_by_modality": "true"})))
