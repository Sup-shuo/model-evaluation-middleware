from __future__ import annotations

import unittest
from pathlib import Path

from model_evaluation.core.app import Application
from model_evaluation.core.config.parsing import load_yaml_strict


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "model_evaluation"


class PublicExampleTests(unittest.TestCase):
    def test_user_examples_use_explicit_schema_13_profile_selection(self):
        system_paths = [ROOT / "config" / "system.yaml"]
        system_paths.extend(sorted((ROOT / "config" / "systems").glob("*.yaml")))
        for path in system_paths:
            system = load_yaml_strict(path.read_text(encoding="utf-8"))
            self.assertEqual(system["schema_version"], "1.3", path.name)
            defaults = (system.get("profiles") or {}).get("defaults") or {}
            self.assertNotIn("backend", defaults, path.name)
            self.assertNotIn("evaluator", defaults, path.name)

        evaluation_paths = [ROOT / "config" / "evaluation.yaml"]
        evaluation_paths.extend(sorted((ROOT / "config" / "evaluations").glob("*.yaml")))
        for path in evaluation_paths:
            evaluation = load_yaml_strict(path.read_text(encoding="utf-8"))
            self.assertEqual(evaluation["schema_version"], "1.3", path.name)
            self.assertIsInstance(evaluation.get("backend"), dict, path.name)
            self.assertIsInstance(evaluation.get("evaluator"), dict, path.name)
            self.assertTrue(evaluation["backend"].get("profile"), path.name)
            self.assertTrue(evaluation["evaluator"].get("profile"), path.name)

    def test_examples_cover_two_models_and_multiple_execution_stacks(self):
        app = Application(PACKAGE_ROOT, ROOT)
        cases = [
            ("nvidia", "smoke_bbh_08b", "qwen-example", "nvidia", "vllm"),
            ("mlu", "smoke_bbh_08b", "qwen-example", "mlu", "vllm"),
            ("metax", "smoke_bbh_08b", "qwen-example", "metax", "vllm"),
            (
                "cpu_llama_cpp",
                "smoke_bbh_llama_cpp",
                "llama-gguf-example",
                "cpu",
                "llama_cpp",
            ),
        ]
        seen_models = set()
        seen_hardware = set()
        seen_backends = set()
        for system_id, evaluation_id, model_id, hardware, backend in cases:
            bundle = app.load_user_config(
                ROOT / "config" / "systems" / f"{system_id}.yaml",
                ROOT / "config" / "evaluations" / f"{evaluation_id}.yaml",
            )
            self.assertEqual(bundle.evaluation["models"], [model_id])
            self.assertEqual(bundle.generated["selected_profiles"]["hardware"], hardware)
            self.assertEqual(bundle.generated["selected_profiles"]["backend"], backend)
            seen_models.add(model_id)
            seen_hardware.add(hardware)
            seen_backends.add(backend)

        self.assertGreaterEqual(len(seen_models), 2)
        self.assertGreaterEqual(len(seen_hardware), 3)
        self.assertGreaterEqual(len(seen_backends), 2)

    def test_models_do_not_own_machine_paths_or_capacity(self):
        app = Application(PACKAGE_ROOT, ROOT)
        forbidden = {
            "devices",
            "model_root",
            "cache_root",
            "results_root",
            "environment",
            "gpu_memory_utilization",
            "max_num_seqs",
            "num_concurrent",
            "port",
        }
        for path in sorted((ROOT / "config" / "models").glob("*.yaml")):
            model = load_yaml_strict(path.read_text(encoding="utf-8"))
            serialized_keys = set()

            def collect(value):
                if isinstance(value, dict):
                    serialized_keys.update(value)
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)

            collect(model)
            self.assertFalse(forbidden & serialized_keys, path.name)


if __name__ == "__main__":
    unittest.main()
