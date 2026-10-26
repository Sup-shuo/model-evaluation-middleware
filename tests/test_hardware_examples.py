from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from model_evaluation.adapters.runtime.cann import impl as cann
from model_evaluation.core.app import Application
from model_evaluation.core.config.deployment import resolve_deployment_profile
from model_evaluation.core.process.env import prepare_process_for_environment
from model_evaluation.core.registry.adapter_registry import AdapterClient
from model_evaluation.onboarding import HARDWARE_TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "hardware"
SYSTEMS = (
    ("mthreads-s5000.yaml", "mthreads", "musa", "s5000"),
    ("ascend-910b.yaml", "ascend", "cann", "ascend"),
)


class HardwareExampleTests(unittest.TestCase):
    def _compile(self, filename):
        app = Application(ROOT / "model_evaluation", ROOT)
        with patch.object(AdapterClient, "invoke", side_effect=AssertionError("compile must not probe")):
            bundle = app.load_user_config(EXAMPLES / filename, EXAMPLES / "bbh.yaml")
        return bundle

    def test_compile_platform_and_separate_environments_without_probes(self):
        for filename, device, runtime, hardware in SYSTEMS:
            with self.subTest(system=filename):
                bundle = self._compile(filename)
                platform = bundle.specs.resolve("platform", bundle.generated["platform_id"])
                self.assertEqual(platform["device"], {"adapter": device, "devices": ["0"]})
                self.assertEqual(platform["runtime"], {
                    "adapter": runtime,
                    "parameters": {"root": "/REPLACE_WITH_" + runtime.upper() + "_ROOT"},
                })
                self.assertEqual(platform["backend_environment"], {"provider": "current", "profile": "current"})
                self.assertEqual(platform["evaluation_environment"], {
                    "provider": "conda", "profile": "/REPLACE_WITH_LM_EVAL_CONDA_ENV",
                    "parameters": {"executable": "/REPLACE_WITH_CONDA/bin/conda"},
                })
                self.assertEqual(bundle.generated["selected_profiles"]["hardware"], hardware)
                self.assertNotIn("runtime_env_patch", platform["evaluation_environment"])

    def test_compile_one_model_one_device_and_no_cpu_offload(self):
        for filename, _, runtime, _ in SYSTEMS:
            with self.subTest(system=filename):
                bundle = self._compile(filename)
                self.assertEqual(len(bundle.matrix_spec["models"]), 1)
                model_id = bundle.matrix_spec["models"][0]
                model = bundle.specs.resolve("model", model_id)
                self.assertEqual(model["experiment_id"], "coder3101-qwen35-4b-heretic")
                self.assertEqual(model["source"]["ref"], "coder3101/Qwen3.5-4B-heretic")
                platform = bundle.specs.resolve("platform", bundle.generated["platform_id"])
                override = bundle.matrix_spec["per_model_overrides"][model_id]
                self.assertEqual(override["platform"]["device"]["devices"], ["0"])
                deployment = bundle.specs.resolve("deployment", bundle.generated["deployment_id"])
                effective, _ = resolve_deployment_profile(deployment, model, platform, override["deployment"])
                self.assertEqual(effective["compatibility"]["runtime_families"], [runtime])
                self.assertEqual(effective["management"]["mode"], "managed")
                self.assertEqual(effective["parameters"]["tensor_parallel_size"], 1)
                self.assertEqual(effective["parameters"]["max_model_len"], 4096)
                args = effective["parameters"]["extra_args"]
                self.assertEqual(args[args.index("--cpu-offload-gb") + 1], "0")
                self.assertEqual(effective["model_location"]["local_path"],
                                 "/REPLACE_WITH_MODEL_ROOT/coder3101/Qwen3.5-4B-heretic")

    def test_full_bbh_and_fixed_seeds_are_shared(self):
        evaluation = yaml.safe_load((EXAMPLES / "bbh.yaml").read_text(encoding="utf-8"))
        self.assertEqual(evaluation["benchmarks"], ["bbh"])
        self.assertTrue(evaluation["offline"])
        self.assertEqual(evaluation["models"][0]["resources"]["device_count"], 1)
        for filename, _, _, _ in SYSTEMS:
            with self.subTest(system=filename):
                bundle = self._compile(filename)
                compiled = bundle.specs.resolve("evaluation", bundle.generated["evaluation_id"])
                self.assertNotIn("execution", compiled)
                self.assertEqual(compiled["parameters"]["batch_size"], 1)
                for key in ("numpy_random_seed", "torch_random_seed", "fewshot_random_seed", "request_seed"):
                    self.assertEqual(compiled["parameters"][key], 1234)
                benchmark = bundle.specs.resolve("benchmark", "bbh")
                self.assertEqual(benchmark["metadata"]["task_count"], 24)
                self.assertEqual(benchmark["protocol"]["task"], "leaderboard_bbh")

    def test_public_paths_and_no_private_deployment_values(self):
        forbidden = ("/root/", "/home/", "/mnt/", "/data", "/afs/", "/Users/",
                     "ssh ")
        for path in sorted(EXAMPLES.rglob("*")):
            if not path.is_file():
                continue
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                for value in forbidden:
                    self.assertNotIn(value, text)
        for filename, _, _, _ in SYSTEMS:
            value = yaml.safe_load((EXAMPLES / filename).read_text(encoding="utf-8"))
            self.assertEqual(value["paths"], {"cache": "cache", "results": "results"})
            self.assertEqual(value["models"]["root"], "/REPLACE_WITH_MODEL_ROOT")

    def test_onboarding_status_preserves_validation_boundaries(self):
        ascend = HARDWARE_TEMPLATES["ascend"].validation_status.lower()
        self.assertIn("910b", ascend)
        self.assertIn("full", ascend)
        self.assertIn("contract-tested", HARDWARE_TEMPLATES["amd"].validation_status.lower())
        metax = HARDWARE_TEMPLATES["metax"].validation_status.lower()
        self.assertIn("c500", metax)
        self.assertIn("full", metax)

    def test_supported_validate_cli_accepts_both_examples(self):
        for filename, _, _, _ in SYSTEMS:
            with self.subTest(system=filename):
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "eval-manager"), "validate",
                     "--system-config", str(EXAMPLES / filename),
                     "--evaluation-config", str(EXAMPLES / "bbh.yaml")],
                    cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    text=True, capture_output=True, timeout=30, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                output = json.loads(completed.stdout)
                self.assertTrue(output["ok"])
                self.assertEqual(output["benchmarks"], ["bbh"])
                self.assertEqual(len(output["models"]), 1)

    def test_cann_patch_persists_for_backend_but_not_evaluator(self):
        # Only an artificial runtime fixture is inspected; no installed CANN or device probes.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "lib64").mkdir()
            (root / "python/site-packages").mkdir(parents=True)
            runtime_patch = cann.resolve_environment({"parameters": {"root": directory}}, {})["env_patch"]
            process = {"argv": ["python", "-c", "pass"]}

            def wrapper(prepared):
                return {**prepared, "argv": ["/example/environment/bin/python", "-c", "pass"]}

            backend = prepare_process_for_environment(
                process, process_owner="backend", wrap=wrapper,
                base_patches=(("device", {"set": {"ASCEND_RT_VISIBLE_DEVICES": "0"}}),
                              ("runtime", runtime_patch)),
            )
            self.assertEqual(backend["env_patch"]["set"]["ASCEND_HOME_PATH"], str(root.resolve()))
            self.assertEqual(backend["env_patch"]["set"]["ASCEND_RT_VISIBLE_DEVICES"], "0")
            self.assertEqual(backend["env_patch"]["prepend_path"]["PYTHONPATH"],
                             [str(root / "python/site-packages")])
            evaluator = prepare_process_for_environment(process, process_owner="evaluator", wrap=wrapper)
            self.assertNotIn("ASCEND_HOME_PATH", evaluator["env_patch"]["set"])
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", evaluator["env_patch"]["set"])
            self.assertNotIn("PYTHONPATH", evaluator["env_patch"]["prepend_path"])


if __name__ == "__main__":
    unittest.main()
