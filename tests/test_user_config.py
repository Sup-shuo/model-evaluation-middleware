from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "model_evaluation"
sys.path.insert(0, str(ROOT))

from model_evaluation.core.app import Application


def _dump(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


class UserConfigTests(unittest.TestCase):
    def _external_docs(self, root: Path, *, two_profiles: bool = True) -> tuple[dict, dict]:
        backends = {
            "remote_a": {
                "type": "generic_openai",
                "mode": "external",
                "endpoint": {"base_url": "http://127.0.0.1:9999/v1", "auth": {"mode": "none"}},
            }
        }
        evaluators = {
            "lm_eval_a": {
                "type": "lm_eval",
                "root": str(root),
                "environment": {"type": "current"},
            }
        }
        defaults = {"backend": "remote_a", "evaluator": "lm_eval_a"}
        if two_profiles:
            backends["remote_b"] = {
                "type": "generic_openai",
                "mode": "external",
                "endpoint": {"base_url": "http://127.0.0.1:9998/v1", "auth": {"mode": "none"}},
            }
            evaluators["lm_eval_b"] = {
                "type": "lm_eval",
                "root": str(root),
                "environment": {"type": "current"},
                "parameters": {"batch_size": 3},
            }
        system = {
            "schema_version": "1.2",
            "system": {"name": "test-host"},
            "profiles": {"defaults": defaults, "backend": backends, "evaluator": evaluators},
            "models": {},
            "paths": {"cache": str(root / "cache"), "results": str(root / "results")},
        }
        evaluation = {
            "schema_version": "1.2",
            "models": ["model-A", {"name": "model-B"}],
            "benchmarks": ["mmlu"],
            "offline": True,
        }
        if two_profiles:
            evaluation["profiles"] = {"backend": "remote_b", "evaluator": "lm_eval_b"}
            evaluation["evaluator"] = {"batch_size": 2}
        return system, evaluation

    def _managed_vllm_docs(
        self,
        root: Path,
        *,
        runtime: str = "cpu",
        compatibility: list[str] | None = None,
        devices: list[int | str] | None = None,
        device_type: str = "cpu",
    ) -> tuple[dict, dict]:
        if compatibility is None:
            compatibility = [runtime]
        system = {
            "schema_version": "1.2",
            "system": {"name": "managed-host"},
            "profiles": {
                "defaults": {"hardware": "local", "backend": "vllm_local", "evaluator": "lm_eval_local"},
                "hardware": {
                    "local": {
                        "type": device_type,
                        "runtime": {"type": runtime},
                    }
                },
                "backend": {
                    "vllm_local": {
                        "type": "vllm",
                        "mode": "managed",
                        "compatibility": {"runtime_families": compatibility},
                        "environment": {"type": "current"},
                        "executable": "/bin/true",
                    }
                },
                "evaluator": {
                    "lm_eval_local": {
                        "type": "lm_eval",
                        "root": str(root),
                        "environment": {"type": "current"},
                    }
                },
            },
            "models": {"root": str(root / "models")},
            "paths": {"cache": str(root / "cache"), "results": str(root / "results")},
        }
        evaluation = {
            "schema_version": "1.2",
            "models": ["model-A"],
            "benchmarks": ["mmlu"],
        }
        if devices is not None:
            evaluation["resources"] = {"devices": devices}
        return system, evaluation

    def _write_docs(self, root: Path, system: dict, evaluation: dict) -> tuple[Path, Path]:
        system_path = root / "system.yaml"
        evaluation_path = root / "evaluation.yaml"
        _dump(system_path, system)
        _dump(evaluation_path, evaluation)
        return system_path, evaluation_path

    def test_external_profile_selection_generates_evaluation_only_platform(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            self.assertEqual(bundle.generated["selected_profiles"], {"backend": "remote_b", "evaluator": "lm_eval_b"})
            platform = app.specs.resolve("platform", bundle.generated["platform_id"])
            self.assertEqual(platform["evaluation_environment"]["provider"], "current")
            self.assertNotIn("device", platform)
            self.assertNotIn("runtime", platform)
            self.assertNotIn("backend_environment", platform)
            deployment = app.specs.resolve("deployment", bundle.generated["deployment_id"])
            self.assertEqual(deployment["endpoint"]["base_url"], "http://127.0.0.1:9998/v1")
            evaluation_spec = app.specs.resolve("evaluation", bundle.generated["evaluation_id"])
            self.assertEqual(evaluation_spec["parameters"]["batch_size"], 2)

    def test_profile_defaults_are_used_when_evaluation_omits_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            evaluation.pop("profiles", None)
            evaluation.pop("evaluator", None)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            bundle = Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)
            self.assertEqual(bundle.generated["selected_profiles"], {"backend": "remote_a", "evaluator": "lm_eval_a"})

    def test_single_profile_per_kind_is_automatic_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root, two_profiles=False)
            system["profiles"].pop("defaults", None)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            bundle = Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)
            self.assertEqual(bundle.generated["selected_profiles"], {"backend": "remote_a", "evaluator": "lm_eval_a"})

    def test_multiple_profiles_without_default_or_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system["profiles"].pop("defaults", None)
            evaluation.pop("profiles", None)
            evaluation.pop("evaluator", None)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "多个 profile"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_unknown_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            evaluation["profiles"]["backend"] = "missing_backend"
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaises(Exception):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_external_backend_rejects_hardware_selection_and_devices(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system["profiles"]["hardware"] = {"unused": {"type": "cpu", "runtime": {"type": "cpu"}}}
            evaluation["profiles"]["hardware"] = "unused"
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "不使用本地 Hardware profile"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

            evaluation["profiles"].pop("hardware")
            evaluation["resources"] = {"devices": [0]}
            _dump(evaluation_path, evaluation)
            with self.assertRaisesRegex(Exception, "不使用本地 resources.devices"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_external_backend_rejects_backend_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system["profiles"]["backend"]["remote_b"]["environment"] = {"type": "current"}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "不能配置 backend.environment"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_evaluator_environment_is_explicit_not_core_defaulted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system["profiles"]["evaluator"]["lm_eval_b"].pop("environment")
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "必须显式选择 EnvironmentProvider"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_managed_backend_environment_is_explicit_not_core_defaulted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system["profiles"]["backend"]["vllm_local"].pop("environment")
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "必须显式选择 EnvironmentProvider"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_managed_backend_requires_explicit_runtime_compatibility_even_single_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system["profiles"]["backend"]["vllm_local"].pop("compatibility")
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "必须显式声明 compatibility.runtime_families"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_runtime_family_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root, runtime="cpu", compatibility=["cuda"])
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "Profile compatibility"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_runtime_compatibility_is_core_owned_not_backend_parameter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root, runtime="cpu", compatibility=["cpu"])
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            deployment = app.specs.resolve("deployment", bundle.generated["deployment_id"])
            self.assertEqual(deployment["compatibility"]["runtime_families"], ["cpu"])
            self.assertNotIn("runtime_families", deployment.get("parameters", {}))

    def test_unspecified_devices_are_not_defaulted_to_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            platform = app.specs.resolve("platform", bundle.generated["platform_id"])
            self.assertNotIn("devices", platform["device"])
            self.assertNotIn("deployment", bundle.matrix_spec.get("overrides", {}))

    def test_vllm_tp_derives_only_from_explicit_selected_device_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root, devices=["gpu-A", "gpu-B"])
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            bundle = Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)
            self.assertEqual(bundle.matrix_spec["overrides"]["deployment"]["parameters"]["tensor_parallel_size"], 2)

    def test_hardware_profile_devices_are_machine_default_and_evaluation_can_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system["profiles"]["hardware"]["local"]["devices"] = [2, "gpu-B"]
            system_path, evaluation_path = self._write_docs(root, system, evaluation)

            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            platform = app.specs.resolve("platform", bundle.generated["platform_id"])
            self.assertEqual(platform["device"]["devices"], ["2", "gpu-B"])
            self.assertEqual(
                bundle.matrix_spec["overrides"]["deployment"]["parameters"]["tensor_parallel_size"],
                2,
            )

            evaluation["resources"] = {"devices": [0]}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            platform = app.specs.resolve("platform", bundle.generated["platform_id"])
            self.assertEqual(platform["device"]["devices"], ["0"])
            self.assertEqual(
                bundle.matrix_spec["overrides"]["deployment"]["parameters"]["tensor_parallel_size"],
                1,
            )

    def test_same_catalog_model_and_evaluation_resolve_for_mlu_and_nvidia_systems(self):
        """Only the system document changes; model and evaluation stay byte-identical."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            catalog = root / "models"
            catalog.mkdir()
            _dump(catalog / "qwen.yaml", {
                "schema_version": "1.0",
                "id": "qwen-portable",
                "source": {"type": "local", "ref": "Qwen/Portable"},
                "architecture": "qwen",
                "quantization": "bf16",
                "backends": {
                    "vllm": {"max_model_len": 4096, "trust_remote_code": True},
                    "llama_cpp": {"context_length": 4096},
                },
            })
            evaluation = {
                "schema_version": "1.2",
                "model_catalog": str(catalog),
                "models": ["qwen-portable"],
                "benchmarks": ["mmlu"],
                "backend": {"seed": 1234},
            }
            evaluation_path = root / "evaluation.yaml"
            _dump(evaluation_path, evaluation)

            def machine(name: str, device: str, runtime: str, device_id: int) -> Path:
                host = root / name
                model_dir = host / "models" / "Qwen" / "Portable"
                model_dir.mkdir(parents=True)
                lm_root = host / "lm-eval"
                (lm_root / "lm_eval").mkdir(parents=True)
                system = {
                    "schema_version": "1.2",
                    "system": {"name": name},
                    "profiles": {
                        "defaults": {"hardware": "accelerator", "backend": "inference", "evaluator": "eval"},
                        "hardware": {
                            "accelerator": {"type": device, "devices": [device_id], "runtime": {"type": runtime}}
                        },
                        "backend": {
                            "inference": {
                                "type": "vllm",
                                "mode": "managed",
                                "compatibility": {"runtime_families": [runtime]},
                                "environment": {"type": "current"},
                                "executable": "vllm",
                            }
                        },
                        "evaluator": {
                            "eval": {
                                "type": "lm_eval",
                                "root": str(lm_root),
                                "environment": {"type": "current"},
                                "parameters": {"require_clean_framework": False},
                            }
                        },
                    },
                    "models": {"root": str(host / "models")},
                    "paths": {"cache": str(host / "cache"), "results": str(host / "results")},
                }
                path = host / "system.yaml"
                _dump(path, system)
                return path

            resolved = {}
            for name, device, runtime, device_id in (
                ("mlu-host", "mlu", "neuware", 2),
                ("nvidia-host", "nvidia", "cuda", 0),
            ):
                app = Application(PACKAGE_ROOT, ROOT)
                bundle = app.user_config.load(machine(name, device, runtime, device_id), evaluation_path)
                platform = app.specs.resolve("platform", bundle.generated["platform_id"])
                deployment = app.specs.resolve("deployment", bundle.generated["deployment_id"])
                resolved[name] = (platform, deployment, bundle)

            self.assertEqual(resolved["mlu-host"][0]["device"]["devices"], ["2"])
            self.assertEqual(resolved["nvidia-host"][0]["device"]["devices"], ["0"])
            self.assertEqual(resolved["mlu-host"][0]["runtime"]["adapter"], "neuware")
            self.assertEqual(resolved["nvidia-host"][0]["runtime"]["adapter"], "cuda")
            for platform, deployment, bundle in resolved.values():
                self.assertEqual(platform["backend_environment"]["provider"], "current")
                self.assertEqual(deployment["backend"]["adapter"], "vllm")
                self.assertEqual(bundle.matrix_spec["models"], list(bundle.generated["model_ids"]))
                model_id = next(iter(bundle.generated["model_ids"]))
                self.assertEqual(
                    bundle.matrix_spec["per_model_overrides"][model_id]["deployment"]["parameters"],
                    {"max_model_len": 4096, "trust_remote_code": True},
                )

    def test_machine_capacity_parameters_come_from_system_not_model(self):
        """A machine can tune service capacity without changing Model/Evaluation."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            tools = root / "bin"
            tools.mkdir()
            smi = tools / "nvidia-smi"
            smi.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *index,name,memory.total,uuid*) echo '0, NVIDIA Test GPU, 24576, GPU-test-0' ;;\n"
                "  *driver_version*) echo '555.42.02' ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            smi.chmod(0o755)
            system, evaluation = self._managed_vllm_docs(
                root, runtime="cuda", compatibility=["cuda"], device_type="nvidia",
            )
            system["profiles"]["backend"]["vllm_local"]["parameters"] = {
                "gpu_memory_utilization": 0.3,
                "max_num_seqs": 2,
                "num_concurrent": 1,
            }
            catalog = root / "models"
            catalog.mkdir()
            _dump(catalog / "qwen.yaml", {
                "schema_version": "1.0",
                "id": "qwen-portable",
                "source": {"type": "local", "ref": "Qwen/Portable"},
                "backends": {"vllm": {"max_model_len": 4096, "trust_remote_code": True}},
            })
            (root / "models-root" / "Qwen" / "Portable").mkdir(parents=True)
            system["models"] = {"root": str(root / "models-root")}
            evaluation["model_catalog"] = str(catalog)
            evaluation["models"] = ["qwen-portable"]
            evaluation["backend"] = {"seed": 17}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with patch.dict(os.environ, {"PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")}):
                matrix_plan, bundle = Application(PACKAGE_ROOT, ROOT).user_matrix_plan(
                    system_path, evaluation_path,
                )
            model_id = next(iter(bundle.generated["model_ids"]))
            self.assertEqual(
                bundle.matrix_spec["per_model_overrides"][model_id]["deployment"]["parameters"],
                {"max_model_len": 4096, "trust_remote_code": True},
            )
            params = matrix_plan["plans"][0]["resolved"]["specs"]["deployment"]["parameters"]
            self.assertEqual(params["gpu_memory_utilization"], 0.3)
            self.assertEqual(params["max_num_seqs"], 2)
            self.assertEqual(params["num_concurrent"], 1)
            self.assertEqual(params["seed"], 17)
            self.assertEqual(params["max_model_len"], 4096)

    def test_nvidia_cuda_system_builds_plan_with_simulated_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            tools = root / "bin"
            tools.mkdir()
            smi = tools / "nvidia-smi"
            smi.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *index,name,memory.total,uuid*) echo '0, NVIDIA Test GPU, 24576, GPU-test-0' ;;\n"
                "  *driver_version*) echo '555.42.02' ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            smi.chmod(0o755)
            model_root = root / "host" / "models"
            (model_root / "Qwen" / "Portable").mkdir(parents=True)
            lm_root = root / "host" / "lm-eval"
            (lm_root / "lm_eval").mkdir(parents=True)
            catalog = root / "models"
            catalog.mkdir()
            _dump(catalog / "qwen.yaml", {
                "schema_version": "1.0",
                "id": "qwen-portable",
                "source": {"type": "local", "ref": "Qwen/Portable"},
                "context_length": 4096,
                "backend": {"max_model_len": 4096},
            })
            evaluation_path = root / "evaluation.yaml"
            _dump(evaluation_path, {
                "schema_version": "1.2",
                "model_catalog": str(catalog),
                "models": ["qwen-portable"],
                "benchmarks": ["mmlu"],
            })
            system_path = root / "system.yaml"
            _dump(system_path, {
                "schema_version": "1.2",
                "system": {"name": "nvidia-test"},
                "profiles": {
                    "defaults": {"hardware": "nvidia", "backend": "vllm", "evaluator": "lm_eval"},
                    "hardware": {"nvidia": {"type": "nvidia", "devices": [0], "runtime": {"type": "cuda"}}},
                    "backend": {
                        "vllm": {
                            "type": "vllm",
                            "compatibility": {"runtime_families": ["cuda"]},
                            "environment": {"type": "current"},
                            "executable": "vllm",
                        }
                    },
                    "evaluator": {
                        "lm_eval": {
                            "type": "lm_eval",
                            "root": str(lm_root),
                            "environment": {"type": "current"},
                            "parameters": {"require_clean_framework": False},
                        }
                    },
                },
                "models": {"root": str(model_root)},
                "paths": {"cache": str(root / "cache"), "results": str(root / "results")},
            })
            with patch.dict(os.environ, {"PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")}):
                matrix_plan, bundle = Application(PACKAGE_ROOT, ROOT).user_matrix_plan(system_path, evaluation_path)
            plan = matrix_plan["plans"][0]
            self.assertEqual(plan["resolved"]["platform"]["device"]["vendor"], "nvidia")
            self.assertEqual(plan["resolved"]["platform"]["runtime"]["family"], "cuda")
            self.assertEqual(plan["resolved"]["platform"]["device_env_patch"]["set"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(plan["resolved"]["specs"]["deployment"]["compatibility"]["runtime_families"], ["cuda"])
            self.assertEqual(bundle.generated["selected_profiles"], {
                "backend": "vllm", "evaluator": "lm_eval", "hardware": "nvidia",
            })

    def test_profile_tensor_parallel_setting_is_not_overridden_by_device_derivation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root, devices=[0, 1])
            system["profiles"]["backend"]["vllm_local"]["parameters"] = {"tensor_parallel_size": 1}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            bundle = Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)
            self.assertNotIn("tensor_parallel_size", bundle.matrix_spec.get("overrides", {}).get("deployment", {}).get("parameters", {}))
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            dep = app.specs.resolve("deployment", bundle.generated["deployment_id"])
            self.assertEqual(dep["parameters"]["tensor_parallel_size"], 1)

    def test_platform_functional_parameters_are_direct_not_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root, runtime="cuda", compatibility=["cuda"], device_type="nvidia")
            system["profiles"]["hardware"]["local"]["runtime"]["root"] = "/opt/cuda"
            system["profiles"]["hardware"]["local"]["runtime"]["parameters"] = {"nvcc": "/usr/local/cuda/bin/nvcc"}
            system["profiles"]["backend"]["vllm_local"]["environment"] = {"type": "conda", "profile": "vllm", "executable": "/opt/conda/bin/conda"}
            system["profiles"]["evaluator"]["lm_eval_local"]["environment"] = {"type": "conda", "profile": "eval", "executable": "/opt/conda/bin/conda"}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            platform = app.specs.resolve("platform", bundle.generated["platform_id"])
            self.assertEqual(platform["runtime"]["parameters"]["root"], "/opt/cuda")
            self.assertEqual(platform["runtime"]["parameters"]["nvcc"], "/usr/local/cuda/bin/nvcc")
            self.assertEqual(platform["backend_environment"]["parameters"]["executable"], "/opt/conda/bin/conda")
            self.assertEqual(platform["evaluation_environment"]["parameters"]["executable"], "/opt/conda/bin/conda")
            self.assertNotIn("middleware", platform.get("metadata", {}))

    def test_environment_object_is_provider_generic_and_conda_profile_is_adapter_declared(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system["profiles"]["evaluator"]["lm_eval_local"]["environment"] = {
                "type": "conda", "profile": "lm-eval", "executable": "/opt/conda/bin/conda"
            }
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            platform = app.specs.resolve("platform", bundle.generated["platform_id"])
            self.assertEqual(platform["evaluation_environment"]["provider"], "conda")
            self.assertEqual(platform["evaluation_environment"]["profile"], "lm-eval")

    def test_backend_user_parameter_typo_is_rejected_by_selected_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            evaluation["backend"] = {"max_modle_len": 8192}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "max_modle_len"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_evaluator_user_parameter_typo_is_rejected_by_selected_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            evaluation["evaluator"] = {"bacth_size": 4}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "bacth_size"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_selected_adapter_must_exist_even_without_parameters(self):
        cases = (("backend", "remote_b", "type"), ("evaluator", "lm_eval_b", "type"))
        for kind, profile, field in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                root = Path(td).resolve()
                system, evaluation = self._external_docs(root)
                system["profiles"][kind][profile][field] = f"missing_{kind}"
                system_path, evaluation_path = self._write_docs(root, system, evaluation)
                with self.assertRaisesRegex(Exception, "adapter not found"):
                    Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_selected_device_runtime_and_environment_adapters_must_exist(self):
        mutations = [
            ("device", lambda s: s["profiles"]["hardware"]["local"].update({"type": "missing_device"})),
            ("runtime", lambda s: s["profiles"]["hardware"]["local"]["runtime"].update({"type": "missing_runtime"})),
            ("environment", lambda s: s["profiles"]["backend"]["vllm_local"].update({"environment": {"type": "missing_environment"}})),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                root = Path(td).resolve()
                system, evaluation = self._managed_vllm_docs(root)
                mutate(system)
                if label == "runtime":
                    system["profiles"]["backend"]["vllm_local"]["compatibility"] = {"runtime_families": ["missing_runtime"]}
                system_path, evaluation_path = self._write_docs(root, system, evaluation)
                with self.assertRaisesRegex(Exception, "adapter not found"):
                    Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_evaluator_type_must_match_preset_framework(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system["profiles"]["evaluator"]["lm_eval_b"]["type"] = "reference_eval"
            system["profiles"]["evaluator"]["lm_eval_b"]["preset"] = "lm_eval_current"
            system["profiles"]["evaluator"]["lm_eval_b"].pop("root")
            system["profiles"]["evaluator"]["lm_eval_b"].pop("parameters", None)
            evaluation.pop("evaluator", None)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "type/preset 不一致"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_external_backend_does_not_require_models_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            self.assertEqual(system["models"], {})
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            bundle = Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)
            self.assertTrue(bundle.matrix_spec["models"])

    def test_managed_path_backend_requires_models_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system["models"] = {}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "需要本地模型根目录"):
                Application(PACKAGE_ROOT, ROOT).user_config.load(system_path, evaluation_path)

    def test_managed_backend_without_path_policy_can_omit_models_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system["profiles"]["backend"]["vllm_local"] = {
                "type": "ollama",
                "mode": "managed",
                "compatibility": {"runtime_families": ["cpu"]},
                "environment": {"type": "current"},
            }
            system["models"] = {}
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            deployment = app.specs.resolve("deployment", bundle.generated["deployment_id"])
            self.assertNotIn("model_location", deployment)

    def test_backend_defaults_come_from_versioned_adapter_user_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._managed_vllm_docs(root)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.user_config.load(system_path, evaluation_path)
            deployment = app.specs.resolve("deployment", bundle.generated["deployment_id"])
            self.assertEqual(deployment["parameters"]["port"], 8091)
            manifest = app.registry.get("backend", "vllm").identity.manifest
            self.assertEqual(manifest["user_config"]["schema_version"], "1.0")
            self.assertNotIn("user_config", manifest.get("implementation", {}))

    def test_reference_evaluator_proves_second_framework_path_without_core_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system["profiles"]["evaluator"]["ref"] = {"type": "reference_eval", "environment": {"type": "current"}}
            evaluation["profiles"]["evaluator"] = "ref"
            evaluation.pop("evaluator", None)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            plan, bundle = app.user_matrix_plan(system_path, evaluation_path)
            self.assertTrue(plan["plans"])
            for child in plan["plans"]:
                self.assertEqual(child["resolved"]["binding_adapter"], "reference_eval")

    def test_user_config_can_build_external_matrix_plan_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            plan, bundle = app.user_matrix_plan(system_path, evaluation_path)
            self.assertTrue(plan["matrix_id"].startswith("matrix-"))
            self.assertEqual(len(plan["plans"]), 2)
            self.assertEqual(bundle.results_root, str((root / "results").resolve()))

    def test_failed_user_config_load_preserves_last_complete_overlay_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root, two_profiles=False)
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            app = Application(PACKAGE_ROOT, ROOT)
            bundle = app.load_user_config(system_path, evaluation_path)
            before = app.specs.overlay_snapshot()

            # This fails only after platform/deployment/evaluation/model specs
            # have been generated in the resolver's private repository.
            evaluation["benchmarks"] = ["mmlu", "missing-benchmark"]
            _dump(evaluation_path, evaluation)
            with self.assertRaisesRegex(Exception, "missing-benchmark"):
                app.load_user_config(system_path, evaluation_path)

            self.assertEqual(app.specs.overlay_snapshot(), before)
            model_id = bundle.matrix_spec["models"][0]
            self.assertEqual(app.specs.resolve("model", model_id)["source"]["ref"], "model-A")

    def test_loaded_bundles_can_be_planned_after_a_later_load_replaces_app_overlays(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            app = Application(PACKAGE_ROOT, ROOT)

            root_a = root / "a"
            root_a.mkdir()
            system_a, evaluation_a = self._external_docs(root_a, two_profiles=False)
            system_a["system"]["name"] = "machine-a"
            evaluation_a["models"] = ["model-A"]
            system_a_path, evaluation_a_path = self._write_docs(root_a, system_a, evaluation_a)
            bundle_a = app.load_user_config(system_a_path, evaluation_a_path)

            root_b = root / "b"
            root_b.mkdir()
            system_b, evaluation_b = self._external_docs(root_b, two_profiles=False)
            system_b["system"]["name"] = "machine-b"
            evaluation_b["models"] = ["model-B"]
            system_b_path, evaluation_b_path = self._write_docs(root_b, system_b, evaluation_b)
            bundle_b = app.load_user_config(system_b_path, evaluation_b_path)

            plan_a = app.build_user_matrix_plan(bundle_a)
            plan_b = app.build_user_matrix_plan(bundle_b)
            model_a = plan_a["plans"][0]["resolved"]["specs"]["model"]
            model_b = plan_b["plans"][0]["resolved"]["specs"]["model"]
            self.assertEqual(model_a["source"]["ref"], "model-A")
            self.assertEqual(model_b["source"]["ref"], "model-B")
            self.assertNotEqual(bundle_a.specs, bundle_b.specs)
            self.assertNotEqual(bundle_a.generated["platform_id"], bundle_b.generated["platform_id"])

    def test_results_default_and_relative_path_are_project_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            system, evaluation = self._external_docs(root)
            system["paths"].pop("results")
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            bundle = Application(PACKAGE_ROOT, ROOT).load_user_config(system_path, evaluation_path)
            self.assertEqual(bundle.results_root, str((ROOT / "results").resolve()))

            system["paths"]["results"] = "results/custom"
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            bundle = Application(PACKAGE_ROOT, ROOT).load_user_config(system_path, evaluation_path)
            self.assertEqual(bundle.results_root, str((ROOT / "results" / "custom").resolve()))

            system["paths"]["results"] = "../outside"
            system_path, evaluation_path = self._write_docs(root, system, evaluation)
            with self.assertRaisesRegex(Exception, "不能越过项目根目录"):
                Application(PACKAGE_ROOT, ROOT).load_user_config(system_path, evaluation_path)


    def test_named_environment_profiles_and_model_role_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            system,evaluation=self._managed_vllm_docs(root)
            system['profiles']['environment']={
                'backend_default':{'type':'current'},
                'eval_default':{'type':'current'},
                'model_special':{'type':'venv','profile':str(root/'special-venv')},
            }
            venv_bin=root/'special-venv'/'bin'; venv_bin.mkdir(parents=True); (venv_bin/'python').symlink_to(Path(__import__('sys').executable).resolve())
            system['profiles']['backend']['vllm_local']['environment']='backend_default'
            system['profiles']['evaluator']['lm_eval_local']['environment']='eval_default'
            evaluation['models']=[
                {'id':'same-source-default','ref':'model-A','label':'Default env'},
                {'id':'same-source-special','ref':'model-A','label':'Special env','environments':{'backend':'model_special'}},
            ]
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            app=Application(PACKAGE_ROOT, ROOT); bundle=app.user_config.load(system_path,evaluation_path)
            self.assertEqual(len(bundle.matrix_spec['models']),2)
            first,second=bundle.matrix_spec['models']
            self.assertNotEqual(first,second)
            self.assertNotIn(first,bundle.matrix_spec.get('per_model_overrides',{}))
            patch=bundle.matrix_spec['per_model_overrides'][second]
            self.assertEqual(patch['platform']['backend_environment']['provider'],'venv')
            self.assertEqual(patch['platform']['backend_environment']['profile'],str((root/'special-venv').resolve()))
            model_a=app.specs.resolve('model',first); model_b=app.specs.resolve('model',second)
            self.assertEqual(model_a['source']['ref'],model_b['source']['ref'])
            self.assertEqual(model_a['experiment_id'],'same-source-default')
            self.assertEqual(model_b['experiment_id'],'same-source-special')
            self.assertEqual(model_b['label'],'Special env')
            # Verify the per-model environment survives Matrix expansion and Planner
            # resolution, not only UserConfigResolver serialization.
            (root/'models'/'model-A').mkdir(parents=True)
            matrix_plan,_=Application(PACKAGE_ROOT, ROOT).user_matrix_plan(system_path,evaluation_path)
            by_exp={child['resolved']['specs']['model']['experiment_id']:child for child in matrix_plan['plans']}
            self.assertEqual(by_exp['same-source-default']['resolved']['platform']['backend_environment']['provider'],'current')
            self.assertEqual(by_exp['same-source-special']['resolved']['platform']['backend_environment']['provider'],'venv')
            self.assertEqual(by_exp['same-source-special']['resolved']['platform']['backend_environment']['identity'],str((root/'special-venv').resolve()))

    def test_evaluation_environment_override_beats_system_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            system,evaluation=self._external_docs(root,two_profiles=False)
            system['profiles']['environment']={'default_eval':{'type':'current'},'isolated_eval':{'type':'venv','profile':str(root/'eval-venv')}}
            venv_bin=root/'eval-venv'/'bin'; venv_bin.mkdir(parents=True); (venv_bin/'python').symlink_to(Path(__import__('sys').executable).resolve())
            system['profiles']['evaluator']['lm_eval_a']['environment']='default_eval'
            evaluation['environments']={'evaluator':'isolated_eval'}
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            app=Application(PACKAGE_ROOT, ROOT); bundle=app.user_config.load(system_path,evaluation_path)
            platform=app.specs.resolve('platform',bundle.generated['platform_id'])
            self.assertEqual(platform['evaluation_environment']['provider'],'venv')
            self.assertEqual(platform['evaluation_environment']['profile'],str((root/'eval-venv').resolve()))

    def test_model_environment_override_beats_evaluation_environment_override(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            system,evaluation=self._managed_vllm_docs(root)
            for name in ('eval-backend','model-backend'):
                bindir=root/name/'bin'; bindir.mkdir(parents=True); (bindir/'python').symlink_to(Path(__import__('sys').executable).resolve())
            system['profiles']['environment']={
                'system-backend':{'type':'current'},
                'eval-backend':{'type':'venv','profile':str(root/'eval-backend')},
                'model-backend':{'type':'venv','profile':str(root/'model-backend')},
            }
            system['profiles']['backend']['vllm_local']['environment']='system-backend'
            evaluation['environments']={'backend':'eval-backend'}
            evaluation['models']=[{'id':'m','ref':'model-A','environments':{'backend':'model-backend'}}]
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            app=Application(PACKAGE_ROOT, ROOT); bundle=app.user_config.load(system_path,evaluation_path)
            base_platform=app.specs.resolve('platform',bundle.generated['platform_id'])
            self.assertEqual(base_platform['backend_environment']['profile'],str((root/'eval-backend').resolve()))
            model_id=bundle.matrix_spec['models'][0]
            patch=bundle.matrix_spec['per_model_overrides'][model_id]['platform']['backend_environment']
            self.assertEqual(patch['profile'],str((root/'model-backend').resolve()))

    def test_same_evaluation_reuses_across_two_conda_and_shared_conda_machines(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            evaluation={
                'schema_version':'1.2','models':['model-A'],'benchmarks':['mmlu'],
                'resources':{'devices':[0]},
            }
            evaluation_path=root/'evaluation.yaml'; _dump(evaluation_path,evaluation)

            def machine(name, *, backend_env, eval_env, environment_profiles):
                host=root/name; host.mkdir()
                fake_conda=host/'conda'
                fake_conda.write_text(f'#!/bin/sh\necho "{sys.executable}"\n')
                fake_conda.chmod(0o755)
                profiles={}
                for env_name,env_profile in environment_profiles.items():
                    profiles[env_name]={'type':'conda','profile':env_profile,'executable':str(fake_conda)}
                model_root=host/'models'; (model_root/'model-A').mkdir(parents=True)
                lm_root=host/'lm-harness'; (lm_root/'lm_eval').mkdir(parents=True)
                system={
                    'schema_version':'1.2','system':{'name':name},
                    'profiles':{
                        'defaults':{'hardware':'local','backend':'vllm_local','evaluator':'lm_eval_local'},
                        'environment':profiles,
                        'hardware':{'local':{'type':'cpu','runtime':{'type':'cpu'}}},
                        'backend':{'vllm_local':{'type':'vllm','mode':'managed','compatibility':{'runtime_families':['cpu']},'environment':backend_env,'executable':'vllm'}},
                        'evaluator':{'lm_eval_local':{'type':'lm_eval','root':str(lm_root),'environment':eval_env,'parameters':{'require_clean_framework':False}}},
                    },
                    'models':{'root':str(model_root)},
                    'paths':{'cache':str(host/'cache'),'results':str(host/'results')},
                }
                path=host/'system.yaml'; _dump(path,system); return path

            system_a=machine('machine-a',backend_env='backend-conda',eval_env='eval-conda',environment_profiles={'backend-conda':'backend-env','eval-conda':'eval-env'})
            system_b=machine('machine-b',backend_env='shared-conda',eval_env='shared-conda',environment_profiles={'shared-conda':'shared-env'})
            plan_a,_=Application(PACKAGE_ROOT, ROOT).user_matrix_plan(system_a,evaluation_path)
            plan_b,_=Application(PACKAGE_ROOT, ROOT).user_matrix_plan(system_b,evaluation_path)
            a=plan_a['plans'][0]; b=plan_b['plans'][0]
            self.assertEqual(a['resolved']['platform']['backend_environment']['identity'],'backend-env')
            self.assertEqual(a['resolved']['platform']['evaluation_environment']['identity'],'eval-env')
            self.assertEqual(b['resolved']['platform']['backend_environment']['identity'],'shared-env')
            self.assertEqual(b['resolved']['platform']['evaluation_environment']['identity'],'shared-env')
            self.assertEqual(a['resolved']['specs']['deployment']['parameters']['executable'],'vllm')
            self.assertEqual(b['resolved']['specs']['deployment']['parameters']['executable'],'vllm')

    def test_environment_provider_shorthand_respects_profile_required(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._external_docs(root,two_profiles=False)
            # current has profile_required=false and remains a useful shorthand.
            system['profiles']['evaluator']['lm_eval_a']['environment']='current'
            sp,ep=self._write_docs(root,system,evaluation)
            bundle=Application(PACKAGE_ROOT, ROOT).load_user_config(sp,ep)
            self.assertEqual(bundle.generated['selected_profiles']['evaluator'],'lm_eval_a')
            # conda/venv require an explicit environment name/prefix and may not
            # silently reuse the provider name as the profile.
            system['profiles']['evaluator']['lm_eval_a']['environment']='conda'
            _dump(sp,system)
            with self.assertRaisesRegex(Exception,'要求显式 profile/name'):
                Application(PACKAGE_ROOT, ROOT).load_user_config(sp,ep)

    def test_environment_object_rejects_profile_and_name_together(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._external_docs(root,two_profiles=False)
            system['profiles']['evaluator']['lm_eval_a']['environment']={
                'type':'current','profile':'current','name':'other',
            }
            sp,ep=self._write_docs(root,system,evaluation)
            with self.assertRaises(Exception):
                Application(PACKAGE_ROOT, ROOT).load_user_config(sp,ep)

    def test_machine_and_evaluation_config_can_default_from_environment_variables(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._external_docs(root,two_profiles=False)
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            with patch.dict(os.environ,{'MODEL_EVAL_SYSTEM_CONFIG':str(system_path),'MODEL_EVAL_EVALUATION_CONFIG':str(evaluation_path)},clear=False):
                bundle=Application(PACKAGE_ROOT, ROOT).load_user_config()
            self.assertEqual(bundle.system['system']['name'],'test-host')
            self.assertEqual(bundle.evaluation['models'][0],'model-A')

    def test_recursive_user_model_catalog_keeps_evaluation_as_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root,devices=[0])
            model_dir=root/'models'/'qwen'; model_dir.mkdir(parents=True)
            _dump(model_dir/'qwen_awq.yaml',{
                'schema_version':'1.0','id':'qwen-awq','label':'Qwen AWQ',
                'source':{'type':'hf','ref':'org/qwen-awq','revision':'abc123'},
                'architecture':'qwen','quantization':'awq','format':'safetensors',
                'context_length':32768,'tokenizer':{'ref':'org/qwen-tokenizer'},
                'trust_remote_code':True,'chat_template':'/templates/qwen.jinja',
                'backend':{'max_model_len':8192},
                'provenance':{'policy':'pinned'},
                'metadata':{'family':'qwen'},
            })
            evaluation['models']=['qwen-awq']
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            app=Application(PACKAGE_ROOT, ROOT); bundle=app.load_user_config(system_path,evaluation_path)
            generated_id=next(iter(bundle.generated['model_ids']))
            model=app.specs.resolve('model',generated_id)
            self.assertEqual(model['experiment_id'],'qwen-awq')
            self.assertEqual(model['source'],{'type':'hf','ref':'org/qwen-awq','revision':'abc123'})
            self.assertEqual(model['quantization'],'awq')
            self.assertEqual(model['format'],'safetensors')
            self.assertEqual(model['context_length'],32768)
            self.assertEqual(model['tokenizer']['ref'],'org/qwen-tokenizer')
            self.assertTrue(model['trust_remote_code'])
            self.assertEqual(model['metadata']['family'],'qwen')
            patch=bundle.matrix_spec['per_model_overrides'][generated_id]
            self.assertEqual(patch['deployment']['parameters']['max_model_len'],8192)
            self.assertTrue(bundle.generated['model_catalog']['enabled'])
            self.assertEqual(Path(bundle.generated['model_catalog']['root']),root/'models')

    def test_model_catalog_ignores_macos_metadata_files(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root,devices=[0])
            catalog=root/'models'; catalog.mkdir()
            _dump(catalog/'portable.yaml',{
                'schema_version':'1.0','id':'portable-model',
                'source':{'type':'hf','ref':'org/portable-model'},
            })
            (catalog/'._portable.yaml').write_bytes(b'\x00AppleDouble')
            nested=catalog/'._metadata'; nested.mkdir()
            (nested/'ignored.yaml').write_bytes(b'\x00AppleDouble')
            evaluation['models']=['portable-model']
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            bundle=Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)
            self.assertEqual(len(bundle.generated['model_ids']),1)

    def test_catalog_logical_tokenizer_resolves_under_each_system_model_root(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            catalog=root/'catalog'; catalog.mkdir()
            _dump(catalog/'negentropy.yaml',{
                'schema_version':'1.0',
                'id':'jackrong-negentropy-claude-opus47-4b',
                'source':{'type':'hf','ref':'Jackrong/Negentropy-claude-opus-4.7-4B'},
                'tokenizer':'coder3101/Qwen3.5-4B-heretic',
            })
            evaluation={
                'schema_version':'1.2',
                'model_catalog':str(catalog),
                'models':['jackrong-negentropy-claude-opus47-4b'],
                'benchmarks':['mmlu'],
            }
            evaluation_path=root/'evaluation.yaml'; _dump(evaluation_path,evaluation)

            for machine in ('mlu-model-root','a100-model-root'):
                machine_root=root/machine
                model_root=machine_root/'models'
                (model_root/'Jackrong'/'Negentropy-claude-opus-4.7-4B').mkdir(parents=True)
                (model_root/'coder3101'/'Qwen3.5-4B-heretic').mkdir(parents=True)
                system,_=self._managed_vllm_docs(machine_root,devices=[0])
                system['models']['root']=str(model_root)
                system_path=machine_root/'system.yaml'
                _dump(system_path,system)

                app=Application(PACKAGE_ROOT, ROOT)
                bundle=app.load_user_config(system_path,evaluation_path)
                model_id=next(iter(bundle.generated['model_ids']))
                model=app.specs.resolve('model',model_id)
                expected=str((model_root/'coder3101'/'Qwen3.5-4B-heretic').resolve())
                self.assertEqual(model['tokenizer'],{'ref':'coder3101/Qwen3.5-4B-heretic'})
                matrix_plan=app.build_user_matrix_plan(bundle)
                plan=matrix_plan['plans'][0]
                self.assertEqual(
                    plan['resolved']['specs']['deployment']['model_location']['tokenizer_path'],
                    expected,
                )
                self.assertEqual(
                    plan['resolved']['deployment_resolution']['resolved_tokenizer_path'],
                    expected,
                )

    def test_catalog_namespaced_backends_selects_only_active_backend_parameters(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root,devices=[0])
            catalog=root/'models'; catalog.mkdir()
            _dump(catalog/'portable.yaml',{
                'schema_version':'1.0','id':'portable','source':{'type':'hf','ref':'org/portable'},
                'backends':{
                    'vllm':{'max_model_len':8192,'gpu_memory_utilization':0.6},
                    # This is valid llama.cpp configuration but deliberately not
                    # part of the selected vLLM deployment parameters.
                    'llama_cpp':{'context_length':16384,'extra_args':['--mlock']},
                },
            })
            evaluation['models']=['portable']
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            app=Application(PACKAGE_ROOT, ROOT); bundle=app.load_user_config(system_path,evaluation_path)
            generated_id=next(iter(bundle.generated['model_ids']))
            params=bundle.matrix_spec['per_model_overrides'][generated_id]['deployment']['parameters']
            self.assertEqual(params,{'max_model_len':8192,'gpu_memory_utilization':0.6})
            self.assertNotIn('context_length',params)

    def test_catalog_namespaced_backend_parameters_use_selected_adapter_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root)
            catalog=root/'models'; catalog.mkdir()
            _dump(catalog/'invalid.yaml',{
                'schema_version':'1.0','id':'invalid','source':{'type':'hf','ref':'org/invalid'},
                'backends':{
                    'vllm':{'max_modle_len':8192},
                    # An unselected namespace is not interpreted by the vLLM
                    # adapter and may use its own Backend's parameter vocabulary.
                    'llama_cpp':{'context_length':8192},
                },
            })
            evaluation['models']=['invalid']
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            with self.assertRaisesRegex(Exception,r'backends\.vllm.*max_modle_len'):
                Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)

    def test_catalog_rejects_legacy_backend_together_with_namespaced_backends(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root)
            catalog=root/'models'; catalog.mkdir()
            _dump(catalog/'ambiguous.yaml',{
                'schema_version':'1.0','id':'ambiguous','source':{'type':'hf','ref':'org/ambiguous'},
                'backend':{'max_model_len':4096},
                'backends':{'vllm':{'max_model_len':8192}},
            })
            evaluation['models']=['ambiguous']
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            with self.assertRaisesRegex(Exception,'validation failed|不能同时配置'):
                Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)

    def test_catalog_legacy_backend_remains_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root)
            catalog=root/'models'; catalog.mkdir()
            _dump(catalog/'legacy.yaml',{
                'schema_version':'1.0','id':'legacy','source':{'type':'hf','ref':'org/legacy'},
                'backend':{'max_model_len':6144},
            })
            evaluation['models']=['legacy']
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            app=Application(PACKAGE_ROOT, ROOT); bundle=app.load_user_config(system_path,evaluation_path)
            generated_id=next(iter(bundle.generated['model_ids']))
            self.assertEqual(
                bundle.matrix_spec['per_model_overrides'][generated_id]['deployment']['parameters'],
                {'max_model_len':6144},
            )

    def test_catalog_model_evaluation_override_is_temporary_and_deep_merged(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root,devices=[0])
            catalog=root/'models'; catalog.mkdir()
            model_path=catalog/'llama.yaml'
            original={
                'schema_version':'1.0','id':'llama-long','label':'Llama Long',
                'source':{'type':'hf','ref':'org/llama','revision':'deadbeef'},
                'backends':{
                    'vllm':{'max_model_len':32768,'gpu_memory_utilization':0.7},
                    'llama_cpp':{'context_length':32768},
                },
                'environments':{'backend':'special-vllm'},
            }
            _dump(model_path,original)
            system['profiles']['environment']={
                'special-vllm':{'type':'venv','profile':str(root/'special')},
                'temporary-vllm':{'type':'venv','profile':str(root/'temporary')},
            }
            evaluation['models']=[{
                'id':'llama-long',
                'overrides':{
                    'backend':{'max_model_len':8192},
                    'environments':{'backend':'temporary-vllm'},
                },
            }]
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            app=Application(PACKAGE_ROOT, ROOT); bundle=app.load_user_config(system_path,evaluation_path)
            generated_id=next(iter(bundle.generated['model_ids']))
            patch=bundle.matrix_spec['per_model_overrides'][generated_id]
            self.assertEqual(patch['deployment']['parameters'],{
                'max_model_len':8192,'gpu_memory_utilization':0.7,
            })
            self.assertEqual(patch['platform']['backend_environment']['profile'],str((root/'temporary').resolve()))
            self.assertEqual(yaml.safe_load(model_path.read_text()),original)

    def test_catalog_typo_duplicate_id_and_override_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._managed_vllm_docs(root)
            catalog=root/'models'; (catalog/'nested').mkdir(parents=True)
            base={'schema_version':'1.0','id':'known','source':{'type':'hf','ref':'org/model'}}
            _dump(catalog/'one.yaml',base)
            evaluation['models']=['missing']
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            with self.assertRaisesRegex(Exception,'不存在的模型 catalog id'):
                Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)
            _dump(catalog/'nested'/'two.yaml',base)
            evaluation['models']=['known']; _dump(evaluation_path,evaluation)
            with self.assertRaisesRegex(Exception,'catalog id 重复'):
                Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)
            (catalog/'nested'/'two.yaml').unlink()
            evaluation['models']=[{'id':'known','overrides':{'id':'changed'}}]; _dump(evaluation_path,evaluation)
            with self.assertRaisesRegex(Exception,'validation failed|不能修改模型身份字段'):
                Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)
            evaluation['models']=[{'id':'known','overrides':{'source':{'type':'hf','ref':'other/model'}}}]
            _dump(evaluation_path,evaluation)
            with self.assertRaisesRegex(Exception,'validation failed|请新建一份 model catalog 配置'):
                Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)

    def test_explicit_catalog_and_inline_model_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._external_docs(root,two_profiles=False)
            catalog=root/'catalog'; catalog.mkdir()
            _dump(catalog/'registered.yaml',{
                'schema_version':'1.0','id':'registered','source':{'type':'hf','ref':'org/registered'},
            })
            evaluation['model_catalog']='catalog'
            evaluation['models']=['registered',{'id':'adhoc','ref':'org/adhoc','source_type':'hf'}]
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            bundle=Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)
            self.assertEqual(set(bundle.generated['model_ids'].values()),{'registered','adhoc'})

    def test_backend_evaluator_string_shorthand_selects_profiles(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._external_docs(root)
            evaluation.pop('profiles',None); evaluation.pop('evaluator',None)
            evaluation['backend']='remote_b'; evaluation['evaluator']='lm_eval_b'
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            bundle=Application(PACKAGE_ROOT, ROOT).load_user_config(system_path,evaluation_path)
            self.assertEqual(bundle.generated['selected_profiles'],{'backend':'remote_b','evaluator':'lm_eval_b'})

    def test_lm_eval_limit_accepts_integer_count_and_fraction(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve(); system,evaluation=self._external_docs(root,two_profiles=False)
            for value in (1,0.25):
                evaluation['evaluator']={'limit':value}
                system_path,evaluation_path=self._write_docs(root,system,evaluation)
                app=Application(PACKAGE_ROOT, ROOT); bundle=app.load_user_config(system_path,evaluation_path)
                evaluation_spec=app.specs.resolve('evaluation',bundle.generated['evaluation_id'])
                self.assertEqual(evaluation_spec['parameters']['limit'],value)

    def test_config_id_shorthand_resolves_systems_and_evaluations_directories(self):
        app=Application(PACKAGE_ROOT, ROOT)
        self.assertEqual(app._user_config_path('mlu',env_name='X',legacy_name='system.yaml',catalog_dir='systems'),ROOT/'config'/'systems'/'mlu.yaml')
        self.assertEqual(app._user_config_path('smoke_bbh_08b',env_name='X',legacy_name='evaluation.yaml',catalog_dir='evaluations'),ROOT/'config'/'evaluations'/'smoke_bbh_08b.yaml')

    def test_doctor_executes_backend_and_evaluator_probes_inside_selected_venvs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            backend_env=root/'backend-venv'; backend_bin=backend_env/'bin'; backend_bin.mkdir(parents=True)
            backend_python=backend_bin/'python'
            backend_python.write_text(
                '#!/bin/sh\n'
                'case "$1" in\n'
                '  */adapters/backend/vllm/preflight.py)\n'
                "    echo '{\"schema_version\":\"1.0\",\"status\":\"passed\",\"facts\":{\"backend\":\"vllm\",\"platform\":\"fixture\",\"weights_loaded\":false}}'\n"
                '    exit 0\n'
                '    ;;\n'
                'esac\n'
                f'exec {str(Path(sys.executable).resolve())!r} "$@"\n'
            )
            backend_python.chmod(0o755)
            vllm=backend_bin/'vllm'; vllm.write_text('#!/bin/sh\necho fake-vllm-1.0\n'); vllm.chmod(0o755)
            eval_env=root/'eval-venv'; eval_bin=eval_env/'bin'; eval_bin.mkdir(parents=True)
            (eval_bin/'python').symlink_to(Path(sys.executable).resolve())
            lm_root=root/'lm-harness'; pkg=lm_root/'lm_eval'; pkg.mkdir(parents=True); (pkg/'__init__.py').write_text('VALUE=1\n')
            model_root=root/'models'; (model_root/'model-A').mkdir(parents=True)
            system={
                'schema_version':'1.2','system':{'name':'doctor-host'},
                'profiles':{
                    'defaults':{'hardware':'local','backend':'vllm_local','evaluator':'lm_eval_local'},
                    'environment':{
                        'backend':{'type':'venv','profile':str(backend_env)},
                        'eval':{'type':'venv','profile':str(eval_env)},
                    },
                    'hardware':{'local':{'type':'cpu','runtime':{'type':'cpu'}}},
                    'backend':{'vllm_local':{'type':'vllm','mode':'managed','compatibility':{'runtime_families':['cpu']},'environment':'backend','executable':'vllm'}},
                    'evaluator':{'lm_eval_local':{'type':'lm_eval','root':str(lm_root),'environment':'eval','parameters':{'require_clean_framework':False}}},
                },
                'models':{'root':str(model_root)},
                'paths':{'cache':str(root/'cache'),'results':str(root/'results')},
            }
            evaluation={'schema_version':'1.2','models':['model-A'],'benchmarks':['mmlu'],'resources':{'devices':[0]}}
            system_path,evaluation_path=self._write_docs(root,system,evaluation)
            proc=subprocess.run([sys.executable,str(ROOT/'eval-manager'),'doctor','--format','json','--system-config',str(system_path),'--evaluation-config',str(evaluation_path)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            out=json.loads(proc.stdout); self.assertTrue(out['ok'])
            checks=out['runs'][0]['checks']
            self.assertEqual(checks['backend_environment']['status'],'ok')
            probes=checks['backend_environment']['preflight']['probes']
            self.assertEqual([p['id'] for p in probes],['backend.import','model.config'])
            self.assertIn('fake-vllm-1.0',probes[0]['stdout'])
            self.assertFalse(probes[1]['result']['facts']['weights_loaded'])
            self.assertEqual(checks['evaluator_environment']['status'],'ok')


if __name__ == "__main__":
    unittest.main()
