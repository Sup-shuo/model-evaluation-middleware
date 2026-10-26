from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "model_evaluation"
sys.path.insert(0, str(ROOT))

import model_evaluation.core.planner as planner_module
from model_evaluation.core.errors import AdapterProtocolError, CompatibilityError, SchemaValidationError
from model_evaluation.core.orchestrator import Orchestrator
from model_evaluation.core.registry.operation_contracts import validate_operation_output
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.sdk.runtime import AdapterError

REVISION = "0123456789abcdef0123456789abcdef01234567"


def load_adapter(kind, name):
    path = PACKAGE / "adapters" / kind / name / "impl.py"
    spec = importlib.util.spec_from_file_location("test_metadata_" + name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetadataOnlyTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="model-eval-metadata-test-")
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name).resolve()
        self.schemas = SchemaStore(PACKAGE / "schemas")
        self.binding = load_adapter("binding", "lm_eval.bbh")
        self.evaluator = load_adapter("evaluator", "lm_eval")
        self.source = self.root / "harness" / "lm_eval" / "tasks" / "leaderboard" / "bbh_mc"
        self.source.mkdir(parents=True)
        (self.source / "_leaderboard_bbh.yaml").write_text(
            "group: leaderboard_bbh\ntask:\n  - leaderboard_bbh_boolean_expressions\n"
        )
        (self.source / "_fewshot_template_yaml").write_text("dataset_path: upstream\n")
        (self.source / "boolean_expressions.yaml").write_text(
            "task: leaderboard_bbh_boolean_expressions\ndataset_name: boolean_expressions\n"
            "include: _fewshot_template_yaml\n"
        )
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "boolean_expressions.json").write_text(
            json.dumps({"examples": [{"input": "True", "target": "True"}]})
        )
        self.parameters = {
            "tool_root": str(self.root / "harness"),
            "provenance_policy": "metadata-only",
            "framework_revision": REVISION,
            "require_clean_framework": False,
        }
        self.inputs = {
            "benchmark": {
                "id": "bbh",
                "metrics": ["accuracy"],
                "protocol": {"fewshot": 3},
                "dataset": {"provider": "bbh_local", "parameters": {"integrity_policy": "basic"}},
            },
            "dataset_artifact": {
                "dataset_id": "bbh",
                "revision": "fixture-revision",
                "root": str(self.data),
                "files": [{"path": str(self.data / "boolean_expressions.json")}],
                "metadata": {"integrity_policy": "basic", "content_verified": False},
            },
            "evaluation": {"framework": {"adapter": "lm_eval"}, "parameters": self.parameters},
            "staging_root": str(self.root / "staging"),
        }

    def forbid_hashes(self):
        return patch.object(hashlib, "sha256", side_effect=AssertionError("unexpected hash calculation"))

    def build_metadata_task(self):
        with self.forbid_hashes(), patch.object(
            self.binding.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=REVISION)
        ):
            return self.binding.build_task(self.inputs, {})

    def test_metadata_binding_never_hashes_files_or_protocol(self):
        with self.forbid_hashes(), patch.object(
            self.binding.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=REVISION)
        ):
            task = self.binding.build_task(self.inputs, {})
            fingerprint = self.binding.protocol_fingerprint(self.inputs, {})
        self.assertIsNone(task["protocol_fingerprint"])
        self.assertEqual(fingerprint, {"protocol_fingerprint": None, "recording_mode": "metadata-only"})
        self.assertEqual(task["provenance"]["framework_revision"], REVISION)
        self.assertFalse(task["provenance"]["source_fingerprinted"])
        self.assertTrue(task["artifacts"])
        self.assertTrue(all(set(item) == {"path"} for item in task["artifacts"]))
        manifest = json.loads((Path(task["task_root"]) / "PROTOCOL_MANIFEST.json").read_text())
        self.assertEqual(manifest["recording_mode"], "metadata-only")
        self.assertEqual(len(manifest["rows"]), 1)
        self.assertEqual(len(manifest["source_task_files"]), 3)
        self.assertNotIn("source_protocol_sha256", manifest)
        self.assertNotIn("source_yaml_sha256", manifest["rows"][0])
        self.schemas.validate("framework_task_artifact", task)
        validate_operation_output(
            self.schemas, "binding", "build_task", task, input_obj=self.inputs
        )
        validate_operation_output(
            self.schemas, "binding", "protocol_fingerprint", fingerprint, input_obj=self.inputs
        )

    def test_dataset_basic_never_hashes_content(self):
        dataset = load_adapter("dataset", "bbh_local")
        with self.forbid_hashes(), patch.object(
            dataset, "_expected", return_value=[("boolean_expressions", 1, "0" * 64)]
        ):
            artifact = dataset._artifact(self.data, "basic")
            verified = dataset.verify(
                {"benchmark": self.inputs["benchmark"], "artifact": artifact}, {}
            )
            resolution = dataset.resolve({"benchmark": self.inputs["benchmark"]}, {})
        self.assertTrue(verified["valid"])
        self.assertNotIn("fingerprint", artifact)
        self.assertNotIn("sha256", artifact["files"][0])
        self.assertNotIn("expected_manifest_sha256", resolution)

    def test_metadata_mode_keeps_path_checks(self):
        task = self.build_metadata_task()
        Orchestrator._verify_task_artifacts(task, Path(self.inputs["staging_root"]))
        task["artifacts"][0]["path"] = str(self.root / "outside.yaml")
        with self.assertRaises(CompatibilityError):
            Orchestrator._verify_task_artifacts(task, Path(self.inputs["staging_root"]))

    def test_metadata_mode_rejects_missing_and_symlink_artifacts(self):
        task = self.build_metadata_task()
        task["artifacts"] = [{"path": str(Path(task["task_root"]) / "missing.yaml")}]
        with self.assertRaises(CompatibilityError):
            Orchestrator._verify_task_artifacts(task, Path(self.inputs["staging_root"]))
        linked = Path(task["task_root"]) / "linked.yaml"
        linked.symlink_to(self.source / "_leaderboard_bbh.yaml")
        task["artifacts"] = [{"path": str(linked)}]
        with self.assertRaises(CompatibilityError):
            Orchestrator._verify_task_artifacts(task, Path(self.inputs["staging_root"]))

    def test_null_digest_requires_explicit_mode_in_schema(self):
        task = self.build_metadata_task()
        for mutation in ("no_provenance", "strict", "fingerprinted"):
            changed = copy.deepcopy(task)
            if mutation == "no_provenance":
                changed.pop("provenance")
            elif mutation == "strict":
                changed["provenance"]["policy"] = "strict"
            else:
                changed["provenance"]["source_fingerprinted"] = True
            with self.subTest(mutation=mutation), self.assertRaises(SchemaValidationError):
                self.schemas.validate("framework_task_artifact", changed)

    def test_metadata_schema_requires_nonempty_task_root(self):
        task = self.build_metadata_task()
        for value in (None, ""):
            changed = copy.deepcopy(task)
            if value is None:
                changed.pop("task_root")
            else:
                changed["task_root"] = value
            with self.subTest(value=value), self.assertRaises(SchemaValidationError):
                self.schemas.validate("framework_task_artifact", changed)

    def test_legacy_policy_rejects_contradictory_recording_mode(self):
        output = {"protocol_fingerprint": "a" * 64, "recording_mode": "metadata-only"}
        with self.forbid_hashes():
            for policy in ("strict", "migration"):
                inputs = copy.deepcopy(self.inputs)
                inputs["evaluation"]["parameters"]["provenance_policy"] = policy
                with self.subTest(policy=policy), self.assertRaises(AdapterProtocolError):
                    validate_operation_output(
                        self.schemas, "binding", "protocol_fingerprint", output, input_obj=inputs
                    )

    def test_metadata_schema_rejects_digest_claim(self):
        task = self.build_metadata_task()
        task["protocol_fingerprint"] = "a" * 64  # Fixed legacy-shaped fixture; never computed.
        with self.assertRaises(SchemaValidationError):
            self.schemas.validate("framework_task_artifact", task)

    def test_runtime_rejects_null_without_requested_mode(self):
        task = self.build_metadata_task()
        legacy_inputs = copy.deepcopy(self.inputs)
        legacy_inputs["evaluation"]["parameters"]["provenance_policy"] = "migration"
        for operation, output in (
            ("build_task", task),
            ("protocol_fingerprint", {"protocol_fingerprint": None, "recording_mode": "metadata-only"}),
        ):
            with self.subTest(operation=operation), self.assertRaises(AdapterProtocolError):
                validate_operation_output(
                    self.schemas, "binding", operation, output, input_obj=legacy_inputs
                )
        for output in (
            {"protocol_fingerprint": None},
            {"protocol_fingerprint": "a" * 64, "recording_mode": "metadata-only"},
        ):
            with self.assertRaises(AdapterProtocolError):
                validate_operation_output(
                    self.schemas, "binding", "protocol_fingerprint", output, input_obj=self.inputs
                )

    def test_existing_hashed_artifact_remains_readable(self):
        task = self.build_metadata_task()
        task["protocol_fingerprint"] = "a" * 64
        task["provenance"].update(policy="strict", strict=True, source_fingerprinted=True)
        for artifact in task["artifacts"]:
            artifact["sha256"] = "b" * 64
        with self.forbid_hashes():
            self.schemas.validate("framework_task_artifact", task)
            Orchestrator._verify_task_artifacts(task, Path(self.inputs["staging_root"]))

    def test_framework_revision_is_observed_without_git_status(self):
        calls = []
        def git_read(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[-2:], ["rev-parse", "HEAD"])
            return SimpleNamespace(returncode=0, stdout=REVISION)
        with self.forbid_hashes(), patch.object(self.evaluator.subprocess, "run", side_effect=git_read):
            root, revision, dirty = self.evaluator._framework_source(self.parameters)
        self.assertEqual(root, self.root / "harness")
        self.assertEqual(revision, REVISION)
        self.assertIsNone(dirty)
        self.assertEqual(len(calls), 1)

    def test_metadata_mode_rejects_clean_tree_requirement(self):
        parameters = dict(self.parameters, require_clean_framework=True)
        with self.forbid_hashes(), patch.object(self.evaluator.subprocess, "run") as process:
            with self.assertRaisesRegex(AdapterError, "conflicts"):
                self.evaluator._framework_source(parameters)
            process.assert_not_called()

    def test_observed_revision_mismatch_is_rejected(self):
        with self.forbid_hashes(), patch.object(
            self.evaluator.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="f" * 40)
        ):
            with self.assertRaisesRegex(AdapterError, "revision mismatch"):
                self.evaluator._framework_source(self.parameters)
            with self.assertRaisesRegex(AdapterError, "revision mismatch"):
                self.binding._revision(self.root / "harness", self.inputs["evaluation"])

    def test_missing_revision_never_uses_declared_revision_as_observed(self):
        with self.forbid_hashes(), patch.object(
            self.evaluator.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")
        ), patch.object(self.evaluator, "read_git_head", return_value=None):
            with self.assertRaisesRegex(AdapterError, "revision mismatch"):
                self.evaluator._framework_source(self.parameters)

    def test_generic_binding_rejects_mode_before_any_hash(self):
        generic = load_adapter("binding", "lm_eval")
        with self.forbid_hashes():
            for operation in ("requirements", "build_task", "protocol_fingerprint"):
                with self.subTest(operation=operation), self.assertRaisesRegex(AdapterError, "bbh"):
                    generic.OPERATIONS[operation](self.inputs, {})

    def test_metadata_mode_rejects_strict_dataset_in_all_binding_operations(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["benchmark"]["dataset"]["parameters"]["integrity_policy"] = "strict"
        with self.forbid_hashes():
            for operation in ("requirements", "build_task", "protocol_fingerprint"):
                with self.subTest(operation=operation), self.assertRaisesRegex(AdapterError, "basic"):
                    self.binding.OPERATIONS[operation](inputs, {})

    def test_planner_rejects_strict_dataset_before_dataset_resolve(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["benchmark"]["dataset"]["parameters"]["integrity_policy"] = "strict"
        evaluation = dict(inputs["evaluation"], binding={"adapter": "lm_eval.bbh"})
        deployment = {"id": "fixture", "management": {"mode": "external"}, "backend": {"adapter": "fixture"}}
        platform = {"evaluation_environment": {"provider": "fixture", "profile": "fixture"}}
        bundle = {
            "platform": platform, "deployment": deployment, "evaluation": evaluation,
            "benchmark": inputs["benchmark"], "model": {"id": "fixture"},
        }
        calls = []
        binding = self.binding
        class Client:
            last_warnings = []
            def __init__(self, kind):
                self.identity = SimpleNamespace(kind=kind, name="fixture", version="1.0")
            def invoke(self, operation, input_obj, **kwargs):
                kind = self.identity.kind
                calls.append((kind, operation))
                if kind == "dataset":
                    raise AssertionError("Dataset resolve must not run for the conflicting policy")
                if kind == "binding":
                    return binding.requirements(input_obj, {})
                if kind == "environment":
                    return {}
                return {"schema_version": "1.0", "requirements": []}
        clients = {kind: Client(kind) for kind in ("backend", "environment", "dataset", "binding", "evaluator")}
        planner = planner_module.Planner(
            project_root=ROOT, schemas=self.schemas,
            specs=SimpleNamespace(resolve_bundle=lambda run: bundle),
            registry=SimpleNamespace(get=lambda kind, name: clients[kind]),
        )
        with self.forbid_hashes(), patch.object(
            planner_module, "validate_run_overrides"
        ), patch.object(
            planner_module, "resolve_deployment_profile", return_value=(deployment, {})
        ), patch.object(
            planner_module, "resolve_evaluation_profile", return_value=(evaluation, {})
        ), patch.object(
            planner_module, "assess_model_provenance", return_value={}
        ), patch.object(
            planner_module, "adapter_parameters", return_value={}
        ), patch.object(
            planner_module, "facts_from_environment", return_value={}
        ):
            with self.assertRaisesRegex(AdapterError, "basic"):
                planner.build({})
        self.assertIn(("binding", "requirements"), calls)
        self.assertNotIn(("dataset", "resolve"), calls)

    def test_normalized_result_retains_explicit_recording_mode(self):
        task = self.build_metadata_task()
        raw = {"results": {task["task_id"]: {"acc_norm,none": 0.5}}}
        inputs = {
            "raw_result_root": str(self.root),
            "task": task,
            "run_metadata": {"run_id": "fixture-run", "model": {"id": "fixture"}, "benchmark": {"id": "bbh"}},
        }
        with self.forbid_hashes(), patch.object(
            self.evaluator, "_find_result", return_value=(self.root / "native.json", raw)
        ), patch.object(self.evaluator, "_sample_artifacts", return_value=[]), patch.object(
            self.evaluator, "_token_usage", return_value=None
        ):
            result = self.evaluator.normalize(inputs, {})
        self.assertIsNone(result["metadata"]["protocol_fingerprint"])
        self.assertEqual(result["metadata"]["protocol_recording_mode"], "metadata-only")


if __name__ == "__main__":
    unittest.main()
