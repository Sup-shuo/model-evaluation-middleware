from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.config.parsing import load_yaml_strict
from model_evaluation.core.registry.adapter_registry import (
    AdapterRegistry,
    _adapter_process_env,
)
from model_evaluation.core.registry.plugin_discovery import (
    ENTRY_POINT_GROUP,
    AdapterCandidate,
    index_candidates,
)
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.matrix_config import MatrixSchemas
from model_evaluation.environment_snapshot import (
    controller_environment_snapshot,
    requirements_lock_text,
)
from model_evaluation.onboarding import initialize_project
from model_evaluation.commands.adapters import check_adapter_root


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "model_evaluation"


class _FakeDistribution:
    def __init__(self, root: Path, name: str = "example-adapter-package"):
        self.root = root
        self.metadata = {"Name": name}

    def locate_file(self, value: Path) -> Path:
        return self.root / value


class _FakeEntryPoints(list):
    def select(self, *, group: str):
        return self if group == ENTRY_POINT_GROUP else []


def _plugin_entry_point(root: Path, *, name: str = "device.thirdparty"):
    distribution = _FakeDistribution(root)
    return SimpleNamespace(
        name=name,
        module="example_plugin.adapters.device.thirdparty",
        attr=None,
        extras=(),
        dist=distribution,
    )


def _write_plugin(root: Path) -> Path:
    directory = root / "example_plugin" / "adapters" / "device" / "thirdparty"
    directory.mkdir(parents=True)
    manifest = {
        "adapter_api": "1.0",
        "kind": "device",
        "name": "thirdparty",
        "version": "1.0.0",
        "operations": ["probe"],
        "schema_versions": {"device_descriptor": "1.0"},
    }
    entry = directory / "adapter"
    entry.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = manifest ]; then\n"
        f"  printf '%s\\n' '{json.dumps(manifest, separators=(',', ':'))}'\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    entry.chmod(0o755)
    # Discovery must not import this package.
    (root / "example_plugin" / "__init__.py").write_text(
        "raise RuntimeError('plugin package must not be imported during discovery')\n",
        encoding="utf-8",
    )
    return entry


class PluginAndOnboardingTests(unittest.TestCase):
    def setUp(self):
        self.schemas = SchemaStore(PACKAGE_ROOT / "schemas")
        self.user_schemas = MatrixSchemas(PACKAGE_ROOT / "schemas" / "user")

    def test_installed_adapter_entry_point_is_discovered_without_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            builtin = root / "builtin"
            builtin.mkdir()
            entry = _write_plugin(root)
            points = _FakeEntryPoints([_plugin_entry_point(root)])
            with patch(
                "model_evaluation.core.registry.plugin_discovery.metadata.entry_points",
                return_value=points,
            ):
                registry = AdapterRegistry(builtin, self.schemas)
                identity = registry.get("device", "thirdparty").identity
                self.assertEqual(identity.path, entry.resolve())
                self.assertEqual(identity.version, "1.0.0")

    def test_builtin_adapters_launch_with_the_controller_python(self):
        env = _adapter_process_env()
        self.assertEqual(
            Path(env["MODEL_EVAL_CONTROLLER_PYTHON"]),
            Path(sys.executable).resolve(),
        )
        entries = sorted((PACKAGE_ROOT / "adapters").glob("*/*/adapter"))
        self.assertEqual(len(entries), 25)
        for entry in entries:
            source = entry.read_text(encoding="utf-8")
            self.assertIn(
                'exec "${MODEL_EVAL_CONTROLLER_PYTHON:-python3}" -B -S',
                source,
                entry,
            )

    def test_duplicate_adapter_candidates_are_rejected(self):
        first = AdapterCandidate("device", "same", Path("/one"), "builtin")
        second = AdapterCandidate("device", "same", Path("/two"), "entry-point:two")
        with self.assertRaisesRegex(ConfigError, "duplicate adapter device/same"):
            index_candidates([first, second])

    def test_adapter_check_validates_an_isolated_external_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            external = root / "external"
            adapter_dir = external / "device" / "thirdparty"
            adapter_dir.mkdir(parents=True)
            source_entry = _write_plugin(root)
            (adapter_dir / "adapter").write_bytes(source_entry.read_bytes())
            (adapter_dir / "adapter").chmod(0o755)
            report = check_adapter_root(external, self.schemas)
            self.assertTrue(report["ok"])
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["adapters"][0]["kind"], "device")
            self.assertEqual(report["adapters"][0]["name"], "thirdparty")

    def test_init_creates_minimal_project_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            created = initialize_project(root, hardware="nvidia")
            self.assertEqual(len(created), 4)
            self.assertTrue((root / "config" / "system.yaml").is_file())
            self.assertTrue((root / "config" / "models" / "example.yaml").is_file())
            self.assertTrue((root / "config" / "evaluation.yaml").is_file())
            self.assertTrue((root / "results").is_dir())
            generated_documents = (
                ("user_system", root / "config" / "system.yaml"),
                ("user_model", root / "config" / "models" / "example.yaml"),
                ("user_evaluation", root / "config" / "evaluation.yaml"),
            )
            for schema_name, path in generated_documents:
                document = load_yaml_strict(path.read_text(encoding="utf-8"))
                self.user_schemas.validate(schema_name, document)
            original = (root / "config" / "system.yaml").read_text(encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "refuses to overwrite"):
                initialize_project(root, hardware="mlu")
            self.assertEqual(
                (root / "config" / "system.yaml").read_text(encoding="utf-8"),
                original,
            )

    def test_controller_environment_snapshot_is_sorted_and_lockable(self):
        snapshot = controller_environment_snapshot()
        names = [package["name"] for package in snapshot["packages"]]
        self.assertEqual(names, sorted(set(names)))
        self.assertEqual(snapshot["scope"], "controller-python-environment")
        lock = requirements_lock_text(snapshot)
        for package in snapshot["packages"]:
            self.assertIn(f"{package['name']}=={package['version']}", lock)


if __name__ == "__main__":
    unittest.main()
