from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml
from unittest.mock import patch

from model_evaluation.core.app import Application
from model_evaluation.core.config.catalog import resolve_config_reference, scan_config_catalog
from model_evaluation.core.config.migration import migrate_document, migrate_entries
from model_evaluation.core.files import atomic_text as real_atomic_text
from model_evaluation.core.config.model_catalog import resolve_model_entries
from model_evaluation.core.errors import ConfigError
from model_evaluation.commands.configuration import handle_config_command


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "model_evaluation"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


class ConfigManagementTests(unittest.TestCase):
    def test_nested_system_and_evaluation_ids_resolve_below_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            system = project / "config" / "systems" / "lab" / "mlu.yaml"
            evaluation = project / "config" / "evaluations" / "bbh" / "smoke.yaml"
            _write(system, {"schema_version": "1.3"})
            _write(evaluation, {"schema_version": "1.3"})
            self.assertEqual(
                resolve_config_reference(project, "lab/mlu", catalog_dir="systems").resolve(),
                system.resolve(),
            )
            self.assertEqual(
                resolve_config_reference(project, "bbh/smoke", catalog_dir="evaluations").resolve(),
                evaluation.resolve(),
            )
            with self.assertRaises(ValueError):
                resolve_config_reference(project, "../outside", catalog_dir="systems")

    def test_model_catalog_listing_uses_declared_global_id(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            _write(
                project / "config" / "models" / "qwen" / "small.yaml",
                {"schema_version": "1.0", "id": "qwen-small"},
            )
            entries = scan_config_catalog(project, "model")
            self.assertEqual([(entry.kind, entry.reference) for entry in entries], [("model", "qwen-small")])

    def test_catalog_ignores_metadata_directories_at_every_depth(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            _write(
                project / "config" / "models" / "valid.yaml",
                {"schema_version": "1.0", "id": "valid"},
            )
            _write(
                project / "config" / "models" / "._metadata" / "ignored.yaml",
                {"schema_version": "1.0", "id": "ignored"},
            )
            entries = scan_config_catalog(project, "model")
            self.assertEqual([entry.reference for entry in entries], ["valid"])

    def test_config_list_returns_nonzero_when_catalog_contains_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            path = project / "config" / "models" / "invalid.yaml"
            path.parent.mkdir(parents=True)
            path.write_text("schema_version: [", encoding="utf-8")
            args = Namespace(
                cmd="config",
                config_action="list",
                kind="model",
                format="json",
            )
            with redirect_stdout(StringIO()), self.assertRaisesRegex(SystemExit, "2"):
                handle_config_command(args, Application(PACKAGE_ROOT, project))

    def test_nested_evaluation_still_discovers_project_model_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            _write(
                project / "config" / "models" / "qwen" / "small.yaml",
                {
                    "schema_version": "1.0",
                    "id": "qwen-small",
                    "source": {"type": "local", "ref": "Qwen/Small"},
                },
            )
            evaluation_path = project / "config" / "evaluations" / "teams" / "smoke.yaml"
            _write(evaluation_path, {"schema_version": "1.3"})
            rows, catalog_root, enabled = resolve_model_entries(
                Application(PACKAGE_ROOT, ROOT),
                {"models": ["qwen-small"]},
                evaluation_path,
            )
            self.assertTrue(enabled)
            self.assertEqual(catalog_root, (project / "config" / "models").resolve())
            self.assertEqual(rows[0]["id"], "qwen-small")

    def test_evaluation_12_migration_requires_explicit_framework_profiles(self):
        document = {
            "schema_version": "1.2",
            "models": ["m"],
            "benchmarks": ["bbh"],
            "backend": {"seed": 1},
            "evaluator": {"limit": 1},
        }
        with self.assertRaises(ConfigError):
            migrate_document("evaluation", document)
        migrated, changed = migrate_document(
            "evaluation",
            document,
            backend_profile="vllm",
            evaluator_profile="lm_eval",
        )
        self.assertTrue(changed)
        self.assertEqual(migrated["schema_version"], "1.3")
        self.assertEqual(migrated["backend"], {"profile": "vllm", "parameters": {"seed": 1}})
        self.assertEqual(migrated["evaluator"], {"profile": "lm_eval", "parameters": {"limit": 1}})
        Application(PACKAGE_ROOT, ROOT).matrix_schemas.validate("user_evaluation", migrated)

    def test_evaluation_12_migration_preserves_explicit_legacy_profile_selection(self):
        document = {
            "schema_version": "1.2",
            "profiles": {"hardware": "gpu", "backend": "served", "evaluator": "scope"},
            "models": ["m"],
            "benchmarks": ["bbh"],
            "backend": {},
            "evaluator": {},
        }
        migrated, _ = migrate_document("evaluation", document)
        self.assertEqual(migrated["profiles"], {"hardware": "gpu"})
        self.assertEqual(migrated["backend"], {"profile": "served"})
        self.assertEqual(migrated["evaluator"], {"profile": "scope"})

    def test_system_12_migration_removes_hidden_framework_defaults(self):
        document = {
            "schema_version": "1.2",
            "system": {"name": "host"},
            "profiles": {
                "defaults": {"hardware": "gpu", "backend": "vllm", "evaluator": "lm_eval"},
                "hardware": {},
                "backend": {},
                "evaluator": {},
            },
            "paths": {"cache": "/tmp/cache"},
        }
        migrated, changed = migrate_document("system", document)
        self.assertTrue(changed)
        self.assertEqual(migrated["schema_version"], "1.3")
        self.assertEqual(migrated["profiles"]["defaults"], {"hardware": "gpu"})

    def test_catalog_write_migration_is_preflighted_before_any_file_changes(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            system_path = project / "config" / "systems" / "legacy.yaml"
            evaluation_path = project / "config" / "evaluations" / "legacy.yaml"
            _write(
                system_path,
                {
                    "schema_version": "1.2",
                    "system": {"name": "legacy"},
                    "profiles": {"defaults": {"backend": "vllm", "evaluator": "lm_eval"}},
                    "paths": {"cache": "/tmp/cache"},
                },
            )
            _write(
                evaluation_path,
                {
                    "schema_version": "1.2",
                    "models": ["m"],
                    "benchmarks": ["bbh"],
                    "backend": {},
                    "evaluator": {},
                },
            )
            before = system_path.read_text(encoding="utf-8")
            args = Namespace(
                cmd="config",
                config_action="migrate",
                kind=None,
                write=True,
                backend_profile=None,
                evaluator_profile=None,
                format="json",
            )
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
                handle_config_command(args, Application(PACKAGE_ROOT, project))
            self.assertEqual(system_path.read_text(encoding="utf-8"), before)

    def test_catalog_write_migration_rolls_back_earlier_files_on_io_failure(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            first = project / "config" / "systems" / "a.yaml"
            second = project / "config" / "systems" / "b.yaml"
            document = {
                "schema_version": "1.2",
                "system": {"name": "legacy"},
                "profiles": {"defaults": {"backend": "vllm"}},
                "paths": {"cache": "/tmp/cache"},
            }
            _write(first, document)
            _write(second, document)
            before = {path: path.read_text(encoding="utf-8") for path in (first, second)}
            entries = scan_config_catalog(project, "system")
            calls = 0

            def flaky_write(path, text):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated write failure")
                real_atomic_text(path, text)

            with patch(
                "model_evaluation.core.config.migration.atomic_text",
                side_effect=flaky_write,
            ), self.assertRaisesRegex(ConfigError, "rolled back"):
                migrate_entries(entries, write=True)

            for path in (first, second):
                self.assertEqual(path.read_text(encoding="utf-8"), before[path])

    def test_single_write_migration_preserves_file_mode(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            path = project / "config" / "systems" / "system.yaml"
            _write(
                path,
                {
                    "schema_version": "1.2",
                    "system": {"name": "legacy"},
                    "profiles": {"defaults": {"backend": "vllm"}},
                    "paths": {"cache": "/tmp/cache"},
                },
            )
            path.chmod(0o640)
            entry = scan_config_catalog(project, "system")[0]
            migrate_entries([entry], write=True)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_migration_rolls_back_when_permission_restore_fails(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td)
            path = project / "config" / "systems" / "system.yaml"
            document = {
                "schema_version": "1.2",
                "system": {"name": "legacy"},
                "profiles": {"defaults": {"backend": "vllm"}},
                "paths": {"cache": "/tmp/cache"},
            }
            _write(path, document)
            before = path.read_text(encoding="utf-8")
            entry = scan_config_catalog(project, "system")[0]
            real_chmod = Path.chmod
            calls = 0

            def flaky_chmod(target, mode):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("simulated chmod failure")
                return real_chmod(target, mode)

            with patch(
                "model_evaluation.core.config.migration.Path.chmod",
                autospec=True,
                side_effect=flaky_chmod,
            ), self.assertRaisesRegex(ConfigError, "rolled back"):
                migrate_entries([entry], write=True)

            self.assertEqual(path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
