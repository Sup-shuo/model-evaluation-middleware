from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "model_evaluation_build_release",
    ROOT / "scripts" / "build_release.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILD_RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_RELEASE)


class BuildReleaseVersionTests(unittest.TestCase):
    def test_pep440_accepts_supported_release_phases(self):
        cases = {
            "4.1.0-alpha31": "4.1.0a31",
            "4.1.0-beta1": "4.1.0b1",
            "4.1.0-rc1": "4.1.0rc1",
            "4.1.0": "4.1.0",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(BUILD_RELEASE.pep440(source), expected)

    def test_pep440_rejects_unknown_or_incomplete_versions(self):
        for value in ("4.1", "4.1.0-alpha", "4.1.0-preview1", "04.1.0"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                BUILD_RELEASE.pep440(value)

    def test_check_version_accepts_matching_stable_and_prerelease_forms(self):
        cases = {
            "4.1.0-alpha31": "4.1.0a31",
            "4.1.0-beta1": "4.1.0b1",
            "4.1.0-rc1": "4.1.0rc1",
            "4.1.0": "4.1.0",
        }
        for source, packaged in cases.items():
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                package = root / "model_evaluation"
                package.mkdir()
                (package / "VERSION.txt").write_text(source + "\n", encoding="utf-8")
                (root / "pyproject.toml").write_text(
                    f'[project]\nversion = "{packaged}"\n',
                    encoding="utf-8",
                )
                self.assertEqual(BUILD_RELEASE.check_version(root), source)

    def test_check_version_rejects_mismatched_package_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "model_evaluation"
            package.mkdir()
            (package / "VERSION.txt").write_text("4.1.0-rc1\n", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                '[project]\nversion = "4.1.0"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "version mismatch"):
                BUILD_RELEASE.check_version(root)


if __name__ == "__main__":
    unittest.main()
