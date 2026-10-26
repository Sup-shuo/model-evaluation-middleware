from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_evaluation.adapters.runtime.cann import impl as cann
from model_evaluation.sdk.runtime import AdapterError


class CannRuntimeEnvironmentTests(unittest.TestCase):
    def test_resolves_existing_runtime_paths_without_shell_inheritance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            relatives = (
                "lib64", "lib64/plugin/opskernel", "lib64/plugin/nnengine",
                "opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64",
                "python/site-packages", "toolkit", "bin", "tools/ccec_compiler/bin",
            )
            for relative in relatives:
                (root / relative).mkdir(parents=True, exist_ok=True)
            with patch.object(cann.platform, "machine", return_value="aarch64"), patch.dict(
                os.environ, {"ASCEND_OPP_PATH": "/unrelated/opp", "PYTHONPATH": "/unrelated/python"}
            ):
                resolved = cann.resolve_environment({"parameters": {"root": str(root)}}, {})["env_patch"]
            self.assertEqual(resolved["set"], {
                "ASCEND_HOME_PATH": str(root), "ASCEND_AICPU_PATH": str(root),
                "ASCEND_TOOLKIT_HOME": str(root), "ASCEND_OPP_PATH": str(root / "opp"),
                "TOOLCHAIN_HOME": str(root / "toolkit"),
            })
            self.assertEqual(resolved["prepend_path"]["PYTHONPATH"], [
                str(root / "python/site-packages"),
                str(root / "opp/built-in/op_impl/ai_core/tbe"),
            ])
            self.assertEqual(resolved["prepend_path"]["PATH"], [
                str(root / "bin"), str(root / "tools/ccec_compiler/bin"),
            ])
            self.assertEqual(resolved["prepend_path"]["LD_LIBRARY_PATH"], [
                str(root / relative) for relative in relatives[:4]
            ])

    def test_sparse_older_root_does_not_inject_nonexistent_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "lib64").mkdir()
            resolved = cann.resolve_environment({"parameters": {"root": str(root)}}, {})["env_patch"]
            self.assertEqual(resolved["set"], {
                "ASCEND_HOME_PATH": str(root), "ASCEND_AICPU_PATH": str(root),
                "ASCEND_TOOLKIT_HOME": str(root),
            })
            self.assertEqual(resolved["prepend_path"], {"LD_LIBRARY_PATH": [str(root / "lib64")]})

    def test_unconfigured_runtime_keeps_empty_patch(self):
        with patch.dict(os.environ, {"ASCEND_HOME_PATH": "", "ASCEND_TOOLKIT_HOME": ""}):
            self.assertEqual(cann.resolve_environment({"parameters": {}}, {}), {"env_patch": {}})

    def test_missing_configured_root_remains_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(AdapterError) as raised:
                cann.resolve_environment({"parameters": {"root": str(Path(temporary) / "missing")}}, {})
            self.assertEqual(raised.exception.code, "DEPENDENCY_MISSING")


if __name__ == "__main__":
    unittest.main()
