from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from model_evaluation.adapters.runtime.cann import impl as cann
from model_evaluation.adapters.runtime.cuda import impl as cuda
from model_evaluation.adapters.runtime.neuware import impl as neuware
from model_evaluation.adapters.runtime.rocm import impl as rocm


class RuntimePathPortabilityTests(unittest.TestCase):
    def test_runtime_roots_have_no_implicit_installation_directory(self):
        with patch.dict(
            os.environ,
            {
                "CUDA_HOME": "",
                "CUDA_PATH": "",
                "ROCM_PATH": "",
                "NEUWARE_HOME": "",
                "ASCEND_HOME_PATH": "",
                "ASCEND_TOOLKIT_HOME": "",
            },
            clear=False,
        ):
            self.assertIsNone(cuda._root({"parameters": {}}))
            self.assertIsNone(rocm._root({"parameters": {}}))
            self.assertIsNone(neuware._root({"parameters": {}}))
            self.assertIsNone(cann._root({"parameters": {}}))

    def test_runtime_roots_accept_system_parameters(self):
        configured={"parameters": {"root": "/example/vendor-runtime"}}
        self.assertEqual(str(cuda._root(configured)), "/example/vendor-runtime")
        self.assertEqual(str(rocm._root(configured)), "/example/vendor-runtime")
        self.assertEqual(str(neuware._root(configured)), "/example/vendor-runtime")
        self.assertEqual(str(cann._root(configured)), "/example/vendor-runtime")

    def test_runtime_roots_accept_explicit_environment_variables(self):
        with patch.dict(
            os.environ,
            {
                "CUDA_HOME": "/env/cuda",
                "ROCM_PATH": "/env/rocm",
                "NEUWARE_HOME": "/env/neuware",
                "ASCEND_HOME_PATH": "/env/ascend",
            },
            clear=False,
        ):
            self.assertEqual(str(cuda._root({"parameters": {}})), "/env/cuda")
            self.assertEqual(str(rocm._root({"parameters": {}})), "/env/rocm")
            self.assertEqual(str(neuware._root({"parameters": {}})), "/env/neuware")
            self.assertEqual(str(cann._root({"parameters": {}})), "/env/ascend")


if __name__ == "__main__":
    unittest.main()
