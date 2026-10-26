from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from model_evaluation.adapters.device.mthreads import impl as device
from model_evaluation.adapters.runtime.musa import impl as runtime
from model_evaluation.sdk.runtime import AdapterError


CSV = "0, MTT S5000, uuid-0, 81920 [MiB]\\n2, MTT S5000, uuid-2, 81920 [MiB]\\n".replace("\\n", "\n")


class DeviceTests(unittest.TestCase):
    def test_probe_order_and_memory(self):
        with patch.object(device.shutil, "which", return_value="/bin/mthreads-gmi"), patch.object(
            device.subprocess, "run", return_value=CompletedProcess([], 0, CSV, "")
        ):
            value = device.probe({"requested_devices": [2, 0]}, {})
        self.assertEqual([x["id"] for x in value["devices"]], ["2", "0"])
        self.assertEqual(value["devices"][0]["memory_bytes"], 80 * 1024**3)
        self.assertEqual(value["vendor"], "mthreads")

    def test_missing_device_and_duplicate_rejected(self):
        with patch.object(device, "_inventory", return_value=[{"id": "2"}]):
            for requested in ([0], [2, 2]):
                with self.assertRaises(AdapterError):
                    device.probe({"requested_devices": requested}, {})

    def test_visibility(self):
        self.assertEqual(device.visibility({"devices": [2, 3]}, {})["env_patch"]["set"],
                         {"MUSA_VISIBLE_DEVICES": "2,3"})
        self.assertEqual(device.visibility({"devices": []}, {})["env_patch"]["set"],
                         {"MUSA_VISIBLE_DEVICES": ""})
        with self.assertRaises(AdapterError):
            device.visibility({"devices": ["2,3"]}, {})

    def test_bad_csv_and_failed_tool(self):
        for result in (CompletedProcess([], 0, "garbage", ""),
                       CompletedProcess([], 1, "", "failed"),
                       CompletedProcess([], 0, "0, card, uid, N/A", "")):
            with patch.object(device.shutil, "which", return_value="/bin/gmi"), patch.object(
                device.subprocess, "run", return_value=result
            ), self.assertRaises(AdapterError):
                device.probe({}, {})

    def test_missing_tool_and_timeout(self):
        with patch.object(device.shutil, "which", return_value=None), self.assertRaises(AdapterError):
            device.probe({}, {})
        with patch.object(device.shutil, "which", return_value="/bin/gmi"), patch.object(
            device.subprocess, "run", side_effect=TimeoutExpired("gmi", 1)
        ), self.assertRaises(AdapterError):
            device.probe({}, {})


class RuntimeTests(unittest.TestCase):
    def test_probe_and_environment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include").mkdir()
            (root / "lib").mkdir()
            (root / "include/musa.h").write_text("#define MUSA_VERSION 50200\n")
            inputs = {"parameters": {"root": directory}}
            with patch.object(runtime.shutil, "which", return_value="/bin/gmi"), patch.object(
                runtime.subprocess, "run",
                return_value=CompletedProcess([], 0, "5.2.0-server\n", "")
            ):
                value = runtime.probe(inputs, {})
            self.assertEqual(value["version"], "5.2.0")
            self.assertEqual(value["family"], "musa")
            self.assertEqual(value["driver_version"], "5.2.0-server")
            self.assertEqual(runtime.resolve_environment(inputs, {})["env_patch"]["set"]["MUSA_HOME"],
                             str(root.resolve()))

    def test_invalid_root(self):
        with self.assertRaises(AdapterError):
            runtime.resolve_environment({"parameters": {"root": "relative"}}, {})
        with self.assertRaises(AdapterError):
            runtime.probe({"parameters": {"root": "/nonexistent-musa-root"}}, {})


if __name__ == "__main__":
    unittest.main()
