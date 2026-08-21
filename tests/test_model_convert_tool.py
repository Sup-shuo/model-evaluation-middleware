from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "model_convert.py"


def run_tool(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )


class ModelConvertToolTests(unittest.TestCase):
    def checkpoint(self, root: Path, *, architecture: str, quantization: dict) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps(
                {
                    "architectures": [architecture],
                    "model_type": "qwen3_5",
                    "dtype": "bfloat16",
                    "quantization_config": quantization,
                }
            ),
            encoding="utf-8",
        )
        (source / "model.safetensors").write_bytes(b"structural-test-weight")
        return source

    def test_routes_are_explicit_and_never_automatic(self) -> None:
        completed = run_tool("routes", "--format", "json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        routes = json.loads(completed.stdout)
        names = {route["name"] for route in routes}
        self.assertIn("compressed-tensors-to-bf16", names)
        self.assertIn("awq-to-bf16", names)

        with tempfile.TemporaryDirectory() as temporary:
            source = self.checkpoint(
                Path(temporary),
                architecture="Qwen3_5ForConditionalGeneration",
                quantization={
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                },
            )
            inspected = run_tool("inspect", str(source), "--format", "json")
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            payload = json.loads(inspected.stdout)
            self.assertFalse(payload["automatic_conversion"])
            self.assertTrue(payload["checkpoint"]["loadable"])
            self.assertEqual(payload["checkpoint"]["status"], "complete")
            self.assertGreater(payload["checkpoint"]["weight_size_bytes"], 0)
            self.assertEqual(
                payload["compatible_manual_routes"],
                [
                    "compressed-tensors-to-bf16",
                    "compressed-tensors-contract-to-bf16",
                ],
            )

    def test_convert_dry_run_selects_internal_route_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.checkpoint(
                root,
                architecture="Qwen3_5ForConditionalGeneration",
                quantization={
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                },
            )
            output = root / "derived"
            completed = run_tool(
                "convert",
                "--route",
                "compressed-tensors-to-bf16",
                "--source",
                str(source),
                "--output",
                str(output),
                "--source-ref",
                "owner/model",
                "--dry-run",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            command = payload["command"]
            self.assertFalse(payload["automatic_conversion"])
            self.assertIn("convert_ct_qwen_to_dense_bf16.py", command[1])
            self.assertIn(str(root / ".derived.converting"), command)
            self.assertFalse(output.exists())
            self.assertFalse((root / ".derived.converting").exists())

    def test_check_reports_missing_index_shards_and_transient_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.checkpoint(
                Path(temporary),
                architecture="Qwen3_5ForConditionalGeneration",
                quantization={"quant_method": "awq"},
            )
            (source / "model.safetensors").unlink()
            (source / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (source / "download.incomplete").write_bytes(b"partial")
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.0.weight": "model-00001-of-00002.safetensors",
                            "layer.1.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )

            completed = run_tool("check", str(source), "--format", "json")
            self.assertEqual(completed.returncode, 1, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertFalse(payload["checkpoint"]["loadable"])
            self.assertEqual(payload["checkpoint"]["status"], "incomplete")
            self.assertEqual(
                payload["checkpoint"]["index"]["missing_shards"],
                ["model-00002-of-00002.safetensors"],
            )
            self.assertEqual(payload["checkpoint"]["transient_files"], ["download.incomplete"])

    def test_check_allows_complete_index_but_reports_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.checkpoint(
                Path(temporary),
                architecture="Qwen3_5ForConditionalGeneration",
                quantization={"quant_method": "awq"},
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model.safetensors"}}),
                encoding="utf-8",
            )
            (source / "old.safetensors.incomplete").write_bytes(b"stale")

            completed = run_tool("check", str(source), "--format", "json")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["checkpoint"]["loadable"])
            self.assertFalse(payload["checkpoint"]["clean"])
            self.assertEqual(payload["checkpoint"]["status"], "usable-with-warnings")

    def test_check_rejects_unsafe_index_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = self.checkpoint(
                Path(temporary),
                architecture="Qwen3_5ForConditionalGeneration",
                quantization={"quant_method": "awq"},
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
                encoding="utf-8",
            )

            completed = run_tool("check", str(source), "--format", "json")
            self.assertEqual(completed.returncode, 1, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertIn("unsafe", payload["checkpoint"]["issues"][0])

    def test_route_mismatch_and_existing_output_fail_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.checkpoint(
                root,
                architecture="Qwen3_5ForConditionalGeneration",
                quantization={
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "version": "gemm",
                },
            )
            mismatch = run_tool(
                "convert",
                "--route",
                "compressed-tensors-to-bf16",
                "--source",
                str(source),
                "--output",
                str(root / "derived"),
                "--source-ref",
                "owner/model",
                "--dry-run",
            )
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("requires quantization_method", mismatch.stderr)

            output = root / "derived"
            output.mkdir()
            existing = run_tool(
                "convert",
                "--route",
                "awq-to-bf16",
                "--source",
                str(source),
                "--output",
                str(output),
                "--source-ref",
                "owner/model",
                "--dry-run",
            )
            self.assertEqual(existing.returncode, 2)
            self.assertIn("refusing to overwrite", existing.stderr)

    def test_tools_are_separate_from_project_automation_scripts(self) -> None:
        implementation = ROOT / "tools" / "model_conversion"
        self.assertTrue(implementation.is_dir())
        self.assertTrue((implementation / "convert_ct_qwen_to_dense_bf16.py").is_file())
        self.assertFalse((ROOT / "scripts" / "model_conversion").exists())


if __name__ == "__main__":
    unittest.main()
