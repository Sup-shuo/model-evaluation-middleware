from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.matrix_config import MatrixSchemas
from model_evaluation.core.matrix_export import (
    build_execution_export,
    export_execution_plans,
)


ROOT = Path(__file__).resolve().parents[1]


def _matrix_plan(count: int = 5) -> dict:
    return {
        "matrix_id": "matrix-" + "a" * 24,
        "plans": [
            {"schema_version": "1.0", "plan_id": f"plan-{index:024x}"}
            for index in range(1, count + 1)
        ],
    }


class MatrixExportTests(unittest.TestCase):
    def setUp(self):
        self.schemas = MatrixSchemas(ROOT / "model_evaluation" / "schemas" / "user")

    def test_round_robin_export_is_deterministic_and_preserves_exact_plans(self):
        plan = _matrix_plan()
        manifest, shards = build_execution_export(plan, shards=2)
        self.schemas.validate("matrix_execution_export", manifest)
        for shard in shards:
            self.schemas.validate("matrix_execution_shard", shard)
        self.assertEqual([row["plan_count"] for row in manifest["shards"]], [3, 2])
        self.assertEqual(
            [row["index"] for row in shards[0]["plans"]],
            [1, 3, 5],
        )

        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "export"
            report = export_execution_plans(
                plan,
                output,
                shards=2,
                schemas=self.schemas,
            )
            self.assertEqual(report["output_dir"], str(output.resolve()))
            saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, manifest)
            for index, child in enumerate(plan["plans"], 1):
                path = output / "plans" / f"{index:06d}-{child['plan_id']}.json"
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), child)
            with self.assertRaisesRegex(ConfigError, "already exists"):
                export_execution_plans(plan, output, shards=2, schemas=self.schemas)

    def test_export_rejects_invalid_shard_counts(self):
        plan = _matrix_plan(2)
        for shards in (0, 3, True):
            with self.subTest(shards=shards):
                with self.assertRaises(ConfigError):
                    build_execution_export(plan, shards=shards)


if __name__ == "__main__":
    unittest.main()
