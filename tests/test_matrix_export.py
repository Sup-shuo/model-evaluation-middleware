from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.matrix_config import MatrixSchemas
from model_evaluation.core.matrix_export import (
    build_execution_export,
    build_scheduler_jobs,
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


def _resource_matrix_plan(weights: list[int]) -> dict:
    plan = _matrix_plan(len(weights))
    for index, (child, count) in enumerate(zip(plan["plans"], weights), 1):
        child.update(
            {
                "run_spec": {
                    "model": f"model-{index}",
                    "benchmark": "bbh",
                    "deployment": "vllm",
                    "evaluation": "lm_eval",
                },
                "resources": [
                    {"kind": "device", "id": f"physical-{device}", "exclusive": True}
                    for device in range(count)
                ],
                "compatibility": {"status": "compatible"},
                "resolved": {
                    "platform": {
                        "backend_environment": {
                            "provider": "conda",
                            "identity": "/environments/backend",
                        },
                        "evaluation_environment": {
                            "provider": "conda",
                            "identity": "/environments/evaluator",
                        },
                    },
                    "specs": {
                        "model": {"experiment_id": f"model-{index}"},
                        "benchmark": {"id": "bbh"},
                        "platform": {
                            "device": {"adapter": "nvidia", "devices": list(range(count))},
                            "runtime": {"adapter": "cuda"},
                        },
                        "deployment": {
                            "backend": {"adapter": "vllm"},
                            "management": {"mode": "managed"},
                        },
                        "evaluation": {"framework": {"adapter": "lm_eval"}},
                    }
                },
            }
        )
    return plan


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
            self.assertEqual(len(list((output / "jobs").glob("*.json"))), len(plan["plans"]))
            with self.assertRaisesRegex(ConfigError, "already exists"):
                export_execution_plans(plan, output, shards=2, schemas=self.schemas)

    def test_resource_balanced_export_uses_logical_requirements_without_device_ids(self):
        plan = _resource_matrix_plan([4, 3, 2, 1])
        jobs = build_scheduler_jobs(plan)
        for job in jobs:
            self.schemas.validate("matrix_scheduler_job", job)
            self.assertNotIn("physical-", json.dumps(job))
            self.assertNotIn("/environments/", json.dumps(job))
        manifest, shards = build_execution_export(
            plan,
            shards=2,
            strategy="resource_balanced",
        )
        self.schemas.validate("matrix_execution_export", manifest)
        self.assertEqual(manifest["strategy"], "resource_balanced")
        self.assertEqual(
            [row["requirements"]["resource_weight"] for row in manifest["shards"]],
            [5, 5],
        )
        for shard in shards:
            self.schemas.validate("matrix_execution_shard", shard)

    def test_export_keeps_execution_compatibility_groups_in_separate_shards(self):
        plan = _resource_matrix_plan([2, 1, 2, 1])
        for child in plan["plans"][2:]:
            platform = child["resolved"]["specs"]["platform"]
            platform["device"]["adapter"] = "mlu"
            platform["runtime"]["adapter"] = "neuware"
            deployment = child["resolved"]["specs"]["deployment"]
            deployment["backend"]["adapter"] = "vllm_mlu"

        manifest, shards = build_execution_export(
            plan,
            shards=2,
            strategy="resource_balanced",
        )
        self.schemas.validate("matrix_execution_export", manifest)
        self.assertEqual(
            [shard["requirements"]["accelerator_types"] for shard in shards],
            [["mlu"], ["nvidia"]],
        )
        self.assertEqual(
            [shard["requirements"]["runtime_families"] for shard in shards],
            [["neuware"], ["cuda"]],
        )

        with self.assertRaisesRegex(ConfigError, "compatibility_groups=2"):
            build_execution_export(
                plan,
                shards=1,
                strategy="resource_balanced",
            )

    def test_export_treats_evaluator_as_part_of_execution_compatibility(self):
        plan = _resource_matrix_plan([1, 1])
        plan["plans"][1]["resolved"]["specs"]["evaluation"]["framework"][
            "adapter"
        ] = "evalscope"
        _, shards = build_execution_export(
            plan,
            shards=2,
            strategy="round_robin",
        )
        evaluators = []
        jobs = build_scheduler_jobs(plan)
        jobs_by_id = {job["job_id"]: job for job in jobs}
        for shard in shards:
            evaluators.append(
                {
                    jobs_by_id[row["job_id"]]["intent"]["evaluator"]
                    for row in shard["plans"]
                }
            )
        self.assertEqual(evaluators, [{"evalscope"}, {"lm_eval"}])

    def test_export_treats_environment_capability_as_execution_compatibility(self):
        plan = _resource_matrix_plan([1, 1])
        plan["plans"][1]["resolved"]["platform"]["evaluation_environment"][
            "identity"
        ] = "/environments/evaluator-v2"
        _, shards = build_execution_export(
            plan,
            shards=2,
            strategy="round_robin",
        )
        environment_ids = [
            shard["requirements"]["execution_compatibility"][
                "evaluator_environment"
            ]
            for shard in shards
        ]
        self.assertEqual(len(set(environment_ids)), 2)

    def test_export_rejects_invalid_shard_counts(self):
        plan = _matrix_plan(2)
        for shards in (0, 3, True):
            with self.subTest(shards=shards):
                with self.assertRaises(ConfigError):
                    build_execution_export(plan, shards=shards)


if __name__ == "__main__":
    unittest.main()
