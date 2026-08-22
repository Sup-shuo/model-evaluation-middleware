from __future__ import annotations

import unittest
from pathlib import Path

from model_evaluation.core.capability_vocabulary import vocabulary_scope
from model_evaluation.core.compatibility import evaluate
from model_evaluation.core.schema.validator import SchemaStore


ROOT = Path(__file__).resolve().parents[1]


class CapabilityVocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schemas = SchemaStore(ROOT / "model_evaluation" / "schemas")

    def test_core_failure_has_schema_valid_structured_diagnostic(self) -> None:
        report = evaluate(
            {"schema_version": "1.0", "requirements": [
                {"path": "runtime.family", "op": "equals", "value": "cuda"}
            ]},
            {"runtime.family": "cpu"},
        )
        self.assertFalse(report.compatible)
        self.assertEqual(report.diagnostics[0]["vocabulary"], "core")
        self.schemas.validate("capability_diagnostic", report.diagnostics[0])

    def test_extension_capability_is_not_rejected_by_a_closed_enum(self) -> None:
        report = evaluate(
            {"schema_version": "1.0", "requirements": [
                {"path": "vendor.feature.experimental_mode", "op": "exists"}
            ]},
            {},
        )
        self.assertFalse(report.compatible)
        self.assertEqual(vocabulary_scope("vendor.feature.experimental_mode"), "extension")
        self.assertEqual(report.diagnostics[0]["vocabulary"], "extension")
        self.schemas.validate("capability_diagnostic", report.diagnostics[0])


if __name__ == "__main__":
    unittest.main()
