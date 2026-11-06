import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from model_evaluation.adapters.evaluator.evalscope.completeness import validate_completeness
from model_evaluation.adapters.evaluator.evalscope import impl
from model_evaluation.adapters.binding.evalscope import impl as binding
from model_evaluation.sdk.runtime import AdapterError


class CompletenessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = {"task_id": "fixture", "metadata": {"evalscope_result_validation": {
            "subset_counts": {"a": 2, "b": 2}, "require_images": True}}}
        self.rows = [{"num": 4}]
        self.details = [{"subset_name": k, "num": 2} for k in ("a", "b")]
        self.data = {kind: [] for kind in ("predictions", "reviews")}
        for category in ("a", "b"):
            for index in range(2):
                metadata = {"category": category, "index": index}
                common = {"messages": [{"content": [{"type": "image", "image": "data:image/jpeg;base64,AA"}]}]}
                self.data["predictions"].append(dict(common, metadata=metadata,
                    model_output={"choices": [{"stop_reason": "stop"}]}))
                self.data["reviews"].append(dict(common, sample_score={
                    "sample_metadata": metadata, "score": {"value": {"acc": 0}}}))
        self.write()

    def write(self):
        for kind, rows in self.data.items():
            directory = self.root / kind
            directory.mkdir(exist_ok=True)
            (directory / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def check(self):
        validate_completeness(self.root, self.task, self.rows, self.details)

    def test_full(self):
        self.check()

    def test_no_policy_retains_legacy_contract(self):
        validate_completeness(self.root, {"metadata": {}}, [], [])

    def test_smoke_subset_limit(self):
        self.task["metadata"]["evalscope_dataset_args"] = {"subset_list": ["a"]}
        self.task["metadata"]["evalscope_result_validation"]["sample_limit"] = 1
        self.data = {k: v[:1] for k, v in self.data.items()}
        self.rows, self.details = [{"num": 1}], [{"subset_name": "a", "num": 1}]
        self.write()
        self.check()

    def test_missing_extra_duplicate_and_mismatched_identity(self):
        original = copy.deepcopy(self.data)
        for kind in original:
            for mode in ("missing", "extra", "duplicate", "different"):
                with self.subTest(kind=kind, mode=mode):
                    self.data = {k: copy.deepcopy(v) for k, v in original.items()}
                    rows = self.data[kind]
                    if mode == "missing":
                        rows.pop()
                    elif mode == "extra":
                        rows.append(copy.deepcopy(rows[0]))
                    elif mode == "duplicate":
                        rows[-1] = copy.deepcopy(rows[-2])
                    else:
                        meta = rows[-1].get("metadata") or rows[-1]["sample_score"]["sample_metadata"]
                        meta["index"] = 99
                    self.write()
                    with self.assertRaises(AdapterError):
                        self.check()

    def test_missing_images_and_structured_records(self):
        original = copy.deepcopy(self.data)
        for kind in original:
            for field in ("messages", "model_output" if kind == "predictions" else "sample_score"):
                with self.subTest(kind=kind, field=field):
                    self.data = {k: copy.deepcopy(v) for k, v in original.items()}
                    self.data[kind][0].pop(field)
                    self.write()
                    with self.assertRaises(AdapterError):
                        self.check()

    def test_wrong_report_counts(self):
        for details in ([], [{"subset_name": "a", "num": 4}], self.details + [{"subset_name": "unknown", "num": 2}]):
            with self.subTest(details=details), self.assertRaises(AdapterError):
                validate_completeness(self.root, self.task, self.rows, details)

    def test_normalize_enforces_before_publishing(self):
        directory = self.root / "reports"
        directory.mkdir()
        (directory / "fixture.json").write_text(json.dumps({"metric_name": "acc", "score": 1, "num": 4}))
        with self.assertRaisesRegex(AdapterError, "subset counts"):
            impl.normalize({"raw_result_root": str(self.root), "task": self.task}, {})

    def test_binding_carries_policy_and_limit_without_hashing(self):
        benchmark = {"id": "fixture", "protocol": {"evalscope_result_validation":
            {"subset_counts": {"a": 2}, "require_images": True}}, "metrics": []}
        with patch.object(binding, "_fingerprint", return_value="not-computed"):
            task = binding.build_task({"benchmark": benchmark,
                "dataset_artifact": {"materialization": {"kind": "virtual"}},
                "evaluation": {"execution": {"sample_limit": 1}}}, {})
        self.assertEqual(task["metadata"]["evalscope_result_validation"]["sample_limit"], 1)

    def test_official_six_subsets_250(self):
        names = ("coarse perception", "fine-grained perception", "instance reasoning",
                 "logical reasoning", "math", "science & technology")
        self.task["metadata"]["evalscope_result_validation"]["subset_counts"] = dict.fromkeys(names, 250)
        self.data = {kind: [] for kind in self.data}
        for subset, name in enumerate(names):
            for index in range(250):
                metadata = {"category": name, "index": subset * 250 + index}
                message = [{"content": [{"type": "image", "image": "data:image/jpeg;base64,AA"}]}]
                self.data["predictions"].append({"metadata": metadata, "messages": message,
                    "model_output": {"choices": [{"stop_reason": "max_tokens"}]}})
                self.data["reviews"].append({"messages": message, "sample_score": {
                    "sample_metadata": metadata, "score": {"value": {"acc": 0}}}})
        self.rows = [{"num": 1500}]
        self.details = [{"subset_name": name, "num": 250} for name in names]
        self.write()
        self.check()

    def test_strict_reuses_context_and_error_rejection(self):
        for output in ({"choices": [{"stop_reason": "model_length"}]},
                       {"choices": [], "error": "unavailable"}):
            self.data["predictions"][0]["model_output"] = output
            self.write()
            with self.assertRaisesRegex(AdapterError, "generation failure"):
                self.check()

    def test_malformed_and_non_numeric_review(self):
        self.data["reviews"][0]["sample_score"]["score"] = {"value": {"acc": "wrong"}}
        self.write()
        with self.assertRaisesRegex(AdapterError, "numeric review"):
            self.check()
        (self.root / "reviews" / "samples.jsonl").write_text("not json\n")
        with self.assertRaisesRegex(AdapterError, "invalid reviews JSON"):
            self.check()
