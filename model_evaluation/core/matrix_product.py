from __future__ import annotations

import copy
from pathlib import Path

from model_evaluation.core.batch_product import render_batch_tables
from model_evaluation.core.config.parsing import load_json_strict
from model_evaluation.core.files import atomic_json, atomic_text
from model_evaluation.core.result_product import inspect_run_product


class MatrixProductMixin:
    def _load_finished_product(self, run_dir: Path) -> dict:
        report = inspect_run_product(run_dir, self.app.schemas)
        if report["outcome"] != "success":
            raise ValueError(
                f"terminal outcome is not success: {report['outcome']!r}"
            )
        return load_json_strict((run_dir / "result.json").read_text(encoding="utf-8"))

    def _load_finished_legacy_run(
        self,
        run_dir: Path,
        result_path: Path,
        terminal_path: Path,
    ) -> dict:
        terminal = load_json_strict(terminal_path.read_text(encoding="utf-8"))
        outcome = str(
            terminal.get("outcome") or terminal.get("status") or ""
        ).lower()
        if outcome not in {"success", "succeeded"}:
            raise ValueError(f"terminal outcome is not success: {outcome!r}")
        result = load_json_strict(result_path.read_text(encoding="utf-8"))
        self.app.schemas.validate("canonical_result", result)
        return result

    def _validate_success_record(
        self,
        rec: dict,
        plan: dict,
    ) -> tuple[bool, str | None, dict | None]:
        try:
            raw = rec.get("run_dir")
            if not raw:
                raise ValueError("missing run_dir")
            run_dir = self._stored_path(
                raw,
                self.results_root,
                label="run_dir",
                require_dir=True,
            )
            if run_dir == self.results_root:
                raise ValueError(
                    f"run_dir must be a child of results root: {run_dir}"
                )
            # A batch consumes the complete public run product through the same
            # checker exposed to users. SHA manifests are deliberately not part
            # of that product or a prerequisite for comparing completed runs.
            recorded = rec.get("result_path") or rec.get("canonical_result_path")
            recorded_name = Path(str(recorded)).name if recorded else None
            use_product = (
                recorded_name == "result.json"
                or (
                    recorded_name != "canonical_result.json"
                    and (
                        (run_dir / "result.json").is_file()
                        or (run_dir / "terminal.json").is_file()
                    )
                )
            )
            if use_product:
                result_path = run_dir / "result.json"
                terminal_path = run_dir / "terminal.json"
                config_path = run_dir / "config" / "run_config.json"
                kind = "run product"
            else:
                result_path = run_dir / "canonical_result.json"
                terminal_path = run_dir / "terminal_record.json"
                config_path = run_dir / "config" / "execution_plan.json"
                kind = "legacy run"
            for label, path in (
                ("result", result_path),
                ("terminal", terminal_path),
                ("config", config_path),
            ):
                if not path.is_file():
                    raise ValueError(
                        f"{kind} is missing {label} file: {path.relative_to(run_dir)}"
                    )
            saved_config = load_json_strict(config_path.read_text(encoding="utf-8"))
            if saved_config.get("plan_id") != plan.get("plan_id"):
                raise ValueError(f"{kind} belongs to a different child plan")
            if use_product:
                result = self._load_finished_product(run_dir)
            else:
                result = self._load_finished_legacy_run(
                    run_dir,
                    result_path,
                    terminal_path,
                )
            expected_framework = plan["resolved"]["specs"]["evaluation"][
                "framework"
            ]["adapter"]
            model_spec = (
                ((plan.get("resolved") or {}).get("specs") or {}).get("model") or {}
            )
            model_meta = model_spec.get("metadata") or {}
            expected_model = (
                model_spec.get("experiment_id")
                or model_meta.get("experiment_id")
                or model_spec.get("id")
                or plan["run_spec"]["model"]
            )
            identity_matches = (
                result.get("run_id") == run_dir.name
                and result.get("model") == expected_model
                and result.get("benchmark") == plan["run_spec"]["benchmark"]
                and result.get("framework") == expected_framework
            )
            if not identity_matches:
                raise ValueError(
                    "result identity disagrees with child plan/run directory"
                )
            if recorded:
                stored = self._stored_path(
                    recorded,
                    run_dir,
                    label="result_path",
                    require_file=True,
                )
                if stored != result_path:
                    raise ValueError(
                        "recorded result path disagrees with relocated run directory"
                    )
            return True, None, result
        except Exception as exc:
            return False, str(exc), None

    def _write_status(
        self,
        path: Path,
        matrix_id: str,
        status: dict[str, dict],
    ) -> None:
        runs = sorted(status.values(), key=lambda row: int(row.get("index", 0)))
        atomic_json(path, {"matrix_id": matrix_id, "runs": runs})

    @staticmethod
    def _public_run(row: dict) -> dict:
        keys=(
            'index','plan_id','model_id','model_label','model_ref','benchmark','platform','deployment','evaluation',
            'status','attempts','started_at','finished_at','run_dir','result_path','warnings_count','cleanup_status','error',
        )
        return {key: copy.deepcopy(row[key]) for key in keys if key in row}

    @staticmethod
    def _model_identity(plan: dict) -> tuple[str, str, str]:
        model_spec = (
            ((plan.get("resolved") or {}).get("specs") or {}).get("model") or {}
        )
        metadata = model_spec.get("metadata") or {}
        fallback = plan["run_spec"]["model"]
        model_id = str(
            model_spec.get("experiment_id")
            or metadata.get("experiment_id")
            or fallback
        )
        model_label = str(
            model_spec.get("label")
            or metadata.get("label")
            or model_spec.get("experiment_id")
            or metadata.get("experiment_id")
            or fallback
        )
        model_ref = str((model_spec.get("source") or {}).get("ref") or "")
        return model_id, model_label, model_ref

    @classmethod
    def _not_run_record(cls, index: int, plan: dict) -> dict:
        model_id, model_label, model_ref = cls._model_identity(plan)
        run_spec = plan["run_spec"]
        return {
            "index": index,
            "plan_id": plan["plan_id"],
            "model": run_spec["model"],
            "model_id": model_id,
            "model_label": model_label,
            "model_ref": model_ref,
            "platform": run_spec["platform"],
            "deployment": run_spec["deployment"],
            "benchmark": run_spec["benchmark"],
            "evaluation": run_spec["evaluation"],
            "status": "not_run",
            "attempts": 0,
        }

    @classmethod
    def _fill_not_run_records(
        cls,
        matrix_plan: dict,
        status: dict[str, dict],
    ) -> None:
        for index, plan in enumerate(matrix_plan["plans"], 1):
            if plan["plan_id"] not in status:
                status[plan["plan_id"]] = cls._not_run_record(index, plan)

    def _finalize_batch(
        self,
        batch_dir: Path,
        matrix_plan: dict,
        status: dict[str, dict],
        *,
        hard_stop: bool,
        keep_going: bool,
        interrupted: bool = False,
    ) -> dict:
        rows = sorted(status.values(), key=lambda row: int(row.get("index", 0)))
        products = []
        plan_by_id = {plan["plan_id"]: plan for plan in matrix_plan["plans"]}
        for row in rows:
            if row.get("status") != "success":
                continue
            ok, error, result = self._validate_success_record(
                row,
                plan_by_id[row["plan_id"]],
            )
            if not ok or result is None:
                row["status"] = "failed"
                row["error"] = {
                    "type": "ResultValidationError",
                    "message": error or "success result invalid",
                }
                continue
            products.append((row, result))
        tables = render_batch_tables(products)
        rows = sorted(status.values(), key=lambda row: int(row.get("index", 0)))
        counts = {
            "planned": len(matrix_plan["plans"]),
            "success": sum(row.get("status") == "success" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
            "interrupted": sum(row.get("status") == "interrupted" for row in rows),
            "not_run": sum(row.get("status") == "not_run" for row in rows),
        }
        outcome = (
            "interrupted"
            if interrupted
            else "success"
            if counts["success"] == counts["planned"]
            else "failed"
        )
        summary = {
            "schema_version": "1.0",
            "batch_id": batch_dir.name,
            "matrix_id": matrix_plan["matrix_id"],
            "outcome": outcome,
            **counts,
            "warnings": sum(
                int(row.get("warnings_count", 0) or 0) for row in rows
            ),
            "hard_stop": hard_stop,
            "continue_on_error": keep_going,
        }
        # batch_status.json and matrix_plan.json are resumable internal state;
        # the following five files are the lightweight user-facing product.
        self.app.matrix_schemas.validate("matrix_batch_summary", summary)
        public_runs = [self._public_run(row) for row in rows]
        self.app.matrix_schemas.validate("matrix_batch_runs", public_runs)
        self._write_status(
            batch_dir / "batch_status.json",
            matrix_plan["matrix_id"],
            status,
        )
        atomic_json(batch_dir / "summary.json", summary)
        atomic_json(
            batch_dir / "runs.json",
            public_runs,
        )
        for name, text in tables.items():
            atomic_text(batch_dir / name, text)
        return summary
