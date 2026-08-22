# CLI reference

This page is a compact command index. Use `eval-manager <command> --help` for
all arguments. `<system>` and `<evaluation>` may be file paths or catalog IDs;
commands use `config/system.yaml` and `config/evaluation.yaml` when both are
omitted.

## Project and configuration

| Command | Purpose |
|---|---|
| `eval-manager init [path] --hardware <type>` | Create a minimal project without overwriting existing files |
| `eval-manager config list [--kind ...]` | List System, Model, and Evaluation catalog entries |
| `eval-manager config show <kind> <reference>` | Display one stored configuration document |
| `eval-manager config check` | Validate the catalog and optional System/Evaluation pair |
| `eval-manager config migrate` | Preview supported Schema migrations; add `--write` only after review |
| `eval-manager environment-snapshot` | Record the Controller Python environment and optionally write a requirements lock |

## Pre-run checks

| Command | Purpose |
|---|---|
| `eval-manager validate` | Validate and resolve the selected user configuration |
| `eval-manager doctor` | Check local hardware, environments, executables, framework, and model material |
| `eval-manager check` | Run validation, Doctor, plan preview, and read-only resource checks together |
| `eval-manager explain` | Explain why a selected model/backend/machine combination can or cannot run |

These commands accept `--system-config` and `--evaluation-config`. Add
`--smoke` to freeze one sample per task into a newly resolved plan without
modifying YAML. Use `--format json` on `doctor`, `check`, and `explain` for
automation.

## Plan and execute

| Command | Purpose |
|---|---|
| `eval-manager plan ... -o <plan.json>` | Resolve user configuration or a RunSpec into a frozen plan |
| `eval-manager run ...` | Resolve user configuration and execute it directly |
| `eval-manager run-plan <plan.json>` | Execute a saved ExecutionPlan or MatrixPlan without re-resolving configuration |

`run` and `run-plan` accept `--results-root` and `--cache-root` overrides.
`run-plan` also supports `--resume-dir` and `--continue-on-error` when its input
is a MatrixPlan. Add `--render-summary` to write TXT and SVG views for successful
runs. A saved plan is immutable, so `--smoke` belongs on the earlier user-config
command, not on `run-plan`.

## Matrix lifecycle

| Command | Purpose |
|---|---|
| `eval-manager matrix-validate <matrix>` | Validate a MatrixSpec |
| `eval-manager matrix-expand <matrix>` | Preview the concrete RunSpec combinations |
| `eval-manager matrix-plan <matrix> -o <plan.json>` | Resolve and freeze every child ExecutionPlan |
| `eval-manager matrix-run <matrix>` | Plan and execute a MatrixSpec locally and serially |
| `eval-manager run-matrix-plan <plan.json>` | Execute or resume a saved MatrixPlan locally |
| `eval-manager matrix-export <plan.json> -o <dir>` | Export exact child plans and scheduler-neutral jobs/shards |

See [Matrix execution lifecycle](matrix-execution.md) for the complete local and
external-scheduler flow.

## Adapters and protocol

| Command | Purpose |
|---|---|
| `eval-manager adapters` | List discovered built-in and external Adapters |
| `eval-manager adapter-check <root>` | Validate an external Adapter root and its manifests |
| `eval-manager schema-check` | Validate all packaged protocol Schemas |
| `eval-manager demo --render-summary` | Run the hardware-free Reference E2E |

## Result products

| Command | Purpose |
|---|---|
| `eval-manager result-check <path>` | Validate a completed run or Matrix batch product |
| `eval-manager inspect <path>` | Validate and render the same product for humans |
| `eval-manager inspect <path> --format json` | Return the inspection report for automation |

See [Final result product protocol](result-product.md) for JSON documents, the
Python consumer API, Batch products, and TXT/SVG projections.
