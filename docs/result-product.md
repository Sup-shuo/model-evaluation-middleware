# Final result product protocol

The run directory is the public product boundary of the middleware. Version 1.0
defines the following formal documents:

| File | Purpose | Presence |
|---|---|---|
| `result.json` | Run identity, normalized summary metrics and artifact references | Required on success |
| `metrics.json` | Summary, group and task metrics | Required with `result.json` |
| `terminal.json` | Final outcome, local timestamps and cleanup status | Always required |
| `failure.json` | Failure stage, primary error, log excerpts and cleanup result | Required on failure |
| `config/run_config.json` | Effective run selection, resolved Specs and Adapter identities | Always required after initialization |
| `config/runtime_versions.json` | Observed Backend, Evaluator and environment versions | Required on success |

Their schemas are distributed with the Python package as
`model_evaluation/schemas/`, including dedicated schemas for the result,
metrics, terminal, failure, run configuration, and runtime-version products.
Result Product Schema `1.0` is frozen for the 4.1 release line: incompatible
changes require a new schema version. `terminal.json` is written last and is the public completion
marker; a directory without it is not a completed result product.

Validate a directory with either command:

```bash
eval-manager result-check results/<run-id>
eval-manager inspect results/<run-id>
eval-manager inspect results/<run-id> --format json
```

A synthetic, sanitized, schema-valid directory is available at
[`examples/result_example/`](../examples/result_example/). It is intended for
format discovery and consumer integration tests; its score is illustrative and
must not be reported as a model benchmark result.

The check covers JSON Schema, cross-file run/model/benchmark identities, metric
agreement, success/failure file rules, and confined artifact paths. It does not
verify cryptographic provenance and does not turn a result into tamper-proof
evidence. The product records what was run and preserves the framework output;
trust and governance remain outside this glue layer.

Python consumers can use the same validation boundary without parsing CLI
output:

```python
from model_evaluation.results import load_run

run = load_run("results/<run-id>")
run.metrics.summary()
run.metrics.groups()
run.metrics.tasks()
run.runtime()
run.artifacts()
```

The SDK returns defensive copies and read-only artifact descriptors. It does not
create another result format or reinterpret framework metrics.

`--render-summary` is available on `demo`, `run`, `run-plan`, `matrix-run`, and
`run-matrix-plan`. When enabled, the CLI writes `result-summary.txt` and
`result-summary.svg` after each successful run. Failed runs are not rendered.
These are human-readable projections, not additional protocol documents; they
do not change or recompute the normalized metrics. Existing successful results
can be rendered with `scripts/print_result.py`.

Matrix batches are also public products. `_batches/<batch-id>/summary.json` and
`runs.json` have dedicated schemas, while the three TSV tables provide summary,
group, and task projections:

```text
results/_batches/<batch-id>/
├── summary.json
├── runs.json
├── metrics.tsv
├── group_metrics.tsv
└── task_metrics.tsv
```

`eval-manager result-check` accepts either a run directory or a batch directory.
For a batch, it validates every referenced successful run, regenerates the
summary, group, and task projections from those run products, and requires exact
agreement with all three TSV files. Inspect a batch with the same product API:

```bash
eval-manager result-check results/_batches/<batch-id>
eval-manager inspect results/_batches/<batch-id>
```

Framework-native data remains under `raw/`. Per-sample files remain under
`samples/` and are present only when the evaluator was explicitly configured to
emit them. `config/` records the effective run and observed runtime versions;
`logs/` contains operational logs.
