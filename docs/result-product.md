# Final result product protocol

The run directory is the public product boundary of the middleware. Version 1.0
defines four JSON documents:

| File | Purpose | Presence |
|---|---|---|
| `result.json` | Run identity, normalized summary metrics and artifact references | Required on success |
| `metrics.json` | Summary, group and task metrics | Required with `result.json` |
| `terminal.json` | Final outcome, local timestamps and cleanup status | Always required |
| `failure.json` | Failure stage, primary error, log excerpts and cleanup result | Required on failure |

Their schemas are distributed with the Python package as
`model_evaluation/schemas/{result,metrics,terminal,failure}.schema.json`.
Schema version `1.0` is frozen for the alpha31 line: incompatible changes require
a new schema version.

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

Framework-native data remains under `raw/`. Per-sample files remain under
`samples/` and are present only when the evaluator was explicitly configured to
emit them. `config/` records the effective run and observed runtime versions;
`logs/` contains operational logs.
