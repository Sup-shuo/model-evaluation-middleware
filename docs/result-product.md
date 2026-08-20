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

The check covers JSON Schema, cross-file run/model/benchmark identities, metric
agreement, success/failure file rules, and confined artifact paths. It does not
verify cryptographic provenance and does not turn a result into tamper-proof
evidence. The product records what was run and preserves the framework output;
trust and governance remain outside this glue layer.

Framework-native data remains under `raw/`. Per-sample files remain under
`samples/` and are present only when the evaluator was explicitly configured to
emit them. `config/` records the effective run and observed runtime versions;
`logs/` contains operational logs.
