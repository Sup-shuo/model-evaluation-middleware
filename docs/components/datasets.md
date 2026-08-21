# Datasets and Bindings

Dataset and Binding are separate because locating/verifying data is different
from translating a benchmark into an evaluation framework's task format.

## Built-in Dataset Adapters

| Dataset Adapter | Purpose | Current use |
|---|---|---|
| `bbh_local` | Resolve, prepare, and verify the local BBH dataset | Full and smoke BBH paths |
| `local_files` | Describe a user-managed local dataset tree | Generic local integration |
| `virtual` | Produce a dataset artifact without physical model data | CPU Mock demo |
| … | External Dataset Adapter | Deployment-specific datasets |

## Built-in Bindings

| Binding | Purpose |
|---|---|
| `lm_eval` | Generic lm-evaluation-harness task construction |
| `lm_eval.bbh` | BBH-specific lm-evaluation-harness protocol |
| `evalscope` | EvalScope task construction |
| `reference_eval` | Mock task construction |

## Execution relationship

```text
BenchmarkSpec
    |
    v
Dataset Adapter -> DatasetArtifact
                         |
                         v
                 Binding Adapter -> FrameworkTaskArtifact
                                             |
                                             v
                                      Evaluator Adapter
```

The Dataset Adapter owns asset location, preparation, and integrity policy.
The Binding owns task IDs, generated framework files, metric contract, and the
protocol fingerprint. The Evaluator runs the framework and normalizes output.

## Cache and offline operation

System `paths.cache` is the shared materialization root. Individual Adapters
decide subdirectory layout and required offline environment variables.

`basic` integrity normally checks existence, readability, and structure needed
for execution. An Adapter may expose `strict` verification when stronger
content checks are useful. These checks support repeatable operation; they do
not turn the project into an evidence or anti-tamper system.

## Adding a dataset or benchmark

- Add a Dataset Adapter only when material resolution or verification behavior
  differs from existing providers.
- Add a Binding when a benchmark needs framework-specific task generation.
- Reuse the existing Evaluator when its process and output contract already fit.
- Keep benchmark semantics out of Core and machine System files.

See [Adding an Adapter](../adapters/adding-an-adapter.md) for manifests, user
parameter Schemas, RPC, and validation.
