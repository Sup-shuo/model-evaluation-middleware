# Evaluation configuration

An Evaluation file selects reusable Model and benchmark identities and records
the temporary choices for one experiment.

```yaml
schema_version: "1.3"

models:
  - id: qwen35-08b-base
    resources:
      device_count: 1

benchmarks:
  - bbh

backend:
  profile: vllm
  parameters:
    seed: 1234
    pythonhashseed: 1234

evaluator:
  profile: lm_eval
  parameters:
    batch_size: 1
    random_seed: 0
    numpy_random_seed: 1234
    torch_random_seed: 1234
    fewshot_random_seed: 1234
    request_seed: 1234
    pythonhashseed: 1234

offline: true

execution:
  mode: serial
  continue_on_error: false
```

## Select profiles for one run

Every Evaluation explicitly selects one registered inference Backend and one
Evaluator. This keeps the execution intent visible in the file instead of
silently inheriting framework choices from System:

```yaml
backend:
  profile: vllm

evaluator:
  profile: evalscope
```

System remains the machine inventory: it defines what those profile names mean
and which environments they use. An optional `profiles.hardware` selects a
hardware profile when the System registers more than one.

## Device pool and per-model count

`system.profiles.hardware.<name>.devices` is the physical device pool exposed
to this middleware. Evaluation can request a different count for each model:

```yaml
models:
  - id: qwen35-08b-base
    resources:
      device_count: 1
  - id: qwen36-27b-derived-bf16
    resources:
      device_count: 2
```

For a managed Backend, Core assigns the first N devices from the selected pool.
The vLLM Adapter derives `tensor_parallel_size` from that model's assigned
count unless an explicit Backend parameter overrides it. A count larger than
the pool fails during validation; the middleware does not guess capacity or
spill inference to CPU. If `device_count` is omitted, that model keeps the
entire selected pool.

For a one-off run, top-level `resources.devices` can narrow or reorder the
System pool before per-model counts are applied. Physical device IDs stay out
of reusable Model catalog files.

## Temporary model override

An Evaluation may override a loading parameter without changing the catalog:

```yaml
models:
  - id: qwen35-08b-base
    overrides:
      backend:
        max_model_len: 8192
```

Do not override source, architecture, format, quantization, or experiment
identity. Create a new Model declaration when those facts change.

## Smoke and full evaluation

Reuse the same Evaluation file for both modes. `--smoke` is available on
`validate`, `doctor`, `check`, `explain`, `plan`, and `run`; add it when checking
connectivity:

```bash
eval-manager check --system-config <system> --evaluation-config <evaluation> --smoke
eval-manager run --system-config <system> --evaluation-config <evaluation> --smoke
```

The Controller freezes `execution.mode: smoke` and `sample_limit: 1` into the
resolved Evaluation Spec; Evaluator Adapters translate that intent to their
framework-specific limit. The source YAML is unchanged. Omit `--smoke` for the
full benchmark. A smoke result validates integration only and must not be
presented as a complete benchmark score. Saved RunSpec/MatrixPlan files are
immutable inputs, so `--smoke` is intentionally rejected by `run-plan` and
`run-matrix-plan`.

## Matrix and external scheduling

The built-in matrix path expands a finite set of Model, Platform, Deployment,
Benchmark, and Evaluation axes. Local execution remains serial with resource
locks. For larger work, export a protocol 1.2 bundle for an external scheduler:

```bash
eval-manager matrix-export /tmp/matrix-plan.json \
  -o /tmp/matrix-jobs \
  --shards 8 \
  --strategy resource_balanced
```

Each `jobs/*.json` file contains logical accelerator count/type, Runtime,
Backend, Evaluator intent, and claim counts without physical device IDs. The
bundle also retains exact `plans/*.json` files for workers with compatible
paths and environments. `round_robin` remains available for deterministic
legacy-style sharding. Jobs with different accelerator, Runtime, Backend,
management, Evaluator, or resolved environment capabilities are kept in
separate shards; the
requested shard count must provide at least one shard for every compatibility
group. The scheduler owns distribution; each worker returns the same result
product. See [Matrix execution lifecycle](../matrix-execution.md) for plan creation,
local execution, export, worker execution, resume, and Batch validation.

## Before execution

```bash
eval-manager check \
  --system-config nvidia \
  --evaluation-config full_bbh

eval-manager plan \
  --system-config nvidia \
  --evaluation-config full_bbh \
  --output /tmp/full-bbh-plan.json
```

Review the selected devices, environments, model path, resources, and
benchmark before calling `eval-manager run`.
