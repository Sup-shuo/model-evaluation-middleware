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

An explicit small `evaluator.limit` is useful for connectivity smoke:

```yaml
evaluator:
  profile: lm_eval
  parameters:
    limit: 1
    log_samples: true
```

Remove the limit for a full benchmark. A smoke run validates integration only;
it must not be presented as a complete benchmark score.

## Matrix and external scheduling

The built-in matrix path expands a finite set of Model, Platform, Deployment,
Benchmark, and Evaluation axes. Local execution remains serial with resource
locks. For larger work, export the resolved execution-plan bundle and submit it
to an external scheduler:

```bash
eval-manager matrix-export --help
```

The scheduler owns distribution; each worker returns the same result product.

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
