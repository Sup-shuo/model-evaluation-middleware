# Evaluation configuration

An Evaluation file selects reusable Model and benchmark identities and records
the temporary choices for one experiment.

```yaml
schema_version: "1.2"

models:
  - qwen35-08b-base

benchmarks:
  - bbh

backend:
  seed: 1234
  pythonhashseed: 1234

evaluator:
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

System defaults are used unless Evaluation selects a named profile:

```yaml
profiles:
  hardware: nvidia
  backend: vllm
  evaluator: evalscope
```

Only those profiles and their referenced environments enter the plan.

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
