# System configuration

A System file describes one machine or deployment: what hardware and Runtime
exist, which execution environments and framework profiles are available, and
where models, cache, and results live.

```yaml
schema_version: "1.3"

system:
  name: nvidia-vllm

metadata:
  timezone: Asia/Shanghai
  result_platform: nvidia

profiles:
  defaults:
    hardware: nvidia

  environment:
    vllm-env:
      type: current
    lm-eval-env:
      type: conda
      profile: /opt/conda/envs/lm_eval_env
      executable: /opt/conda/bin/conda

  hardware:
    nvidia:
      type: nvidia
      devices: [0]
      runtime:
        type: cuda
        root: /usr/local/cuda

  backend:
    vllm:
      type: vllm
      executable: /usr/local/bin/vllm
      environment: vllm-env
      compatibility:
        runtime_families: [cuda]
      parameters:
        gpu_memory_utilization: 0.8
        max_num_seqs: 8
        num_concurrent: 8

  evaluator:
    lm_eval:
      type: lm_eval
      root: /opt/lm-evaluation-harness
      environment: lm-eval-env

models:
  root: /data

paths:
  cache: cache
  results: results
```

## Profile groups

| Group | Purpose |
|---|---|
| `environment` | Resolve or wrap the executable context (`current`, Conda, venv) |
| `hardware` | Select devices and their Runtime |
| `backend` | Configure a managed, attached, or external inference service |
| `evaluator` | Configure the evaluation framework and its environment |

Profile names are local aliases. The `type` selects the Adapter. This lets one
machine register multiple Backend or Evaluator choices without duplicating its
hardware and path configuration.

## Rules that affect portability

- `hardware.devices` is the device pool this machine exposes to the middleware.
  Evaluation may narrow/reorder the pool and assign a `device_count` per model;
  reusable Model files must never contain physical device IDs.
- Managed Backends declare `compatibility.runtime_families`; Core compares
  capabilities rather than guessing compatibility from profile names.
- Backend and Evaluator may reference different environments.
- Service concurrency, memory utilization, executable paths, and device
  selection belong here, not in Model.
- Relative cache and result paths resolve from the project root and may not
  escape it with `..`. Absolute paths may point to controlled shared storage.
- Use one canonical model root. Avoid describing the same physical directory
  through multiple symlink aliases because that creates confusing run records.
- `metadata.timezone` controls display and result timestamps; use an IANA name
  such as `Asia/Shanghai`.
- `metadata.result_platform` is the short platform prefix used in run names.

## Multiple evaluation frameworks

Register each framework environment and profile under a distinct name:

```yaml
profiles:
  environment:
    lm-eval-env:
      type: conda
      profile: /opt/conda/envs/lm_eval_env
      executable: /opt/conda/bin/conda
    evalscope-env:
      type: conda
      profile: /opt/conda/envs/evalscope_env
      executable: /opt/conda/bin/conda

  evaluator:
    lm_eval:
      type: lm_eval
      root: /opt/lm-evaluation-harness
      environment: lm-eval-env
    evalscope:
      type: evalscope
      environment: evalscope-env
      parameters:
        executable: evalscope
        expected_version: 1.10.0
```

An Evaluation selects one explicitly with `evaluator.profile`; the unused
profile remains inactive. For setup guidance see [Environment isolation](environments.md) and
[Evaluation frameworks](../components/evaluators.md).

## Platform examples

The repository includes examples under `config/systems/`. Treat them as site
templates: copy one and replace machine paths, devices, environments, and
capacity values. Validation evidence for a named platform is documented
separately in the [compatibility matrix](../compatibility.md).
