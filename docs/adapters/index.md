# Adapter guide

Adapters are isolated subprocess integrations that exchange versioned JSON
objects with Core. They let hardware, frameworks, and datasets evolve without
adding vendor- or framework-name branches to orchestration code.

## Adapter kinds

| Kind | Responsibility | Built-in examples |
|---|---|---|
| Device | Discover devices and establish visibility | `cpu`, `nvidia`, `mlu`, `metax`, `amd`, `ascend` |
| Runtime | Resolve SDK/runtime environment facts | `cpu`, `cuda`, `neuware`, `maca`, `rocm`, `cann` |
| Environment | Resolve interpreter and wrap processes | `current`, `conda`, `venv` |
| Backend | Preflight, start/attach, probe service, snapshot | `vllm`, `generic_openai`, `ollama`, `llama_cpp`, `reference` |
| Dataset | Resolve, prepare, verify, snapshot data | `bbh_local`, `local_files`, `virtual` |
| Binding | Translate Benchmark + Dataset into framework tasks | `lm_eval`, `lm_eval.bbh`, `evalscope`, `reference_eval` |
| Evaluator | Preflight, plan evaluation, normalize, snapshot | `lm_eval`, `evalscope`, `reference_eval` |

List the adapters discovered by the current installation:

```bash
eval-manager adapters
```

Built-in presence is not a real-machine support claim. Check
[compatibility and validation status](../compatibility.md) before selecting a
production path.

## Discovery sources

Core combines three sources and rejects duplicate kind/name identities:

1. Adapters shipped in `model_evaluation/adapters/`;
2. absolute development roots listed in `MODEL_EVAL_ADAPTER_PATHS`;
3. installed Python distributions exposing the
   `model_evaluation.adapters` entry-point group.

For development:

```bash
export MODEL_EVAL_ADAPTER_PATHS=/absolute/path/to/my-adapters
eval-manager adapter-check /absolute/path/to/my-adapters
eval-manager adapters
```

For an installed plugin, declare entry points without importing plugin code
during discovery:

```toml
[project.entry-points."model_evaluation.adapters"]
"device.example" = "my_package.adapters.device.example"
```

The target module directory contains the executable launcher and manifest.
Runtime invocation still occurs through the Adapter subprocess protocol.

## Choose the smallest extension

- New accelerator: usually Device + Runtime; reuse an existing Backend if its
  vendor build provides the same service contract.
- New inference engine: Backend Adapter; do not duplicate evaluator behavior.
- New evaluation framework: Evaluator + Binding; reuse Dataset Adapters where
  their artifacts already fit.
- New dataset source: Dataset Adapter; add a Binding only when framework task
  generation differs.
- New model: normally Model YAML only, not an Adapter.

Continue with [Adding an Adapter](adding-an-adapter.md). The normative object
definitions and operations are in the
[Architecture and Adapter protocol](../../ARCHITECTURE_AND_ADAPTER_PROTOCOL.md).
