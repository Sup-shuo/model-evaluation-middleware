# Components and integrations

The middleware connects existing hardware runtimes, inference engines,
datasets, and evaluation frameworks through versioned Adapters. This page is a
navigation view; the [compatibility matrix](../compatibility.md) is the source
of validation status.

| Component | Built-in surface | Configuration guide |
|---|---|---|
| Hardware and Runtime | CPU/CPU, NVIDIA/CUDA, Cambricon MLU/Neuware, MetaX/MACA, AMD/ROCm, Ascend/CANN, … | [Hardware and Runtimes](hardware.md) |
| Inference Backend | vLLM, generic OpenAI-compatible service, Ollama, llama.cpp, Reference Backend, … | [Inference Backends](inference-backends.md) |
| Evaluation framework | lm-evaluation-harness, EvalScope, Reference Evaluator, … | [Evaluation frameworks](evaluators.md) |
| Dataset and Binding | BBH local, local files, virtual, lm-eval bindings, EvalScope binding, … | [Datasets and Bindings](datasets.md) |
| Model material | Local/Hugging Face references, Safetensors, quantized and derived material, text/VL, … | [Model guide](../models/index.md) |

## How the parts connect

```text
System selects Device + Runtime + Environments + Backend + Evaluator
Model supplies identity + material + Backend loading requirements
Evaluation selects Model + Benchmark + temporary overrides
                              |
                              v
            Dataset -> Binding -> Evaluator -> Result product
                       ^
                       |
                    Backend service
```

The Backend owns model inference. The evaluation framework owns task behavior
and metric calculation. The middleware validates, plans, launches or attaches,
passes canonical objects between them, and normalizes the saved result product.

## Built-in versus validated

“Built-in” means the Adapter manifest, implementation, Schema, and contract are
present. It does not mean every cross-product of hardware, Runtime, Backend,
model format, Dataset, and Evaluator has completed real-machine validation.
Validation records use three explicit levels:

- full real-machine E2E;
- real-machine smoke E2E;
- contract-tested.

When deploying a new combination, start from a nearby System example and run
`eval-manager check` before any smoke or full evaluation.

## Add another integration

Most integrations should be delivered as an Adapter, not a Core branch. See
the [Adapter inventory](../adapters/index.md) to choose the kind and
[Adding an Adapter](../adapters/adding-an-adapter.md) for the contract and
validation flow.
