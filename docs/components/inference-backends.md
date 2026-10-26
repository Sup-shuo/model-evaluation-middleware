# Inference Backends

A Backend Adapter describes service requirements, preflight checks, startup or
attachment, readiness, capabilities, and a runtime snapshot. It delegates model
execution to an upstream engine or an official model-family API and does not
change model algorithms.

| Backend | Upstream | Default mode | Repository validation |
|---|---|---|---|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Managed | Real E2E on NVIDIA, MLU, MetaX, Moore Threads S5000, and Ascend 910B stacks listed in the compatibility matrix |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) | Managed | Contract + loopback protocol smoke |
| TinyLLaVA family | Checkpoint-embedded official `model.chat` code | Managed | Contract + loopback protocol smoke |
| Generic OpenAI-compatible service | External service | External | Contract-tested |
| Ollama | [ollama/ollama](https://github.com/ollama/ollama) | Managed | Contract-tested |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Managed | Contract-tested |
| Reference Backend | Included mock implementation | Managed | CPU Mock E2E |
| … | External Adapter | Managed/attached/external | Deployment-specific |

Vendor vLLM implementations follow the same Backend contract. Relevant
upstreams include [Cambricon vllm-mlu](https://github.com/Cambricon/vllm-mlu),
[MetaX vLLM](https://github.com/MetaX-MACA/vLLM-metax),
[Moore Threads vLLM-MUSA](https://github.com/MooreThreads/vllm-musa), and
[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend).
Choose the platform-compatible engine/environment in System; all use the
existing `vllm` Backend Adapter. See the [S5000/910B configuration examples](../../examples/hardware/README.md).

## Management modes

- `managed`: Core owns the Backend process lifecycle for the run.
- `attached`: the service already exists in the current deployment.
- `external`: another system owns the remote service.

External and attached services do not require a fake local Device/Runtime. The
Backend still reports a Service Descriptor containing endpoints, model
identity, ownership, limits, authentication references, and capabilities.

## Managed vLLM example

```yaml
profiles:
  environment:
    vllm-env:
      type: current

  backend:
    vllm:
      type: vllm
      executable: /usr/local/bin/vllm
      environment: vllm-env
      compatibility:
        runtime_families: [cuda]
      parameters:
        port: 8091
        gpu_memory_utilization: 0.8
        max_num_seqs: 8
        num_concurrent: 8
```

Machine capacity and service settings belong to System. Stable checkpoint
loading requirements such as `dtype`, `max_model_len`, or
`trust_remote_code` belong under the selected namespace in Model.

## Python-native model execution

Use `vllm` when the installed engine supports the checkpoint. Use
`transformers` for a standard text `AutoModelForCausalLM` checkpoint that must
run directly in Python. Use a model-family Backend such as `tinyllava` only when
the official inference API has family-specific inputs or outputs.

Both Python-native Backends reuse one internal Native Backend service. It
provides health and model discovery, OpenAI-compatible Completions/Chat where
the Driver truthfully supports them, optional tokenize/detokenize endpoints,
bounded request size and concurrency, and normal managed-process cleanup. YAML
selects a named Adapter; it cannot name an arbitrary Python module or function.

Register a dedicated inference environment in System:

```yaml
profiles:
  environment:
    transformers-env:
      type: conda
      profile: transformers-runtime

  backend:
    transformers-native:
      type: transformers
      environment: transformers-env
      parameters:
        port: 8091
        device: auto
        dtype: bfloat16
        allow_cpu_offload: false

    tinyllava-native:
      type: tinyllava
      environment: transformers-env
      parameters:
        port: 8092
        device: auto
        dtype: float16
        allow_cpu_offload: false
```

Then put stable loading values in the matching Model namespace:

```yaml
backends:
  transformers:
    max_model_len: 4096
    trust_remote_code: false
  tinyllava:
    max_model_len: 4096
    trust_remote_code: true
```

The generic Transformers Driver is text-only. TinyLLaVA accepts at most one
system message plus one user turn containing text and exactly one `image_url`
part. Its modern checkpoint must embed `AutoModelForCausalLM` in `auto_map`;
legacy TinyLLaVA material is rejected before weight loading. Neither path
enables CPU/disk offload unless System explicitly sets
`allow_cpu_offload: true`.

The Native Backend service has no authentication and defaults to loopback.
Keep it on `127.0.0.1` unless an external network boundary provides equivalent
access control. TinyLLaVA checkpoints execute repository-provided Python code;
enable that Adapter only for model material you trust.

## OpenAI-compatible service example

Use the `generic_openai` Backend for a service owned elsewhere. Configure its
endpoint and secret reference according to the Adapter parameter Schema. Keep
secrets in the environment or secret provider; do not write their values into
committed YAML or result examples.

## Model compatibility

The Adapter checks declared requirements and actual checkpoint metadata where
available. A model name is not sufficient to infer serialization or Runtime
support. If material conversion is required, use the standalone manual tool
and register a separate derived Model; execution never converts automatically.
