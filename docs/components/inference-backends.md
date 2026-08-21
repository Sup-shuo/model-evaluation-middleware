# Inference Backends

A Backend Adapter describes service requirements, preflight checks, startup or
attachment, readiness, capabilities, and a runtime snapshot. It does not
reimplement model execution.

| Backend | Upstream | Default mode | Repository validation |
|---|---|---|---|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Managed | Real E2E on NVIDIA, MLU, and MetaX stacks listed in the compatibility matrix |
| Generic OpenAI-compatible service | External service | External | Contract-tested |
| Ollama | [ollama/ollama](https://github.com/ollama/ollama) | Managed | Contract-tested |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Managed | Contract-tested |
| Reference Backend | Included mock implementation | Managed | CPU Mock E2E |
| … | External Adapter | Managed/attached/external | Deployment-specific |

Vendor vLLM implementations follow the same Backend contract. Relevant
upstreams include [Cambricon vllm-mlu](https://github.com/Cambricon/vllm-mlu)
and [MetaX vLLM](https://github.com/MetaX-MACA/vLLM-metax).

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
