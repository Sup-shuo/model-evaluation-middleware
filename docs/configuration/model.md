# Model configuration

A Model file assigns a stable experiment identity to model material and records
the loading facts that should travel across machines.

```yaml
schema_version: "1.0"

id: qwen35-08b-base
label: Qwen3.5 0.8B Base BF16

source:
  type: local
  ref: Qwen/Qwen3.5-0.8B-Base

architecture: qwen3_5
quantization: bf16
format: safetensors
context_length: 4096

backends:
  vllm:
    max_model_len: 4096
    trust_remote_code: true

provenance:
  policy: migration
```

## Identity and material

`id` is the identity selected by Evaluation and written to results.
`source.ref` describes the upstream reference or path relative to the selected
System's `models.root`. Registering a Model does not download its material.

Use `source.type: hf` or `registry` to describe origin, not to request an
automatic download. For a locally managed Backend, make the referenced model
directory available before execution.

## Field ownership

Put stable model facts here: revision, tokenizer, architecture, format,
quantization, context length, chat template, trust-remote-code, and
Backend-specific loading requirements. Keep devices, Runtime roots, executable
paths, memory policy, and service concurrency in System.

Use Backend namespaces such as `backends.vllm` or `backends.llama_cpp` instead
of one unscoped parameter set. The selected Backend receives only its own
loading parameters.

## Quantized and derived material

A repository name containing `AWQ` is not enough to identify the checkpoint's
real serialization. Inspect its `config.json`, tokenizer, and weight index.
Compatibility notes may advise that a particular Runtime requires conversion:

```yaml
metadata:
  runtime_compatibility:
    mlu_vllm:
      status: conversion_required
      reason: The deployed Backend does not support this checkpoint format
      recommended_model_id: example-derived-bf16
      advisory_only: true
```

This is advisory metadata. Validation and execution never convert weights
automatically. If a manually converted artifact is used, create a new Model ID
and `source.ref`, and describe `derived_from` plus the transformation. Do not
overwrite the original model identity.

The standalone manual tool is documented in the [Model guide](../models/index.md#manual-inspection-and-conversion).

## Catalog organization

The loader discovers `.yaml` and `.yml` files recursively under
`config/models/`. Subdirectories are optional and do not change IDs:

```text
config/models/
├── README.md
├── qwen/
│   ├── official/
│   └── community/
└── vision-language/
```

Keep one declaration per YAML file and make `id` unique across the entire
catalog. See [representative model examples](../models/index.md).
