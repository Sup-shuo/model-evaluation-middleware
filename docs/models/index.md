# Model catalog guide

Model configuration is a catalog, not a download manager or model registry.
Each entry gives reusable model material a stable experiment identity and
records the loading facts needed by supported Backends.

## Representative examples

The repository intentionally documents a few patterns instead of listing every
model.

| Pattern | Example | What it demonstrates |
|---|---|---|
| Safetensors model | [`qwen_example.yaml`](../../config/models/qwen_example.yaml) | Local material, architecture, context, and vLLM loading parameters |
| GGUF model | [`llama_gguf_example.yaml`](../../config/models/llama_gguf_example.yaml) | A distinct model family and llama.cpp-specific loading parameters |

These files are examples of catalog structure. Availability and validation
remain machine-specific; consult `eval-manager check` and the
[compatibility matrix](../compatibility.md).

## Minimal declaration

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
```

If System declares `models.root: /data`, the managed Backend resolves this
example to `/data/Qwen/Qwen3.5-0.8B-Base`.

## Source and availability

`source.type` and `source.ref` describe origin or material location. They do not
request a download. It is valid to register a model before it is materialized,
but avoid guessing architecture, quantization, format, or context from the
repository name. Confirm them from model metadata after download.

## Backend-specific loading

Place stable requirements below the Backend namespace:

```yaml
backends:
  vllm:
    dtype: bfloat16
    max_model_len: 4096
  llama_cpp:
    context_length: 4096
```

Only the selected Backend receives its namespace. Hardware devices, tensor
parallel policy, memory utilization, executable paths, ports, and concurrency
belong to System.

## Compatibility notes

Compatibility metadata explains why a model needs different material on a
particular Runtime without changing execution behavior:

```yaml
metadata:
  runtime_compatibility:
    mlu_vllm:
      status: conversion_required
      reason: The deployed Backend does not support this checkpoint format
      recommended_model_id: example-derived-bf16
      advisory_only: true
```

The note is consumed as guidance by validation and inspection surfaces. It does
not trigger conversion or silently substitute another Model ID.

## Derived model identity

A converted artifact is a distinct model product:

```yaml
id: example-derived-bf16
source:
  type: local
  ref: owner/model-derived-bf16
quantization: bf16

metadata:
  derived_from:
    id: example-original
    ref: owner/model-original
    quantization: compressed-tensors
  transformation:
    kind: dequantize
    output_dtype: bfloat16
```

Never overwrite the source directory or reuse the original ID. A text-only
artifact derived from a vision-language model must explicitly state that the
vision capability was removed.

## Manual inspection and conversion

Model material operations are standalone tools and never part of
`eval-manager run`:

```bash
python tools/model_convert.py inspect /data/OWNER/MODEL
python tools/model_convert.py check /data/OWNER/MODEL
python tools/model_convert.py routes
```

`inspect` and `check` read checkpoint metadata, Safetensors indexes, referenced
shards, empty or missing files, download residue, and approximate weight size.
They do not load the Backend or automatically tune resources.

Choose a conversion route explicitly and use a new output directory:

```bash
python tools/model_convert.py convert \
  --route compressed-tensors-to-bf16 \
  --source /data/OWNER/MODEL \
  --output /data/OWNER/MODEL-derived-bf16 \
  --source-ref OWNER/MODEL \
  --dry-run
```

Remove `--dry-run` only after reviewing the plan. The conversion implementation
validates structure and tensor equivalence before publishing the new directory
and refuses to overwrite an existing destination.

## Organize a large catalog

Subdirectories are supported but optional:

```text
config/models/
├── README.md
├── qwen/
│   ├── official/
│   └── community/
├── vision-language/
└── derived/
```

Directories help humans navigate; they do not create namespaces. Keep one
model per YAML file and make every `id` globally unique.

## Checklist

1. Verify the real checkpoint metadata and material path.
2. Create one stable Model ID.
3. Put only model-owned loading facts in the Backend namespace.
4. Add advisory compatibility notes when a Runtime needs different material.
5. Run `eval-manager check` on every target System.
6. Use a smoke run before reporting a full evaluation.
