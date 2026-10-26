# Ascend 910B validation

Status: full text BBH E2E on the recorded stack, including result publication
and owned-process cleanup.

## Environment

| Component | Observed environment |
| --- | --- |
| Accelerator | Ascend 910B; single-NPU and TP2 integration |
| Driver | 26.0.rc1, confirmed from the container-mounted driver version record |
| CANN | Toolkit installation metadata 9.1.0; historical runtime probes recorded `unknown` |
| Inference Python | Container Python 3.12.13 |
| Backend | vLLM-Ascend 0.23.0 with vLLM 0.23.0+empty; OpenAI-compatible Completions API |
| Inference packages | torch 2.10.0+cpu; torch_npu 2.10.0.post4; Transformers 5.5.4 |
| Evaluator | Separate lm-eval environment; lm-evaluation-harness 0.4.13.dev0; Transformers 5.14.1 |
| Harness revision | `f4d4b3de3ee6741a7151a9fe74945ee515262f4c` |
| Protocol | Local leaderboard BBH multiple-choice subset, 3-shot, 24 tasks, 5,761 examples |

The torch package suffix `+cpu` describes that distribution; actual backend
records identify the NPU execution platform. These runs used no CPU offload.
CANN and driver installation metadata were checked separately;
historical `runtime_versions.json` files remain unchanged.

## Exercised coverage

The validated path covers unconverted artifacts and separately materialized
BF16 derivatives, with explicit single-device or tensor-parallel settings.
Derived BF16 results establish that execution path, not native AWQ/NVFP4
support. Record the effective artifact and any tokenizer override in the
Model configuration when reproducing an evaluation.

Validation covers text, including VL language paths; it does not establish
vision or audio evaluation.

## Runtime environment

`device/ascend` sets `ASCEND_RT_VISIBLE_DEVICES`.
`runtime/cann` reconstructs backend library, Python, OPP/tiling, and tool
paths from the configured CANN root for managed backend/preflight processes.
This avoids relying on an interactive shell's CANN activation. The evaluator
retains its own Environment binding.

Start the Controller from the intended container inference Python when the
Backend uses `current`, or select a separate explicit Environment. Configure
the installed CANN root and backend executable in System. Container login
defaults and host-driver mounts remain deployment settings; the middleware
does not rewrite them.

If startup reports `aclrtMallocHostWithCfg` 207001 (host pinned-memory
allocation), inspect driver/backend logs and memory limits before retrying.
Treat this as a runtime allocation failure, not a model-format incompatibility.

## Reproduce the workflow

The [portable example](../../examples/hardware/README.md) uses
`coder3101/Qwen3.5-4B-heretic`, one NPU, separate backend/evaluator environments,
and explicit CPU offload 0. Replace all path placeholders and choose an
available device before running from the project root:

```bash
python3 -B ./eval-manager validate --system-config examples/hardware/ascend-910b.yaml --evaluation-config examples/hardware/bbh.yaml
python3 -B ./eval-manager evaluate --system-config examples/hardware/ascend-910b.yaml --evaluation-config examples/hardware/bbh.yaml --smoke
python3 -B ./eval-manager evaluate --system-config examples/hardware/ascend-910b.yaml --evaluation-config examples/hardware/bbh.yaml
```

For a larger model, explicitly expose the intended System device pool and
select the required Evaluation device count and tensor-parallel configuration;
the single-NPU example is not a universal large-model profile.
See [device selection](../components/hardware.md#device-selection-and-parallelism).

Upstream: [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend).
