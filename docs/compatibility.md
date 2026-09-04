# Compatibility matrix

Support statements use three levels so that a built-in Adapter is not confused
with a production-validated stack.

| Hardware / service | Runtime | Backend | Evaluator | Status |
|---|---|---|---|---|
| NVIDIA A100 | CUDA | vLLM | lm-evaluation-harness | Real full BBH E2E |
| Cambricon MLU | Neuware | vLLM-MLU | lm-evaluation-harness | Real full BBH E2E |
| MetaX C500 | MACA | vLLM-MetaX | lm-evaluation-harness | Real single-device BBH smoke E2E |
| CPU | CPU | reference backend | reference evaluator | Hardware-free Mock E2E + integration tests |
| AMD GPU | ROCm | Adapter contract only | — | Contract-tested |
| Ascend NPU | CANN | Adapter contract only | — | Contract-tested |
| OpenAI-compatible service | — | generic OpenAI Adapter | configured evaluator | Contract-tested |
| Standard CausalLM | selected Runtime | Transformers native service | configured chat/completions evaluator | Contract + loopback protocol smoke |
| TinyLLaVA Factory checkpoint | selected Runtime | TinyLLaVA family native service | configured chat evaluator | Contract + loopback protocol smoke |
| NVIDIA A100 | CUDA | existing vLLM OpenAI-compatible service | EvalScope 1.10.0 | Real one-sample GSM8K smoke E2E |
| Local service | — | Ollama / llama.cpp | configured evaluator | Contract-tested |

“Real full BBH E2E” means service startup, 24 BBH tasks, 5,761 effective
examples, final result publication and owned-process cleanup completed on a real
machine. It is not a promise that every model supported by the underlying engine
works.

For the validated MetaX stack (C500, MACA Runtime 3.8.0.23, vLLM 0.23.0,
and the recorded vLLM-MetaX 0.23.0 plugin build), the compressed-tensors weight
kernel supports group sizes 64 and 128. The vLLM Backend preflight owns this
version-qualified rule; the Device Adapter reports only device facts. On the
exact validated stack, another weight group size is rejected before startup
with the matched versions and capability source. Unknown or upgraded stack
versions are not rejected by this static rule and remain subject to the native
vLLM model-config preflight. A group-size-32 checkpoint on the recorded stack
should use a separately materialized BF16 model identity.

“Real single-device BBH smoke E2E” means the real device/runtime probes, model
startup, OpenAI-compatible service capability checks, 24 BBH tasks with one
effective example per task, result publication and owned-process cleanup passed.
It is an integration result, not a complete benchmark score.

“Real one-sample GSM8K smoke E2E” means an existing external service was checked,
EvalScope executed the explicit test split and zero-shot protocol, its `mean_acc`
was normalized to canonical `accuracy`, the final Result Product passed `result-check`,
and cleanup correctly left the unowned service running. It is an integration
result, not a representative model score.

“Contract-tested” means manifests, JSON Schema, RPC objects and planning behavior
are covered. “Loopback protocol smoke” additionally starts the shared Native
Backend service, probes its declared capabilities and makes a real local request
with a fake Driver; it does not load model weights. Before production use, run
`validate`, `doctor`, a smoke evaluation, and the intended benchmark on the
target stack.

“Hardware-free Mock E2E” means the real Core/Matrix/process/result path ran with
the loopback reference components and virtual dataset. Its `contract_ok` metric
is a software integration signal, not a model quality score or hardware claim.

The same Model and Evaluation documents can be reused across machines when the
System document provides the appropriate device, runtime, environments, paths
and capacity. Backend-specific model loading requirements remain in the Model
catalog; machine capacity remains in System.
