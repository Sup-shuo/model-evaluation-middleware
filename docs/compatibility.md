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
| OpenAI-compatible service | — | existing service Backend | EvalScope | Contract-tested; runtime smoke pending |
| Local service | — | Ollama / llama.cpp | configured evaluator | Contract-tested |

“Real full BBH E2E” means service startup, 24 BBH tasks, 5,761 effective
examples, final result publication and owned-process cleanup completed on a real
machine. It is not a promise that every model supported by the underlying engine
works.

“Real single-device BBH smoke E2E” means the real device/runtime probes, model
startup, OpenAI-compatible service capability checks, 24 BBH tasks with one
effective example per task, result publication and owned-process cleanup passed.
It is an integration result, not a complete benchmark score.

“Contract-tested” means manifests, JSON Schema, RPC objects and planning behavior
are covered. Before production use, run `validate`, `doctor`, a smoke evaluation,
and the intended benchmark on the target stack.

“Hardware-free Mock E2E” means the real Core/Matrix/process/result path ran with
the loopback reference components and virtual dataset. Its `contract_ok` metric
is a software integration signal, not a model quality score or hardware claim.

The same Model and Evaluation documents can be reused across machines when the
System document provides the appropriate device, runtime, environments, paths
and capacity. Backend-specific model loading requirements remain in the Model
catalog; machine capacity remains in System.
