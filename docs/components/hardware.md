# Hardware and Runtimes

Hardware is represented by a Device Adapter plus its Runtime Adapter. System
configuration selects physical devices and the installed Runtime root.

| Hardware | Runtime | Built-in Adapter | Current repository validation |
|---|---|---|---|
| CPU | CPU | `device/cpu` + `runtime/cpu` | Mock E2E with Reference Backend/Evaluator |
| NVIDIA GPU | CUDA | `device/nvidia` + `runtime/cuda` | Full A100 E2E with vLLM + lm-eval + BBH |
| Cambricon MLU | Neuware | `device/mlu` + `runtime/neuware` | Full MLU E2E with vLLM-MLU + lm-eval + BBH |
| MetaX GPU | MACA | `device/metax` + `runtime/maca` | Full C500 BBH E2E with the version-qualified vLLM-MetaX stack in the compatibility matrix |
| Moore Threads S5000 | MUSA | `device/mthreads` + `runtime/musa` | Full BBH E2E with MUSA 5.2.0 + vLLM-MUSA 0.28.0 + lm-eval |
| AMD GPU | ROCm | `device/amd` + `runtime/rocm` | Contract-tested |
| Ascend 910B | CANN | `device/ascend` + `runtime/cann` | Full text BBH E2E on the recorded vLLM-Ascend stack, single-NPU and TP2 |
| … | … | External Adapter | Deployment-specific |

Upstream references include [CUDA samples](https://github.com/NVIDIA/cuda-samples),
[ROCm](https://github.com/ROCm/ROCm),
[Ascend samples](https://github.com/Ascend/samples), and the
[MetaX-MACA organization](https://github.com/MetaX-MACA),
[Moore Threads torch_musa](https://github.com/MooreThreads/torch_musa),
[vLLM-MUSA](https://github.com/MooreThreads/vllm-musa), and
[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend).

## System example

```yaml
profiles:
  defaults:
    hardware: nvidia

  hardware:
    nvidia:
      type: nvidia
      devices: [0, 1, 2, 3]
      runtime:
        type: cuda
        root: /usr/local/cuda
```

The profile name (`nvidia`) is a local alias; `type` selects the Device or
Runtime Adapter. Device numbers and Runtime paths belong to System because they
change between machines.

## Device selection and parallelism

`hardware.devices` is the ordered pool exposed by this System. Evaluation can
set `models[].resources.device_count`; each model receives the first N devices
from that pool. The selected count may feed Adapter-owned derivation such as
vLLM tensor parallel size. A top-level `resources.devices` can select and
reorder members of the pool for one run, but cannot introduce a physical device
that the selected System hardware profile did not expose. The middleware does not silently choose
another device, move work to CPU, or tune memory limits after a failure.

Use `eval-manager check` and `eval-manager explain` to inspect the effective
selection and resource advice before execution.

## Validation boundary

Passing Adapter contract tests verifies manifest, RPC, Schema, and planning
behavior. Real-machine status additionally requires an actual Backend,
Evaluator, model, and Dataset run. See the detailed
[A100](../validation/nvidia-a100.md),
[MLU](../validation/cambricon-mlu.md),
[C500](../validation/metax-c500.md), [S5000](../validation/s5000.md), and
[910B](../validation/ascend-910b.md) records.
The [portable S5000/910B examples](../../examples/hardware/README.md) demonstrate
one shared Model/Evaluation and two machine-specific Systems. They require
installed vendor runtimes and real paths before execution.
