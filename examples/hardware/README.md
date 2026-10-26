# S5000/MUSA and Ascend 910B/CANN examples

These portable System/Evaluation examples use the same
`coder3101/Qwen3.5-4B-heretic` model and built-in `bbh` benchmark with
vLLM + lm-evaluation-harness, following the validated text BBH integration path.
These files replace deployment-specific paths with placeholders; they are
not copies of an active machine configuration or a promise of identical scores.

| System file | Device adapter | Runtime adapter | Default |
| --- | --- | --- | --- |
| `mthreads-s5000.yaml` | `mthreads` | `musa` | physical device 0, TP=1 |
| `ascend-910b.yaml` | `ascend` | `cann` | physical device 0, TP=1 |

Both explicitly disable CPU offload. The shared `bbh.yaml` requests one
device and has no sample limit. The benchmark is the project's 24-task
leaderboard multiple-choice BBH subset, not all possible BBH tasks.

## Prepare an existing environment

Nothing here installs packages, creates environments, downloads weights,
or prepares the offline dataset cache. Before running an evaluation, provide:

- A working platform driver/runtime and platform-compatible vLLM/PyTorch stack.
- The middleware's controller dependencies (see project requirements).
- An existing lm-evaluation-harness checkout and a Conda environment with its
  required dependencies. The controller/backend environment need not match it.
- Local model files under
  `/REPLACE_WITH_MODEL_ROOT/coder3101/Qwen3.5-4B-heretic` and the complete
  offline BBH cache under the configured cache root.

Replace every `/REPLACE_WITH_...` value in a copy of the chosen System file.
Set its actual physical device ID and an unused local service port. Runtime
`root` must identify the installed MUSA or CANN directory; it is not a driver
installer. Configure `paths.cache` for the prepared dataset cache. Relative
`cache` and `results` paths resolve under the middleware project root.

The backend uses `current`: run the controller from the prepared inference
environment, with its `vllm` executable available, or select an explicit
isolated Environment profile. The evaluator uses the separate named
`lm-eval` Conda profile. The absolute Conda executable and environment prefix
are placeholders, not assumptions about an installed Conda location.

The CANN Runtime adapter reconstructs backend runtime variables and existing
library, Python, OPP/tiling, and tool directories from the selected root.
Backend startup and backend preflight processes receive Device + Runtime
patches before the selected Environment wrapper; the wrapper cannot rewrite
already-owned variables. Evaluator processes use only their own Environment
binding, not the backend Device/Runtime patch. Sourcing an interactive
`set_env.sh` is not a substitute for configuring the persistent runtime root.

## Validate without hardware access

From the middleware checkout, these commands validate schemas and compile
both examples without probing hardware, importing vLLM, or starting services.
Placeholders are intentionally accepted at this stage; validation does not
prove that those paths or dependencies exist.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 eval-manager validate --system-config examples/hardware/mthreads-s5000.yaml --evaluation-config examples/hardware/bbh.yaml
PYTHONDONTWRITEBYTECODE=1 python3 eval-manager validate --system-config examples/hardware/ascend-910b.yaml --evaluation-config examples/hardware/bbh.yaml
```

After replacing placeholders and preparing dependencies, select one System
file (or your edited copy). For example:

```bash
python3 eval-manager doctor --system-config examples/hardware/mthreads-s5000.yaml --evaluation-config examples/hardware/bbh.yaml
python3 eval-manager evaluate --smoke --system-config examples/hardware/mthreads-s5000.yaml --evaluation-config examples/hardware/bbh.yaml
python3 eval-manager evaluate --system-config examples/hardware/mthreads-s5000.yaml --evaluation-config examples/hardware/bbh.yaml
```

For Ascend, substitute `examples/hardware/ascend-910b.yaml` in all three
commands. `doctor` and `plan` perform live checks; only `validate` above is
a no-probe compile check. `--smoke` limits each task to one sample for that
invocation; omit it for the full suite. `evaluate` starts real inference.
Successful runs use the standard
`results/<run-id>/` layout.

The examples use metadata-only framework provenance and migration model
provenance so placeholders need no invented revision locks. For a pinned
reproduction, record the actual model/framework revisions and environment
versions and configure the corresponding provenance policy before running.
Seeds are fixed, but versions, kernels, and device backends can still change
results.
