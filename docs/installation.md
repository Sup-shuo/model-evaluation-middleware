# Installation and first evaluation

This guide starts with a clean checkout and ends with a validated result
directory. It keeps the Controller, inference Backend, and evaluation framework
as separate environments because they solve different jobs and may require
incompatible dependencies.

## 1. Install the Controller

Python 3.10 or newer is required. From a source checkout:

```bash
python -m pip install -e .
eval-manager schema-check
eval-manager adapters
```

For a built wheel, install that artifact and use the console script:

```bash
python -m pip install model_evaluation_middleware-*.whl
eval-manager schema-check
eval-manager adapters
```

The Controller needs only the middleware and its declared dependencies. Do not
install vendor runtimes, vLLM, lm-eval, and EvalScope into this environment just
to make the CLI import successfully.

## 2. Verify the control path

```bash
eval-manager demo --render-summary
```

The Mock demo uses CPU, a virtual Dataset, and Reference Backend/Evaluator. A
successful result confirms installation, planning, execution, publication, and
result validation without claiming hardware or model quality.

## 3. Prepare the inference environment

Install the inference engine by following its upstream instructions or use an
existing vendor container. The middleware does not install drivers, device SDKs,
or inference engines automatically.

For the exercised NVIDIA path, the Backend executable is vLLM and exposes
OpenAI-compatible Completions and Chat Completions endpoints. Confirm it in the
Backend environment:

```bash
vllm --version
```

See [Inference Backends](components/inference-backends.md) for managed,
attached, and external service configuration.

## 4. Prepare an evaluation environment

Use a dedicated environment for each evaluation framework. For the exercised
lm-evaluation-harness path:

```bash
python -m venv /opt/venvs/lm-eval
/opt/venvs/lm-eval/bin/python -m pip install -e /opt/lm-evaluation-harness
```

For EvalScope:

```bash
conda create -p /opt/conda/envs/evalscope_env python=3.10 pip
conda run -p /opt/conda/envs/evalscope_env \
  python -m pip install evalscope==1.10.0
```

Pin the version or framework revision used by your deployment. Register the
environment in System rather than activating it before every Controller command.
See [Environment isolation](configuration/environments.md) and
[Evaluation frameworks](components/evaluators.md).

## 5. Create the user configuration

```bash
eval-manager init my-evaluation --hardware nvidia
cd my-evaluation
```

`--hardware` also accepts `mlu`, `metax`, `amd`, `ascend`, and `cpu`. The command
does not overwrite existing files. Replace every generated `REPLACE_WITH_*`
value, then configure these three documents:

| Document | Required decisions | Detailed guide |
|---|---|---|
| System | Device, Runtime, environments, Backend/Evaluator, model root, cache, results | [System](configuration/system.md) |
| Model | Identity, source, architecture, format, quantization, context, Backend loading parameters | [Model](configuration/model.md) |
| Evaluation | Models, benchmarks, profile selection, seeds, limits, execution mode | [Evaluation](configuration/evaluation.md) |

## 6. Register a model

Place the model below the machine's `models.root`, then create a catalog entry.
For example, if System declares `models.root: /data`:

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

The resolved model directory is `/data/Qwen/Qwen3.5-0.8B-Base`. Registering a
model never downloads or converts it. More text, quantized, multimodal, and
derived examples are in the [Model guide](models/index.md).

## 7. Select the evaluation

```yaml
schema_version: "1.3"

models:
  - qwen35-08b-base

benchmarks:
  - bbh

backend:
  profile: vllm
  parameters:
    seed: 1234

evaluator:
  profile: lm_eval
  parameters:
    batch_size: 1
    log_samples: true

offline: true

execution:
  mode: serial
  continue_on_error: false
```

Keep the Evaluation reusable. The CLI `--smoke` flag applies a temporary
one-sample-per-task limit in the resolved plan without changing this YAML. A
smoke result must not be reported as a complete score.

## 8. Check before execution

```bash
eval-manager check \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml --smoke
```

`check` performs validation, Doctor, plan preview, and read-only resource checks.
If it fails, request a human-readable explanation:

```bash
eval-manager explain \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml --smoke
```

Resolve missing paths, environments, model material, ports, or compatibility
requirements before running.

## 9. Run and inspect

```bash
eval-manager run \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml \
  --smoke --render-summary
```

Then validate the final product:

```bash
eval-manager result-check results/<run-id>
eval-manager inspect results/<run-id>
eval-manager inspect results/<run-id> --format json
```

The optional `result-summary.txt` and `result-summary.svg` project saved data;
they do not recompute metrics. Continue with the
[Result product protocol](result-product.md).

## 10. Reuse the evaluation on another machine

Keep Model and Evaluation unchanged and select another System:

```bash
eval-manager check --system-config mlu --evaluation-config bbh --smoke
eval-manager check --system-config nvidia --evaluation-config bbh --smoke
```

Remove `--smoke` after connectivity succeeds to run the complete Evaluation.
This works when both System files provide compatible Hardware, Runtime, Backend,
Evaluator, environments, and model material. Validation status for one stack is
not automatically inherited by another; consult the
[Compatibility matrix](compatibility.md).

## Troubleshooting entry points

Use the narrowest command that answers the current question:

| Symptom or question | Command |
|---|---|
| Is the catalog or selected YAML valid? | `eval-manager config check` or `eval-manager validate` |
| Are hardware, environments, executables, frameworks, and model files ready? | `eval-manager doctor` |
| Why is this model/backend/machine combination blocked? | `eval-manager explain` |
| Is everything ready before service startup? | `eval-manager check` |
| Is a completed Run or Batch Result internally consistent? | `eval-manager result-check <path>` |

These checks report the failing layer; they do not download models, convert
weights, silently change resource settings, or start a distributed scheduler.
