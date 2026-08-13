# Model Evaluation Middleware

> Portable evaluation orchestration across hardware, inference backends, model
> formats, and evaluation frameworks.

Model Evaluation Middleware is a configuration-driven glue layer around existing
inference engines and evaluation frameworks. It separates machine inventory,
model identity, and experiment selection, then records complete task metrics,
native framework output, effective configuration, runtime versions, and logs in
one predictable result directory.

**English summary:** keep the model and evaluation files stable, switch only the
System file when moving a workload between NVIDIA/CUDA, Cambricon MLU/Neuware,
CPU/llama.cpp, or another adapter-backed environment.

This project is not a new inference engine, benchmark suite, model registry,
experiment-tracking platform, or tamper-proof evidence product. Its scope is
portable execution, reproducible configuration, process lifecycle management,
and useful result publication.

## Why this exists

Real evaluation deployments often need to connect:

- an inference backend such as vLLM;
- an evaluator such as lm-evaluation-harness;
- local models and offline datasets on shared storage;
- an inference API that must be started, probed, used, and cleaned up safely.

The middleware connects those pieces without importing vendor SDKs or evaluation
frameworks into Core:

```text
System + Model + Evaluation
            |
    validate / doctor / plan
            |
Device -> Runtime -> Environment -> Backend
            |
Dataset -> Binding -> Evaluator
            |
 result.json + task metrics + raw output + logs
```

## Status

The current release is an alpha intended for engineers comfortable editing YAML.
The vLLM + lm-evaluation-harness + BBH path has been exercised on both NVIDIA
and Cambricon accelerators. Other built-in adapters are contract-tested, but a
real `doctor` and smoke run are still required on every new host.

For engineers who already have a backend, evaluator environment, model mount,
and dataset cache, the project is usable today. It is not yet a one-command tool
for first-time users. The remaining maturity work is mainly onboarding rather
than a redesign of the execution core:

- an `init` assistant that generates a private System file;
- clearer dependency/cache bootstrap diagnostics;
- real-host smoke coverage for more built-in adapters;
- a smaller set of curated recipes for common backend/evaluator combinations.

Site inventories, debug queues, private model catalogs, and historical results
are intentionally not part of that roadmap or the public source tree.

## Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

Check the bundled contracts and adapters:

```bash
./eval-manager schema-check
./eval-manager adapters
```

## Configuration model

Only three user-facing object types are required:

| File | Owns | Must not own |
|---|---|---|
| System | hardware, device IDs, runtime, environments, executables, paths and capacity | experiment identity |
| Model | model identity, source, format and backend-specific loading requirements | host paths or device IDs |
| Evaluation | selected model IDs, benchmark, seeds, limits and run-local overrides | long-lived machine inventory |

`model_evaluation/presets/` contains packaged protocol/default templates used by
Core. It is not a second user configuration directory. Normal users edit only
the root `config/` tree or private files outside the repository.

## Included examples

The public tree deliberately contains placeholders rather than real deployment
paths:

| Example | Hardware/runtime | Backend | Model/format |
|---|---|---|---|
| `nvidia` + `smoke_bbh_08b` | NVIDIA / CUDA | managed vLLM | Qwen / Safetensors |
| `mlu` + `smoke_bbh_08b` | Cambricon MLU / Neuware | managed vLLM | the same Qwen model |
| `cpu_llama_cpp` + `smoke_bbh_llama_cpp` | CPU | managed llama.cpp | Llama / GGUF |

Replace all `/opt`, `/data/models`, and `/var/cache` example paths before use.
The examples do not download models or datasets.

All host-specific filesystem paths are owned by the System configuration. Model
files carry logical model references; Evaluation files carry experiment choices.
Production Python contains no model mount, framework checkout, environment, or
executable installation path. See
[docs/deployment.md](docs/deployment.md#path-ownership-and-portability) for the
enforced path boundary.

## First run

Copy the examples into the ignored private directory:

```bash
mkdir -p config/private
cp config/systems/nvidia.yaml config/private/my-system.yaml
cp config/evaluations/smoke_bbh_08b.yaml config/private/my-evaluation.yaml
```

Edit `config/private/my-system.yaml` to point at your backend environment,
lm-evaluation-harness checkout, model root, and cache. Place the example model at:

```text
<models.root>/Qwen/Qwen3.5-0.8B-Base
```

Then run the workflow in order:

```bash
./eval-manager validate \
  --system-config config/private/my-system.yaml \
  --evaluation-config config/private/my-evaluation.yaml

./eval-manager doctor \
  --system-config config/private/my-system.yaml \
  --evaluation-config config/private/my-evaluation.yaml

./eval-manager plan \
  --system-config config/private/my-system.yaml \
  --evaluation-config config/private/my-evaluation.yaml \
  --output /tmp/model-eval-plan.json

./eval-manager run \
  --system-config config/private/my-system.yaml \
  --evaluation-config config/private/my-evaluation.yaml
```

- `validate` checks user schemas and adapter parameters.
- `doctor` probes hardware, environments, framework dependencies, and model
  configuration without starting the long-lived model service.
- `plan` shows the fully resolved workload.
- `run` starts or attaches to the backend, evaluates, saves output, and cleans up
  owned processes.

To move the same Qwen evaluation from NVIDIA to MLU, keep the Model and
Evaluation unchanged and select the MLU System:

```bash
./eval-manager run \
  --system-config config/systems/mlu.yaml \
  --evaluation-config config/evaluations/smoke_bbh_08b.yaml
```

## Smoke versus full BBH

`smoke_bbh_08b.yaml` sets `evaluator.limit: 1` and is only a compatibility test.
`full_bbh.yaml` has no limit and should produce 24 tasks with 5,761 effective
examples for the included BBH profile.

## Results

By default each run is saved below `results/`:

```text
results/<model>_<benchmark>_YYYYMMDD-HHMMSS/
├── result.json
├── metrics.json
├── raw/
├── samples/                 # when log_samples=true
├── config/
│   ├── run_config.json
│   └── runtime_versions.json
├── logs/
├── terminal.json
└── failure.json             # failures only
```

`metrics.json` contains summary, group, and per-task metrics; `raw/` preserves
the native evaluator output. Successful runs remove transient `.run/` state.
The project does not generate proof/evidence bundles for normal results.

Render a saved result as a terminal table and SVG:

```bash
python scripts/print_result.py results/<run-id> \
  --text results/<run-id>/result-summary.txt \
  --svg results/<run-id>/result-summary.svg
```

## Private deployments and public repositories

Do not commit real cluster paths, usernames, host aliases, endpoint credentials,
or operational result directories. Store private files under the ignored
`config/private/` directory or outside the repository and pass absolute paths to
the CLI. See [docs/deployment.md](docs/deployment.md).

Before publishing a fork, run:

```bash
python scripts/check_public_tree.py
```

The CI workflow runs the same privacy check.

## Extending adapters

Built-in adapter kinds are Device, Runtime, Environment, Backend, Dataset,
Binding, and Evaluator. An adapter is an executable JSON protocol boundary; Core
does not import the implementation framework. See
[ARCHITECTURE_AND_ADAPTER_PROTOCOL.md](ARCHITECTURE_AND_ADAPTER_PROTOCOL.md) for
object contracts and lifecycle rules.

## Development

```bash
python scripts/check_public_tree.py
python tests/static_contract_check.py
python -m unittest discover -s tests -p 'test_*.py'
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
