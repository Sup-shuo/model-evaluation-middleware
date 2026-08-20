# Model Evaluation Middleware

[简体中文](docs/README.zh-CN.md)

A configuration-driven glue layer for recording and reproducing model
evaluations across hardware, inference backends, models, datasets, and
evaluation frameworks. It orchestrates existing tools and publishes a stable
result product; it is not an inference engine, benchmark implementation, or
tamper-proof evidence system.

```text
System + Model + Evaluation
            ↓
 validate / doctor / plan / resource check
            ↓
 Device → Runtime → Environment → Backend
            ↓
 Dataset → Binding → Evaluator
            ↓
 result.json + metrics.json + raw/ + logs/
```

## Try it in 10 seconds — no GPU or NPU required

Install the controller dependencies and run the self-contained demo from the
repository root:

```bash
python -m pip install -r requirements.txt
./eval-manager demo
```

The demo uses the CPU Runtime, loopback Reference Backend, `dataset/virtual`,
`binding/reference_eval`, and `evaluator/reference_eval`. It still exercises
the production configuration resolver, Matrix planner, process manager, result
publisher, and consistency checks. It downloads no model, uses no network, and
typically finishes within 10 seconds:

```json
{
  "demo": "reference",
  "ok": true,
  "report": {
    "benchmark": "mock_demo",
    "cleanup": "clean",
    "framework": "reference_eval",
    "model": "mock-model",
    "outcome": "success",
    "summary": {
      "contract_ok": {"value": 1}
    }
  }
}
```

The demo produces a normal result directory. `contract_ok=1` only means that
the middleware path completed; it is **not a model quality score** and does not
claim that any physical accelerator or evaluation framework was validated.

## Scope

| This project does | This project does not |
|---|---|
| Resolve System, Model, and Evaluation configuration | Download or train models |
| Discover devices, runtimes, and execution environments | Reimplement inference engines |
| Start or attach to a Backend and verify required capabilities | Reimplement benchmarks |
| Bind Datasets to Evaluators | Schedule distributed clusters |
| Save effective configuration, versions, complete metrics, and raw output | Provide model governance or experiment tracking |
| Clean up processes that it started | Provide forensics, tamper resistance, or trusted evidence |

Here, reproducibility means recording the effective configuration and
observable runtime versions so that the evaluation can be reconstructed on
another machine. It does not promise bit-level equality across hardware and
does not turn results into cryptographic evidence.

## Current validation coverage

| Level | Current coverage | Meaning |
|---|---|---|
| Full hardware E2E | NVIDIA A100/CUDA and Cambricon MLU/Neuware; vLLM + lm-eval + BBH | 24 tasks, 5,761 samples, result publication, and cleanup passed |
| Hardware smoke E2E | MetaX C500/MACA; vLLM-MetaX + lm-eval + BBH | Single-device service and 24-subtask smoke passed; this is not a full accuracy evaluation |
| Mock E2E | CPU + Reference Backend/Evaluator + virtual Dataset | Software execution and result-product path passed |
| Contract-tested | AMD/ROCm, Ascend/CANN, Ollama, llama.cpp, generic OpenAI, and others | Manifest, Schema, RPC, and planning behavior passed; this is not production hardware validation |

The project therefore has broad protocol coverage and focused hardware
coverage. The presence of an Adapter does not mean that every combination has
passed production validation. See the [compatibility matrix](docs/compatibility.md)
and [sanitized hardware records](docs/validation/) for exact claims.

## Installation and first real evaluation

Python 3.10 or newer is required. From source:

```bash
python -m pip install -r requirements.txt
./eval-manager schema-check
./eval-manager adapters
```

After installing a wheel, the same CLI is available as a console script:

```bash
eval-manager schema-check
```

Create a minimal project without overwriting existing files:

```bash
mkdir my-evaluation
cd my-evaluation
eval-manager init . --hardware nvidia
```

`--hardware` also supports `metax`, `mlu`, `amd`, `ascend`, and `cpu`. Replace
the generated `REPLACE_WITH_*` placeholders, check the configuration, and run:

```bash
eval-manager check \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml

eval-manager run \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml
```

`check` combines configuration validation, Doctor, plan preview, and read-only
resource checks without starting a model service. For step-by-step diagnosis,
use `validate`, `doctor`, `plan`, and `explain` separately.

## Three configuration layers

| Configuration | Question it answers | It should not contain |
|---|---|---|
| System | What does this machine provide, where is it, and which environments should be used? | Model experiment identity or benchmark selection |
| Model | What is this model and how should each Backend load it? | Device indices, machine paths, or memory utilization |
| Evaluation | Which models and benchmarks are selected for this run, and what temporary overrides apply? | Long-lived model definitions or driver installation details |

Typical layout:

```text
config/
├── systems/                 # NVIDIA, MLU, MetaX, and other machine profiles
├── models/                  # One model per file; group by family or provider
├── evaluations/             # Smoke, full, and one-off experiment selections
├── system.yaml              # Generic template used by init
└── evaluation.yaml          # Generic template used by init
```

Model and Evaluation files can remain byte-for-byte identical across machines;
only the System changes:

```bash
./eval-manager check --system-config mlu --evaluation-config smoke_bbh_08b
./eval-manager check --system-config nvidia --evaluation-config smoke_bbh_08b
```

Machine paths, devices, Runtime, Backend/Evaluator environments, and capacity
settings belong to System. Model source, architecture, quantization, context,
and Backend-specific loading options belong to Model. Seeds, sample limits, and
selection belong to Evaluation. See the [configuration guide](docs/configuration.md)
for fields, override precedence, cache behavior, and reproducibility rules.

`model_evaluation/presets/` contains internal normalization presets shipped with
the package; it is not a second user-configuration system. Normal users maintain
only the root `config/`. `model_evaluation/examples/mock/` provides the
self-contained installed `demo`.

## Common workflows

```bash
# Hardware-free demo with a final JSON report
./eval-manager demo

# Complete pre-run check; use --format json for automation
./eval-manager check --system-config mlu --evaluation-config smoke_bbh_08b

# Explain why the current combination can run or is blocked
./eval-manager explain --system-config mlu --evaluation-config smoke_bbh_08b

# Generate a plan or execute an evaluation
./eval-manager plan --system-config mlu --evaluation-config smoke_bbh_08b -o /tmp/plan.json
./eval-manager run  --system-config mlu --evaluation-config smoke_bbh_08b

# Validate and inspect a completed result
./eval-manager result-check results/<run-id>
./eval-manager inspect results/<run-id>
./eval-manager inspect results/<run-id> --format json
```

Other entry points:

- `init`: create a project skeleton without overwriting existing files;
- `schema-check` / `adapters`: inspect Core Schemas and discovered Adapters;
- `adapter-check`: validate an external Adapter root before installation;
- `environment-snapshot`: optionally export the Controller Python environment;
- `matrix-export`: shard a saved Matrix plan for an external scheduler;
- `run-plan` / `run-matrix-plan`: execute saved plans or resume a batch.

Core deliberately remains a single-node, serial Matrix executor with resource
locking. At larger scale, Slurm, Kubernetes, Ray, or an internal scheduler
should consume exported child plans instead of turning this project into a
distributed scheduler.

Model-format conversion is not part of the automatic evaluation workflow.
When a Backend cannot load a checkpoint format, the operator must explicitly
use the root-level manual tool to create a separate derivative:

```bash
python tools/model_convert.py inspect /data/OWNER/MODEL
python tools/model_convert.py routes
python tools/model_convert.py convert --help
```

`eval-manager` never invokes this tool. It does not overwrite the source model
or an existing output. See the [configuration guide](docs/configuration.md) for
routes and derivative Model registration rules.

## Result product

Default run name:

```text
<platform>_<model-id>_<backend>_<benchmark-id>_YYMMDD-HHMM
```

Results are written to the project-level `results/` directory:

```text
results/<run-id>/
├── result.json              # Run identity and normalized summary
├── metrics.json             # Summary, group, and per-task metrics
├── terminal.json            # Final outcome, local time, and cleanup status
├── failure.json             # Present only on failure
├── raw/                     # Complete framework-native output
├── samples/                 # Present only when explicitly enabled and emitted
├── config/                  # Effective configuration and observable versions
└── logs/                    # Backend and Evaluator logs
```

The four top-level JSON objects each have an independent Schema. `result-check`
and `inspect` validate Schemas, cross-file identity and metric consistency,
success/failure rules, and public artifact path boundaries. They do not provide
cryptographic proof.

Python applications can consume the same protocol directly:

```python
from model_evaluation.results import load_run

run = load_run("results/<run-id>")
summary = run.metrics.summary()
tasks = run.metrics.tasks()
runtime = run.runtime()
artifacts = run.artifacts()
```

See the [result-product protocol](docs/result-product.md). To create a terminal
summary or report view:

```bash
python scripts/print_result.py results/<run-id> \
  --text results/<run-id>/result-summary.txt \
  --svg results/<run-id>/result-summary.svg
```

TXT and SVG files only project an existing result. They do not recompute scores
and are not proof artifacts.

## Adapter extensions

Seven Adapter kinds cover Device, Runtime, Environment, Backend, Dataset,
Binding, and Evaluator. Each Adapter is a JSON-over-stdio subprocess with a
`manifest.json`; Core does not import vendor SDKs or evaluation frameworks.

Third-party Adapters can be discovered through Python entry points without
being committed to this repository:

```toml
[project.entry-points."model_evaluation.adapters"]
"backend.my_engine" = "my_eval_plugin.adapters.backend.my_engine"
```

During development, `MODEL_EVAL_ADAPTER_PATHS` can point to one or more absolute
Adapter roots. Startup fails if built-in, development, or installed entry-point
Adapters declare the same kind/name; silent replacement is not allowed.

See [Architecture and Adapter Protocol](ARCHITECTURE_AND_ADAPTER_PROTOCOL.md)
for extension objects, RPCs, failure semantics, and review checklists.

## Repository layout

```text
model-evaluation-middleware/
├── model_evaluation/        # Single installable Python package
│   ├── core/                # Configuration, planning, execution, resources, results
│   ├── adapters/            # Built-in Adapters
│   ├── sdk/                 # Stable SDK for external Adapters
│   ├── schemas/             # Public object and user-configuration Schemas
│   ├── presets/             # Internal normalization presets
│   ├── examples/mock/       # Hardware-free demo shipped in the wheel
│   └── commands/            # CLI command layer
├── config/                  # User-maintained System, Model, and Evaluation files
├── tests/                   # Unit, integration, and static boundary tests
├── scripts/                 # Release, privacy, and result-view automation
├── tools/                   # Standalone manual inspection/conversion utilities
├── results/                 # Runtime output; excluded from wheel and release ZIP
└── eval-manager             # Source-tree entry point
```

This is a single-package application repository, so it does not keep a `src/`
wrapper containing only one package. Internal directories follow actual
responsibilities instead of merging Core, Adapters, and protocols merely to
reduce the directory count.

## Documentation

- [中文 README](docs/README.zh-CN.md)
- [Configuration guide](docs/configuration.md)
- [Result-product protocol](docs/result-product.md)
- [Compatibility matrix](docs/compatibility.md)
- Sanitized hardware records: [NVIDIA A100](docs/validation/nvidia-a100.md),
  [Cambricon MLU](docs/validation/cambricon-mlu.md), and
  [MetaX C500](docs/validation/metax-c500.md)
- [Architecture and Adapter Protocol](ARCHITECTURE_AND_ADAPTER_PROTOCOL.md)
- [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), and
  [Changelog](CHANGELOG.md)

Development checks:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tests/static_contract_check.py
python3 scripts/build_release.py
```
