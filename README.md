# Model Evaluation Middleware

[Chinese (Simplified)](docs/README.zh-CN.md)

A model-evaluation glue layer for engineering teams. It connects hardware,
runtime environments, inference backends, models, datasets, and evaluation
frameworks into a recordable, reproducible, and portable execution path, then
publishes a consistent result product.

```mermaid
flowchart LR
    CONFIG["Configuration<br/>System · Model · Evaluation"]
    CORE["Core<br/>Validate · Check · Plan · Run"]
    ADAPTERS["Adapters<br/>Hardware · Backend · Dataset · Evaluator"]
    ENV["Execution environment<br/>CPU / GPU / NPU · inference and evaluation frameworks"]
    RESULTS["Result product<br/>result · metrics · raw · config · logs"]

    CONFIG --> CORE --> ADAPTERS --> ENV
    ENV --> CORE --> RESULTS

    classDef config fill:#e8f1ff,stroke:#2563eb,color:#0f172a;
    classDef core fill:#ede9fe,stroke:#7c3aed,color:#0f172a;
    classDef adapter fill:#fff7ed,stroke:#ea580c,color:#0f172a;
    classDef external fill:#ecfdf5,stroke:#059669,color:#0f172a;
    classDef product fill:#f8fafc,stroke:#334155,color:#0f172a;

    class CONFIG config;
    class CORE core;
    class ADAPTERS adapter;
    class ENV external;
    class RESULTS product;
```

## Understand the project in 10 seconds, without a GPU or NPU

Install the Controller dependencies and run this from the repository root:

```bash
python -m pip install -r requirements.txt
./eval-manager demo
```

`demo` uses the CPU Runtime, a loopback Reference Backend, `dataset/virtual`,
`binding/reference_eval`, and `evaluator/reference_eval`. It still exercises
the real configuration parser, Matrix, process management, result publication,
and consistency validation. It does not download a model, access the network,
or use a physical GPU/NPU. It normally returns within 10 seconds:

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

The demo still creates a normal result directory. `contract_ok=1` only means
that the glue pipeline completed. It is **not a model-quality score** and does
not claim that any physical hardware or real evaluation framework passed.

## Project scope

| In scope | Out of scope |
|---|---|
| Parse System, Model, and Evaluation configurations | Download or train models |
| Discover devices, runtimes, and execution environments | Reimplement inference engines |
| Start or connect to a Backend and validate required capabilities | Reimplement benchmarks |
| Bind a Dataset to an Evaluator | Distributed cluster scheduling |
| Save effective configuration, versions, complete metrics, and raw output | Model governance or experiment-tracking platforms |
| Clean up processes started by this project after failure | Forensics, tamper resistance, or trusted proof |

Here, “reproducible” means recording the effective configuration and observable
runtime versions so that an evaluation can be reconstructed on another machine.
It does not promise bit-level equality across hardware and does not present
results as tamper-proof evidence.

## Current validation coverage

| Level | Current coverage | Meaning |
|---|---|---|
| Full real-machine E2E | NVIDIA A100/CUDA and Cambricon MLU/Neuware; vLLM + lm-eval + BBH | 24 tasks, 5,761 samples, result publication, and cleanup passed |
| Real-machine smoke E2E | MetaX C500/MACA; vLLM-MetaX + lm-eval + BBH | Single-device service and 24-subtask smoke passed; this is not a full accuracy evaluation |
| Mock E2E | CPU + Reference Backend/Evaluator + virtual Dataset | Software execution and result-product pipeline passed |
| Contract-tested | AMD/ROCm, Ascend/CANN, Ollama, llama.cpp, generic OpenAI, and others | Manifest, Schema, RPC, and planning behavior passed; this is not production hardware validation |

The protocol surface is intentionally broader than the current real-machine
coverage. The presence of an Adapter directory does not mean that combination
has completed production validation. See the [compatibility matrix](docs/compatibility.md)
and [sanitized machine records](docs/validation/) for precise claims.

## Installation and the first real evaluation

Python 3.10 or newer is required. For a source checkout:

```bash
python -m pip install -r requirements.txt
./eval-manager schema-check
./eval-manager adapters
```

After installing the wheel, use the console script with the same name:

```bash
eval-manager schema-check
```

Create a minimal project without overwriting existing files:

```bash
mkdir my-evaluation
cd my-evaluation
eval-manager init . --hardware nvidia
```

`--hardware` also accepts `metax`, `mlu`, `amd`, `ascend`, and `cpu`. Replace
the generated `REPLACE_WITH_*` values, check the configuration, and then run:

```bash
eval-manager check \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml

eval-manager run \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml
```

`check` combines configuration validation, Doctor, plan preview, and read-only
resource checks without starting the model service. For step-by-step diagnosis,
use `validate`, `doctor`, `plan`, and `explain` separately.

## Three configuration types

| Configuration | Question it answers | Must not contain |
|---|---|---|
| System | What does this machine provide, where is it, and which environments should be used? | Model experiment identity or benchmark selections |
| Model | What is this model, and how should a particular Backend load it? | Device IDs, machine paths, or GPU memory ratios |
| Evaluation | Which models and benchmarks are selected for this run, and what is temporarily overridden? | Long-lived model definitions or driver-installation details |

Typical layout:

```text
config/
├── systems/                 # NVIDIA, MLU, MetaX, and other machine profiles
├── models/                  # One model per file; may be grouped by family/provider
├── evaluations/             # Smoke, full, and one-off evaluation selections
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
parameters belong to System. Model source, architecture, quantization, context,
and Backend-namespaced loading parameters belong to Model. Seeds, sample limits,
and selections for the current run belong to Evaluation. See the
[configuration guide](docs/configuration.md) for fields, precedence, caching,
and reproduction rules.

`model_evaluation/presets/` contains packaged normalization defaults, not a
second user-configuration tree. Users normally maintain only the root-level
`config/`. `model_evaluation/examples/mock/` provides the self-contained
installed `demo` example.

## Common workflows

```bash
# Hardware-free demo that returns the final JSON report
./eval-manager demo

# Complete pre-run checks; use --format json for automation
./eval-manager check --system-config mlu --evaluation-config smoke_bbh_08b

# Explain why the current combination can run or is blocked
./eval-manager explain --system-config mlu --evaluation-config smoke_bbh_08b

# Generate a plan or execute an evaluation
./eval-manager plan --system-config mlu --evaluation-config smoke_bbh_08b -o /tmp/plan.json
./eval-manager run  --system-config mlu --evaluation-config smoke_bbh_08b

# Validate and inspect a final result
./eval-manager result-check results/<run-id>
./eval-manager inspect results/<run-id>
./eval-manager inspect results/<run-id> --format json
```

Other commands:

- `init`: create a minimal project skeleton without overwriting files;
- `schema-check` / `adapters`: inspect Core Schemas and discovered Adapters;
- `adapter-check`: validate an external Adapter root before installation;
- `environment-snapshot`: optionally export the Controller Python environment;
- `matrix-export`: shard a saved Matrix plan for an external scheduler;
- `run-plan` / `run-matrix-plan`: execute saved plans or resume a batch.

Core remains a single-host, serial Matrix executor with resource locks. At
larger scale, Slurm, Kubernetes, Ray, or an internal scheduler should consume
the exported child plans instead of turning this project into a distributed
scheduler.

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

Results are written under the project-level `results/` directory:

```text
results/<run-id>/
├── result.json              # Run identity and normalized summary
├── metrics.json             # Summary, group, and per-task metrics
├── terminal.json            # Final state, local time, and cleanup result
├── failure.json             # Present only after failure
├── raw/                     # Complete framework-native output
├── samples/                 # Present only when explicitly enabled and produced
├── config/                  # Effective configuration and observable versions
└── logs/                    # Backend and Evaluator logs
```

Each of the four top-level JSON files has an independent Schema. `result-check`
and `inspect` validate their Schemas, cross-file identity and metric consistency,
success/failure rules, and public artifact-path boundaries. They do not produce
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

See the [result product protocol](docs/result-product.md). To create a terminal
or report view from an existing result:

```bash
python scripts/print_result.py results/<run-id> \
  --text results/<run-id>/result-summary.txt \
  --svg results/<run-id>/result-summary.svg
```

The TXT/SVG files are projections of saved results. They do not recompute the
score and are not proof artifacts.

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

During development, `MODEL_EVAL_ADAPTER_PATHS` may contain one or more absolute
Adapter roots. Duplicate kind/name pairs across built-ins, development roots,
and installed entry points are rejected instead of being silently overridden.

See the [architecture and Adapter protocol](ARCHITECTURE_AND_ADAPTER_PROTOCOL.md)
for extension objects, RPCs, failure semantics, and checklists.

## Repository layout

```text
model-evaluation-middleware/
├── model_evaluation/        # Single installable Python package
│   ├── core/                # Configuration, planning, execution, resources, results
│   ├── adapters/            # Built-in Adapters
│   ├── sdk/                 # Stable SDK for external Adapters
│   ├── schemas/             # Public-object and user-configuration Schemas
│   ├── presets/             # Internal normalization presets
│   ├── examples/mock/       # Hardware-free demo packaged in the wheel
│   └── commands/            # CLI command layer
├── config/                  # User-maintained System, Model, and Evaluation files
├── tests/                   # Unit, integration, and static-boundary tests
├── scripts/                 # Project release and result-view automation
├── tools/                   # Standalone manual inspection/conversion tools
├── results/                 # Generated output; excluded from wheel/release ZIP
└── eval-manager             # Source-tree entry point
```

This is a single-package application repository, so it does not keep a
redundant `src/` wrapper containing only one package. Source code is grouped by
real responsibilities rather than merging Core, Adapters, and protocols merely
to reduce the directory count.

## Documentation

- [Configuration guide](docs/configuration.md): System / Model / Evaluation,
  caching, and reproduction;
- [Result product protocol](docs/result-product.md): final JSON and Python API;
- [Compatibility matrix](docs/compatibility.md): real-machine, Mock, and
  contract-tested boundaries;
- [NVIDIA A100](docs/validation/nvidia-a100.md),
  [Cambricon MLU](docs/validation/cambricon-mlu.md), and
  [MetaX C500](docs/validation/metax-c500.md): sanitized machine records;
- [Architecture and Adapter protocol](ARCHITECTURE_AND_ADAPTER_PROTOCOL.md);
- [Contributing](CONTRIBUTING.md), [security boundary](SECURITY.md), and
  [changelog](CHANGELOG.md).

Development verification:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tests/static_contract_check.py
python3 scripts/build_release.py
```
