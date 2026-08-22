# Configuration overview

Users maintain three configuration types. Keeping them separate is what makes
an evaluation portable between machines without mixing hardware paths into the
model identity or benchmark intent.

| Configuration | Owns | Detailed guide |
|---|---|---|
| System | Hardware, Runtime, environments, Backend/Evaluator profiles, model root, cache, result paths | [System configuration](configuration/system.md) |
| Model | Model identity, source, architecture, format, quantization, context, and Backend loading parameters | [Model configuration](configuration/model.md) |
| Evaluation | Models, benchmarks, selected profiles, seeds, limits, and one-run overrides | [Evaluation configuration](configuration/evaluation.md) |

The Controller resolves these user documents into internal canonical Specs.
Files under `model_evaluation/presets/` are packaged implementation data and
are not user configuration.

## Recommended layout

```text
config/
├── systems/
│   ├── nvidia.yaml
│   └── mlu.yaml
├── models/
│   ├── README.md
│   └── qwen/
│       └── qwen35_08b_base.yaml
├── evaluations/
│   └── bbh.yaml             # Reused for smoke and full runs
├── system.yaml
└── evaluation.yaml
```

System and Evaluation may be supplied as a file path or as an ID resolved
recursively below `config/systems/` and `config/evaluations/`; for example,
`--evaluation-config teams/bbh` resolves
`config/evaluations/teams/bbh.yaml`. Model files are discovered recursively
below `config/models/`; their directories organize the catalog but do not
create namespaces. Evaluations always select a model by its globally unique
`id`, not by file path.

Use `eval-manager config list`, `config show`, `config check`, and the
dry-run-first `config migrate` workflow for catalog maintenance. See
[Configuration management](configuration/management.md).

## Selection and precedence

Evaluation must explicitly select `backend.profile` and `evaluator.profile`.
System registers the available implementations and environments but does not
silently choose either framework. Hardware may be selected through
`evaluation.profiles.hardware`; otherwise the System hardware default, or its
only registered hardware profile, is used. Only selected profiles and their
referenced environments enter the plan.

One-run overrides in Evaluation take precedence over reusable Model and System
defaults without modifying either catalog entry. Identity-changing fields such
as model source, architecture, format, or quantization require a new Model ID
instead of an override.

The CLI resolves configuration in this order:

1. `--system-config` / `--evaluation-config`;
2. `MODEL_EVAL_SYSTEM_CONFIG` / `MODEL_EVAL_EVALUATION_CONFIG`;
3. `config/system.yaml` / `config/evaluation.yaml`.

## Ownership rules

- Put the exposed device pool, Runtime roots, executable paths, environments,
  ports, concurrency, and capacity policy in System.
- Put source, tokenizer, architecture, format, quantization, context, and
  stable Backend-specific loading requirements in Model.
- Put Backend/Evaluator selection, benchmark choice, seeds, temporary limits,
  per-model device counts, and temporary parameter overrides in Evaluation.
- Put Dataset-to-framework behavior in Dataset/Binding Adapters and benchmark
  presets, not in machine profiles.

## Environment isolation

A Backend and an Evaluator may use different Conda/venv environments. One
System can register any practical number of named environments and evaluator
profiles. See [Environment isolation](configuration/environments.md).

## Dataset and cache boundary

`paths.cache` declares the materialization root. Dataset and Evaluator Adapters
own their internal cache layout and offline variables. A typical BBH chain is:

```text
BenchmarkSpec(bbh)
  -> DatasetProvider(bbh_local)
  -> FrameworkBinding(lm_eval.bbh)
  -> Evaluator(lm_eval)
```

The default `basic` integrity policy checks the files required to run. A
Dataset Adapter may expose a stronger `strict` policy. Core does not require
all model and dataset assets to be registered as cryptographic evidence.

## Validate before running

```bash
eval-manager check \
  --system-config nvidia \
  --evaluation-config teams/bbh --smoke

eval-manager explain \
  --system-config nvidia \
  --evaluation-config teams/bbh --smoke
```

`--smoke` is a transient CLI mode: it freezes one sample per task into the
resolved execution plan without changing the Evaluation file. Omit the flag for
a full benchmark. `check` combines Schema validation, Doctor, plan preview, and
read-only resource checks. `explain` presents the effective compatibility and
resource decision without starting a model service.

## Related guides

- [Installation and first evaluation](installation.md)
- [Model catalog examples](models/index.md)
- [Configuration management](configuration/management.md)
- [Components and integrations](components/index.md)
- [Compatibility and validation status](compatibility.md)
- [Result product protocol](result-product.md)
