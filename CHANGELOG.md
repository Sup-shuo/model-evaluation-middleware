# Changelog

## 4.1.0-rc1

- Upgraded user System and Evaluation documents to schema 1.3: every run now
  selects Backend and Evaluator profiles explicitly.
- Added per-model `resources.device_count` allocation from the System device
  pool, including Adapter-owned tensor-parallel derivation and validation.
- Updated onboarding, Mock, public examples, documentation, and tests for the
  new configuration contract.
- Split configuration compilation, execution planning, run lifecycle, and
  orchestration into focused Core modules with explicit resolved-object Schemas.
- Added configuration catalog listing, checking, and preview-first migration,
  plus smoke overlays that do not rewrite Evaluation YAML.
- Added a canonical capability vocabulary and richer diagnostics shared by
  planning, compatibility checks, and Adapter operation validation.
- Extended Matrix exports with scheduler-neutral jobs, resource-balanced
  sharding, resumable batch products, and dedicated inspection Schemas.
- Added CLI, configuration-management, capability, Matrix lifecycle, and result
  product documentation for the release-candidate surface.

## 4.1.0-alpha33

### Integration and documentation

- Added contract-tested EvalScope Binding and Evaluator adapters for existing
  OpenAI-compatible Chat Completions services.
- Added task-oriented installation, component, configuration, model-catalog,
  and Adapter extension guides while keeping deployment records private.
- Extended the standalone model-conversion tool with structural Safetensors
  readiness checks; it remains explicit, non-destructive, and outside normal
  evaluation execution.
- Retained the sanitized real MLU full-BBH result projection and reduced the
  public model catalog to two portable examples.

## 4.1.0-alpha32

### Portable workflow and public API

- Added `check` and `explain` over one structured pre-run workflow that combines
  configuration validation, Doctor, plan preview and read-only resource checks.
- Added a read-only `model_evaluation.results.load_run` API for the stable result
  product and deterministic `matrix-export` bundles for external schedulers.
- Added a hardware-free `eval-manager demo` using the same Matrix, process and
  result-publication path as real evaluations.
- Added opt-in `--render-summary` output for successful runs and a checked-in,
  sanitized result-product example for consumer integration tests.
- Added project-relative cache roots and refreshed the configuration and result
  documentation without changing the System/Model/Evaluation ownership boundary.
- Added an explicitly invoked, non-destructive checkpoint inspection and Qwen
  conversion toolkit. It remains separate from `eval-manager`, never converts
  automatically, and refuses to overwrite source or existing output paths.

### Adapter coverage

- Added MetaX Device and MACA Runtime adapters plus contract tests and a sanitized
  single-device validation record.
- Kept the compatibility statement explicit: a smoke run proves that one tested
  stack starts and cleans up; it is not a full model-quality evaluation.

## 4.1.0-alpha31

### Product and onboarding

- Froze independent 1.0 schemas for `result.json`, `metrics.json`,
  `terminal.json` and `failure.json`.
- Added `result-check`, human/JSON `inspect`, human-readable `doctor`,
  `adapter-check` and a non-overwriting `init` command.
- Added external Adapter roots and installed Python entry-point discovery.
- Added an optional Controller environment snapshot and strict dependency file.

### Maintainability

- Split result publication, backend preflight, run finalization, runtime version
  capture, Matrix product/config handling, procfs and stale-process recovery out
  of the former orchestration hotspots.
- Added a compatibility matrix, sanitized validation records, contribution
  guidance and a security policy.
- Added Linux CI for Python 3.10 and 3.12 plus release and installed-wheel gates.
- Preserved the product boundary: runtime/configuration records support
  reproduction, not cryptographic evidence or tamper-proof claims.

## 4.1.0-alpha30

### Configuration and portability

- Separated machine-owned System configuration, reusable Model identity and
  experiment-owned Evaluation selection.
- Added recursive one-model-per-file catalogs and ID-based configuration
  selection.
- Kept host paths, devices, environments, executables and service capacity in
  System configuration rather than production Python or reusable Model files.
- Added portable NVIDIA/CUDA, Cambricon MLU/Neuware and CPU examples using
  placeholder paths.
- Made model and dataset byte digests optional. Dataset integrity defaults to
  `basic`; `strict` remains opt-in.

### Evaluation and results

- Added structured Backend and Evaluator preflight checks using the same
  Device → Runtime → Environment wrapping as execution.
- Added offline BBH support, fixed seeds, short local-time run identifiers and
  complete summary/group/task metrics.
- Published native framework output, optional samples, effective configuration,
  observed runtime versions and readable logs in a project-local `results/`
  directory.
- Added a terminal/SVG result renderer and graceful owned-process cleanup.
- Removed default evidence/hash bundles from normal results.

### Architecture

- Adopted one root-level `model_evaluation/` application package and removed
  the redundant one-child `src/` wrapper.
- Renamed packaged `specs/` to `presets/`; project-level `config/` remains the
  user-maintained configuration entry point.
- Retained Adapter Protocol 1.0 compatibility and optional Backend preflight for
  third-party integrations.
