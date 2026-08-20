# Changelog

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
