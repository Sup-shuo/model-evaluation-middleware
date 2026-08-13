# Changelog

## 4.1.0-alpha30

### Portable configuration

- Separated machine-owned System configuration, reusable Model identity, and
  run-local Evaluation selection.
- Added recursive one-model-per-file catalogs and backend-specific model loading
  namespaces.
- Added generic NVIDIA/CUDA/vLLM, MLU/Neuware/vLLM, and CPU/llama.cpp examples.
- Removed runtime installation-directory assumptions from Python; vendor roots,
  framework checkouts, executables, model mounts, caches, and result locations
  are selected by System configuration or explicit environment variables.

### Execution and results

- Added backend and evaluator environment isolation, hardware/runtime patches,
  bounded preflight, service capability probing, process-group cleanup, and
  serial multi-model execution.
- Added complete `result.json`, group/task `metrics.json`, native evaluator
  output, optional samples, effective configuration, runtime versions, and logs.
- Added a text/SVG renderer for saved task-level results.
- Kept external model and dataset digests optional; the product records
  reproducibility inputs but does not claim tamper-proof evidence.

### Project layout and publishing

- Uses one root-level `model_evaluation/` source package; no redundant one-child
  `src/` wrapper and no build step for normal use.
- Renamed internal specs to `presets/` to distinguish them from
  user-maintained `config/`.
- Added public-tree privacy checks, private-config guidance, CI, and Apache-2.0.
