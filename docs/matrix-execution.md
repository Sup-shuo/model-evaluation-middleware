# Matrix execution lifecycle

Matrix execution has two inputs: the normal System/Model/Evaluation catalog for
most users, and an explicit MatrixSpec for advanced cross-product workflows.
Both routes freeze exact child ExecutionPlans before execution. Core executes
locally and serially; it exports plans for external schedulers but does not
become a distributed scheduler.

```text
System + Model + Evaluation       explicit MatrixSpec
              |                         |
              +------ resolve ----------+
                         |
                    MatrixPlan
                   /          \
          local executor      matrix-export
                |                  |
          Batch Result       jobs + shards + exact plans
                                   |
                            external scheduler/workers
                                   |
                              Run Results
```

## Route A: user configuration

This is the recommended route for ordinary evaluations, including mixed model
sizes with per-model `device_count` values:

```bash
eval-manager check \
  --system-config <system> \
  --evaluation-config <evaluation>

eval-manager plan \
  --system-config <system> \
  --evaluation-config <evaluation> \
  -o /tmp/evaluation-plan.json
```

`plan` writes a MatrixPlan even when the selection contains one model. Execute
the same resolved intent directly or from the saved plan:

```bash
# Resolve and execute in one command
eval-manager run --system-config <system> --evaluation-config <evaluation>

# Execute the already frozen plan
eval-manager run-plan /tmp/evaluation-plan.json
```

Add `--smoke` to `check`, `plan`, or `run` when one sample per task is intended.
Do not add it to a saved plan.

## Route B: explicit MatrixSpec

Use an explicit MatrixSpec when the axes themselves are the product input:

```bash
eval-manager matrix-validate <matrix-spec.yaml>
eval-manager matrix-expand <matrix-spec.yaml> -o /tmp/matrix-runs.json
eval-manager matrix-plan <matrix-spec.yaml> -o /tmp/matrix-plan.json
```

`matrix-expand` previews combinations. `matrix-plan` resolves every combination,
checks compatibility and resources, and freezes child ExecutionPlans. Review the
plan before execution.

## Local serial execution

Execute directly from the MatrixSpec:

```bash
eval-manager matrix-run <matrix-spec.yaml> --render-summary
```

Or execute the saved MatrixPlan without resolving configuration again:

```bash
eval-manager run-matrix-plan /tmp/matrix-plan.json --render-summary
```

A local Matrix execution acquires machine resource locks, runs children
serially, and publishes `results/_batches/<batch-id>/`. Use
`--continue-on-error` when independent children may continue after an ordinary
run failure. Cleanup-critical failures still stop the batch. Resume the same
saved plan against its existing batch directory with:

```bash
eval-manager run-matrix-plan /tmp/matrix-plan.json \
  --resume-dir results/_batches/<batch-id>
```

## External scheduler export

Export a saved MatrixPlan rather than teaching Core to submit Slurm, Kubernetes,
Ray, or site-specific jobs:

```bash
eval-manager matrix-export /tmp/matrix-plan.json \
  -o /tmp/matrix-jobs \
  --shards 8 \
  --strategy resource_balanced
```

The output contains:

```text
matrix-jobs/
├── manifest.json
├── plans/                  # Exact executable child ExecutionPlans
├── jobs/                   # Scheduler-neutral intent and logical resources
└── shards/                 # Compatible, resource-balanced job groups
```

The scheduler reads logical requirements from `jobs/` and `shards/`, selects a
worker compatible with the frozen paths, environments, and device assignment,
and runs each referenced plan without rewriting it:

```bash
eval-manager run-plan /tmp/matrix-jobs/plans/<child-plan>.json
```

Exported workers publish normal Run Result products. Dispatch, retries, transfer,
and distributed aggregation remain scheduler responsibilities; local
`matrix-run` and `run-matrix-plan` are the commands that publish the built-in
Batch Result product.

## Validate products

```bash
eval-manager result-check results/_batches/<batch-id>
eval-manager inspect results/_batches/<batch-id>
eval-manager result-check results/<run-id>
```

A Batch check validates its summary, run index, three TSV projections, and every
referenced successful Run Result. See [Final result product protocol](result-product.md).
