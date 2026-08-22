# Evaluation frameworks

An Evaluator Adapter performs framework preflight, plans the evaluation
process, and normalizes native output into the final result product. Benchmark
semantics and metric algorithms remain owned by the selected framework and
Binding.

| Evaluator | Upstream | Built-in integration | Repository validation |
|---|---|---|---|
| lm-evaluation-harness | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Evaluator + generic and BBH Bindings | Full BBH E2E on NVIDIA and MLU; C500 smoke E2E |
| EvalScope | [modelscope/evalscope](https://github.com/modelscope/evalscope) | Evaluator + Binding | Existing-service GSM8K smoke E2E on NVIDIA A100 |
| Reference Evaluator | Included mock implementation | Evaluator + Binding | CPU Mock E2E |
| … | External Adapter | Evaluator and, when needed, Binding | Deployment-specific |

## Register multiple frameworks

One System can hold several evaluation environments and profiles:

```yaml
profiles:
  environment:
    lm-eval-env:
      type: conda
      profile: /opt/conda/envs/lm_eval_env
      executable: /opt/conda/bin/conda
    evalscope-env:
      type: conda
      profile: /opt/conda/envs/evalscope_env
      executable: /opt/conda/bin/conda

  evaluator:
    lm_eval:
      type: lm_eval
      root: /opt/lm-evaluation-harness
      environment: lm-eval-env
    evalscope:
      type: evalscope
      environment: evalscope-env
      parameters:
        executable: evalscope
        expected_version: 1.10.0
```

Select one for a run in Evaluation:

```yaml
evaluator:
  profile: evalscope
  parameters: {}
```

Unused environments are not activated. A separate environment is recommended
when evaluator dependency ranges conflict, but compatible roles may share one.

## Adding another evaluation framework

Most frameworks require:

1. an Evaluator Adapter for preflight, process planning, normalization, and
   runtime snapshot;
2. a Binding Adapter that turns the canonical Benchmark and Dataset artifact
   into the framework's task representation;
3. framework-specific environment configuration in System;
4. contract tests, then a real smoke before claiming real-machine support.

Core should not gain framework-name branches. Follow
[Adding an Adapter](../adapters/adding-an-adapter.md) and reuse canonical
Service, Dataset, Task, Process, and Result objects.
