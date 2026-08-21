# Environment isolation

The Controller, inference Backend, and Evaluator are separate execution roles.
They may share one environment when dependencies are compatible, but the
configuration does not require that.

| Role | Typical contents | Configured in |
|---|---|---|
| Controller | Middleware, PyYAML, jsonschema | The shell or installed package |
| Backend | Vendor Runtime integration and inference engine | `profiles.environment` + `profiles.backend` |
| Evaluator | Dataset/evaluation framework and tokenizer stack | `profiles.environment` + `profiles.evaluator` |

## Environment Adapter types

- `current`: use the Python/process environment that launched the Controller.
- `conda`: resolve commands through a named or path-based Conda environment.
- `venv`: use an existing Python virtual environment directory.

The middleware selects and wraps an existing environment. It does not install
drivers, vendor SDKs, inference frameworks, or evaluator packages during a
normal run.

## Recommended pattern

Use a dedicated environment when frameworks pin conflicting versions:

```yaml
profiles:
  environment:
    backend-env:
      type: venv
      profile: /opt/venvs/vllm
    evaluator-env:
      type: conda
      profile: /opt/conda/envs/lm_eval_env
      executable: /opt/conda/bin/conda

  backend:
    vllm:
      type: vllm
      environment: backend-env

  evaluator:
    lm_eval:
      type: lm_eval
      root: /opt/lm-evaluation-harness
      environment: evaluator-env
```

The evaluator may call an OpenAI-compatible service while remaining outside
the Backend environment. It does not need to load model weights itself.

## Reproducibility options

Normal runs record observed runtime versions in the result product. This is a
snapshot, not a dependency lock. When a stricter Controller environment record
is needed, export one explicitly:

```bash
eval-manager environment-snapshot --help
```

Pin evaluator and Backend versions according to the deployment policy, and
record framework revisions in their profiles. Bit-level equality across
hardware or frameworks is not implied by an environment snapshot.
