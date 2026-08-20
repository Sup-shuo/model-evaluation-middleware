# Sanitized NVIDIA validation record

This record intentionally omits usernames, private mount paths, hostnames and run
identifiers. It documents product coverage, not trusted evidence.

## Environment

- GPU: NVIDIA A100-SXM4-40GB
- GPU count: one for the normal runs; two with tensor parallelism for the 27B run
- Driver: 550.90.07
- Backend: vLLM 0.25.1, OpenAI-compatible Completions API
- Evaluator: lm-evaluation-harness 0.4.13.dev0
- Harness revision: `f4d4b3de3ee6741a7151a9fe74945ee515262f4c`
- Transformers: 5.14.1
- Dataset: local BBH, 3-shot, 24 tasks, 5,761 effective examples

## Exercised product behavior

- Separate Backend and Evaluator Python environments
- Single-GPU and two-GPU tensor-parallel managed service startup
- `local-completions` prompt-logprobs/echo capability check
- Full result, task metrics, raw output and optional sample publication
- Graceful owned-process cleanup with no remaining listener

## Reproduction shape

```bash
eval-manager validate --system-config <nvidia-system> --evaluation-config <full-bbh>
eval-manager doctor   --system-config <nvidia-system> --evaluation-config <full-bbh>
eval-manager run      --system-config <nvidia-system> --evaluation-config <full-bbh>
eval-manager inspect  results/<run-id>
```

Machine-specific paths and credentials are deliberately not published. A user
must supply them in the System configuration.
