# Sanitized Cambricon MLU validation record

This record intentionally omits usernames, private mount paths, hostnames and run
identifiers. It documents product coverage, not trusted evidence.

## Environment

- Accelerator: Cambricon MLU, one visible device per run
- Runtime: Neuware 4.7.2
- Backend: vLLM 0.21.0 with the vLLM-MLU plugin
- Evaluator: lm-evaluation-harness 0.4.13.dev0
- Harness revision: `f4d4b3de3ee6741a7151a9fe74945ee515262f4c`
- Transformers: 5.14.1
- Dataset: local BBH, 3-shot, 24 tasks, 5,761 effective examples

## Exercised product behavior

- System-owned MLU selection and Neuware environment patching
- Managed vLLM service startup and readiness checks
- `local-completions` prompt-logprobs/echo capability check
- Full result, task metrics, raw output and optional sample publication
- Graceful owned-process cleanup with no remaining listener

## Reproduction shape

```bash
eval-manager validate --system-config <mlu-system> --evaluation-config <full-bbh>
eval-manager doctor   --system-config <mlu-system> --evaluation-config <full-bbh>
eval-manager run      --system-config <mlu-system> --evaluation-config <full-bbh>
eval-manager inspect  results/<run-id>
```

Machine-specific paths and credentials are deliberately not published. A user
must supply them in the System configuration.
