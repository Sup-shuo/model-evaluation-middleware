# Sanitized MetaX C500 validation record

This record intentionally omits usernames, private mount paths, hostnames and run
identifiers. It documents product coverage, not trusted evidence.

Current coverage includes full text BBH E2E on the recorded stack, including
result publication and clean owned-process cleanup. The smoke workflow below
is an onboarding check, separate from full benchmark validation.

## Initial smoke environment

- Accelerator: MetaX C500 64 GiB, one selected device
- Runtime: MACA 3.8.0.23, driver 3.6.11
- Backend: vLLM 0.23.0 with vLLM-MetaX 0.23.0
- Evaluator: lm-evaluation-harness 0.4.13.dev0
- Harness revision: `f4d4b3de3ee6741a7151a9fe74945ee515262f4c`
- Transformers: 5.14.1
- Dataset: local BBH, 3-shot, 24 tasks, one effective example per task

## Exercised product behavior

- System-owned MetaX selection and MACA environment patching
- Managed vLLM service startup and readiness checks
- `local-completions` prompt-logprobs/echo capability check
- Result, task metrics, raw output and sample publication
- Graceful owned-process cleanup with no remaining listener

This was a real integration smoke, not a complete benchmark score.

## Reproduction shape

```bash
eval-manager validate --system-config <metax-system> --evaluation-config <bbh-evaluation> --smoke
eval-manager doctor   --system-config <metax-system> --evaluation-config <bbh-evaluation> --smoke
eval-manager run      --system-config <metax-system> --evaluation-config <bbh-evaluation> --smoke
# Omit --smoke for the complete configured benchmark.
eval-manager run      --system-config <metax-system> --evaluation-config <bbh-evaluation>
eval-manager inspect  results/<run-id>
```

Machine-specific paths must be supplied in the System configuration.
