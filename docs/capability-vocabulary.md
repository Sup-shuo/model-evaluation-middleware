# Capability vocabulary and diagnostics

Capability paths are the small interoperability language shared by Core and
Adapters. Stable Core terms cover common facts such as `device.vendor`,
`runtime.family`, `runtime.version`, `runtime.compatible_device_vendors`,
`service.context_length`, tokenizer availability, and the common service probes
`service.generation`, `service.chat`, `service.completion_logprobs`,
`service.echo`, `service.tokenize`, and `service.detokenize`. Adapter-owned
extensions remain valid and do not require a Core release.

The vocabulary is deliberately not a closed allow-list. Core validates the
shape of a requirement and evaluates its operator; it does not attempt to know
every hardware, inference-engine, or evaluator feature.

An unmet requirement exposes a structured diagnostic alongside its readable
message:

```json
{
  "code": "CAPABILITY_REQUIREMENT_FAILED",
  "severity": "error",
  "path": "runtime.family",
  "operator": "in",
  "expected": ["cuda"],
  "actual": "cpu",
  "optional": false,
  "vocabulary": "core",
  "message": "selected runtime is not allowed"
}
```

The same record appears in a frozen `ExecutionPlan` and in failure diagnostics
when execution-time facts no longer satisfy the plan. Consumers should use
`code`, `path`, `expected`, and `actual` for automation, while treating
`message` as presentation text. Unknown paths are classified as `extension`,
not rejected.
