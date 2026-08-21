# Adding an Adapter

This guide covers the integration workflow. The
[Architecture and Adapter protocol](../../ARCHITECTURE_AND_ADAPTER_PROTOCOL.md)
remains the normative contract.

## 1. Select one responsibility

Choose Device, Runtime, Environment, Backend, Dataset, Binding, or Evaluator.
Split responsibilities when an integration spans more than one kind. For
example, a new evaluation framework normally needs an Evaluator plus a Binding,
not changes to Core.

## 2. Create the directory contract

```text
my-adapters/
└── evaluator/
    └── example/
        ├── adapter
        ├── manifest.json
        ├── user_parameters.schema.json
        ├── impl.py
        └── runner.py
```

`adapter` is an executable launcher supporting:

```text
adapter manifest
adapter invoke
```

`manifest.json` is the identity source. It declares Adapter API version, kind,
name, implementation version, operations, canonical Schema versions, and an
optional versioned user-configuration contract.

## 3. Use the public SDK and canonical objects

External Adapters may import the public Adapter SDK. They must not import Core
internals or another Adapter. Exchange only the canonical request and response
objects defined by the protocol.

The invocation is one JSON request on stdin and one JSON response on stdout:

```json
{
  "api_version": "1.0",
  "request_id": "req-example",
  "operation": "probe",
  "input": {},
  "context": {}
}
```

Write diagnostics to stderr. Any extra stdout text breaks the protocol.

## 4. Declare user parameters

Adapter-owned parameters belong in `user_parameters.schema.json` and are
referenced by `manifest.user_config`. Avoid a second hidden configuration DSL
inside implementation code. Machine values still belong in System, model
loading facts in Model, and one-run choices in Evaluation.

## 5. Return plans, not owned processes

Backend and Evaluator Adapters return canonical process and preflight plans.
Core owns the process lifecycle, readiness wait, logs, timeout, and cleanup.
Dataset and Binding Adapters return Dataset and Framework Task artifacts rather
than calling the evaluator directly.

## 6. Validate the root

```bash
eval-manager adapter-check /absolute/path/to/my-adapters
MODEL_EVAL_ADAPTER_PATHS=/absolute/path/to/my-adapters \
  eval-manager adapters
```

Then test every declared operation with valid and invalid requests. Verify
Schema errors, timeouts, non-zero exits, stderr diagnostics, capability
mismatches, and cleanup behavior.

## 7. Package as an external plugin

Expose each Adapter directory through the package entry-point group:

```toml
[project.entry-points."model_evaluation.adapters"]
"evaluator.example" = "my_package.adapters.evaluator.example"
"binding.example" = "my_package.adapters.binding.example"
```

After wheel installation, run `eval-manager adapters` and `adapter-check` on the
installed root as part of package CI. Duplicate identities are rejected rather
than silently shadowed.

## 8. State validation honestly

Use distinct labels:

- contract-tested: manifest, Schema, RPC, and planning tests pass;
- smoke E2E: a small real integration run succeeds;
- full E2E: the declared full benchmark and expected sample count succeed on
  the named hardware/framework stack.

Do not infer production readiness for every matrix combination from one passing
Adapter contract.

## Review checklist

- One clear Adapter responsibility.
- Executable launcher and valid manifest.
- User parameters have their own JSON Schema.
- No imports between Adapters and no Core-internal imports.
- No secrets printed or persisted in public results.
- Structured failures and bounded operations.
- Contract tests plus the appropriate real smoke/full validation record.
- Documentation links to the upstream project and states validation level.
