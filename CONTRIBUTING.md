# Contributing

Contributions should preserve the project boundary: this repository coordinates
existing hardware, runtimes, inference backends, datasets and evaluators. It is
not an inference engine, benchmark implementation, experiment tracker or
tamper-proof evidence system.

## Development flow

1. Create a focused branch and keep machine-private paths, credentials, model
   weights, caches and result directories out of the change.
2. Add or update JSON Schema before changing a public object.
3. Put vendor/framework behavior in an Adapter, not in Core conditionals.
4. Add tests for the success path and at least one actionable failure path.
5. Run:

```bash
python -m pip install -r requirements.txt
python ./eval-manager schema-check
python tests/static_contract_check.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/build_release.py
```

For a third-party Adapter, prefer a separate Python distribution with the
`model_evaluation.adapters` entry point. Use `eval-manager adapter-check <root>`
before publishing it. New hardware/backend compatibility claims need a real
target-machine smoke and the intended benchmark; contract tests alone are not
sufficient.

Keep code readable: one responsibility per helper, no compressed multi-statement
control flow, bounded subprocess/network operations, and explicit cleanup.

## Public documentation and examples

Describe integration capabilities, validation levels, environment requirements,
and reproducible configuration examples. Keep internal evaluation campaign
details private: completed or successful run/model totals, per-platform counts,
conversion totals, pending or failed model lists, download inventories,
operational timelines, and unpublished scores. Redact these details from prose,
tables, screenshots, SVGs, attachments, and release descriptions before upload.

Software versions, device requirements, benchmark protocol parameters, and
synthetic examples may remain when needed to explain usage. A real result
example requires explicit approval for publication and a review of its metadata;
approval for an example does not authorize publishing campaign-wide statistics.
Review the content itself before publishing; a path/secret scan alone cannot
identify sensitive operational information.
