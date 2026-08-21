# Sanitized result-product example

`cpu_example-model_reference_example-benchmark_260101-1200/` is a synthetic,
schema-valid success result. It demonstrates the public directory and JSON
contracts without containing a real benchmark score, user name, host name,
network address, model inventory, or machine-specific checkout path.

From the repository root:

```bash
./eval-manager result-check \
  examples/result_example/cpu_example-model_reference_example-benchmark_260101-1200

./eval-manager inspect \
  examples/result_example/cpu_example-model_reference_example-benchmark_260101-1200
```

The committed TXT and SVG files are sanitized presentation projections. The
four top-level JSON documents remain the stable machine-readable protocol.
