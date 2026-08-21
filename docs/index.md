# Documentation

Use the shortest path that matches your goal. The pages are intentionally
layered: the installation guide produces a working run, configuration pages
explain user-owned YAML, component pages describe integrations, and Adapter
pages define extension work.

## Use the project

1. [Installation and first evaluation](installation.md)
2. [Configuration overview](configuration.md)
3. [System configuration](configuration/system.md)
4. [Model configuration](configuration/model.md)
5. [Evaluation configuration](configuration/evaluation.md)
6. [Environment isolation](configuration/environments.md)
7. [Result product protocol](result-product.md)

## Choose an integration

- [Components and integrations](components/index.md)
- [Hardware and runtimes](components/hardware.md)
- [Inference Backends](components/inference-backends.md)
- [Evaluation frameworks](components/evaluators.md)
- [Datasets and Bindings](components/datasets.md)
- [Models and model material](models/index.md)
- [Compatibility matrix](compatibility.md)

## Extend or audit the project

- [Adapter inventory and discovery](adapters/index.md)
- [Adding an Adapter](adapters/adding-an-adapter.md)
- [Architecture and Adapter protocol](../ARCHITECTURE_AND_ADAPTER_PROTOCOL.md)
- [Final result product protocol](result-product.md)
- [Contributing](../CONTRIBUTING.md)
- [Security boundary](../SECURITY.md)

## Real-machine records

- [NVIDIA A100](validation/nvidia-a100.md)
- [Cambricon MLU](validation/cambricon-mlu.md)
- [MetaX C500](validation/metax-c500.md)

These records show exercised stacks. They do not expand the validation status
of other built-in Adapters or every model accepted by an upstream engine.
