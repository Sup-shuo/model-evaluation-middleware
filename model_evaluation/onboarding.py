from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model_evaluation.core.errors import ConfigError


@dataclass(frozen=True)
class HardwareTemplate:
    device: str
    runtime: str
    runtime_family: str
    runtime_root: str | None
    runtime_parameters: tuple[tuple[str, str], ...] = ()
    validation_status: str = "Adapter contract-tested; real-machine E2E pending"


HARDWARE_TEMPLATES = {
    "nvidia": HardwareTemplate(
        "nvidia", "cuda", "cuda", "/usr/local/cuda",
        validation_status="Real full E2E: NVIDIA A100/CUDA",
    ),
    "mlu": HardwareTemplate(
        "mlu", "neuware", "neuware", "/usr/local/neuware",
        validation_status="Real full E2E: Cambricon MLU/Neuware",
    ),
    "metax": HardwareTemplate(
        "metax",
        "maca",
        "maca",
        "/opt/maca",
        (("driver_root", "/opt/mxdriver"), ("driver_tool", "/usr/bin/mx-smi")),
        validation_status="Real smoke E2E: MetaX C500/MACA",
    ),
    "amd": HardwareTemplate("amd", "rocm", "rocm", "/opt/rocm"),
    "ascend": HardwareTemplate(
        "ascend", "cann", "cann", "/usr/local/Ascend/ascend-toolkit/latest"
    ),
    "cpu": HardwareTemplate(
        "cpu", "cpu", "cpu", None,
        validation_status="Reference Mock E2E; vLLM template is contract-tested",
    ),
}


def _yaml_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _system_yaml(
    hardware: str,
    *,
    models_root: str,
    evaluator_root: str,
    cache_root: str,
    results_root: str,
    backend_executable: str,
    devices: tuple[int | str, ...],
) -> str:
    selected = HARDWARE_TEMPLATES[hardware]
    runtime_root = f"\n        root: {selected.runtime_root}" if selected.runtime_root else ""
    runtime_parameters = ""
    if selected.runtime_parameters:
        values = "\n".join(
            f"          {name}: {value}" for name, value in selected.runtime_parameters
        )
        runtime_parameters = f"\n        parameters:\n{values}"
    devices_yaml = (
        f"\n      devices: {_yaml_scalar(list(devices))}" if devices else ""
    )
    return f'''schema_version: "1.3"

system:
  name: {hardware}-local

metadata:
  timezone: Asia/Shanghai

profiles:
  defaults:
    hardware: local

  environment:
    controller:
      type: current

  hardware:
    local:
      type: {selected.device}{devices_yaml}
      runtime:
        type: {selected.runtime}{runtime_root}{runtime_parameters}

  backend:
    vllm:
      type: vllm
      executable: {_yaml_scalar(backend_executable)}
      compatibility:
        runtime_families: [{selected.runtime_family}]
      environment: controller

  evaluator:
    lm_eval:
      type: lm_eval
      root: {_yaml_scalar(evaluator_root)}
      environment: controller

models:
  root: {_yaml_scalar(models_root)}

paths:
  cache: {_yaml_scalar(cache_root)}
  results: {_yaml_scalar(results_root)}
'''


def _model_yaml(model_id: str, model_ref: str) -> str:
    return f'''schema_version: "1.0"

id: {_yaml_scalar(model_id)}
label: {_yaml_scalar(model_id)}

source:
  type: local
  ref: {_yaml_scalar(model_ref)}

backends:
  vllm:
    max_model_len: 4096

provenance:
  policy: migration
'''


def _evaluation_yaml(model_id: str, benchmark: str) -> str:
    return f'''schema_version: "1.3"

models:
  - {_yaml_scalar(model_id)}

benchmarks:
  - {_yaml_scalar(benchmark)}

backend:
  profile: vllm
  parameters:
    seed: 1234
    pythonhashseed: 1234

evaluator:
  profile: lm_eval
  parameters:
    batch_size: 1
    random_seed: 0
    numpy_random_seed: 1234
    torch_random_seed: 1234
    fewshot_random_seed: 1234
    request_seed: 1234
    pythonhashseed: 1234

offline: true

execution:
  mode: serial
  continue_on_error: false
'''


_GITIGNORE = '''results/
cache/
runtime/
build/
dist/
*.egg-info/
__pycache__/
*.pyc
.pytest_cache/
'''


def initialize_project(
    target: str | Path,
    *,
    hardware: str,
    model_id: str = "example-model",
    model_ref: str = "REPLACE_WITH_MODEL_RELATIVE_PATH",
    models_root: str = "/REPLACE_WITH_MODEL_ROOT",
    evaluator_root: str = "/REPLACE_WITH_LM_EVALUATION_HARNESS",
    cache_root: str = "/REPLACE_WITH_CACHE_ROOT",
    results_root: str = "results",
    benchmark: str = "bbh",
    backend_executable: str = "vllm",
    devices: tuple[int | str, ...] | None = None,
) -> list[Path]:
    if hardware not in HARDWARE_TEMPLATES:
        raise ConfigError(f"unsupported onboarding hardware template: {hardware}")
    selected_devices = tuple(
        devices if devices is not None else (() if hardware == "cpu" else (0,))
    )
    if hardware == "cpu" and selected_devices:
        raise ConfigError("CPU onboarding template does not accept accelerator device IDs")
    if hardware != "cpu" and not selected_devices:
        raise ConfigError("accelerator onboarding template requires at least one device ID")
    if len(set(selected_devices)) != len(selected_devices):
        raise ConfigError("onboarding device IDs must be unique")
    required = {
        "model_id": model_id,
        "model_ref": model_ref,
        "models_root": models_root,
        "evaluator_root": evaluator_root,
        "cache_root": cache_root,
        "results_root": results_root,
        "benchmark": benchmark,
        "backend_executable": backend_executable,
    }
    empty = sorted(name for name, value in required.items() if not str(value).strip())
    if empty:
        raise ConfigError(f"init values must not be empty: {', '.join(empty)}")
    root = Path(target).expanduser().resolve()
    planned = {
        root / "config" / "system.yaml": _system_yaml(
            hardware,
            models_root=models_root,
            evaluator_root=evaluator_root,
            cache_root=cache_root,
            results_root=results_root,
            backend_executable=backend_executable,
            devices=selected_devices,
        ),
        root / "config" / "evaluation.yaml": _evaluation_yaml(model_id, benchmark),
        root / "config" / "models" / "example.yaml": _model_yaml(model_id, model_ref),
        root / ".gitignore": _GITIGNORE,
    }
    conflicts = [path for path in planned if path.exists()]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise ConfigError(f"init refuses to overwrite existing files: {joined}")

    for path, text in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return sorted(planned)
