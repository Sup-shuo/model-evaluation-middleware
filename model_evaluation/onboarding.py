from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_evaluation.core.errors import ConfigError


@dataclass(frozen=True)
class HardwareTemplate:
    device: str
    runtime: str
    runtime_family: str
    runtime_root: str | None


HARDWARE_TEMPLATES = {
    "nvidia": HardwareTemplate("nvidia", "cuda", "cuda", "/usr/local/cuda"),
    "mlu": HardwareTemplate("mlu", "neuware", "neuware", "/usr/local/neuware"),
    "amd": HardwareTemplate("amd", "rocm", "rocm", "/opt/rocm"),
    "ascend": HardwareTemplate(
        "ascend", "cann", "cann", "/usr/local/Ascend/ascend-toolkit/latest"
    ),
    "cpu": HardwareTemplate("cpu", "cpu", "cpu", None),
}


def _system_yaml(hardware: str) -> str:
    selected = HARDWARE_TEMPLATES[hardware]
    runtime_root = f"\n        root: {selected.runtime_root}" if selected.runtime_root else ""
    devices = "\n      devices: [0]" if hardware != "cpu" else ""
    return f'''schema_version: "1.2"

system:
  name: {hardware}-local

metadata:
  timezone: Asia/Shanghai

profiles:
  defaults:
    hardware: local
    backend: vllm
    evaluator: lm_eval

  environment:
    controller:
      type: current

  hardware:
    local:
      type: {selected.device}{devices}
      runtime:
        type: {selected.runtime}{runtime_root}

  backend:
    vllm:
      type: vllm
      executable: vllm
      compatibility:
        runtime_families: [{selected.runtime_family}]
      environment: controller

  evaluator:
    lm_eval:
      type: lm_eval
      root: /REPLACE_WITH_LM_EVALUATION_HARNESS
      environment: controller

models:
  root: /REPLACE_WITH_MODEL_ROOT

paths:
  cache: /REPLACE_WITH_CACHE_ROOT
  results: results
'''


_MODEL_YAML = '''schema_version: "1.0"

id: example-model
label: Example local model

source:
  type: local
  ref: REPLACE_WITH_MODEL_RELATIVE_PATH

backends:
  vllm:
    max_model_len: 4096

provenance:
  policy: migration
'''


_EVALUATION_YAML = '''schema_version: "1.2"

models:
  - example-model

benchmarks:
  - bbh

backend:
  seed: 1234
  pythonhashseed: 1234

evaluator:
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


def initialize_project(target: str | Path, *, hardware: str) -> list[Path]:
    if hardware not in HARDWARE_TEMPLATES:
        raise ConfigError(f"unsupported onboarding hardware template: {hardware}")
    root = Path(target).expanduser().resolve()
    planned = {
        root / "config" / "system.yaml": _system_yaml(hardware),
        root / "config" / "evaluation.yaml": _EVALUATION_YAML,
        root / "config" / "models" / "example.yaml": _MODEL_YAML,
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
