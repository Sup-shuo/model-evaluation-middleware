# Configuration guide

用户只维护三类文档：System、Model 和 Evaluation。Core 将它们解析为随包发布的内部
Profiles/Specs；`model_evaluation/presets/` 不需要用户编辑。

## System

System 描述一台机器上的设备、Runtime、执行环境、Backend、Evaluator 和路径：

```yaml
schema_version: "1.2"
system:
  name: nvidia-vllm

metadata:
  timezone: Asia/Shanghai
  result_platform: nvidia

profiles:
  defaults:
    hardware: nvidia
    backend: vllm
    evaluator: lm_eval

  environment:
    vllm-env:
      type: conda
      profile: /opt/conda/envs/vllm
      executable: /opt/conda/bin/conda
    eval-env:
      type: venv
      profile: /opt/venvs/lm-eval

  hardware:
    nvidia:
      type: nvidia
      devices: [0]
      runtime:
        type: cuda
        root: /usr/local/cuda

  backend:
    vllm:
      type: vllm
      executable: vllm
      environment: vllm-env
      compatibility:
        runtime_families: [cuda]
      parameters:
        gpu_memory_utilization: 0.8
        max_num_seqs: 8
        num_concurrent: 8

  evaluator:
    lm_eval:
      type: lm_eval
      root: /opt/lm-evaluation-harness
      environment: eval-env

models:
  root: /data/models

paths:
  cache: cache
  results: results
```

关键规则：

- `hardware.devices` 是机器默认设备；`evaluation.resources.devices` 只用于单次覆盖。
- Backend 与 Evaluator 可以使用不同 Conda/venv；`current` 表示 Controller Python。
- managed Backend 必须显式声明允许的 `runtime_families`。
- 显存比例、服务并发和设备选择属于 System，不属于 Model。
- `paths.cache` 与 `paths.results` 的相对路径以项目根解析且不能用 `..` 越界；绝对
  路径可用于机器共享存储。
- 模型根使用机器规范路径。符号链接会被解析，但统一真实路径可避免日志出现两套身份。

## Model Catalog

每个长期模型定义只写一次：

```yaml
schema_version: "1.0"
id: qwen35-08b-base
label: Qwen3.5 0.8B Base BF16

source:
  type: local
  ref: Qwen/Qwen3.5-0.8B-Base

architecture: qwen3_5
quantization: bf16
format: safetensors
context_length: 4096

backends:
  vllm:
    max_model_len: 4096
    trust_remote_code: true

provenance:
  policy: migration
```

`id` 是实验身份，`source.ref` 是来源或相对物化路径。可复用字段包括 revision、
tokenizer、architecture、quantization、format、context length、chat template、
`trust_remote_code` 和按 Backend 命名的加载参数。

新增配置优先使用 `backends.vllm`、`backends.llama_cpp` 等命名空间。这样切换 Backend
时只取当前实现对应的参数。`dtype` 若是某个 checkpoint 在某 Backend 上稳定运行的长期
要求，属于这个命名空间；设备、显存和并发仍属于 System。

`source.type: hf` 或 `registry` 只描述来源，不表示 Core 会下载模型。需要本地权重的
managed Backend 使用 System `models.root` 与 `source.ref` 定位已物化目录。

权重尚未下载时可以先登记身份，但不要根据仓库名称猜 architecture、quantization、
format 或 context。物化后以 `config.json`、Tokenizer 和权重索引为准。

若目标 Backend 不支持原量化格式，项目一级 `tools/model_convert.py` 提供人工转换入口。
它与 `eval-manager` 完全分离：`validate`、`doctor`、`check`、`plan` 和 `run` 都不会
自动转换模型。先只读检查实际 checkpoint 元数据与可选路线：

```bash
python tools/model_convert.py inspect /data/OWNER/MODEL
python tools/model_convert.py routes
```

转换必须由使用者显式选择路线、来源和新的输出目录。例如：

```bash
python tools/model_convert.py convert \
  --route compressed-tensors-to-bf16 \
  --source /data/OWNER/MODEL \
  --output /data/OWNER/MODEL-derived-bf16 \
  --source-ref OWNER/MODEL
```

可先加 `--dry-run` 查看将要调用的内部实现。工具根据 `config.json` 判断真实量化方法，
不会根据仓库名中的 `AWQ` 字样猜测，也不会把 compressed-tensors 直接改名成 AWQ。
转换实现写入同级临时目录，完成内置的结构、张量和加载检查后才原子发布，并拒绝覆盖
已有目录；失败时保留临时目录供人工检查，不删除来源。

派生物必须使用新的 Model ID，不改写原目录；只有完成键/形状、逐张量等价和目标
Backend smoke 后，才能将其列为可用物料。多模态模型的纯文本派生物必须明确标记视觉
能力已经移除。`tools/model_conversion/` 保存统一入口使用的转换与校验实现；
`tools/model_convert.py` 是唯一公开入口，不是自动工作流。

## Evaluation

Evaluation 只保存一次实验的选择与临时差异：

```yaml
schema_version: "1.2"

models:
  - qwen35-08b-base

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
```

临时覆盖引用 catalog ID：

```yaml
models:
  - id: qwen35-08b-base
    overrides:
      backend:
        max_model_len: 8192
```

覆盖不会修改 Catalog。来源、架构、量化、格式和实验 ID 属于模型身份；改变它们时应
新建 Model，而不是把结果挂在旧 ID 下。Backend/Evaluator 字符串表示选择 System
profile，对象表示参数覆盖。

## Dataset and cache

`paths.cache` 是物化根，不是全局数据策略。Dataset/Evaluator Adapter 决定其内部布局
和离线环境变量。例如 BBH 链路是：

```text
BenchmarkSpec(bbh)
  → DatasetProvider(bbh_local)
  → FrameworkBinding(lm_eval.bbh)
  → Evaluator(lm_eval)
```

外部资产检查由 Adapter 决定：

- `basic`：默认检查存在、可读和结构正确；
- `strict`：用户显式选择时执行 Adapter 自己的摘要或更强校验。

Core 不要求模型和数据全部预登记 SHA，也不把结果声明成可信证据。

## Reproducibility boundary

评测应显式固定参与采样、请求顺序和服务执行的种子。保存的
`config/runtime_versions.json` 记录可观察到的 Adapter、Python 环境、Runtime/Driver、
Backend 和 Evaluator 版本，但它不是完整 lock file。

建议：

1. Model 使用可定位 revision，或记录本地权重来源；
2. 固定评测框架 revision 和 task 定义；
3. 固定所有 seeds、batch/concurrency、few-shot 与 generation 参数；
4. 保留完整运行目录，而不是只复制总分；
5. 跨硬件以指标容差比较，不要求 bitwise 一致。

Controller Python 可以单独导出：

```bash
eval-manager environment-snapshot \
  -o controller-environment.json \
  --requirements-lock controller-requirements.lock
```

仓库的 `requirements-strict.txt` 是发布验证过的 Controller 基线。Backend 与 Evaluator
若使用独立环境，应分别导出自己的 Conda/venv 快照，不要把三者错误合并成一份全局锁。
