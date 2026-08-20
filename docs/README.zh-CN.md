# Model Evaluation Middleware

[English](../README.md)

一个面向工程团队的模型评测胶水层：把硬件、运行环境、推理 Backend、模型、
数据集与评测框架连接成可记录、可复现、可迁移的执行链，并交付统一结果产品。

![模型评测胶水层架构](assets/architecture.zh-CN.svg)

## 10 秒建立感知：不需要 GPU/NPU

安装 Controller 依赖后，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
./eval-manager demo
```

`demo` 使用 CPU Runtime、回环 Reference Backend、`dataset/virtual`、
`binding/reference_eval` 和 `evaluator/reference_eval`，但仍经过正式的配置解析、
Matrix、进程管理、结果发布和一致性检查。它不下载模型、不访问外网，也不调用实体
GPU/NPU，通常在 10 秒内返回：

```json
{
  "demo": "reference",
  "ok": true,
  "report": {
    "benchmark": "mock_demo",
    "cleanup": "clean",
    "framework": "reference_eval",
    "model": "mock-model",
    "outcome": "success",
    "summary": {
      "contract_ok": {"value": 1}
    }
  }
}
```

它生成的仍是正式结果目录。`contract_ok=1` 只表示胶水链路跑通，**不是模型能力
分数**，也不代表任何硬件或真实评测框架已经通过。

## 这个项目负责什么

| 负责 | 不负责 |
|---|---|
| 解析 System、Model、Evaluation | 下载或训练模型 |
| 发现设备、Runtime 与执行环境 | 重新实现推理引擎 |
| 启动或连接 Backend，并验证实际能力 | 重新实现 benchmark |
| 把 Dataset 绑定到 Evaluator | 分布式集群调度 |
| 保存配置、版本、完整指标和原始输出 | 模型治理、实验追踪平台 |
| 在失败后清理自己启动的进程 | 取证、防篡改或可信证明 |

本项目的“可复现”含义是记录实际生效配置和可观察运行版本，帮助在另一台机器上重建
同一评测；它不承诺跨硬件 bit-level 一致，也不把结果包装成防篡改证据。

## 当前验证范围

| 等级 | 当前范围 | 含义 |
|---|---|---|
| 实机完整 E2E | NVIDIA A100/CUDA、Cambricon MLU/Neuware；vLLM + lm-eval + BBH | 24 个任务、5761 条样本、结果发布和清理通过 |
| 实机 smoke E2E | MetaX C500/MACA；vLLM-MetaX + lm-eval + BBH | 单卡服务与 24 个子任务 smoke 通过，不是完整精度评测 |
| Mock E2E | CPU + Reference Backend/Evaluator + virtual Dataset | 软件执行与结果产品链路通过 |
| Contract-tested | AMD/ROCm、Ascend/CANN、Ollama、llama.cpp、generic OpenAI 等 | Manifest、Schema、RPC 和计划行为通过，不等于生产实机验收 |

因此当前状态是“协议覆盖较广、实机覆盖集中”。目录里存在某个 Adapter，不代表该组合
已经完成生产验证。准确声明见[兼容性矩阵](compatibility.md)和
[脱敏实机记录](validation/)。

## 安装与第一次真实评测

要求 Python 3.10 或更高版本。源码方式：

```bash
python -m pip install -r requirements.txt
./eval-manager schema-check
./eval-manager adapters
```

安装 wheel 后使用同名 console script：

```bash
eval-manager schema-check
```

创建一个不会覆盖已有文件的最小工程：

```bash
mkdir my-evaluation
cd my-evaluation
eval-manager init . --hardware nvidia
```

`--hardware` 也支持 `metax`、`mlu`、`amd`、`ascend` 和 `cpu`。填写生成文件中的
`REPLACE_WITH_*` 后，先检查，再运行：

```bash
eval-manager check \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml

eval-manager run \
  --system-config config/system.yaml \
  --evaluation-config config/evaluation.yaml
```

`check` 组合配置验证、Doctor、计划预览和只读资源检查，不启动模型服务。需要逐步排错
时可分别执行 `validate`、`doctor`、`plan` 和 `explain`。

## 三类配置

| 配置 | 回答的问题 | 不应包含 |
|---|---|---|
| System | 这台机器有什么、在哪里、使用哪些环境 | 模型实验身份、benchmark 组合 |
| Model | 这个模型是什么、针对某 Backend 如何加载 | 设备号、机器路径、显存比例 |
| Evaluation | 这次选择哪些模型与 benchmark、临时覆盖什么 | 长期模型定义、驱动安装细节 |

典型目录：

```text
config/
├── systems/                 # NVIDIA、MLU、MetaX 等机器配置
├── models/                  # 一模型一文件，可按模型族/来源方分目录
├── evaluations/             # smoke、完整评测和一次性实验选择
├── system.yaml              # init 使用的通用模板
└── evaluation.yaml          # init 使用的通用模板
```

Model 和 Evaluation 可以在不同机器保持字节不变，只替换 System：

```bash
./eval-manager check --system-config mlu --evaluation-config smoke_bbh_08b
./eval-manager check --system-config nvidia --evaluation-config smoke_bbh_08b
```

机器路径、设备、Runtime、Backend/Evaluator 环境和容量参数属于 System。模型来源、
架构、量化、上下文与按 Backend 命名的加载参数属于 Model。种子、样本限制和本次选择
属于 Evaluation。完整字段、覆盖优先级、缓存与复现规则见
[配置指南](configuration.md)。

`model_evaluation/presets/` 是随包发布的内部规范化预设，不是第二套用户配置；普通用户
只维护根目录 `config/`。`model_evaluation/examples/mock/` 只为安装态 `demo` 提供自包含
示例。

## 常用工作流

```bash
# 无硬件演示，直接输出最终 JSON 报告
./eval-manager demo

# 完整的运行前检查；自动化使用 --format json
./eval-manager check --system-config mlu --evaluation-config smoke_bbh_08b

# 解释为什么当前组合可以运行或被阻止
./eval-manager explain --system-config mlu --evaluation-config smoke_bbh_08b

# 生成计划或执行评测
./eval-manager plan --system-config mlu --evaluation-config smoke_bbh_08b -o /tmp/plan.json
./eval-manager run  --system-config mlu --evaluation-config smoke_bbh_08b

# 验证和查看最终结果
./eval-manager result-check results/<run-id>
./eval-manager inspect results/<run-id>
./eval-manager inspect results/<run-id> --format json
```

其他入口：

- `init`：生成不覆盖现有文件的工程骨架；
- `schema-check` / `adapters`：查看 Core Schema 与已发现 Adapter；
- `adapter-check`：在安装前独立检查一个外部 Adapter root；
- `environment-snapshot`：可选导出 Controller Python 环境；
- `matrix-export`：把已保存 Matrix plan 分片给外部调度器；
- `run-plan` / `run-matrix-plan`：执行已保存计划或恢复批次。

Core 保持单机、串行 Matrix 与资源锁定位。大规模任务应由 Slurm、Kubernetes、Ray 或
内部调度器消费导出的 child plans，而不是让这个项目演变为分布式调度系统。

模型格式转换不属于评测自动流程。需要把目标 Backend 不支持的 checkpoint 生成独立
派生物时，由使用者显式运行项目一级工具：

```bash
python tools/model_convert.py inspect /data/OWNER/MODEL
python tools/model_convert.py routes
python tools/model_convert.py convert --help
```

它不会由 `eval-manager` 自动调用，也不会覆盖原模型或已有输出。具体路线和派生 Model
登记规则见[配置指南](configuration.md)。

## 结果产品

默认运行名：

```text
<platform>_<model-id>_<backend>_<benchmark-id>_YYMMDD-HHMM
```

默认结果位于项目一级 `results/`：

```text
results/<run-id>/
├── result.json              # 运行身份和标准化 summary
├── metrics.json             # summary、group 与每个 task 的指标
├── terminal.json            # 最终状态、本地时间和清理结果
├── failure.json             # 仅失败时存在
├── raw/                     # 完整框架原始输出
├── samples/                 # 仅显式启用且框架实际产出时存在
├── config/                  # 实际配置与可观察运行版本
└── logs/                    # Backend / Evaluator 日志
```

四个顶层 JSON 各有独立 Schema。`result-check` 和 `inspect` 会检查 Schema、跨文件身份与
指标一致性、成功/失败规则，以及公开产物路径约束。它们不做密码学证明。

Python 程序可以直接消费同一协议：

```python
from model_evaluation.results import load_run

run = load_run("results/<run-id>")
summary = run.metrics.summary()
tasks = run.metrics.tasks()
runtime = run.runtime()
artifacts = run.artifacts()
```

协议细节见[结果产品协议](result-product.md)。需要终端截图或报告视图时：

```bash
python scripts/print_result.py results/<run-id> \
  --text results/<run-id>/result-summary.txt \
  --svg results/<run-id>/result-summary.svg
```

TXT/SVG 只投影已有结果，不重算分数，也不是证明文件。

## Adapter 扩展

七类 Adapter 分别处理 Device、Runtime、Environment、Backend、Dataset、Binding 和
Evaluator。每个 Adapter 是一个带 `manifest.json` 的 JSON-over-stdio 子进程；Core
不 import 厂商 SDK 或评测框架。

第三方 Adapter 可以通过 Python entry point 自动发现，无需提交到主仓库：

```toml
[project.entry-points."model_evaluation.adapters"]
"backend.my_engine" = "my_eval_plugin.adapters.backend.my_engine"
```

开发态还可通过 `MODEL_EVAL_ADAPTER_PATHS` 指定一个或多个绝对 Adapter root。内置、
开发目录与已安装 entry point 出现同 kind/name 时会拒绝启动，不允许静默覆盖。

扩展对象、RPC、失败语义和检查清单见
[架构与 Adapter 协议](../ARCHITECTURE_AND_ADAPTER_PROTOCOL.md)。

## 项目结构

```text
model-evaluation-middleware/
├── model_evaluation/        # 单一可安装 Python 包
│   ├── core/                # 配置、规划、执行、资源和结果整理
│   ├── adapters/            # 内置 Adapter
│   ├── sdk/                 # 外部 Adapter 的稳定 SDK
│   ├── schemas/             # 公共对象与用户配置 Schema
│   ├── presets/             # 内部规范化预设
│   ├── examples/mock/       # wheel 内置的无硬件演示
│   └── commands/            # CLI 命令层
├── config/                  # 用户维护的 System、Model、Evaluation
├── tests/                   # 单元、集成与静态边界测试
├── scripts/                 # 项目发布与结果视图自动化脚本
├── tools/                   # 独立的手动检查、转换和校验工具
├── results/                 # 运行产物；不进入 wheel/发布 ZIP
└── eval-manager             # 源码树入口
```

这是单包应用仓库，不保留只有一个子包的 `src/` 包装层。源码内部按真实职责分层，不为
减少目录数量而把 Core、Adapter 和协议混在一起。

## 文档导航

- [配置指南](configuration.md)：System / Model / Evaluation、缓存与复现；
- [结果产品协议](result-product.md)：最终 JSON 与 Python 消费接口；
- [兼容性矩阵](compatibility.md)：实机、Mock 与 contract-tested 边界；
- [NVIDIA A100](validation/nvidia-a100.md)、
  [Cambricon MLU](validation/cambricon-mlu.md)、
  [MetaX C500](validation/metax-c500.md)：脱敏实机记录；
- [架构与 Adapter 协议](../ARCHITECTURE_AND_ADAPTER_PROTOCOL.md)；
- [贡献指南](../CONTRIBUTING.md)、[安全边界](../SECURITY.md)、[版本变化](../CHANGELOG.md)。

开发验证：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tests/static_contract_check.py
python3 scripts/build_release.py
```
