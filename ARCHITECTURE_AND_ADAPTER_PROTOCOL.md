# 架构与 Adapter 协议

本文描述稳定的层边界、交换对象、执行不变量和扩展接口。具体安装、配置示例与命令见 README。

## 1. Scope 与 Non-goals

本项目是模型评测工程中间层。它把机器硬件、运行环境、推理 Backend、模型、Dataset、Framework Binding 和 Evaluator 组合成可迁移的执行链，并把不同评测框架的输出整理成统一结果产品。

项目负责：

- 解析 System、Model Catalog 和 Evaluation 配置；
- 探测硬件、Runtime 和执行 Environment；
- 规划并管理 Backend/Evaluator 进程；
- 在模型启动前检查依赖、任务和缓存；
- 执行 Dataset → Binding → Evaluator 链路；
- 输出 `result.json`、逐任务 `metrics.json`、框架原始结果、可选 samples、实际配置和日志；
- 固定影响随机性的运行参数，记录当前能够探测的关键复现配置与版本。

项目不负责：

- 实现新的推理引擎、模型、Dataset 或 benchmark；
- 充当调度平台、训练平台、模型注册中心或通用实验追踪数据库；
- 提供密码学防篡改、长期取证、合规审计或远端资产存证；
- 强制所有外部模型和数据预登记 SHA；
- 通过 Core 中的厂商/框架分支猜测环境兼容性。

`runtime_versions.json` 只是随结果保存的实验条件，不是锁文件、资产证明或审计证据；
它不会扫描全部已安装包，也不扩大上述边界。

运行时内部会短暂保存计划、preflight 和进程状态，以支持执行与失败排错；成功后删除这些临时状态。它们不是独立的审计产品。

## 2. 分层模型

```text
User Configuration
  System + Model Catalog + Evaluation
                    │
                    ▼
UserConfigResolver / Schema validation
                    │
                    ▼
Canonical Specs
  Platform + Deployment + Model + Benchmark + Evaluation
                    │
                    ▼
Planner
  compatibility + resource claims + ordered stages
                    │
                    ▼
Orchestrator
  Device → Runtime → Environment → Backend
  Dataset → Binding → Evaluator
                    │
                    ▼
Result Product
  result + metrics + raw + samples + config + logs
```

三类用户配置必须保持正交：

- System：这台机器有什么、在哪里、可选哪些 profile；
- Model：这个长期模型是什么、如何加载；
- Evaluation：本次选择哪些模型和 benchmark，以及临时参数覆盖。

解析器把用户层配置转换为标准 Spec。Planner 和 Orchestrator 只消费标准 Spec，不解释用户 YAML 的便捷写法。

## 3. Spec 边界

### PlatformProfile

描述本地执行事实的选择方式：DeviceAdapter、RuntimeAdapter、Backend Environment 和 Evaluation Environment。Environment 是执行角色属性；Backend 和 Evaluator 可以在不同 Python 环境。

纯 external/attached Backend 的 Platform 可以是 evaluation-only，不应伪造本地 Device/Runtime。

### DeploymentProfile

描述如何获得模型服务：

- `managed`：Core 启动并拥有 Backend 进程；
- `attached`：服务已存在于当前机器/环境；
- `external`：服务由外部系统管理。

managed Deployment 必须显式声明 `compatibility.runtime_families`。兼容性不是 Backend 参数，Core 也不按 Adapter 名称猜测。

### ModelSpec

描述模型来源与实验身份。`id`/`experiment_id` 是结果身份，`source.ref` 是权重来源；二者不能混为一谈。通用字段可以包含 revision、tokenizer、architecture、quantization、format、context length、chat template 和 trust-remote-code。

Backend 特有参数由 Backend Adapter 的用户参数 Schema 接受，并进入 Deployment override，不进入 Core 特殊字段。

当目标 Backend 无法读取现有权重格式时，派生模型必须使用新的 ModelSpec 身份和
新的 `source.ref`，不得覆盖或冒充原模型。转换工具属于离线运维工具，不进入 Core
或 Adapter 的在线执行路径；它应保留原目录、记录转换方法，并在发布派生目录前完成
明确的键/形状与数值等价检查。若从多模态模型只提取语言塔，派生身份必须声明已移除
视觉能力，文本 benchmark 结果也不能被解释成原模型的多模态能力结论。

### BenchmarkSpec

描述 DatasetProvider、revision、评测协议、inference 类型、few-shot 和标准指标。Benchmark 不绑定具体机器。

### EvaluationProfile

选择 EvaluatorAdapter 和默认 FrameworkBinding，配置框架根、revision、batch/concurrency、随机种子、timeout 和 metric mapping。

### RunSpec / MatrixSpec

RunSpec 引用五个轴：Model、Platform、Deployment、Benchmark、Evaluation。MatrixSpec 对这些轴做有界笛卡尔积、排除规则和单模型 override。展开必须确定、有限且去重。

## 4. Adapter 生态

每个 Adapter 位于独立目录，包含：

```text
adapter                 # 可执行 JSON-over-stdio 入口
manifest.json           # 唯一身份与能力声明
impl.py / helpers       # 实现
user_parameters.schema.json  # 可选，Adapter 自有用户参数
```

`manifest.json` 是 identity 单一来源，至少声明：

- `adapter_api`；
- `kind`、`name`、`version`；
- `operations`；
- 所交换 canonical object 的 schema versions；
- `implementation`；
- 可选的版本化 `user_config`。

`implementation` 只描述载体，不承载隐藏配置 DSL。用户参数策略必须位于 `manifest.user_config`，其参数空间由 Adapter 自己的 JSON Schema 约束。

### Adapter 类别

| kind | 职责 | 核心 operations |
|---|---|---|
| Device | 设备发现和可见性 | probe, visibility, snapshot |
| Runtime | 运行时事实和环境补丁 | probe, resolve_environment, snapshot |
| Environment | 解释器解析与进程包装 | resolve, wrap_process, snapshot |
| Backend | 服务需求、预检、启动、探活 | requirements, plan_preflight?, plan_start, probe_service, snapshot |
| Dataset | 定位、物化、验证数据 | resolve, prepare, verify, snapshot |
| Binding | Benchmark 到框架 task 的绑定 | requirements, build_task, protocol_fingerprint |
| Evaluator | 框架预检、执行计划、结果标准化 | requirements, plan_preflight?, plan_evaluate, normalize, snapshot |

Adapter 之间禁止相互 import。跨组件通信只通过 canonical objects；外部 Adapter 可使用公共 SDK，不需要 import Core。

## 5. Adapter RPC

入口支持：

```text
adapter manifest
adapter invoke
```

请求：

```json
{
  "api_version": "1.0",
  "request_id": "req-...",
  "operation": "probe",
  "input": {},
  "context": {}
}
```

成功响应：

```json
{
  "api_version": "1.0",
  "request_id": "req-...",
  "ok": true,
  "output": {},
  "warnings": []
}
```

失败响应包含结构化 `error.code/message/retryable/details`。stdout 只允许协议 JSON；诊断写 stderr。Core 对请求、响应、operation input 和 canonical output 分层校验，并对调用设置边界时间。

Adapter subprocess 使用净化后的 Controller 环境。Core 不把所有宿主环境变量、secret 或 `PYTHONPATH` 无条件传入插件。

## 6. 关键交换对象

### EnvPatch

声明 `set`、`unset`、`prepend_path`。Core 按明确顺序合并：

```text
Device → Runtime → role Environment → Process-local
```

冲突必须显式处理；Environment 负责最终解释器/可执行文件包装。`current` 指运行 Core 的解释器，不能从 Adapter 的 `PATH` 重新猜测。

### ProcessSpec

描述 argv、cwd、EnvPatch、secret references、stdin/stdout/stderr、timeout、readiness 和 metadata。Adapter 只能计划进程，不能拥有长期服务生命周期；进程所有权属于 Core。

### BackendPreflightPlan

由 Backend Adapter 返回结构化 probes，区分 dependency 与 model phase。每个 probe 都经过与实际 Backend 相同的 Device、Runtime、Environment 包装。

结构化 probe 同时检查退出码与 `PreflightProbeResult.status`；二者不一致视为失败。dependency 失败时不得进入 model phase。

### BackendStartPlan / ServiceDescriptor

StartPlan 描述 managed 进程、attach 信息、readiness、shutdown 和兼容旧 Adapter 的 dependency probe。Core 启动后通过 Adapter `probe_service` 得到 ServiceDescriptor。

ServiceDescriptor 是能力事实，不只是 URL。它应声明 model identity、ownership、协议端点、auth、tokenizer、context/并发限制和 CapabilitySet。

### CapabilitySet / RequirementSet

Backend/Environment 提供事实，Evaluator/Binding 声明需求，Core 做通用比较。Core 不写 `if backend == vllm` 或 `if evaluator == lm_eval` 的兼容分支。

### DatasetArtifact

描述 dataset id、revision、root、文件引用、物化类型和可选 metadata/fingerprint。

外部资产完整性由 Dataset Adapter 负责：

- `basic`：检查运行所需的存在性、可读性和结构；
- `strict`：按具体 Adapter 契约验证内容并提供 fingerprint。

Core 不要求所有 DatasetArtifact 一定带逐文件 SHA 或 fingerprint。revision 表示声明的来源身份，不能假装成内容摘要。

### FrameworkTaskArtifact

Binding 将 Benchmark + DatasetArtifact + Evaluation 转换为框架任务，返回 task id/root、生成文件、execution、metric contract、provenance 和 `protocol_fingerprint`。

Protocol fingerprint 必须来自实际影响任务语义的规范化输入，例如 benchmark protocol、dataset revision、integrity policy、framework revision 和生成任务文件；basic 数据策略不需要扫描数据内容。

### CanonicalResult

Evaluator 的 normalize 返回：

- run/model/benchmark/framework identity；
- 标准化 summary metrics；
- `breakdowns.summary/groups/tasks`；
- 原始框架结果引用；
- 可选 sample artifacts；
- 最少量实现 metadata。

`breakdowns` 是正式产品字段，不放在 metadata：

```json
{
  "summary": {
    "id": "leaderboard_bbh",
    "kind": "group",
    "metric_namespace": "canonical",
    "metrics": {"accuracy": {"value": 0.5}},
    "native_metrics": {"acc_norm,none": {"value": 0.5}}
  },
  "groups": {
    "leaderboard_bbh": {
      "metrics": {},
      "canonical_metrics": {},
      "subtasks": ["..."]
    }
  },
  "tasks": {
    "leaderboard_bbh_boolean_expressions": {
      "metrics": {},
      "canonical_metrics": {},
      "sample_count": {"original": 250, "effective": 1},
      "num_fewshot": 3,
      "version": "1.0",
      "config": {}
    }
  }
}
```

未定义的 stderr（例如框架输出 `N/A`）应省略，不能伪造为 0。逐样本文件只有在框架实际产出时才列入 `sample_artifacts`，新产品字段不要求 SHA。

## 7. 结果产品边界

默认单次运行目录是：

```text
results/<model>_<benchmark>_<Beijing short timestamp>/
├── result.json
├── metrics.json
├── raw/
├── samples/        # 可选
├── config/
├── logs/
├── terminal.json
└── failure.json    # 仅失败
```

### result.json

指标入口。包含本次运行身份、模型/benchmark/framework、summary metrics 和结果时间元数据。运行 outcome、起止时间与 cleanup 状态由同层的 `terminal.json` 表达；用户不需要理解 ExecutionPlan、Adapter manifest 或生命周期状态机。

### metrics.json

CanonicalResult `breakdowns` 的稳定投影：summary、groups、tasks。组任务必须完整保留每个子任务，而不是只交付一个平均分。字段适合人读和脚本消费，不携带不必要的实现快照。

### raw/

无损保存评测框架原始结果。lm-eval 原始 JSON 中的 results、groups、task configs、versions、n-samples、duration 等不得因标准化而丢失。

### samples/

仅在用户启用 `log_samples` 且框架产出逐样本文件时存在。目录保存框架提供的 prompt/response/choice/loglikelihood 等记录；中间层不凭空重建缺失样本。

### config/ 与 logs/

config 保存本次实际生效的用户配置、模型/评测参数和框架配置；logs 保存 Backend/Evaluator 可读日志。内部 canonical Specs 可以作为诊断信息保留，但不应占据用户结果入口。

### 内部临时状态

ExecutionPlan、preflight、Dataset/Task staging 和进程状态位于隐藏的 `.run/`。成功运行发布结果后删除 `.run/`；失败运行暂留它帮助定位问题。默认产品不生成 `evidence/` 或逐结果哈希清单，也不宣称密码学防篡改、合规追踪或长期取证。

Batch 产品应聚合 child `result.json` 和 `metrics.json`，同时保留 child run 路径。恢复执行所需的内部状态与面向用户的 batch summary 分离。

## 8. 执行阶段与不变量

典型阶段：

```text
PLATFORM_READY
DATA_READY
TASK_READY
EVALUATOR_PREFLIGHT
SERVICE_STARTING
SERVICE_READY
EVALUATING
NORMALIZING
CLEANING
```

关键顺序：

1. 规划时探测 Device/Runtime/Environment 与静态 requirements；
2. 执行前重新验证计划、Adapter 和运行事实；
3. Dataset prepare/verify；
4. Binding 生成任务并确认 fingerprint；
5. Evaluator dependency/task/cache preflight；
6. managed Backend preflight、启动与 readiness；
7. Service capability 与 Evaluator requirements 比较；
8. 再次验证 task 与 strict dataset；
9. 评测、normalize、产品结果落盘；
10. 无论成功失败都进入 bounded cleanup。

不变量：

- Evaluator task/cache preflight 必须早于昂贵的 Backend startup；
- planning facts 在 execution 使用前必须重新验证；
- attached/external Backend 不得被 Core 当作 owned process 终止；
- managed Backend 的端口声明与真实启动参数必须一致；
- Binding 返回的 fingerprint 必须与独立 `protocol_fingerprint` operation 一致；
- CanonicalResult identity 必须与正在执行的 run/model/benchmark/evaluator 一致；
- normalize 必须保留完整 raw result 并输出所有可识别 task/group 指标；
- cleanup 失败不得被成功结果掩盖。

## 9. 进程、资源与失败安全

Core 对 managed process 负责：

- 在创建子进程后立即建立 provisional ownership；
- 记录 PID、PGID、启动 identity 与 ownership record；
- 只在所有权可证明时向 owned process group 发信号；
- 优先 graceful shutdown，超时后才做有界强制终止；
- PGID/identity 模糊时不盲目杀进程，而是报告 cleanup incomplete；
- 中断、Backend 失败、Evaluator 失败和 normalize 失败都进入相同 finally cleanup。

资源管理至少覆盖全局 run lock、设备、端口和 dataset cache lock。声明资源与真实进程参数不一致属于执行错误。

失败产品应在根层提供 `failure.json`、`terminal.json` 和相关日志 tail/路径，让用户无需进入内部状态目录即可定位主要错误；`.run/` 仅作为补充调试材料保留。

## 10. Secret 与环境安全

- 持久化配置只能引用 `secret://...`，不得内联 token/password/key；
- SecretStore 在执行前解析，ProcessSpec 使用 `secret_env`；
- 计划、结果、日志摘要和结构化诊断必须走统一 redaction；
- Adapter subprocess 环境按 allowlist 构造，不继承任意宿主 secret；
- bearer 上游凭据通过 Core 控制的本地代理注入，不直接交给评测框架；
- 路径必须 resolve/confine，拒绝 workspace 越界和不安全 symlink；
- Adapter stdout 只承载协议，避免日志污染 JSON RPC。

安全目标是避免共享服务器上的误杀、secret 泄露和路径越界，不是构建通用沙箱或零信任取证系统。

## 11. Reproducibility 边界

可复现需要同时固定：

- Model source/revision、quantization、tokenizer/chat template；
- Benchmark protocol、dataset revision 与适用的 integrity policy；
- Framework root/revision 和 task binding fingerprint；
- Backend/Evaluator 参数与 generation/few-shot 配置；
- Python、NumPy、Torch、few-shot、request、Backend 与 `PYTHONHASHSEED`；
- 设备数量/identity、Runtime/Environment 版本和并发策略。

种子必须进入真正执行的命令/环境，不能只写在配置文件。lm-eval 的框架原始 config/version/n-samples 和中间层的 resolved config 一起帮助复现实验。

以下情况即使种子相同也可能不逐 bit 一致：不同硬件 kernel、不同 Runtime/Backend 版本、浮点归约顺序、并发请求调度、外部服务漂移、未固定数据/模型 revision。项目承诺把已知控制面显式化，不承诺跨任意硬件逐 token 完全一致。

## 12. 扩展规则

### 新硬件

增加 DeviceAdapter 和需要时的 RuntimeAdapter，返回通用 descriptors/capabilities/EnvPatch；不要修改 Core 硬件分支。System profile 选择它们。

### 新 Backend

实现 requirements、plan_start、probe_service、snapshot；推荐实现结构化 plan_preflight。用户参数、默认 mode、派生参数和 model-location policy 放入 versioned `manifest.user_config`。

### 新 Dataset

实现 resolve/prepare/verify/snapshot，明确 basic/strict 语义。Adapter 自己负责缓存布局与内容校验策略；不要把评测框架逻辑写进 DatasetProvider。

### 新 Binding

实现 requirements/build_task/protocol_fingerprint。生成文件必须位于 staging root；fingerprint 覆盖所有任务语义输入。Dataset 与 Evaluator 通过 Binding 解耦。

### 新 Evaluator

实现 requirements/plan_evaluate/normalize/snapshot；推荐 task/data preflight。normalize 必须提供 summary、breakdowns 和 raw result；有逐样本输出时列出 sample artifacts。

### 扩展检查清单

1. manifest identity 与路径一致；
2. operation input/output 均有正式 Schema；
3. Adapter 无跨实现 import、无长期进程所有权；
4. 参数默认值属于 Adapter 或 profile，不进入 Core；
5. secret、路径和 subprocess 环境遵守安全边界；
6. 结果保留 raw、逐任务指标和必要配置；
7. 脱敏检查、静态契约、Schema 和单测全部通过。

## 13. 发布与目录边界

源码、协议 Schema、内置 Presets 和 Adapter 构成运行目录。项目直接从源码运行，不要求 wheel 或 ZIP。`results/`、cache、runtime 临时状态和 build/dist 不进入 Git 仓库，也不应在发布过程中被删除。

公开仓库的门禁是脱敏扫描、Schema、静态合同和 Linux 单测。运行结果不生成防篡改证据，也不要求用户为模型、数据或结果预登记 SHA。

长期稳定性优先级：

```text
Canonical object schemas
  > Adapter API / operation contracts
  > Spec semantics
  > user-config convenience syntax
  > built-in implementation details
```

兼容变更应优先增加可选字段；破坏 canonical object 或 Adapter API 的变更必须升级对应 schema/API version。
