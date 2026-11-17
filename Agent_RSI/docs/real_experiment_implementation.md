# Agent RSI 真实实验实施方案

## 结论

真实实验不从 InterCode 开始扩展为正式数据集。InterCode 只用于验证环境适配器、轨迹记录和奖励读取是否工作。主实验使用 `IQuestLab/ModularRSI_2000_Instances`，原因是它提供 1000 个终端任务和 1000 个软件工程任务，每个任务都采用 Harbor 本地任务格式，包含环境、指令、测试与参考解法，并且论文使用的 Harness 正好分成 Agent Loop、Tool Use、Observation Management、Context Management 和 Task Completion Detection 五个模块。

正式评测使用 Terminal-Bench 2.1 和 SWE-bench Verified。Terminal-Bench 由 Harbor 执行和评分；SWE-bench 使用官方 `swebench` CLI 和 Docker evaluator。不要自行重新实现这两个 evaluator。

## 研究对象

首个白盒模型固定为 `Qwen/Qwen3-8B`。它是 dense decoder-only 模型，Hugging Face 模块结构稳定，适合使用现有 `HFBackend` 和 PyTorch forward hooks。首轮复现实验完成后，再用同系列 14B 模型做规模复核。另用 `Qwen/Qwen3.5-9B` 做系统层架构迁移，只报告冻结 Harness 的行为迁移，不把其混合 DeltaNet/Attention 层纳入主机制比较。不要在首版使用 MoE 模型，因为专家路由会引入额外机制变量。

Agent 使用 mini-SWE-agent 风格的线性循环，但执行环境统一接入 Harbor：

```text
instruction
  -> model reasoning and action
  -> shell tool call
  -> environment observation
  -> append to message history
  -> next model call
  -> completion or step limit
```

需要自行实现一个 Harbor `BaseAgent` 适配器，使模型推理运行在容器外，shell 动作在任务容器内执行。不要修改 Harbor verifier，也不要让 Agent 访问 `tests/`、`solution/`、参考补丁或验证日志。

## 数据集与固定划分

### Pilot

使用 ModularRSI terminal domain 中官方标记为 `train` 的 120 个任务。按任务类别分层，并用种子 2026 固定抽取 30 个任务。每个任务在相同 H0 下运行 4 个随机种子。

Pilot 只回答：

1. Harbor 适配器能否运行任务并读取 reward；
2. 激活能否按模型调用、Agent step 和反思 Token 对齐；
3. 是否能够获得同任务 success/failure 配对；
4. 结构化更新能否被编译、加载和重跑。

Pilot 结果不得作为论文最终性能结果。

### Main mechanism split

对 terminal domain 的 120 个 train 任务按类别分层，固定划分为：

- mechanism train: 80 tasks；
- mechanism dev: 20 tasks；
- mechanism test: 20 tasks。

每个任务生成 5 个 rollout。Probe 只在 train 拟合，在 dev 选择层、Token 位置、正则强度和 intervention site，在 test 只报告一次。

同任务 rollout 分组为：

- contrastive: 同时存在成功和失败；
- negative: 全部失败；
- positive: 全部成功；
- unstable: reward 或环境状态不一致，需要排除或单独报告。

机制主分析使用 contrastive 组，但必须在全部任务上报告覆盖率，避免只选择容易产生正结果的任务。

### Harness evolution split

Harness 更新使用同一组 120 个任务，但角色固定：

- evolve: 80 tasks；
- dev gate: 20 tasks；
- internal held-out: 20 tasks。

任何候选更新只能使用 evolve 轨迹提出。候选先在触发它的 mini-batch 上重跑，再在 dev gate 上决定接受或拒绝。internal held-out 不参与候选生成、层选择、阈值选择或版本选择。

### Cross-domain replication

在 software-engineering domain 上使用相同的 80/20/20 协议复核。不要把 terminal 和 SWE 任务随机混在一个 split 中。分别演化后做交叉迁移：

```text
terminal-evolved harness -> SWE held-out
SWE-evolved harness      -> terminal held-out
```

### Final external evaluation

1. Terminal-Bench 2.1：完整任务集，Harbor 官方 verifier。
2. SWE-bench Verified：开发期间使用预注册的 100-task repository-stratified subset，最终结果运行完整 500 tasks。
3. 可选：SWE-bench Multilingual，用于检查 Harness 是否只适配 Python repository。

SWE-bench Verified 公开任务可能存在模型记忆和 benchmark 过拟合问题，因此它只作为外部行为评价。Harness 更新不得读取 gold patch、test patch 或 evaluator 输出细节。

## 评测工具

| 目的 | 采用工具 | 是否自行实现 |
|---|---|---|
| ModularRSI task 执行 | Harbor | 复用 |
| Terminal-Bench 2.1 | Harbor | 复用 |
| SWE-bench 生成轨迹 | mini-SWE-agent loop 或 Harbor mini-swe-agent adapter | 复用并加 Hook |
| SWE-bench 最终评分 | 官方 `swebench eval` | 复用 |
| GEPA | 官方 `gepa` 的 `GEPAAdapter` | 写项目适配器 |
| AutoSaddler | 官方 V2 engine、event store、EvoDAG 思路 | 写 ModularRSI/Harbor scenario plugin |
| 模型激活 | Transformers + PyTorch forward hooks | 扩展现有 `HFBackend` |
| Linear Probe | scikit-learn LogisticRegression | 复用 |
| ReCast | 作者代码或按论文训练 encoder | 优先复用 |
| Patching/Steering | 项目现有 Hook；必要时参考 TransformerLens | 自行实现 Agent 序列对齐 |
| SAE | SAELens | 第二阶段复用 |
| 统计 | task-cluster bootstrap、paired permutation、McNemar | 自行实现小型统计模块 |

Harbor 输出 reward、trial、agent trajectory 和 verifier 结果。SWE-bench 官方 evaluator输出每个 instance 是否 resolved 及运行日志。项目只负责把两种输出转换成统一 EpisodeRecord，不能改变官方判分。

## Harness 表示

Harness 固定为五个具名组件：

```json
{
  "agent_loop": "...",
  "tool_use": "...",
  "observation_management": "...",
  "context_management": "...",
  "completion_detection": "..."
}
```

首版更新不是自由代码生成，而是结构化操作：

- ADD_GUARD：增加执行前置或完成条件；
- ADD_VALIDATION：要求显式验证；
- RETRY_ON_FAILURE：在指定错误后重试；
- REORDER：调整规划、执行和检查顺序；
- RETAIN：保留关键观察；
- COMPRESS：压缩低价值观察；
- RETRIEVE：按条件检索历史经验；
- STOP_RULE：修改终止条件；
- NO_OP：不更新。

每个操作由确定性 compiler 转换成组件文本或受控代码 diff。这样可以把机制标签固定为：

```text
failure step
root cause
target module
update operator
scope
```

自由代码更新只在结构化实验稳定后加入。

## 统一轨迹记录

每个 episode 必须保存：

- dataset、task_id、split、task category；
- model id、revision、tokenizer revision；
- harness version 和五个组件内容哈希；
- rollout seed、temperature、step limit；
- 每个 Agent step 的 prompt、action、tool args、observation、reward；
- 环境终态和官方 verifier 结果；
- 反思文本、Token IDs、绝对 Token 位置和阶段边界；
- 激活文件路径及层、组件、dtype、shape；
- failure label、root cause、target module、operator；
- proposal、candidate diff、accepted diff；
- same-batch、dev、held-out 结果；
- parent version、child version 和回滚记录。

激活不要直接写入 JSONL。JSONL 保存索引，激活使用分片 `safetensors` 或 `npz`，并按 run/task/rollout/model-call 组织。

## 系统基线

所有基线必须共享任务顺序、rollout seeds、模型、最大步骤、Token 预算和 verifier 调用预算。

### H0

固定五模块 Harness，不反思、不更新。它是所有 gain 的共同起点。

### Text Reflection

失败后生成反思，并在同一任务重试时加入上下文；任务结束后不持久化。该基线测量 test-time retry 收益。

### GEPA

使用官方 GEPA。Candidate 是五组件文本映射。项目实现：

- `evaluate(batch, candidate)`：调用 Harbor；
- `make_reflective_dataset(...)`：将轨迹、reward 和失败标签变成组件级反馈；
- `propose_new_texts(...)`：使用 GEPA 默认 proposer。

GEPA 只优化文本组件，不访问隐藏激活。

### AutoSaddler-style

优先复用 AutoSaddler V2 的事件存储、候选版本、恢复与 EvoDAG。需要编写 scenario plugin，把 ModularRSI task、Harbor evaluator 和五模块 Harness 交给 AutoSaddler。若官方 V2 当时仍没有 Terminal-Bench 插件，再实现兼容的最小 plugin，不复制整个优化器。

该基线执行：mini-batch diagnosis、structured patch、same-batch verification、dev gate、reflection 和 EvoDAG evolution。它不访问隐藏激活。

### Ours

与 AutoSaddler-style 使用相同候选数和评价预算。唯一新增输入为从白盒模型状态得到的：

- Probe posterior；
- causal site；
- patch/ablation/steering 后的 update distribution。

最终候选仍必须通过相同 same-batch 和 dev gate。

## 机制基线

1. Text classifier：任务、轨迹、反馈和反思文本 TF-IDF 或 embedding classifier。
2. Linear Probe：逐层、逐阶段 LogisticRegression。
3. ReCast：pattern、deviation、context encoder、contrastive loss 和 ranking loss。
4. Random matched：同层、同 Token 位置、等范数随机扰动。
5. Cross-task donor：来自不同任务但相同标签的 donor。
6. Sequential patch：字段窗口内连续 Token patch，而不是只 patch 反思末尾。
7. Ablation：zero、mean、direction removal。
8. Steering：正负方向和多个强度，报告 dose-response。

每次干预必须同时报告更新选择变化、语言连贯性、非目标标签变化和最终任务行为，不能只报告 logit 变化。

## 主要指标

### 机制指标

- task-disjoint macro-F1、AUROC、Hit@1、MRR；
- 正确信息首次可解码的 Token 和层；
- hidden Probe 相对 text classifier 的增益；
- intervention 后正确 module/operator 概率变化；
- donor-choice rate；
- 随机控制校正后的 causal effect；
- 干预特异性和输出连贯性。

### Harness 指标

- same-task repair rate；
- evolve/dev/internal-held-out pass rate；
- external Pass@1 或 resolved rate；
- regression rate；
- retained-success rate；
- accepted updates、rejected updates、rollback 数；
- steps、policy tokens、wall time、verifier calls；
- patch size 和被修改组件数。

### 统计

- 性能比例使用任务级 paired bootstrap 95% CI；
- 同任务二元成功差异使用 McNemar 检验；
- 连续 causal effect 使用以 task 为 cluster 的 bootstrap 或 permutation test；
- Harness evolution 至少运行 3 个独立 seed；
- 层、站点和干预强度只在 dev 选择。

## 实施顺序

1. 定义统一数据合同和五模块 Harness。
2. 编写 Harbor `BaseAgent`，复用 mini-SWE-agent 线性循环。
3. 编写 Harbor trial 到 EpisodeRecord 的转换器。
4. 在 30 个 ModularRSI terminal task 上跑 H0 smoke test。
5. 接入 HFBackend，按 model call 采集选定层残差流。
6. 生成 success/failure pairs，运行 text classifier 和 Linear Probe。
7. 实现单点、随机和 sequential patching。
8. 实现结构化 update compiler 和版本管理。
9. 接入 GEPA adapter。
10. 接入 AutoSaddler scenario plugin。
11. 运行 120-task 主机制实验。
12. 在 SWE domain 复核。
13. 冻结 Harness 后运行 Terminal-Bench 2.1 和 SWE-bench Verified。

## 学校机器运行环境

代码开发与数据 manifest 可以在当前机器完成。正式运行环境建议：

- Ubuntu 22.04 或更新版本；
- Docker；
- Python 3.12 和 uv；
- NVIDIA GPU 与匹配的 PyTorch/CUDA；
- 24 GB GPU 运行 7B pilot，48 GB GPU 更适合 14B 或较长上下文；
- 至少 16 CPU、64 GB RAM 和 500 GB 可用 SSD；
- Docker 与模型缓存放在高速本地磁盘。

第一次同步时解析并固定所有 Git commit、Hugging Face revision、Docker image digest 和数据 manifest hash，之后不再跟随 `main`。
