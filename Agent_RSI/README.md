# Agent RSI

本仓库实现面向真实软件工程任务的 Agent RSI 系统。执行层使用 Harbor，Agent 采用 mini-SWE-agent 风格的线性循环，基础模型为可读取中间激活的 Qwen3 dense decoder-only 模型。系统不再包含购物篮或其他合成任务。

## 已实现的闭环

1. Harbor 启动 ModularRSI、Terminal-Bench 或转换后的 SWE-bench 任务，并调用官方 verifier 评分。
2. `WhiteboxHarborAgent` 在宿主机加载模型，在隔离任务容器中执行 shell 动作；每次模型调用保存 residual、attention 和 MLP 激活。
3. 失败轨迹经过白盒诊断，生成一个受约束的结构化更新提案。
4. `HarnessCompiler` 只允许预注册操作修改五个 Harness 模块，不执行模型生成的代码。
5. 候选 Harness 必须通过原失败批次非退化和 task-disjoint dev 配对门控，才能成为下一代父版本。
6. 演化结束后冻结版本，在未参与选择的 held-out 任务上评测。版本、父子关系、接受/拒绝和回滚事件均写入磁盘。

五个可更新模块为 `agent_loop`、`tool_use`、`observation_management`、`context_management` 和 `completion_detection`。允许的更新操作为 `ADD_GUARD`、`ADD_VALIDATION`、`RETRY_ON_FAILURE`、`REORDER`、`RETAIN`、`COMPRESS`、`RETRIEVE`、`STOP_RULE` 和 `NO_OP`。

这里的 RSI 与 Harness offline learning 的区别在于：offline learning 只从固定日志学习一次 Harness；本系统中通过门控的子版本会在下一代真实任务执行中成为父版本，产生新的轨迹、激活、失败诊断和更新提案。外部评测始终使用冻结版本，不参与更新。

## 代码结构

| 路径 | 作用 |
|---|---|
| `rsi/adapters/harbor_agent.py` | Harbor Agent、Qwen3 推理、容器工具调用和逐调用激活采集 |
| `rsi/adapters/harbor_records.py` | Harbor 官方结果到统一 EpisodeRecord 的转换 |
| `rsi/harness/` | 五模块 Harness schema 与结构化更新编译器 |
| `rsi/evolution/version_store.py` | 版本谱系、接受/拒绝、active 指针和回滚记录 |
| `rsi/datasets.py` | ModularRSI 的 evolve/dev/held-out 固定切分 |
| `rsi/formal.py` | 真实 pilot 与多代 evolve 主入口 |
| `rsi/mechanism/` | task-disjoint 文本分类器和逐层线性 Probe |
| `configs/real_pilot.json` | Qwen3、Harbor、激活层和评测配置 |
| `scripts/run_external_eval.sh` | 冻结 Harness 的 Terminal-Bench/SWE-bench 外部评测入口 |
| `third_party/ModularRSI` | 固定提交的官方 ModularRSI/Harbor 子模块 |

## 正式运行入口

所有输出目录必须是新目录。下面的 `DATASET_ROOT` 指向官方 ModularRSI 数据根目录，其下应有 `tb/manifest.json`、`tb/tasks/`、`swe/manifest.json` 和 `swe/tasks/`。

```bash
python -m rsi --config configs/real_pilot.json pilot \
  --dataset-root "$DATASET_ROOT" --domain tb \
  --max-tasks 30 --out runs/pilot-tb

python -m rsi --config configs/real_pilot.json evolve \
  --dataset-root "$DATASET_ROOT" --domain tb \
  --generations 3 --candidates 3 \
  --out runs/evolve-tb
```

机制数据采集和 Probe 使用同一真实任务执行器，并按任务划分训练、开发和测试集：

```bash
python -m rsi.mechanism.collect \
  --config configs/real_pilot.json \
  --dataset-root "$DATASET_ROOT" --domain tb \
  --out runs/mechanism-tb
```

冻结版本的外部评测：

```bash
HARNESS_PATH=runs/evolve-tb/lineage/active-harness.json \
DATASET=terminal-bench@2.1 \
bash scripts/run_external_eval.sh

HARNESS_PATH=runs/evolve-swe/lineage/active-harness.json \
DATASET_PATH=/path/to/harbor-swebench-tasks \
bash scripts/run_external_eval.sh
```

`DATASET_PATH` 应是 Harbor 格式的 SWE-bench 任务目录，可由 SWE-bench 的 Harbor 转换工具生成。实际模型、Docker、GPU 和完整数据在学校 Linux 机器上运行；当前仓库保存代码、固定清单和实验协议，不把大规模模型或任务数据提交到 Git。

## 输出与审计

每个任务运行保存 Harbor 原始结果、`episodes.jsonl`、Harness 快照、逐次模型调用的激活文件和汇总指标。`lineage/versions/` 保存所有候选版本，`lineage/events.jsonl` 保存门控决定，`lineage/active-harness.json` 是可直接用于外部评测的冻结版本。

本地静态验证覆盖五模块 schema、结构化操作编译、版本接受/拒绝/回滚以及固定任务切分。真实 benchmark 分数必须在具备相应数据、容器和模型的 Linux GPU 机器上产生。