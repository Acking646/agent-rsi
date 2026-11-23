# 任务书与当前代码对应

| 研究任务 | 当前实现 | 状态与边界 |
|---|---|---|
| RSI 统一记录 | `rsi/contracts.py`、`rsi/adapters/harbor_records.py` | 已把真实 Harbor 轨迹、官方 verifier、失败标签、提案和版本统一为可审计记录 |
| 五模块 Harness | `rsi/harness/schema.py` | 已实现 Agent Loop、Tool Use、Observation Management、Context Management、Completion Detection |
| 结构化更新空间 | `rsi/harness/compiler.py` | 已实现预定义操作编译和确定性 diff；不允许模型自由改系统代码 |
| 真实 Agent 执行 | `rsi/adapters/harbor_agent.py` | 已实现宿主机 Qwen3 推理、Harbor 容器命令执行、上下文管理和完成判断 |
| 多粒度状态采集 | `InstrumentedModel` | 已在每次模型调用采集指定层 residual、attention、MLP 输出并与调用/步骤关联 |
| 失败诊断与提案 | `generate_structured_update()` | 已从失败 Episode 生成根因、责任步骤、目标模块和结构化操作；弱证据允许 NO_OP |
| 跨轮次 RSI | `rsi/formal.py`、`rsi/evolution/version_store.py` | 已实现父版本执行、失败提案、候选编译、同批次复验、dev 门控、版本写回和下一代复用 |
| 泛化评测 | `rsi/datasets.py`、`scripts/run_external_eval.sh` | 已实现 task-disjoint held-out；冻结 Harness 可在 Terminal-Bench 和 Harbor 格式 SWE-bench 上运行 |
| 机制读出 | `rsi/mechanism/probe.py`、`collect.py` | 已实现文本控制和逐层线性 Probe，训练/dev/test 按任务隔离 |

## 方案中的因果链

当前数据结构把三个对象直接连起来：每个 Agent step 记录产生动作的 model call；该 model call 指向对应激活文件；失败诊断记录责任步骤、目标 Harness 模块和更新操作。由此可检验：

```text
Token 前缀与生成阶段
        ↓
指定层/组件的内部激活
        ↓
失败归因、目标模块与更新操作
        ↓
编译后的 Harness 子版本
        ↓
同批次、dev 与 held-out 的官方 verifier 结果
```

当前已完成相关性读出和真实更新闭环。Sequential activation patching、必要性消融、方向 steering、ReCast 表示训练，以及官方 GEPA/AutoSaddler 对照应作为后续 GPU 实验模块接入；配置文件不会把这些尚未执行的实验伪装成已有结果。

## Harness offline learning 与 RSI

单次从历史轨迹学习 Harness 属于 offline learning。这里的 RSI 需要接受后的 `active-harness.json` 成为下一代的真实执行条件，再由新轨迹触发新诊断和新候选。`events.jsonl` 明确记录每次接受或拒绝，因此可以检查系统是否真的发生跨轮自改进，而不是只对固定日志做一次优化。