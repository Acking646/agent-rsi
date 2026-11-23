# Agent 优先的 RSI 文献与内部机制接口

## 研究定位

本项目应先把 Agent 作为研究对象，再选择内部机制工具。核心对象不是一般数学推理中的 Chain of Thought，而是具有任务状态、工具调用、环境反馈、跨步依赖和持久更新工件的 Agent 执行闭环：

```text
任务与环境状态
→ Agent 执行轨迹
→ 失败结果与验证信号
→ 关键错误和根因归因
→ 反思与修复选择
→ Skill Memory Harness 更新
→ 后续任务调用更新
→ 修复 泛化 回归和成本
```

Probe、SAE、activation patching 和 steering 用于解释这条 Agent 链中的内部计算。它们不是独立研究目标。

## 第一组 失败位置与因果标签

### Who and When Pro

本地目录：`agent论文/失败归因与表示学习`

这项工作提供大规模 Agent 失败归因基准，检验模型能否定位责任 Agent 和关键步骤。适合作为失败归因任务、人工标签和困难样本来源。

内部机制接口：

- 对每个 Agent step 的隐藏状态训练责任步骤 Probe；
- 比较直接归因文本、文本分类器与隐藏状态 Probe；
- 在同一轨迹内替换候选步骤表示，观察责任步骤分数是否移动；
- 分析责任 Agent、责任 step 和根因类别是否由不同层表示。

### Executable Prefix Values for Locating Agent Failures

本地目录：`agent论文/失败归因与表示学习`

该方法从失败轨迹的中间状态重启 Agent，估计每个前缀继续成功的概率，并识别 point of no return。它把“第一处看起来错误”改造成可执行反事实标签。

内部机制接口：

- 用 prefix value 代替纯人工责任标签；
- 检验内部失败信号与真实价值下降是否同步；
- 将 value 突降前后的激活构造成最小对比样本；
- 检验内部错误信号是根因前出现、根因处形成，还是只在后续症状处变强。

### Why Do Multi Agent LLM Systems Fail MAST

本地目录：`agent论文/失败归因与表示学习`

MAST 提供多 Agent 系统失败分类和真实失败轨迹。它适合把当前单 Agent 真实任务系统扩展到多 Agent 失败轨迹，使研究覆盖规范理解、Agent 间不一致、协调失败和验证失败。

内部机制接口：

- 比较单 Agent 局部错误和跨 Agent 传播错误；
- 分析发送方、接收方与协调器中的错误表示；
- 对通信消息和接收后的隐藏状态分别做 patching；
- 检验失败信息是否在 Agent 间传递时丢失或被重解释。

## 第二组 成功失败轨迹到 Skill 和 Memory

### SkillForge

本地目录：`agent论文/轨迹技能与记忆`

SkillForge 对同一任务的成功和失败执行进行配对，提炼造成结果差异的关键程序性选择。这种数据结构比任意跨任务 donor 更适合内部机制研究，因为任务目标和环境基本保持不变。

内部机制接口：

- 使用同任务成功失败对作为 clean 和 donor；
- 定位两条轨迹首次出现行为差异的位置；
- 在差异前后进行 sequential activation patching；
- 检验提炼出的 Skill 内容是否对应可干预内部方向；
- 区分文本相似性、行为选择和最终成功的变化。

### SKILLREVISE

本地目录：`agent论文/轨迹技能与记忆`

SkillRevise 从执行证据诊断初始 Skill 的缺陷，检索修复原则，执行锚定编辑并重跑候选。它为“内部诊断状态到实际 Skill 修改”提供了完整外部闭环。

内部机制接口：

- 分别 Probe defect detection、repair principle 和 edit operator；
- 在诊断结束、原则检索和编辑提案位置记录激活；
- 干预诊断状态后，检查编辑位置和补丁类型是否改变；
- 继续运行重试和迁移任务，检查内部干预是否改变真实 Skill 效用。

### When Not to Imitate

本地目录：`agent论文/轨迹技能与记忆`

Boundary Aware Skill Memory 为 Skill 增加适用条件、禁止条件和失败修复条件。它表明成功经验不是无条件可复用规则。

内部机制接口：

- 构造 Skill 内容相同但适用状态不同的最小对；
- Probe 当前状态是否满足 Skill activation boundary；
- 干预 boundary representation，检查 Agent 是否错误调用或正确抑制 Skill；
- 将 Skill 检索、Skill 激活和 Skill 遵循拆成三个独立端点。

### Agent Skills Can Be Harmful

本地目录：`agent论文/轨迹技能与记忆`

该工作对同一任务比较 Skill 引导、无 Skill 和语义匹配 Skill 运行，将退化归因到具体 Skill。它提供了比“有记忆优于无记忆”更干净的差分设计。

内部机制接口：

- 比较 Skill 注入前后对状态表示的改变；
- 检查有害 Skill 是否覆盖了原本正确的环境表示；
- 对 Skill 相关方向进行消融，检验能否恢复无 Skill 行为；
- 使用 semantically matched skill 排除仅由长度和主题产生的激活变化。

### ReasoningBank

本地目录：`agent论文/轨迹技能与记忆`

ReasoningBank 从成功和失败经历中蒸馏可检索的推理策略，是外部推理记忆路线的重要行为基线。

内部机制接口：

- 比较原始轨迹、自然语言记忆和无记忆条件；
- 检验记忆是否让相关改进状态更早出现；
- 区分记忆被检索、被内部表示和被实际遵循；
- 通过错误记忆、同义记忆和随机记忆检验因果性。

## 第三组 Harness 持久更新和实际收益

### Harness Updating Is Not Harness Benefit

本地目录：`agent论文/Harness与持久修复`

该工作将两种能力分开：生成有用 Harness 更新的能力，以及任务 Agent 调用并遵循更新的能力。弱模型可能生成看似合理的更新，却无法在执行中激活或遵循它们。

内部机制接口：

- 在更新生成模型中研究 diagnosis 和 proposal state；
- 在执行模型中研究 retrieval、activation 和 following state；
- 对相同 Harness 使用不同能力模型，避免把更新质量和使用能力混为一谈；
- 检验内部 Harness 表示能否预测最终收益而不只是更新文本质量。

### AEL Evolving Agent Harness in Open Ended Environments

本地目录：`agent论文/Harness与持久修复`

AEL 使用快时间尺度 bandit 逐回合选择记忆检索策略，并在慢时间尺度通过 diagnose before prescribe 的反思生成新策略。其机制实验表明，只有环境状态要求不同检索行为时，反思更新才产生收益。

内部机制接口：

- 将环境 regime 作为因果变量；
- Probe 当前 regime、策略适用性和 plateau 判断；
- 在稳定 regime 与变化 regime 中分别干预反思状态；
- 检验内部状态变化是否只在真正需要新策略时影响行为。

### Living Harness

本地目录：`agent论文/Harness与持久修复`

Living Harness 将每条完成轨迹和评价信号转换成 episodic memory 与 state graph，使失败模式、触发条件、恢复动作和转换规则在后续任务中持续积累。

内部机制接口：

- 将 state graph 节点作为明确的外部 Agent 状态；
- 比较 graph retrieval 前后模型内部状态；
- patch 触发条件、恢复动作和状态转换的表示；
- 追踪内部状态如何转化为外部 graph 更新，再如何在下一轮重新进入模型。

## 与现有核心论文的组合

新增 Agent 论文负责定义问题、数据和端点；现有内部机制论文提供分析方法：

| Agent 研究变量 | 数据或标签来源 | 内部机制方法 |
|---|---|---|
| 关键失败步骤 | Who and When Pro、MAST、AgentRx | 逐步 Probe、层选择、ReCast |
| 因果失败位置 | Executable Prefix Values | value 对齐 Probe、前后 patching |
| 成功失败差异 | SkillForge | sequential activation patching |
| Skill 缺陷和修复操作 | SkillRevise、RESKILL | 分阶段 Probe、方向注入和消融 |
| Skill 激活边界 | When Not to Imitate | 最小对、boundary steering |
| Skill 引发的退化 | Agent Skills Can Be Harmful | 有无 Skill 差分、反向消融 |
| 记忆检索和使用 | ReasoningBank、AEL | retrieval activation following 分解 |
| Harness 更新质量 | AutoSaddler、Evo Harness | proposal state 和补丁类型干预 |
| Harness 实际收益 | Harness Updating Is Not Harness Benefit | 执行期激活和遵循状态 |
| 跨回合持久状态 | Living Harness | 内部状态与 state graph 双向追踪 |

## 建议的实验主线

### 实验一 Agent 失败状态形成

使用可重启环境生成成功失败配对轨迹，以 executable prefix value 标定因果失败点。在每个 Agent step 的 observation、thought、action 和 feedback 边界记录激活，研究错误信息何时形成、在哪些层可读，以及是否早于显式失败文本。

### 实验二 从诊断状态到修复选择

采用 SkillRevise 或 RESKILL 式结构，将修复过程拆成 evidence、responsible step、root cause、affected artifact 和 repair operator。分别训练 Probe，并在阶段边界进行 donor patch、方向注入、方向消融和随机对照。

### 实验三 从内部干预到持久收益

将 clean 和 intervened proposal 从相同初始 Harness 独立写回，分别运行原失败任务、同模板新任务、跨模板任务和旧任务回归测试。报告 repair gain、near transfer、far transfer、regression 和成本。

### 实验四 更新调用和遵循

固定同一份更新工件，改变执行模型能力和任务状态，区分更新是否被检索、是否在内部激活、是否被遵循，以及遵循后是否真正提高成功率。

## 结论边界

Agent 优先的机制研究应保持四个结论层级：

1. 轨迹中存在与失败或修复相关的文本证据；
2. 模型隐藏状态可解码相应信息；
3. 干预隐藏状态会改变 Agent 的诊断或修复行为；
4. 行为变化会产生可验证、可泛化且无明显回归的持久改进。

只有第四层证据才能支持 Agent RSI 能力提升。前三层分别属于轨迹分析、内部诊断和行为因果机制。
