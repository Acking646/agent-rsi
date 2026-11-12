# Agent RSI 前沿文献与实验设计路线图

## 文档范围

本文整理截至 2026 年 11 月 12 日与 Agent 递归自我改进、失败归因、反思内部状态、因果干预和持久更新最相关的前沿工作。检索范围包括 ACL 2026、EMNLP 2026、NeurIPS 2025–2026、ICLR 2026–2027 在审投稿以及 arXiv 最新预印本。

论文状态需要严格区分：ACL Anthology、ICLR Proceedings 和 NeurIPS Proceedings 中的论文属于正式论文集；ICLR 2027 active submissions 尚未完成录用；只在 arXiv 出现的工作应标为预印本。

本项目要研究的完整因果链是：

```text
执行失败
→ 定位关键错误步骤
→ 从轨迹和反馈形成内部改进状态
→ 内部状态驱动反思与更新选择
→ 生成 Prompt、Memory、Skill 或 Harness 修改
→ 独立验证并持久写回
→ 在后续未见任务中产生收益
```

现有工作通常只覆盖其中一段。本项目的核心机会，是把内部状态形成、因果干预、实际更新工件和跨任务效果连成一条可检验的证据链。

## 最优先阅读的论文

| 优先级 | 论文 | 与本项目的关系 | 可直接借鉴的方法 |
|---|---|---|---|
| S | [A Mechanistic Study of Automated Failure Attribution in LLM-based Agentic Systems](https://arxiv.org/abs/2610.28811) | 最接近失败归因内部机制 | 分层线性 Probe、文本与内部知识差距、内部信息利用 |
| S | [From Latent Signals to Reflection Behavior](https://arxiv.org/abs/2602.01999) | 研究反思状态如何逐层形成 | 层与生成时间双轴分析、三阶段因果链、目标干预 |
| S | [Triggering Chain-of-Thought via Latent Feature Interventions](https://aclanthology.org/events/emnlp-2026/) | 用 SAE 找到并干预推理特征 | 特征注入、抑制、残差增量注入、表面语言控制 |
| S | [Hidden Error Awareness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2605.09502) | 严格区分诊断信号和因果变量 | Probe、文本对照、steering、patching 失败结果 |
| S | [AutoSaddler](https://arxiv.org/abs/2608.23041) | 从失败轨迹到持久 Harness 更新 | 深度诊断、结构化补丁、验证选择、版本轨迹 |
| S | [RESKILL](https://arxiv.org/abs/2609.15684) | 显式连接失败假设和候选修复 | repair state、覆盖归因、重测条件更新 |
| S | [Evo-Harness](https://arxiv.org/abs/2608.15071) | 从单次上下文编译可复用 Skill | trajectory-to-skill compilation、在线写回、迁移评估 |
| S | [Self-Evolving Agents Can Repair Failures but Struggle to Generalize from Them](https://arxiv.org/abs/2610.22984) | 揭示修复与泛化之间的鸿沟 | CFOE、失败专用演化、repair-generalization gap |
| A | [ReCast](https://arxiv.org/abs/2610.11334) | 学习面向归因的步骤表示 | 层选择、pattern/deviation 特征、对比与排序目标 |
| A | [Mechanistic Interpretability of CoT via Sequential Activation Patching](https://arxiv.org/abs/2608.22332) | 将单点 patching 扩展到连续生成过程 | 逐 Token patching、多 Head 联合干预、随机和跨问题控制 |

## 内部状态与因果机制

### State over Tokens

[Characterizing the role of reasoning tokens in large language models as state over tokens](https://www.nature.com/articles/s42256-026-01303-y) 于 2026 年发表于 Nature Machine Intelligence。它提出，推理 Token 不应只被视为供人阅读的解释，而应视为跨自回归生成步骤传递计算状态的外部载体。

本项目可以将该框架扩展到 improvement reasoning：反思 Token 是外部化状态载体，残差流、Attention、MLP 和稀疏特征是内部实现，Prompt、Memory、Skill 和代码补丁则是跨调用持久化的 Agent 外部状态。三者不能混为一谈。

### Automated Failure Attribution 的内部机制

[A Mechanistic Study of Automated Failure Attribution](https://arxiv.org/abs/2610.28811) 发现，模型直接回答责任步骤时可能很弱，但隐藏状态中已经编码了足够的失败归因信息，形成 knows-but-won't-say gap。

可以将其任务扩展为同时预测：

- 执行是否失败；
- 最早关键错误步骤；
- 错误根因；
- 需要修改的 Agent 组件；
- 推荐修改操作；
- 候选更新是否会通过验证。

这类结果只能证明信息可解码。要证明模型真正使用这些信息，仍需因果干预。

### 反思状态的形成轨迹

[From Latent Signals to Reflection Behavior](https://arxiv.org/abs/2602.01999) 围绕反思开始位置追踪逐层激活，观察到 latent-control、semantic-pivot 和 behavior-overt 三个阶段。

本项目可以检验相似但更贴近改进决策的阶段：

```text
反馈读取
→ 错误定位
→ 原因归因
→ 修改对象选择
→ 修改操作选择
→ 更新提案输出
```

不能预先假定这些阶段存在，应通过跨层 Probe、logit lens、因果替换和消融共同检验。

### SAE 特征干预

[Triggering Chain-of-Thought via Latent Feature Interventions](https://aclanthology.org/events/emnlp-2026/) 的核心设计是比较直接回答与 CoT 条件，找出差异性 SAE 特征，再进行双向干预：注入检验充分性，抑制检验必要性。

适合迁移到本项目的做法包括：

- 用残差增量而不是完整 SAE 重构替换，减小重构误差；
- 在不同提示措辞和任务上验证特征是否泛化；
- 设置只有反思措辞、没有实际诊断内容的 pseudo-reflection；
- 同时报告操作选择变化和最终任务收益；
- 对候选特征进行随机特征、相同频率特征和等范数方向控制。

### 诊断性不等于因果性

[Hidden Error Awareness](https://arxiv.org/abs/2605.09502) 表明，线性 Probe 可以非常准确地预测推理是否错误，但 activation steering、best-of-N、自我修正和 activation patching 都不能稳定修复错误。

因此，本项目应将证据分成三个等级：

1. Encodable：内部表示可被 Probe 读取；
2. Behaviorally used：替换、注入或消融会改变更新选择；
3. Improvement-effective：变化后的更新通过验证，并提高未见任务表现。

只有第一等级时，结论应是诊断性信号，而不是因果改进状态。

### 时序 Activation Patching

[Sequential Activation Patching](https://arxiv.org/abs/2608.22332) 指出，推理机制跨多个生成 Token 展开，单个静态位置的 patching 不足以定位时序因果效应。

正式实验应在以下位置保存并干预激活：

- 任务执行结束；
- 环境反馈结束；
- 反思进度 25%、50%、75% 和 100%；
- 错误定位字段结束；
- 原因判断字段结束；
- 修改对象和修改操作形成位置；
- proposal readout 之前。

同时使用单点、连续窗口和多站点联合 patching，比较状态是集中、逐步形成还是分布式维持。

## 失败定位与传播

### AgentRx

[AgentRx](https://arxiv.org/abs/2602.02475) 将异构执行日志标准化，并结合工具 schema、环境规则、轨迹和约束验证定位关键失败步骤。它适合用于构造较客观的外部标签，避免完全依赖另一个 LLM 的自由判断。

### ReCast

[ReCast](https://arxiv.org/abs/2610.11334) 发现原始隐藏状态对根因步骤与下游受污染步骤的分离有限，因此学习 attribution-oriented step representations。其方法包括相关层选择、pattern 与 deviation 特征、上下文编码器、对比损失和排序损失。

原始隐藏状态 Probe 应是主实验；ReCast 式表示学习适合作为增强分析。训练后的表示属于外部分析器产生的表示，不能被描述为模型原生内部状态。

### DUOTRACE

[Detect Before You Attribute](https://arxiv.org/abs/2608.29646) 先从长轨迹检测异常子结构，再把聚焦证据交给归因模型。它结合语义与结构双视角表示、Tree-LSTM、前缀链增强和异常检测，适合作为长轨迹压缩和候选筛选模块。

### Error-Propagation Modeling

[Error-Propagation Modeling for Failure Attribution](https://arxiv.org/abs/2610.11600) 区分根因、传播、持续循环和最终症状，并通过反事实验证判断修复某一步是否足以恢复执行。它提示本项目的责任标签应优先定义为 decisive error，而不是最早出现异常措辞的步骤。

### Trajectory-Centered Error Records

[Dependable Long-Horizon LLM Agents](https://arxiv.org/abs/2611.13254) 使用 origin、propagation、detection、response 和 consequence 记录一次失败。它发现很多轨迹中已经存在反驳错误的证据，但 Agent 没有利用。

这一结构可以替换当前简单的 fault 标签：

```text
错误从何处进入
→ 通过什么状态传播
→ 哪些观察与它矛盾
→ Agent 是否检测到
→ 是否采取有效修复
→ 对最终结果造成什么影响
```

## 反思与错误恢复

### Failure Makes the Agent Stronger

[Failure makes the agent stronger](https://aclanthology.org/2026.findings-acl.618/) 将错误工具调用、证据化诊断、结构化反思、正确后续调用和最终结果组织成训练轨迹，并用程序验证反思格式、工具参数和执行结果。

建议采用以下结构化反思：

```json
{
  "observed_failure": "...",
  "evidence": ["..."],
  "responsible_step": "...",
  "root_cause": "...",
  "affected_component": "...",
  "repair_operator": "...",
  "expected_effect": "..."
}
```

这些字段可以分别与 Token 边界和激活对应，避免所有分析都使用一个笼统的动作标签。

### Echo Trap 与多轮反思

[Escaping the Echo Trap](https://aclanthology.org/2026.acl-long.1636/) 指出多轮反思可能只重复已有内容，并使用树状策略优化比较从同一状态分叉出的反思路径。

本项目可在相同任务、轨迹和反馈下生成多个反思分支，比较不同分支的内部状态、更新选择与留出收益，从而把最终结果归因到具体反思分叉，而不是平均归给整条轨迹。

### 早期失败信号的局限

[Last Step Matters](https://aclanthology.org/2026.emnlp-main.475/) 发现长程 Agent 的早期不确定性对最终失败预测较弱，最后阶段才明显增强，部分原因是中途路径切换。

因此需要报告随执行进度变化的预测性能，并按路径切换、任务长度和任务类型分组，不能只报告最终位置的高 Probe 分数。

## 从轨迹到持久更新

### AutoSaddler

[AutoSaddler](https://arxiv.org/abs/2608.23041) 将 Harness 优化建模为离线学习：从小批量失败轨迹进行深度诊断，生成将 Harness 视为代码的结构化补丁，并通过验证集选择持久更新。

其最重要的实验结论是：

- 深度调试优于浅层反思；
- 目标明确的小修改优于不受约束的自由重写；
- 泛化感知选择优于只修复触发任务；
- 更新需要保存父版本、候选差异、验证结果和最终部署版本。

当前固定动作集合适合机制实验。正式扩展时可以逐步加入 Prompt、Tool 描述、Memory 写回、Skill、Hook 和 Agent 循环逻辑，但不宜一开始就开放全部代码空间。

### RESKILL

[RESKILL](https://arxiv.org/abs/2609.15684) 显式维护失败假设、假设权重、候选 Skill 补丁、补丁覆盖关系和重测结果。一次失败重测不会清空状态，而会改变假设权重并影响下一轮修复。

这为内部状态与外部 repair state 的对接提供了直接模板。需要额外控制 LLM 判断的覆盖关系，包括随机覆盖、均匀覆盖和人工抽样验证。

### Evo-Harness

[Evo-Harness](https://arxiv.org/abs/2608.15071) 将一次任务中的长上下文编译成较短的可复用 Skill Harness，并连续写回经验。其迁移划分应直接采用：原任务重试、同主题新任务和跨任务族迁移。

### Harness 搜索与联合优化

- [MESH-Harness](https://arxiv.org/abs/2610.05300)：把 Harness 分成可替换模块，用 bandit 和坐标搜索在有限预算内重组模块。
- [VERSE](https://arxiv.org/abs/2610.02616)：同时演化执行 Harness 和负责修改 Harness 的优化器。
- [HarnessFlow](https://arxiv.org/abs/2611.06254)：用 MCTS 搜索完整可执行 Harness，保存候选和评估历史。
- [ParetoSaddler](https://arxiv.org/abs/2611.02079)：联合优化质量和成本，用 Bayesian bootstrap 判断候选是否具有可信进步。

这些方法适合后续扩大修改空间。首版机制论文仍应冻结模型权重、限制候选操作，以保持因果解释清晰。

## 泛化、评价器和闭环风险

### 修复不等于泛化

[Self-Evolving Agents Can Repair Failures but Struggle to Generalize](https://arxiv.org/abs/2610.22984) 提出 Cold-Start Failure-Only Evolution，发现失败经验常能修复原任务，却很少转化为未见任务上的持续能力提升。

应分别报告：

```text
Repair Gain
= 原失败任务更新后成功率 - 更新前成功率

Generalization Gain
= 未见任务更新后成功率 - 更新前成功率
```

只有 Generalization Gain 才能支持能力增长结论。

### ReasoningBank

[ReasoningBank](https://arxiv.org/abs/2509.25140) 从成功和失败轨迹中提炼通用推理策略并写入可检索记忆。适合作为记忆路线基线，并应比较失败经验、成功经验、混合经验、原始轨迹检索、结构化策略检索和随机记忆。

### GEPA

[GEPA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) 从执行轨迹生成自然语言反思，提出 Prompt 候选，并使用 Pareto 前沿保存和合并互补更新。它是反思驱动 Prompt 更新的重要外部行为基线，但不能替代内部机制分析。

### Co-Cheating

[False Frontiers](https://arxiv.org/abs/2609.39102) 是 ICLR 2027 在审方向，当前应引用为预印本。它发现自演化 proposer 与 solver 可能逐渐共享同一个错误，使内部奖励上升而外部正确率不升。

本项目应避免让同一个模型同时生成反思、判断补丁并验收补丁。至少加入独立环境验证器、不同模型 Judge、来源隔离验证和人工抽样审计。

### Verifier Overfitting

[Learning the Task or Learning the Verifier?](https://arxiv.org/abs/2611.08318) 表明，暴露验证器标准会使 Agent 针对判定规则优化，而不是真正学习任务。校准集验收不应向提案模型暴露完整验证器实现，并应准备语义等价的替代验证器和未公开留出验证器。

## 正式实验设计

### 任务因子解耦

当前受控环境中，缺参数、报价过期和漏验算与三个更新动作基本一一对应。这会导致结果可能只是错误类型分类。

正式数据至少需要：

| 设计 | 目的 |
|---|---|
| 相同根因，不同操作 | 分离归因与动作选择 |
| 不同根因，相同操作 | 检验动作表示是否独立于错误类别 |
| 相同反馈，不同上下文 | 排除反馈关键词到动作的直接映射 |
| 固定任务和轨迹，只改变反馈 | 识别反馈的独立作用 |
| 同义反馈 | 排除词面记忆 |
| 错误反馈 | 检验模型是否盲从外部评价 |
| 无反馈 | 检验能否从环境证据自行归因 |

### 表示解码

在每个阶段分别预测最终成功、关键错误步骤、根因、修改对象、修改操作和候选是否通过验收。

必要对照包括：任务文本、反馈文本、反思文本的 TF-IDF；Token 长度；关键词；随机标签；最终层；输出 logits；无反思条件。所有站点选择只使用开发集。

### 因果干预

干预组至少包括：

- self patch；
- 同任务异条件 donor；
- 同标签 donor；
- 异标签 donor；
- 等范数随机扰动；
- 均值替换；
- 方向注入和方向消融；
- SAE 特征注入和抑制；
- 连续 Token patching；
- 多站点联合 patching。

除 argmax 变化外，还应报告 donor 与 target 的 logit margin 变化、更新对象变化、实际工件差异和验证后的留出收益。

### 从内部干预到实际更新

每次干预都应从同一个初始 Agent 版本独立产生两条分支：

```text
Clean state
→ Clean proposal
→ Clean artifact
→ Validation
→ Holdout

Intervened state
→ Intervened proposal
→ Intervened artifact
→ Validation
→ Holdout
```

只有这种设计才能证明内部状态变化穿过了完整闭环，而不只是改变了一个选项 Token。

### 评价与统计

建议报告：

- 按 task_id 聚类 bootstrap 置信区间；
- 配对干预效应；
- repair、near transfer 和 far transfer；
- 旧任务回归率；
- 更新接受率；
- 每次有效更新的 Token、API、环境调用和墙钟时间成本；
- 跨模型、Prompt 和随机种子的稳定性；
- 独立评价器与同模型评价器的差异。

## 建议的论文主张

一个与现有工作有明确差异、且能由实验支持的主张是：

> Agent 在失败反馈后的反思过程中，逐步形成可解码的错误定位、原因归因和更新选择表示；其中只有部分表示对修改选择具有因果作用，而这种因果作用并不自动转化为跨任务能力提升。本文建立从内部状态形成、因果干预到持久更新收益的完整评测链。

这个表述同时容纳正结果和负结果：如果某个 Probe 很强但干预失败，可以报告其为诊断性状态；如果干预改变选择但不能提高留出表现，可以报告行为因果性与改进有效性的分离。

## 推荐实施顺序

1. 重构数据生成器，使根因、操作和反馈词面解耦。
2. 将反思输出改成结构化字段，并记录精确 Token 边界。
3. 完成跨阶段、跨层的线性 Probe 和文本对照。
4. 将反思结束单点干预扩展为时序 patching。
5. 增加 SAE 特征的注入、抑制和残差增量控制。
6. 将干预后的提案转化为独立更新工件，并运行校准、回归和留出评估。
7. 分开报告修复收益与泛化收益。
8. 在机制链稳定后，再扩展自由文本 Prompt、Skill 和 Harness 代码修改。
