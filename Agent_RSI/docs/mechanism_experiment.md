# Agent RSI 内部改进状态实验框架

以《Agent_RSI方法实现(1).docx》§2.1–§2.7 为设计依据。首版用可复现的购物篮计算任务，把环境反馈、反思 Token、模型激活、更新提案和实际写回关联起来。Python 3.10+。

## 运行

解压后进入 `agent_rsi_framework`。输出目录由程序创建，每次运行使用新路径。

```bash
python -m pip install -r requirements.txt
python -m rsi --out runs/toy
```

`toy` 后端刻意把功能标签写入合成激活，用来验证实验管线。它的准确率、Probe 分数和干预效应仅是程序检查结果。

用真实本地模型运行。选择 `model.model.layers` 下包含 `self_attn` 和 `mlp` 的 Llama/Qwen 类 decoder-only 模型。下面的层号按 0 开始，根据模型深度修改。

```bash
python -m pip install -r requirements-hf.txt
python -m rsi --backend hf --model /path/to/local/checkpoint --device cuda \
  --layers 2 6 10 --reflection-tokens 48 --cases-per-split 3 \
  --out runs/local-pilot
```

默认根据 dev 集线性 Probe 的 macro F1 选择残差流层。用 `--site 6 mlp` 可预先固定干预点，目标层也必须列在 `--layers` 中。模型权重和 tokenizer 来源写入 `manifest.json`，正式实验可用 `--revision` 固定版本。代码对 A–D 读出检查单 Token 条件。遇到 tokenizer 不兼容时，在 `backends.py` 中调整决策后缀或选项字符串，保持所有标签均可在同一前缀后单 Token 读出。

可选 SAE 使用 `--sae-width 128 --sae-steps 400`。它只在选定站点和反思结束锚点的训练样本上拟合，用测试样本记录重建误差。小样本默认配置适合接通代码，正式解释前需要扩大训练语料、检查稀疏性及跨种子稳定性。SAE 特征方向依据训练集类别均值差与解码器列向量的对齐选择，干预保留原始重建残差。

## Word 方案与代码位置

| 文档内容 | 代码入口 | 当前实现 |
|---|---|---|
| §2.1 执行、反馈、反思、更新 | `environment.py`、`__main__.py` | 四种受控任务条件，外部验证器记录结果、定位、根因、推荐操作 |
| §2.2 逐 Token 分阶段采集 | `backends.py` | 反馈结束、逐反思 Token、反思结束锚点、更新决策位置，采集残差流、Attention 输出和 MLP 输出 |
| §2.3 解码与 SAE | `analysis.py`、`sae.py` | 四个标签的线性 Probe、TF-IDF 对照、可选 SAE |
| §2.4 状态形成 | `analysis.py` | 比较反思长度百分位上的 Probe 表现，保留精确 Token 索引供后续分析 |
| §2.5 因果干预 | `analysis.py` | 同任务配对替换、自替换、等范数随机扰动、方向消融和注入、可选 SAE 特征替换与消融 |
| §2.6–2.7 工件与跨轮状态 | `environment.py`、`__main__.py` | 提案转为结构化规则，校准集验收，保留旧版本、候选 diff、实际 diff、下一轮输入及留出表现 |

环境有四种条件：成功、缺少必要输入、目录报价过期、遗漏验算。执行轨迹由 Python 环境生成，真实模型后端负责反思及更新选择。当前更新工件是三种声明式规则及保留选项，分别映射到 Prompt、Memory 和 Skill。规则由 Python 执行器消费，并随下一轮输入传入模型。当前实现属于固定选择集的受控试验，自由生成 Prompt/Memory/Skill 工件和 LLM 驱动的任务执行需要扩展环境接口。

HF 后端先贪心生成反思，再以相同反思 Token 重放完整因果前缀。反思末尾附加固定标记，补丁作用于该标记的最后一个 Token。决策读取发生在后续固定后缀的最后位置。重放关闭 KV cache，控制组和干预组复用目标前缀，补丁改变指定模块的输出。每个阶段的边界由 Token ID 构造并持久化。残差流此处指每个 decoder block 的输出，Attention/MLP 指对应子模块输出。

## 基线代码怎么借鉴

下表前五项对应 Word 中的工作。仓库链接用于定位原作者实现。当前框架是独立实现，实验报告应标为受控变体。

| 工作 | 适合复用的部分 | 当前框架对应位置 |
|---|---|---|
| [RISE](https://arxiv.org/abs/2407.18219) | 用任务、前次尝试、反馈、下一次修正组织多轮数据，同任务前后评估 | `episodes.jsonl`、`stream.jsonl`。训练循环可留到研究参数更新时再接入 |
| [Gödel Agent](https://github.com/Arvid-pku/Godel_Agent) | 将更新提出、执行和结果检验拆开 | `proposal()` 与 `assess_update()`，当前更新动作受限于声明式规则 |
| [DGM](https://github.com/jennyzzt/dgm) | 版本档案、父子关系、补丁和评估记录。本文的研究问题可先用线性版本日志 | `stream.jsonl` 中的前后版本、候选与实际 diff |
| [GEPA](https://github.com/gepa-ai/gepa) | 轨迹与反馈如何交给反思模块、如何产生候选 Prompt 并验证 | `context_for()`、`assess_update()`，作为文本反馈和 Prompt 更新路线 |
| [ReasoningBank](https://github.com/google-research/reasoning-bank) | 成功与失败轨迹的经验提炼、检索和写回接口 | `baselines.py` 中的轻量轨迹检索对照仅复用旧提案，适合先与真实实现的记忆写回区分 |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) / [Zhang & Nanda](https://arxiv.org/abs/2309.16042) | 激活缓存与 Hook 替换模式，比较 clean、donor、patched 和扰动对照 | `backends.py`、`analysis.py`，本版使用原生 PyTorch Hook |
| [SAELens](https://github.com/decoderesearch/SAELens) | 用更充分训练的 SAE 替换本地小型实现，复用稀疏特征读取与干预接口 | `sae.py` |

可直接运行的行为对照包括总选 A、同模型跳过反思、训练轨迹 TF-IDF 最近邻复用旧提案，以及完整反思。线性 Probe 另有任务文本、验证器反馈、反思文本三个 TF-IDF 对照。机制控制组包括同一激活回填和等范数随机扰动。四种候选操作的 logit 顺序是 manifest 中的 A、B、C、D。

## 输出和检查顺序

| 输出 | 查看内容 |
|---|---|
| `manifest.json` | 模型、解码设置、任务组划分、更新代码映射 |
| `episodes.jsonl` 与 `activations/*.npz` | 从 run_id、Token 绝对位置追溯到各层和组件激活。NPZ 的 token_positions 给出矩阵行对应的位置 |
| `baselines.json` 与 `probes.json` | 行为基线与分阶段解码结果。使用 train 拟合、dev 选站点、test 报告 |
| `interventions.jsonl` | clean 与干预的候选 logits、操作变化、donor/target 记录和锚点 |
| `intervention_updates.jsonl` | 把两臂提案从同一初始版本独立送入校准验收，记录工件差异和留出任务表现 |
| `stream.jsonl` | 跨任务连续使用同一个更新状态，记录接受或拒绝、版本变化和实际重输入 |
| `summary.json` | 总览、配对 margin 改变量、按 task_id 聚类 bootstrap 的区间及留出表现 |

默认案例中，同一个 task_id 的多个条件共同进入同一数据切分。干预中保持目标序列固定，在同任务的另一失败条件取 donor 激活。donor 条件的任务字段、轨迹和反馈可以同时变化，因此该设计检验的是候选站点对读出的因果贡献。要识别反馈在路径中的独立作用，可进一步增加固定任务和轨迹、仅改变外部反馈的配对，加入同义反馈和随机反馈对照。

四类环境标签在这个受控任务中有明确映射，不能从其 Probe 结果推断模型内部存在四种相互独立的状态。正式评估时应增加多种根因对应同一操作、同一根因因任务上下文而采用不同操作的样本。正式因果效应应同时检查行为方向、随机对照以及验收后的任务收益。

## 首版验证

在当前环境完成默认 toy 端到端运行。96 条轨迹，32 个测试配对，每个配对包含 5 个干预组。自替换的 logit 差变化为 0，预设 donor 替换会改变提案，后续独立回放会产生可追溯的工件变化。三个声明式更新分别通过校准并在下一轮重输入。真实模型和可选 SAE 需要 Torch，当前环境未进行运行验证。
