Agent RSI 相关论文整理

来源文档：D:\jhli\agent论文\研究方案\Agent_RSI方法实现.docx
整理日期：2026-11-26

一、最值得优先精读
1. State over Tokens（Levy et al., 2026）
   对应：反思 Token 作为状态载体、Token 前缀到内部计算状态的理论切入点。
2. Recursive Introspection: Teaching Language Model Agents How to Self-Improve（RISE, 2024）
   对应：任务执行—反馈—自我修正的递归闭环。
3. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning（2025/ICLR 2026）
   对应：从轨迹和反馈生成自然语言反思，再优化 Prompt；可作为外部更新基线。
4. ReasoningBank（2025/ICLR 2026）
   对应：从成功/失败经历提炼可检索推理策略并写回长期记忆；可作为 Memory 更新基线。
5. Gödel Agent / Darwin Gödel Machine
   对应：Agent 自修改逻辑、模块、代码与开放式候选搜索；用于界定 RSI 系统层范围。
6. Unfaithful CoT / Reasoning Models Don't Always Say What They Think / Thought Anchors
   对应：反思文本不等于真实内部计算；支持从文本分析转向激活与因果干预。
7. Towards Best Practices of Activation Patching
   对应：Activation Patching 的对齐、指标、对照和实验设计。
8. Sparse Autoencoders Find Highly Interpretable Features / Do SAEs Identify Reasoning Features?
   对应：SAE 特征提取、稀疏特征解释及其在推理状态定位中的局限。

二、目录
01_RSI_Agent自我改进
- RISE_Recursive_Introspection_2407.18219.pdf
- Reflexion_2303.11366.pdf
- Self_Refine_2303.17651.pdf
- Godel_Agent_2410.04444.pdf
- Darwin_Godel_Machine_2505.22954.pdf
- GEPA_Reflective_Prompt_Evolution_2507.19457.pdf
- ReasoningBank_2509.25140.pdf
- Voyager_2305.16291.pdf

02_内部状态与反思忠实性
- State_over_Tokens_2512.12777.pdf
- Turpin_Unfaithful_CoT_2305.04388.pdf
- Reasoning_Models_Dont_Always_Say_2505.05410.pdf
- Thought_Anchors_2506.19143.pdf
- When_CoT_Fails_2604.23351.pdf

03_因果机制解释
- Activation_Patching_Best_Practices_2309.16042.pdf
- ROME_Locating_Editing_Factual_Associations_2202.05262.pdf

04_SAE与特征分析
- Sparse_Autoencoders_Interpretable_Features_2309.08600.pdf
- Scaling_Evaluating_SAEs_2406.04093.pdf
- Do_SAEs_Identify_Reasoning_Features_2601.05679.pdf
- Step_Level_Sparse_Autoencoder_2603.03031.pdf

三、与本方案的映射
A. RSI 外部闭环：RISE、Reflexion、Self-Refine、GEPA、ReasoningBank、Voyager。
B. Agent 自修改边界：Gödel Agent、Darwin Gödel Machine。
C. 内部状态与反思忠实性：State over Tokens、Turpin、Reasoning Models Don't Always Say、Thought Anchors、When CoT Fails。
D. 因果验证：Activation Patching Best Practices、ROME、Causal Abstraction（建议后续补充）。
E. SAE/特征：Sparse Autoencoders、Scaling SAEs、Do SAEs Identify Reasoning Features、Step-Level SAE。

四、建议阅读顺序
RISE → GEPA → ReasoningBank → State over Tokens → Turpin → Thought Anchors → Activation Patching → SAE 系列 → Gödel Agent → Darwin Gödel Machine。

说明：部分 ICLR/OpenReview 页面下载受网络或反爬限制；已优先使用 arXiv 对应版本。个别 2026 年论文按当前可访问的 arXiv 版本保存，正式引用前请核对最终会议版本和 DOI。
