# 原始源码与实验接口

`upstream/reference_files/` 保存四个作者仓库中选定的原始文件和各自许可证。`reference_files_manifest.json` 逐文件记录网页地址、长度、SHA-256 和 Git blob SHA-1。网页使用公开 `main` 分支，Git 提交号未能确认。Git blob 哈希标识文件内容。原始文件用于阅读和后续接入，运行本包实验只调用 `rsi/`。

| 上游 | 随包原始文件及入口 | 当前实验对应点 |
|---|---|---|
| DGM | `DGM_outer.py` 的 `choose_selfimproves()`、`update_archive()`，`self_improve_step.py` 的 `self_improve()`，以及 `coding_agent.py` | `baseline_suite.py` 的 `select_parent()` 对照官方 `score_child_prop` 权重，`Version` 与 `search()` 负责版本档案及评估。当前变异对象为三种规则 |
| GEPA | `src/gepa/core/adapter.py` 的 `GEPAAdapter.evaluate()`、`make_reflective_dataset()` 和 `EvaluationBatch`，`src/gepa/api.py` 的 `optimize()` | 当前用 `reflection_decision()` 和 `search()` 做声明式候选比较。若接完整 GEPA，可把同一环境的轨迹、得分及反馈分别映射到 `EvaluationBatch` 和反思数据集 |
| ReasoningBank | `WebArena/memory_management.py` 的 `select_memory()`、`screening()`，`WebArena/induce_memory.py` 的 `format_trajectory()`、`format_examples()` | 对照 `Memory`、`memory_key()` 和 `retrieve()`。原实现有嵌入及缓存，受控基线使用哈希向量，保留有验证结果的重试经验 |
| Gödel Agent | `src/agent_module.py` 的 `Agent.evolve()`、`action_adjust_logic()`、`action_evaluate_on_task()`，以及 `src/logic.py` | 对照 `godel_sequential` 的提出、重试验证和持久化。当前环境使用声明式规则 |
| RISE | 检查了论文摘要和 HTML 版，未核验到专用作者实现 | `rise_retry` 单独测试同一任务中的反馈驱动重试，沿用冻结模型 |
| mi4afa | 核对了论文指向的公开仓库，未确认可再分发许可证 | 本包的 Probe 和 logit 干预继续使用独立实现 |

## 许可证

DGM 与 ReasoningBank 为 Apache-2.0，GEPA 与 Gödel Agent 为 MIT。许可证与各自源码并置。四组原始文件均按下载字节保留。选定文件不构成完整的上游 Python 项目，运行上游完整方法时使用其仓库和依赖配置。

## 接入顺序

1. 先用 `python -m rsi.baseline_suite --out runs/baselines-toy` 得到相同任务流上的审计日志。
2. 若只替换基线实现，保留 `environment.execute()` 的验证器和原有校准／留出分割，替换相应方法内部的提案或检索逻辑。
3. 若接完整论文实现，独立配置其环境和依赖，并记录完整源码提交、模型和数据版本。分开报告受控任务与论文环境的结果。
4. 做机制比较时，保持候选操作读出、Token 锚点和运行顺序一致。替换基线后重新检查分组切分和随机干预对照。
