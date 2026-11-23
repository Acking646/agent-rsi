# 上游代码与本系统接口

| 上游资产 | 固定位置 | 在本系统中的作用 |
|---|---|---|
| ModularRSI / Harbor | `third_party/ModularRSI` | 任务生命周期、Docker 环境、官方 verifier、五模块研究对象和 ModularRSI 数据格式 |
| SWE-bench Harbor adapter | 外部转换工具 | 将 SWE-bench 实例转换为 Harbor 本地任务；本 runner 通过 `DATASET_PATH` 接收转换结果 |
| GEPA 参考文件 | `upstream/reference_files/` | 后续系统基线接入时对照其 reflective dataset、candidate proposal 与 evaluation batch 接口 |
| DGM / Gödel Agent 参考文件 | `upstream/reference_files/` | 对照版本谱系、候选选择和持续更新流程，不直接参与当前正式 runner |
| ReasoningBank 参考文件 | `upstream/reference_files/` | 后续增加检索记忆基线时对照轨迹格式化、筛选和检索接口 |

本项目的执行与计分只通过固定版本的 Harbor 完成。上游参考文件用于方法核对，不以行为近似冒充官方基线。接入任何完整基线时，应保持任务清单、rollout 数、模型、官方 verifier、预算和随机种子一致，并单独记录上游仓库提交与依赖环境。