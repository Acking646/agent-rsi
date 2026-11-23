# ModularRSI 数据

本仓库固定使用 `IQuestLab/ModularRSI_2000_Instances` 的 revision `610b70d487a17c666e67f3c75dc06bf7a34fd569`。`data/manifests/` 保存官方 Terminal-Bench 域和 SWE-bench 域的清单及 revision；完整任务数据不进入 Git。

数据根目录应具有以下结构：

```text
ModularRSI_2000_Instances/
├── tb/
│   ├── manifest.json
│   └── tasks/<task-name>/task.toml
└── swe/
    ├── manifest.json
    └── tasks/<task-name>/task.toml
```

`rsi.datasets.modularrsi_splits()` 只使用每个域的 120 个官方 train 任务，以固定 seed 洗牌后划分为 80 个 evolve、20 个 dev 和 20 个 held-out；pilot 使用 evolve 的前 30 个任务。任务名称、角色和顺序可由 selection digest 审计。官方 pool 不参与 Harness 更新门控，可留作额外外部评测。

Windows 无法完整检出数据中带冒号的路径，因此完整任务数据应放在 Linux 文件系统。`scripts/bootstrap_modularrsi.sh` 可按固定 revision 下载并对照本仓库清单核验。代码运行入口见根目录 `README.md`。