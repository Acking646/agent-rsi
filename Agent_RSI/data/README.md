# ModularRSI 官方代码与数据

## 当前本地状态

官方 ModularRSI 代码通过 Git submodule 固定在 `third_party/ModularRSI`，提交为 `ed1e545bc60f050b404474e51ee18021777a0e23`。仓库中可直接看到五个可演化模块：

- `agent_loop`
- `tools`
- `observation`
- `context_mgmt`
- `verification`，对应本项目所称的 Completion Detection

`tool_helper` 是工具辅助库，不是第六个独立演化模块。

完整数据没有直接复制到主 Git 历史。仓库保存了固定 revision `610b70d487a17c666e67f3c75dc06bf7a34fd569` 的两份官方 manifest：

- `manifests/tb_manifest.json`：1000题，120 train，880 pool；
- `manifests/swe_manifest.json`：1000题，120 train，880 pool。

## 为什么 Windows 不能完整检出

数据中存在路径片段 `2024-01-15T10:00:00Z`。冒号在 Windows NTFS 文件名中非法，因此官方数据 Git 仓库在 Windows checkout 时会失败。数据必须在 Linux 文件系统中检出；这与磁盘空间、Git LFS和访问权限无关。

## 学校 Linux 机器下载

从主仓库根目录运行：

```bash
bash Agent_RSI/scripts/bootstrap_modularrsi.sh
```

默认下载到：

```text
Agent_RSI/.local/ModularRSI_2000_Instances
```

也可以指定独立高速数据盘：

```bash
MODULARRSI_DATA_DIR=/data/agent-rsi/ModularRSI_2000_Instances \
  bash Agent_RSI/scripts/bootstrap_modularrsi.sh
```

脚本会固定代码与数据提交，并用仓库内 manifest 对下载结果做字节级核对。

## 与当前代码的关系

`python -m rsi` 仍然执行购物篮 toy 环境，用于检查 Probe、Activation Patching、更新验收和版本写回的数据管线。它不会自动运行 ModularRSI。

真实实验应以 `third_party/ModularRSI` 中的 Harbor 和五模块 Terminus-2 Modular Harness 为执行系统，再把本项目的 Qwen3 白盒 backend、激活采集、Probe和因果干预接到模型调用边界。数据下载只是第一步，并不等于适配器已经完成。