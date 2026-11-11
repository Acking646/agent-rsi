# Agent RSI 研究代码包

原始研究方案位于 `docs/Agent_RSI方法实现(1).docx`。代码包含两组实验。机制实验定位反思激活并干预更新决策。基线套件比较不同方法如何利用反馈、记忆和版本管理。

## 快速运行

Python 3.10 或更新版本。以下命令在解压后的 `Agent_RSI_Research_Package` 目录执行。每次运行需要一个新的输出目录。

```bash
python -m pip install -r requirements.txt
python -m rsi --out runs/mechanism-toy
python -m rsi.baseline_suite --config configs/pilot.json --out runs/baselines-toy
```

`toy` 通过预设的合成激活和任务标签检查数据管线。演化分数和干预数值只反映程序设计。购物篮计算任务是脚本化环境，包含缺参数、报价过期、漏验算和成功四种条件。所有方法共享这个环境、任务顺序、外部验证器和操作集合。

## 目录

| 位置 | 内容 |
|---|---|
| `docs/Agent_RSI方法实现(1).docx` | 用户提供的研究方案原件 |
| `docs/mechanism_experiment.md` | §2.1–§2.7 与源码对应关系、干预方法、结果字段 |
| `docs/baseline_protocol.md` | 每个基线的具体实现、论文原版与简化实现的差别、评估口径 |
| `rsi/__main__.py` | 机制实验入口 |
| `rsi/backends.py`、`rsi/analysis.py`、`rsi/sae.py` | 模型后端、Token 对齐、Probe、干预与可选 SAE |
| `rsi/environment.py` | 任务、验证器、更新工件与写回验收 |
| `rsi/baselines.py` | 机制实验的固定选择、无反思与近邻检索对照 |
| `rsi/baseline_suite.py` | 八种方法的共享任务流评估 |
| `configs/pilot.json` | 任务流、样本数、搜索候选数、检索阈值 |
| `docs/upstream_code_map.md` | 上游函数到本项目接口的对照 |
| `upstream/reference_files/` | 四个作者仓库的关键原始源码和许可证 |
| `upstream/reference_files_manifest.json` | 逐文件来源和内容哈希 |
| `upstream/sources.json` | 论文链接、仓库来源、借鉴位置及核验状态 |
| `upstream/fetch.py` | 下载公开 Git 仓库的源码快照，记录提交、SHA-256 和许可证文件 |
| `examples/` | 默认配置下验证通过的两份模拟运行摘要 |

## 换成真实模型

两个入口共用冻结的 Hugging Face 模型后端。模型需要提供 Llama/Qwen 风格的 decoder blocks。下面的层号从 0 起算，按模型深度修改。用本地路径可避免运行时下载模型。

```bash
python -m pip install -r requirements-hf.txt
python -m rsi --backend hf --model /path/to/checkpoint --device cuda --layers 2 6 10 --cases-per-split 3 --out runs/mechanism-hf
python -m rsi.baseline_suite --backend hf --model /path/to/checkpoint --device cuda --layers 2 --out runs/baselines-hf
```

如 tokenizer 无法将 A–D 候选编码成相同前缀后的独立单 Token，后端会报出具体标签。可在 `backends.py` 中调整决策后缀或标签，并重新校验。当前环境验证了模拟后端的两组实验，真实 checkpoint 和 SAE 仍需在有 PyTorch 的环境运行验证。

## 上游源码

包内附 DGM、GEPA、ReasoningBank、Gödel Agent 的关键原始文件和许可证。整库 ZIP 下载超时，逐文件下载成功。原文件从公开 `main` 分支下载，附 SHA-256 和 Git blob 哈希，未记录未核实的提交号。受控实验运行本项目的独立实现。逐项来源和接口见 `docs/upstream_code_map.md`。

```bash
python upstream/fetch.py --list
python upstream/fetch.py dgm
python upstream/fetch.py gepa
```

默认解析仓库 HEAD 并下载解析出的确切提交。使用输出清单中的 `commit` 配合 `--commit` 复现。脚本只下载源码快照和仓库内已有的许可证文件，不执行安装脚本。仓库尚未确认的条目可在确认作者链接后用 `--repo` 指定。
