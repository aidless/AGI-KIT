# AGI 研究完整套装 (AGI Research Kit)

[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)  [![CI](https://github.com/aidless/AGI-KIT/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/AGI-KIT/actions/workflows/ci.yml)

> 面向 CPU 小机器的 tool-use Agent 研究套件：反思回路、策略记忆、持续学习安全门与有界 schema 变更。

AGI Research Kit is a CPU-oriented research prototype for tool-use
agents with reflection, semantic strategy memory, a continual-learning
safety gate, and bounded schema mutation.

**Publication status:** the unified manuscript is a submission draft
and has not been submitted. See `papers/preprint_unified_en.md`,
`papers/00_INDEX_en.md`, and `papers/PUBLISHING.md`. The historical
three-run result is descriptive only; do not reuse the superseded
`p<0.01`, 30% static-baseline, or internal 4.5-score claims.

The current controlled L1 evaluation uses a fixed 30-task manifest
(20 held-out test tasks), identical prompts/tools/budgets/seeds across
arms, raw JSONL traces, and exact paired analysis. It demonstrates one
auditable tool-evidence correction (19/20 Static versus 20/20 L1) but
does not support a general superiority claim from one discordant pair.

The source code is licensed under the Apache License 2.0. Model weights
and external datasets retain their own licenses.

## 测试与复算

### 单元测试（2026-11-01 实测）

```bash
python -m pytest tests/unit -q     # 43 passed（2026-11-01 基线）
```

### 受控 L1 配对分析复算

README 顶部"19/20 Static 对 20/20 L1"的**唯一不一致对**来自
`logs/controlled_arithmetic/run-llama3b-20260802/`。复算：

```bash
python scripts/analyze_controlled_arithmetic.py \
    logs/controlled_arithmetic/run-llama3b-20260802
# 输出：n_pairs=20, l1_wins=1, l1_losses=0, discordant_pairs=1,
#       l1_activations=1, two_sided_exact_mcnemar_p=1.0
```

即 19/20 (Static) vs 20/20 (L1)，仅 1 对不一致，且 L1 干预激活 1 次——这是
可审计的修正（可归因到 reflection-retry 激活），但 McNemar p=1.0，**不构成
L1 优于 Static 的一般性声明**（README 顶部已如此限定）。

各受控运行逐对复算（2026-11-01 实测）：

| 运行 | n_pairs | L1 wins/losses | 不一致对 | L1 激活 | 精确 McNemar p |
|---|---:|---:|---:|---:|---:|
| `heldout-qwen25-20260802T1223` | 20 | 0/0 | 0 | 0 | 1.0 |
| `run-llama3b-20260802` | 20 | 1/0 | 1 | 1 | 1.0 |
| `run-qwen0.6b-20260802` | 20 | 1/4 | 5 | 0 | 0.375 |
| `smoke-llama-20260802` | 2 | 0/0 | 0 | 0 | 1.0 |
| `smoke-llama-v3-20260802` | 2 | 0/0 | 0 | 0 | 1.0 |

### 提交前预检

```bash
python scripts/preflight_submission.py   # 提交前完整性预检（干净 worktree 下）
```

## 证明什么 / 不证明什么

**证明什么**：
- 受控 L1 评测：固定 30 任务清单（20 held-out 测试任务）、arm 间
  prompt/tools/budget/seeds 匹配、原始 JSONL trace、精确配对分析，能产出一组
  可审计的逐对修正（`scripts/analyze_controlled_arithmetic.py` 复算）。
- 安全门 12/12、SchemaMutator 18/18 阻断 + 12/12 合法控制、自我进化环
  10 轮的保守拒绝与 delta=0.0 提升，均为**可复算的工程性质**（见
  `papers/00_INDEX_en.md` Evidence Summary）。

**不证明什么**：
- 从**一个**不一致对（19/20 vs 20/20，McNemar p=1.0）**不能**推广出
  "L1 反思验证普遍优于 Static"——README 顶部已如此声明，且无一般性优越
  声明可支撑。
- 历史三跑结果（62.5% / 62.5% / 56.2%）是**描述性**的：seed 控制与产物
  溯源不完整，不得作为能力声明复用（README 顶部已警示废弃 `p<0.01`、
  30% static-baseline、内部 4.5-score 声明）。
- 评测饱和（77.8% final-emission 对每种配置）说明该消融**不区分**组件效应。
- 本仓库为**研究原型**，非生产系统：模型权重与外部数据集保留各自许可证。

## 目录结构

```
AGI-KIT/
├── src/agi_kit/          # 可安装的包：agents / tools / rag / llms / evals
│   ├── agents/           # ReAct、plan-execute 等 agent 实现
│   ├── tools/            # 工具实现
│   ├── rag/              # 检索层
│   └── loop.py, reflect.py, meta.py, strategy_miner.py  # agent 循环 + 反思 + 策略记忆
├── experiments/          # 约 36 个可运行的 study / eval 脚本（不属于包）
│   ├── hello_agent.py    # 最小可跑 Tool-Use Agent(ReAct)
│   └── eval_gaia2.py     # GAIA / GAIA2 评测脚手架
├── tests/unit/           # 单元测试（make test）
├── papers/               # 稿件、图、TMLR 投稿包
├── configs/              # 实验配置
├── data/                 # 数据集与本地模型产物（体积大，约 400 MB，已随仓库提交）
├── logs/                 # 评测输出（大体积 trace / 采样文件由 .gitignore 排除）
├── Makefile, pyproject.toml  # 打包入口：make dev / make test / make lint / make serve
└── scripts/              # 辅助与构建脚本（含 Windows 用的 setup_env.ps1 / download_agi_set.ps1）
```

## 一键安装(在 PowerShell 里)

```powershell
cd path\to\AGI-KIT      # 你的克隆目录（不要用别人机器上的绝对路径）
.\scripts\setup_env.ps1         # 装 Python 3.12 + Ollama + Python 包
.\scripts\download_agi_set.ps1 -HFToken "hf_你的token"   # 下载模型
```

安装包本身、跑测试与静态检查（跨平台，不依赖 PowerShell）：

```bash
make dev      # 可编辑安装 + 开发依赖
make test     # 单元测试
make lint     # ruff / black / isort / mypy
```

模型清单(约 30 GB,按 `tier` 可选):

| 模型 | 大小 | 用途 |
|---|---:|---|
| Qwen3-1.7B | 3.8 GB | 主力 Agent LLM |
| Qwen3-0.6B | 1.4 GB | 最小通用 LLM, CPU 可跑 |
| SmolLM3-3B | 5.7 GB | HF 官方 3B, 推理强 |
| SmolVLM-256M-Instruct | 0.5 GB | 视觉 / GUI Agent |
| Qwen3-ASR-0.6B | 1.4 GB | 语音识别 |
| SmolLM2-135M / 360M | 0.5 GB | 极小模型研究 |
| Qwen3-4B-Instruct-2507 | 8 GB | 可选, 4B tool-use 主力 |
| Qwen-AgentWorld-35B-A3B | 70 GB | 可选, AGI 前沿 MoE |

## 跑第一个 Agent

```powershell
# 1. 启动 Ollama 后台(它会跑在 http://127.0.0.1:11434)
ollama serve

# 2. 拉模型(首次,会下载到 C:\Users\<你>\.ollama)
# 完整复现实验前必须先拉取下列四个 Ollama 模型。
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
ollama pull qwen2.5:3b
ollama pull llama3.2:1b

# 3. 跑 ReAct Agent
.venv\Scripts\python.exe experiments\hello_agent.py --backend ollama --model qwen3:1.7b

# 或者纯 HF transformers(走本地 models/ 目录)
.venv\Scripts\python.exe experiments\hello_agent.py --backend transformers --model models\Qwen\Qwen3-1.7B
```

## 跑 GAIA2 评测

```powershell
# 单题调试
.venv\Scripts\python.exe experiments\eval_gaia2.py --single "用 calculator 算 2**10,然后 echo 答案"

# 跑 GAIA2 评测集前 20 题(自动写日志到 logs/)
.venv\Scripts\python.exe experiments\eval_gaia2.py --subset meta-agents-research-environments/gaia2 --limit 20
```

## 下一步研究路径建议

- **Week 1**: 跑通 hello_agent.py,理解 ReAct 循环
- **Week 2**: 跑 GAIA2 评测,看 Qwen3-1.7B 准确率
- **Week 3**: 加新工具(浏览器 / RAG / 代码执行),看能力变化
- **Week 4**: 对比 SmolLM3-3B vs Qwen3-1.7B,分析小模型差距
- **进阶**: 学 [smol-course](https://huggingface.co/learn/smol-course) 自己 SFT
- **进阶**: 学 [agents-course](https://huggingface.co/learn/agents-course) 高级 Agent

## 资源链接

- [Hugging Face Learn](https://huggingface.co/learn)
- [smol-course](https://huggingface.co/learn/smol-course)
- [agents-course](https://huggingface.co/learn/agents-course)
- [GAIA Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

## 故障排查

| 问题 | 解决 |
|---|---|
| `ollama` 命令找不到 | 重启 PowerShell 让 PATH 生效,或用绝对路径 `C:\Program Files\Ollama\ollama.exe` |
| 模型下载慢 | 设 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像 |
| 显存不够 | 用 GGUF Q4 量化版,或换 Qwen3-0.6B |
| Agent 输出非法 JSON | 改用更强的模型,或在 `hello_agent.py` 调整 `parse_action` |

## 文件清单

- `scripts/download_agi_set.ps1` - 30 GB 模型下载(支持断点续传)
- `scripts/setup_env.ps1` - Python + Ollama + 依赖一键安装
- `experiments/hello_agent.py` - ReAct Agent 最小实现
- `experiments/eval_gaia2.py` - GAIA2 评测脚手架
- `README.md` - 本文件