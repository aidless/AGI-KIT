# AGI 研究完整套装 (AGI Research Kit)

[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)  [![CI](https://github.com/aidless/AGI-KIT/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/AGI-KIT/actions/workflows/ci.yml)

> 本地小机器 + Agent 研究 + AGI 兴趣 - 一站式研究工作台

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

## 目录结构

```
agi-research-kit/
  models/                 # 模型权重(Qwen3-1.7B 等)
  data/                   # 数据集(GAIA2 / smoltalk 等)
  experiments/            # 实验代码
    hello_agent.py        # 最小可跑 Tool-Use Agent(ReAct)
    eval_gaia2.py         # GAIA / GAIA2 评测脚手架
  logs/                   # 评测结果（运行日志不入库，见 .gitignore）
  scripts/
    setup_env.ps1         # Python + Ollama 安装
    download_agi_set.ps1  # 30 GB 模型下载
  python/                 # Python 安装包缓存
  .venv/                  # Python 虚拟环境
```

## 一键安装(在 PowerShell 里)

```powershell
cd "F:\agent to AGI\agi-research-kit"
.\scripts\setup_env.ps1         # 装 Python 3.12 + Ollama + Python 包
.\scripts\download_agi_set.ps1 -HFToken "hf_你的token"   # 下载模型
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

[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)  [![CI](https://github.com/aidless/AGI-KIT/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/AGI-KIT/actions/workflows/ci.yml/badge.svg)
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