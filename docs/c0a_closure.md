# C0a 闭合报告（GAIA2 canonical runner 可复现性门）

> 提交范围：`eae54f5`（C0A-08 observation-transform 补丁 + 回归测试）、`2a350d7`（C0A-05 权威工具目录 dump）。
> 冻结环境：ARE `meta-agents-research-environments@7946367413129784139e785ae4c351090002a0bb`（MIT，`third_party/are.provenance.json`），模型 `qwen2.5:3b`（digest `357c53fb…`），Ollama 服务端 `NUM_PARALLEL=4, KEEP_ALIVE=5m`。
> 场景：pinned 单场景 `0144_vi1f41…` = `scenario_universe_23_qqp7ys`（fix8 历史场景，任务文本严格匹配）。

## 1. 门状态：C0a 全部闭合

| 门 | 状态 | 证据 |
| --- | --- | --- |
| C0A-01 干净安装 | ✅ | ARE 1.2.0 于干净 py3.12 venv，258 模块逐文件一致（用户执行） |
| C0A-02 provenance | ✅ | `third_party/are.provenance.json`（archive sha `df4c0200…`） |
| C0A-03 runner 导入 | ✅ | `--help` 冒烟 |
| C0A-04 场景 manifest | ✅ | `third_party/c0a04_ambiguity_manifest.json`（32 文件 × SHA-256） |
| C0A-05 工具目录 | ✅ | **权威运行时对象 dump = 101 工具**（`2a350d7` 起自动写入 `<out>/tool_catalog.json`；含 `SystemApp__wait_for_notification` 与 `RentAFlat__save_apartment`） |
| C0A-06 counter 语义 | ✅ | 缺失目标写动作/额外 wait 均失败（用户 fixture） |
| C0A-07 迭代上限 | ✅ | 修复 fixture：`finally` 块对错误步也递增 `iterations`/`planning_counter`；默认终止只查 `iterations >= max_iterations`，`total_iterations=120` 惰性（`tests/test_c0a_iteration_error_semantics.py`） |
| C0A-08 observation 保真 | ✅ | root-cause 复现 + 修复补丁 + 回归测试（`tests/test_c0a_observation_compaction.py`）；三路诊断见 §2 |
| C0A-09 user action | ✅ | 无 UserProxy 时 `send_message_to_user` 不自动生成回复（用户 fixture） |
| A12 / N1 门 | ✅ | v1 失败在 canonical runner 上复现（§2） |

## 2. 三路诊断结果（6 run，`logs/c0a08_diagnostic_summary.json` 精简版；完整产物见 `third_party/c0a08_runs_artifact.json`）

| run | 步数 | wait | save_apartment | tokens | 时长(s) |
| --- | --- | --- | --- | --- | --- |
| raw_truncate_3500 ×2 | 45/26 | 30/19 | **0** | 177K/103K | 1452/1067 |
| legacy_id_name ×2 | 40/40 | 30/31 | **0** | 161K/161K | 319/317 |
| decision_fields ×2 | 40/40 | 30/30 | **0** | 161K/161K | 307/315 |

**结论（研究结论，写入 PR）：**
1. **C0a 已闭合；v1 失败（wait-loop）在 canonical runner 上可复现**：`legacy_id_name` 两次 run 均 40 步、wait 30–31、`save_apartment=0`，A12 门通过。
2. **observation transform 未带来行为改善**：三变换失败模式逐项相同（wait 30±1、forced 8–9、save 0）→ "压缩导致空转"作为支配性原因被证伪；`decision_fields` 保留为保真与成本治理措施，其增益仅限"保真 + 成本"，不声称行为提升。
3. **机制证据**：模型从不调用 RentAFlat 读工具（从未看见公寓数据）；v1 guard 的 forced 读重定向到任务无关 app（`Calendar__get_all_tags`）。
4. **Phase 1 v2 预注册方向**：测试**任务相关候选集**（forced/fallback 按 pending goal 的 app 优先；目标 app 读/写动作完备进入候选面）与**事件依赖等待门控**（`wait_eligible()`）。
5. **工具面口径**：正则提取混入 oracle/gold 动作名（历史与当前均为 111、零差异）；权威口径 = ARE 初始化运行时对象 = **101**。今后工具面漂移以 `tool_catalog.json` 为准，正则仅调试。

## 3. 仓库内文件与外部工件

**入库**：`eae54f5` + `2a350d7`、`third_party/are.provenance.json`、`third_party/c0a04_ambiguity_manifest.json`、`third_party/c0a08_runs_artifact.json`、`docs/c0a_closure.md`（本文件）、`logs/c0a08_diagnostic_summary.json`、`tests/test_c0a_iteration_error_semantics.py`、`tests/test_c0a_observation_compaction.py`、`.gitignore`（`logs/c0a08_*/`）。

**外置工件（附 PR/release）**：`c0a08_runs.zip`（178,867 bytes，sha256 `795cb80827584d04f9571c2dedf1090e3d92d2c8d60012ca938834246b5fd1c4`）——6 个完整 run 目录（environment_manifest/llm_progress/canonical_summary/trace/tool_catalog/stdout/stderr），生成 commit `2a350d7`，模型 digest `357c53fb…`，服务端 env 见 `third_party/c0a08_runs_artifact.json`。

## 4. 遗留与后续

- N1 分支 (a) 已通过；不需要分支 (b)。
- 基线冻结前其余事项（A02–A08、A11、analysis plan 冻结、dev/holdout 隔离）属于 0B/0C 范围，未在本提交内。
- `eae54f5`/`2a350d7` 未推送，等待与本报告合并为单一 PR。
