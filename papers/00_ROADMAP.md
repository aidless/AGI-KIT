# 5 篇 TMLR 投稿论文路线图

目标:产出 5 篇 TMLR 级别论文,基于现有 AGI Kit 实验。

## Paper 1: Self-Critique as a First-Class Abstraction for Small Tool-Use Agents
**作者主题**:L1 反思原语
**核心 novelty**: 把 self-critique 提升为一类编程原语(不是 ad-hoc prompt)
**实验**:
  - L1 Reflector vs no-reflection baseline on 50 GAIA2-style tasks
  - Rule-only vs LLM-only vs Hybrid scoring ablation
  - Hindsight quality analysis (human-rated on 100 samples)
**目标 venue**: TMLR
**预期页数**: ~12 pages

## Paper 2: Semantic Strategy Memory with Rule-Based Meta-Control for Tool-Use Agents
**作者主题**:L2 Playbook + MetaController
**核心 novelty**: 把 strategy memory 和 meta-controller 形式化为闭包元层
**实验**:
  - Playbook size vs task success curve
  - MetaController rule-based vs LLM-based ablation
  - Stuck-detection latency reduction
**目标 venue**: TMLR
**预期页数**: ~12 pages

## Paper 3: Continual Learning Loop with A/B Safety Gate for Self-Improving Agents
**作者主题**:L3 ContinualLoop + safety gate
**核心 novelty**: 引入 A/B 安全门 + 经验回放 buffer 形式化
**实验**:
  - Multi-generation (7+ generations) evaluation
  - Safety gate false-rejection analysis
  - Buffer size vs retrain quality
**目标 venue**: TMLR
**预期页数**: ~14 pages

## Paper 4: Bounded Recursive Self-Modification in Small Language Model Agents
**作者主题**:L4 SchemaMutator + ToolFactory + PromptMutator
**核心 novelty**: 受限递归自改(sandbox + version control + eval gate)
**实验**:
  - SchemaMutator 修改阈值后系统稳定性
  - ToolFactory 真实合成工具 vs ground truth
  - PromptMutator 版本谱系追踪
**目标 venue**: TMLR
**预期页数**: ~13 pages

## Paper 5: An End-to-End Architecture for Self-Improving Tool-Use Agents on Consumer Hardware
**作者主题**:L1-L4 集成系统
**核心 novelty**: 1.7B 模型 + 完整 L1-L4 闭环在 5GB RAM 上跑通
**实验**:
  - 完整 50-100 episode 实验 + 代际曲线
  - 与静态基线对比的 ablation
  - 计算开销分析
**目标 venue**: TMLR
**预期页数**: ~16 pages (系统论文)