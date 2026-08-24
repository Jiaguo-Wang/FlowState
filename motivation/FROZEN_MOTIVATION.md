# Frozen Motivation Snapshot

**Freeze date: 2026-08-24**

本文档冻结 FlowState Motivation 阶段的现有证据、索引和解释边界。冻结范围包括 WP1 recurrent-state capacity、WP2 replay/recovery cost、WP3 workflow-dependent checkpoint value，以及支撑这些结论的 SGLang code-path study、runtime validation 和 controlled formal results。

## Frozen result set

- WP1 checkpoint capacity characterization：[`artifacts/checkpoint_size_20260819/`](artifacts/checkpoint_size_20260819/)
- Runtime/instrumentation gap validation：[`artifacts/runtime_validation_gap_replay_20260819/`](artifacts/runtime_validation_gap_replay_20260819/)
- WP2 replay cost characterization：[`artifacts/replay_cost_20260819/`](artifacts/replay_cost_20260819/)
- WP3 workflow boundary value characterization：[`artifacts/workflow_boundary_value_20260819/`](artifacts/workflow_boundary_value_20260819/)
- WP3 direct eviction causal validation：[`artifacts/wp3a_direct_eviction_20260819/`](artifacts/wp3a_direct_eviction_20260819/)
- Current frozen controlled formal result：[`artifacts/wp3b_gate_20260820/formal_k4_20260821_102355/`](artifacts/wp3b_gate_20260820/formal_k4_20260821_102355/)
- SGLang recurrent-state code-path study：[`docs/sglang_recurrent_state_path.md`](docs/sglang_recurrent_state_path.md)

## Freeze rules

1. **Existing formal results must not be overwritten.** 已有 formal summary、CSV、JSON、JSONL、logs、plots、metadata、patches、findings 和实验脚本应保持原样。
2. **Future experiments must create new timestamped directories.** 新实验、复跑、扩展配置、修正分析或替代 policy 必须写入新的时间戳目录，不得复用或覆盖现有 artifact 目录。
3. **Raw artifacts remain the source of truth.** 论文文字、README、索引和后续汇总必须引用 raw CSV、JSON、JSONL、logs、runtime evidence 及其对应冻结分析；索引文档不替代原始证据。
4. 如发现需要更正或补充的解释，应新增带日期的 addendum 或新 artifact 目录，并明确引用原始结果；不得静默修改冻结数值。
5. 新目录应保留实验环境、输入、命令、有效性门、raw outputs、分析脚本和结果摘要之间的可追溯关系。

## Interpretation boundary

冻结的 Motivation 结果支持以下论证：recurrent checkpoints 具有可观的容量成本；physical prefix frontier `H` 与 executable state frontier `E` 的错位会产生 recovery gap `G = H - E` 和真实恢复代价；checkpoint 的未来价值依赖 workflow 位置。

这些结果是特定 SGLang、模型、硬件和 controlled workload 下的 evidence。它们**不表示完整 FlowState 系统、通用在线选择策略或生产实现已经完成**，也不应被外推为跨模型、硬件、并发度和 workload 的无条件结论。

论文视角的索引见 [`README.md`](README.md)，实验级索引见 [`artifacts/README.md`](artifacts/README.md)。

