# TraceLab 最终离线评估协议

## 冻结结论

主 cohort 正式冻结为 **C128**。正式 workload 采用每个非空 `provider × context bucket × concurrency scale` 最多 5 个确定性快照、全局 active-run-set 去重、`X>0` demand gate，以及 exact-parent demand X 归一化预算。此决定在任何 policy performance 被观察之前完成。

当前协议可进入 128K recovery profiler 扩展，但**不能进入正式 policy comparison**：正式 recovery profiler 尚未独立验证到 128K gap。

## 为什么冻结 C128

- C32 仅产生 3 个 representative snapshots
- C64 产生 8 个 representative snapshots
- C128 产生 15 个且已有 Medium/Large workload
- 相比 C64，C128 eligible runs 增加 162.401%，selected unique runs 增加 120.690%
- C256 虽覆盖更多 workload，但要求明显更大的 recovery-model 有效域
- 该选择在任何 policy performance 被观察之前冻结
- 完整 TraceLab 中大于 128K 的数据继续保留用于 workload characterization，不从报告中删除。

## 正式采样结果

固定 seed 为 `20260826`，每层最多 5 个快照。先按 SHA-256 排名，再跳过 active-run set 完全相同的后项；不补造空层，也不使用 Phi、policy 或 recovery objective。

| 项目 | 数值 |
|---|---:|
| C128 eligible runs | 27,888 |
| Characterization snapshots | 59 |
| X=0 excluded | 2 |
| Formal snapshots | 57 |
| Small / Medium / Large | 29 / 23 / 5 |
| Claude / Codex | 34 / 23 |
| Unique active runs | 165 |
| Duplicate active-run sets | 0 |

## Snapshot 结构

N 保留 snapshot 之前已经生成的全部逻辑 recurrent checkpoints；X 只表示当前 known pending 所需的 distinct exact-parent states。没有执行 exact-parent、value、Phi 或 recency pruning。

| 指标 | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| W | 4.281 | 4.000 | 8.000 | 9.000 | 11.000 |
| N | 25.386 | 15.000 | 55.000 | 66.000 | 134.000 |
| P | 1.860 | 1.000 | 5.000 | 6.000 | 6.000 |
| X | 1.860 | 1.000 | 5.000 | 6.000 | 6.000 |
| N/X | 20.294 | 11.000 | 48.000 | 66.000 | 134.000 |

## Demand-relative budget

正式公式为 `K(r)=max(1, floor(X*r))`，其中 X 是 exact-parent demand。当前线性 TraceLab workload 中 X=P 是数据性质，不是算法假设。Candidate-relative 预算正式淘汰：历史 checkpoint 会令 N/X 偏大，并使 50%/75% 档缺少有效 state pressure。

| Ratio | K count | K mean | K median | K P90 | K max | K<X |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 57 | 1.000 | 1.000 | 1.000 | 1.000 | 38.596% |
| 50% | 57 | 1.211 | 1.000 | 2.000 | 3.000 | 38.596% |
| 75% | 57 | 1.368 | 1.000 | 3.000 | 4.000 | 38.596% |
| 100% | 57 | 1.860 | 1.000 | 5.000 | 6.000 | 0.000% |

## 冻结 policy metadata

- KVFlow-style：所有 known pending 的 `steps_to_execution=1`；同 STE 时复用 Global-LRU recency。TraceLab 不激活 richer DAG-distance signal，也不从 future round distance、tool count、elapsed time 或 anchor depth 构造 STE。
- Marconi-style：与 Global-LRU 共用 recency；FLOP proxy 使用同 workflow 线性 ancestry 上 parent-relative incremental token span；`alpha=1.0`，不调参、不读取未来 round。TraceLab 原始 `token_pos=0` checkpoint 保留零 span，不据此删除 candidate。
- FlowState：只使用 known pending、known anchor、线性 lineage 与冻结 recovery model；不使用 recency、STE、future prefix 或 future round。

## Recovery model 有效域门禁

C128 中 anchor 最大为 131072；当 K<X 且没有 retained compatible checkpoint 时，E 可以为 0，G 因而可达到 anchor。正式 Phi-based comparison 前，独立 recovery profiler 必须验证至 128K。禁止线性外推、把 gap clamp 到 32K，或用 32K cost 替代更大 gap。本步骤没有修改 Phi。

## 冻结指标与权重

Primary：

- `total_predicted_recovery_cost_ms`
- `mean_predicted_recovery_cost_ms_per_pending`
- `total_recovery_gap_tokens`
- `mean_recovery_gap_tokens`

Secondary：

- `executable_hit_ratio`
- `p95_recovery_gap_tokens`
- `max_recovery_gap_tokens`

Structural metadata：X、N、N/X、active workflows。主聚合对每个 selected snapshot 等权；允许另报 pending-weighted secondary aggregate，但必须分开。主 policy set 为 Global-LRU、KVFlow-style、Marconi-style、FlowState。Oracle 仅可在小 N 时作为 optional exact audit，不能改变主 workload 或参数。

## 完整性门禁

- Cohort frozen: **PASS**
- Sampling frozen: **PASS**
- Demand-relative budget frozen: **PASS**
- Policy metadata frozen: **PASS**
- Future leakage violations: **0**
- LLM-level branching violations: **0**
- Ready for profiler extension: **PASS**
- Ready for policy comparison: **NO**，因为 recovery profiler 尚未独立验证至 128K。

本协议没有运行 policy、Phi 或 GPU；未来不得根据 policy performance 反向修改 cohort、采样、预算、metadata、指标或权重。
