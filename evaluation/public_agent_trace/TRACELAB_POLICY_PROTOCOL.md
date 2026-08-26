# TraceLab 策略评估协议冻结

## 技术摘要

正式 TraceLab policy cohort 冻结为 3,196 个严格闭合且完整 run 最大输入不超过 32K 的 Agent Runs。固定 seed `20260826` 生成 3 个 provider × concurrency snapshots；本步骤 policy runs=0、Phi calls=0、Oracle runs=0。

有效域与无泄漏 gate 均通过，但非平凡竞争性为 WEAK，KVFlow metadata 为 WEAK：TraceLab 只能安全赋予所有已知 pending `steps_to_execution=1`，不能激活 DAG-distance signal。因此 Step 10D readiness=WEAK，不得据此调整 workload 或预算。

## 32K 有效域在 policy 之前冻结

| Provider | Strict runs | Rounds | Pending | Candidates | Mean rounds/run | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 3,196 | 8,113 | 5,106 | 8,113 | 2.538 | 1 | 6 | 9 | 38 |
| claude | 638 | 1,941 | 1,412 | 1,941 | 3.042 | 2 | 6 | 8 | 21 |
| codex | 2,558 | 6,172 | 3,694 | 6,172 | 2.413 | 1 | 6 | 9 | 38 |

该规则保证任意 planning gap 均位于 `[0, 32768]`。大于 32K 的 runs 仍保留在 Step 10C characterization 中，但不进入当前正式 Phi-based comparison。

## 实际支持三个非空并发 strata

固定 sampling 只按 provider 与 trace-observed concurrency scale 分层，不读取 policy value、Phi 或 gap。最终 Small=2、Medium=1、Large=0；empty strata=3。

| Snapshot 指标 | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Active workflows | 3.667 | 3 | 6 | 6 | 6 |
| Candidates | 14.333 | 16 | 18 | 18 | 18 |
| Pending | 3.667 | 3 | 6 | 6 | 6 |

共有 11 个不同 runs 出现在 snapshots 中。跨 provider 时间不混合，并发仍只解释为 trace-observed overlap。

## Budget contention 先报告、不调参

| Ratio | K median | K/N mean | K/W mean | K/P mean | K<W | K<P | K>=W | K>=P | Exact-parent capacity sufficient |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 4 | 0.231 | 1.111 | 1.111 | 66.667% | 66.667% | 33.333% | 33.333% | 33.333% |
| 50% | 8 | 0.481 | 2.278 | 2.278 | 0.000% | 0.000% | 100.000% | 100.000% | 100.000% |
| 75% | 12 | 0.713 | 3.389 | 3.389 | 0.000% | 0.000% | 100.000% | 100.000% | 100.000% |

`K<W` 表示预算无法为每个 active workflow 各留一个状态；`K<P` 和 exact-parent capacity 表示预算是否足以同时保护所有当前 pending。这里不执行任何 selection。

## Exact-parent 与 spacing 只做结构审计

exact-parent availability=100.000%。每个 pending 只检查同 workflow、相同完整线性 lineage、`token_pos == anchor_pos` 的候选。

exact parent 丢失后，最近严格线性兼容祖先的 gap：median=253、P90=529、P95=17394、max=17394 tokens。若没有更早兼容 checkpoint，gap 按 anchor 到零计算。

| Gap 上限 | 比例 |
|---:|---:|
| <=4K | 90.909% |
| <=8K | 90.909% |
| <=16K | 90.909% |
| <=32K | 100.000% |

spacing 只使用 snapshot 当前与历史 candidates，不调用 Phi。

## 三类 policy metadata 已预注册

### KVFlow-style

所有 known pending 的 `steps_to_execution=1`；相同优先级时使用与 Global-LRU 完全相同的 recency。不得从未来 round 数、tool 数、anchor 深度或真实未来等待时间生成 STE。TraceLab 不验证 KVFlow 的 DAG-distance 优势。

### Marconi-style

recency 与 Global-LRU 共用 checkpoint `known_at_time` 全序；FLOP proxy 使用同 workflow 线性 ancestry 上 parent-relative incremental token span；`alpha=1.0`，不搜索、不调参。所有 span 只来自 snapshot 当前与历史 token positions。

### FlowState

只允许 known pending、known anchor、linear lineage 和正式 frozen Phi。future prefix、future round、recency 与 STE 不进入 FlowState metadata；本步骤没有读取或调用 Phi。

## Step 10D 指标与解释边界

Primary：total/mean predicted recovery cost，以及 total/mean recovery gap。Secondary：executable-hit ratio、max gap、P95 gap。主 baseline 只包含 Global-LRU、KVFlow-style、Marconi-style、FlowState。Oracle 仅允许在小 candidate snapshot 做 optional exact audit，本步骤未运行。

下一步结果只能称为 `trace-driven offline policy evaluation`，不能称为真实 runtime latency。

## Gate

| Gate | 结果 |
|---|---|
| Profiler-supported cohort | PASS |
| Non-trivial state contention | WEAK |
| KVFlow metadata well-defined | WEAK |
| Marconi metadata well-defined | PASS |
| FlowState metadata leakage-free | PASS |
| Ready for Step 10D | WEAK |

WEAK 不授权修改 budget、metadata 或 sampling；它只记录 TraceLab 缺少显式 DAG signal，以及当前 cohort 的结构竞争强度。
