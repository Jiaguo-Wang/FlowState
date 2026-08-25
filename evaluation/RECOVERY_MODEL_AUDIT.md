# Step 9B Recovery Cost Model 与 Tail-Latency 审计

## 技术摘要

- 冻结 Phi 在 32K gap 上与 Step 9B 实测增量接近，但在 4K、8K、16K 上明显高估；这构成中间 gap 区间的 calibration drift。
- SOTA-signal K8 中 Marconi-style 与 FlowState 的 total gap 都是 163,840 tokens，但 gap 分布不同。Phi 略偏好 FlowState，真实 TTFT 则偏好只保留 0/8K gap 的 Marconi-style。
- FlowState 当前优化的是平均恢复成本，不是 P95 或最大 gap。Scalable K4 与 SOTA-signal K8 都显示平均目标与 tail recovery 可以分离。
- 本报告只审计冻结 evaluation data，不用 Step 9B 数据重新拟合 Phi。

## 审计范围与指标口径

- 数据源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/sota_latency_20260825_113526_839592`。
- measured records：690；warmup 不进入审计统计。
- gap histogram 先按 equivalence class 去重，再用 class multiplicity 恢复一次逻辑 workload。
- TTFT mean/P95 使用 10 次 measured repetition，并按 class multiplicity 加权。
- measured incremental TTFT 以全 benchmark 的 G=0 加权 TTFT mean 为基线。
- relative error 定义为 absolute error / measured incremental TTFT；G=0 的分母为零，因此记为不适用。
- frozen-Phi predicted cost 直接调用当前 `RecoveryCostModel.estimate()`，没有重新拟合。

## Phi 在中间 gap 区间存在校准漂移

G=0 的真实 TTFT mean 基线为 **39.802 ms**。

| Gap | Phi (ms) | Measured incremental TTFT (ms) | Absolute error (ms) | Relative error |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 不适用 |
| 4096 | 178.729 | 131.588 | 47.141 | 35.83% |
| 8192 | 378.589 | 290.719 | 87.871 | 30.23% |
| 16384 | 772.491 | 625.997 | 146.495 | 23.40% |
| 32768 | 1499.508 | 1505.932 | 6.424 | 0.43% |

Phi 对 4K/8K/16K 的增量成本均为高估，而 32K 基本校准。该漂移只说明当前 WP2 profile 与 Step 9B request shape/运行路径之间存在外推误差；Step 9B 本身不能用于 post-hoc refit。

## 四个代表点的平均恢复与尾恢复

Histogram 列按 `G=0 / 4K / 8K / 16K / 32K` 给出逻辑请求数。

| Point | Policy | Gap histogram | Mean gap | P95 gap | Phi mean cost (ms) | TTFT mean (ms) | TTFT P95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Scalable N16 K4 | Global-LRU | 14 / 15 / 15 / 16 / 0 | 7441.1 | 16384 | 345.327 | 311.581 | 666.482 |
| Scalable N16 K4 | KVFlow-style | 14 / 15 / 15 / 16 / 0 | 7441.1 | 16384 | 345.327 | 312.306 | 669.894 |
| Scalable N16 K4 | Marconi-style | 14 / 15 / 15 / 16 / 0 | 7441.1 | 16384 | 345.327 | 310.832 | 663.738 |
| Scalable N16 K4 | FlowState | 24 / 15 / 15 / 3 / 3 | 5529.6 | 16384 | 252.930 | 251.196 | 687.724 |
| Scalable N16 K12 | Global-LRU | 42 / 15 / 3 / 0 / 0 | 1433.6 | 4096 | 63.612 | 87.411 | 175.718 |
| Scalable N16 K12 | KVFlow-style | 42 / 15 / 3 / 0 / 0 | 1433.6 | 4096 | 63.612 | 86.897 | 176.079 |
| Scalable N16 K12 | Marconi-style | 42 / 15 / 3 / 0 / 0 | 1433.6 | 4096 | 63.612 | 87.287 | 176.360 |
| Scalable N16 K12 | FlowState | 52 / 7 / 1 / 0 / 0 | 614.4 | 4096 | 27.162 | 59.633 | 173.512 |
| SOTA-signal K4 | Global-LRU | 10 / 0 / 20 / 0 / 10 | 12288.0 | 32768 | 564.172 | 564.191 | 1565.083 |
| SOTA-signal K4 | KVFlow-style | 10 / 0 / 15 / 0 / 15 | 15360.0 | 32768 | 704.286 | 710.738 | 1547.499 |
| SOTA-signal K4 | Marconi-style | 10 / 0 / 20 / 0 / 10 | 12288.0 | 32768 | 564.172 | 560.209 | 1546.523 |
| SOTA-signal K4 | FlowState | 16 / 0 / 20 / 0 / 4 | 7372.8 | 32768 | 339.245 | 337.099 | 1552.886 |
| SOTA-signal K8 | Global-LRU | 20 / 0 / 10 / 0 / 10 | 10240.0 | 32768 | 469.524 | 489.988 | 1561.044 |
| SOTA-signal K8 | KVFlow-style | 20 / 0 / 10 / 0 / 10 | 10240.0 | 32768 | 469.524 | 491.096 | 1563.630 |
| SOTA-signal K8 | Marconi-style | 20 / 0 / 20 / 0 / 0 | 4096.0 | 8192 | 189.295 | 187.235 | 339.711 |
| SOTA-signal K8 | FlowState | 32 / 0 / 4 / 0 / 4 | 4096.0 | 32768 | 187.810 | 219.788 | 1542.487 |

## Phi 排序只在 SOTA-signal K8 发生严格反转

`ranking_same` 按 Phi 的严格偏序检查；Phi objective 完全相同的策略允许真实 TTFT 打破平局。

| Point | Phi ranking | Measured TTFT ranking | ranking_same |
|---|---|---|---|
| Scalable N16 K4 | FlowState &lt; {Global-LRU, KVFlow-style, Marconi-style} | FlowState &lt; Marconi-style &lt; Global-LRU &lt; KVFlow-style | True |
| Scalable N16 K12 | FlowState &lt; {Global-LRU, KVFlow-style, Marconi-style} | FlowState &lt; KVFlow-style &lt; Marconi-style &lt; Global-LRU | True |
| SOTA-signal K4 | FlowState &lt; {Global-LRU, Marconi-style} &lt; KVFlow-style | FlowState &lt; Marconi-style &lt; Global-LRU &lt; KVFlow-style | True |
| SOTA-signal K8 | FlowState &lt; Marconi-style &lt; {Global-LRU, KVFlow-style} | Marconi-style &lt; FlowState &lt; Global-LRU &lt; KVFlow-style | False |

## SOTA-signal K8：相同 total gap，不同分布

### Marconi-style gap distribution

| Gap tokens | Logical requests | Fraction |
|---:|---:|---:|
| 0 | 20 | 50.0% |
| 8192 | 20 | 50.0% |

### FlowState gap distribution

| Gap tokens | Logical requests | Fraction |
|---:|---:|---:|
| 0 | 32 | 80.0% |
| 8192 | 4 | 10.0% |
| 32768 | 4 | 10.0% |

| Policy | Total gap | Frozen Phi total (ms) | Measured incremental TTFT total (ms) | Weighted TTFT mean (ms) |
|---|---:|---:|---:|---:|
| Marconi-style | 163840 | 7571.789 | 5814.377 | 187.235 |
| FlowState | 163840 | 7512.389 | 7186.603 | 219.788 |

FlowState 用 4 个 32K gap 和 4 个 8K gap 换取了 32 个零 gap；Marconi-style 则是 20 个零 gap和 20 个 8K gap。两者 total gap 相同。冻结 Phi 预测 FlowState 总成本略低，但 Step 9B 显示 8K 实际增量比 Phi 低很多、32K 则基本符合 Phi，因此 Marconi-style 的 measured incremental total 和 weighted TTFT mean 都更低。

## Tail 结果来自真实 gap distribution

### Scalable K4

三个 baseline 都没有 32K gap；FlowState 虽降低 mean gap 并增加零 gap，但留下 3/60（5%）个 32K gap。FlowState 在不超过 16K 的累计比例恰好为 95%，因此 P95 会取其 16K 观测的上边界；baseline 没有 32K gap 请求，P95 则落在更宽的 16K 请求群体内部。这解释了两者 P95 gap 都是 16K，但 FlowState 的实测 TTFT P95 略高。

### SOTA-signal K8

Marconi-style 的最大 gap 与 P95 gap 都是 8K。FlowState 有 4/40（10%）个 32K gap，所以 P95 gap 直接升至 32K；这解释了 FlowState TTFT P95 约 1.542 秒，而 Marconi-style 约 0.340 秒。

## 限制与稳健性

- 该审计是描述性 calibration check，不是新的模型拟合或因果实验。
- Step 9B gap 分组汇总混合了四个代表点和四种策略；它适合检查统一 Phi 的外推一致性，但不能单独定位 drift 来自 request shape、runtime path 还是计时边界。
- P95 使用冻结 benchmark 的加权经验分布定义；边界上恰好 5% 的深 gap 会使分位点对相邻样本敏感。
- frozen Phi objective 优化 mean recovery cost，不提供 tail-latency 保证。

## 建议的下一步

如果后续实验需要把 Phi 用作跨 workload 的定量预测器，应单独运行独立 Recovery Profiler recalibration：冻结与目标 benchmark 相同的模型、SGLang 配置、request shape 和计时边界，并使用与 Step 9B 不重叠的新数据完成校准与 held-out validation。当前 Step 9B evaluation data 只保留用于审计，不参与拟合。

## 仍需回答的问题

- 中间 gap 的 drift 是由 request shape、Mamba cache 配置、计时路径还是其他 runtime 状态造成，需要独立 profiler 才能区分。
- 若论文目标包含 tail SLO，需要另行定义 tail-aware objective 或约束；本步骤不修改当前 mean objective。
