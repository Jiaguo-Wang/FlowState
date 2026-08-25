# SOTA 信号受控压力场景 v1

## 目的

本 workload 独立验证：当 KVFlow-style、Marconi-style、FlowState 及其他 evaluation policy 原本依赖的信息维度都具有真实差异时，各策略如何分配同一个 recurrent checkpoint budget。它不替换或改写既有 `controlled_multiworkflow_v1` 与 `scalable_multiworkflow_v2`。

场景、metadata 和预算在运行任何策略比较前固定，不能根据结果反向调整。

## 完整四因素阶乘

场景完整交叉四个二值因素：

- anchor depth：`8192`、`32768`
- pending fanout：`1`、`4`
- steps-to-execution：`1`、`3`
- recency class：`old`、`recent`

`2×2×2×2` 产生十六个独立 workflow，每个组合恰好出现一次。每个 anchor level 同时包含全部 fanout、steps 和 recency 组合，其他因素也满足相同的完整交叉性质。因此不存在“深 checkpoint 必然更新”或“高 fanout 必然更接近执行”等人为相关性。

## 核心状态与待续请求

每个 workflow 只有一个位于 anchor 的 main checkpoint，不增加 shallow checkpoint。十六个 candidate 大小相同，且初始 `recurrent_resident=True`、`fa_resident=True`。

每个 workflow 按 fanout 创建一个或四个 pending continuation，总计四十个。continuation 的 `anchor_pos` 与 `resident_fa_frontier` 都等于对应 anchor；main lineage 为 `("P",)`，待续 lineage 为 `("P", "B编号")`。workflow 标识隔离不同工作流，禁止跨 workflow compatibility。

## 冻结的 SOTA metadata

steps-to-execution 直接来自 workflow factor。一个 workflow 的所有待续分支继承相同的 `1` 或 `3`，不从运行结果产生。

recency 来自固定的 pre-decision access history：先按 checkpoint ID 依次访问全部 `old` checkpoint，再按 checkpoint ID 依次访问全部 `recent` checkpoint。随后机械映射成递增 last-access rank，保证任意 recent rank 严格大于任意 old rank。recency 不读取 anchor、fanout 或 steps。

每个 workflow 只有一个 candidate，因此没有 candidate parent。Marconi incremental replay span 等于 checkpoint 的 token position：8K candidate 使用 `8192.0` proportional FLOP units，32K candidate 使用 `32768.0`。所有 memory 相同，`alpha` 固定为 `1.0`，不进行调参。

## 策略信息边界

- Global-LRU：只使用 recency。
- KVFlow-style：使用 steps-to-execution，并只在 priority 平局时回退到 recency。
- Marconi-style：使用 recency 与 incremental FLOPs/memory。
- Workflow-Only：只使用 compatible future continuation coverage。
- Recovery-Only：只使用单个 continuation 的 recovery/depth value。
- Equal-Share：只使用公开且固定的 workflow order。
- FlowState：继续使用核心 workflow compatibility、recovery cost 和 set-dependent marginal coverage，不读取 steps 或 recency。
- Oracle：在本场景内独立枚举全部子集，精确最小化同一 `sum Phi(G)` objective。

## 预算与输出

主要预算点为 `K={4,8,12}`，`K=16` 仅用于 full-budget sanity。每个策略记录 selected checkpoint IDs、对应 workflow factor tuples、总 recovery gap、planning EPR、估计 recovery cost、已用 checkpoint/bytes，以及所选因素各层级的数量。

这些结果属于 planning/offline correctness 与机制解释，不是 GPU runtime 或性能结论。
