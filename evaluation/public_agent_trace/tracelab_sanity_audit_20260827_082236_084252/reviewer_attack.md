# Reviewer-attack questions

## 1. 为什么 Marconi-style 弱于 LRU？

冻结 Marconi utility 同时奖励 recency 与 parent-relative FLOP efficiency，但不读取当前 pending coverage、E/G 或 Phi(G,T)。它可能保留计算跨度较大却不服务当前 demand 的状态。排序复核零 mismatch，因此现有证据指向目标语义差异，而非实现错误。

## 2. 这个 Marconi 比较是否不公平？

它应被明确称为 Marconi-style snapshot adaptation，而非原系统端到端复现。recency 与 LRU 共用，FLOP proxy、alpha=1.0 均在观察结果前冻结，未调参；公平性来自相同 candidate、K 和统一 evaluator，外部有效性限制仍需披露。

## 3. KVFlow 为什么与 LRU 接近？

TraceLab 没有显式 LLM-level DAG，冻结协议令全部 known pending 的 STE=1。同 priority 内 KVFlow 回退到 LRU recency，仅会把无 compatible future 的 candidate 排到 priority=1 candidate 之后，因此大量退化为 LRU。

## 4. 为什么 K=X 时 FlowState cost=0？

X 是 distinct exact-parent demands 数，K=X 恰好 demand-sufficient。审计中 FlowState exact-parent set 违规为 0，因此每个 continuation 都有 E=T、G=0。

## 5. TraceLab 是否天然偏向 FlowState？

主 cohort 预注册为 X>=2，确实聚焦多状态竞争而非自然事件频率；但采样、X、预算和 metadata 在 policy comparison 前冻结，且未用 policy outcome 采样。结果适用于该结构覆盖 cohort，不能外推为完整 TraceLab population average。

## 6. 为什么 X>=4 紧预算优势变小？

25% 时 mean K/X=0.195，每个快照实际只有 K=1。两策略都只能覆盖少量 demand，共同未覆盖项主导总成本；到 50%/75% 后 FlowState 才有多个 slot 按位置感知边际收益连续分配。

## 7. estimated recovery cost 是否等于真实 TTFT？

不等于。它是独立 H100 profiler 校准的增量 recovery estimate；Step 10E 没有 GPU 或真实 TraceLab token replay。它支持 objective-level 离线比较，不应被表述为该 trace 上实测 latency speedup。

## 8. TraceLab 没有 DAG 是否削弱结论？

是。它削弱对 branching workflow 和 richer KVFlow STE 的结论。当前证据只覆盖由真实 round 顺序构造的线性 lineage 与 immediate tool-call continuation。

## 9. FlowState 使用同一个 evaluator，胜出是否是同义反复？

Step 10E 直接检验的是 allocator 与正式 executable-state objective 的对齐，不能单独证明真实 latency superiority。独立 held-out profiler 与受控 H100 runtime evidence提供外部支撑，但 TraceLab offline 结果本身必须标为 modeled objective comparison。

## 10. 为什么 baseline 在 K=X 时仍会漏掉 exact parent？

K=X 只等于 demand 数，不等于全部历史 candidate 数 N。baseline 可能把 slot 用于较浅兼容 checkpoint 或当前无 pending dependency 的历史状态；它们没有读取 exact-parent recovery demand。

## 11. 105 个快照足以做 population claim 吗？

不足。105-set 是 deterministic stratified structural sample，稀有 Codex、Medium/Large 和高 X 被有意提高权重。bootstrap 量化该样本内部不确定性，不修复相对于自然事件频率的 sampling bias。

## 12. 100% 点能否作为核心性能提升？

不能。它是 demand-sufficient correctness sanity point，展示当预算足够覆盖所有已知 exact-parent demand 时目标应归零；它不是有限资源区间中的主要 tradeoff，也不是实测 GPU speedup。
