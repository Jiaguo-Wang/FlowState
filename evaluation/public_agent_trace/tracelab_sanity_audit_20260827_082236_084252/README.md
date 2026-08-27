# TraceLab 结果 sanity 与 reviewer-attack 审计

本审计只读取 Step 10E 冻结 artifact。没有重新采样、运行策略、调用 GPU、修改 recovery model 或 protocol。

## 核心结论

- Marconi 实现复核：**PASS**。排序与冻结 recency、parent-relative FLOP efficiency、alpha=1.0 完全一致。
- Marconi 在部分 TraceLab 快照弱于 LRU/KVFlow，来自它与 current-pending recovery objective 的信号错位，不是已发现的实现错误。
- FlowState 与 Marconi selection 不同的案例为 397；其中 exact-parent coverage、兼容深度和 lineage redundancy reduction 可以重叠。
- 主集合 X>=4 共 13 个快照；25% 时 mean K=1.000，mean K/X=0.195。
- K=X 时 FlowState exact-parent set 违规为 0，全部 gap 为零：True。

## 解释边界

100% 是 demand-sufficient sanity point，不是核心性能卖点。TraceLab 成本是独立校准 Phi(G,T) 的离线估计，不等于该 trace 上直接测得的 TTFT。TraceLab 没有显式 LLM-level DAG，因此 KVFlow 的 richer STE 信号没有被激活。
