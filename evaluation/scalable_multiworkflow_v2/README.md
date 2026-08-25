# 可扩展受控多工作流 v2

## 目的

本 workload 用于离线观察工作流数量扩大到 8 和 16 后，共享 recurrent-state budget 下的 executable coverage 与 recovery cost。构造采用预先固定的完整阶乘设计，不根据任何策略结果反向调整。

## 阶乘构造

锚点深度固定为 4096、8192、16384、32768。

- N=8：四个锚点深度与 fanout={1,4} 的完整 4×2 组合。
- N=16：四个锚点深度与 fanout={1,2,4,8} 的完整 4×4 组合。

工作流按 `(anchor 升序, fanout 升序)` 确定性排列。每个工作流有一个位于 anchor 的 main parent checkpoint。每个 anchor 组只有排序第一的 workflow 增加一个位于 `anchor/2` 的 shallow checkpoint，因此 N=8 有 12 个候选，N=16 有 20 个候选。

每个待续请求的 lineage 为 `("P", "B编号")`，main 与 shallow checkpoint 的 lineage 均为 `("P",)`。不同 workflow 仍由独立的 `workflow_id` 隔离。

## 预算

- N=8：K={2,4,6,8}。
- N=16：K={4,8,12,16}。

这些预算分别对应一 workflow 一个 main checkpoint 基准的 25%、50%、75%、100%。K=N 时不需要保留 redundant shallow checkpoint，也足以达到完整 executable coverage。

## 离线边界

离线分析复用 v1 的 FlowState、Global-LRU、Equal-Share、Recovery-Only、Workflow-Only 和 exact Oracle 实现，并接入 KVFlow-style 与 Marconi-style。两个 SOTA-style 策略使用 [SOTA_BASELINES.md](../SOTA_BASELINES.md) 中先于比较冻结的 metadata：所有直接下一步分支的 steps-to-execution 均为 `1`，共同 recency 来自同一 Global-LRU 全序，Marconi FLOP proxy 使用父候选相对的增量 replay-token span，`alpha=1.0`。

输出中的 EPR 是 `sum(E_p)/sum(T_p)`，不是真实 GPU physical-hit EPR。estimated recovery cost 只作为统一 evaluation metric，不参与 KVFlow-style 或 Marconi-style 的选择。

本目录中的 CSV 与 JSON 是 planning/offline artifact，不是 GPU 或在线顺序执行结果。
