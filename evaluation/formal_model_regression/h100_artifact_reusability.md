# H100 代表点制品可复用性

历史 correctness 来源为
`evaluation/runtime_artifacts/sota_correctness_20260825_104513_833972/summary.json`，历史
latency 来源为 `evaluation/runtime_artifacts/sota_latency_20260825_113526_839592/`。

审计结果：

- Scalable N16 K4：`CHANGED`，旧 latency 制品不能用于最终 FlowState claim；
- Scalable N16 K12：`IDENTICAL`，旧 latency 制品可复用；
- SOTA-signal K4：`IDENTICAL`，旧 latency 制品可复用；
- SOTA-signal K8：`CHANGED`，旧 latency 制品不能用于最终 FlowState claim。

需要后续 GPU 重跑的点为 `Scalable N16 K4` 与 `SOTA-signal K8`。本步骤没有执行重跑。
过去实测 TTFT 不变；这里只用正式 `Phi(G,T)` 重新计算模型预测 objective、均值、总 gap
和 executable hit ratio。
