# RQ3 KVFlow 工作流距离敏感受控比较

状态：**RQ3_KVFLOW_CONTROLLED_READY**。

本实验只执行 CPU 离线选择与公共目标计算，没有读取或重跑 OpenHands/AgentX 正式快照。
三个待续请求在决策时均已知且已物化；STE 直接来自公开 workflow graph，不来自真实未来轨迹。
workflow ancestry 只读取显式 workflow_id 与 lineage_path metadata，不读取 radix tree 或 token LCP。

## 工作负载与信号

- STE：1, 3, 5
- 公共预算：K=1
- KVFlow ranking：CP_NEAR > CP_MID > CP_FAR
- Marconi ranking：CP_FAR > CP_MID > CP_NEAR
- FlowState 边际顺序：CP_FAR > CP_MID > CP_NEAR

## 选择与公共恢复目标

| 策略 | 选择 | C(S) |
|---|---|---:|
| KVFlow Adaptation | CP_NEAR | 2218.544191 ms |
| Marconi Adaptation | CP_FAR | 968.602712 ms |
| FlowState | CP_FAR | 968.602712 ms |
| Exact | CP_FAR | 968.602712 ms |

## 结论

KVFlow 选择 STE=1 的 CP_NEAR，符合 workflow proximity 语义；FlowState 与 Exact 选择 CP_FAR，因为它在公共 C(S) 下提供最大的可执行恢复收益。
因此存在清晰的“更近但恢复价值更低”与“更远但恢复价值更高”的受控权衡。
本结果只比较 KVFlow 的 STE retention signal，不声称复现 KVFlow runtime、prefetching 或完整系统。

## 门禁

- three_distinct_ste_values: PASS
- kvflow_ranking_ste_sensitive: PASS
- workflow_semantics_explicit: PASS
- no_future_leakage: PASS
- no_policy_directed_tuning: PASS
- common_candidate_set_and_budget: PASS
- common_objective_evaluator: PASS
- policy_deterministic: PASS
- near_lower_ste_lower_recovery_value: PASS
- source_integrity: PASS
