# Phase B5-C4 最终报告

**Gate：READY**。这是输入、版本、枚举与比较可复核的门禁，不表示Greedy取胜。

## 主结果

J(S)，单位为模型化恢复成本ms：

|K|FlowState|PCB|WFC|LDR|StandaloneTopK|Exact OPT|
|---|---:|---:|---:|---:|---:|---:|
|1|228.054813355|435.362141026|228.054813355|435.362141026|228.054813355|228.054813355|
|2|130.803083830|281.013185410|130.803083830|281.013185410|130.803083830|48.228728759|
|3|48.228728759|134.034994578|48.228728759|134.034994578|48.228728759|16.411344607|
|4|31.093896177|1.753119344|31.093896177|1.753119344|48.228728759|1.753119344|


七候选全部保留，逻辑成本相同；13准入/15拒绝/0未知；冻结Phi、Token、来源及配置未改变。历史Greedy按B4-C SHA恢复隔离执行，PCB/WFC/LDR按原锁调用，无UNVERIFIED策略。

## 结论

1. 多共享结构打破此前PCB/WFC互补覆盖Exact的观察：K2/K3两者均非最优。
2. FlowState仅K1 Exact；K2/K3/K4分别多支出82.574355071、31.817384152、29.340776833模型化ms。不可回退的浅共享选择及后续重复覆盖占用预算是可观察的失效机制。
3. FlowState与WFC四条件完全一致。相对StandaloneTopK仅K4更好；该静态策略选中被N27支配的N21，造成零边际重复。
4. K_min=4；树形支持成立。当前事实支持重新评估是否应采用适合受限树结构的精确求解器，不能继续以这些实例论证现有Greedy的普遍优势。本轮不实现、不调优任何新求解器。

## 验证与范围

独立Decimal核对128子集、24策略结果和448个边际变化全部PASS。来源摘要前后不变，详见input_integrity.json、validation.json、exact_verification.json；完整数值及归因见policy_comparison.csv、results.json、allocation_difference_analysis.md。

本轮只有一个Trace-derived Controlled Composition快照；四预算不是四独立样本。EPOCH_ADMISSIBLE不自动成为RESUME_VALIDATED。没有GPU运行、Eviction或Reconciliation，不报告TTFT、端到端收益或真实物理容量竞争。没有进行Phi位置项消融，也没有证明一般非树共享结构的算法性质。

已完成，停止并等待审查，不进入下一阶段。
