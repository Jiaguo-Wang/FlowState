# T2：现有实现的真实性与公平性

结论：已有代码足以支撑若干**明确命名的策略级对照**，不足以声称完整 Marconi、KVFlow 或 TOPAS 系统复现。下列差异均保留，不修改策略或历史产物。

## 1. Marconi Adaptation 的六项差异

现入口：[sota_policies.py:96–140,216–258](../../sota_policies.py)，`MarconiStylePolicy.select` / `_build_marconi_metrics`。

|项目|作者实现|本项目实际实现|影响|
|---|---|---|---|
|Recency|距上次访问时间倒数再归一化|直接对传入顺序/排名归一化|单独顺序可能一致，但与效率加权后的排序不保证相同|
|计算节省|模型分层 FLOPs、相对父节点|旧版按 Workflow/Lineage 找父候选，再用 ΔL；新合成直接用 L|两种代理本身也不同，不能合并为同一个公式的复现|
|资源分母|SSM 与 Attention 状态字节|`candidate.memory_bytes`；合成为1|只对应等成本 Recurrent 留存抽象，排除了原方法的重要异构成本|
|α|过去 trace bootstrap 调参|冻结 α=1|可作为固定适配参数，不是原论文完整参数协议|
|候选与更新|每次驱逐更新树上叶子/单孩子集合和分数|当前 resident 候选一次静态排序取 K|不保留原始准入、结构保护和逐步重新估值|
|物理操作|叶子 KV+SSM 或中间节点合并|选集交给共同 Recurrent-only 执行器|安全执行可以统一，但不等于原系统效果|

旧元数据函数：[sota_metadata.py:78–133](../../sota_metadata.py)。其父关系显式要求同 Workflow 与 Lineage，不能直接套到新跨 Workflow 严格域，也不能用它否定已准入跨 Workflow 边。新合成调用：[run_cpu.py:30–42](../../kvflow_multiworkflow_cpu_main/run_cpu.py)，明确 `flop=L`、`memory_bytes=1`、`alpha=1.0`。

**门禁：原论文 Marconi PARTIAL；严格的原策略复现尚不可直接执行。**旧适配数学公式可复现，但应称“Recency–Token-Value / Marconi-inspired retention adaptation”，不能把标签省略成 Marconi。没有真实 FLOPs 不必删除其历史数字，却必须保留代理限定。

## 2. KVFlow 有两个不同的本地实现

1. 旧 `sota_policies.py:26–94,181–197` 的 `KVFlowStylePolicy`：对旧 `is_compatible` 匹配的请求取最小外部 STE，按 recency 同分；`sota_metadata.py:63–75` 还存在统一 STE=1。它不是 ASG+Fixed Prefix 的新实现，不能用于新 Strict 比较。
2. 新 [kvflow_inspired_retention_cpu/policy.py](../../kvflow_inspired_retention_cpu/policy.py)：`compute_ste:18–42`、`priorities:49–82`、`select:85–92`，消费完整声明并验证实际 Token 前缀；无 Phi 和 Workflow 相等判断。未知依赖/映射保守拒绝，ID 升序同分。此实现可以复用，**不应重新发明打分替代缺失输入**。

新实现保留 STE 与固定前缀传播，改写了对象和可行域：候选为完整 Recurrent 状态、FA 保持；统一K项静态保留，不模拟原 KV 叶子删除、CPU prefetch 或状态调度。Unknown拒绝比原论文未知工作流等优先级退化更严格。它仍须由外围 Strict Epoch 验证物理驻留、别名与安全释放；`priorities()` 本身没有证明 GPU 驻留。

**门禁：Synthetic 声明输入上的 KVFlow-inspired CPU READY；Strict20/真实来源接入 BLOCKED；完整 KVFlow 系统 PARTIAL，不能直接移植运行。**原论文的固定前缀启发式分段路线见 [来源审计](paper_source_audit.md)，当前数据缺少支持该路线的历史证据。

特别风险：当前 C 的 Ready 调用关联 STE 大量同为0，ID决定多个保留顺位；候选含祖先和末端。其劣势可能来自同分和 Recurrent 粒度适配，不应统称为 STE 机制无效。应逐项保留 priority、同分组、传播关联和 E/G。

## 3. 内部策略与真实 Strict 接线

WFC/PCB/LDR 的 `common.Epoch` 消费外部提供的兼容集合，数据加载器必须只传 EPOCH_ADMISSIBLE。新 Strict 下不能再次套旧 `is_compatible`。新 `GlobalOptimizer.plan_strict()` 已直接调用严格 `frontiers`，规模超限才走 Greedy。

Exact、Greedy、1-Swap、Standalone、WFC、PCB、LDR 的输入与排序定义见 [机制矩阵](baseline_mechanism_matrix.md)。本次只读取 [E3 原始 CSV](../../greedy_swap_structure_e3_step1/condition_results.csv) 的140行，确认：

|方法|达到冻结 Exact 目标的条件数（20条件）|身份|
|---|---:|---|
|Exact|20|内部最优参照；当前生产也有门内分支|
|Greedy|5|内部动态边际方法|
|1-Swap|18|实验性局部改进，非生产策略|
|WFC|5|内部无 Phi 的动态覆盖对照|
|StandaloneTopK|5|内部独立价值对照|
|PCB / LDR|各5|内部简单推荐策略|

Greedy与WFC全部20条件同集同前沿，不能报告为“优于强基线”；Greedy相对Standalone的额外优势仅4条件。旧 [573条件审查](../../p1_strong_baseline_validation/) 中 Greedy=PCB=LDR=WFC 是给定旧逻辑矩阵的观察，不是一般定理。[E5逐条件解释](../../strong_baseline_position_audit_e5_step0/final_report.md)。

最新汇总 `paper_strict_evidence_extraction/baseline_input_contract.csv` 中仍有文字不一致：Standalone 写“17/20同J”却同时列4个差异条件；KVFlow“固定前缀祖先priority=0”忽略未Ready Agent的非零STE。本文以原始140行和实际 `policy.py` 为准，不传播这些简写。

## 4. 已完成比较及边界

|证据|实际比较与分母|可保留结论|不可保留结论|
|---|---|---|---|
|旧24 Runtime|24快照×3策略×3重复=216生命周期；FS/LRU/Marconi适配|既有核验复现 FS 相对 LRU平均逐快照TTFT改善43.057902%，相对适配24.264401%；Marconi对比22胜2负|不是完整Marconi；不包含控制成本；不具备当前逐请求严格认证|
|Strict20|5快照×4K×7内部策略=140行|相同严格矩阵下的J与选集质量|缺LRU/Marconi/KVFlow输入，不能算成这些SOTA的失败|
|Synthetic Sequential|2轮×2K×5策略=20行|共同静态十调用目标全部持平|不是10个同时Ready；CPU10末端域漏掉实际10祖先，不能当公平GPU预算结果|
|Synthetic A/B/C|3结构×2K×5策略=30行|当前J：FS对LRU、Marconi代理各6胜；KVFlow4胜2平；WFC2胜4平|全部CPU声明输入；未来静态评价反向结果不能遗漏，也不是在线JCT|
|C/K3正式GPU|FS/ST/WFC×4请求×5重复=60主臂|特定冻结保留集合下的GPU请求比较|没有LRU/Marconi/KVFlow正式臂；即使Marconi代理与ST同集，也不能继承为其完整选择/控制计时|

旧24来源：[原始审计](../../legacy_runtime_evidence_p0_audit/final_report.md)、[统计JSON](../../legacy_runtime_evidence_p0_audit/summary.json)。Synthetic来源：[原始CSV](../../kvflow_multiworkflow_cpu_main/cpu_comparison.csv)、[纠偏报告](../../kvflow_multiworkflow_cpu_review/review_report.md)。正式GPU：[protocol.json](../../c_formal_perf_gpu/protocol.json)。分母不可跨族相加成总体SOTA胜率。

本次只读核对 C/K3 的五策略当前 J：FS 154.348956、Marconi代理 348.234011、WFC 471.874343、LRU 582.034711、KVFlow-inspired 627.614977，单位均为**模型化ms**。全部短T非零项处于原模型目标位置校准范围外；不得据此认定实际延迟排序。

## 5. 需要纠正的旧报告概括

- C/K5 存在四叶+JOIN的双零集合；“当前与未来必然冲突”不成立。Greedy只选4项是零边际停机，不是没有容量。
- 选最深末端后其浅祖先的同请求额外收益为0，不是仍有“约64Token收益”。
- 最正式60臂只支持内部策略保留集比较；线上Hybrid双分支正确性另有 [Spec8门禁](../../paper_hybrid_solver_gpu_gate.md)，二者不能合并为全在线SOTA性能证明。
- Marconi/KVFlow官方共享能力真实存在；不能通过弱化描述制造 FlowState 创新点。
