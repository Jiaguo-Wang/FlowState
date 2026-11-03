# T2：当前能够比较外部保留机制，尚不能宣称完整 SOTA 系统对照

**总门禁：PARTIAL。** 原论文与现有适配的差异已明确，有限范围的公平对照可执行；最接近 FlowState 的 Hybrid 基线 Marconi 仍缺忠实策略输入/机制，TOPAS 不适合当前固定保留问题的直接替代。审查日期为2026-10-26。未运行GPU、模型、策略评估或测试，未修改生产代码、论文与冻结输入。

## 1. 最重要的结论

1. **当前最缺的不是更多内部启发式，而是忠实的 Hybrid 缓存价值基线。**Marconi与问题最接近，但现适配用访问排名、Token跨度/位置代理、固定α和静态Top-K；它没有复现原论文的完整准入、FLOP/字节和动态树驱逐。历史结果可保留为适配比较，不能包装为击败完整Marconi。
2. **KVFlow-inspired已有可用CPU代码，合成数据也有合法声明输入。**其余真实来源Strict20没有决策时ASG及Fixed Prefix证据，不能补造。新核对的NeurIPS正式版还允许按持续命中启发式分段；当前归档同样缺少该过程的历史信息。
3. **TOPAS已经联合优化驻留与请求分配。**不能说已有工作都只做局部缓存排序。但其JCT/调度问题不同，公开可核验代码未定位、核心输入也缺，不能抽出任意分数冒称TOPAS。
4. **内部强基线必须保留。**Strict20中1-Swap达到Exact 18/20，Greedy和WFC均5/20且全部同集同前沿。不存在“Greedy普遍优于强简单策略”的依据。

用户最后的问题停在“当前最缺少哪一种具有…”。第1项按完整任务目标回答其最可能涉及的外部证据缺口；不假设还有未收到的附加要求。

## 2. 方法门禁与论文命名

|比较对象|当前门禁|论文可使用的名称与结论|
|---|---|---|
|Marconi官方完整方法|PARTIAL；直接新Strict运行未就绪|作为最相关Hybrid工作；不得写已完整复现|
|现有Marconi Adaptation|READY仅限已冻结代理公式的CPU对照；作为忠实SOTA仍PARTIAL|明确标注Token-value代理、α=1、固定候选、Recurrent-only|
|KVFlow-inspired Synthetic保留策略|READY（CPU声明输入）|KVFlow-inspired Recurrent retention；非完整官方KVFlow|
|KVFlow-inspired真实Strict20|BLOCKED|输入不完整，不报告虚构胜负|
|完整KVFlow服务系统|PARTIAL|预取/调度/代码版本及Hybrid对象适配未闭合|
|TOPAS作为当前K项保留基线|BLOCKED|相关工作；无法把联合调度问题无损缩成当前接口|
|内部7策略Strict20|READY，已有结果|模型化分配证据；非外部SOTA系统性能|

详细来源：[论文与源码审计](paper_source_audit.md)；逐方法/逐数据域：[输入矩阵](input_availability_matrix.md)；实际公式与改写：[现有实现审计](existing_adapter_audit.md)。READY不跨范围传播。

## 3. 现有数字能支持多少

|证据|可用数字/统计分母|必要限制|
|---|---|---|
|旧24 OpenHands GPU|216生命周期；对LRU改善43.057902%，对Marconi适配24.264401%；后者22胜2负|按24快照相对改善等权平均；协调后单输出TTFT，不含控制，不是当前Strict/完整官方系统|
|新Strict20|5Snapshot×4预算；Exact20/20、1-Swap18/20、Greedy/WFC/ST/PCB/LDR各5/20达到Exact|本次读取140行CSV核实；多预算不是独立工作负载；未执行Marconi/KVFlow合法选集|
|Synthetic A/B/C|6条件30行；FS对KVFlow-inspired4胜2平，对Marconi代理6胜|只比较当前模型化J；访问是合成声明；未来静态评价保留反向结果|
|C/K3 CPU|FS154.348956、Marconi代理348.234011、KVFlow-inspired627.614977 modeled ms|相同16候选K3；短T外推；不能当实际GPU延迟|
|C/K3正式GPU|60主臂：FS、Standalone、WFC各20臂|未包含Marconi/KVFlow；Marconi代理与Standalone同集不能升级为完整策略性能|

数值文件：[旧统计](../../legacy_runtime_evidence_p0_audit/summary.json)、[Strict20 CSV](../../greedy_swap_structure_e3_step1/condition_results.csv)、[合成CSV](../../kvflow_multiworkflow_cpu_main/cpu_comparison.csv)、[正式GPU协议](../../c_formal_perf_gpu/protocol.json)。没有跨数据族合并总体胜率。

## 4. 对论文差异定位的影响

- 可写：在固定FA、完整Strict兼容、已知Pending和逻辑K下，FlowState对可执行Recurrent状态集合估值；用同一控制器隔离检验保留决策。
- 需收窄：Marconi已经考虑Hybrid匹配和计算/内存价值；KVFlow已经处理共享前缀和Workflow距离；TOPAS已经联合选择状态与请求。FlowState的差异不能建立在否认这些能力上。
- 不可写：CPU J低等于GPU更快；内部WFC/PCB/LDR是外部SOTA；当前Strict20已胜过KVFlow；合成代理Marconi等于原方法；固定K证明真实HBM压力；预测收益证明控制成本后净收益。
- 当前生产入口已是门内Exact/超域Greedy，且最新 [Spec8](../../paper_hybrid_solver_gpu_gate.md)完成双分支GPU正确性；这不改变历史正式矩阵的冻结选集执行方式，也不是完整在线SOTA性能认证。

## 5. 唯一推荐的最小补充对照

只推荐已就绪的 **KVFlow-inspired Synthetic策略级CPU共同输入证据包**：复用全部A/B/C、K3/5及已有实现，明确声明图、Token、同分、祖先预算和当前/未来评价，补齐与Standalone的机制解释。已有基础结果不必重跑，不增加随机搜索或新工作负载。完整方案见 [baseline_experiment_plan.md](baseline_experiment_plan.md)。

这项补充能回答执行距离优先与当前联合恢复收益的差异，但**不能填补忠实Marconi或完整SOTA系统性能的空缺**。Marconi升级所需真实访问、GDN成本/字节、树驱逐和历史α条件尚未闭合，故不把它列为立即执行的实验。TOPAS也不安排替代打分实验。

## 6. 交付与可复核性

- [paper_source_audit.md](paper_source_audit.md)：正式论文、作者仓库、版本和实现位置。
- [baseline_mechanism_matrix.md](baseline_mechanism_matrix.md)：外部机制与内部策略定义。
- [input_availability_matrix.md](input_availability_matrix.md)：逐数据族输入与MISSING项。
- [existing_adapter_audit.md](existing_adapter_audit.md)：策略真实性、历史数字及冲突纠偏。
- [baseline_experiment_plan.md](baseline_experiment_plan.md)：仅就绪范围的对照方案。
- [source_identity.json](source_identity.json)：本次读取关键文件SHA、前后不变检查、作者提交及网络核查范围。

外部论文优先使用正式版，TOPAS仅能核实预印本；未找到代码不表述为不存在。全文中UNKNOWN不是反证。完成审查后停止，不移植、不执行补充实验。
