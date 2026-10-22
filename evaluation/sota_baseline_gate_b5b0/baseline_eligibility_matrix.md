# Baseline 准入决策

|对象|门禁|现有输入与执行资格|主要原因分类|本轮建议|
|---|---|---|---|---|
|PCB/LDR|READY（已有冻结比较）|严格矩阵、位置、T、共同Phi完整；原始结果可复核|工作负载局部判别力有限；非输入缺失|继续保留全部胜平负，不重新实现|
|WFC|READY（已有冻结比较）|严格边、位置与当前前沿可用于动态覆盖数/推进量|覆盖数优先与收益冲突已见；总体样本小|保留g003全部严格差异和其余持平|
|StandaloneTopK|READY（已有冻结比较）|冻结独立收益完整|静态同链冗余可区分；不是完整SOTA|保留五个严格差异与11个持平|
|Marconi Eviction Adaptation|PARTIAL|真实创建与Native匹配历史、树和模型尺寸已存在；父节点、时钟、GDN/GQA公式及独立Bootstrap尚未闭合|输入合同缺失＋资源/模型语义不等价|值得进入CPU输入合同补齐，尚不允许宣称忠实适配READY|
|Marconi完整原系统|BLOCKED（当前固定候选任务）|原生准入、树分裂/合并、Hybrid字节容量无法由本轮固定候选recurrent-only K完整表达|系统语义不等价|不以适配替代全系统复现|
|旧B4-C Marconi Adaptation|READY仅限既有代理结果复核|alpha=1、增量Token代理、旧recency字段均已冻结|与原始Marconi忠实性不同|不得作为已完成官方Marconi比较的证据|
|KVFlow steps-aware adaptation|BLOCKED|缺声明执行DAG、Agent Step和时点证据|关键数据缺失＋steps与兼容性语义不等价|暂缓主实验，不推造steps|
|KVFlow论文系统复现|BLOCKED|同时缺原生依赖、分层/预取合同及论文版本归属|数据缺失＋系统语义不等价＋版本UNVERIFIED|不启动实现|

PCB实际读取共同Phi，但仅按每请求独立最佳项推荐；WFC不读取Phi，动态使用改善请求数和Token推进量。因此PCB与FlowState持平不是“成本模型无用”的证据；g003的WFC失败也没有独立隔离位置项。

上述READY只对应已有CPU冻结算法比较的输入资格，不表示GPU输出、逐边恢复或性能资格。B4-B1的68条EPOCH_ADMISSIBLE不因R2-C1b-2验证一个目标而全部成为RESUME_VALIDATED。

没有将“实验实例结构简单”“模型/资源口径不等价”“缺少声明DAG”“论文版本无法溯源”合并成一个笼统数据不足结论。详见各输入门禁和workload_structure_audit.md。
