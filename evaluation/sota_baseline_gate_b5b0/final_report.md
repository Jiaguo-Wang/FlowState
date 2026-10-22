# Phase B5-B0：工作负载判别力与SOTA输入可行性

## 总体门禁：PARTIAL

已完成冻结结果复核及结构审计。现有四快照可以区分部分机制，但不能证明FlowState相对强简单策略的普适优势。Marconi输入有实质基础但尚需合同补齐；KVFlow缺关键执行图输入，应暂缓主实验实现。

本轮只解析已有JSON/CSV、源码和SHA；没有启动GPU、Docker、Replay，没有重跑分配、实现Baseline或更改冻结结果。既有输出目录中的早期脚本和metrics保留不动，本轮新增完整报告及只读核验记录。

## 1. 权威证据与版本

B4-B1根manifest及g001–g004组manifest通过，四张snapshot摘要与summary一致。B4-C冻结evidence_manifest通过，结果来自results.json而不是抄取报告表格。B5-A两个本地仓库HEAD与指定commit一致且无工作树差异。

B4-C历史输入锁中的 `flowstate/optimizer.py` 与当前源码有一项摘要差异；已由R2-A接入Strict Epoch的历史变更解释，R2-A报告已有16/16回归。本轮未调用当前优化器，未把当前源码冒称原执行版本，也未覆盖旧锁。其他本轮核对的B4-C输入摘要一致。完整路径、摘要和清单计数见evidence_index.md、integrity_review.json和source_hashes.json。

g000、g005/g006、旧573条件不纳入总体。四个Snapshot×四预算是16个相关条件，不是16个独立工作流。四组主层均A、兼有B/C标签，不能拆成三个独立分层总体。

## 2. 原始结果核验

|项目|结果|
|---|---:|
|候选 / Pending / 配对|56 / 16 / 224|
|EPOCH_ADMISSIBLE / REJECTED / UNKNOWN|68 / 156 / 0|
|正收益边 / 跨Workflow正收益边|68 / 12|
|存在跨Pending共享的Snapshot|4/4|
|存在恢复收益重叠的Snapshot|4/4|
|K_min|各组均4|
|K<K_min的条件|12/16（每组K1–3）|
|FlowState匹配Exact|12/16|
|PCB匹配Exact|8/16|
|WFC匹配Exact|8/16|
|PCB或WFC至少一个匹配Exact|16/16，二者互补|

Greedy全部非最优条件：g001/K4（+68.218394974）、g002/K3（+49.016811025）、g002/K4（+82.987847211）、g004/K4（+48.608817340）modeled ms。

FlowState相对PCB为8胜4平4负，相对WFC为4胜12平0负，相对StandaloneTopK为5胜11平。没有任何条件同时严格优于PCB和WFC。逐条件成本、最终集合和Exact差距见condition_policy_review.csv；所有读取的7策略配对中同目标异最终集合为0，选择顺序不同不计作不同集合。

## 3. 为什么PCB/WFC容易达到Exact

**具体结构已证实，化简在这些结构假设下可证明；其普遍频率未验证。** 每组只有一个覆盖全部四Pending的共享B1，其他候选按Pending可分，且深层专属项支配本链浅层项。最优解可归入“选择B1”或“不选择B1”两个解族，固定该选择后剩余问题可分。四个不同最深项又分别是达到各最大前沿的唯一方式，所以K_min=4。

PCB代表独立深层项的选择；WFC本批先选B1再做专属推进。g003由独立深层项获胜，其他组在部分小预算下共享优先获胜，预算增大时可能切换。冻结结果确认PCB/WFC恰好覆盖16条件的Exact最优；这不是一般定理，更不是一个已经实现的可部署Portfolio。

当前足以说明：共享收益确实进入目标，静态重复覆盖会浪费预算，覆盖数量优先可能低于恢复收益排序；同时Greedy的不可交换早期选择也会失败。不能说明位置项具有独立分配收益、复杂全局分配普遍必要或CPU改善等于GPU性能改善。

缺少的检验维度是多个不同非单例支持域的共享层、它们与专属状态在紧预算下的竞争，以及能隔离Token排序与位置成本项的预注册条件；不是简单增加候选或降低K。不能根据胜负找新样本或构造非法Token前缀关系。

## 4. Marconi能否进入实现

**PARTIAL；值得先做CPU输入合同补齐，不宜直接进入正式效用评估。** 原始创建、命中、Token树、模型尺寸不是空白。四组各有两条跨Workflow历史匹配记录，可直接核查B1来源；现成last_access_order却只更新同Workflow同位置，不能直接当作原生recency。

父节点也有真实区别：g001 A1的直接树父节点是2437-token的FA-only Node19，最近驻留Recurrent祖先是2432-token的B1；现成同Workflow增量代理却为3520。原始Marconi的父节点、GDN/GQA FLOPs、Hybrid内存分母与recurrent-only固定K不能无说明混用。独立Bootstrap窗口和起始树没有冻结证据。

α主方案按用户已确定的官方FLOPs Savings选择，Hit Rate仅敏感性分析；不再把这点列为待用户拍板。禁止用测试J/Phi选α。B5-A“缺失输入：无”“无需树信息”的判断被本轮实际数据核查限定，旧报告保持不动。

## 5. KVFlow是否应暂缓

**BLOCKED；应暂缓。** Pending、Session/Lineage、Token前缀和EPOCH_ADMISSIBLE均不是声明式Agent执行DAG。缺Agent Step映射、AND/OR/条件依赖、当前执行位置和声明时点证据，不能忠实计算steps。B5-A由Pending图导出距离的合同前提未成立。

继续区分论文优先级机制、当前混合HEAD阈值保持与默认非分层LRU；当前HEAD无法证明等于论文实验版本。已有LRU或固定steps=1不能换名成完整KVFlow。应补真实声明式工作流元数据及版本边界，而不是先编写代理距离。

## 6. 下一轮唯一最高优先级任务

**B5-B1：Marconi适配输入合同闭合与CPU规范化门禁。** 仅围绕现有真实创建/Native Match事件、实际前缀父关系，冻结GDN/GQA计算和资源分母、独立Bootstrap来源及初始化时钟规则；先验证不会读取测试Epoch结果。若Bootstrap或模型语义仍无法闭合，明确保留PARTIAL，不先上主表跑胜负。

本建议不授权本轮实现任何Baseline或新增数据。KVFlow暂缓；PCB/WFC现有严格负例与持平应继续保留。完成本轮后停止，等待人工审查。

## 交付索引

- evidence_index.md：来源、版本和摘要差异。
- workload_structure_audit.md：逐候选支持、K_min、16条件完整共同成本及结构归因。
- marconi_input_gate.md：逐输入状态、真实事件、父节点与Bootstrap风险。
- kvflow_input_gate.md：执行依赖与兼容性分离、版本边界及缺失字段。
- baseline_eligibility_matrix.md：独立资格判定与原因分类。
- condition_policy_review.csv、condition_review.json、structure_statistics.json：复核数据。
- native_access_evidence.json：当前物理身份可映射的历史命中记录，不等价于完整输出恢复认证。
