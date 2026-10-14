# P4-G1 Evaluation 重组：Phase 1 交付

已完成基于冻结原始结果的主矩阵、三张图和章节重组建议。只执行CPU读数、汇总和绘图，没有执行baseline、GPU、SGLang、replay；没有修改论文、PPT、算法或旧冻结结果。

最新论文正文及P4-G0未在已搜索路径定位，因此本文是证据驱动的章节设计，**不宣称完成现稿逐段核对**。这一项保持UNKNOWN，不影响已有数据矩阵的恢复。

## 推荐结构

实验设置 → 恢复代价刻画与模型准确性 → 总体分配质量 → 强基线验证/工作负载区分力 → 机制归因与消融 → 请求级Runtime → 控制与E2E → 限制。恢复刻画和模型准确性作为分配主实验的测量基础；案例解释机制，不替代主实验；系统测量独立验证模型成本是否转化为实际请求收益。

详细的每节研究问题、假设、工作负载、对照、指标、图表、结果与缺口见[evaluation_outline.md](evaluation_outline.md)。

## 本轮实际交付

- [统一矩阵](overall_allocation_matrix.csv)：48行，191个snapshot、573个snapshot–budget条件、8方法；[完整阅读表](allocation_tables.md)。
- [逐案例证据](per_case_evidence.csv)：4584条，包含准确原始文件/行号、共同评分、历史原始成本、集合、覆盖与K_min。
- [Figure A 总体分配质量](overall_allocation_quality.pdf)：两数据集全部正式baseline；FlowState/PCB/LDR/WFC共享同一重合曲线。
- [Figure B 强基线](strong_baseline_validation.pdf)：各方法573/573目标、集合、E/G相同；顺序差异9/12/21。
- [Figure C 预算和区分力](budget_workload_discriminativeness.pdf)：各预算逻辑竞争与共享覆盖，明确不等于物理竞争。
- [Exact子集表](exact_opt_subset_quality.csv)：482/482 objective匹配；OpenHands434/434、AgentX48/48，不混入非可穷举条件。
- [机制组织](mechanism_attribution_plan.md)、[Runtime组织](runtime_evaluation_plan.md)、[Claim与缺口](claim_evidence_gaps.md)。
- [复现脚本](build_evaluation.py)、[证据与指标定义](evidence_index.md)、[源文件完整性](source_integrity.json)、[校验摘要](verification.json)。

## 必须保留的结果与限制

强baseline全平不是FlowState优势，也不能解释为算法在任何场景都无价值。OpenHands的独立正收益支持使PCB可精确化简，即便90个低预算条件受限仍等价；AgentX所有正式预算都能达到全候选前沿，无法惩罚不同排序。66/69正收益共享、48/69多请求新增收益、9/69排序反转是结构/过程证据，不是独特最终收益；69条件仅来自23快照。

Full Coverage与达到候选最大前沿不同：OpenHands FlowState为71/168、156/168、156/168；AgentX全部snapshot存在不可消除residual。主矩阵不能把不可消除部分算成选择器失败，也不能在预算提高后删掉残差以美化曲线。

RQ4真实TTFT均值164.705/129.097/94.025 ms可以支持指定协议下的请求级改善。优化后E2E请求执行163.191/127.987/91.152 ms来自另一批次，不能直接与TTFT当同指标比较。优化后FlowState控制131.904 ms，但epoch对LRU的mean差526.612 ms、95%CI[-1057.440,2214.943]；对Marconi为−549.649 ms、CI[-1784.913,699.396]。两者均未建立稳定净收益。初始批次的非显著结果和所有负例同样保留。

serving-path测量被正确性资格门禁阻塞，0/216正式生命周期；其性能为UNKNOWN。含探针epoch的剩余时间未独立细分，不能扣掉估计的诊断成本制造新serving结果。

## 校验范围

逐条连接正式来源并核对snapshot ID、budget、digest、selected set与成本；原始/共同评分最大允许差1e-9 modeled ms。四强方法共同评分的差精确为0；历史求和顺序的微小差另存原值。覆盖判定在全部4584条记录上核对整数G=0与C≤1e-9一致。两批E2E各216 records、864 requests重新汇总后与冻结summary逐项一致；RQ4从864条请求恢复均值并核对冻结配对均值。CI、分层与CV直接引用原正式分析，不重做实验。

已检查三张PNG预览与PDF生成；source_integrity.json记录被读取输入的SHA-256及前后相等。只证明清单中输入未变，不把它扩大成全仓库扫描结果。

## 四个最终回答

1. **新结构能覆盖已有证据吗？** 能覆盖本轮查到的正式分配、强基线、机制、模型、Runtime与负面证据；最新论文/P4-G0映射仍待核对，不能声称已完成论文逐段整合。
2. **哪些主图现在完成？** A总体分配、B强基线全平、C预算与区分力均已完成并附脚本。Runtime表与来源已整理；其进一步图形排版可直接用冻结数据，但不能画未完成的serving实验。
3. **哪些主张必须弱化？** 强workflow-aware基线优势、position-aware带来的最终增益、一般global allocation的独特收益、原生共享容量竞争、稳定生产E2E加速。请求TTFT、模型准确性、去冗余和Exact子集质量可按范围保留。
4. **下一步先做什么？** 先做数学区分力与物理资源条件研究，再过原生事件资格门禁后决定是否采新cohort；不建议盲目增加数据集或重跑相同工作负载。若论文坚持系统净收益主张，补齐正确性等价的serving测量也是必需项，不能由重组或减去探针估计替代。

本轮到此停止，不实现策略、接口或新增实验。
