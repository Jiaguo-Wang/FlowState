# Gate 1 / Step 3：兼容性与保留容量合同一致性审计

**BLOCKED**。2026-10-12。

阻断的是把现有AgentX正式目标解释为**对应B2物理状态可以直接执行的恢复收益**：Formal保留70条正收益兼容边，而同一B2输入的完整Raw前缀谓词结果为假；Formal通过逻辑集合重编码消除了这个约束，却没有提供状态转换或实际恢复认证。这是有证据支持的语义冲突，不只是字段缺失。它不推翻四策略在同一抽象矩阵上573/573持平的算术事实，也不证明抽象分配问题不存在。

保留资源合同可以自洽定义，但当前K仍是单次候选选择数量，而非已实施的全局持久配额或物理池上限。未修改代码、矩阵、收益或预算，未重新评分、运行策略、GPU、Engine或Replay。

## 1. 两套定义的完整链条

| 维度 | Raw兼容 | Formal兼容 |
|---|---|---|
| 生成入口 | `evaluation/agentx_runtime_snapshot_collection.py:884–895` | `evaluation/rq3_agentx_formal_policy_evaluation.py:327–425` |
| 基础输入 | 此次回放materialized request的token序列、实测recurrent_prefix_length、pending token序列 | B1冻结corpus派生conversation/祖先关系和active target，加B2物理L/T |
| ID | conversation_id::reqNNNN；conversation_id::pendingNNNN，包含trace/派生分支作用域 | 同一ID全集，逐集合校验，不重命名为另一请求 |
| 核心判断 | `_prefix_match(req.token_ids[:L],pending_tokens)`，函数`:647` | B1严格逻辑祖先、pending inherited、逻辑累计位置≤active target，再以B2的L≤T过滤 |
| 同conversation | token前缀满足则可计入 | B1显式排除自身conversation |
| 字面token prefix | 要求实际合成回放token前缀相同 | 不要求；模块开头明确声明Raw token前缀字段不进入selector/目标 |
| L≤T | Raw阶段不直接检查；前缀匹配只要求不长于pending完整tokens | `encode_usable_compatibility:190–195`显式检查L≤planning_target |
| FA frontier | 另行测R/T，但不进入Raw列表判据 | T=min输入长度,FA frontier)，成为过滤上界 |
| recurrent驻留 | 先检查exact/fallback node与FA/recurrent驻留，失败abort | 沿用B2布尔驻留、node、digest；不再认证新增边的token状态 |
| lineage | 不用逻辑祖先过滤 | 将usable集合的laminar簇编码为新人工lineage；不是真实token前缀树 |
| 时点 | 历史回放结束、checkpoint检查及pending查询之后的同一次collector | 离线读取同份B2观测并重建B1在t的关系；没有新runtime状态观测 |

B1原定义源码：`evaluation/agentx_runtime_final_correctness_audit.py:87–115`（自身排除、继承、严格祖先、位置与target）；`:117–174`按已完成请求构造exact set，位置是累计逻辑request_hash_token_positions。Formal改用B2实际recurrent_prefix_length作为收益深度，不能把累计逻辑位置和实际缓存token位置当成同一种量。

`encode_usable_compatibility`在`:198–229`生成“兼容簇:<摘要>”lineage，让`flowstate/state_catalog.py:60`的普通prefix判据复现指定矩阵。`:416–425`证明的是**编码保真**，不是SGLang对这些状态的执行认证。`actual_lineage_paths`仅作为元数据保存，不能把编码path称为实际继承路径。

历史`agentx_runtime_final_audit_20260906_194000/corrected_runtime_snapshots.jsonl`将Raw列表保存为raw_compatible_pending_ids，再以B1列表替换compatible_pending_ids，**未产生新的GPU状态**。旧“corrected_exactly_matches_b1”只能证明列表符合B1。原源码`:364–368`甚至明确记载逻辑祖先成立但独立合成token并非pending字面前缀的情况；本轮不沿用“纠正”一词推断runtime合法性。

## 2. 23个快照、568个候选的全量差异

本轮对每个candidate×该snapshot全部pending输出一行，含一致和不一致关系，见`compatibility_diff.csv`。ID采用完整字符串精确映射，不截断req编号；ID全集匹配，UNMAPPED=0。候选驻留、L/T都来自同一B2记录，不以不同实验时间解释关系改变。

| 指标 | 数量 |
|---|---:|
| Raw与最终Formal集合一致的候选 | 236/568 |
| 集合不同的候选 | 332/568，分布于全部23快照 |
| 含Formal新增关系的候选 | 48 |
| 含Raw独有关系的候选 | 285 |
| 同时包含两种方向差异的候选 | 1 |
| Formal新增边 | 70 |
| Raw独有边 | 413 |
| 两边共同为真的边 | 44 |
| 差异边合计 | 483 |
| ID无法映射 | 0 |
| 定义级原因无法归类 | 0 |
| 可以用已证实观测时间变化解释的差异 | 0；没有这类证据 |

候选计数和边计数不是同一分母。Formal正边共114=70+44，Raw正边共457=413+44。

### 413条Raw独有边

以下分类来自历史逐边归因，已按完整candidate/pending ID联接，不把“逻辑不允许”误称为“runtime执行失败”：

| 原因 | 边数 |
|---|---:|
| 同conversation自身匹配，B1显式排除 | 250 |
| 非逻辑祖先但实际token前缀匹配 | 71 |
| SPAWN/non-inherited前缀匹配 | 11 |
| 前述条件之外仍不满足B1冻结逻辑关系 | 81 |

最后81条对应历史分类`physical_prefix_match_outside_frozen_logical_relation`；不猜测每条缺失的active-target数值。已知完整predicate链决定不在B1集合，不代表每条有重新归档的原生父子事件。

### L>T与两阶段统计差异

B1至最终Formal总计删去6条L>T关系，全部属于Raw原本也不含的边，所以历史“Raw vs B1”为76新增、334个不同候选、234个一致；本轮“Raw vs最终objective_usable”为70新增、332不同、236一致。不是审计计数矛盾。

483条差异边中69条同时满足L>T，均在Raw-only方向；它们本身已被B1排除，不能把这69条重复当成“仅由Formal L/T过滤导致”的差异。**仅靠L/T过滤新增的Raw-vs-Formal差异为0。** CSV保留独立L_gt_T列，避免把血缘差异与长度差异强行归为互斥单原因。

### 定义可解释不等于执行冲突消失

70条Formal-only边均有L>0、L≤T、正式正收益，分布21个快照；根据Raw生成代码，它们对这次已定义的pending token序列未通过完整prefix谓词。Raw不是调用SGLang“拒绝恢复”API的记录，而是collector在已解析physical prefix上计算的完整布尔关系。因此：

- 不能从空列表断言checkpoint对所有其他可能目标均不可执行。
- 可以确认：对于本次完整列举的pending输入，记录的prefix必要条件没有成立。
- Formal没有检查该必要条件，也没有state-transform机制来让错误历史的recurrent state变成正确状态。
- 原始完整token数组未归档，本轮不能报告首次不同token位置；owner/generation及两个pending实际恢复日志也缺失。这里的冲突认证依赖冻结collector结果及其生成代码，不伪称重新比较了原始token数组。

因此“全部差异具有定义来源”不足以满足PASS的“未留下影响恢复收益合法性的关键冲突”。

## 3. 代表案例 g000 / req0000

**结论：INCONSISTENT——相对于这次B2实际token状态的可执行解释。** 其B1抽象逻辑关系可成立，定义来源明确，但不能升级为实际恢复认证；实际恢复本身仍EXECUTION_UNVERIFIED。

来源：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/runtime_snapshots.jsonl:1`；同目录`per_snapshot/snap_02bc0afb13f7a2d9efa86c28511261d85c0e_2184.185000/snapshot.json`；正式manifest和R:505–507。

- 完整candidate：`02bc0afb13f7a2d9efa86c28511261d85c0e::req0000`，物理node18，L=37760，FA/recurrent驻留，identity=`exact:203673b2e66b2bdddc1ed693a9ab22ba086c10272e357acb8e6d6c36d745a818`。
- 正式pending：同trace的`::fa:000::pending0002`与`::fa:002::pending0000`，T均110527。B1将其编码为inherited严格后代；L≤T使两条边进入正式矩阵。
- Raw列表为空，表示req0000的这段真实回放token前缀没有通过与任何本次pending输入的prefix比较；不是缺失列表字段。
- node18和digest认证该checkpoint本身的历史前缀及驻留，不认证目标pending有该前缀。源checkpoint自身planning_target=37759也不能拿来替代目标pending的T=110527；L大于自身T一token不是这两条边差异的原因。
- request_rows记录req0000执行输入37760、输出1；没有目标两个pending沿node18恢复成功的记录。candidate逻辑owner是root conversation；物理slot号、owner代际和完整引用事件未随该candidate归档，UNKNOWN。
- 两个pending的派生lineage支持B1逻辑祖先；实际token-prefix谓词结果不支持此次物理恢复。不能凭父子命名、FA hit或人工兼容簇补足。

本轮未改边、未重算任何objective。对已有selected IDs仅作关系交叉引用：63/69条件至少一个旧策略选中了Formal-only边的源candidate；FlowState为54/69，LFU60、LRU56、Marconi52。结果见`affected_existing_selections.json`。这些是**选中源关联**，不代表每条边都决定最终E或目标误差；本轮禁止重新评分，因而不报告“修正后的成本”或策略胜负。PCB/LDR/WFC与FlowState集合相同，其54条件同样需限制执行解释。

## 4. 生命周期与一致性边界

| 顺序 | 当前路径与证据 | 缺失/限制 |
|---|---|---|
| 请求执行/结束 | AgentX collector`:710`起串行generate；request_rows有请求、completion及metrics | 不是原生Agent paused/ready事件 |
| 物化checkpoint | collector`:775–843`inspect exact node，失败时`:440–515`回退到实际radix边界 | 逻辑请求ID与实际物理prefix深度可不同；无slot generation账本 |
| 候选准入 | node存在、FA路径/recurrent驻留、当前identity；正式manifest保存L与digest | 准入证明源状态，不证明所有目标边 |
| pending形成 | collector`:327`current-context；B1按active_targets派生 | MISSING_INTERFACE：原生Ready/Queued因果事件 |
| 状态读取 | 源checkpoint检查后逐pendingFA查询；独占串行runner提供静止假设 | 各RPC批处理不是全局原子性；没有一般并发admission封闭证明 |
| allocation | Formal AgentX为CPU读取冻结矩阵；OpenHands runner重建view后计算 | AgentX未在此后执行所选集合与pending恢复 |
| reconciliation | OpenHands独立控制实验经idle RPC、digest验证、adapter逐目标回收 | 非全局事务；没有自动全池持久quota，失败可部分完成 |
| resume | F1/WP3A/OpenHands有受控请求COW/恢复记录 | 无70条AgentX新增边的实际恢复认证 |

分支共享的是只读checkpoint源；工作目标独立，不能共享可写recurrent状态。F1的2→3与2→5复制是直接物理证据，但属于另一工作负载，不能填补AgentX边认证。

`RuntimeCheckpointHandle`与node/prefix验证提供当前身份检查；`EpochHandles`防止epoch引用重用，不等于slot generation。active/host/session锁由adapter拒绝驱逐；未完成COW、tracking、transfer的完整所有权和别名安全仍需受限模式及调用方保证，不能声称已覆盖任意并发。

离线B1以已完成请求与当时active target构造，没有证据显示本轮额外引入未来请求；但这种current-context不是原生Ready，未知请求事件不能由事后DAG重建替代。“没有发现未来泄露”和“已证明合法待续”不同。

## 5. 保留容量与统一接口

详见`retention_contract.md`与`runtime_interface_matrix.csv`。

现有K=max(1,floor(r×candidate_count))是**当轮等大小候选选择数量**。候选驻留数>K合法，是因为池P较大，K不是当前物理上限；没有证据表明旧实验已经实施临时lease或长期K_ret。

单次reconciliation只覆盖输入候选。g000物理evictable110而候选109已经证明不能用candidate_count等同全池persistent占用。即使候选中只剩K项，也不能直接得出全局H≤K。状态归还slot可复用，预分配HBM不必下降。

四个环境各自审计，不跨环境相加：B2 P192、F1 P471、WP3A P16、OpenHands P28。B2末次空闲43–185；F1最大32子请求仍空闲406；WP3A是对称组件干预；OpenHands是有headroom的串行受控执行。它们均不证明一个自然原生shared epoch已经满足资源合同下K_ret<K_min。

合同本身无不可协调矛盾：独占admission边界、完整owner/rank/generation、合法边证书、可撤销借用与真实reservation允许空闲阶段C>K_ret。但**当前缺少长期quota、全池role账目、lease归还与全入口admission联动**。这是有限接口缺口，不应通过更换K口径隐藏。

| 阶段 | 判定 | 边界 |
|---|---|---|
| Observe | PARTIAL | idle census/FA查询存在，native Ready与跨RPC封闭缺失 |
| Build Executable State View | PARTIAL | exact handle有验证，AgentX人工lineage不认证实际边 |
| Allocate | SUPPORTED | 仅指有效输入关系上的等大小逻辑分配 |
| Reconcile | PARTIAL | 限定idle、无引用、映射与FA保护；无长期全池合同 |
| Resume | PARTIAL | 受控真实恢复成立，不能外推到Formal-only边 |

## 6. 对论文主张的影响

| Claim | 状态 | 证据与允许范围 |
|---|---|---|
| Physical FA-KV Hit ≠ Executable Hybrid-State Hit | VERIFIED | WP3A组件干预与恢复、原始CSV/driver事件支持限定设置；报告引用的server_full.log现缺失，不伪称重新读到 |
| Workflow-conditioned Executable Compatibility | CONDITIONALLY_SUPPORTED | 精确前缀/身份/状态有效时可支持；把AgentX全部正式边视为B2 runtime边的强表述CONTRADICTED |
| Recovery Benefit Model | CONDITIONALLY_SUPPORTED | Phi固定、抽象合法边上的计算有效；70条新增边缺合法恢复基础，不能由拟合准确性补救 |
| Global Recurrent Checkpoint Allocation | CONDITIONALLY_SUPPORTED | 抽象cardinality问题成立；当前K不是已实施的全局物理retention配额 |
| Runtime Reconciliation | CONDITIONALLY_SUPPORTED | OpenHands受控idle原语与门禁；不保证任意alias/active/COW/跨rank事务与长期quota |
| End-to-End Improvement | UNVERIFIED | 既有优化后E2E没有稳定显著净收益；不同批次不能拼接。本轮不重新统计 |

AgentX影响范围是21快照的70条Formal-only正边及其依赖解释，不是说其余所有边都获得原生认证。全部23快照存在Raw/Formal定义差异，但Raw-only边通常是逻辑范围排除，不应直接当作算法错误。OpenHands504条件不受本次AgentX重编码冲突的同一路径影响；其物理预算claim仍需单独限定。

Q5判断：**需要限定或修正部分系统假设。** F(S)作为给定合法收益矩阵的抽象问题仍自洽；论文系统化解释必须要求兼容边绑定实际历史/模型状态，并把逻辑K与物理总池、长期quota区分。当前证据不能继续无条件将AgentX正式矩阵作为已实现的Executable Hybrid-State语义示范。不要求因本次审计另造优化算法。

## 7. 五个直接回答与停止

1. **为什么有差异？** 两种定义：实际token前缀 vs B1严格逻辑祖先累计位置关系，再以物理L/T过滤并重编码。236候选一致、332不同；70 Formal-only、413 Raw-only；无ID错配。定义来源解释完整，但执行合法性没有因此恢复。
2. **Formal足以代表Runtime兼容吗？** 不足。新增边缺共同token状态/模型身份、owner/generation与实际恢复证据；其中Raw已记录对应prefix谓词不成立。不能仅补一句“逻辑兼容”继续声称相同物理执行语义。
3. **K是什么？** 当轮逻辑选择数量，不是物理P，不是已生效的长期全局K_ret。
4. **Runtime能执行保留分配吗？** 在独占idle、exact映射、无保护引用、FA隔离等条件下可执行单次candidate reconciliation；统一长期资源合同仍缺全池owner/generation、quota/reservation/lease和admission门禁。
5. **Problem Formulation是否成立？** 抽象问题成立；需要限定或修正系统假设与AgentX执行证据解释。当前不能跨过本Gate直接做性能对比。

**唯一最高优先级下一步：先确定并审核一套绑定实际pending token历史和checkpoint物理身份的Executable Compatibility准入合同，明确如何处置Formal-only边的证据缺口。** 在人工批准之前不修改或重建旧矩阵；只有准入语义明确后再考虑原生事件接口与容量执行验证。此次不实施。

## 8. 完整性与复现

`audit.py`只读取冻结关系、保存全量边对照、提取已有selected关联和census；没有调用模型评分或任何selector。复现入口：`PYTHONDONTWRITEBYTECODE=1 python3 evaluation/gate1_step3_compatibility_retention_contract_audit/audit.py`。只写本目录。

原始矩阵、568候选、23快照和69预算条件均保留，未按结果过滤。P3所用16个临时SGLang源码路径当前缺失，见前轮及本轮manifest；不能用其他分支冒充。完整token数组、slot generation和原生Agent ready事件缺失，不填推测值。差异CSV为完整候选×pending网格，筛选different=True即483条差异边。报告的BLOCKED专指可执行语义认证受阻，不影响只读审计交付完成。
