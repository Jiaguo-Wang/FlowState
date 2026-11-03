# Section III 模型与 Runtime 语义一致性审计（只读）

审计问题：**Section III 定义的可执行恢复前沿、候选集合与预算约束，能否准确表达当前 FlowState 实现及已完成 GPU 实验中的状态分配语义？**

依据：`flowstate/` 包全部源码、`evaluation/exact_planner_contract_gate_b5d1/planner.py`、
`evaluation/c_formal_perf_gpu/{control,runner}.py`（含其 import 链 `exact_reconciliation_correctness_gate_b5d3`、
`multishare_runtime_gate_b5c3`、`runtime_control_closure_r2c0/r2c1a`）、`evaluation/p1_strong_baseline_validation/{common,pcb,ldr,wfc}.py`，
以及 B5-C4 / C Pilot / c_formal_perf_gpu 冻结产物。零 GPU、零修改；所有判断给出代码或冻结实验依据。

---

## 一、论文公式与代码路径的逐项对应

| Section III 对象 | 实现位置 | 对应关系与差异 |
|---|---|---|
| 可执行恢复前沿 \(E_p(S)=\max(\{L_c: c\in S, c\preceq_t p, L_c\le T_p\}\cup\{0\})\) | [strict_epoch.py:175-180](../flowstate/strict_epoch.py#L175-L180) `StrictAllocationEpoch.frontiers` | **逐字对应**：`L_c`=候选 `token_pos`；`c⪯_t p`=该 (c,p) 边 `status=='EPOCH_ADMISSIBLE'`；`max(...+[0])` 即 `∪{0}`。函数注释明确"只使用显式准入边，完全不调用 Workflow 或 Lineage 谓词" |
| 兼容关系 \(\preceq_t\)（严格入口） | [strict_epoch.py:131-144](../flowstate/strict_epoch.py#L131-L144)（边校验）、[:183](../flowstate/strict_epoch.py#L183)（14 条件） | 三态边（EPOCH_ADMISSIBLE/REJECTED），**UNKNOWN 在分配入口被拒绝**（L137）；`L_le_T` 与 `exact_token_prefix` 为 14 个证据条件之二（L141-143 强制与 status 一致）；完整矩阵 `len(edges)==n×m`（L144） |
| 兼容谓词（Legacy 入口） | [state_catalog.py:60-73](../flowstate/state_catalog.py#L60-L73) `is_compatible`；[executable_state.py:11-22](../flowstate/executable_state.py#L11-L22) | **另一条入口的另一个关系**：同 workflow + lineage 前缀 + `token_pos≤planning_target`。Legacy 的 `T_p=min(anchor, FA前沿)`（[workflow.py:24-28](../flowstate/workflow.py#L24-L28)）把 FA 驻留折进目标；严格入口的 `T_p` 来自冻结快照且要求 `controlled_input_ready`（[strict_epoch.py:128-130,166](../flowstate/strict_epoch.py#L128-L130)）。论文的 \(\preceq_t\) 必须指前者（严格边），两者不得混写 |
| 恢复缺口 \(G_p=T_p-E_p\)、目标 \(J\) | [planner.py:117-120](exact_planner_contract_gate_b5d1/planner.py#L117-L120) `score`；[optimizer.py:59-62](../flowstate/optimizer.py#L59-L62) | `J(S)=Σ_p Phi(T_p-E_p(S), T_p)`，Phi=`position_aware_quadratic_v1`（[recovery_model.py:39-71](../flowstate/recovery_model.py#L39-L71)，定义域 T≤131,072、G≤T 在 `_validate_inputs` 强制）。拒绝边收益强制为 0 且矩阵元素有限非负（[planner.py:110-116](exact_planner_contract_gate_b5d1/planner.py#L110-L116)） |
| 预算 \(K\) | [strict_epoch.py:95,105](../flowstate/strict_epoch.py#L95)；[optimizer.py:64-65,116](../flowstate/optimizer.py#L64-L65)；[strict_control.py:103](../flowstate/strict_control.py#L103) | K=检查点**个数**；`plan_strict` 以 `K×size` 字节传入共用循环再 `capacity=budget//size`，等大小下与计数严格等价；`validate_plan` 断言 `len(selected)≤K`。等大小是硬约束（[strict_epoch.py:127](../flowstate/strict_epoch.py#L127)、[optimizer.py:107-114](../flowstate/optimizer.py#L107-L114)） |
| 候选集合 \(\mathcal C_t\) | 构造见 [planner.py:48-67](exact_planner_contract_gate_b5d1/planner.py#L48-L67) `from_frozen`；运行时对账见 [c_formal_perf_gpu/control.py:14-40](c_formal_perf_gpu/control.py#L14-L40) `physical_cpu_census` | 候选=冻结快照/物理 census 的全部驻留 Recurrent 节点；C/K3 为 16 个（8 owner×{L,L−64}），逐 Token 前缀+L 唯一映射回 CPU checkpoint_id |

---

## 二、关键假设的实现依据（文件·函数·位置）

1. **Token History 与配置认证**：候选/Pending 原始 Token 逐字节摘要一致（[strict_epoch.py:118-122](../flowstate/strict_epoch.py#L118-L122)）、`configuration_digest` 全域同一（L120）、快照 SHA256 前置校验（L151-152；[planner.py:50](exact_planner_contract_gate_b5d1/planner.py#L50)）；`same_configuration/same_boundary` 在 14 条件内。
2. **完整 Recurrent Components 驻留**：`StrictCandidate` 强制 `recurrent_resident and fa_resident`（[strict_epoch.py:125](../flowstate/strict_epoch.py#L125)）；物理视图要求 `conv_complete/temporal_complete` 且组件≥2、空槽不得残留组件（[strict_control.py:127-131](../flowstate/strict_control.py#L127-L131)）；`prepare` 逐候选复核 node/slot/component 身份与字节切片大小（[planner.py:83-97](exact_planner_contract_gate_b5d1/planner.py#L83-L97)）。
3. **Attention 侧连续有效前沿**：FA 不在分配域内，协调前后 `attention_digest/non_recurrent_digest` 必须不变（[strict_control.py:170](../flowstate/strict_control.py#L170)）；实验侧对 FA map 索引与内容整体哈希（[c_formal_perf_gpu/control.py:169-172,200-215](c_formal_perf_gpu/control.py#L169-L215)，仅豁免目标请求 E 后合法重算区间）；活观察 `attention_frontier` 与 CPU 声明逐 Pending 相等（L133-135）。**H（FA 前沿）与 E（Recurrent 可执行前缀）是两个量**，请求合同分别断言 `physical_fa_hit==H`、`executable_prefix==E`（[runner.py:257-260](c_formal_perf_gpu/runner.py#L257-L260)）。
4. **Scheduler 静止边界与 Strict Epoch**：物理视图要求 `idle and admission_closed and not transfers_pending`（[strict_control.py:117](../flowstate/strict_control.py#L117)）；候选节点 `references==0、protected==False、host_copy==False`（L125）——活动请求保护状态在 Epoch 内**不存在**（必须先静止）；桥接单次消费、驱逐必须处于有效连续控制区间（L243-259）；适配器侧另有版本/能力/并发校验（[adapters/sglang.py:123,533-559](../flowstate/adapters/sglang.py#L533-L559)）。
5. **执行控制前身份与物理映射重验**：`bind` 按完整 Token 精确匹配且要求**唯一**物理节点（[strict_control.py:141-152](../flowstate/strict_control.py#L141-L152)）；`preflight` 在首个驱逐前一次性重验 bind 等价与 Keep/Evict 物理无重叠（L155-163）；适配器驱逐前 `_validate_target` 重验节点身份/前缀摘要/驻留（[adapters/sglang.py:332-338](../flowstate/adapters/sglang.py#L332-L338)）。
6. **UNKNOWN/REJECTED**：见 §一；评估侧再加断言"快照无 UNKNOWN 边"（[e3 audit.py:37](greedy_swap_structure_e3_step1/audit.py#L37)）。
7. **未选中状态必须实际释放**：`StrictReconciler` 对**补集**逐项 `evict_mamba_only`（仅 Mamba 组件、不级联，[adapters/sglang.py:416-417](../flowstate/adapters/sglang.py#L416-L417)），每次驱逐前后 `postcheck`（[strict_control.py:166-188](../flowstate/strict_control.py#L166-L188)）：被逐节点槽位/组件必须清空、释放槽位记账闭合、最终驻留集合==selected 且 ≤K；失败即 fail-stop 账本（含已完成驱逐）+ invalidate，不重试不回滚（L87-92,191-229）。
8. **COW / 槽位共享 / Extra-buffer**：请求期 Native COW 只复制恢复源（CUDA 0.14–0.39ms，产物留档），不改变保留集与 E——协调后所选组件内容前后哈希相等（[control.py:195-196](c_formal_perf_gpu/control.py#L195-L196)）；`no_slot_alias` 条件 + 视图级 slot/component 无别名（[strict_control.py:133-138](../flowstate/strict_control.py#L133-L138)）、`prepare` 强制每候选单槽（[planner.py:88](exact_planner_contract_gate_b5d1/planner.py#L88)）；物理池容量记账闭合（used+free==capacity，L137-138；C/K3 为 16+12=28）。逻辑 K 是**槽位计数**，不是 HBM 字节保证——等大小槽使计数≈字节，但显存压力/碎片未建模未测。

---

## 三、候选闭包是否成立；是否需要 \(E_p^0\)

**闭包在严格路径上由构造强制，且有运行时检测**：`bind` 要求"驻留 Recurrent 节点集合 == 候选绑定集合"，任何候选域外驻留 Recurrent 状态直接失败（[strict_control.py:150-151](../flowstate/strict_control.py#L150-L151) "存在候选域外 Recurrent State"）；实验侧 census 把未映射额外物理状态一律视为未选中并真实驱逐、不免费保留祖先（[control.py:66-73](c_formal_perf_gpu/control.py#L66-L73)）。C Pilot 实测 16=16 一一映射、`unmapped_extras=0`；formal 每臂 census 16/16/0。因此：

- **候选集合以外的合法可复用 Recurrent 状态：在严格 Epoch 不存在**（存在即 bind 失败）；活动保护状态在静止边界上已被清除（references==0）；非候选 FA 状态存在且复用，但属于 H_p，不是 E_p 的输入。
- **"逻辑未选中、物理仍保留"会改变恢复前沿的情况：会被三层拦截**——postcheck 驻留校验、census 对账、以及请求合同 `executable_prefix==E` 硬断言。formal 的 g1_CB_SA BLOCKED 事件（形状预热在 E<T 格意外物化被逐驱逐状态，正式准入被抬高）正是第 3 层实际触发的证据：**该类偏差可被检测，且被拦截于计时之前**。
- **\(E_p^0\) 判断：当前已验证模型不需要引入**。`∪{0}` 的冷基底在"闭包 + 补集完全驱逐 + 静止边界"三点同时成立时是**准确**的（非近似）：Epoch 后一切 Recurrent 驻留恰为 S，请求原生匹配深度不可能超过 max S 前沿（断言保证）。但这三点是**实现强制的行为，不是公理**——Legacy 入口（`controller.reconcile` + `is_compatible`）接受调用方任意候选清单、无闭包/驱逐完备性检查，Epoch 之间的在线 Admission 也不受该模型约束。**结论：Section III 应把 \(E_p^0=0\) 写成闭包假设的推论并显式列出该假设；仅当论文主张推广到无闭包的一般在线 Runtime 时才需要引入 \(E_p^0\)（当前无实现、无实验）。**

---

## 四、模型与已完成实验的对应关系

| 检查项 | B5-C4 | C Shared/Branched K=3（Pilot 17 臂） | c_formal_perf_gpu（门禁2+主矩阵60） |
|---|---|---|---|
| 冻结选集 vs 实际驻留 | CPU 纯规划；驻留经 B5C3 冻结物理证据链认证（`from_frozen` B5 物理格式 + `prepare` 逐项复核节点/槽/组件/字节），本轮无新 GPU 驱逐 | census 16=16 逐 Token+L 唯一映射；物化轮次合同（末端@round0、祖先@round1，均在冻结预热内、决策前） | 同左，每臂 census 16/16/0 + CPU↔GPU 保留映射断言（[control.py:75-79](c_formal_perf_gpu/control.py#L75-L79)） |
| 未选中是否真实驱逐 | 不适用（纯规划） | 每臂 13 项驱逐全数返回，`operations_all_returned` 12/12，无免费保留祖先 | 同左；操作序列==`logical_eviction_ids` 逐项 `returned`（[control.py:197-198](c_formal_perf_gpu/control.py#L197-L198)） |
| Native E/G vs 逻辑目标 | 计划层 `validate_plan` 重算前沿/成本一致（[strict_control.py:105-109](../flowstate/strict_control.py#L105-L109)） | 12/12 格 E/T/G == CPU 冻结合同（合同认证非推断）；E>0 时存在**唯一**合法恢复源（[control.py:141-145](c_formal_perf_gpu/control.py#L141-L145)） | 62/62 臂断言 `fa_hit==H、executable_prefix==E、batch_prefix_length==E、extend==len−E、Prefill区间==E/G 合同`；形状预热准入==计划 E |
| 非候选状态提供额外恢复 | 无（规划域=证据域） | unmapped_extras=0；H=T 恒成立，策略杠杆仅在 E | 同左 + 请求窗口 Triton 新增编译==0、audit==2 |
| 正确性门禁能证/不能证 | 能证：冻结证据下计划合同；不能证：物理执行 | 能证：受控生命周期内闭包/驱逐/E-G/FA 保护/Cold 逐 Token 一致（12/12）；不能证：并发在线、HBM 压力、非冻结到达流 | 能证：上述全部 × 5 轮配对 + 机器模式量化；不能证：生产 E2E/Goodput（前审计已判定） |

受控成立条件（三实验共同）：冻结预热物化候选域、静止边界单 Scheduler 控制区间、等大小单槽组件、FA 全保（H=T）、每臂独立生命周期。**一般在线 Runtime（多活动请求、Epoch 间 Admission、真实容量压力、非等大小）不在证明范围内。**

---

## 五、在线求解器真实性

- **Marginal Greedy 的实际入口**：生产库唯一选择器 [optimizer.py:78-174](../flowstate/optimizer.py#L78-L174) `_select_candidates`（动态边际、tol 1e-9、同分按 ID、负边际即异常）；严格入口 `plan_strict`（L52-76）、Legacy 入口 `select`+`StateController.reconcile`（[controller.py:55-112](../flowstate/controller.py#L55-L112)）。
- **Exact 独立参考入口**：[planner.py:121-139](exact_planner_contract_gate_b5d1/planner.py#L121-L139)——`allowed` 规模门（n≤12、|P|≤4、K≤4、子集≤794），超域 `LimitError` 且仅在显式授权时 Fallback 到 Greedy（L151-159）；同分规则=容差内最少项+字典序。
- **`policy_plans.json` 的使用方式**：三策略选集由 CPU 侧离线产出并冻结；GPU 每臂在线做的是**认证与落实**，不是重新选择——`from_frozen→prepare→finish(冻结选集)`（[control.py:109-121](c_formal_perf_gpu/control.py#L109-L121)，注释明确"不调用 planner.allowed"），随后 E/T/G 与 CPU 期望逐格比对（L126-130）。
- **Observe / Select / Reconcile 在线步骤**：在线执行=final_heavy_observation（观察）、cpu_physical_mapping（census 对账）、strict_epoch_authentication_and_plan（认证+构计划）、apply_time_freshness_preflight（bind+preflight 重验）、strict_reconciliation_with_postchecks（13 项真实驱逐+逐项后检）、FA 内容审计；**Select 本身未在线执行**（实验设计使选择可复现且与计时解耦）。
- **WFC/StandaloneTopK/PCB/LDR 的身份**：冻结评估实现于 [p1_strong_baseline_validation/](p1_strong_baseline_validation/)（common/pcb/ldr/wfc.py，哈希锁于 b4c input_integrity 并被 e3 复核），**不是 Runtime 组件**；StandaloneTopK 为评估脚本内静态独立收益排序。

**Section III/IV 应采用的表述**：§III 定义严格 Epoch 上的分配问题与 \(E_p,J,K\)（§一表）；§IV 如实写"控制器在线执行 Observe→Authenticate→Bind→Evict→Postcheck；实验中策略选集为冻结 CPU 输出（`policy_plans.json`）在线认证落实；生产求解入口为 Marginal Greedy，Exact 为带规模门的独立参考求解器；WFC/PCB/LDR/StandaloneTopK 为冻结基线实现"。

---

## 六、成立 / 不成立 / 证据不足的条件与 Section III 必须修改的内容

**成立（有实现+实验双重依据）**：E_p 公式与 `frontiers` 逐字对应；三态准入边关系（UNKNOWN 拒绝）；K 的计数语义与预算断言；闭包+补集驱逐+E_p^0=0；H/E 分离与 FA 保护；fail-stop 协调；C/K3 与 formal 全臂的 E/G 合同认证。

**不成立/未实现（不得写成已有能力）**：无闭包的一般在线恢复（Legacy 入口无强制）；非等大小/多槽候选（`prepare` 明确"未验证多槽候选成本"）；Epoch 间 Admission 与并发活动请求下的模型；HBM 级容量保证；\(E_p^0>0\) 的建模（无实现）。

**Section III 必须修改（7 项，均属"完善假设"而非改公式）**：
1. 把 \(\preceq_t\) 明确定义为冻结准入边（14 证据条件、三态、UNKNOWN 拒绝、完整 n×m 矩阵），并与 Legacy lineage 谓词入口作出区分；
2. 显式列出**候选闭包假设**（\(\mathcal C_t\)=静止边界上全部驻留 Recurrent 状态；bind 域闭合检查强制）及**静止边界假设**（idle、无 Admission/传输、引用/保护清零）；
3. 显式列出**补集驱逐协调语义**（未选中⇒真实驱逐；失败 fail-stop 终止生命周期、不回滚），并声明 \(E_p^0=0\) 是 2+3 的推论——推广到无闭包在线场景才需引入 \(E_p^0\)（当前无依据）；
4. 声明**等大小、单槽**候选假设与 K 的槽位计数语义；写明逻辑容量≠物理显存保证；
5. 写清 \(T_p\) 语义（冻结快照 planning target、controlled_input_ready）与 H_p（FA 前沿）分离、协调期 FA 不变性；
6. 声明 Phi 定义域（T≤131,072、G≤T）与"J 为 modeled ms"单位约定；
7. 按§五给出求解器与控制步骤的准确表述（冻结选集在线认证落实；Greedy 生产入口；Exact 规模门；基线为冻结评估实现）。

---

## 审计判定：**READY**

当前公式能够准确表述已验证的目标场景：\(E_p(S)\)、\(\preceq_t\)、\(\mathcal C_t\)、\(K\)、\(J\) 与严格路径实现逐项对应（§一），其前提（闭包、静止、补集驱逐、等大小、FA 分离）在 B5-C4 证据链、C Pilot 17/17 与 formal 62/62 中全部被实测合同强制成立（§四）；全部缺口都是**假设未成文**（§六 7 项修改）而非模型与实现的冲突，因此无需收窄研究范围或调整模型，也无需引入 \(E_p^0\)。判定范围限定为"已验证的严格 Epoch 受控场景"；对一般在线 Runtime 的外推不包含在本判定内。
