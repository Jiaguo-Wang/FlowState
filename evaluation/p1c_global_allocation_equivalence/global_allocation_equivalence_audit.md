# P1-C：Global Allocation 等价性与工作负载结构审计

日期：2026-10-12。结论：**正式总体缺乏四种方法最终结果的区分力；OpenHands 的可分结构有精确化简证明，AgentX 的持平则有预算与唯一前沿实现集合证书。四种算法并非一般等价，FlowState 贪心也并非一般最优。**

本轮只读正式源码与记录；仅在隔离目录执行 CPU 结构汇总及三个数学构型、四个预算条件的验证。没有重跑正式 selector、GPU、Engine、Replay，没有修改策略、冻结总体、Phi 或论文。数值均为 modeled ms，不是实测延迟。三个预算是相同快照的重复条件，不进行独立样本显著性推断。

## 1. 证据范围与现有事实

主要来源如下，后文 R:n 表示 R 文件第 n 行，其一行对应一个完整预算条件。

- R：`evaluation/p1_strong_baseline_validation/per_case_results.jsonl:1`，573 行；包含冻结输入、收益矩阵、完整输出、基线轨迹、首次分歧与正式来源文件/行号。
- 冻结规则：同目录 `implementation_lock.json:1`、`pcb.py:5`、`ldr.py:4`、`wfc.py:4`、`common.py:57`。本轮四个锁定摘要逐一匹配；FlowState optimizer、Phi、兼容性、workflow 和 executable_state 五个文件亦与 P1-B 的 source_integrity_before.json 历史摘要匹配。
- FlowState：`flowstate/optimizer.py:34`；模型：`flowstate/recovery_model.py:39`、`:58`；兼容性：`flowstate/state_catalog.py:60`；目标：`flowstate/workflow.py:25`。
- K_min 原算法：`evaluation/p1_strong_baseline_validation/evaluate.py:10`，只覆盖各 pending 的全候选最大可达前沿。
- 前序解释：`evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md`、`evaluation/p0_allocation_mechanism_audit/final_report.md`、`evaluation/p1_strong_baseline_validation/final_report.md`、`evaluation/p4_evaluation_restructuring/final_report.md`。
- 正式原始结果：`evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl` 和 `rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl`。R 的 `source_file/source_line` 给出逐条件映射。

| 数据集 | 独立快照 / 预算条件 | 三个基线各自同集合、同逐请求 E/G、同目标 | PCB / LDR / WFC 顺序不同 |
|---|---:|---:|---:|
| OpenHands | 168 / 504 | 504/504 | 0 / 3 / 3 |
| AgentX | 23 / 69 | 69/69 | 9 / 9 / 18 |
| 合计 | 191 / 573 | 573/573 | 9 / 12 / 21 |

本轮直接比较保存的集合、逐请求记录、浮点目标与实际顺序；成本差均严格为 0.0，不依赖扩大容差。每种基线相对 FlowState 的严格胜/平/负为 0/573/0。数据来源 R:1–573；本轮摘要 `structure_summary.json`、完整证书 `frontier_certificates.json`。既有 482 个可穷举条件匹配 Exact OPT 是另一条有限总体事实，不能据此证明一般贪心最优，也没有重算该实验。

## 2. 四种冻结决策规则

| 方法 | 每轮读取和分数 | 排序、去重与终止 | Phi 的角色 |
|---|---|---|---|
| FlowState | L、T、兼容性与当前集合 E；逐候选计算 C(S)−C(S∪{c}) | 所有驻留且等大小候选；最大正 marginal；差值在 1e−9 ms 内按候选 ID 字典序；达到 K 或最佳收益 ≤0 停止 | 真实用于每轮决策 |
| PCB | 对每个 pending 找最大 Phi(T,T)−Phi(T−L,T) 的单候选推荐 | 只保留 >0 的推荐；每请求同收益按候选 ID；全局按收益降序、候选 ID、pending ID；重复 ID 跳过后继续扫描；达到 K 或推荐耗尽 | 真实用于静态推荐、全局排序；不更新 marginal |
| LDR | 每请求最深兼容 L>0 的推荐 | 按 L 降序、候选 ID、pending ID；唯一 ID 占预算；达到 K 或推荐耗尽 | 选择阶段不读；只在共同评价中使用 |
| WFC | 当前 E；a_p=max(0,L−E_p)，不兼容置零；q 为 a_p>0 的请求数，r=Σa_p | 每步按 q 降序、r 降序、候选 ID；更新 E；达到 K 或 q=0 停止 | 选择阶段不读；只在共同评价中使用 |

LDR/WFC 的输入对象虽然带有 T，但选择函数没有读取 T；T 对其可用候选的影响已体现在冻结兼容矩阵。PCB 并不计算跨请求联合收益；多请求推荐相同 ID 只计一次预算，也不为已被覆盖的请求重新寻找次优推荐。WFC 的 q 优先于 r，不能简称为“最大总推进”。

共同评价 `common.py:57` 使用 E_p=max({L_c: c∈S且兼容}∪{0})、G_p=T_p−E_p、C(S)=ΣPhi(G_p,T_p)。正式代码的 K 为等大小内存预算整除，`optimizer.py:65–85`；R 已保存对应逻辑 K。本轮不修改这些映射。

## 3. 为什么当前总体容易持平

### 3.1 正收益支持图

按每个快照只计一次的候选出现次数统计，避免将三个预算重复计入：

| 数据集 | 帮助 0 / 1 / 2 个 pending 的候选数 | 至少一个共享正收益候选的快照 |
|---|---:|---:|
| OpenHands | 12 / 2264 / 0 | 0/168 |
| AgentX | 484 / 54 / 30 | 22/23 |

这些是“快照内候选出现次数”，不是跨时间去重的物理状态数。本轮同时检查兼容度数与正收益度数，两者在当前输入逐项一致。来源 R 的 `input.candidates` 与 `benefit_matrix_ms`；完整聚合见 `structure_summary.json`。AgentX 的 66/69 共享预算条件来自 22 个快照，不是 66 个独立工作流。

OpenHands 每个正收益候选最多帮助一个 pending，符合第 4 节的可分定理；零收益候选不影响最优目标。AgentX 有共享，不能套用该定理。其 compatibility 是正式冻结输入的关系，本轮没有新原生 fork/同时 ready 或 GPU 物理共享证据，不把逻辑矩阵重新解释成这些证据。

### 3.2 预算和唯一前沿证书

OpenHands 的 K_min 全为 4；90/504 条件 K<4，全部位于 25% 预算；其余 414 条件 K≥4。AgentX K_min=2/1/0 的条件分别为 27/39/3，全部 K≥K_min。

本轮增加的是只读结构证书，不是新分配实验：对每个 E_p^max>0，查找满足 L_c=E_p^max 的兼容候选。**全部 573 条件中，每个这样的 pending 恰有一个最大前沿实现者。** 将这些必须保留的 ID 求并集 U，逐条件验证 |U| 等于冻结 K_min；在全部 483 个预算足够条件中，四种实际最终集合均恰等于 U。证书逐项保存最大实现者、U、K、Case ID 和 R 行号。

这比“K 大所以会持平”更强：这些记录中四种方法的实际轨迹仅选择了最终必须的状态，没有提前花预算保留额外、最终冗余的祖先。**K≥K_min 本身不保证任意贪心算法得到 U**；第 5 节数学乙二正好反驳该推断。

- OpenHands 的 90 个紧预算条件：独立支持使 PCB 与精确最优的结构一致；保存的 LDR/WFC 决策也恰好没有在预算截断处发生收益排序冲突。后者是数据事实，不是所有不同 T 输入上的定理。R:1 的 `rq3-openhands-main-g000-round2`（完整 ID 以原行及证书为准）是 K=2、K_min=4 的实例，选择 A/C 最深项，而非全保留。
- AgentX 的全部条件：达到候选可实现最深前沿，仍可有不可消除 residual；不能改写为所有 G=0。22 个共享快照的最终选择已达到候选上界，重排这些相同必要项不能改善目标。
- 全候选上界、预算足够和最终实际算法达到上界是三个不同命题；本轮逐项核对后二者，不从前者推断。

### 3.3 排序变化究竟发生在哪里

完整清单见第 7 节及 `order_difference_evidence.json`，包括所有候选、T、兼容集合、各策略轨迹、首次分歧和最终 E/G。以下不是根据相同终点反推过程：

- OpenHands g051：C/D 的 L 都为 7808，T 为 7808/7999；共同前缀 A/B 之后，FlowState 的边际值分别为 299.473119577 / 299.520864654，选择 D 在前。LDR/WFC 因同位置或同 q/r 按 ID 选择 C 在前。K=5/10/15 都能容纳四个唯一最深项，因此顺序变化未被预算截断。来源 R:133–135。
- AgentX g000/g007/g014：FlowState 首选共享总收益项，PCB/LDR 首选单分支更深推荐。二者恰是不同 pending 的唯一最深实现者，K 均≥2，第二步补入另一个后达到上界。这里是真实记录中的联合收益排序效应，但不是最终目标增益。
- AgentX g005/g006/g010/g017/g018/g021：WFC 首选 q=2 的共享项，FlowState 首选更大 modeled marginal 的 q=1 项；共享浅项对另一 pending 仍是唯一最深实现者，之后仍必须保留。两种顺序最终都只选这两个必要项。不同 T、共享覆盖和非线性成本确实可能改变排序，不能说当前规则从未体现差异。

## 4. 数学分析：充分条件与边界

以下证明以非负、固定 b，空集收益零、等大小 cardinality 约束、无额外物理耦合为前提。数学上的精确最大化与浮点实现分开：若非相等收益差落入 FlowState 的 1e−9 容差，可能与 PCB 精确浮点排序不同；声称同 ID 集合还需最大者唯一或一致的同分规则，不能只证明同目标。

### 4.1 独立支持下的精确化简

若不同 workflow 的候选只影响本 workflow，则 F(S)=Σ_w F_w(S∩C_w)，仍需在各 workflow 间分配 K；一个 workflow 内多个共享 pending 的子问题未因此消失。

更强条件是每个正收益候选最多影响一个 pending。把候选按其唯一正收益请求分组 C_p，令 v_p=max_c b_pc，选择达到最大值的 c_p。任一可行 S 在每组只保留收益最大的一项不降低目标且不增加预算，再将该项替换为 c_p 也不降低目标。于是最优目标就是最大的 min(K,m) 个正 v_p 之和，m 为 v_p>0 的请求数。反向选择这些 c_p 达到该上界，证明完成。共享零收益或不兼容项无影响，K=0 或 m=0 返回空；K>m 不必填满；同分时有多个最优解。

PCB 实现这一化简。FlowState 在精确比较下每步选择尚未服务的最大 v_p，选定该请求的最好项后同组其他项 marginal=0，因此也实现化简。OpenHands 符合该充分条件；这是根据输入结构给出的证明，与已有 Exact OPT 的有限样本匹配无关。

### 4.2 深度支配、T 的影响和 LDR

同一 pending、同 T 下，Phi 对 G 单调递增，则更深兼容项的 b 不小于浅项。若要对整个集合目标证明 c₂ 支配 c₁，还要求 c₂ 覆盖 c₁ 的所有正收益请求且对每个请求 b_pc₂≥b_pc₁。分支专属后代可能丢失共享祖先的其他分支，因此“更深”不自动意味着全局支配。

正式 Phi 用 Ki-token 表示为 a g+b g t+c g²，其中 a=37.828150、b=0.345974143、c=−0.156201917。因此

B(l,t)=Phi(t,t)−Phi(t−l,t)=[a+(b+2c)t]l−c l²，且 b+2c=0.033570309。

对 0≤g≤t≤128，∂Phi/∂g=a+b t+2c g≥a+(b+2c)t>0，满足单调性。对相同 t，B 严格随 l 增长。因此“独立正收益支持 + 各请求 T 相同 + 同分一致”足以使 LDR、PCB、FlowState 采用同一最好请求顺序。更一般只需这些请求的最大 L 排序与 B(L,T) 排序在相关预算边界一致。T 不同则不保证；g051 已证明即使 L 相同也能改变排序。T 系数较小只是公式性质，不是可以忽略它的证明。

### 4.3 WFC 和 marginal 的关系

在当前集合前沿 e 下，一个请求由 e 推进到 l 的收益为

B(l,t)−B(e,t)=(l−e)[a+(b+2c)t−c(l+e)]。

因此相同 r 并不意味着相同 marginal；q 优先还可能压过任意较大的 r。一个足够条件仍是独立支持、公共 T：未服务组的正收益候选 q=1、e=0，已服务组最深项之外都为零，于是 WFC 在组间按最大 L 排序，与上述三者一致。

对于一般共享输入，只有当**每个可达前缀**的 q/r 排序与上述收益和排序一致（包括 tie）时，才能归纳得到同序；这是可检验的充分条件，不是现有数据自动保证。即使 Phi 是正系数线性 gap 模型，FlowState 只最大化 r，WFC 仍首先最大化 q；还需 q 相同或 q/r 顺序相容，不能仅凭线性便声称等价。

### 4.4 共享、紧预算也可以持平，但不保证

若一个候选同时是所有有效 pending 唯一最深实现者，则四者在 K≥1 时都选它；共享并不自动产生区分力。即使 K<K_min，也可能在预算截断内选择相同的占优项。

可直接证明的紧预算共享构型：两个独立 workflow，每个各有两个 pending，只各提供一个共享候选；两候选具有相同 T、不同正 L，K=1<K_min=2。PCB/LDR 各 workflow 的两个推荐去重；FlowState 收益为 2B(L,T)，WFC q 同为2再按 2L，因此四者均选更深项。这是解析存在性说明，未计入正式数据或第 5 节验证结果。

失效条件包括：跨请求 T 导致深度与收益顺序反转、共享总收益与单请求推荐不一致、q 优先与收益大小冲突、先选共享祖先后又需要多个深分支造成预算冗余、不同大小或额外耦合约束。满足其中某项只意味着可能失效，不保证 FlowState 获胜。

## 5. 最小合法反例：仅为数学与 CPU 机制验证

所有例子使用未修改的四个真实选择函数与正式 Phi；相同候选、每候选一个相同逻辑单位、全部驻留、相同预算；ID 为 a<b<c，T=anchor=FA frontier，严格按冻结同分规则。合法谱系由 `is_compatible` 产生，不手填收益矩阵。单位 Ki=1024 tokens。下文成本均为总 C(S)，完整逐候选收益、FlowState 每轮 marginal、PCB/LDR 推荐序列、WFC 排名与微型穷举最优值见 `analytical_examples.json`。这些状态是假设存在的合法数学状态，不是 GPU 已物化证据。

**甲：不同 T 使相同深度产生不同收益。** 两个独立 workflow，p1:T=8Ki、p2:T=128Ki；a:L=4Ki 仅兼容 p1，b:L=4Ki 仅兼容 p2，c:L=2Ki 仅兼容 p1，K=1。FlowState/PCB 选 b；LDR 的深度 tie 与 WFC 的 q/r tie 均按 ID 选 a。这里选择边界只差恢复估值，不能归因于额外在线信息。

**乙：共同祖先和两条深分支。** 两个 pending 均 T=8Ki，根谱系 r 在 L=4Ki 保存 a，然后真实意义上的数学 fork 为 r/x、r/y；各分支在 L=6Ki 保存 b/c，仅分别兼容 p1/p2。共同 ancestor 在 fork 之前，两个深分支在 fork 之后，不是仅 prompt 相同。

- K=1：FlowState 的 2B(4,8) 大于单分支 B(6,8)，选 a；WFC q=2 也选 a。PCB 各请求只推荐最深 b/c，LDR 同样，按 ID 选 b。
- K=2：相同输入不变。FlowState/WFC 先 a，之后 b/c 增益同分，选 b；PCB/LDR 直接保留 b,c。此时 **FlowState 严格更差**，且 K=K_min=2 仍未达到全候选前沿。它反驳“预算足够就保证贪心达到上界”，也反驳“一般 FlowState 贪心必不劣于 PCB”。

**丙：覆盖数量不代表收益大小。** 将乙的 fork/共享 a 改为 L=1Ki，其余 b/c:L=6Ki、T=8Ki、K=1。WFC 优先 q=2 的 a；FlowState/PCB/LDR 选择单分支 b。用同一合法模型即可产生 WFC 严格损失。

本轮没有做随机搜索或筛选真实工作负载；只验证上述三个构型（乙有两个预算）。不声称是所有模型下候选数的严格最小下界。第 4.4 节的另一个紧预算持平构型只作符号推导。

## 6. 科研判断与后续建议

**现有负载缺乏区分力：已验证。** 对最终集合、E/G 和 modeled objective，全量持平，无独特收益。

**当前结构允许简单方法达到同等结果：有可分证明及逐条件证书。** OpenHands 的 A1 结构使 PCB 精确化简，90 个紧预算条件仍属于这个范畴；AgentX 的实际候选最大前沿由极少数唯一必要项实现，所有正式预算都容纳它们，实际轨迹也都没有浪费预算。这两个原因应分别表述。

**是否无法判断主要原因：不能笼统回答未知，也不能作过度因果归因。** 当前总体中上述结构障碍已明确；但不能从冻结观察给出它们在更广泛真实应用中各占多少、或在新容量合同下谁更重要。反例证明一般不等价，也证明 FlowState 可以失败；它们不是性能优势证据。不能用相对 StandaloneTopK 的去冗余收益替代相对 PCB/LDR/WFC 的额外收益。

最多两个后续设计，均未执行：

1. **预注册 CPU 结构机制研究。** 独立于冻结总体，系统枚举小型合法谱系、固定 Phi、T、L 与事先规定预算，按独立/共享、K 与 K_min、排序冲突分类；共同报告所有四种策略相对 Exact OPT 的胜平负和贪心失败。目标是区分结构性等价域与启发式偶然一致，不按 FlowState 胜负筛选。其结论只属于数学模型。
2. **原生真实共享队列的可行性验证，再决定是否分配比较。** 事先固定应用、native fork/ready 身份、allocation barrier 与物理 retention 合同；完整保留所有准入事件，报告能否自然产生共享祖先、多个深分支以及真实 K<K_min。不事后降低预算或回筛旧样本。只有可行性通过才另行批准策略比较与必要 runtime 测量，本轮不启动。

当前可以保留有限总体匹配、共享收益影响排序、去冗余机制等结论；必须弱化“一般全局边际优化在这些工作负载上提供强基线之外额外收益”和“位置感知模型已转化为最终分配优势”。无需为本次审计修改算法。完成本报告后停止。

## 7. 全部顺序差异的输入与选择证据

以下每行代表同一快照的三个冻结预算；没有遗漏顺序差异快照。每个 Case ID 需与表内 budget/K 配对。全量输入包含未选候选，保存在 R 指定行及 `order_difference_evidence.json`；下表只缩写 checkpoint 前缀，不改变 ID，完整 ID 可由该文件恢复。

| Case ID | R 行号（25/50/75%） | K（三预算） / K_min | 与 FlowState 顺序不同的方法 |
|---|---|---|---|
| rq3-openhands-main-g051-round5 | 133/134/135 | 5/10/15 / 4 | LDR,WFC |
| rq3-agentx-g000-epoch109 | 505/506/507 | 27/54/81 / 2 | PCB,LDR |
| rq3-agentx-g005-epoch8 | 520/521/522 | 2/4/6 / 2 | WFC |
| rq3-agentx-g006-epoch35 | 523/524/525 | 8/17/26 / 2 | WFC |
| rq3-agentx-g007-epoch21 | 526/527/528 | 5/10/15 / 2 | PCB,LDR |
| rq3-agentx-g010-epoch38 | 535/536/537 | 9/19/28 / 2 | WFC |
| rq3-agentx-g014-epoch9 | 547/548/549 | 2/4/6 / 2 | PCB,LDR |
| rq3-agentx-g017-epoch12 | 556/557/558 | 3/6/9 / 2 | WFC |
| rq3-agentx-g018-epoch30 | 559/560/561 | 7/15/22 / 2 | WFC |
| rq3-agentx-g021-epoch149 | 568/569/570 | 37/74/111 / 2 | WFC |

### rq3-openhands-main-g051-round5

来源 R:133；原始 `/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:133`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[11200, 8320, 7808, 7999]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| A_TURN_005 | 11200 | p1 |
| B_TURN_005 | 8320 | p2 |
| C_TURN_005 | 7808 | p3 |
| D_TURN_004 | 7808 | p4 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | A_TURN_005 → B_TURN_005 → D_TURN_004 → C_TURN_005 | 7.554497620 |
| PCB | A_TURN_005 → B_TURN_005 → D_TURN_004 → C_TURN_005 | 7.554497620 |
| LDR | A_TURN_005 → B_TURN_005 → C_TURN_005 → D_TURN_004 | 7.554497620 |
| WFC | A_TURN_005 → B_TURN_005 → C_TURN_005 → D_TURN_004 | 7.554497620 |

LDR 首次分歧在第 3 步，共同前缀 E=[11200, 8320, 0, 0]：FlowState 选 D_TURN_004（q=1、r=7808、marginal=299.520864654）；该基线选 C_TURN_005（q=1、r=7808、marginal=299.473119577）。

WFC 首次分歧在第 3 步，共同前缀 E=[11200, 8320, 0, 0]：FlowState 选 D_TURN_004（q=1、r=7808、marginal=299.520864654）；该基线选 C_TURN_005（q=1、r=7808、marginal=299.473119577）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g000-epoch109

来源 R:505；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:1`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[110527, 20223, 110527]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0000 | 37760 | p1,p3 |
| req0005 | 51328 | p1 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0000 → req0005 | 9197.818797961 |
| PCB | req0005 → req0000 | 9197.818797961 |
| LDR | req0005 → req0000 | 9197.818797961 |
| WFC | req0000 → req0005 | 9197.818797961 |

PCB 首次分歧在第 1 步，共同前缀 E=[0, 0, 0]：FlowState 选 req0000（q=2、r=75520、marginal=3481.852412350）；该基线选 req0005（q=1、r=51328、marginal=2470.221830882）。

LDR 首次分歧在第 1 步，共同前缀 E=[0, 0, 0]：FlowState 选 req0000（q=2、r=75520、marginal=3481.852412350）；该基线选 req0005（q=1、r=51328、marginal=2470.221830882）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g005-epoch8

来源 R:520；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:16`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[54527, 1344]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0000 | 384 | p1,p2 |
| req0001 | 43840 | p1 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0001 → req0000 | 605.802845913 |
| PCB | req0001 → req0000 | 605.802845913 |
| LDR | req0001 → req0000 | 605.802845913 |
| WFC | req0000 → req0001 | 605.802845913 |

WFC 首次分歧在第 1 步，共同前缀 E=[0, 0]：FlowState 选 req0001（q=1、r=43840、marginal=1982.352784905）；该基线选 req0000（q=2、r=768、marginal=29.101911990）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g006-epoch35

来源 R:523；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:19`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[125248, 125759]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0001 | 48448 | p1,p2 |
| req0011 | 103104 | p2 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0011 → req0001 | 6832.743154525 |
| PCB | req0011 → req0001 | 6832.743154525 |
| LDR | req0011 → req0001 | 6832.743154525 |
| WFC | req0001 → req0011 | 6832.743154525 |

WFC 首次分歧在第 1 步，共同前缀 E=[0, 0]：FlowState 选 req0011（q=1、r=103104、marginal=5807.509134874）；该基线选 req0001（q=2、r=96896、marginal=4668.125454620）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g007-epoch21

来源 R:526；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:22`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[73279, 88959, 88959]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0001 | 40064 | p1,p2 |
| req0002 | 44800 | p2 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0001 → req0002 | 9221.218683271 |
| PCB | req0002 → req0001 | 9221.218683271 |
| LDR | req0002 → req0001 | 9221.218683271 |
| WFC | req0001 → req0002 | 9221.218683271 |

PCB 首次分歧在第 1 步，共同前缀 E=[0, 0, 0]：FlowState 选 req0001（q=2、r=80128、marginal=3646.365109330）；该基线选 req0002（q=1、r=44800、marginal=2081.553760983）。

LDR 首次分歧在第 1 步，共同前缀 E=[0, 0, 0]：FlowState 选 req0001（q=2、r=80128、marginal=3646.365109330）；该基线选 req0002（q=1、r=44800、marginal=2081.553760983）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g010-epoch38

来源 R:535；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:31`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[94079, 94079]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0002 | 44160 | p1,p2 |
| req0019 | 86080 | p1 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0019 → req0002 | 3556.676289099 |
| PCB | req0019 → req0002 | 3556.676289099 |
| LDR | req0019 → req0002 | 3556.676289099 |
| WFC | req0002 → req0019 | 3556.676289099 |

WFC 首次分歧在第 1 步，共同前缀 E=[0, 0]：FlowState 选 req0019（q=1、r=86080、marginal=4542.999186872）；该基线选 req0002（q=2、r=88320、marginal=4109.691493535）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g014-epoch9

来源 R:547；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:43`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[42559, 42559]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0000 | 34304 | p1,p2 |
| req0001 | 34560 | p1 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0000 → req0001 | 809.007211135 |
| PCB | req0001 → req0000 | 809.007211135 |
| LDR | req0001 → req0000 | 809.007211135 |
| WFC | req0000 → req0001 | 809.007211135 |

PCB 首次分歧在第 1 步，共同前缀 E=[0, 0]：FlowState 选 req0000（q=2、r=68608、marginal=2978.561876055）；该基线选 req0001（q=1、r=34560、marginal=1501.712928553）。

LDR 首次分歧在第 1 步，共同前缀 E=[0, 0]：FlowState 选 req0000（q=2、r=68608、marginal=2978.561876055）；该基线选 req0001（q=1、r=34560、marginal=1501.712928553）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g017-epoch12

来源 R:556；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:52`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[78463, 42303, 15744]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0000 | 320 | p2,p3 |
| req0001 | 41984 | p2 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0001 → req0000 | 4643.435568925 |
| PCB | req0001 → req0000 | 4643.435568925 |
| LDR | req0001 → req0000 | 4643.435568925 |
| WFC | req0000 → req0001 | 4643.435568925 |

WFC 首次分歧在第 1 步，共同前缀 E=[0, 0, 0]：FlowState 选 req0001（q=1、r=41984、marginal=1870.390037366）；该基线选 req0000（q=2、r=640、marginal=24.267784471）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g018-epoch30

来源 R:559；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:55`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[126847, 124287]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0001 | 47360 | p1,p2 |
| req0012 | 105344 | p2 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0012 → req0001 | 6745.074698151 |
| PCB | req0012 → req0001 | 6745.074698151 |
| LDR | req0012 → req0001 | 6745.074698151 |
| WFC | req0001 → req0012 | 6745.074698151 |

WFC 首次分歧在第 1 步，共同前缀 E=[0, 0]：FlowState 选 req0012（q=1、r=105344、marginal=5963.868027112）；该基线选 req0001（q=2、r=94720、marginal=4548.133889245）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

### rq3-agentx-g021-epoch149

来源 R:568；原始 `/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:64`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。

输入 T 向量（tokens，按冻结 pending 顺序）：[106367, 111423, 1727, 3839]。候选的兼容索引用 p1、p2…对应此顺序。

| 最终必要项 | L（tokens） | 兼容 pending |
|---|---:|---|
| req0000 | 384 | p1,p3 |
| req0008 | 96704 | p1 |

| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |
|---|---|---:|
| FlowState | req0008 → req0000 | 7239.810795909 |
| PCB | req0008 → req0000 | 7239.810795909 |
| LDR | req0008 → req0000 | 7239.810795909 |
| WFC | req0000 → req0008 | 7239.810795909 |

WFC 首次分歧在第 1 步，共同前缀 E=[0, 0, 0, 0]：FlowState 选 req0008（q=1、r=96704、marginal=5294.784966778）；该基线选 req0000（q=2、r=768、marginal=29.743931855）。

持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。

## 8. 解析反例的逐步数值

由 `audit.py` 直接调用冻结实现；此处仅汇总数学验证。穷举只作用于这些 3 候选小例子，不作用于正式总体。

| 数学条件 | K | FlowState 顺序 / C | PCB 顺序 / C | LDR 顺序 / C | WFC 顺序 / C | 小例子 Exact OPT |
|---|---:|---|---|---|---|---:|
| 数学甲 | 1 | b / 8095.002144368 | b / 8095.002144368 | a / 8111.115892688 | a / 8111.115892688 | 8095.002144368 |
| 数学乙一 | 1 | a / 319.769083808 | b / 395.337701084 | b / 395.337701084 | a / 319.769083808 | 319.769083808 |
| 数学乙二 | 2 | a,b / 240.451620524 | b,c / 161.134157240 | b,c / 161.134157240 | a,b / 240.451620524 | 161.134157240 |
| 数学丙 | 1 | b / 395.337701084 | b / 395.337701084 | b / 395.337701084 | a / 553.035416150 | 395.337701084 |

### 数学甲 的评分过程

候选独立收益向量（pending 顺序同定义，modeled ms）：a=154.886080560,0.000000000；b=0.000000000,170.999828880；c=76.818232612,0.000000000。

FlowState 前缀 []：a marginal=154.886080560，b marginal=170.999828880，c marginal=76.818232612；选择 b。

PCB 推荐排序：p2:b（170.999828880） → p1:a（154.886080560）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

LDR 推荐排序：p1:a（4096.000000000） → p2:b（4096.000000000）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

WFC 前缀 []：a（q=1,r=4096） → b（q=1,r=4096） → c（q=1,r=2048）；选择 a。

### 数学乙一 的评分过程

候选独立收益向量（pending 顺序同定义，modeled ms）：a=154.886080560,154.886080560；b=234.203543844,0.000000000；c=0.000000000,234.203543844。

FlowState 前缀 []：a marginal=309.772161120，b marginal=234.203543844，c marginal=234.203543844；选择 a。

PCB 推荐排序：p1:b（234.203543844） → p2:c（234.203543844）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

LDR 推荐排序：p1:b（6144.000000000） → p2:c（6144.000000000）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

WFC 前缀 []：a（q=2,r=8192） → b（q=1,r=6144） → c（q=1,r=6144）；选择 a。

### 数学乙二 的评分过程

候选独立收益向量（pending 顺序同定义，modeled ms）：a=154.886080560,154.886080560；b=234.203543844,0.000000000；c=0.000000000,234.203543844。

FlowState 前缀 []：a marginal=309.772161120，b marginal=234.203543844，c marginal=234.203543844；选择 a。

FlowState 前缀 ['a']：b marginal=79.317463284，c marginal=79.317463284；选择 b。

PCB 推荐排序：p1:b（234.203543844） → p2:c（234.203543844）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

LDR 推荐排序：p1:b（6144.000000000） → p2:c（6144.000000000）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

WFC 前缀 []：a（q=2,r=8192） → b（q=1,r=6144） → c（q=1,r=6144）；选择 a。

WFC 前缀 ['a']：b（q=1,r=2048） → c（q=1,r=2048）；选择 b。

### 数学丙 的评分过程

候选独立收益向量（pending 顺序同定义，modeled ms）：a=38.252914389,38.252914389；b=234.203543844,0.000000000；c=0.000000000,234.203543844。

FlowState 前缀 []：a marginal=76.505828778，b marginal=234.203543844，c marginal=234.203543844；选择 b。

PCB 推荐排序：p1:b（234.203543844） → p2:c（234.203543844）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

LDR 推荐排序：p1:b（6144.000000000） → p2:c（6144.000000000）；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。

WFC 前缀 []：a（q=2,r=2048） → b（q=1,r=6144） → c（q=1,r=6144）；选择 a。

## 9. 完整性与复核边界

`integrity.json` 记录本轮所用策略、核心模型、P1-B 记录、两个正式原始结果及前序报告的 SHA-256 前后对照；全部匹配。四项 implementation_lock 摘要全部通过，573 条 FlowState 输出另与各自原始正式文件的 Case ID、snapshot digest、预算、集合及成本交叉核对。

本轮生成：`audit.py`、`append_evidence.py`、`structure_summary.json`、`frontier_certificates.json`、`order_difference_evidence.json`、`analytical_examples.json`、`integrity.json` 和本报告。前两个脚本只写本隔离目录；可用 `PYTHONDONTWRITEBYTECODE=1 python3 evaluation/p1c_global_allocation_equivalence/audit.py` 复核。`append_evidence.py` 仅更新本报告附录。

超出旧实验手册的新信息已明确标记：全量唯一最大前沿证书来自冻结输入的 CPU 只读分析；解析反例来自人工合法数学构型，未进入正式 population。尚未提供新的原生共享事件或 runtime 证据，因此真实新工作流中的收益与额外控制成本保持 UNKNOWN。
