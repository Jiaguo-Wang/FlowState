# FlowState 工作流结构与预算敏感性最终实验报告

日期：2026-10-28。判定：**READY**（任务 1/2/4 完成，任务 3 按条件句判 NOT_SUPPORTED 并如实记录；同日 P0-1 收尾复核完成：Strict20 最优集合唯一性 20/20、结构性非嵌套 11/15，过强结论与 GPU 证据归属已修正，见 §4.1/§5/§6）。全程 CPU-only 只读：未启动 GPU、未修改冻结成本函数/候选准入语义/历史正式数据/Runtime/正式 Solver、未筛选有利实例。

产物目录 [paper_workload_structure_final/](paper_workload_structure_final/)：
[legacy_recheck_scan.py](paper_workload_structure_final/legacy_recheck_scan.py)（573 复核+AgentX 扫描，0.4s 可复算）、[strict20_sensitivity.py](paper_workload_structure_final/strict20_sensitivity.py)、[strict20_optima_nesting.py](paper_workload_structure_final/strict20_optima_nesting.py)（P0-1 最优集唯一性/嵌套性全枚举复算）、[agentx_budget_scan.csv](paper_workload_structure_final/agentx_budget_scan.csv)（736 行）、[legacy573_recheck.json](paper_workload_structure_final/legacy573_recheck.json)（579 检查）、[strict20_sensitivity.csv](paper_workload_structure_final/strict20_sensitivity.csv)（80 行）、[strict20_optima_nesting.json](paper_workload_structure_final/strict20_optima_nesting.json)/[.csv](paper_workload_structure_final/strict20_optima_nesting.csv)（20 条件×全子集枚举）、[generator_feasibility.json](paper_workload_structure_final/generator_feasibility.json)、两份 source_hashes.json。上游只读引用：gate1_step1_allocation_equivalence（573 条件冻结矩阵）、greedy_swap_structure_e3_step1（Strict20 七策略冻结结果）、T1 结构审计（paper_audits/t1_structure_budget）。

## 0. 口径（全部 modeled ms，Phi 冻结系数，非 GPU 延迟）

目标 J(S)=Σ_p Phi(T_p−E_p, T_p)，E_p=选中且兼容候选的最大 position；benefit_ms 为其逐格线性化证书。**命名**：本文「Strict20」指 5 个**真实轨迹衍生的受控共享分支实例**（快照 w1–w4、B5-C4，Trace-derived Controlled Composition 运行时采集，见任务 3）× K∈{1..4} 共 20 个条件，**不是合成实例**。四方法同一目标：**Exact**（K<K_min 全枚举、K≥K_min 代数最优并小组合数枚举交叉验证、并列取 (大小,ID 字典序) 最小）、**Marginal Greedy**（Φ 边际最大，零边际停）、**WFC**（(可推进 Pending 数 q, token 推进 r, ID) 排序，q=0 停，P1 B3 直译）、**StandaloneTopK**（独立总 benefit 排序取前 K 正收益，e3 直译）。**实现校准：在 69 条 AgentX 原 budget 条件上，Greedy 与 WFC 分别 69/69 逐位复现冻结 flowstate/wfc 选集，Phi 成本复算 69/69**——扫描结果与历史正式数据同一语义。

## 1. 任务 1：历史 573 条件只读复核（579 检查全 PASS）

逐条重算结构指标与证书公式，并与 gate1 statistics 冻结字段核对：OpenHands 504/504 支持可分（共享 0、重叠 504、紧预算 90）→ SEPARABLE_SUPPORT_TOPK 代数证书，Greedy 504/504 达最优；AgentX 69/69 K≥K_min（66 共享+3 可分）→ 66 条 ALL_CANDIDATE_FRONTIERS_ATTAINED + 3 条 TopK 证书，Greedy 69/69 达最优；四策略（FlowState/PCB/LDR/WFC）cost 573×4 全部复算一致（Phi 域与 benefit 线性化互证 <1e-6）。聚合与 [paper_allocation_structure_legacy573.csv](paper_allocation_structure_legacy573.csv) 冻结行逐项相等。573/573 无 Strict Runtime 认证（口径不变）。

## 2. 任务 2：AgentX 固定绝对预算扫描（23 快照 × K∈{1..8} × 4 方法 = 736 行）

同快照跨 budget_ratio 矩阵逐字节一致（校验过）；扫描只改 K，执行历史/兼容矩阵/候选状态/成本模型零改动。Exact 在 K<K_min（本数据仅 K_min=2 的 9 个快照于 K=1）用全枚举（C(n,1)≤149），K≥K_min 用代数式并对 C(n,K)≤200,000 的 37 个 (快照,K) 格额外枚举复核代数式全部一致。

| 方法 | K=1（含 9 个 K<K_min 紧预算格） | K=2..8 |
|---|---|---|
| Marginal Greedy | **23/23 最优且同集（gap=0，含 9/9 紧预算格）** | **23/23 最优且同集（全部 K）** |
| WFC | 17/23；**6 个失败全部是 K_min=2 且有共享的快照，gap=433..5265ms（中位 1139）** | 23/23 最优且同集（全部 K） |
| StandaloneTopK | 23/23 最优且同集 | 18..20/23 达 Exact J；**7 个快照持续非同集（gap 至 2333.7ms），不随 K 增大修复** |

要点：①AgentX 结构（嵌套/不交支持、K_min≤2）下 **Marginal Greedy 对预算完全不敏感**，连 K<K_min 也精确（K=1 时边际最大=单点最深，贪心天然选对）；②**WFC 的失败模式是紧预算下的覆盖数偏置**——K=1 时它选"推进最多 Pending"的共享候选而非最深候选，预算一旦 ≥K_min 立即恢复精确；③SA 不感知共享，在 7 个共享快照上持续选重复覆盖集，增加预算不能修复（选集含共享大候选，K 增大只加深同一覆盖）。

## 3. 任务 3：Strict20 生成器可行性 = NOT_SUPPORTED

五快照（w1–w4、B5-C4）均为**真实轨迹衍生的运行时采集产物**（workload_type=Trace-derived Controlled Composition；含引擎配置/模板 sha256 指纹与完整采集链 capture_manifest/runtime.log/gpu_before/physical_state_census，逐项核验在位，见 [generator_feasibility.json](paper_workload_structure_final/generator_feasibility.json)）。仓内不存在支持固定参数（共享度/分支深度）与随机种子的 Strict20 结构生成器——引用快照的代码全部为审计/验证脚本（读入方）。按 spec 条件句**不生成合成实例**。因此本轮全部实例（573 历史条件 + Strict20 + AgentX 23 快照）均为真实轨迹衍生，合成压力实例出现次数 = 0；"真实 vs 合成"对照表退化为单列事实，如实报告。

## 4. 任务 4：四方法预算敏感性与非嵌套最优集合（两域汇总）

### 预算敏感性（gap_vs_exact 中位 ms）

| 域 | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| Strict20（5 快照，K_min=4）：Greedy / WFC / SA | 0 | **77.7** | 31.8 | 29.3（SA 36.6） |
| AgentX（23 快照，K_min≤2）：Greedy / WFC / SA | 0 / **1139.4（6 格）** / 0 | 0 / 0 / 0* | 0 / 0 / 0* | 0 / 0 / 0* |

*SA 在 K≥2 有 7 个快照持续 gap（中位仍 0：失败集中于少数共享快照，max 2333.7ms）。

**敏感性形状非单调且结构依赖**：Strict20 在 K=2 出现 gap 峰值（首次允许"两分支联合替换祖先"而 Greedy 已在 step1 锁定共享祖先 N20），K3/K4 递减但不清零（K4 预算充分仍 29.3 中位）；AgentX 上 Greedy/WFC 全程零 gap（WFC 仅 K=1 紧预算谷）。

### 非嵌套最优集合比例（相邻预算转移的选集包含检验）

| 域 | Exact 终集非嵌套 | Greedy 终集非嵌套 |
|---|---|---|
| Strict20（5 快照 × 3 转移） | **11/15（K1→K2 5/5、K2→K3 4/5、K3→K4 2/5）** | **0/15** |
| AgentX 扫描（23 快照 × 7 转移） | **0/161** | **0/161** |

这是"不可撤销早期选择"的形式化证据：**Greedy 终集跨预算恒嵌套（贪心前缀结构，两域 176 次转移零例外）**；当 Exact 最优集本身嵌套（AgentX 嵌套/不交支持、OpenHands 可分）时该性质无害——Greedy 全程最优；当 Exact 需要跨预算重组（Strict20 层次共享：K1 公共祖先在 K2 最优集消失）时，前缀锁定与最优路径结构性冲突，15/20 条件次优。本数据内观察到的关联是单向的：Greedy 出现损失的条件均伴随 Exact 非嵌套；其逆（非嵌套 ⟹ 必有损失）与一般充分性均未证明，不作定理表述。

### 4.1 最优集合唯一性与嵌套性复核（P0-1 收尾，只读全枚举）

对 5 快照 × K∈{1..4} 共 20 条件，用冻结目标对全部 |S|≤K 的子集（每条件 2^7=128 个）穷举，容差 1e-9 下收集**全部**达到最优 J* 的集合（[strict20_optima_nesting.py](paper_workload_structure_final/strict20_optima_nesting.py)，明细 [strict20_optima_nesting.json](paper_workload_structure_final/strict20_optima_nesting.json)）。**枚举校准：按 (大小, ID 字典序) 最小 tie-break 选出的终集与 e3 冻结 Exact 选集 20/20 逐位一致**；冻结 Exact 选集与既有 11/15 统计不变。

| 复核项 | 结果 |
|---|---|
| 最优集合唯一的条件 | **20/20**（无任何并列最优） |
| 相邻转移（15 个）中 tie-break 终集非嵌套 | 11（K1→K2 5/5、K2→K3 4/5、K3→K4 2/5，与既有统计一致） |
| 其中**结构性非嵌套**（不存在任何嵌套最优集合对 S_K⊆S_{K+1}） | **11/11 全部**——因最优集唯一，tie-break 非嵌套即必然结构，任何并列选择/tie-breaking 都无法恢复嵌套 |
| 存在嵌套最优对的转移 | 4（B5-C4 K3→K4、w3 K2→K3、w3 K3→K4、w4 K3→K4；即嵌套转移本身，无额外自由度） |
| 存在 K=1→4 完整嵌套最优解链的快照 | **0/5** |

**一句结论：Strict20 的 15 个相邻预算转移中，11 个的非嵌套属于必然的最优解结构（每条件最优集合唯一，"按固定 tie-breaking 选出的最优集合非嵌套"与"任何最优解都无法形成嵌套结构"两口径完全重合）；且 5 个快照无一存在 K1→K4 完整嵌套最优链。**

### 选择差异（Strict20 20 条件两两同集）

Greedy–WFC 20/20 同集；Greedy–SA 16/20；Exact 与三者均 5/20（仅 K1）。AgentX 扫描：Greedy 184/184 格与 Exact 同集（**该 184/184 为当前 AgentX 结构（嵌套/不交支持、K_min≤2）与预算范围 K∈{1..8} 内的实验发现，不构成任意嵌套工作负载下的一般定理**——对照域 Strict20 上同样规则即 15/20 次优）；WFC 178/184（6 个失败格全在 K=1）；SA 157/184。

## 5. 必答四问

**Q1 哪些结构能由简单规则精确求解？** ①支持可分（正收益支持≤1）：逐 Pending 行最大 TopK 代数精确——OpenHands 504/504、AgentX 3/69；②嵌套/不交支持且 K≥K_min：Greedy 与 WFC 都精确——AgentX 66/69；③K=1（任意结构，除 WFC 偏置）：Greedy/SA 精确——AgentX 23/23、Strict20 5/5；④AgentX 紧预算 K=1<K_min：Greedy 仍 9/9 精确（边际最大=最深单点）。本两域数据中的共同伴随条件是"Exact 最优集跨预算嵌套 ⟹ 简单规则无损"（充分方向在全部观测条件成立）；一般充要条件未证，不作定理表述。

**Q2 哪些条件下共享状态形成真实预算竞争？** ①层次共享+中间预算（Strict20 K2–K4，K<K_min=4 期间）：gap 中位 77.7/31.8/29.3ms，且 K4 预算充分仍不清零；②K=1 紧预算+共享候选对 WFC（AgentX 6 快照，gap 433–5265ms）；③C/K3（K=3<K_min=4，既有 GPU 配对证据：FS−StandaloneTopK Pending B −163.2ms、FS−WFC Pending B −84.4ms，两条基线机制相互独立，归属见 §6）。判别式：存在共享候选且预算不足以"先买共享再补分支"或规则偏置（q 覆盖数/独立总量）压过深度时，竞争成为实际损失。

**Q3 哪些情况下 Greedy 产生不可撤销的早期选择？** 机制证据链：Greedy 终集跨预算嵌套 0/176 违反（结构性无法撤销）；Strict20 15 个失败条件的分歧全部在 step1（选共享祖先 N20∉Exact 最优集，K3 时其边际贡献已为 0 仍占名额；w1/w2-K3 需三同步替换，固定首改善 1-Swap 也停在局部最优）；对照域（AgentX/OpenHands）Exact 也嵌套，同样不可撤销但无代价。**不可撤销 ⟸ 恒成立（贪心前缀结构）；在 Strict20 中，造成损失的条件均为 Exact 结构性非嵌套（11/15，最优集唯一，见 §4.1）——这是当前数据集内的观察，一般充要性未证。**

**Q4 真实轨迹衍生 vs 合成结构中各出现多少次？** 真实轨迹衍生：573 历史条件（OpenHands 168 快照 504 条件 + AgentX 23 快照 69 条件）+ Strict20 五快照 20 条件；合成压力实例：**0**（生成器不存在，NOT_SUPPORTED）。上述全部现象（TopK 精确、全前沿精确、WFC 覆盖偏置、非嵌套/不可撤销）均来自真实轨迹衍生实例。

## 6. 主文与 Appendix 划分建议

**主文（Evaluation）**：①可分⇒TopK 精确 + K≥K_min⇒Greedy/WFC 精确（573 条件双证书，说明"重叠存在≠需复杂求解器"）；②层次共享下 gap 非单调（Strict20 K2 峰值 77.7 中位）与预算充分仍不清零；③非嵌套 11/15 vs Greedy 嵌套 0/176 的不可撤销机制（一句话：贪心前缀恒嵌套，Strict20 中损失条件均为 Exact 结构性非嵌套——11/15 转移无任何嵌套最优对，0/5 快照有完整嵌套最优链）；④基线失败模式与 GPU 证据**分别归属、不混用**：WFC 紧预算覆盖偏置（AgentX K=1，6/9 失败，gap 千级 ms）↔ C/K3 GPU 配对 **FS−WFC Pending B −84.4ms** [−92.0,−69.8]；StandaloneTopK 独立排序冗余（不感知共享，7 快照持续次优）↔ C/K3 GPU 配对 **FS−StandaloneTopK Pending B −163.2ms** [−167.3,−154.9]（E 5120→0：D7168+D7104 同父遮蔽挤占 B5120；数据源 [paper_pending_aggregate.csv](c_formal_perf_gpu/paper_pending_aggregate.csv)）。
**Appendix**：573 逐条复核清单与证书推导、AgentX 736 行扫描明细、SA 逐 K 失败快照明细、生成器可行性记录、四方法实现与 69/69 校准、口径与 Phi 系数出处。

## 7. 复算索引

| 结论 | 数据来源 | 口径 |
|---|---|---|
| 573 复核聚合 | legacy573_recheck.json | 逐条件重算 vs statistics 冻结字段，579 检查 |
| AgentX 扫描 | agentx_budget_scan.csv | 23 快照×K1-8×4 方法；Exact 枚举/代数双轨互证 |
| Strict20 敏感性 | strict20_sensitivity.csv（自 e3 冻结 140 行重聚合） | 逐 K gap/同集 |
| 非嵌套比例 | 两域扫描选集 | 相邻 K 终集包含检验（15+161 转移） |
| 最优集唯一性/结构性非嵌套 | strict20_optima_nesting.json/.csv | 20 条件全子集枚举（容差 1e-9），tie-break 终集==e3 冻结 20/20 |
| 生成器可行性 | generator_feasibility.json | 五快照来源指纹+采集链在位核验 |

**最终判定：READY。**
