# P1 Baseline 与工作负载区分力设计审查

日期：2026-10-08（Asia/Shanghai）  
最终状态：**DESIGN_PARTIAL**

公平且独立的 baseline 可以定义，但现有正式总体不足以证明一般共享场景下 Global Executable-State Allocation 相对简单 per-continuation retention 的独特优势。建议将后续 CPU-only 对照定位为等价性核验与价值模型必要性检查；不建议以当前总体直接启动以“一般共享分配优势”为目标的 P1，更没有新增 GPU 实验的必要前提。

本轮只写本文档。未实现或执行新 baseline，未运行既有分析脚本中的 selector 回放，未启动 GPU、SGLang、请求、trajectory replay，未改变正式 population、budget、eligibility、compatibility 或模型。本文的伪代码是设计定义，不是可执行实现。

## 证据范围和命名

标记约定：**已验证数据**指冻结记录及其结构计数；**数学推导**指给定条件下的证明；**设计**指尚未实现的方法；**解析例**不计入真实总体。

未在当前 FlowState 仓库及所搜索的 `/home/wjg/code` 论文文件中定位到 Section 3–4、6.4 的可读论文稿件原文。因此不能逐字认证现稿。以下以任务提供的正式定义、实现和冻结模型文档为依据，对 Section 6.4 的建议是 claim 边界审查，不声称读过缺失的段落。

主要证据索引（路径以仓库根目录为基准）：

| 编号 | 文件、位置 | 支持内容 |
|---|---|---|
| S1 | `flowstate/workflow.py:24`，`PendingContinuation.planning_target` | T=min(anchor_pos, resident_fa_frontier) |
| S2 | `flowstate/state_catalog.py:60`，`is_compatible` | 相同 workflow、祖先路径前缀及 L≤T |
| S3 | `flowstate/executable_state.py:11`、`:25` | E 为最深兼容位置，G=T−E |
| S4 | `evaluation/FORMAL_RECOVERY_MODEL.md:5`；`flowstate/recovery_model.py:39`、`:58`、`:77` | 冻结 Phi、定义域、收益分解及严格单调性 |
| S5 | `flowstate/optimizer.py:48`、`:65`、`:74`、`:85`、`:112`、`:124` | resident eligibility、等大小、cardinality、贪心、1e-9 ms 比较容差、正收益停止 |
| S6 | `evaluation/p0_allocation_mechanism_audit/final_report.md:15`、`:39`、`:77`、`:91` | 全量 W/T/L、结构分类、ties 与机制边界 |
| S7 | `evaluation/p0_allocation_mechanism_audit/all_cases.jsonl` | 573 行逐案例结构与完整来源 source_file/source_line；本轮仅读取结构与已有结果 |
| S8 | `evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl`；同目录 `aggregate_results.json` | OpenHands 504 cases；原始 objective、兼容矩阵及 Exact OPT |
| S9 | `rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl`；同目录 `formal_snapshot_manifest.json`、`aggregate_results.json` | AgentX 69 cases、23 snapshots，正式逻辑兼容与 objective 可用兼容 |
| S10 | `evaluation/rq3_agentx_formal_policy_evaluation.py:1101` | manifest 保存兼容集合，未保存完整真实 lineage 路径 |
| S11 | `evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726/per_case_results.jsonl` | 两个总体的冻结 StandaloneTopK，`without_set_dependent` |
| S12 | `evaluation/SOTA_BASELINES.md:1`；`evaluation/sota_policies.py:100`、`:253` | 现有策略仅是 adaptation，Marconi 价值信号不是 FlowState Phi |
| S13 | `rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/analysis_snapshot_level/final_report.md:1` | 24 snapshots、216 runs 的既有 TTFT 证据；不包含本文新 baseline |

证据定位补充：P0 的 `case_A.json` / `case_C.json` 对应 OpenHands g186/K4 与 g138/K4；`case_D.json` / `case_E.json` 对应 AgentX 两个严格胜例；`case_F.json` 对应 AgentX 联合排序存在但最终持平。每个文件保存原始 ID、矩阵及源行号。

## A. 独立 continuation 情形的精确化简

### A.1 定理及必要的建模前提

令有限集合 P、C，固定非负收益 b(p,c)，不兼容时定义为零；空集的 max 定义为零。目标为

\[
F(S)=\sum_{p\in P}\max(\{b_{p,c}:c\in S\}\cup\{0\}),\qquad |S|\le K.
\]

K 是非负整数；checkpoint 大小相同；checkpoint 的收益与可行性在本 epoch 固定。假设：

- A1：每个 checkpoint 的正收益支持集大小至多为 1。
- A2：唯一预算是等大小 checkpoint 的 cardinality K。
- A3：除此之外没有联合选择约束、必选项或随选择变化的物理可行性约束。

定义

\[
C_p^+=\{c:b_{p,c}>0\},\quad v_p=\max(\{b_{p,c}:c\in C\}\cup\{0\}).
\]

对于 v_p>0，取一个 c_p∈argmax_c b(p,c)。设 m 为 v_p>0 的 continuation 数，按 v_p 非增排序为 v_(1),…,v_(m)。则

\[
\operatorname{OPT}(K)=\sum_{i=1}^{\min(K,m)}v_{(i)}.
\]

选取这些 continuation 的 c_p 即达到原问题 Exact OPT。

这里 A1–A3 是保证上述化简对所有实例成立的一组**充分条件，不是每个实例最优的必要条件**。有共享时也可能偶然、或因预算足够而等价；不能从违反 A1 推断一定有性能差异。真正需要的是固定的可分收益与对应可行集合，而不是“独立 workflow”这个名称本身。

### A.2 证明

A1 使不同 p 的 C_p^+ 两两不交。任取可行 S，令 Q(S) 为获得正收益的 continuation 集合。一个选中 checkpoint 至多激活一个 p，故 |Q(S)|≤|S|≤K；并且

\[
F(S)=\sum_{p\in Q(S)}\max_{c\in S}b_{p,c}
\le\sum_{p\in Q(S)}v_p
\le\sum_{i=1}^{\min(K,m)}v_{(i)}.
\]

反过来，选取前 min(K,m) 个 continuation 的 c_p。因为各 C_p^+ 不交，这些 c_p 是不同的 checkpoint，占用恰好 min(K,m) 个预算单位；它们给出右侧上界，故最优。等价地，任何 S 在同一 p 上选多个候选都可缩减为其中收益最大的一个，而不降低 F、不增加预算。

由于 C(S)=C(空集)−F(S)，此结论也精确最小化同一冻结 recovery objective；没有改变目标。

### A.3 边界、tie 和数值实现

- K=0、C 为空或 m=0：返回空集；收益为零。T=0 或没有正收益兼容候选的 p 不消耗预算。
- K≥m：选每个有效 p 的最佳 checkpoint 后停止；不以零收益项填满 K。可以存在多个不同的同值最优集合。
- 同一 p 的精确收益 tie：取最小 checkpoint ID。跨 p 的精确收益 tie：先比较其 c_p 的 ID，再比较 continuation ID。正收益支持不交时 c_p 不会重复。
- 若只要求 objective 最优，任何精确 tie-break 都成立；若要求与 FlowState 的 selected IDs 相同，还要使用相同 tie 规则和停止约定。
- 定理针对精确算术。现有 FlowState 在 S5 用 1e-9 ms 判断近似 tie，但以 gain≤0 停止；不能把“差值在容差内”写成实数意义的相等。设计中的排序键按精确值定义；未来实现需显式记录分数、沿用正式 objective 比较容差并核验近 tie。不得调大 epsilon 制造优势，也不得承诺未经验证的逐 bit 集合等价。
- 对数值误差敏感的逐 ID 一致性是实现验证问题，不是推翻上述组合优化定理的证据。

条件放宽的边界：共享候选可用一份预算改善多个 p，使 |Q(S)|≤|S| 失效（解析例 1）；不等大小时是按权重预算的选择问题，不能按收益直接取 K 个；若存在 must-keep、slot 冲突、分组配额、联合释放或 transfer 限制，替换/合并候选不一定可行。本文不把逻辑 cardinality 结论外推到所有物理 pool 配置。

### A.4 FlowState 在此子类的关系

精确 greedy 每次应选一个尚未服务 p 的 c_p：同一 p 的最佳项总不弱于其余项，选中后该 p 所有其他项 marginal 均为零，其余 p 的收益不变。因此在 A1–A3 下 FlowState greedy 与上述 Top-K 化简 objective 等价；唯一最优分数及一致 tie-break 时集合也等价。

这不是由“482 个 tractable instances 上匹配 Exact OPT”反推的结论。434 个 OpenHands 与 48 个 AgentX 的冻结 Exact OPT 匹配是有限样本观测（S8/S9 的 exact_opt）；证明的依据是支持集不交和可行集合。共享场景没有一般 exact-greedy 定理，解析例 2 直接展示合法树形结构下 greedy 非最优。

### A.5 OpenHands 是否满足

**已验证数据（S6–S8）：**168 snapshots、504 snapshot-budget cases；每个 snapshot 有 4 个独立 workflow 的 pending。每个正收益 checkpoint 仅对应一个 pending，正收益共享计数为 0/504；原 snapshot 保存四个 workflow 的真实路径。候选等大小，正式 allocator 仅施加 K，未把额外物理耦合放入该离线目标。

| 条件 | 结论 | 限定 |
|---|---|---|
| A1 | 满足 | 冻结 compatibility/benefit matrix；不以 workflow 名字代替矩阵检查 |
| A2 | 满足 | 正式逻辑分配模型；S5 等大小校验与 budget_bytes/size |
| A3 | 满足于该离线问题 | 不是一般 runtime slot ownership 或并发 epoch 原子性的证明 |

因此 B1 是 **OpenHands 的精确等价求解对照**。90/504 cases 的 K<4，414/504 的 K≥4；后者连跨 p 的有限预算取舍也不需要。每条链包含许多 checkpoints 会让朴素 StandaloneTopK 重复占位，但不会增加上述化简问题的组合难度。

## B. 三类 baseline 的完整设计

### B.0 公平性约定

所有方法读取同一 allocation epoch、同一 eligible C、同一当前 P、正式 compatibility、L、T 和 K。输入不得混入未来 invocation、真实后续输出长度、策略结果或 Exact OPT 答案。AgentX 必须使用 S9 的正式逻辑兼容及 L≤T 后的 objective 可用集合，不改用原始 B2 runtime 的另一套 compatibility。

集合 S 的预算按**唯一冻结 checkpoint ID**计数。同一个 checkpoint 被多个 p 推荐只保留一次；这不是每个 p 各复制一份。不同 ID 即使内容或位置相同也不能自动合并。物理 realization 需使用既有 canonical checkpoint/handle 映射与相同 reconciliation substrate；若存在跨 ID 的物理别名，须先证明完整状态、所有 ranks 和所有权代际的等价性，不能仅按 slot 整数去重。当前离线 snapshot 不提供一般物理别名证明，标为 UNKNOWN；本设计不改变 frozen logical budget。

共同 objective 可以用于选择完成后的统一评分，但 B2/B3 的决策不得读取它。若未来计时，必须区分输入/observation、selector、reconciliation 和评价诊断成本；新 heuristic 不使用 Phi 的事实不意味着它可以免付自身实际所需的 runtime 信息读取。三个名字均为本文定义的算法，不称为完整 KVFlow、TOPAS 或 Marconi。

### B1. Per-Continuation Best-Checkpoint（PCB）

**用途：**相同 benefit 模型下的可分求解对照；不是价值模型独立的竞争者，也不是完整 FlowState 的复制。它使用 b(p,c)，但不维护 set-dependent marginal，不扫描共享组合。

定义：对每个 p 选择正收益最大的 c_p；精确 tie 用最小 c ID。将有效 p 按 `(-b(p,c_p), c_p.ID, p.ID)` 排序。按此静态顺序扫描，把未在 S 中的 c_p 加入，直到 |S|=K 或列表耗尽；遇到重复 c_p 只跳过，继续扫描后面的 p，不浪费 slot。不会为已经选过的 p 再寻找次优 checkpoint，不根据已保留集合重新计算 ranking。

```text
输入：当前 P、C、compatibility、T、L、K、冻结 Phi
S ← 空集
对于每个 p：
    在兼容候选中确定正收益最大的 c_p；同值按 checkpoint ID
    若不存在正收益候选，则不产生推荐
将推荐按（收益降序，checkpoint ID 升序，continuation ID 升序）排列
依次扫描推荐：
    若 |S| = K，则结束
    若 c_p 不在 S，则加入 S
返回 S
```

共享情形下，该定义始终可行：只保留候选中的唯一 IDs，至多 K 个；一个共享 c_p 自动惠及其全部 compatible p，但排序不会把这些额外收益重新相加。被所有 p 的独立最佳项压过的共享祖先可能从推荐列表消失（例 1）。不能把这个缺陷隐瞒，也不能事后补一个 FlowState 式 marginal 规则。

复杂度 O(|P||C|+|P|log|P|)，临时推荐 O(|P|)，可流式求各 p 最佳；不要求保存完整 benefit matrix。读取 Phi/T 属于公平、明确的模型共享。

**等价范围：**A1–A3 下 Exact OPT；更一般地，若 K≥m（有效 p 数），其去重后的推荐并集包含每个 p 的最佳项，达到全候选可实现的前沿，即使存在共享也 Exact OPT。若推荐并集小于 m，只需要 K 足够容纳该并集。共享且预算受限时无最优保证，也不保证一定弱于 FlowState。

### B2. Lineage-Aware Deepest Retention（LDR）

**用途：**测试“知道兼容关系、每条 lineage 留最深可用位置”的简单长度代理是否已经足够。它不调用 Phi、不计算 F 或恢复成本 marginal。

对每个 p，在 compatible 且 L>0 的候选中取最大 L 的 c_p；相同 L 取最小 c ID。将推荐按 `(-L(c_p), c_p.ID, p.ID)` 排序，按 B1 的唯一 ID 规则填充至 K 或推荐耗尽。L 是从空集恢复所避免的 token 长度，不是 ms；T 仅参与冻结 compatibility，不进入加权得分。

```text
输入：当前 P、C、compatibility、L、K
S ← 空集
对于每个 p：
    在兼容且 L>0 的候选中取最深 c_p；同位置按 checkpoint ID
按（L 降序，checkpoint ID 升序，continuation ID 升序）排列推荐
依次扫描：
    若 |S| = K，则结束
    若 c_p 不在 S，则加入 S
返回 S
```

重复覆盖：每个 p 只推荐一个 checkpoint，避免同一 p 的浅层重复；共享推荐去重，但不同推荐对第三个 p 的覆盖重叠不更新排序。提前停止不填入 L=0、无兼容或无推荐项。复杂度 O(|P||C|+|P|log|P|)。信息只来自当前状态，无历史 FLOP 元数据或未来知识。

在冻结 Phi 对固定 T 严格单调的条件下，LDR 与 PCB **每个 p 的最佳 checkpoint 相同**。区别仅在有限预算下跨 p 按 token 长度还是按位置感知收益排序。K≥m 时两者均达到全候选前沿。A1 下且所有 T 相同，b 是 L 的同一个严格递增函数，因此跨 p 排序也与 PCB/FlowState 等价；T 不同则可能不同（例 3）。存在共享且 K 受限时，LDR 也可能遗漏共享祖先。

### B3. Workflow-Aware Frontier Coverage（WFC）

**用途：**采用无拟合系数的“先改善更多待续前沿，再比较总 token 推进”规则，主动考虑共享，而不使用完整位置感知 recovery objective。名称中的 workflow-aware 指使用正式当前兼容关系，不是假定所有 pending 来自独立工作流。

当前 E_p 初始为 0，选择后按最深兼容 L 更新。对于未选 c：

\[
d_{p,c}(S)=\begin{cases}\max(0,L_c-E_p(S)),&c\preceq p,\\0,&\text{其他},\end{cases}
\]
\[
q_c(S)=\#\{p:d_{p,c}(S)>0\},\quad r_c(S)=\sum_p d_{p,c}(S).
\]

按 `(q 降序, r 降序, checkpoint ID 升序)` 选择。这里 q 统计**前沿被严格改善**的 continuation，不是仅统计 E=0 的首次命中；部分恢复的 p 仍可被继续改善。没有引入可调权重。

```text
输入：当前 P、C、compatibility、L、K
S ← 空集；每个 E_p ← 0
当 |S| < K：
    为每个未选 c 计算当前 q_c 与 r_c
    若没有 q_c>0 的候选，则结束
    选择 q 最大、继而 r 最大、继而 ID 最小的 c
    S ← S ∪ {c}
    对所有兼容 p，E_p ← max(E_p,L_c)
返回 S
```

同一 ID 只占一份预算；共享 c 同时更新所有 compatible p。当前零 token 推进候选不会被选入。token 整数比较不需要 ms epsilon。复杂度可为 O(K|P||C|)，维护 E 需要 O(|P|)；它不保证比现有 FlowState 实现更快，因为 observation/reconciliation 可能占主导。

**独立价值与缺陷：**WFC 显式检查共享与重复覆盖，但 q 主导排序，会偏向给多个 p 很小的推进，而不是给一个 p 很大的恢复收益；r 也忽略绝对位置导致的单 token 成本变化。它不是完整 F 的贪心，也不是把 Phi 名字换成别的名字。A1 下 q 只有 0/1；每次最深候选被选后该 p 的其他项失效，故等价于 LDR。共享场景则可能与 PCB、LDR、FlowState 都不同；不预设哪一个更好。

### B.4 候选比较和实施约束

| 方法 | 使用冻结 Phi | 使用当前集合 | 共享处理 | 研究用途 | 主要风险 |
|---|---|---|---|---|---|
| PCB | 是，逐 p 独立收益 | 只做 ID 去重 | 共享收益自然实现，但不联合排序 | 精确化简对照 | 不应伪装成独立价值模型或预设落后 |
| LDR | 否 | 只做 ID 去重 | 同上 | 最小 workflow-aware 强对照 | token 长度忽略 T；推荐集合可能漏祖先 |
| WFC | 否 | 维护 E 与 token 推进 | 显式多 p 前沿覆盖 | **最值得额外审查的独立 heuristic** | 覆盖数量优先可能牺牲总恢复成本 |

如果只增加一个具有独立方法价值的 heuristic，优先 WFC：它不会把缺乏任何共享意识的弱对照当作一般性分配基线。PCB 仍应作为必要的化简校验，LDR 用于区分位置感知成本是否必要。全部只是在此提出；需要另行授权后才可实现，不能按 P0 的胜例调权重、调整 tie 或补规则。

## C. 当前 workload 的区分力

### C.1 新核对的结构事实：不是新 baseline 结果

本轮只从 S7 的已保存矩阵、K、pending_count 和已有 C(C) 读取以下属性；没有调用 selector、没有输出任何新方法的 selected set 或性能表：

| 属性 | OpenHands | AgentX |
|---|---:|---:|
| snapshots / snapshot-budget cases | 168 / 504 | 23 / 69 |
| 每 snapshot 的 pending 数 | 4 | 2–8 |
| K≥全部 pending 数的 cases | 414 | 52 |
| K≥有正收益候选的 pending 数 m 的 cases | 414 | 66 |
| K<m 的 cases | 90 | 3 |
| 同一矩阵含共享正收益列和单 pending 正收益列 | 0 | 27（9 snapshots） |
| 全候选仍存在正恢复 residual 的 cases | 36（12 snapshots） | 69（23 snapshots） |
| 冻结 FlowState 达到全候选可实现前沿 | 414 | 69 |

AgentX 的 3 个 K<m cases 是：

| snapshot | K / m | 唯一正收益 checkpoint | L | 正式原始 JSONL 行 |
|---|---|---|---:|---:|
| rq3-agentx-g002-epoch6 | 1 / 2 | `4cb074e3cbf03d1709e2b44cbe575807813b::req0000` | 448 | S9，7 |
| rq3-agentx-g009-epoch6 | 1 / 2 | `6ec3296ff5a1f4f0ed4507093863aa53a4cc::req0000` | 320 | S9，28 |
| rq3-agentx-g016-epoch6 | 1 / 2 | `b699b717b623acb2b6c06db1196258fbb4a7::req0000` | 448 | S9，49 |

每个 case 只有一个正收益 checkpoint，它服务两个 p；其余候选的所有 b 都是零。**数学推导：**66 cases 由 K≥m 保证 PCB 和 LDR 推荐并集可完整保留；余下 3 cases 的两个推荐是同一个唯一正收益 checkpoint，去重后占一份预算。因此当前 AgentX 全体在该设计的 PCB/LDR 下都能达到全候选前沿。结合冻结 FlowState 已达到该前沿，可推导 objective 等价；这不是声称运行得到新的“69/69 baseline 实验结果”。工程实现、数值和 handle 映射仍未验证。

这个结论比“只有 2/69 相对 StandaloneTopK 改善”更有区分力：那两个胜例 K=8、7，而各自仅有 2 个 pending，并不存在容纳各分支最佳 checkpoint 的预算困难；困难来自静态 TopK 把许多预算单位耗在重复覆盖上。

### C.2 五层证据必须分开

下表“严格改善”只引用已经运行的 FlowState vs StandaloneTopK，不代表 PCB/LDR/WFC。评级针对每行所描述的区分力，而不是断言不存在任何结构。

| 场景 | 形式结构可表达 | OpenHands 正式出现及顺序/目标证据 | AgentX 正式出现及顺序/目标证据 | 当前评级 | 独立 runtime 归因 |
|---|---|---|---|---|---|
| 1. 独立 continuation、链内冗余 | 是，收益支持不交 | 504 cases；144 严格胜例首分歧静态候选 marginal=0；属于可精确化简子类 | 同一 pending 收益重叠 33 cases（11 snapshots）；完整真实 workflow 独立性 UNKNOWN；两胜例有冗余重分配 | OpenHands：SUPPORTED；AgentX 独立性：PARTIALLY_SUPPORTED | 新 baseline 均 UNKNOWN；旧策略总体有 S13 |
| 2. 共享祖先同时服务多个 p | 是，祖先前缀及 L≤T | 0；NOT_OBSERVED | 66/69 正收益共享；48/69 实际多 p marginal；9/69 联合排序反转，但不能隔离出同时多分支收益导致的严格优势 | AgentX：PARTIALLY_SUPPORTED（结构和顺序已支持，独特最终优势未支持） | UNKNOWN |
| 3. 部分共享与部分分支专属收益 | 是，树上祖先和后代候选 | 没有共享；NOT_OBSERVED | 27 cases/9 snapshots；两个严格胜例属于此类，但首分歧新增收益只有一个 p | PARTIALLY_SUPPORTED；“分支专属”不等于已经证明“独立 workflow” | UNKNOWN |
| 4. 独立收益相近但剩余 marginal 不同 | 是，不要求任意构造 benefit matrix | g138/K4：静态下一项 181.568、替代项 176.541 ms；首分歧当前 marginal 为 0 与 176.541 ms，最终改善 174.070 ms | 两严格胜例有明确 marginal 反转；未预先定义“相近”的数值门槛 | PARTIALLY_SUPPORTED；“相近”的总体发生率 UNKNOWN，不事后设阈值 | UNKNOWN |
| 5. 覆盖重叠且 K 真正限制可实现前沿 | 是，必须有共享/分支最深状态的保留取舍 | 90 cases K<4，有容量限制，但仍是可分 Top-K；不能证明一般共享组合难度 | 有 33 重叠 cases，但全体 FlowState 都达到全候选前沿；3 个 K<m cases 只有一个有用共享候选，无竞争 | OpenHands：PARTIALLY_SUPPORTED；AgentX 所需的共享预算取舍：NOT_OBSERVED | UNKNOWN |
| 6. 无法覆盖规划目标、残余不可消除 | 是，max L<T 或无可用候选 | 36 cases/12 snapshots 的 C(C)>0 | 69 cases/23 snapshots 的 C(C)>0，且已达到可实现前沿 | SUPPORTED（残余存在）；不能当成组合难度证据 | 模型残余不直接量化 runtime，UNKNOWN |
| 7. 不同集合、相同 E/G 和 objective | 是，等位置替代或冗余项 | 相对 StandaloneTopK 287 cases；全部相同逐 p 前沿 | 22 cases；67 ties 中 45 同集合、22 不同集合 | SUPPORTED | 不能推导 runtime 必然相同；UNKNOWN |

AgentX 的 66/48/9 cases 分别涉及 22/16/3 个不同 snapshots。仅有的两个严格胜例涉及两个不同 snapshots，且都在 25% budget；不是 69 个独立样本。三个预算是同一 snapshot 的重复条件。未来区间估计应以 snapshot 为 paired/cluster unit，不能把预算、continuation 或候选当成独立样本。

“联合排序反转”的已有定义是：实际选中候选的**总**贡献较大，但其**最大单个 p**贡献比某另一候选小；它证明求和改变相对排序，不等于完整的反事实策略胜负。StandaloneTopK 也使用跨 p 独立收益之和，因此该性质不是 FlowState 独占。

S13 支持旧 LRU/Marconi 与 FlowState 的总体 request-level TTFT 差异；不能自动映射为上述七种机制各自的 runtime 因果效应，也没有 PCB/LDR/WFC 的 latency 测量。模型 residual 大不等于净 runtime 收益大。

## D. 三个小型解析区分例

以下均为 **ANALYTICAL EXAMPLE**，非真实 workload、非正式实验、非 GPU latency。集合与选择由公式手工推导；不将数字并入任何冻结统计。不允许用此处结构去筛选有利的正式 snapshots。

### D.1 共享祖先可能优于逐 p 最佳推荐，K=1

两个 pending：p1 路径 `(w,a)`、p2 路径 `(w,b)`，T1=T2=10。候选 s 路径 `(w)`、L=4、兼容 {p1,p2}；a 路径 `(w,a)`、L=6、兼容 {p1}；b 路径 `(w,b)`、L=6、兼容 {p2}。分叉在共享位置 4 之后，位置与祖先关系合法。ID 顺序 a<b<s，大小均为一单位。

使用解析恢复函数 Phi(G,T)=G（任意一致长度/成本单位）。对每个固定 T，它非负、Phi(0,T)=0、导数为 1，严格单调；故 b(p,c)=L_c，收益矩阵不是任意填数：

| c | L | b(p1,c) | b(p2,c) |
|---|---:|---:|---:|
| s | 4 | 4 | 4 |
| a | 6 | 6 | 0 |
| b | 6 | 0 | 6 |

空集 C=20。K=1：

| 方法 | 手工选择 | E 向量 | F | C |
|---|---|---|---:|---:|
| PCB | a：每个 p 推荐自己最深项；6 同分按 ID | (6,0) | 6 | 14 |
| LDR | a：6 同分按 ID | (6,0) | 6 | 14 |
| WFC | s：q=2，a/b 的 q=1 | (4,4) | 8 | 12 |
| FlowState | s：总 marginal 8>6 | (4,4) | 8 | 12 |
| 冻结定义的 StandaloneTopK（仅手工说明） | s：独立总收益 8>6 | (4,4) | 8 | 12 |

枚举可行单元素集即可确认最优成本为 12。此例证明共享可能破坏 A1 的化简，并不证明 set-dependent update 比 StandaloneTopK 更好，也不等于真实数据已出现这种受限选择。

### D.2 同一合法共享树，K=2：FlowState greedy 也可能落后

沿用 D.1 的两个 pending、三个 checkpoints、位置与 Phi，只把解析例 K 设为 2；这不是修改正式 budget。PCB/LDR 取 a,b，达到两个 p 各自最深位置。FlowState 首先选 s（8>6），之后 a/b 相对 s 各增加 2，按 ID 选 a。WFC 同样先 s，再在 q=1、r=2 的 tie 中选 a。

| 方法 | 手工选择顺序/集合 | E 向量 | F | C |
|---|---|---|---:|---:|
| PCB | a,b | (6,6) | 12 | 8 |
| LDR | a,b | (6,6) | 12 | 8 |
| WFC | s,a | (6,4) | 10 | 10 |
| FlowState | s,a | (6,4) | 10 | 10 |
| StandaloneTopK（手工） | s,a | (6,4) | 10 | 10 |

三个两元素集合中，{a,b} 成本 8，{s,a} 和 {s,b} 成本 10；故 Exact OPT=8。该例没有非树形交叉 compatibility、负收益、不等大小或未来信息，说明“合法共享 lineage”并不赋予 greedy 一般精确最优性。也说明新增 baseline 必须容许简单方法获胜，不能按 P0 定制一个必败对照。

### D.3 冻结 Phi 下，位置权重可区别于 token 长度，K=1

两个独立 workflow，pA 路径 `(wA)`、pB 路径 `(wB)`。a 仅兼容 pA，L_a=1024、T_A=131072；b 仅兼容 pB，L_b=1088、T_B=8192。候选路径分别与所属 pending 相同；大小相同、K=1，符合 A1–A3。

此例直接使用冻结 Phi，而非前两例的解析线性函数。记 g=G/1024、t=T/1024、l=L/1024，令 A=37.828150、B=0.345974143、D=−0.156201917，则

\[
\Phi(g,t)=Ag+Bgt+Dg^2,\qquad
b(l,t)=Al+(B+2D)tl-Dl^2.
\]

在 0≤g≤t≤128 中，导数 A+Bt+2Dg≥A+(B+2D)t>0，所以两个收益都由合法、严格单调的冻结函数产生。a 使用 (l,t)=(1,128)，b 使用 (1.0625,8)：

- b(pA,a)=42.281351469 ms；b(pB,b)=40.654094322 ms。
- C(空集)=8266.001973248 ms。

| 方法 | 手工选择理由 | selected set | C，模型 ms |
|---|---|---|---:|
| PCB | 42.281351469>40.654094322 | {a} | 8223.720621779 |
| LDR | 1088>1024 tokens | {b} | 8225.347878926 |
| WFC | 两者 q=1；r 分别 1024、1088 | {b} | 8225.347878926 |
| FlowState | 首步 marginal 即各自 b | {a} | 8223.720621779 |
| StandaloneTopK（手工） | 各自只服务一个 p，按同一 b | {a} | 8223.720621779 |

差异约 1.627257 ms，刻意不解释为运行时收益。这说明跨 p 的位置感知成本可能有独立价值；它不需要复杂共享，也不证明真实 OpenHands 的 90 个受限 cases 一定出现该排序差异。本轮没有运行 LDR/WFC 去验证这种出现率。

三个例子的兼容集合均来自真实可实现的路径前缀结构；没有把任意非层次集合覆盖矩阵冒充 ancestry。一般工作负载的难度必须结合这种结构限制讨论，不能仅从 submodularity 推断某个任意集合覆盖难例一定可实现。

## E. Claim–Evidence 对照与未来真实 workload 条件

### E.1 当前 Section 6.4 的边界

| 拟声称内容 | 当前证据与判断 | 最小修改方向 |
|---|---|---|
| 在冻结 tractable 子集匹配 Exact OPT | 支持有限观测：434+48=482 cases，S8/S9；不推出一般最优性 | 明确 tractable、已评估实例和 frozen objective |
| OpenHands 证明一般共享状态全局优化必要 | 不支持；A1–A3 下可按逐 p 最佳再 Top-K 精确求解 | 改为独立 workflows 间预算竞争、朴素静态冗余保留的代价 |
| 相对 Marconi 的优势全部来自 marginal updates | 不支持；P0 中 245 个 OpenHands Marconi 胜例与 StandaloneTopK 持平 | 分离价值信号差异与 set-dependent 更新的额外作用 |
| 相对 StandaloneTopK 的 144/504、2/69 严格改善 | 支持这些冻结结果和首次分歧的冗余消除解释 | 不推广为胜过所有 workflow-aware retention |
| AgentX 共享收益影响实际分配排序 | 支持：48 多 p marginal、9 联合排序反转 cases | 同时给出 16、3 个独立 snapshots，限定为顺序/贡献证据 |
| AgentX 两胜例证明同时多分支收益造成独特优势 | 不支持；两例该步新增收益仅有一个 p | 改为共享祖先可用性下的冗余消除和另一分支恢复 |
| 大 residual 或大量候选说明分配组合困难 | 不支持；AgentX 69 cases 已达到全候选前沿 | 报告有效需求数、是否真正预算受限及不可消除 residual |
| 同 objective 意味着同 runtime 或 control overhead | 不支持；相同前沿不约束 reconciliation/control 代价 | 分离 modeled allocation、runtime execution 和 control-inclusive 指标 |
| PCB/LDR 已经通过正式 baseline 测试 | 不支持；本轮只有数学推导 | 明确“按定义可推导”，待未来实现验证，不新增性能结果表 |
| 当前论文 Section 6.4 的原文逐句正确 | UNKNOWN：未定位稿件原文 | 将以上边界用于后续人工逐句核对，不声称本轮已编辑论文 |

### E.2 真实且可审计的区分条件

缺少的不是简单地“更多候选”或“更复杂名字的 lineage”。更有区分力的条件是：在同一真实 allocation boundary 上，有多个**确实尚未完成且可续接**的分支；存在共同祖先与分支专属、更深且当前驻留的 checkpoint；共享与专属候选的正收益存在重叠；K 真正限制保留各 p 最深状态的并集，形成“较浅共享覆盖多个 p”与“较深专属覆盖部分 p”的取舍。

- 更多真正 pending continuations 只有在其需求不能由一两个公共最佳 checkpoint 同时满足时才有用；只增加无可用候选的 p 会扩大 residual，未必增加可分配难度。
- lineage 深度本身不充分；独立长链仍可化简。祖先共享与后代专属状态的位置/目标关系，以及有效预算约束更关键。
- K 按全部 candidate 数的比例设置，可能被大量零收益或重复候选放大；应**报告**总 candidate 数、正收益候选数、有效 p 数及前沿是否可同时实现。不能事后删候选或降低 frozen K 来放大收益。
- 真实 parent/fork 关系需有可核查 provenance、fork 时点、父状态 identity、token-prefix 继承与当时驻留证据。不得将相似文本、任务标签或事后拼接的祖先路径当成可复用状态。
- 必须记录 checkpoint 创建/驻留与 pending 就绪的时间先后、当前 T、attention-side frontier 和候选 eligibility；不能把未来会产生的 checkpoint 或下一步分支加入当前边界。
- 新数据如确有必要，应另立 cohort，先固定任务来源、时间窗口、全部符合 gate 的纳入规则、epoch 捕获规则和预算协议，之后 policy-blind 收集并全量报告。可按预注册结构层作描述，但不得依据哪个策略获胜重新选择样本，更不能替换当前 168/23 的正式总体。
- 解析例仅证明可区分条件存在，不能转化成“真实数据支持”。合法 prefix-tree 的共享集合受到结构限制；不要人为构造不可能的交叉血缘覆盖。

当前真实数据中，共享结构和混合列已经有证据，故并非需要凭空新增一种拓扑；主要缺少的是**真正受预算限制的共享/专属收益取舍**。对“位置感知收益是否胜过简单长度代理”，现有 OpenHands 90 个受限 cases 有潜在区分力，但未执行前不得声称存在收益或指定幅度。

## F. P1 是否值得执行及优先级

### F.1 明确判断

1. **PCB 对 OpenHands 是等价求解对照。** 值得用于排除弱 baseline 造成的错误叙事，不能作为预设将被 FlowState 击败的对照。
2. **WFC 是最有独立研究价值的额外 heuristic。** 它考虑共享与当前覆盖，且不使用完整 Phi；LDR 是必要的简单参照。PCB 则校验问题是否已退化。
3. **当前数据不足以建立一般共享分配优势。** AgentX 的结构饱和使 PCB/LDR 的理论结果已被限定；OpenHands 的非平凡部分主要能检查跨 p 的位置权重，不能验证共享组合机制。
4. **缺口主要是有效收益重叠与预算紧约束的交互。** 更多真实分支、复杂 lineage 是可能的来源，不是充分条件。
5. **Section 6.4 可以保留冻结策略之间的有限比较、静态冗余代价和观测到的共享排序。** 应弱化“简单 workflow-aware 方法不足”“一般 exact 最优”和“多分支共享导致独特净收益”等表述。
6. **建议后续有限 CPU-only 对照，但不是证明一般优势的正式 P1 启动理由。** 价值在于验证等价性、检测实现不一致、检查独立 proxy 是否足够；允许甚至预期出现大量 ties 或 heuristic 胜例。
7. **本轮没有必须新增 GPU 才能回答的数学/结构问题。** 只有当问题改为新策略实际 control cost、reconciliation 代价、request latency 或净 E2E 收益时，才需要其独立 runtime 测量；不能拼接旧 GPU 结果代替。

### F.2 后续建议表（未执行）

| 候选工作 | 预期科研价值 | 实现成本 | 结论风险 | 建议 |
|---|---|---|---|---|
| PCB 的 CPU-only 等价性核验 | 高：揭示基线差距、校验 K/去重/compatibility | 低至中：纯 selector，但需边界与数值一致性核验 | 对 FlowState 的性能区分预期很低；若不等价应先查实现 | 建议作为验证，不作为宣传新收益的实验 |
| LDR 的冻结总体 CPU-only 对照 | 中至高：分离位置模型与 workflow-aware retention | 低 | AgentX 预期结构性持平；OpenHands 只在受限、异 T 子类有潜力 | 建议完整总体报告，不仅报告有差异子集 |
| WFC 的冻结总体 CPU-only 对照 | 高：更公平的共享感知独立 heuristic | 中：维护 E、去重与严格 tie 规则 | 可能持平、优于或劣于 FlowState；不能事后改规则 | 独立 heuristic 第一优先级，但不保证当前数据足够有区分力 |
| 新真实 cohort 的 policy-blind 采集 | 潜在高：补足共享/专属的真实预算取舍 | 高：需要可审计 lineage、pending barrier 和驻留证据 | 稀有结构、采样偏差及 runtime 支持风险 | 暂缓；先确认来源自然出现该结构，另行预注册和授权 |
| 新 GPU/E2E 实验 | 仅对 runtime claim 必要 | 高 | control cost 可能抵消 allocation 改善，旧数据不可替代 | 当前不建议启动；先完成公平设计与 CPU 层解释 |

若后续获准执行 CPU-only：固定本文三种算法，不调参；复用全部 504+69 cases，不新增/删选总体；使用既有 Phi 做共同评价；先验证合法性和边界，再报告按预算的 paired objective、逐 p E/G、selected set、不可消除 residual 及所有 ties/负例。跨预算推断以 snapshot 为聚类单位；不能把 69 cases 写成 69 个独立 snapshots。若需要 Exact OPT，只能引用已有结果或另行明确授权与预注册，不以本文为执行许可。

**最终选择 DESIGN_PARTIAL 的原因：**baseline 定义公平且可实现，数学上存在合法区分条件；但当前真实 workload 尚不能支撑一般共享分配相对简单 per-continuation retention 的优势。不是 NO_JUSTIFIED_P1，因为核验强简单对照和位置代理具有明确科研价值；也不是 DESIGN_READY，因为“现有真实数据足够有区分力”这一关键条件尚未成立。

## 完整性与停止边界

审查输入集合包含 P0 所记录的冻结源、P0 产物以及本轮补读的正式定义和既有 runtime 报告，共 1769 个文件。审查前的路径—SHA-256 映射整体摘要为 `84db2b846fd3f9bc167d5c5175d8b7b9bd7644c592825a08bed1779bebf742bf`；P0 原冻结源摘要核对 PASS。本文写完后再次只读核验同一集合，文件数与整体摘要完全相同，完整性复核 PASS。

唯一新增交付文件为 `evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md`。没有新 baseline 实现、性能比较产物或正式结果变更。本轮到设计审查为止，立即停止，不进入实验执行。
