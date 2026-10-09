# P0 分配机制与案例审计

**PARTIALLY_SUPPORTED**

已冻结结果支持去冗余后的跨待续请求预算分配；OpenHands 的独立工作流竞争证据明确。AgentX 有真实共享收益和联合排序效应，但不能把两个相对 StandaloneTopK 的严格胜例归因于同时覆盖多个分支的新增收益。以下统计单位均为 snapshot-budget case，三个预算不是独立 snapshots；本报告不把 573 cases 当成 573 个独立统计样本做推断。所有差值为 baseline C(S) − FlowState C(S)，单位为模型预测 ms，不是 GPU latency。

## 输入与只读边界

- OpenHands：168 snapshots × 3 budgets = 504 cases。正式结果：`evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl`；原始 snapshots：`evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/snapshots/`。
- AgentX：23 snapshots × 3 budgets = 69 cases。正式结果：`rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl`；兼容性 manifest：同目录 `formal_snapshot_manifest.json`。
- 两个总体的完整 StandaloneTopK 决策均存在：`evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726/per_case_results.jsonl`，使用 `without_set_dependent`；没有临时新增 baseline。
- 全量解释数据：[all_cases.jsonl](all_cases.jsonl)。每行包含来源文件及行号、候选与兼容矩阵、T/L/E/G、所有策略成本、全部 standalone benefit、每个贪心前缀的所有剩余候选 marginal、u、首个分歧与最终零删除边际。
- 分析脚本：[analyze.py](analyze.py)；报告与矢量图脚本：[report.py](report.py)。没有修改原始结果、策略或候选总体。

## 全量收益分布

| 数据集 | baseline | 胜 / 平 / 负 | 平均差值, ms | 中位差值, ms | P25 / P75, ms | 最大差值, ms | 不同选中集合 |
|---|---|---|---:|---:|---|---:|---:|
| OpenHands | Marconi | 386 / 118 / 0 | 112.021 | 63.263 | 4.931 / 198.634 | 582.242 | 503 |
| OpenHands | LRU | 306 / 198 / 0 | 164.282 | 164.197 | 0.000 / 274.096 | 727.167 | 504 |
| OpenHands | LFU | 504 / 0 / 0 | 273.104 | 193.699 | 100.175 / 392.561 | 1114.912 | 504 |
| OpenHands | StandaloneTopK | 144 / 360 / 0 | 59.301 | 0.000 | 0.000 / 144.524 | 753.200 | 431 |
| AgentX | Marconi | 16 / 53 / 0 | 77.177 | 0.000 | 0.000 / 0.000 | 1967.475 | 69 |
| AgentX | LRU | 16 / 53 / 0 | 415.158 | 0.000 | 0.000 / 0.000 | 5309.014 | 69 |
| AgentX | LFU | 12 / 57 / 0 | 277.608 | 0.000 | 0.000 / 0.000 | 5309.014 | 66 |
| AgentX | StandaloneTopK | 2 / 67 / 0 | 66.807 | 0.000 | 0.000 / 0.000 | 2333.666 | 24 |

| 数据集 | 预算 | vs Marconi 胜/平/负 | vs LRU | vs LFU | vs StandaloneTopK |
|---|---|---|---|---|---|
| OpenHands | 25% | 163/5/0 | 168/0/0 | 168/0/0 | 96/72/0 |
| OpenHands | 50% | 131/37/0 | 90/78/0 | 168/0/0 | 46/122/0 |
| OpenHands | 75% | 92/76/0 | 48/120/0 | 168/0/0 | 2/166/0 |
| AgentX | 25% | 12/11/0 | 11/12/0 | 7/16/0 | 2/21/0 |
| AgentX | 50% | 3/20/0 | 3/20/0 | 4/19/0 | 0/23/0 |
| AgentX | 75% | 1/22/0 | 2/21/0 | 1/22/0 | 0/23/0 |

OpenHands 的三预算 W/T/L 与冻结 `rq3_sanity_structure_audit_20260904_121500/win_tie_loss.json` 完全一致；四个正式策略在两个总体、三个预算下的平均 C(S) 均与原 aggregate_results.json 一致。没有负例。近零正收益采用预先明确的描述性区间 (1e-9, 1] ms：两个总体对 Marconi 都没有此类 case；没有更改正式 1e-9 ms tie 容差。

## 机制分类及分布

分类允许重叠，不可相加为 100%。A 的单位是同一个可兼容 pending 的收益覆盖；OpenHands 同时有完整 lineage 路径佐证，AgentX 以正式祖先兼容集合为准。B 的严格容量竞争条件是 K 小于正收益独立工作流数；K≥工作流数时，静态 TopK 仍可能因冗余占用而漏掉其他工作流。AgentX 未在本轮恢复真实完整 workflow/lineage metadata，因此 B 显示 UNKNOWN；逐案例 JSON 中 B=null、汇总 unknown_count=69，不能把未确认当成不存在。

| 数据集 | 分类 | 数量 / 占比 | vs Marconi 平均 / 中位 [最小, 最大], ms | vs StandaloneTopK 平均 / 中位 [最小, 最大], ms |
|---|---|---|---|---|
| OpenHands | A：同一 pending 的候选收益重叠 | 504 / 100.00% | 112.021 / 63.263 [0.000, 582.242] | 59.301 / 0.000 [0.000, 753.200] |
| OpenHands | A：StandaloneTopK 最终集合存在零删除边际 | 431 / 85.52% | 106.224 / 44.593 [0.000, 582.242] | 69.345 / 0.000 [0.000, 753.200] |
| OpenHands | A：Marconi 最终集合存在零删除边际 | 440 / 87.30% | 116.985 / 64.486 [0.000, 582.242] | 59.285 / 0.000 [0.000, 753.200] |
| OpenHands | B：确认存在独立工作流 | 504 / 100.00% | 112.021 / 63.263 [0.000, 582.242] | 59.301 / 0.000 [0.000, 753.200] |
| OpenHands | B：K 少于正收益独立工作流数 | 90 / 17.86% | 130.191 / 99.874 [0.000, 412.964] | 76.187 / 0.000 [0.000, 390.316] |
| OpenHands | C：正式结构允许 checkpoint 覆盖多个 pending | 0 / 0.00% | 不适用 | 不适用 |
| OpenHands | C：至少一个 checkpoint 对多个 pending 有正独立收益 | 0 / 0.00% | 不适用 | 不适用 |
| OpenHands | C：实际贪心某一步对多个 pending 有正新增收益 | 0 / 0.00% | 不适用 | 不适用 |
| OpenHands | C：选中共享 checkpoint 的删除会增加成本 | 0 / 0.00% | 不适用 | 不适用 |
| OpenHands | C：联合求和与最大单项贡献发生相对排序反转 | 0 / 0.00% | 不适用 | 不适用 |
| OpenHands | D：与 StandaloneTopK 集合不同但成本相同 | 287 / 56.94% | 51.709 / 26.914 [0.000, 540.605] | 0.000 / 0.000 [0.000, 0.000] |
| OpenHands | D：与 Marconi 集合不同但成本相同 | 117 / 23.21% | 0.000 / 0.000 [0.000, 0.000] | 5.418 / 0.000 [0.000, 221.286] |
| OpenHands | E：与 Marconi 成本相同 | 118 / 23.41% | 0.000 / 0.000 [0.000, 0.000] | 5.373 / 0.000 [0.000, 221.286] |
| OpenHands | E：对 Marconi 改善在 (1e-9, 1] ms | 0 / 0.00% | 不适用 | 不适用 |
| OpenHands | E：对 Marconi 成本更高 | 0 / 0.00% | 不适用 | 不适用 |
| AgentX | A：同一 pending 的候选收益重叠 | 33 / 47.83% | 155.560 / 0.000 [0.000, 1967.475] | 139.687 / 0.000 [0.000, 2333.666] |
| AgentX | A：StandaloneTopK 最终集合存在零删除边际 | 24 / 34.78% | 129.862 / 0.000 [0.000, 1432.519] | 192.070 / 0.000 [0.000, 2333.666] |
| AgentX | A：Marconi 最终集合存在零删除边际 | 69 / 100.00% | 77.177 / 0.000 [0.000, 1967.475] | 66.807 / 0.000 [0.000, 2333.666] |
| AgentX | B：确认存在独立工作流 | UNKNOWN | UNKNOWN | UNKNOWN |
| AgentX | B：K 少于正收益独立工作流数 | UNKNOWN | UNKNOWN | UNKNOWN |
| AgentX | C：正式结构允许 checkpoint 覆盖多个 pending | 69 / 100.00% | 77.177 / 0.000 [0.000, 1967.475] | 66.807 / 0.000 [0.000, 2333.666] |
| AgentX | C：至少一个 checkpoint 对多个 pending 有正独立收益 | 66 / 95.65% | 80.685 / 0.000 [0.000, 1967.475] | 69.844 / 0.000 [0.000, 2333.666] |
| AgentX | C：实际贪心某一步对多个 pending 有正新增收益 | 48 / 69.57% | 4.772 / 0.000 [0.000, 35.306] | 0.000 / 0.000 [0.000, 0.000] |
| AgentX | C：选中共享 checkpoint 的删除会增加成本 | 66 / 95.65% | 80.685 / 0.000 [0.000, 1967.475] | 69.844 / 0.000 [0.000, 2333.666] |
| AgentX | C：联合求和与最大单项贡献发生相对排序反转 | 9 / 13.04% | 4.144 / 0.000 [0.000, 12.432] | 0.000 / 0.000 [0.000, 0.000] |
| AgentX | D：与 StandaloneTopK 集合不同但成本相同 | 22 / 31.88% | 78.679 / 0.000 [0.000, 1432.519] | 0.000 / 0.000 [0.000, 0.000] |
| AgentX | D：与 Marconi 集合不同但成本相同 | 53 / 76.81% | 0.000 / 0.000 [0.000, 0.000] | 0.000 / 0.000 [0.000, 0.000] |
| AgentX | E：与 Marconi 成本相同 | 53 / 76.81% | 0.000 / 0.000 [0.000, 0.000] | 0.000 / 0.000 [0.000, 0.000] |
| AgentX | E：对 Marconi 改善在 (1e-9, 1] ms | 0 / 0.00% | 不适用 | 不适用 |
| AgentX | E：对 Marconi 成本更高 | 0 / 0.00% | 不适用 | 不适用 |

完整 P25/P75/P95 分布见 [summary.json](summary.json)。分类是证据描述，不是额外策略实验：C 联合排序反转要求实际选中 c 的总 marginal 大于某个候选 d，但 c 的最大单个 pending 新增收益小于 d 的最大单项。只比较已经计算的贡献向量，没有构造或运行新 baseline。AgentX 该现象为 9 cases（3 snapshots × 3 budgets），均与 StandaloneTopK 最终成本持平；它说明联合求和影响相对排序，不证明相对 StandaloneTopK 的独特最终收益。

## 持平原因

| 数据集 | 与 StandaloneTopK 持平原因 | cases | 达到所有候选可实现的最大前沿 | C(S)=0 |
|---|---|---:|---:|---:|
| OpenHands | 不同集合但每个待续请求的可执行前沿相同 | 287 | 287 | 267 |
| OpenHands | 选中集合相同 | 73 | 20 | 20 |
| AgentX | 不同集合但每个待续请求的可执行前沿相同 | 22 | 22 | 0 |
| AgentX | 选中集合相同 | 45 | 45 | 0 |

- OpenHands：360/504（71.43%）持平，其中 73 集合相同，287 集合不同；后者全部逐 pending 的 E/G 相同，静态额外保留不会改善目标。267 个不同集合 tie 已零成本，其余 20 个仍有非零但候选无法消除的残余 gap。
- AgentX：67/69（97.10%）持平，其中 45 集合相同、22 集合不同。67 个都达到所有现有候选可实现的前沿，但全部 C(S)>0；不能把非零不可恢复残余误称为分配失败，也不能把不同集合误称为收益。
- 所有不同集合 objective ties 都具有逐 pending 相同前沿；没有通过不同 pending 成本正负抵消制造的 tie。

## 机制归因与论文边界

1. **相对 Marconi 的主要证据是价值信号与保留位置不同。** OpenHands Marconi 用归一化最近访问值 + alpha×归一化增量计算效率；FlowState 用同一恢复模型下当前可执行前沿的增益。386 个胜例中 245 个与 StandaloneTopK 成本持平；不能把对 Marconi 的全部收益归因于 marginal updates。323 个 Marconi 胜例还包含最终可删除的冗余状态，但这是最终集合诊断，不是 Marconi 内部顺序证据。AgentX 16 个 Marconi 胜例中 14 个与 StandaloneTopK 持平；其原始 utility metadata 缺失，不能补写排序动机。
2. **相对 StandaloneTopK 的额外收益可归因于冗余消除后的重分配。** 全部 OpenHands 144 个、AgentX 2 个严格胜例的首个分歧处，静态下一候选相对共同 retained prefix 的 marginal 为零；FlowState 的替代选择对另一个 pending 有正收益。OpenHands 还明确跨越独立 workflows；这不是因为改变了 standalone score 或 K。
3. **Global allocation 在 OpenHands 的主要可观察作用是独立 workflow 预算竞争。** 四个 workflows 无 cross-continuation sharing；504 cases 都有同一 pending 的收益重叠，90 cases 的 K 少于可服务工作流数。其余 cases 仍可通过避免静态冗余占用而覆盖更多 workflows。不能从此推广“AgentX 也主要是独立 workflows”，因为真实完整血缘未保留在读取的正式 manifest 中。
4. **AgentX 有共享结构和共享收益，但两胜例不证明同时多分支新增收益带来的严格优势。** 69 cases 有结构共享，66 有正收益共享，48 在实际贪心某一步同时增加多个 pending 的收益，9 有联合排序反转；两个严格胜例都在第二步选择共享祖先，但此时其在已覆盖分支上的 marginal 已为零。共享可用性参与了恢复另一个分支，却没有被隔离为独立因果因素；StandaloneTopK 本身也计算所有 pending 的联合独立收益。
5. **当前证据边界。** 支持冻结 allocation objective 下的冗余感知全局分配；不支持“共享联合收益普遍导致相对独立 TopK 严格改善”、所有阶段/任意拓扑最优或运行时加速的泛化结论。当前不必为冗余重分配 claim 新增 baseline；若论文坚持共享建模本身带来独特优势，需要另行控制共享因素的证据，当前未隔离该因素。本轮不启动 P1。

## 七个代表性案例

| 案例 | snapshot | K | vs Marconi, ms | vs StandaloneTopK, ms | 选择依据 |
|---|---|---:|---:|---:|---|
| A | rq3-openhands-main-g186-round4 | 4 | 582.242 | 753.200 | OpenHands 对 Marconi 最大绝对改善；同时也是对 StandaloneTopK 最大改善 |
| B | rq3-openhands-main-g073-round3 | 3 | 115.491 | 0.000 | OpenHands 对 Marconi 严格改善排序的下中位案例 |
| C | rq3-openhands-main-g138-round4 | 4 | 402.325 | 174.070 | OpenHands 对 StandaloneTopK 严格改善排序的下中位案例 |
| D | rq3-agentx-g006-epoch35 | 8 | 271.335 | 2333.666 | AgentX 第一个严格优于 StandaloneTopK 的案例 |
| E | rq3-agentx-g018-epoch30 | 7 | 1114.436 | 2276.008 | AgentX 第二个严格优于 StandaloneTopK 的案例 |
| F | rq3-agentx-g000-epoch109 | 27 | 0.000 | 0.000 | AgentX 不同集合但 objective 相同：按预算、ID 升序首例 |
| G | rq3-openhands-main-g009-round3 | 3 | 0.000 | 0.000 | OpenHands 对 Marconi 持平：按预算、ID 升序首例 |

全部 candidates、原始 IDs、compatibility、selected sets、贪心过程与逐 pending 成本见 [case_studies.md](case_studies.md) 和 case_A.json 至 case_G.json。每个 baseline 的逐 pending ΔC 之和均核对为 aggregate ΔC。两个 AgentX 严格案例的机制相同，仍逐一列出，避免用单例替代完整审查。

## 论文与组会建议

- 论文图 1：[figure_marginal.svg](figure_marginal.svg)。显示案例 A/D/E 首个分歧时，独立高分候选的当前 marginal 已归零。三个面板使用独立标注的尺度，不比较未归一化柱长。
- 论文图 2：[figure_cost.svg](figure_cost.svg)。显示案例 A/B/D/F 的逐 pending 成本，包括中等 Marconi 收益但静态策略持平，以及 AgentX 的高残余成本 tie。最多这两张，不自动编辑论文。
- 组会第 1 页：两个总体、四 baseline 的全量 W/T/L 与 ties；强调不是 GPU latency。
- 组会第 2 页：案例 A/D/E 的首次分歧和余量重分配，使用图 1。
- 组会第 3 页：案例 F 的联合收益存在但最终 tie，配合图 2、未知 metadata 和 claim 边界。

可增加的最小英文分析段落：

> On the frozen allocation cases, FlowState improves modeled recovery cost over StandaloneTopK in 144/504 OpenHands cases and 2/69 AgentX cases, with ties in all remaining cases. At the first divergent selection in every strict win, the static choice has zero marginal benefit relative to the common retained prefix, whereas FlowState selects a checkpoint that benefits another pending continuation. The two AgentX wins involve shared ancestors, but each divergent step contributes new benefit to only one continuation; they therefore support redundancy-aware reallocation rather than an isolated advantage from simultaneous multi-branch gains. Different selected sets alone are not evidence of improvement: 287 OpenHands cases and 22 AgentX cases retain identical per-continuation executable frontiers despite different selections.

## 验证与证据定位

- 573 cases、2865 个策略 objective、573 次现有 GlobalOptimizer CPU 回放、573 个 StandaloneTopK 集合全部核对通过。OpenHands 504 个 Marconi 集合由冻结原始 utility 重新排序核对通过。
- 使用原有 GlobalOptimizer.select，成本钩子仅以冻结兼容矩阵和原 RecoveryCostModel 计算同一 C(S)；没有产生新总体，也没有修改 policy。按正式记录检查 eligibility、K、candidate IDs、T、compatibility 和 digest；AgentX common_input_verified 与各策略 input digest 一致。
- StandaloneTopK 的逐项收益求和排序与消融记录的总成本差展示排序在这 573 cases 中顺序相同。正式 1e-9 ms 比较容差保留；greedy 停止条件仍来自现有代码，不额外设阈值。
- Section 4 分解核对：b(p,c)=Phi(T,T)−Phi(T−L,T)（仅对正式可兼容候选）；u_p=max b；C(S)=C(空集)−sum u_p；Delta=sum max(b−u,0)。逐 pending 成本和冻结 E/G 全部一致。
- 原实现位置：`flowstate/optimizer.py:41`（选择）、`:95`（marginal）、`:110`（tie）、`:124`（停止）；`flowstate/recovery_model.py:60`（Phi）；`flowstate/state_catalog.py:60`（compatibility）；`evaluation/sota_policies.py:100` 与 `:253`（Marconi utility）；`evaluation/rq3_sanity_structure_audit.py:1116`（StandaloneTopK）；`evaluation/rq5_frozen_snapshot_ablation.py:196`（冻结展示顺序）；`evaluation/rq3_agentx_formal_policy_evaluation.py:1101`（正式 manifest 写出字段）。
- Frozen OpenHands W/T/L 报告：`evaluation/runtime_artifacts/rq3_sanity_structure_audit_20260904_121500/STEP_13G_A_FINAL_REPORT.md:232`。全量源行号逐例记录在 all_cases.jsonl 的 source_file/source_line，七例同时列在案例文档中。
- UNKNOWN：AgentX 原始 Marconi metadata 与完整真实 lineage 路径；未从 ID 或 encoded workflow 推断补齐。正式兼容集合、checkpoint L、pending T 和全部结果仍可完整审计。
- 文件 SHA-256 的前后结果与验证门禁见 [integrity.json](integrity.json)。仅独立分析目录有新产物；未启动 GPU、Engine、trajectory 或后续开发。

