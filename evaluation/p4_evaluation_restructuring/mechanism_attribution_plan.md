# 机制归因组织方案

主实验给完整分布；案例只解释机制，不能承担平均性能或因果隔离证明。证据为 P0 `final_report.md:39,91,99`、`case_studies.md:166,423` 和 P1-B `final_report.md:95` 起；真实 JSON 来源以 `per_case_evidence.csv` 为准。

| 展示位置 | 冻结案例或总体 | 展示内容 | 可支持 | 不可支持 |
|---|---|---|---|---|
| 去冗余主案例 | Case C，rq3-openhands-main-g138-round4，25%，K=4 | 四条链、静态 TopK 排序、选择时零 marginal、最终 E/G/C；Marconi 与 TopK 分别列 | TopK 重复覆盖挤占另一 continuation 的状态；FlowState 避免此冗余 | 这不是相对 PCB/LDR/WFC 的优势；四方法该例同集合同目标 |
| 共享解释案例 | Case F，rq3-agentx-g000-epoch109，25%，K=27 | req0000 同时贡献两个 pending，各 1740.926 modeled ms；总 3481.852；FlowState 与 PCB/LDR 首选不同 | 联合收益确实改变顺序 | 最终优势不存在；强方法同集合同前沿，TopK 虽有冗余但也达同目标 |
| 位置感知反例 | P1-B g051-round5，三个预算 | 相同 L=7808，T 不同；FlowState 先 D，LDR/WFC 先 C；成本差为零 | Phi 影响中间排序 | 模型准确性尚未转为该总体的最终增益 |
| 覆盖优先差异 | P1-B g005-epoch8，K=2 | WFC 先共享浅项，FlowState 先深单分支项；最终都留下 | heuristic 与 Phi 方法确实不同 | 顺序不同不等于目标改善 |
| 全量归因 | OpenHands 504、AgentX 69 条件 | TopK 严格胜例 144、2；强基线严格胜例均 0 | 已观测改善主要可由去冗余解释 | 不能写成一般 global 优化已超过简单 workflow-aware 保留 |

Case C 对 Marconi 的差为 402.325 modeled ms，对 StandaloneTopK 为 174.070 modeled ms；必须在表头注明比较对象。Case F 的联合收益与最终持平应放在同一个图注中。AgentX g006/K8 与 g018/K7 是仅有的 TopK 严格胜例，但第二步共享候选只对一个尚未覆盖分支有正新增收益，不能用它们证明“同时多分支收益”的独立因果效果。

## 消融表的组织

`rq5_frozen_snapshot_ablation_20260908_110726/per_case_results.jsonl` 中 full、without_position_aware、without_set_dependent 分别是完整方法、去位置感知和 StandaloneTopK。读取冻结 selection、common_cost，不执行消融选择器。使用同一个 Phi 评分不等于选择规则使用同一个 Phi；在图注区分“决策信号”与“共同评价指标”。

P1-B 的 LDR/WFC 不读取完整 Phi；PCB 使用冻结独立收益并按逐 continuation 规则分配。它们是明确的简化工作流基线，不能命名为完整 KVFlow、TOPAS 或原生 Marconi。

## 数学与经验的边界

当正收益支持互不重叠、checkpoint 等大小、只有 cardinality 预算且没有额外耦合时，逐 continuation 取最大收益，再取至多 K 个最大的正收益 continuation，达到 Exact OPT（P1 证明）。OpenHands 满足这一可分结构；其中 90 个受限条件仍等价，不能归因成“预算总是足够”。

AgentX 全部 K≥K_min_frontier，方法可达到相同的全候选前沿；但该前沿不一定等于 planning target。全部 AgentX snapshot 的不可消除 residual 都大于零，因此 Figure A 的平台不是“全部完全恢复”。

位置感知是否必要、一般集合边际是否独特有效，都需要预算边界实际惩罚错误排序的结构。下一步只建议先研究可实现的兼容集合、收益重叠与物理容量条件；解析反例不能记入真实实验，也不得从策略胜负反向筛选 cohort。

## 冻结消融的全量核对

本轮仅读取比较字段，分预算明细见 `ablation_summary.csv`；未重新运行选择器。

| 数据集 | 消融 | FlowState 胜/平/负 | 最终集合不同 | 顺序不同 |
|---|---|---|---:|---:|
| OpenHands | 去位置感知 | 0/504/0 | 0 | 0 |
| AgentX | 去位置感知 | 0/69/0 | 0 | 0 |
| OpenHands | StandaloneTopK | 144/360/0 | 431 | 431 |
| AgentX | StandaloneTopK | 2/67/0 | 24 | 24 |

去位置感知消融与LDR不是同一个决策规则：前者的冻结结果连顺序都相同；LDR在g051上的顺序差异不能冒称为该消融的结果。这进一步限制了位置感知最终增益主张。以上跨预算计数仅为描述性汇总。
