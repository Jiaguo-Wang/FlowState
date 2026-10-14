# Gate 1 / Step 1：Global Allocation 等价性与工作负载结构审计

**PASS**。日期：2026-10-12。

已独立核对 573 个冻结条件的输入、原始 FlowState 结果、PCB/LDR/WFC 保存的选择与共同评分。四者最终 checkpoint ID 集合、逐请求 E/G 与 modeled objective 全部一致。主要原因有结构证据支持：OpenHands 的正收益支持按请求可分；AgentX 的所有正式预算均可容纳实现全候选最深前沿的必要项。**这不证明四种算法一般等价，也未证明集合感知优化相对这些简单规则具有独立的最终收益。**

本轮没有执行选择器或新增工作负载，只读取旧结果、复算其共同评分和结构统计；没有 GPU、Engine、Runtime 修改或冻结文件修改。此前 P1-C 的数学反例没有在本轮重新运行，也不作为真实工作负载证据。

## 1. 来源、实例与复现范围

| 来源 | 路径与用途 |
|---|---|
| 完整手册 | `evaluation/p4_complete_experimental_evidence/complete_experiment_handbook.md:260`（实验4）、`:367`（实验5） |
| OpenHands 原始结果 | `evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl`，504行 |
| OpenHands 冻结快照 | `evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/snapshots/`，逐实例来源保存在 P0 metadata.source |
| AgentX 原始结果 | `rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl`，69行 |
| AgentX 兼容性与配置证据 | 同目录 `formal_snapshot_manifest.json`；`evaluation/rq3_agentx_formal_policy_evaluation.py:33`、`:401`、`:464` |
| 正式执行入口 | `evaluation/rq3_formal_policy_evaluation.py`、`evaluation/rq3_agentx_formal_policy_evaluation.py`；collector 为 `evaluation/rq3_openhands_neutral_collector.py` |
| 强基线冻结记录 R | `evaluation/p1_strong_baseline_validation/per_case_results.jsonl`，573行；以下 R:n 表示其第 n 行 |
| 强基线入口与锁 | 同目录 `phase1.py`、`evaluate.py:80`、`implementation_lock.json`、`validation.json` |
| 旧结构证据 | `evaluation/p0_allocation_mechanism_audit/all_cases.jsonl`、`evaluation/p1c_global_allocation_equivalence/global_allocation_equivalence_audit.md`；P4-G1 `overall_allocation_matrix.csv` 为既有统一表，本轮不覆盖 |

OpenHands：168 个四工作流快照，三个预算各 168 条件。原采集来自 OpenHands sessions，policy-blind 固定分组、round 2/3/4/5；200 个预定组中 168 个通过，32 个 non_monotonic_inputs 失败且没有补位。采集过程不在本轮重跑。AgentX：23 个正式快照，三个预算各 23 条件。每个快照的 trace ID、OpenHands 每组四条 workflow/trace ID 均在本轮 CSV 的 `trace_ids` 保存；全部来源路径与行号逐行保留。

预算统一为 25%、50%、75%，K=max(1,floor(r×候选数))。这些是相同快照的重复预算条件，不是 573 个独立统计样本，也不是 573 次新 GPU 测量。

| 数据集 | 快照 / 条件 | 每条件 pending 数 | 候选数范围 | K 范围 |
|---|---:|---|---:|---:|
| OpenHands | 168 / 504 | 全部为4 | 8–20 | 2–15 |
| AgentX | 23 / 69 | 2:18、3:18、4:18、5:9、7:3、8:3（数值:条件数） | 6–149 | 1–111 |

数据来自 R:1–573；本轮 `summary.json`、`gate1_step1_instance_statistics.csv` 可逐项重现，范围不表示所有中间整数均出现。

### 字节预算没有被省略

OpenHands 原快照的每项 memory_bytes 均为 51,511,296 bytes；预算变体按 `rq3_formal_policy_evaluation.py:193` 的 create_budget_variant 设为 K×checkpoint_size。AgentX 在 `rq3_agentx_formal_policy_evaluation.py:401` 构造相同大小候选，大小常量来自 `evaluation/controlled_multiworkflow_v1/scenario.py:11` 的 49.125×1024²=51,511,296。

CSV 分别记录 `checkpoint_size_bytes`、`budget_bytes` 和四策略 `used_bytes`，逐项验证不超预算。OpenHands 大小证据标记 RECORDED，AgentX 标记 RECONSTRUCTED_FROM_CODE：它不是 manifest 中独立保存的逐候选物理大小日志，不能冒充显存实测。这里验证的是正式**等大小逻辑计费合同**；物理池容量、COW headroom 与实际显存压力不由这些数值证明。

### 独立输入核验

调用既有只读 `common.load_cases`（`common.py:77`），没有调用 evaluator 的 main：

- 原始结果与 P0、ablation 的 snapshot digest 一致；K、预算比例、候选数与 pending 数一致。
- OpenHands 候选逐项对照 canonical snapshot、等大小与驻留标志；同 workflow、lineage 前缀、L≤T 的兼容规则逐项复核。
- AgentX 对照 `objective_usable_pending_ids` 和原始 common_input_verified / policy_input_digests；不把包含 L>T 的形式关系误当成本目标可用关系。
- 从正式 L/T 和 Phi 复算 benefit，与原始逐请求收益比较；P1-B input_digest 与恢复后的输入一致。
- 保留完整 ID 原字符串，无大小写折叠、截断、末尾 req 编号映射或模糊匹配；验证唯一性后排序仅用于集合比较。不同 trace 中相同 req 编号仍是不同 ID。

因此共同输入是同一候选、兼容性、目标、预算和共同评价模型；**不意味着各策略实际读取同样多的成本信息**。

## 2. 决策规则与结果核对

| 方法 | 真正决策规则 | 同分、过滤、停止 |
|---|---|---|
| FlowState | 当前集合 E 下的 C(S)−C(S∪{c}) 最大化；实际读取 Phi | recurrent_resident、等大小；收益差在1e−9内按ID；最佳收益≤0或用完K停止 |
| PCB | 每 pending 推荐最高正 standalone benefit，再按推荐收益扫描 | 收益>0；精确浮点收益同分按候选ID，再按pending ID；重复ID跳过继续扫描；K或推荐耗尽停止 |
| LDR | 每 pending 推荐最深合法 L，再按L扫描 | L>0；L同分按候选ID，再按pending ID；去重；K或推荐耗尽停止；不读Phi |
| WFC | 动态E；优先最大受益请求数q，再最大总token推进r | q/r同分按ID；q=0或达到K停止；不读Phi |

源码：`flowstate/optimizer.py:34–149`、`evaluation/p1_strong_baseline_validation/pcb.py:5`、`ldr.py:4`、`wfc.py:4`。共同评分为 `common.py:57`，E=max兼容选中L，G=T−E，C=ΣPhi(G,T)，F=C(∅)−C。Phi 参数固定于 `flowstate/recovery_model.py:39`，未重新拟合。

本轮直接复核保存的选择，并重新评分 573×4=2292 个已选集合；不是新执行 baseline。每请求 T/E/G 必须精确相等，成本使用正式 1e−9 ms 复算容差。独立求和与旧记录可能有末位浮点差，不能将它当成策略收益；四种选择在本轮同一 scorer 下的差为零。CSV 的 objective_gap 为三条相对 FlowState 成本差的最大绝对值，分方法 signed gap 也保留；收益差的符号相反。

| 数据集 | 对每个 PCB/LDR/WFC：同ID集合 | 异集合但同目标 | 异集合且异目标 | 无法核实 |
|---|---:|---:|---:|---:|
| OpenHands | 504 | 0 | 0 | 0 |
| AgentX | 69 | 0 | 0 | 0 |
| 合计 | 573 | 0 | 0 | 0 |

“完全相同”同时指最终 ID 集合、逐请求 E/G、同模型目标，不是仅同一汇总平均值，更不是实际 GPU 性能相同。顺序仍可不同：OpenHands PCB/LDR/WFC 为0/3/3条件，AgentX为9/9/18条件，原始逐步证据见R以及旧P1-C完整清单。

没有发现同一 selector 被四个名称重复调用或强制覆写为 FlowState 集合：`evaluate.py:80–104` 独立读取 FlowState 原结果、PCB 已完成的 gate 结果、调用 LDR/WFC 再评分比较。`phase1.py` 执行 PCB 独立推荐；输入 Epoch 不含 FlowState 选择。LDR/WFC 正式执行期间还有 Phi 调用失败 guard。四份 implementation_lock 摘要本轮均匹配。已保存的顺序差异进一步排除了“所有决策过程完全相同”的解释。

## 3. Compatibility / Benefit / Budget 结构

`matrices.jsonl` 每行保存一个条件的 A、B、L、T、候选/请求 ID、预算及来源；矩阵行为 pending、列为候选。不重新定义 b，不以 FA 前缀重叠代替可执行收益。

定义如下：

- 正收益支持 C_c={p:b_pc>0}；shared 表示 |C_c|>1。
- overlap pair 为两个不同候选且 C_c∩C_d≠∅；这是候选对计数，不是 overlap token 数或共享 GPU slot 数。
- 每候选兼容度数与每请求兼容候选度数在 CSV 以有序数组保存，顺序与 matrices 一致。
- K_min_frontier 复用冻结最小覆盖证书：达到全候选 E^max 的最少项数，不要求 G=0。
- budget_pressure=K<K_min；同时独立检查实际最终集合外是否仍有正 marginal 的候选（1e−9 ms 数值比较）。两者在573条记录中逐条一致。不能只用候选数>K。
- 正收益支持 laminar 表示任意两集合不相交或一方包含另一方；它比“一候选仅覆盖一请求”弱，不能单独推出本问题被任意贪心精确求解。

| 指标（条件数） | OpenHands /504 | AgentX /69 |
|---|---:|---:|
| 有共享正收益候选 | 0 | 66 |
| 存在正收益覆盖重叠候选对 | 504 | 33 |
| K<K_min 且最终仍有正边际候选 | 90 | 0 |
| 预算未用满（四者相同） | 323 | 64 |
| 正收益候选数少于K | 0 | 59 |
| 已选前缀改变至少一个剩余候选的 singleton 收益 | 504 | 27 |
| 至少一对剩余候选的严格 singleton/marginal 次序反转 | 494 | 15 |
| FlowState 存在正最优 marginal tie 的步骤 | 82 | 0 |
| 正收益支持集合族为 laminar | 504 | 69 |

每个统计均可按 CSV 布尔列筛选获得全部 Case ID；`prefix_diagnostics.jsonl` 给出每步共同前缀、剩余候选 singleton/marginal、top tie ID。这里“严格次序反转”是**全部剩余候选对**在已保存 FlowState 前缀下的比较；不是 P0 的“共享联合收益导致首选反转”，不能把494/15与既有9/69混为一谈。没有包括已选项，没有把仅从严格顺序变成同分算作反转。

按快照而非预算去重的候选出现次数：OpenHands 的正收益度数0/1/2为12/2264/0，AgentX为484/54/30；来自168/23个快照。共享66条件对应22快照，9等数字若按预算计数不构成独立事件数。这一口径可从矩阵按 snapshot_id 去重恢复，旧P1-C `structure_summary.json` 也给出同值。

## 4. 对情况 A–E 的证据判定

| 情况 | OpenHands | AgentX | 解释边界 |
|---|---:|---:|---|
| A：无共享正收益 | 504 | 3 | 不等于没有同一请求内候选重叠 |
| B：共享但无候选正收益覆盖重叠 | 0 | 33 | 当前定义为严格无重叠，不用含糊的“很少” |
| C：重叠但预算足以达到候选最大前沿 | 414 | 33 | 不等于可以保留全部候选，也不等于G全零 |
| D：紧预算且四者实际顺序一致 | 90 | 0 | 事实是相同实际选择顺序，不声称对所有剩余候选的全排序一致 |
| E：共同实现/配置覆写导致表面等价 | 未发现 | 未发现 | 等大小、tie、退化是已知合同，不是证实了所有潜在实现缺陷不存在 |

类别非互斥。AgentX 无正收益的3条件也是A；虽然实现允许空集，这不是异常默认fallback。预算未用满不等于候选不足：OpenHands没有正收益候选数<K的条件，仍有323次提前停止，因为其余项对已经实现的前沿没有新增收益。

### 可复核归因与代表记录

1. **OpenHands 独立支持 + 预算截断**：`rq3-openhands-main-g000-round2`，R:1，K=2、K_min=4。A/C最深项入选，B/D仍有正收益；紧预算真实存在。所有90个紧预算条件均按同序选择，但数学上只保证PCB与精确最优可分化简，LDR/WFC的跨请求深度/收益顺序在这些数据上恰好兼容。
2. **位置估值改变顺序而不改变终点**：`rq3-openhands-main-g051-round5`，R:133–135，K=5/10/15。C/D最深L都7808，T为7808/7999，FlowState边际为299.473119577/299.520864654 ms，先D；LDR/WFC按ID先C。四个必要项都能保留。
3. **联合共享改变顺序**：`rq3-agentx-g000-epoch109`，R:505–507，25% K=27、K_min=2。FlowState先req0000（共享2请求，3481.852412350 ms），PCB/LDR先req0005（单请求，2470.221830882 ms）；最终均保留二者。完整ID不能只用上述短名跨trace匹配。
4. **覆盖数与收益排序冲突但可补选**：`rq3-agentx-g005-epoch8`，R:520–522，25% K=2。WFC先共享req0000（L384，29.101911990 ms），FlowState先req0001（L43840，1982.352784905 ms）。两项各是某个pending的唯一最大前沿实现者，最终都保留。

本轮对全部条件重新提取每个正 E^max 的唯一实现候选，并对其并集U验证 |U|=冻结K_min。OpenHands全部K_min=4；AgentX的K_min=2/1/0条件为27/39/3。全部483个预算足够条件的实际最终集合恰是U；CSV的 `unique_frontier_realizer_ids` 保存证书。这解释了为什么最终集合不仅同目标，而且连ID都相同。

**不能从 K≥K_min 单独推出任何贪心都达到U。** 当前数据还满足实际路径没有选入U之外的冗余项；这是已核验轨迹事实，不能省略这个前提。

## 5. 可化简条件与可推广边界

**已证明的充分条件**：每个正收益候选最多作用于一个pending、等大小cardinality预算、无其他耦合。在每个请求的候选组内只需保留收益最大项，其余被支配；替换为组内最大项不增加预算且不减少F。再选最大的至多K个正组收益就达到全局上界。这适用于504/573条件（全部OpenHands）；AgentX另3个无正收益条件也退化满足，合计507/573。精确数学比较下PCB与FlowState贪心均实现这一解；同ID结论还需唯一最大者或一致tie，浮点近似tie不得当成严格数学最大化。

**并不满足的错误化简**：“不同候选支持互不重叠”仅在36/573条件成立（全部在AgentX，含3空收益条件）；OpenHands全有同请求候选重叠。无共享不等于无重叠，也不等于 marginal 不变化。

**全部573条件支持集合为laminar**：这是可验证的额外结构，但本轮不提出“laminar必然使四算法等价”的新定理。每个continuation上深度越大收益越大，只能证明该continuation上的支配；分支深状态可能丢失祖先的其他覆盖，不能据此全局删除共享祖先。

**经验事实**：483/573条件具有足够预算且实际达候选上界；90个余下条件是独立支持且实际规则在预算边界一致。Position-aware Phi 可改变g051排序，共享可改变g000排序，但均未改变当前预算下终点。因此既非一般数学等价，也非“集合相关性根本未生效”；是当前可分结构及未截断有意义排序差异的预算/候选组合。

## 6. 可复现产物与限制

- `gate1_step1_instance_statistics.csv`：573行，唯一 instance_id，含输入规模、trace、字节计费、度数、重叠、预算压力、四集合/成本/收益、差值和归因；来源文件/行号均保存。
- `matrices.jsonl`：573个完整兼容/收益矩阵。
- `prefix_diagnostics.jsonl`：保存已有FlowState顺序的逐前缀分析；不执行新优化过程。
- `summary.json`：统计表与同集合/异集合分类。
- `integrity.json`：读取源文件SHA-256、锁定摘要与前后不变核验。
- `audit.py`：只读分析复现入口。命令：`PYTHONDONTWRITEBYTECODE=1 python3 evaluation/gate1_step1_allocation_equivalence/audit.py`。只写本目录。

AgentX 字节计费由代码恢复，未找到独立逐候选物理大小日志；真实并发pending、原生fork、物理预算竞争不能由正式CPU矩阵唯一恢复，保持UNKNOWN。上述限制不影响在已冻结逻辑合同内核实573条决策与解释主要结构，因此本轮判定PASS，**不是对原生共享运行场景或GPU收益的PASS**。

## 7. 四个核心回答

**Q1：为什么选择相同？** OpenHands 的独立正收益支持允许逐请求最大项精确化简；紧预算90条件下简单代理规则也与收益排序在实际选择处一致。其余414个OpenHands和69个AgentX条件均有唯一必要前沿集合U，预算可容纳U，四者实际路径恰好选完U。证据为逐实例矩阵、K_min、前缀诊断和最终集合证书。

**Q2：是否体现多Workflow共享竞争？** 体现OpenHands独立workflow之间的有限预算竞争，以及AgentX冻结兼容图上的共享与部分覆盖重叠；没有出现AgentX的K<K_min，亦不能由派生兼容/pending证明真实同一时刻原生分支排队。因此“共享、重叠且预算截断关键前沿”的真实执行组合仍缺证据。

**Q3：集合感知优化有独立必要性吗？** 集合边际变化和去冗余作用已证实；相对PCB/LDR/WFC的额外最终收益或不可替代性尚未证实。当前573条件中没有这类优势。此结论不否定其他结构上的可能价值，也不把模型成本等价外推为控制开销等价。

**Q4：唯一最高优先级问题。** 在预先固定、具有真实物理含义的容量合同下，原生共享工作流能否产生“共享祖先与多个分支专属深状态均有效，且K<K_min使已观察到的排序差异真正截断最终前沿”的决策边界？应验证该结构是否自然可获得，而非按FlowState胜负挑样本或事后调小K。本轮不实施下一阶段。
