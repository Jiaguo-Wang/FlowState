# 真实多 Workflow 分配结构审计

**INCONCLUSIVE**。现有材料支持独立请求之间的受控分配、真实 token 共享输入，以及一个共同 recurrent 状态的合法恢复；尚不能建立具有代表性的、同一执行边界上已知 Pending 与自然物化候选之间的共享分配问题，并据此证明强简单策略与集合分配的严格差异。

本轮没有新增候选、改预算、重建收益矩阵、运行选择器或 GPU。复核既有完整比较，避免对不合格输入生成貌似正式的新性能数字。AgentX 的既有兼容冲突是其执行解释的阻断项，不把它推广为所有数据源都不可用。

## 1. 全部来源与可用性

|来源|已核对范围|实际可用证据|不足与适用边界|
|---|---|---|---|
|OpenHands 正式快照|168 个快照、504 个预算条件；逐一读取 168 份 runtime_correctness|每快照 4 个待提交输入、8–20 个回放完成后的候选；共同 barrier 实测 residency/FA frontier；全部执行完成与不读未来字段通过|四个会话由实验 runner 组成共同 barrier，不是原生应用同时 Ready 的记录；正式矩阵无跨请求正收益。未把 workflow ID 相同自动视为 token 认证|
|AgentX/Weka|23 个 B2 快照、568 个候选，69 条预算结果；既有全量 393 traces 审计|原始嵌套 subagent、hash blocks、时间字段；受控回放后的 node、prefix digest、Raw prefix 谓词|hash 不是原始 token 文本；链和 current-context Pending 有派生成分；最终 Formal 有 70 条 Raw 不成立的正收益边，不能作运行时合法矩阵|
|LMCache 冻结输入|全部 16 对、32 个文件 SHA 与真实整数数组 LCP 复核|同一目标模型下的 exact prefix；4 对共享完整消息，12 对主要共享 system 消息片段|输入历史不是同时 Pending 证明；没有为全部 16 对保存同边界的自然 checkpoint 集合、FA frontier、在线预算|
|LMCache A5|lmcache_10 全部已有运行结果与保存代码|C0=9984 真实物化，P1/P2 独立合法恢复，recurrent-only eviction 反事实；顺序执行后 recurrent 节点从 1 到 3|C0 位置是受控验证选点，非原生应用缓存策略证据；新增节点仅有 ID 汇总，没有足够 prefix/owner 映射与后续两个 Ready 事件来构造新分配快照|
|LMCache A6|16 对、47 个候选、63 个预算条件|合法 token 矩阵及完整 CPU 策略差异、反例|中点候选是确定性情景构造，Full-FA 假设，不能作为本轮真实负载证据|

来源路径、摘要及逐快照清单见 `evidence_manifest.json`、`snapshot_inventory.csv`。后者包含 **191 个历史快照加 16 个输入对**，这 207 行不是 207 个合格原生分配快照。每行记录 continuation 数、候选数、正共享边、重叠、K_min、预算及来源；无法确定的 LMCache 自然候选结构明确为 UNKNOWN。

## 2. 执行历史、兼容性与时间边界

**OpenHands：受控 barrier 证据成立，跨 Workflow 共享没有得到正式证明。**
`evaluation/rq3_openhands_neutral_collector.py:240–300` 只构造已执行轮次与下一次请求输入，在待续 assistant 标记处停止，不读其输出或更远消息；`:397` 检查请求完成，`:465–477` 用实际输入前缀 inspect checkpoint，`:519–537` 在共同 barrier 重新核对身份和驻留，`:554–584` 查询每个待提交输入 FA frontier。168 份历史记录均通过本轮对应字段复核，详见 `openhands_barrier_checks.json`。

但 collector 的这些检查不能自动代替每条 candidate→pending 的 token 等价检查；正式优化器仍通过 `flowstate/state_catalog.py:60–73` 的 workflow/lineage 与 L≤T 谓词建立边。当前汇总记录中 request_rows 保留长度与测量，未直接保留每条完整 input_ids。**本轮没有把 0 条正式跨请求正边解释为所有历史 token 之间绝无可共享前缀**，也没有自行把不同会话相同 system 文本加入矩阵。若要新增边，必须另有完全一致的目标输入和物化 prefix 证据。

**AgentX：原生归属、派生 ancestry 和可执行 prefix 必须分开。**
`evaluation/agentx_runtime_snapshot_collection.py:647–651,884–895` 对实际合成 token 前缀计算 Raw；`evaluation/rq3_agentx_formal_policy_evaluation.py:327–425` 使用 B1 逻辑关系组合 B2 状态并重编码。Gate 1 Step 3 已记录 70 条 Formal-only 正边分布于 21 个快照；不是简单 ID 映射错误。Raw 空集不代表对所有可能输入都非法，但对于当时被列举的输入，缺少必要前缀条件。本轮不删除这 70 条边后重新评分，因为那将改变正式矩阵且仍不能补出原生 Pending。

原始 t/api_time 的重叠不是 Ready/Queued；B1 active target 也不是原生当前等待恢复请求。来源：`evaluation/p2_shared_cohort_feasibility_audit/final_report.md` 的 B/C 节，以及 `evaluation/gate1_step3_compatibility_retention_contract_audit/audit_report.md` 的 1–2 节。

**LMCache：真实历史前缀不是过去 epoch 可用的未来信息。**
本轮重新验证 16 对文件摘要与数组 LCP，没有重新 tokenize 或改变输入。对其完整历史已知不意味着过去任一时刻这些 token 已可在线获取。A6 C1/C2 中点不得转成“当时自然存在”的候选，A6 的 R=N 不得转成实测 FA。完整消息共享案例为 lmcache_10/12/13/16；lmcache_13 还含短历史被长历史包含的结构，不能直接称原生两分支同时等待。

## 3. 可核验的合法共享案例与自然边界检查

`lmcache_10`：P1=16338、P2=21520 tokens，LCP=10206。A5 的 C0=[0,9984) 对两者 token 一致，独立恢复输出分别等于各自 full recompute；这是现有最直接的共同 recurrent 状态可复用证据。源码与记录：

- `evaluation/submission_p0_lmcache_recovery/input_identity.json`：实际输入路径、SHA、LCP 和 checkpoint prefix 摘要。
- `runtime_evidence.json`：`materializations`、`independent_resumes`、`dual_branch`、`mamba_dependency_counterfactual`。
- `lmcache_runtime_report.md:105–110`：实测恢复、FA 保留和反事实。

DUAL 是先执行 P1、再执行 P2。P1 的 physical/executable=9984/9984；P2 的 physical/executable=10206/9984，后者多出的 FA 命中归因于 P1 留下的共有段。执行后新增 mamba node IDs `[90,92]`，总数 1→3，C0 保持稳定。

这些是**已经执行后自然观察到的节点**，比 A6 中点构造更值得后续调查。但 `lmcache_recovery_smoke.py:731–769` 保存的是 count 和新增 ID 汇总，不保存该 DUAL census 的完整节点映射。不能仅由 ID 反推 L、owner、属于哪个历史、对下一请求是否兼容；也不能把 P2 的 `mamba_branching_seqlen=10176` 直接等同两个新增节点的完整 checkpoint 身份。两请求已执行完成，不能把它们又当成仍 Pending。

只看认证的 C0 子问题，可证明它对两个输入复用；**不能从一个已认证项推导完整自然候选集的 K_min 或预算竞争**。因此本轮不填造这个运行边界的收益矩阵，不给它标注 K_min=2。

## 4. 预算和全部冻结策略结果

以下均为既有 modeled cost，胜/平/负从 FlowState 视角；多预算不是独立统计样本。没有新运行时延结论。

|来源|预算结构|vs PCB / LDR / WFC|vs StandaloneTopK|Exact OPT 已有子集|
|---|---|---|---|---|
|OpenHands|K=2–15；K_min=4；90/504 条件 K<K_min，均在 25% 预算|每个策略 0/504/0|144/360/0|434/434 objective 匹配|
|AgentX 正式逻辑矩阵|K=1–111；K_min=0–2；0/69 条件紧预算|每个策略 0/69/0|2/67/0|48/48 匹配，但不认证矩阵 runtime 合法性|
|LMCache A6 情景|K=0..m；全部16组 K_min=2；16个 K=1 紧预算|PCB/LDR 各 3/58/2；WFC 26/37/0|1/62/0|61/63 匹配，2 个 greedy 失败|

OpenHands 每条正收益候选仅帮助一个请求，重叠发生在同一请求的浅/深 checkpoint；AgentX 正式矩阵 22/23 快照存在正共享，不能忽略其兼容冲突而升级为真实共享。其 continuation 数为 2–8、候选数 6–149，均已逐快照列出。K_min 是覆盖全候选可实现前沿，不等于零 gap，也不是物理池占用。

已复用 P1-B 的所有六类策略相关记录，Exact OPT 从原正式结果的 tractable 字段读取，非由论文数字抄录。详见 `historical_policy_differences.csv`、`historical_exact_opt.csv`。本轮没有满足新增真实共享快照资格的合法冻结矩阵，所以没有执行新的 baseline；不能用合成替代后宣称满足任务目标。

## 5. 差异机制与失败案例

**真正可解释、但不独有于全局共享的改善：**
OpenHands `rq3-openhands-main-g001-round3`、K=3。StandaloneTopK 选择 `B_TURN_002,B_TURN_003,C_TURN_003`，成本 318.540424 ms；FlowState 选择 `B_TURN_003,C_TURN_003,A_TURN_003`，成本 159.270212 ms。B_TURN_003 已覆盖 B 的最深前沿后，B_TURN_002 增益为零；FlowState 将额度用于 A。PCB/LDR/WFC 同样得到 FlowState 结果。因此这一例证明冗余分配问题，不能证明 FlowState 优于强 workflow-aware 方法。完整 ID 与原始行号见逐条件 CSV。

**已有 AgentX 严格例不能越过合法性门禁：**
`rq3-agentx-g006-epoch35/K=8` 与 `rq3-agentx-g018-epoch30/K=7` 对 StandaloneTopK 分别改善 2333.666429、2276.007728 modeled ms。强基线均持平，且收益关系必须受 Step 3 语义限制；不把这两个例子包装成已认证多分支共享优势，也不以它们为依据挑选新样本。

**必须保留的 FlowState 反例，仅为 A6 情景：**
`lmcache_10/K=2` 与 `lmcache_16/K=2`，FlowState 首轮选择共享 C0 后无法撤回，最后 C0+C2 分别比 C1+C2 多 125.634930、113.305779 ms；PCB/LDR 与 Exact OPT 更好。对应 K=1 时，共享 C0 的总收益又更优。它们说明 set-dependent greedy 本身并不保证最优，不是实际运行时失败认证。

A6 `lmcache_12/K=2` 的正例是动态边际更新避开被 C2 部分覆盖的 C0、改选 C1；不能拿它的 midpoint 和 Full-FA 规则补造本轮自然候选。A6 全部负例、tie 和分层保留在原 `evaluation/submission_p1_allocation_structure/paired_summary.json`，本轮未重新筛选或改写。

## 6. 判断与唯一下一阶段场景

- **是否已有真实可复用共享状态？有。** A5 C0 对两个具体输入的受控恢复证据成立，generation 等限制沿用 A5。
- **是否已有可代表自然多 Continuation 分配的完整快照？当前核对范围内不足。** 缺少同边界自然候选映射、真实 Ready/Pending 和可执行 retention quota；OpenHands 的受控独立请求 barrier 不能填补跨请求共享，AgentX 的逻辑图不能填补 token 必要条件。
- **强简单策略与集合分配是否在合格真实共享负载上出现严格差异？尚不能判断。** 现有严格差异分别属于冗余独立链、存在语义限制的历史矩阵或显式情景。

**唯一最值得验证的 Runtime 场景：沿 lmcache_10 已有 A5 顺序执行路径，记录自然新增节点与真实后续恢复边界。** 不采用 A6 中点、不预置 K<K_min：保留执行后每个实际物化节点的 prefix/位置、owner/generation、slot/COW 引用；只有应用真实产生至少两个已 Ready 的续请求时，才在暂停 dispatch 的同边界记录其完整目标 tokens、FA frontier 与候选驻留。先验证自然节点对这些输入的 exact prefix 和安全恢复，再记录已预先确定且实际执行的保留 quota/headroom。

如果该路径不能自然产生两个待续请求，或新增节点无法服务它们，应报告该场景不成立，不能重复提交已完成请求或拼接未来历史来冒充原生 Pending。这一步首先验证资格，不做策略性能比赛，也不预言 FlowState 获胜。

## 7. 完整性与复现

复现：在仓库根目录运行 `PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python evaluation/real_multiworkflow_structure_audit/audit.py`。只读 JSON/JSONL、CSV 与 token 输入，输出本目录，不导入 engine 或选择器。来源前后 SHA 一致见 evidence_manifest.json。未修改旧实验、正式算法或 Phi，未启动 GPU。本轮完成后停止。
