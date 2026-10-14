# Task A6 分配结构审计

**STRUCTURAL_READY：仅为新冻结 Full-FA 情景中的数学结构与 modeled objective 验证。** 不认证新的 GPU 状态、生产 Pending 或物理容量竞争。

## 协议与完成范围

Protocol Lock v1 在本次所有收益计算前写入 protocol_lock.json，包含原文、输入 Manifest SHA-256、Git commit、带时区冻结时间及 Exact OPT 平局规则。这是本次新冻结，不是此前预注册。16 组全部纳入；先调试 lmcache_10，再按相同规则处理其余样本，没有按策略胜负筛选。冻结时间与原文可直接核对 JSON。

身份核查复用接管阶段的 32 个输入和 46 个来源摘要；本轮进一步核对所有案例的目标模型、tokenizer/模板身份及本地文件 SHA-256。原 PCB/LDR/WFC 模块按 implementation_lock.json 校验后直接调用，没有修改旧 573 个条件、Task A/A5 或核心算法。

## 候选、兼容与收益

固定 q=256，C0=floor(LCP/q)q，C1/C2=floor((LCP+N_i)/(2q))q。C0 与原 Manifest 核对一致。C1/C2 必须严格处于 LCP 与本输入末端之间，否则省略。共提出 48 个位置，纳入 **47 个**：lmcache_13 的 C1=83968，低于 LCP=84068，按协议省略，未替补。

每条边对源前缀与目标整数数组逐 token 比较，且 0<L≤T。代码未调用 lineage 判断 LMCache 兼容性，也未构造虚假 workflow identity。矩阵来自比较结果：16 个 C0 均服务两个目标，其余 31 个分支候选各服务自身目标，共 **63 条正收益边**；共享 C0 与分支候选有 **31 对收益支持重叠**。不兼容边收益为零。

所有记录固定 T_i=N_i，planning_target_source=ASSUMED_FULL_FA，physical_fa_evidence=NOT_VALIDATED，pending_event_evidence=UNKNOWN。Phi 来自 evaluation/FORMAL_RECOVERY_MODEL.md 和 flowstate/recovery_model.py:39–70；最大 T=86784，均处于 131072 范围内，非负及单调性检查通过，没有裁剪或重新拟合。

收益矩阵显式保存 T/L/G、空集成本、有效恢复成本与收益。所有 ms 都是该情景模型输出，不是本轮 GPU 测量。

| 案例 | N1 / N2 | LCP | 候选数 | K_min | 当前 context 长度可容纳 |
|---|---|---|---|---|---|
| lmcache_10 | 16338 / 21520 | 10206 | 3 | 2 | True |
| lmcache_01 | 13092 / 30283 | 2719 | 3 | 2 | True |
| lmcache_02 | 27261 / 56753 | 2719 | 3 | 2 | False |
| lmcache_03 | 23167 / 33379 | 2719 | 3 | 2 | True |
| lmcache_04 | 19263 / 46774 | 2719 | 3 | 2 | False |
| lmcache_05 | 17553 / 41728 | 2719 | 3 | 2 | True |
| lmcache_06 | 29580 / 22478 | 2719 | 3 | 2 | True |
| lmcache_07 | 27275 / 32346 | 2719 | 3 | 2 | True |
| lmcache_08 | 30340 / 59593 | 2719 | 3 | 2 | False |
| lmcache_09 | 34495 / 19939 | 2719 | 3 | 2 | True |
| lmcache_11 | 34820 / 59690 | 2719 | 3 | 2 | False |
| lmcache_12 | 21036 / 46397 | 8834 | 3 | 2 | False |
| lmcache_13 | 84070 / 86784 | 84068 | 2 | 2 | False |
| lmcache_14 | 33355 / 62121 | 2719 | 3 | 2 | False |
| lmcache_15 | 34741 / 74519 | 2719 | 3 | 2 | False |
| lmcache_16 | 40340 / 69883 | 35650 | 3 | 2 | False |

GPU_CONTEXT_FEASIBLE 仅按 **A5 context_length=45056** 与完整输入长度比较：7 组通过长度条件、9 组不通过。true 不证明显存、并发、候选物化或 admission 可行；false 的样本仍完整保留于 CPU 审计。

## 预算结构

16/16 的 K_min=2，因此每组 K=1 都为 STRUCTURALLY_TIGHT_IN_SCENARIO。15 组三候选枚举 K=0,1,2,3，lmcache_13 两候选枚举 K=0,1,2，共 **63 个案例–预算条件、378 条策略结果**。K=0 是空集对照；K=m 是可用容量等于候选数的对照，按各策略停止规则不强迫选择零边际候选。

三候选组达到全候选前沿需 C1+C2；lmcache_13 需 C0+C2。全候选前沿仍小于 N_i，故达到 K_min 不代表 G=0。budget_structure_analysis.csv 保存全候选前沿及不可消除 residual。这里 K 是等大小逻辑项数，不是物理 pool 大小，也没有将 A5 的 16 slots 当作 K。

## 策略适配与验证

- FlowState 独立矩阵 greedy 保留 optimizer.py 的逐轮总成本下降、1e-9 ms 比较容差、ID 平局与非正边际停止。仅替换兼容数据来源，不修改核心文件。
- 手工两类合法血缘测试（独立与共享），各 K=0..3，与正式 GlobalOptimizer 逐集合及成本核对，8/8 PASS。这些是代码一致性测试，不是 LMCache 的虚构血缘认证。
- PCB 真实读取 Phi，独立推荐后静态排序和 ID 去重；LDR 仅按深度推荐；WFC 优先推进请求数，再按总 token 推进排序。直接复用 p1_strong_baseline_validation 的锁定实现，不把 WFC 改写成 Phi 边际收益。
- StandaloneTopK 复现 rq3_sanity_structure_audit.py:1116–1131 的静态总独立收益，过滤≤1e-9 项，取 K 后按 ID 返回集合。它的返回顺序不是额外 marginal trace，原静态分数排序保存在 selection_traces.json。
- Exact OPT 穷举所有 |S|≤K 集合，成本平手按 ID 元组字典序，全部最优集合一并保存；不将集合不同但成本相同算作胜利。
- 独立使用 sum_p max b 检查全部 378 条结果及对 Exact OPT 的差距，通过。所有源摘要保持不变。

## 全体与预定分层结果

下表以**案例–预算条件**计数，仅描述，不将同一案例各预算视为独立统计样本；没有显著性检验或 runtime speedup。

| 分层 / 条件数 | 对照 | FlowState 胜 / 平 / 负 |
|---|---|---|
| 全部/63 | PCB | 3 / 58 / 2 |
| 全部/63 | LDR | 3 / 58 / 2 |
| 全部/63 | WFC | 26 / 37 / 0 |
| 全部/63 | StandaloneTopK | 1 / 62 / 0 |
| 全部/63 | Exact OPT | 0 / 61 / 2 |
| 完整消息共享组/15 | PCB | 3 / 10 / 2 |
| 完整消息共享组/15 | LDR | 3 / 10 / 2 |
| 完整消息共享组/15 | WFC | 2 / 13 / 0 |
| 完整消息共享组/15 | StandaloneTopK | 1 / 14 / 0 |
| 完整消息共享组/15 | Exact OPT | 0 / 13 / 2 |
| 部分消息共享组/48 | PCB | 0 / 48 / 0 |
| 部分消息共享组/48 | LDR | 0 / 48 / 0 |
| 部分消息共享组/48 | WFC | 24 / 24 / 0 |
| 部分消息共享组/48 | StandaloneTopK | 0 / 48 / 0 |
| 部分消息共享组/48 | Exact OPT | 0 / 48 / 0 |

FlowState **并不全面优于 PCB/LDR/WFC**。对 PCB 和 LDR 均为 3 胜、58 平、2 负；对 WFC 为 26 胜、37 平、0 负；对 StandaloneTopK 为 1 胜、62 平、0 负；与 Exact OPT 为 61 平、2 负。

FlowState 相对 PCB/LDR 的严格胜例：lmcache_10、13、16 的 K=1，分别减少 155.482828、4295.103948、651.449317 modeled ms。严格负例：**lmcache_10/K=2** 多 125.634930 ms、**lmcache_16/K=2** 多 113.305779 ms；这两个也就是相对 Exact OPT 的全部失败。所有胜平负案例 ID（包括 WFC 的 26 个胜例）完整列于 paired_summary.json，未删除任何负例或 tie。

## 四个完整消息共享案例

| 案例 | K | 策略 | 最终集合 | 情景 modeled cost (ms) |
|---|---|---|---|---|
| lmcache_10 | 1 | FlowState | ["C0"] | 751.209000 |
| lmcache_10 | 1 | PCB | ["C2"] | 906.691828 |
| lmcache_10 | 1 | LDR | ["C2"] | 906.691828 |
| lmcache_10 | 1 | WFC | ["C0"] | 751.209000 |
| lmcache_10 | 1 | StandaloneTopK | ["C0"] | 751.209000 |
| lmcache_10 | 1 | Exact OPT | ["C0"] | 751.209000 |
| lmcache_10 | 2 | FlowState | ["C0", "C2"] | 517.796156 |
| lmcache_10 | 2 | PCB | ["C1", "C2"] | 392.161226 |
| lmcache_10 | 2 | LDR | ["C1", "C2"] | 392.161226 |
| lmcache_10 | 2 | WFC | ["C0", "C2"] | 517.796156 |
| lmcache_10 | 2 | StandaloneTopK | ["C0", "C2"] | 517.796156 |
| lmcache_10 | 2 | Exact OPT | ["C1", "C2"] | 392.161226 |
| lmcache_12 | 1 | FlowState | ["C2"] | 1796.396928 |
| lmcache_12 | 1 | PCB | ["C2"] | 1796.396928 |
| lmcache_12 | 1 | LDR | ["C2"] | 1796.396928 |
| lmcache_12 | 1 | WFC | ["C0"] | 2276.319837 |
| lmcache_12 | 1 | StandaloneTopK | ["C2"] | 1796.396928 |
| lmcache_12 | 1 | Exact OPT | ["C2"] | 1796.396928 |
| lmcache_12 | 2 | FlowState | ["C1", "C2"] | 1205.047609 |
| lmcache_12 | 2 | PCB | ["C1", "C2"] | 1205.047609 |
| lmcache_12 | 2 | LDR | ["C1", "C2"] | 1205.047609 |
| lmcache_12 | 2 | WFC | ["C0", "C2"] | 1457.710177 |
| lmcache_12 | 2 | StandaloneTopK | ["C0", "C2"] | 1457.710177 |
| lmcache_12 | 2 | Exact OPT | ["C1", "C2"] | 1205.047609 |
| lmcache_13 | 1 | FlowState | ["C0"] | 190.075561 |
| lmcache_13 | 1 | PCB | ["C2"] | 4485.179510 |
| lmcache_13 | 1 | LDR | ["C2"] | 4485.179510 |
| lmcache_13 | 1 | WFC | ["C0"] | 190.075561 |
| lmcache_13 | 1 | StandaloneTopK | ["C0"] | 190.075561 |
| lmcache_13 | 1 | Exact OPT | ["C0"] | 190.075561 |
| lmcache_13 | 2 | FlowState | ["C0", "C2"] | 106.968561 |
| lmcache_13 | 2 | PCB | ["C0", "C2"] | 106.968561 |
| lmcache_13 | 2 | LDR | ["C0", "C2"] | 106.968561 |
| lmcache_13 | 2 | WFC | ["C0", "C2"] | 106.968561 |
| lmcache_13 | 2 | StandaloneTopK | ["C0", "C2"] | 106.968561 |
| lmcache_13 | 2 | Exact OPT | ["C0", "C2"] | 106.968561 |
| lmcache_16 | 1 | FlowState | ["C0"] | 2118.293641 |
| lmcache_16 | 1 | PCB | ["C2"] | 2769.742958 |
| lmcache_16 | 1 | LDR | ["C2"] | 2769.742958 |
| lmcache_16 | 1 | WFC | ["C0"] | 2118.293641 |
| lmcache_16 | 1 | StandaloneTopK | ["C0"] | 2118.293641 |
| lmcache_16 | 1 | Exact OPT | ["C0"] | 2118.293641 |
| lmcache_16 | 2 | FlowState | ["C0", "C2"] | 1220.634759 |
| lmcache_16 | 2 | PCB | ["C1", "C2"] | 1107.328980 |
| lmcache_16 | 2 | LDR | ["C1", "C2"] | 1107.328980 |
| lmcache_16 | 2 | WFC | ["C0", "C2"] | 1220.634759 |
| lmcache_16 | 2 | StandaloneTopK | ["C0", "C2"] | 1220.634759 |
| lmcache_16 | 2 | Exact OPT | ["C1", "C2"] | 1107.328980 |

lmcache_10 与 16：K=1 的共享 C0 总收益最大，FlowState/WFC/StandaloneTopK 先选 C0；K=2 时 greedy 不能撤回第一步，选择 C0+C2，但全局最优 C1+C2 可同时取得更深分支前沿。因此同一案例可以在 K=1 胜过 PCB/LDR，在 K=2 反而落败；有集合收益结构不等于 greedy 必然最优。

lmcache_12：C2 独立收益最大；选择后 C0 的 P2 收益已被覆盖，FlowState 转选 C1，而 StandaloneTopK 静态排序保留 C0，造成 252.662567 ms 差距。WFC 首轮优先覆盖数而选 C0，也无法在 K=2 撤回它。

lmcache_13：C1 无效，C0 是 P1 的唯一可用项；K=1 共享收益明显优于 C2。原始历史包含关系仍须注明：短会话历史是长会话历史的前缀，不能把它当原生同时 fork 的证据。

## 为什么仍大量持平

部分消息共享的 12 组，FlowState 与 PCB/LDR 在全部 48 条件持平：C0 仅位于浅 system 前缀，最大分支收益先被选，第二个分支再被选；K=0 和充足预算也不产生最终成本差异。这是固定实例的排序与支配结构，可在 selection_traces.json 中逐项核对，不是一般等价定理。

例如 lmcache_01：K=1 选择 C2，K=2 选择 C1+C2，三者同集合；WFC 因覆盖数优先先选 C0，不能由其结果推断完整 KVFlow 或其他系统的效果。全部案例的选择集合、返回顺序、逐请求 E/G/收益与成本均保留于 policy_comparison.csv；轨迹保留所有 greedy 候选分数和冻结 baseline 返回的证据。

## 两个核心问题与证据边界

**问题一：情景中存在非平凡集合分配结构吗？存在。** 16 组均有 token 合法共享和收益重叠、K=1<K_min，且固定规则下出现严格 modeled objective 差异。结论标签分别为 STRUCTURALLY_TIGHT_IN_SCENARIO 与 STRICT_MODELED_OBJECTIVE_DIFFERENCE。既有 FlowState 胜例，也有 greedy 失败反例，不能归结为必需采用复杂 greedy。

**问题二：距离真实可执行物理预算分配缺什么？** C1/C2 是本次确定性中点探针，不是已有物化事件。47 个纳入候选中只有 lmcache_10/C0 有 A5 历史认证，其余 46 个均 PROPOSED。A5 不认证这里的 Full-FA、共同 Pending 边界和容量合同；旧状态也不代表当前仍驻留。

未来必须因果地先执行并物化分支历史，再形成合法待恢复请求；不可把离线后续 tokens 当作过去 epoch 的在线信息。须同一静止边界取得 FA frontier、全部候选物化及 owner/generation、active/COW 引用、slot 去重和保留配额证据，并验证安全 reconciliation 与两分支恢复。K<K_min 不能替代 GPU 物理池压力或长期 quota 的执行证据。

**最小下一步建议（仅设计，未执行）**：优先 lmcache_10，沿既有 A5 C0=9984 补查协议 C1/C2 的因果物化与同边界同时驻留；核实真实 T、两个 Ready/Pending、实际 retention quota 与 working/COW headroom；再验证删除未选状态后合法恢复、输出和 attention-side 保留。若真实 FA 或容量不符合本情景，应报告差异，不通过事后调整预算追求优势。首先是状态/合同验证，不是性能比较。

## 复现与来源

运行 reproduce.sh，只执行 CPU 审计与报告，不启动 Engine。protocol_lock.json 不由复现脚本改写；analyze.py 验证冻结输入与旧源码摘要后执行。preflight.py 是已完成接管的来源核查入口，不需重复数据准备。

- 来源输入：agentic_workflow_dataset_research/phase2_submission_inputs/runtime_candidate_manifest.jsonl 及各 input_artifact_paths。
- 历史运行证据：evaluation/submission_p0_lmcache_recovery/runtime_evidence.json、input_identity.json、lmcache_runtime_report.md。
- 策略：flowstate/optimizer.py:89–147；evaluation/p1_strong_baseline_validation/pcb.py、ldr.py、wfc.py；evaluation/rq3_sanity_structure_audit.py:1116–1131。
- 模型：evaluation/FORMAL_RECOVERY_MODEL.md；flowstate/recovery_model.py:39–70。
- 结果：candidate_manifest.jsonl、compatibility_matrix.json、benefit_matrix.json、budget_structure_analysis.csv、policy_comparison.csv、case_structure.json、selection_traces.json、paired_summary.json。
- 验证：validation.json、source_integrity_before.json、artifact_manifest.json。takeover_progress.md 是协议补充前的历史接管进度，其缺规则阻塞已由 Protocol Lock v1 解除。

完成本轮后停止。没有新增 GPU Runtime Certification，没有算法优势的生产泛化结论。
