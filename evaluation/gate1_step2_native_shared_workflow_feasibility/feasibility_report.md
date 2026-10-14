# Gate 1 / Step 2：原生共享工作流与有效容量竞争

**PARTIAL**。2026-10-12。找到冻结逻辑共享/重叠结构，也找到旧受控 Runtime 的共同 recurrent 源与独立 COW 分支物理证据；但没有找到同时具有原生 Agent 待续事件、统一可执行兼容边界和可靠保留容量不足的案例。**已核实 A/B/C/D 原生交集为 0；这是检索范围内零个已验证案例，不是证明真实系统中不存在该问题。**

本轮没有新工作负载、策略比较、GPU、Engine、Replay、算法或模型修改。只复核已有矩阵、历史日志、源码与容量记录。所有成本仅为冻结 Phi 的离线收益，不报告新的性能结果。

## 1. 检查范围与统计分母

| 范围 | 本轮记录数 | 统计单位与限制 |
|---|---:|---|
| OpenHands 正式总体 | 504 | 168快照×3预算，不是504个独立epoch |
| AgentX 正式总体 | 69 | 23快照×3预算，关联23条B2历史回放驻留记录 |
| Exp-F1 Hybrid Fork | 7 | 六个子请求规模1/2/4/8/16/32，另一个2子请求指纹修正运行；文件名K是分支数量，不是保留预算 |
| WP3A direct eviction | 20 | 两类目标×两种干预×5次重复，仅一个被测consumer，不是分配总体 |
| 合计 | 600 | 异质历史记录；不能作为600个独立原生工作流样本推断 |

上述600行全部保存在 `scenario_statistics.csv`，包括失败/异常关联记录，不按结果筛选。辅助复用 P2 全量原始字段审查：393条AgentX traces、98,827个invocation、1,697个subagent容器。该统计来自既有 `raw_schema_audit.json` 与 `raw_trace_inventory.json`，**本轮没有重新扫描约1.85GB原始trace全字段**，也没有给393条trace伪造allocation boundary。完整路径见清单。

P2既有census还记有1,206个派生shared-context eligible epoch，其中23个有正式B2边界，另1,183个没有对应驻留/原生ready证据。本轮不重建这些epoch，其可执行P/C/T与K_min保持UNKNOWN，不计入600个已逐记录检查的场景。没有查遍所有外部数据源；不能外推零结果到未审查部署。

## 2. 证据路径和版本

主要目录别名：

- `G1`：`evaluation/gate1_step1_allocation_equivalence/`。
- `R`：`evaluation/p1_strong_baseline_validation/per_case_results.jsonl`，R:n表示第n行。
- `B2`：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/`。
- `F1`：`/home/wjg/code/newfind/artifacts/exp_f1_hybrid_fork_20260814T160725Z/`。
- `WP3A`：`motivation/artifacts/wp3a_direct_eviction_20260819/`。

已读 `evaluation/p2_shared_cohort_feasibility_audit/`、`evaluation/p3_native_instrumentation_feasibility/`、`evaluation/p3_physical_retention_budget_contract/` 的报告与合同；结合原始 B2/runtime_snapshots.jsonl、runtime_correctness.json，F1/raw/*.json、metadata.json 和 WP3A/raw_runs.csv、driver_events.jsonl。

SGLang版本在这些记录中均为0.5.17，但配置不能混用：AgentX B2池192槽；F1 geometry记录no_buffer、实测471槽；WP3A extra_buffer、16槽；OpenHands正式控制路径28槽。F1 metadata 的 `git_commit` 是该实验元数据字段，不能未经核实解释为SGLang提交。

**源码证据的新限制：**P3-G0保存的16个 `/tmp/audit_sglang/sglang-v0517-tag/...` 源文件当前均不存在。本轮保留旧 `source_verification.json` 的历史摘要和定位，但不宣称重新核验了这些runtime源码。没有拿另一个本地SGLang分支替代。`evidence_manifest.json.missing_historical_source_paths`列出全部缺失路径。当前仍可读取FlowState collector、模型与F1 probe/driver源码。

## 3. 四层共享证据逐层判断

### 3.1 原始AgentX：逻辑归属不等于原生fork

`evaluation/agentx_chain_detection.py:298`用最大hash LCP推导parent/depth，并将子请求t写为fork_time；`evaluation/agentx_runtime_snapshot_collection.py:322`生成派生request ID，`:327`使用current-context pending，`:710`开始串行重放。它们不能证明原生parent_request_id、同一时刻Ready/Queued状态或继承某个物理checkpoint。

P2全量schema没有原生fork、ready、dispatch、state-owner/generation事件。subagent容器有agent_id只证明记录组织；API时间区间重叠不能证明待恢复请求同时排队。原模型轨迹为Claude形状数据，后续Qwen状态不是原始执行的状态。

BLOCK_SHAPE说明的真实位置是 `/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614/REPLAY_SEMANTICS.json`；B2目录下同名文件当前不存在。其prefix topology/fork semantics可保留的宣告仅针对构造方法，不补足原生事件。

### 3.2 AgentX B2：驻留存在，但兼容口径有层次

本轮逐项联接23个快照的568个候选：checkpoint ID一致，recurrent_resident与fa_resident为真，正式L等于回放recurrent_prefix_length；T=min(input_token_count,resident_fa_frontier)。这些是历史**回放边界**的状态，不是原生Agent历史驻留。

但原始 `checkpoints[*].compatible_pending_ids` 与正式评价 `objective_usable_pending_ids` 在全部23快照、69预算条件中存在差异。CSV记录每条件不同候选数，不修改任何矩阵。正式评价使用派生workflow兼容，raw记录使用回放/runtime路径口径；它们不是同一证明，不能任选更有利的一方。

例如R:505的 `rq3-agentx-g000-epoch109`，trace `02bc0afb13f7a2d9efa86c28511261d85c0e`：正式req0000位置37760，对两个pending正收益；B2/runtime_snapshots.jsonl:1中该项 `compatible_pending_ids=[]`。该快照有10个候选的raw/formal兼容集合不同。req0000实际有驻留slot，**仍不能推出其对正式两个pending均为runtime合法复用源**。这不是本轮发现CPU评分计算错误，而是不能跨证据层升级兼容主张。

原B2 `runtime_correctness.json`记录 `all_gpu_cleanup_pass=false`，同时记录无unexpected native eviction/rematerialization及23 eligible；本轮保留该负面字段，不把整份历史记录宣称为全部正确性门禁PASS。`fork_semantics_preserved=true`也不能取代原生事件链。

### 3.3 历史F1：原生Runtime复制路径成立，原生Agent工作流未证实

最完整代表：`F1/raw/fork_K2_fpfix.json`。

- `phases.parent_state.leaf_node`：父节点39，边界16384，recurrent源槽2，FA前缀digest `214fdbb069291f13`。
- `phases.fork.mid.req_snapshot.running`：fork-c0、fork-c1分别使用槽3、5；两者前16384个FA索引digest均等于父节点；`waiting=[]`，此时是**两个运行中的请求，不是两个等待allocation的continuation**。
- `phases.fork_cow.copy_records`：2→3、3→4、2→5、5→6；共同父源被复制到独立分支工作槽，后两类链条包含tracking复制，不能把四次全部称为四个fork。
- `phases.divergence`：父指纹前后不变 `982a6658387d5db8`；子槽指纹 `d47be3f8a2e9d46b` / `9fb98ae80540532c`，相互不同且不同于父；两个独立replay样本均token_equal。
- `phases.fork_census_after.mamba_pool`：总471、空闲466、evictable5。状态并存已证实，容量墙没有触及。

`/home/wjg/code/newfind/exp-f1/harness/exp_f1.py:138–172`以async gather并发提交；`:291`为parent_full+预构造suffix。不是仅根据相同prompt推断状态共享：有上面真实COW源、FA索引和指纹；但它仍是受控token工作流，不是已有Agent的原生fork/ready事件。其合法性属于Runtime路径，不能升级为真实应用自然发生的allocation epoch。

七个cell全部检查，六个多子请求cell记为S2（**限定受控Runtime层**）；单子请求cell记S0。K8原始正确性样本有1个token mismatch，六个主cell缺可靠字节指纹，后续2子请求修正运行才补齐指纹。保留这些限制，不以历史GO或“数值漂移”解释替代本轮证明。详情见 `historical_fork_details.json`。

### 3.4 WP3A：组件因果证据不是容量竞争

20个正式episode记录target仅recurrent驱逐、FA保存、槽数变化和consumer恢复。driver_events第一行配置池16、并发1；对称删除target或decoy一槽，不是让多个待续分支在物理紧预算下竞争。Join使用分支输出token序列化，并非合并兄弟recurrent states。没有与本任务相同的P/C/K边界，因此列出但不推导K_min、不算S4。

WP3A报告引用的server_full.log当前本目录不存在；本轮依赖保留的CSV和driver事件，不把缺失日志当作重新读取成功的原始来源。

## 4. 可执行兼容、收益重叠和模型计算

正式573条件的P、C、L、T、A来自G1/matrices.jsonl，逐候选用未修改 `flowstate/recovery_model.py:39,58` 复算b，全部在1e−9 ms内一致；没有把未来下一请求代入current-context。**形式在线信息边界已核对，不等于原生Ready事件已存在。**

E_p(S)=max({L_c:已选且A_pc=1}∪{0})；G=T−E。FA hit本身不能赋予A=1；尤其上节raw/formal差异不能通过FA驻留标志抹平。对F1/WP3A没有统一待续目标的场景，P/T/A/B保持UNKNOWN，不把在运行的request或未来下一轮假定为已Ready。

具体逻辑结构例（不是原生可执行认证）：R:505，pending顺序T=[110527,20223,110527]。

| 候选短名 | L | 正式兼容向量 | b向量，modeled ms |
|---|---:|---|---|
| req0000 | 37760 | [1,0,1] | [1740.926206175,0,1740.926206175] |
| req0005 | 51328 | [1,0,0] | [2470.221830882,0,0] |

二者对p1重叠：更深项提高p1，但共享浅项还服务p3；两个singleton收益向量不可相互支配。因此存在逻辑恢复收益取舍。K_min=2，正式K=27/54/81，实际不用放弃两项中的任何一项。该形式模式不等于两个分支都各有更深状态：P2更严格“双深分支”的计数为0/69，本轮沿用其证据边界，不凭一个singleton便宣称双分支完整竞争。

### A/B/C/D计数

D操作化为存在两个可行单候选方案，其收益向量各在至少一个pending严格更优（1e−9 ms）；这是非平凡**备选取舍**的充分结构证书。是否被迫取舍还需C成立。CSV `nontrivial_allocation`采用此定义，不能单独据此声称容量不足。

| 总体（条件） | A的逻辑代理：共享正收益 | B：正收益覆盖重叠 | C的逻辑代理：K<K_min | D：不可相互支配的备选 | 逻辑A∩B∩C∩D | 完整原生交集 |
|---|---:|---:|---:|---:|---:|---:|
| OpenHands 504 | 0 | 504 | 90 | 504 | 0 | 0已验证 |
| AgentX 69 | 66 | 33 | 0 | 27 | 0 | 0已验证 |
| F1 7 | UNKNOWN（无allocation P/T） | UNKNOWN | UNKNOWN（K不是保留预算） | UNKNOWN | UNKNOWN | 0已验证 |
| WP3A 20 | 0（每行一个consumer） | UNKNOWN | UNKNOWN | 0 | 未建立 | 0已验证 |

对应比例：OpenHands B为100%、逻辑C为90/504≈17.86%；AgentX逻辑A为66/69≈95.65%、B为33/69≈47.83%、D为27/69≈39.13%、逻辑C为0%。正式条件交集0/573；扩展到600条异质历史记录仍为0/600已认证，不能把这个比值当作真实应用发生率。Runtime资源约束下的C没有任何一条被认证；UNKNOWN不当作已证伪。

## 5. 真实容量合同审查

本轮按最大候选前沿覆盖位重新做有限bitmask最小覆盖计算，没有执行策略。结果与冻结K_min逐条一致：OpenHands全部4；AgentX为2/1/0的条件数27/39/3。`capacity_certificates.json`保留前沿与K_min。

离线阈值仅为：K_min=2时正整数紧预算只有K=1；K_min=4时为1/2/3；K_min≤1没有正整数紧预算。这不是建议给AgentX改成K=1，也不产生新性能结果。

物理证据：

1. B2配置max_mamba_cache_size=192，而候选最多149，正式逻辑K仍由候选百分比定义。没有绑定persistent-only quota、working/COW reserve或allocation barrier归还协议。不能从25%命名推断GPU紧缺。
2. F1总池471，六个主规模完成后空闲分别468/466/462/454/438/406。确实有随分支增长的状态消耗，但所有观测规模都未触发准入墙；不能用外推“约235分支”声称已观察到资源竞争。文件里的K是1–32个children，与本任务K语义不同。
3. Qwen TP1完整状态slice含conv+temporal。P3既有fingerprint给出51,511,296 bytes=49.125MiB；F1 geometry的shape为temporal[24,472,32,128,128]、conv[24,472,8192,3]，支持同样的slice大小。
4. F1 probe `f1_probe.py:203–217`把包含padding槽的tensor总bytes除以471个可用槽，得到报告51,620,661 bytes/slot；这带有padding摊销，不能作为真正一次slot copy的字节数。旧结果不修改；本轮只区分该记账口径与完整状态slice大小，避免据此精确推断物理紧预算。
5. COW共享只读源但为各分支复制完整写入目标，并非多分支工作状态共用一个可写槽。普通recurrent池由工作/跟踪/缓存角色共享；归还槽不代表释放底层预分配HBM。碎片、FA页共享不自动减少完整recurrent slice的logical计费。

P3-G1合同结论仍适用：若K就是同一时刻全部驻留状态的硬上限，已驻留C必满足K_min≤|C|≤K，不可能同时K<K_min。要合法观察到超过长期保留预算的候选，须事前规定可撤销借用、真实执行reservation及admission前归还；当前历史正式记录未证明这套合同已经实施。本轮不倒推reserve或挑选较小K来制造竞争。

## 6. S0–S4分层与证据缺口

| 类别 | 正式573 | F1 7 | WP3A 20 | 含义边界 |
|---|---:|---:|---:|---|
| S0 | 507 | 1 | 20 | 该记录没有本任务所需的可验证多pending原生共享关系，不否认某个状态/组件本身有效 |
| S1 | 66 | 0 | 0 | 正式逻辑共享，但原生继承/ready与runtime兼容证据未贯通 |
| S2 | 0 | 6 | 0 | 仅受控Runtime共享源与独立分支执行已证；不是原生Agent同一待续边界 |
| S3 | 0 | 0 | 0 | 没有在既有正式预算中发现共享+重叠+紧预算，不人为补造 |
| S4 | 0 | 0 | 0 | 没有完整原生执行和真实有效资源竞争证书 |

缺失链条：Agent native fork → 稳定request/branch ID → checkpoint materialization与ownership/generation → 同时Ready但未dispatch → 同一barrier的FA/recurrent读数 → 已预注册且可兑现的retention容量合同。现有SGLang Tree-of-Thought fork能力仅有P3历史源码审查，本轮临时源码副本缺失，不能拿它当已执行pilot记录。

F1最完整物理案例与AgentX最完整逻辑矩阵属于不同实验和输入；**不能拼接成一条S4证据链**。同样不能把OpenHands的90个逻辑紧预算与AgentX的66个共享条件取跨总体交集。

## 7. 五个研究问题与唯一下一步

**Q1：真实Agent中有合法共享recurrent checkpoint吗？** 逻辑AgentX兼容图中有；受控SGLang执行中共同源被多个请求合法COW复用有直接记录；原生Agent在同一待续边界复用的完整证据仍UNKNOWN。

**Q2：共享祖先与分支状态有收益竞争吗？** 冻结矩阵中的重叠与不可支配向量已证实，AgentX为33/69有重叠、27/69有取舍备选；这些结果不认证原生继承关系，且既有预算没有迫使放弃关键前沿。

**Q3：有实际依据的有效容量不足吗？** 有真实池大小、完整state大小、COW分支占用与配置，但没有绑定到该原生待续边界并满足K<K_min的保留预算。容量增长趋势不是已发生容量墙。

**Q4：是否找到完整交集？** 0个已验证。正式逻辑总体本身缺少共享与紧预算的交集；其外还缺原生事件/兼容/配额证据。因此既有工作负载结构不足和证据缺失同时存在，不能仅选择其中一种解释。

**Q5：值得进入强基线差异评估吗？** 目前不宜直接进入。已知数据仍不能回答该问题；也不能因COW放大或逻辑共享便宣称Global Allocation具有不可替代创新。研究假设尚有机制可行性，独立必要性未建立。

**唯一最高优先级下一步：**先冻结并验证一个原生Agent admission边界的联合事件—物理保留资源合同，使fork、Ready、实际checkpoint与依据独立服务需求确定的retention预算属于同一证据链。优先解决已有P3指出的接口/证据缺口，不按K_min倒选预算；其是否能自然产生有效竞争应作为可失败的可行性门禁。此次只提出建议，不开发、不运行、不作策略比较。

本轮产物为本报告、600行场景表、证据摘要清单、容量证书、历史Fork细节及独立只读脚本。`evidence_manifest.json`记录所读来源SHA-256、版本/配置、缺失路径、脚本摘要；源文件前后不变。完成后停止。
