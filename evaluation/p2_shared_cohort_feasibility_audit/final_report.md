# P2 真实共享工作负载 cohort 可行性审计

**NOT_FEASIBLE_FROM_EXISTING_DATA**

日期：2026-10-09（Asia/Shanghai）。当前已检查资料不能直接形成同时满足原生 lineage、真实同时 pending、checkpoint eligibility 与有效容量竞争的新 cohort。AgentX 可以继续作为请求形状和派生结构研究来源；不能仅靠重新解析现有字段补齐缺失的原生事件与历史驻留。需要新的联合事件证据来源，不等于必须下载另一个公开数据集。

## A. 原始来源与字段 provenance

原始文件：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl`。SHA-256：`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`，与冻结 B1 INPUTS 一致。全量流式检查 393 条 traces、393 个唯一 trace IDs、98827 个 invocation、1697 个 subagent 容器。原始模型记录示例为 Claude；旧 Qwen/SGLang 的状态是之后重放形状生成的，不能当成原模型历史状态。

| 项目 | 状态 | 证据与限制 |
|---|---|---|
| 原始数据来源 | AVAILABLE | 393 条唯一 trace，原始文件 SHA-256 与 B1 INPUTS 一致；只有 hash block 形状，不是原模型 recurrent state。 证据：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl`; `raw_schema_audit.json` |
| Workflow ID | PARTIAL | 原生 trace id 可追踪且 393/393 唯一；派生 conversation/workflow 分组并非全都来自原生工作流事件。 证据：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl $.id`; `evaluation/agentx_structure_audit.py:369` |
| Parent ID | PARTIAL | 嵌套 subagent 容器证明记录归属；没有 parent_request_id 或继承状态 identity。flat-chain 的 parent_conversation_id 由构建器赋值。 证据：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl 第1行 $.requests[48]`; `evaluation/agentx_structure_audit.py:430` |
| Fork Event | MISSING | 全量原始字段未出现原生 fork 事件；部分 parent/depth 来自最大 hash LCP，fork_time 取子请求 t，不能当成真实继承证明。 证据：`evaluation/agentx_chain_detection.py:298`; `evaluation/agentx_structure_audit.py:517` |
| Branch ID | PARTIAL | 1697 个 subagent 容器有 agent_id；fa/aux/wg 等链 ID 是启发式构建，不可由命名证明真实分支。 证据：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl $.requests[*].agent_id`; `evaluation/agentx_structure_audit.py:399` |
| Request ID | PARTIAL | 可用 trace+JSON 路径关联 invocation；原始 invocation 没有 request_id，::req 编号为派生标识。 证据：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl $.requests[*]`; `evaluation/agentx_runtime_snapshot_collection.py:322` |
| Timestamp | PARTIAL | 98827 invocations 均有 t 和正 api_time；缺 ready/dispatch/fork/native state generation 的同钟事件。绝对时间和跨来源同步 UNKNOWN。 证据：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl t/api_time/duration_ms`; `evaluation/agentx_chain_detection.py:102` |
| Pending 状态 | MISSING | 没有原生 ready/queued/running 状态转换；active conversation 或 API 区间重叠均不足以确认同时待续。 证据：`evaluation/agentx_structure_audit.py:748`; `evaluation/agentx_structure_audit.py:831`; `evaluation/agentx_runtime_snapshot_collection.py:327` |
| Checkpoint Identity | PARTIAL | 旧 replay 保存 checkpoint_id/request_index/node_id/runtime_identity_digest/recurrent_prefix_length；原始轨迹无 checkpoint，未知 epoch 无 identity。 证据：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/runtime_snapshots.jsonl`; `evaluation/agentx_runtime_snapshot_collection.py:831` |
| Residency Evidence | PARTIAL | 仅旧 23 个 replay snapshots 有 recurrent/FA 观测与 census；不是原始执行时驻留，也不能外推到其余 epoch。 证据：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/runtime_snapshots.jsonl`; `/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/`; `evaluation/agentx_runtime_snapshot_collection.py:849` |

全量 schema 的字段与计数见 raw_schema_audit.json；每个 trace 的源行号见 raw_trace_inventory.json。MISSING 表示所检查原始 schema 中没有该类记录，不是声称所有外部系统都没有日志；后者 UNKNOWN。1697 个 subagent 容器的 status 是完成摘要，不能替代 epoch 时点的调度状态。

## B. Fork / Parent / Pending 合法性

**Gate A 未建立。** `agentx_chain_detection.py:298` 调用 `_max_lcp_chain` 后建立 ChainFork，`:308` 将子请求 t 作为 fork_time；`agentx_structure_audit.py:415` 用 depth>0 区分 fork/spawn，`:517` 对 subagent 的首 hash 与 root tail 做 LCP。嵌套 subagent 确实提供结构归属，但没有父 invocation 与继承前缀的原生事件证明。旧“logical inherited ancestry”本身含推导环节，不能因为进入正式 manifest 就升级成原始观测。

**Gate B 未建立。** `agentx_structure_audit.py:748` 在 conversation start 时加入 active_targets，在请求完成事件更新，`:831` 在记录 snapshot 后才移除 end 事件；因而代码还会在 end 同时刻包含刚结束的 conversation，不能直接把其注释中的半开区间当成实现证明。`agentx_runtime_snapshot_collection.py:327` 明确沿用 current-context，而不是未来下一条请求。这个旧实验的 pending 定义不同于本轮要求的“真实已 ready、尚未执行”。

原始全部 invocation 都有 t 和正 api_time。按照记录坐标计算 API 区间重叠，373/393 traces 有至少两个重叠调用，最大重叠 23；这只能说明记录的 API 区间，不能证明同一 allocation boundary 上有两个等待恢复的 continuation，不能用它替代 ready/queued 事件。绝对时钟和跨分支时钟同步仍需额外证明。

这些发现不会修改已冻结数据或 P1-B 的 CPU 等价结论；它们限制的是新 cohort 的真实来源和并发 claim。原仓库根目录 B1 报告甚至含 pytest 临时 canonical root，因此本轮以实际使用的 B1 INPUTS、正式目录及源码为准，未盲信顶层 READY 文案。

## C. 可执行状态与 eligibility

**Gate C/D 部分成立且只限旧 replay 边界。** 旧 `runtime_snapshots.jsonl` 有 23 snapshots、568 个 checkpoint 的 checkpoint_id、request_index、node_id、runtime_identity_digest、recurrent_prefix_length、recurrent_resident 和 fa_resident；pending 有 input_token_count、resident_fa_frontier、planning_target、fa_query_side_effect_free。本轮逐项核对正式 L 等于 recurrent_prefix_length，T=min(input_token_count,R)，所有正式候选与旧 runtime IDs 一致。

collector 在 `agentx_runtime_snapshot_collection.py:710` 按顺序执行已完成请求，然后解析 checkpoint，并在 `:849` 观测 pending FA frontier。`REPLAY_SEMANTICS.json` 明确为 BLOCK_SHAPE 和确定性合成 token。原始 hash/history 不能证明 Qwen recurrent state 当时存在；旧 replay 确实观测到的 node/handle 也只能证明该次 replay 的边界，不证明原始生产时刻或其他 epoch。旧 raw runtime compatibility 与正式逻辑 compatibility 口径不同，本轮容量计算仅用正式 manifest，未混用。

原始数据缺 R_p、state generation/owner 和 checkpoint materialization 事件；无法由 token history 唯一恢复。对于未重放的 epoch，A 可从输入长度形成逻辑量，但是否是当时 ready continuation 的 anchor 未证明；R/T/可用 L/eligibility 均不能伪造。原生真实 lineage 也不能从 token prefix 倒推。

## D. 有效容量竞争和预算协议

精确计算针对原冻结 AgentX 的 23 snapshots×3 budgets=69 cases；不运行 baseline，也不改变任何 K。覆盖仅包含 E_p(C)>0 的 pending；候选覆盖位表示它达到该 pending 的全候选最大前沿。有限 bitmask 动态规划计算 K_min 并保存最小覆盖 witness；零前沿 pending 不凭空占一个 checkpoint。详见 capacity_certificates.json，包含每例要求的九项容量指标、前沿和原始结果行号。

| 预算 | cases | 总候选均值 | 正收益候选均值 | 有效 pending 均值 | 共享候选均值 | 单 pending 候选均值 | K 范围 | K_min 范围 | K/K_min 范围（非零分母） | K<K_min |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 25% | 23 | 24.696 | 3.652 | 1.913 | 1.304 | 2.348 | 1–37 | 0–2 | 1.000–18.500 | 0 |
| 50% | 23 | 24.696 | 3.652 | 1.913 | 1.304 | 2.348 | 3–74 | 0–2 | 2.000–37.000 | 0 |
| 75% | 23 | 24.696 | 3.652 | 1.913 | 1.304 | 2.348 | 4–111 | 0–2 | 3.000–55.500 | 0 |

K_min=2 的 27 cases、K_min=1 的 39 cases、K_min=0 的 3 cases；最后一类 K/K_min 为 null（不适用），不是 0。全部原预算都足以实现最大前沿。单 pending 候选是兼容矩阵中的 singleton 列，不自动代表已经证实原生独立 branch。

更严格的“同一个正收益共享候选下面，至少两个 pending 各有更深 singleton 候选”模式在该正式矩阵中为 0/69；不是说没有共享或分支候选，而是该特定双分支深状态竞争形态没有出现。存在两个预算合法集合可改变 E（如空集与一个正收益项）并不意味着 K 迫使有用选择取舍；本轮不生成策略胜负或人为缩小 K。真正的新 cohort 这种模式和容量受限是否出现，计数为 UNKNOWN。

**旧 candidate-based budget 的局限成立。** 无收益和重复覆盖候选进入总数，K 虽为 25% 仍可能远大于 K_min。继续使用 25%/50%/75% 可保留可比性，但不保证区分力；旧 573 cases 完全不变。新独立 cohort 可预先按实际 recurrent 池和 working/transient/headroom 字节合同确定逻辑 capacity，不能看 K_min 后设置 K=K_min−1，也不能按算法胜负挑预算。物理预算方案和预注册条件见 proposed_cohort_protocol.md。

**K<K_min 是容量竞争证据，不是 FlowState 必胜条件。** 按固定 T，Phi 严格随 G 增加；不同 E 向量有影响成本的可能，但不同分支的得失也可能抵消，不得仅凭 E 不同断言 objective 更低。

## E. 其他 epoch 与本地来源

旧 B1 census 全 393 traces 中记录 1206 个 SHARED_CONTEXT_ELIGIBLE epoch，来自同样 23 条 trace；每 trace 仅取 earliest 的旧正式 population 为 23，故还有 1183 个旧定义下的派生 epoch。这是已保存 census 的计数，不是新建 population。还有 13773 个 shared context-ineligible epoch；它们不能靠截断或外推冻结模型自动转合法。旧 131200 context 阈值也不能替代正式 Phi 的 T≤131072 定义域门禁。

额外 1183 epochs 没有本轮可读取的同 epoch FA/recurrent 驻留观测，也缺原生 pending/fork，因此其 K_min、可用候选与严格合法样本数均 UNKNOWN；没有用 completed request 数量冒充可管理 checkpoint。即使离线再次推导逻辑时序，也无法补出未记录的物理状态。本轮未重新运行结构构建器或 replay、未选择其中有利 epoch。

其他本地来源检查结果：sessions.parquet 的 15000 条 session 顶层 schema 有 session/source/messages/turn 信息，无状态驻留；TraceElephant 五个目录各检查一个确定性首轨迹，见 step_id/agent_id/messages/tool_logs，但没有这些样本的 native fork/ready/state handle；TraceLab DuckDB 只读 schema 有 round 和工具时间事件，但工具调用链接不等于模型状态继承；InferenceX Weka 首条是相同 block-shape schema。限定检查范围和原始路径均在 source_inventory.json，未检查的内部字段/其余全量资格保持 UNKNOWN，不能泛称本地所有数据都不可能有证据。

没有在已核实材料中发现能直接补齐全部门禁的替代来源。新的公开数据源仅提出字段需求，不联网、不下载，也不承诺某个命名数据集拥有 Hybrid state residency。

## F. 后续方案与唯一建议

当前不能从这些文件直接冻结一个新正式 cohort。proposed_cohort_protocol.md 是缺失证据清单与条件性采集设计，不是已注册/已存在的 cohort。要求在采集前绑定来源与版本、真实时间窗口、allocation barrier 捕获规则、原生 pending/fork 状态机、rank-local 完整 checkpoint 证据、attention frontier、物理含义预算、排除日志与统计规则。

后续若获授权，拟比较方法保持 FlowState/PCB/LDR/WFC/StandaloneTopK/LRU/LFU/Marconi Adaptation，Exact OPT 只按事先固定的可穷举范围；当前未运行。模型 C、CPU allocation 与 TTFT/control/reconciliation/E2E 必须分开。Full Coverage 明确定义为全部整数 G=0，与达到 E(C) 区分。多个 budget 为重复条件，同 root workflow 多 epoch 应按 root cluster 聚合/重采样，不得视作独立 n。

**唯一建议的下一步任务：设计真实执行框架与 Hybrid runtime 的联合事件日志及其预注册协议，并核实该框架自然会产生同时 pending 的共享/专属状态竞争；仍不启动正式采集。** 没有原生事件来源，继续扫描 hash-prefix 更多样本无法解决关键缺口。可以继续利用 AgentX 的任务形状/历史 replay 作为参考，但严格真实 cohort 必须有新的事件证据；不一定要另下载数据集。

最大阻碍首先是原生 fork/ready 状态机与历史驻留证据缺失，其次是旧可核验边界不存在 K<K_min 的容量竞争。两者不能通过调整 baseline、回筛 population 或事后降低预算补救。

## G. 完整性和停止

原始 AgentX 文件全量流式 SHA-256 已与冻结输入一致；核心读取文件及 P0/P1/P1-B 冻结证据的前后摘要见 integrity.json。只新增独立审计目录文件；没有创建正式 cohort、重建旧总体、运行 baseline、GPU、SGLang、下载数据或修改论文/PPT。本轮到此停止。

