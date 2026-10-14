"""整理只读可行性证据、缺失字段与未来采集要求，不创建正式 cohort。"""
from pathlib import Path
import json,hashlib,statistics,collections
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RAW='/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl'
B1='/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614'
B2='/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649'
def read(p):return json.loads(Path(p).read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
summary=read(OUT/'feasibility_summary.json');raw=read(OUT/'raw_schema_audit.json');metrics=read(OUT/'capacity_certificates.json');before=read(OUT/'read_hashes_before.json')
fields={
'原始数据来源':dict(status='AVAILABLE',evidence=[RAW,'raw_schema_audit.json'],detail='393 条唯一 trace，原始文件 SHA-256 与 B1 INPUTS 一致；只有 hash block 形状，不是原模型 recurrent state。'),
'Workflow ID':dict(status='PARTIAL',evidence=[RAW+' $.id','evaluation/agentx_structure_audit.py:369'],detail='原生 trace id 可追踪且 393/393 唯一；派生 conversation/workflow 分组并非全都来自原生工作流事件。'),
'Parent ID':dict(status='PARTIAL',evidence=[RAW+' 第1行 $.requests[48]','evaluation/agentx_structure_audit.py:430'],detail='嵌套 subagent 容器证明记录归属；没有 parent_request_id 或继承状态 identity。flat-chain 的 parent_conversation_id 由构建器赋值。'),
'Fork Event':dict(status='MISSING',evidence=['evaluation/agentx_chain_detection.py:298','evaluation/agentx_structure_audit.py:517'],detail='全量原始字段未出现原生 fork 事件；部分 parent/depth 来自最大 hash LCP，fork_time 取子请求 t，不能当成真实继承证明。'),
'Branch ID':dict(status='PARTIAL',evidence=[RAW+' $.requests[*].agent_id','evaluation/agentx_structure_audit.py:399'],detail='1697 个 subagent 容器有 agent_id；fa/aux/wg 等链 ID 是启发式构建，不可由命名证明真实分支。'),
'Request ID':dict(status='PARTIAL',evidence=[RAW+' $.requests[*]','evaluation/agentx_runtime_snapshot_collection.py:322'],detail='可用 trace+JSON 路径关联 invocation；原始 invocation 没有 request_id，::req 编号为派生标识。'),
'Timestamp':dict(status='PARTIAL',evidence=[RAW+' t/api_time/duration_ms','evaluation/agentx_chain_detection.py:102'],detail='98827 invocations 均有 t 和正 api_time；缺 ready/dispatch/fork/native state generation 的同钟事件。绝对时间和跨来源同步 UNKNOWN。'),
'Pending 状态':dict(status='MISSING',evidence=['evaluation/agentx_structure_audit.py:748','evaluation/agentx_structure_audit.py:831','evaluation/agentx_runtime_snapshot_collection.py:327'],detail='没有原生 ready/queued/running 状态转换；active conversation 或 API 区间重叠均不足以确认同时待续。'),
'Checkpoint Identity':dict(status='PARTIAL',evidence=[B2+'/runtime_snapshots.jsonl','evaluation/agentx_runtime_snapshot_collection.py:831'],detail='旧 replay 保存 checkpoint_id/request_index/node_id/runtime_identity_digest/recurrent_prefix_length；原始轨迹无 checkpoint，未知 epoch 无 identity。'),
'Residency Evidence':dict(status='PARTIAL',evidence=[B2+'/runtime_snapshots.jsonl',B2+'/per_snapshot/','evaluation/agentx_runtime_snapshot_collection.py:849'],detail='仅旧 23 个 replay snapshots 有 recurrent/FA 观测与 census；不是原始执行时驻留，也不能外推到其余 epoch。')}
# 其他本地来源只做有边界的 schema 检查，不能从样本缺字段推出全部缺字段。
local=[]
try:
 import pyarrow.parquet as pq
 path=Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet');before[str(path)]=sha(path);f=pq.ParquetFile(path)
 local.append(dict(path=str(path),scope='完整顶层 parquet schema',rows=f.metadata.num_rows,fields=f.schema.names,status='PARTIAL',missing='顶层无调度/驻留事件；messages_json 内是否有足够父子证据 UNKNOWN，不能把 turn 当同时 pending。'))
except ImportError:local.append(dict(path='/home/wjg/data/agentic_coding_trajectories/sessions.parquet',status='UNKNOWN',reason='未读取 schema'))
base=Path('/home/wjg/data/traceelephant/data')
for group in sorted(base.iterdir()):
 fs=sorted(group.rglob('step_records.json'))
 if fs:
  p=fs[0];before[str(p)]=sha(p);r=read(p);meta=p.with_name('trace_metadata.json');before[str(meta)]=sha(meta)
  local.append(dict(path=str(p),scope='该目录字典序首个轨迹，不能代表全库',steps=len(r),fields=sorted({k for x in r for k in x}),metadata_fields=list(read(meta)),status='PARTIAL',missing='多 agent/工具步骤可见；该样本没有系统级 fork、ready 或 Hybrid state handle；其余未读内容 UNKNOWN。'))
try:
 import duckdb
 p=Path('/home/wjg/data/tracelab/v0.0.2/syfi_coding_trace.duckdb');before[str(p)]=sha(p);con=duckdb.connect(str(p),read_only=True)
 schema={t:[x[0] for x in con.execute('describe '+t).fetchall()] for t, in con.execute('show tables').fetchall()};con.close()
 local.append(dict(path=str(p),scope='只读表 schema',schema=schema,status='PARTIAL',missing='timing_events/tool_calls 有时间和工具链接，但这不证明 fork 继承或 checkpoint residency；字段内容覆盖 UNKNOWN。'))
except ImportError:local.append(dict(path='/home/wjg/data/tracelab/v0.0.2/syfi_coding_trace.duckdb',status='UNKNOWN'))
p=Path('/home/wjg/data/inferencex_weka/traces_first30.jsonl');before[str(p)]=sha(p)
with p.open() as f:first=json.loads(next(f))
local.append(dict(path=str(p),scope='仅首条 schema',fields=list(first),request_fields=list(first['requests'][0]),status='PARTIAL',missing='同类 block-shape 请求记录；未证明 native fork、pending、residency，全库符合门禁数 UNKNOWN。'))
inventory=dict(primary_source=dict(path=RAW,provider='semianalysisai/cc-traces-weka-062126（本地 B1 INPUTS 与原始文件）',sha256=before[RAW],version_scope='以此内容摘要固定，未联网核验远端版本',time_scope='每条 trace 的相对 t；绝对采集时间窗口 UNKNOWN',fields=fields),historical_census_root=B1,historical_runtime_root=B2,other_local_sources=local,public_source_search='本轮未联网、未下载；只提出字段需求，未承诺具体公开数据集满足条件。')
write('source_inventory.json',inventory)
summary['status']='NOT_FEASIBLE_FROM_EXISTING_DATA'
summary['gate_results']={
'A':dict(status='NOT_ESTABLISHED',reason='原生 fork/父执行前缀继承事件缺失；LCP/嵌套不足'),
'B':dict(status='NOT_ESTABLISHED',reason='原生 ready/pending 缺失；conversation active 不是等待执行'),
'C':dict(status='PARTIAL',reason='旧 replay 的 A/R/T/L 可核验，新 epoch R 与 native pending anchor UNKNOWN'),
'D':dict(status='PARTIAL',reason='旧 replay 568 个 checkpoint 有身份/驻留关联；其他原始 epoch 未观测物理状态')}
summary['capacity']['conditional_scope']='仅冻结正式 compatibility 下的结构事实，不升级为通过本轮真实 provenance 门禁。'
summary['capacity']['strict_real_shared_tradeoff_count']='UNKNOWN'
summary['recoverability']=dict(recoverable='原始 trace 和 JSON 路径、显式 subagent 归属、请求相对时间、旧 replay 的身份/前沿、旧定义下的其他 epoch 线索',not_recoverable_from_recorded_fields='原生 fork 继承证明、ready/dispatch 状态机、原执行 Hybrid recurrent/attention 驻留；除非另有同时期日志，不能事后补造',recommendation='唯一下一步：形成真实执行框架与 Hybrid runtime 联合事件日志的采集设计及预注册；不启动采集。')
write('feasibility_summary.json',summary)
protocol='''# 新 cohort 的缺失证据与未来采集要求

**状态：条件性方案，尚未注册；未创建新 cohort。** 当前文件不足以直接构建满足严格门禁的新 cohort。下列是后续单独设计任务必须冻结的要求，不是本轮采集授权。

## 1. 来源、版本和范围

不得覆盖 OpenHands 168 / AgentX 23 的正式总体。AgentX 原始文件摘要为 29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276；它可用于任务/时序形状参考，不能补出原生 fork 或驻留。若使用该来源再执行，必须明确为新的受控执行，不声称恢复了原始时刻的状态。

真实执行框架版本、任务来源与许可、容器/模型/tokenizer/runtime commit、记录格式版本、全量入组时间窗口 [start,end)、capture 频率和停止样本数均必须在第一条数据生成前写入注册文件并签名/哈希。当前这些来源绑定参数为 UNKNOWN/待定，不填写虚构时间或 cohort 数量。没有确认自然出现所需结构前，不承诺样本规模或效应大小。

## 2. 联合事件记录

必须记录稳定 root_workflow_id、branch_id、request_id、parent_request_id、原生 fork_event_id、继承前缀长度与 token/model/tokenizer identity、state generation。Fork 必须由框架真实事件和父执行证据确认；LCP 只作一致性校验，不能替代父子继承。不同任务的相同 prompt 不合并为同一 recurrent state。

请求状态机至少包括 ready、queued、dispatch、running、complete/cancel；每个事件有单调序号和同钟 timestamp。记录实际 barrier 起止和在该边界阻止状态变化的机制。pending 定义为边界之前已 ready、尚未 dispatch、未 complete/cancel 的真实 continuation；至少两个，不能用未来请求或已执行请求的上下文凑数。不能仅以 conversation start/end 间隔判定。

原生 epoch 捕获规则必须在观察策略结果前固定，例如每个真实调度分配 barrier 记录一次，再按预先固定的 root-workflow/epoch 序号抽样。不得延迟或重排执行以制造有利需求。所有捕获和排除都保留日志。

## 3. 可执行语义和 eligibility

在同一边界观测 A_p、attention-side R_p、T_p=min(A_p,R_p)，不把历史 token 长度当 R。每个候选记录创建与有效时点、L、checkpoint ID 到 rank-local state handle/slot generation 的映射、所有必需 recurrent/conv 组件、resident/valid/protection 状态。状态 observation 必须非干扰；epoch 内不得发生未受控 request execution、native eviction 或 slot reuse。

候选必须已物化且合法可管理；不同 logical ID 不凭 slot 整数去重。共享推荐的同一 canonical checkpoint 只计一个逻辑单位。每条兼容边同时验证原生 ancestry、模型/缓存兼容和 L≤T；超过 T 的真实 checkpoint 可以留在候选总体但不对该 p 产生可用边，不为了缩小总体删除不利候选。冻结 Phi 定义域为 T≤131072；旧 context census 的 131200 阈值不能自动作为模型门禁。

缺少父事件、pending 状态、状态 identity、rank 证据、同 epoch 一致性或输入越域者，写入排除日志：source ID、epoch、gate、reason、字段缺失、证据哈希。UNKNOWN 不能填零或转 PASS。不能按任何策略胜负排除。

## 4. 预算协议与容量证书

A 协议：如保留 25%/50%/75% candidate-based K，必须全部预先规定，并如实报告无收益/重复候选、有效 pending、K_min_frontier 和宽松程度。不能变更旧 573 cases。该协议利于与旧数据描述性对照，但不保证产生有效容量竞争。

B 协议：新 cohort 可使用实际系统容量预算。先按部署资源合同固定 recurrent 管理池字节 B_r、工作/active state 预留 W_r、transient/COW/headroom H_r，以及 checkpoint 完整组件每 rank 字节 b_r；逻辑上限为 min_r floor((B_r-W_r-H_r)/b_r)。任何额外限额须有明确系统理由。attention/KV 预算保持独立，不挪用容量证明。数值需根据硬件/服务约束在看 workload 的 K_min 或策略结果之前冻结，不能设 K=K_min−1 来强行制造困难，也不能由胜负反向选 K。等大小或 TP 假设不满足时，应重新做独立设计，不临时改正式算法。

本轮没有给新 cohort 选一个具体 K，因为缺少已核准的物理资源合同。K 和 capture 规则冻结后，按事先规定的 shared/unshared、binding/nonbinding 结构层报告全体及子层。结构性富集只能预注册并完整报告原始分母，不能只保留正收益例。

每个 epoch 计算 E_p(C)。令 Q 为 E_p(C)>0 的 p；每个 c 的覆盖集合是其能精确达到这些最深前沿的 p。对该覆盖问题精确求 K_min_frontier，并保存所选 ID 证书或其他可验证最优证明。空 Q 时 K_min=0，K/K_min 不适用；计算不可完成则 UNKNOWN。K<K_min 仅证明有效容量限制，不证明 FlowState 胜出。明确记录共享祖先与至少两个分支更深专属项是否同时存在、收益覆盖是否重叠。

## 5. 后续拟比较的方法与指标

仅在数据独立冻结、门禁通过且另获执行授权后，使用不变 FlowState、PCB、LDR、WFC、StandaloneTopK、LRU、LFU、Marconi Adaptation；不把 adaptation 称作完整外部系统。Exact OPT 仅对预先固定的可穷举阈值内实例，完整公布分母和未求解实例，不因难以穷举而删总体。

共同评分：modeled C(S)、逐 p E/G、selected sets、容量证书、配对胜平负、greedy–Exact gap。Full Coverage 明确定义为所有当前 pending 的整数 G_p=0；达到全候选前沿另外报告，不能混为 Full Coverage。成本 tie 使用冻结 1e-9 modeled ms，不能把非零整数 gap 以成本容差抹掉。

模型成本与 CPU allocation 结果分开。实际 TTFT、control/reconciliation 和 control-inclusive E2E 需后续独立 runtime protocol，本阶段不测，也不把旧结果拼接成新结果。

## 6. 统计与停止规则

多个 budgets 是同 snapshot 的重复条件；同 root workflow 的多个 epoch 有相关性。主配对与重采样以独立 root workflow 为 cluster，预先规定 epoch 权重/聚合，另报每 snapshot 结果。不能把所有请求、epoch、预算摊成独立 n。允许持平、负例和模型选择器非最优结果，完整公布。

未找到自然产生且可审计的紧约束共享结构时，保留可行性失败结论，不增加合成 fork、不放宽 pending/eligibility、不按结果调预算。本文件到未来采集设计为止；本轮未采集、未创建 cohort、未跑策略、未运行 GPU。
'''
(OUT/'proposed_cohort_protocol.md').write_text(protocol)
report=['# P2 真实共享工作负载 cohort 可行性审计','', '**NOT_FEASIBLE_FROM_EXISTING_DATA**','', '日期：2026-10-09（Asia/Shanghai）。当前已检查资料不能直接形成同时满足原生 lineage、真实同时 pending、checkpoint eligibility 与有效容量竞争的新 cohort。AgentX 可以继续作为请求形状和派生结构研究来源；不能仅靠重新解析现有字段补齐缺失的原生事件与历史驻留。需要新的联合事件证据来源，不等于必须下载另一个公开数据集。','', '## A. 原始来源与字段 provenance','',f'原始文件：`{RAW}`。SHA-256：`{before[RAW]}`，与冻结 B1 INPUTS 一致。全量流式检查 393 条 traces、393 个唯一 trace IDs、98827 个 invocation、1697 个 subagent 容器。原始模型记录示例为 Claude；旧 Qwen/SGLang 的状态是之后重放形状生成的，不能当成原模型历史状态。','', '| 项目 | 状态 | 证据与限制 |','|---|---|---|']
for n,v in fields.items():report.append(f'| {n} | {v["status"]} | {v["detail"]} 证据：'+ '; '.join('`'+x+'`' for x in v['evidence'])+' |')
report+=['', '全量 schema 的字段与计数见 raw_schema_audit.json；每个 trace 的源行号见 raw_trace_inventory.json。MISSING 表示所检查原始 schema 中没有该类记录，不是声称所有外部系统都没有日志；后者 UNKNOWN。1697 个 subagent 容器的 status 是完成摘要，不能替代 epoch 时点的调度状态。','', '## B. Fork / Parent / Pending 合法性','', '**Gate A 未建立。** `agentx_chain_detection.py:298` 调用 `_max_lcp_chain` 后建立 ChainFork，`:308` 将子请求 t 作为 fork_time；`agentx_structure_audit.py:415` 用 depth>0 区分 fork/spawn，`:517` 对 subagent 的首 hash 与 root tail 做 LCP。嵌套 subagent 确实提供结构归属，但没有父 invocation 与继承前缀的原生事件证明。旧“logical inherited ancestry”本身含推导环节，不能因为进入正式 manifest 就升级成原始观测。','', '**Gate B 未建立。** `agentx_structure_audit.py:748` 在 conversation start 时加入 active_targets，在请求完成事件更新，`:831` 在记录 snapshot 后才移除 end 事件；因而代码还会在 end 同时刻包含刚结束的 conversation，不能直接把其注释中的半开区间当成实现证明。`agentx_runtime_snapshot_collection.py:327` 明确沿用 current-context，而不是未来下一条请求。这个旧实验的 pending 定义不同于本轮要求的“真实已 ready、尚未执行”。','', '原始全部 invocation 都有 t 和正 api_time。按照记录坐标计算 API 区间重叠，373/393 traces 有至少两个重叠调用，最大重叠 23；这只能说明记录的 API 区间，不能证明同一 allocation boundary 上有两个等待恢复的 continuation，不能用它替代 ready/queued 事件。绝对时钟和跨分支时钟同步仍需额外证明。','', '这些发现不会修改已冻结数据或 P1-B 的 CPU 等价结论；它们限制的是新 cohort 的真实来源和并发 claim。原仓库根目录 B1 报告甚至含 pytest 临时 canonical root，因此本轮以实际使用的 B1 INPUTS、正式目录及源码为准，未盲信顶层 READY 文案。','', '## C. 可执行状态与 eligibility','', '**Gate C/D 部分成立且只限旧 replay 边界。** 旧 `runtime_snapshots.jsonl` 有 23 snapshots、568 个 checkpoint 的 checkpoint_id、request_index、node_id、runtime_identity_digest、recurrent_prefix_length、recurrent_resident 和 fa_resident；pending 有 input_token_count、resident_fa_frontier、planning_target、fa_query_side_effect_free。本轮逐项核对正式 L 等于 recurrent_prefix_length，T=min(input_token_count,R)，所有正式候选与旧 runtime IDs 一致。','', 'collector 在 `agentx_runtime_snapshot_collection.py:710` 按顺序执行已完成请求，然后解析 checkpoint，并在 `:849` 观测 pending FA frontier。`REPLAY_SEMANTICS.json` 明确为 BLOCK_SHAPE 和确定性合成 token。原始 hash/history 不能证明 Qwen recurrent state 当时存在；旧 replay 确实观测到的 node/handle 也只能证明该次 replay 的边界，不证明原始生产时刻或其他 epoch。旧 raw runtime compatibility 与正式逻辑 compatibility 口径不同，本轮容量计算仅用正式 manifest，未混用。','', '原始数据缺 R_p、state generation/owner 和 checkpoint materialization 事件；无法由 token history 唯一恢复。对于未重放的 epoch，A 可从输入长度形成逻辑量，但是否是当时 ready continuation 的 anchor 未证明；R/T/可用 L/eligibility 均不能伪造。原生真实 lineage 也不能从 token prefix 倒推。','', '## D. 有效容量竞争和预算协议','', '精确计算针对原冻结 AgentX 的 23 snapshots×3 budgets=69 cases；不运行 baseline，也不改变任何 K。覆盖仅包含 E_p(C)>0 的 pending；候选覆盖位表示它达到该 pending 的全候选最大前沿。有限 bitmask 动态规划计算 K_min 并保存最小覆盖 witness；零前沿 pending 不凭空占一个 checkpoint。详见 capacity_certificates.json，包含每例要求的九项容量指标、前沿和原始结果行号。','', '| 预算 | cases | 总候选均值 | 正收益候选均值 | 有效 pending 均值 | 共享候选均值 | 单 pending 候选均值 | K 范围 | K_min 范围 | K/K_min 范围（非零分母） | K<K_min |','|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|']
for ratio in (.25,.5,.75):
 rs=[r for r in metrics if r['budget_ratio']==ratio];rat=[r['K_over_K_min_frontier'] for r in rs if r['K_over_K_min_frontier'] is not None]
 avg=lambda k:statistics.mean(r[k] for r in rs)
 report.append(f'| {ratio:.0%} | {len(rs)} | {avg("total_candidates"):.3f} | {avg("positive_benefit_candidates"):.3f} | {avg("effective_pending"):.3f} | {avg("shared_benefit_candidates"):.3f} | {avg("branch_specific_candidates"):.3f} | {min(r["K"] for r in rs)}–{max(r["K"] for r in rs)} | {min(r["K_min_frontier"] for r in rs)}–{max(r["K_min_frontier"] for r in rs)} | {min(rat):.3f}–{max(rat):.3f} | 0 |')
report+=['', 'K_min=2 的 27 cases、K_min=1 的 39 cases、K_min=0 的 3 cases；最后一类 K/K_min 为 null（不适用），不是 0。全部原预算都足以实现最大前沿。单 pending 候选是兼容矩阵中的 singleton 列，不自动代表已经证实原生独立 branch。','', '更严格的“同一个正收益共享候选下面，至少两个 pending 各有更深 singleton 候选”模式在该正式矩阵中为 0/69；不是说没有共享或分支候选，而是该特定双分支深状态竞争形态没有出现。存在两个预算合法集合可改变 E（如空集与一个正收益项）并不意味着 K 迫使有用选择取舍；本轮不生成策略胜负或人为缩小 K。真正的新 cohort 这种模式和容量受限是否出现，计数为 UNKNOWN。','', '**旧 candidate-based budget 的局限成立。** 无收益和重复覆盖候选进入总数，K 虽为 25% 仍可能远大于 K_min。继续使用 25%/50%/75% 可保留可比性，但不保证区分力；旧 573 cases 完全不变。新独立 cohort 可预先按实际 recurrent 池和 working/transient/headroom 字节合同确定逻辑 capacity，不能看 K_min 后设置 K=K_min−1，也不能按算法胜负挑预算。物理预算方案和预注册条件见 proposed_cohort_protocol.md。','', '**K<K_min 是容量竞争证据，不是 FlowState 必胜条件。** 按固定 T，Phi 严格随 G 增加；不同 E 向量有影响成本的可能，但不同分支的得失也可能抵消，不得仅凭 E 不同断言 objective 更低。','', '## E. 其他 epoch 与本地来源','', '旧 B1 census 全 393 traces 中记录 1206 个 SHARED_CONTEXT_ELIGIBLE epoch，来自同样 23 条 trace；每 trace 仅取 earliest 的旧正式 population 为 23，故还有 1183 个旧定义下的派生 epoch。这是已保存 census 的计数，不是新建 population。还有 13773 个 shared context-ineligible epoch；它们不能靠截断或外推冻结模型自动转合法。旧 131200 context 阈值也不能替代正式 Phi 的 T≤131072 定义域门禁。','', '额外 1183 epochs 没有本轮可读取的同 epoch FA/recurrent 驻留观测，也缺原生 pending/fork，因此其 K_min、可用候选与严格合法样本数均 UNKNOWN；没有用 completed request 数量冒充可管理 checkpoint。即使离线再次推导逻辑时序，也无法补出未记录的物理状态。本轮未重新运行结构构建器或 replay、未选择其中有利 epoch。','', '其他本地来源检查结果：sessions.parquet 的 15000 条 session 顶层 schema 有 session/source/messages/turn 信息，无状态驻留；TraceElephant 五个目录各检查一个确定性首轨迹，见 step_id/agent_id/messages/tool_logs，但没有这些样本的 native fork/ready/state handle；TraceLab DuckDB 只读 schema 有 round 和工具时间事件，但工具调用链接不等于模型状态继承；InferenceX Weka 首条是相同 block-shape schema。限定检查范围和原始路径均在 source_inventory.json，未检查的内部字段/其余全量资格保持 UNKNOWN，不能泛称本地所有数据都不可能有证据。','', '没有在已核实材料中发现能直接补齐全部门禁的替代来源。新的公开数据源仅提出字段需求，不联网、不下载，也不承诺某个命名数据集拥有 Hybrid state residency。','', '## F. 后续方案与唯一建议','', '当前不能从这些文件直接冻结一个新正式 cohort。proposed_cohort_protocol.md 是缺失证据清单与条件性采集设计，不是已注册/已存在的 cohort。要求在采集前绑定来源与版本、真实时间窗口、allocation barrier 捕获规则、原生 pending/fork 状态机、rank-local 完整 checkpoint 证据、attention frontier、物理含义预算、排除日志与统计规则。','', '后续若获授权，拟比较方法保持 FlowState/PCB/LDR/WFC/StandaloneTopK/LRU/LFU/Marconi Adaptation，Exact OPT 只按事先固定的可穷举范围；当前未运行。模型 C、CPU allocation 与 TTFT/control/reconciliation/E2E 必须分开。Full Coverage 明确定义为全部整数 G=0，与达到 E(C) 区分。多个 budget 为重复条件，同 root workflow 多 epoch 应按 root cluster 聚合/重采样，不得视作独立 n。','', '**唯一建议的下一步任务：设计真实执行框架与 Hybrid runtime 的联合事件日志及其预注册协议，并核实该框架自然会产生同时 pending 的共享/专属状态竞争；仍不启动正式采集。** 没有原生事件来源，继续扫描 hash-prefix 更多样本无法解决关键缺口。可以继续利用 AgentX 的任务形状/历史 replay 作为参考，但严格真实 cohort 必须有新的事件证据；不一定要另下载数据集。','', '最大阻碍首先是原生 fork/ready 状态机与历史驻留证据缺失，其次是旧可核验边界不存在 K<K_min 的容量竞争。两者不能通过调整 baseline、回筛 population 或事后降低预算补救。','', '## G. 完整性和停止','', '原始 AgentX 文件全量流式 SHA-256 已与冻结输入一致；核心读取文件及 P0/P1/P1-B 冻结证据的前后摘要见 integrity.json。只新增独立审计目录文件；没有创建正式 cohort、重建旧总体、运行 baseline、GPU、SGLang、下载数据或修改论文/PPT。本轮到此停止。','']
(OUT/'final_report.md').write_text('\n'.join(report)+'\n')
# 核验先前冻结审计源，并把所读取的本地补充来源纳入本次摘要范围。
p1=ROOT/'evaluation/p1_strong_baseline_validation';frozen=read(p1/'source_integrity_before.json');before.update(frozen)
man=read(p1/'artifact_manifest.json')
for name,value in man['files'].items():
 p=p1/name
 if sha(p)!=value:raise ValueError('P1-B 产物摘要不一致：'+str(p))
 before[str(p)]=value
before[str(p1/'artifact_manifest.json')]=sha(p1/'artifact_manifest.json')
after={p:sha(p) for p in before}
changed=[p for p in before if before[p]!=after[p]]
assert not changed
write('integrity.json',dict(status='PASS',files_checked=len(before),before=before,after=after,changed_files=changed,scope='审查读取的核心来源、已冻结 P0/P1/P1-B 文件；不声称扫描所有本地数据或证明外部状态没有变化。',gpu_started=False,baseline_executed=False,new_cohort_created=False,raw_agentx_sha256=after[RAW]))
print(summary['status']);print('哈希通过文件数',len(before));print('最终报告已生成。')
