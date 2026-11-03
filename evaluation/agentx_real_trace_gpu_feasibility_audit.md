# P1-A：AgentX 非平凡分配实例可复核，但不能忠实迁移为现有 Strict GPU 对照

审查日期：2026-10-28。**最终判定：BLOCKED。** 本轮只读原始来源、冻结CPU结果、旧Runtime采集产物和源码；仅在内存中复核目标值与预算证书，未新增代码文件，未启动GPU、Engine或Replay，未接触P0-4执行路径。

六个WFC非最优条件确实存在，来自六个不同原始trace所派生的六个快照，各仅有K=1这一条件；并非同一快照六次重复预算。它们是**真实轨迹衍生的Formal分配实例**，不是六个已经严格认证的可执行共享实例。全部六例均缺少原始Prompt/Runtime Token历史；五例选集涉及Raw前缀谓词为假的Formal边，另一例g006则存在改变恢复含义的Raw额外合法前缀。四例还具有逻辑候选映射同一物理Node的记录。因此本轮推荐的合格GPU回放实例数为**0**。

## 1. 来源、快照与真实性

### 1.1 原始数据有真实调用记录，但没有完整执行输入

本地来源为`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl`。本轮重新流式核对整个文件：393条trace，1,847,151,435字节，SHA-256=`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`，与既有身份归档一致。公开来源及官方元数据证据保存在`agentic_workflow_dataset_research/phase2_submission_inputs/weka_agentx_identity_audit.md`、`weka_identity.json`和`official/weka_tree.json`；本轮未联网更新来源版本。

六条原始记录均为`block_size=64`、`hash_id_scope=local`。根字段为id、models、block_size、hash_id_scope、requests；调用字段包括t、model、in/out长度、hash_ids、api_time、ttft、think_time、type；部分trace还含真实subagent容器及其子调用。六条记录及嵌套调用中没有Prompt文本、完整input_token_ids、目标模型Chat Template或原始Recurrent状态。原模型标签包含Claude系列，不能把Qwen重放状态当作原模型状态。

|快照简称|原始trace ID（亦为下文root候选ID前缀）|原始JSONL行|调用数（含嵌套普通调用）|原始模型标签|
|---|---|---:|---:|---|
|g005|61962299108912449a53f411ed3cf2d1c013|154|144|claude-fable-5|
|g006|63d3960fcc384388d5e7e4df8287cd1be081|161|135|claude-opus-4-8|
|g010|77cfef0760bda37fb6f8829cebd9a7871aa5|190|70|claude-opus-4-6|
|g017|cbbac1ac4e0a88cb6ee971ef166fe6a24222|323|130|claude-haiku-4-5-20251001、claude-opus-4-8|
|g018|d77fa1e0115f7fca67ca88b95a053d3a41bc|338|89|claude-fable-5、claude-opus-4-8|
|g021|ecd3ea07309dd8a6b62b2f90fd8c60fa92b2|367|287|claude-fable-5、claude-haiku-4-5-20251001、claude-opus-4-8|

hash_ids是块身份/形状线索，不能逆推出已丢失的原始Token内容。六条记录的原始模型身份也不等于Qwen执行配置身份。既有受控合成器可生成新的合法整数Token串，但不能补回原始历史。

### 1.2 实际处理链

1. `evaluation/agentx_structure_audit.py::_detect_and_build_conversations`（333起）调用`agentx_chain_detection.py::detect_agent_chains`（317起），根据hash LCP、时间及seam规则分解root/flat chains/subagents；`is_inherited`和部分FORK关系是**算法派生**，不是全部来自原生继承事件。逻辑位置使用请求hash块长度的累计和。
2. `_trace_event_timeline`/`_analyze_trace_online_compatibility`（682/702起）以conversation开始、请求完成、结束生成epoch；active conversation被称为Pending，active target是已完成请求的累计位置。`agentx_runtime_population_census.py::_select_earliest_eligible_per_trace`（438起）固定每trace首个合格共享epoch，得到23个边界。
3. `agentx_runtime_snapshot_collection.py::_build_replay_plan`（275起）取t前已完成请求，按完成时间重放；每个请求独立调用`agentx_qwen_replay_preflight.py::_compose_prompt_tokens`（194起），用hash映射64-token块，按in长度裁剪并确定性补尾。**这属于trace形状驱动的受控Token重放**。
4. `_collect_snapshot`（654起）曾真实执行历史输入、查询Node/FA/Mamba驻留和Pending FA前沿；Checkpoint的实际位置L为`recurrent_prefix_length`，不是累计逻辑token_pos。Raw边在884–895附近用`_prefix_match(历史实际前缀, Pending合成输入)`产生。
5. `agentx_runtime_final_correctness_audit.py::logical_compatible_pending_ids`（87起）使用严格逻辑祖先、inherited、位置条件；`corrected_runtime_snapshots.jsonl`替换兼容列表，未产生新GPU状态。
6. `rq3_agentx_formal_policy_evaluation.py::build_agentx_formal_snapshots`（327起）组合B2物理L/T与B1逻辑边，`encode_usable_compatibility`（172起）以人工“兼容簇”lineage编码其laminar集合。模块开头明确：Raw Token兼容不进入目标/选择器。随后旧Formal评估→Gate1矩阵→固定K=1–8扫描。

因此候选并非全是凭空命名：旧B2有真实Qwen合成输入的驻留观测。但**真实物化的源状态 + 人工逻辑正边，并不构成目标请求可执行恢复的证明**。例如g006/root::req0011逻辑token_pos=1,007,936，物理L=103,104；二者绝不能混为同一Token坐标。

### 1.3 时间边界与未来信息

- 候选选择显式要求`request_end_seconds≤t`，历史重放也按该集合排序；这支持“未回放t之后的候选请求”。
- Pending并非自然Agent Ready队列：`_request_index_for_target`（228起）在target>0时指向**最后已完成请求**，target=0时指向首个已开始/在途请求；它不是经过认证的下一次未执行Continuation输入。`_active_request_context_length`也使用最后已完成请求的context。
- 链检测先读取完整trace，再在`_resolve_seams`中做事后链拼接/归属修订，之后才计算epoch。故依赖图与conversation结束边界不具备已归档的逐epoch可见性证明；六例是否具体因某个未来事件改变关系为**UNKNOWN**。不能将其描述为原生在线可用ASG。
- Collector的`future_leakage`是默认False字段；源码只有定义/序列化/汇总，缺少覆盖全部因果可见性的逐项检查。旧报告的0不能替代Pending已准备、依赖当时已知的证据。
- 本轮未发现需要读取未来Assistant文本的构造代码，原始文本本来也不存在；这不补足未来依赖、模型状态或下一请求身份的缺口。

## 2. 六个差异条件：冻结结果与独立CPU复核

来源为`evaluation/paper_workload_structure_final/agentx_budget_scan.csv`：736行数据=23快照×8预算×4策略，共184预算条件；不是184个工作流。本轮读取原矩阵与冻结选集重新评价全部736行，最大J差为4.912×10⁻¹⁰ ms（CSV九位小数舍入），在1e−9 ms容差内；未调用生产Optimizer或重新输出策略实验。

K_min用逐Pending最深前沿的覆盖位掩码DP独立复核，23快照均与冻结值一致。K<K_min的Exact条件在本数据仅K=1，逐一检查全部单候选；K≥K_min则验证冻结Exact集合达到全候选前沿下界。本轮没有对149候选启动无界枚举。

全184条件：Greedy最优184、WFC最优178、StandaloneTopK最优157（27次非最优，K=2–8）。下表六例K均为1、K_min均为2。Checkpoint完整ID为上表trace ID加`::reqNNNN`；Greedy与Exact每例选集相同，单候选集合以花括号表示。

|完整Snapshot ID|n/m|K/K_min|WFC集合|Greedy集合|Exact集合|J_WFC ms|J_Greedy ms|J_Exact ms|WFC−Exact ms|扫描CSV行|
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
|rq3-agentx-g005-epoch8|8/2|1/2|{req0000}|{req0001}|{req0001}|2573.277763859|620.026890944|620.026890944|1953.250872915|164|
|rq3-agentx-g006-epoch35|35/2|1/2|{req0001}|{req0011}|{req0011}|10305.793263812|9166.409583559|9166.409583559|1139.383680254|196|
|rq3-agentx-g010-epoch38|38/2|1/2|{req0002}|{req0019}|{req0019}|6044.829729204|5611.522035867|5611.522035867|433.307693337|324|
|rq3-agentx-g017-epoch12|12/3|1/2|{req0000}|{req0001}|{req0001}|6501.555667633|4655.433414738|4655.433414738|1846.122252896|548|
|rq3-agentx-g018-epoch30|30/2|1/2|{req0001}|{req0012}|{req0012}|10436.816564129|9021.082426262|9021.082426262|1415.734137867|580|
|rq3-agentx-g021-epoch149|149/4|1/2|{req0000}|{req0008}|{req0008}|12519.080584393|7254.039549471|7254.039549471|5265.041034922|676|

### 2.1 Pending坐标、共享域与恢复取舍

P0、P1等仅表示各矩阵`pending_ids`原始顺序，下表给出完整可还原的后缀；同样在前面加对应trace ID。E/G向量均为**Formal模型值**，不是Native Match观测。

**rq3-agentx-g005-epoch8**

Pending顺序：P0=`::fa:000::pending0004`，T=54527；P1=`::sa:subagent_001_97adff5d::pending0000`，T=1344。

|选择|L|正收益支持|E向量|G向量|独立收益 ms|
|---|---:|---|---|---|---:|
|WFC：req0000|384|[0, 1]|[384, 384]|[54143, 960]|29.101911990|
|Exact：req0001|43840|[0]|[43840, 0]|[10687, 1344]|1982.352784905|

全部候选支持集计数：[]：6；[0, 1]：1；[0]：1。

**rq3-agentx-g006-epoch35**

Pending顺序：P0=`::aux:003::pending0000`，T=125248；P1=`::fa:002::pending0008`，T=125759。

|选择|L|正收益支持|E向量|G向量|独立收益 ms|
|---|---:|---|---|---|---:|
|WFC：req0001|48448|[0, 1]|[48448, 48448]|[76800, 77311]|4668.125454620|
|Exact：req0011|103104|[1]|[0, 103104]|[125248, 22655]|5807.509134874|

全部候选支持集计数：[]：23；[0, 1]：2；[1]：10。

**rq3-agentx-g010-epoch38**

Pending顺序：P0=`::fa:000::pending0014`，T=94079；P1=`::fa:001::pending0000`，T=94079。

|选择|L|正收益支持|E向量|G向量|独立收益 ms|
|---|---:|---|---|---|---:|
|WFC：req0002|44160|[0, 1]|[44160, 44160]|[49919, 49919]|4109.691493535|
|Exact：req0019|86080|[0]|[86080, 0]|[7999, 94079]|4542.999186872|

全部候选支持集计数：[]：18；[0, 1]：3；[0]：17。

**rq3-agentx-g017-epoch12**

Pending顺序：P0=`::pending0007`，T=78463；P1=`::sa:subagent_001_d0583ff6::pending0002`，T=42303；P2=`::sa:subagent_002_8573efac::pending0000`，T=15744。

|选择|L|正收益支持|E向量|G向量|独立收益 ms|
|---|---:|---|---|---|---:|
|WFC：req0000|320|[1, 2]|[0, 320, 320]|[78463, 41983, 15424]|24.267784471|
|Exact：req0001|41984|[1]|[0, 41984, 0]|[78463, 319, 15744]|1870.390037366|

全部候选支持集计数：[1, 2]：1；[1]：1；[]：10。

**rq3-agentx-g018-epoch30**

Pending顺序：P0=`::aux:002::pending0000`，T=126847；P1=`::fa:001::pending0009`，T=124287。

|选择|L|正收益支持|E向量|G向量|独立收益 ms|
|---|---:|---|---|---|---:|
|WFC：req0001|47360|[0, 1]|[47360, 47360]|[79487, 76927]|4548.133889245|
|Exact：req0012|105344|[1]|[0, 105344]|[126847, 18943]|5963.868027112|

全部候选支持集计数：[]：17；[0, 1]：2；[1]：11。

**rq3-agentx-g021-epoch149**

Pending顺序：P0=`::fa:000::pending0010`，T=106367；P1=`::sa:subagent_004_4e65601c::pending0031`，T=111423；P2=`::sa:subagent_004_4e65601c:aux:001::pending0000`，T=1727；P3=`::sa:subagent_004_4e65601c:fa:000::pending0003`，T=3839。

|选择|L|正收益支持|E向量|G向量|独立收益 ms|
|---|---:|---|---|---|---:|
|WFC：req0000|384|[0, 2]|[384, 0, 384, 0]|[105983, 111423, 1343, 3839]|29.743931855|
|Exact：req0008|96704|[0]|[96704, 0, 0, 0]|[9663, 111423, 1727, 3839]|5294.784966778|

全部候选支持集计数：[]：140；[0, 2]：1；[0]：8。

六例的直接机制均是WFC按“可推进Pending数量q优先、总Token推进量次之”选择支持两个Pending的状态，而Greedy/Exact选择只支持一个Pending但收益更高的深层状态。g005/g017/g021尤其明显：WFC只推进384/320/384 Tokens，仍因覆盖2个Pending优先。

这是给定Formal矩阵中“覆盖数与恢复收益排序冲突”的实例证据。K=1时Marginal Greedy就是独立总收益最大化，StandaloneTopK在六例也都选Exact同一候选；**这六例不能单独证明动态边际更新优于StandaloneTopK，也不能隔离Position-aware项的独立贡献**。所有支持域均嵌套/单例，不需要非层次交叉关系。

## 3. 同一实例的Raw证据为何不能支持原Formal GPU含义

读取旧`runtime_snapshots.jsonl`和`compatibility_diff.csv`，按完整trace/candidate/pending ID精确联接，没有重建Token、改边或重算Raw策略胜负。

|快照|所选WFC边：Formal真且Raw真/全部|所选Exact边：Formal真且Raw真/全部|逻辑候选数/不同Node数|原选集的GPU复现障碍|
|---|---|---|---|---|
|g005|0/2|0/1|8/8|两种选择全部目标边的Raw谓词为假|
|g006|2/2|1/1|35/34|深层req0011的Raw支持两个Pending，Formal只计P1；39条Raw-only关系，另有Node别名|
|g010|0/2|0/1|38/36|两种选择全部目标边的Raw谓词为假；Node别名|
|g017|0/2|0/1|12/11|WFC共享边及Exact到P1的边均为假；Node别名|
|g018|2/2|0/1|30/28|WFC共享成立，但Exact深层目标边为假；Node别名|
|g021|0/2|1/1|149/149|Exact到P0成立，WFC共享边均为假；149候选超出近期28槽配置|

Raw检查指“旧collector对其当次合成输入计算的完整前缀谓词”，不是六例逐请求真实恢复成功的日志。原始Token数组未落盘，不能给出首次Token分歧位置；上述假边依据冻结Raw结果与其代码，不伪称本轮逐Token重比。

**g006是必须保留的反向发现。** WFC的L=48,448及Exact的L=103,104都曾物化，所选正边都通过Raw；但req0011对`aux:003::pending0000`也为Raw真，Formal却排除此边。若原生Runtime使用该状态，P0可能获得更深恢复，不能继续把Formal E=[0,103104]当作其真实恢复向量。仅保留三条选中正边的真实性不够；完整候选域还含39条Raw-only关系。纠正这些关系会形成**新评价问题**，本轮不执行。

**逻辑K与物理对象也并非一一对应。** g006的`fa:002::req0000`与`fa:002::req0002`同Node、同117,440前缀摘要，后一项标记fallback。g017两个subagent的req0000同Node、同15,744前缀摘要。g010/g018各有两组重复物理Node。不能按逻辑未选就逐ID释放共享对象；现有`flowstate/strict_control.py::bind`（141起）明确拒绝多候选映射同一Node及候选域外Recurrent状态。这里是已知映射冲突，不能以“待观察”淡化。

## 4. GPU物化与Native Resume可用性

|必要条件|现有能力与证据|六例资格|
|---|---|---|
|完整历史输入与公共前缀|原始只有hash块及长度；旧composer为确定性合成，旧artifact只有长度、摘要和Raw布尔列表，无完整Pending/Checkpoint Token数组|BLOCKED：原始输入不可从当前归档忠实恢复；重新合成不等于原始输入认证|
|Qwen源Checkpoint物化|旧B2六例全部ELIGIBLE，记录8/35/38/12/30/149项FA和Recurrent驻留及Node；历史请求确实执行|部分已证实：仅旧合成重放源状态，不证明所有目标边成立或新Engine必然同样物化|
|Strict Epoch|`StrictAllocationEpoch.__post_init__`/`from_snapshot_bytes`要求完整原始Token、配置/epoch摘要、L≤T、准确prefix、三态矩阵且拒绝UNKNOWN|旧快照缺必要字段；五例有明确prefix冲突；g006完整关系不等价|
|物理绑定及公平预算|Strict控制路径要求当前Node/Slot/Conv/Temporal切片身份、零引用、无别名、全域槽位记账、Apply-time重验|旧摘要不满足当前合同；四例多逻辑ID共Node；不能跨生命周期复用旧Node/Slot|
|相同初始状态|既有独立Engine、冻结历史重放、单次Scheduler控制窗口可复用|通用机制存在，六例无当前完整Token输入和Strict物理证据，不能认证各策略等价起始边界|
|Native Match/COW/Cold/TTFT|R2-C1b2、B5-D3、E4等已实现真实匹配、COW、Cold输出及计时|这些是其他受控工作负载；旧AgentX仅执行历史请求及Pending只读FA查询，六例未有正式Pending恢复/Cold逐Token认证|

旧AgentX配置`max_mamba_cache_size=192、mem_fraction_static=0.80`（collector第83起）；近期Qwen严格受控案例使用28槽。不能称为原样复用同一物理配置。若要求全部旧候选同时独占槽，35/38/149项在28槽内无法成立；30逻辑候选虽对应28个Node，也已无必要工作/COW余量的完整认证。8/12项数量可容纳不等于prefix/别名问题已解决。本轮不修改池容量、不删候选、不重新设计物化。

已有可复用接口仅构成未来的技术基础：

- `agentx_runtime_snapshot_collection.py::_build_replay_plan/_collect_snapshot`：旧合成回放和只读前沿查询，不是已接入Strict的三策略恢复Runner。
- `runtime_control_closure_r2c0/bridge.py::observe/scope/slice_identity`：同Scheduler窗口、真实全树与切片身份。
- `flowstate/strict_epoch.py`：完整输入和边的不可变认证；`flowstate/strict_control.py::bind/preflight/StrictReconciler.reconcile_strict`：新鲜绑定、逐项安全释放和Fail-stop。上述是当前实际类与方法，尚无证据表明旧AgentX数据已经接入这条严格路径。
- `runtime_control_closure_r2c1b2/bridge.py`及`c_formal_perf_gpu/runner.py`：已验证的Req/Match/COW/Cold执行机制；其中工作负载装配、候选域和证据格式仍针对其冻结输入，不能直接指向AgentX旧JSON即认为适配完成。

本轮没有实现或修改Runner，也没有因通用API存在而把六例标为READY。

## 5. 代表性实例与唯一下一步建议

**合格GPU回放推荐：0个。** 六例均不能在保持原矩阵、候选及真实历史身份的前提下直接进入忠实GPU对照；不推荐把两个CPU代表案例包装成可执行回放。

仅作为论文CPU结构参照，最多保留以下两例：

- **g005/K1**：8候选、2 Pending，浅共享384与深专属43,840竞争，J差1,953.250872915 modeled ms；结构最紧凑。但三条目标正边全部不满足旧Raw prefix，不具备对应Native恢复含义。
- **g006/K1**：35候选、2 Pending，较浅共享48,448与深103,104竞争，J差1,139.383680254 modeled ms；所选边有Raw依据，但深状态的额外Raw支持使Formal取舍不等同Runtime取舍，又有Node别名及28槽容量问题，不能作为现成GPU回放候选。

唯一建议：先确认能否从原始采集方取得这六条trace的**当时完整输入或无损可渲染消息、原模型/Tokenizer配置及决策边界证据**。若拿不到，应把本批AgentX结果限定为Formal结构与预算敏感性证据，停止尝试忠实回放这六例。已有hash合成工具只支持明确标注的受控形状重放，不能用补造祖先历史来维持旧胜负。本轮不发起数据请求、不生成替代历史、不安排GPU实验。

## 6. 证据索引、复核范围与摘要

本轮只新增本报告。六例原始snapshot位于下列目录；均读取其`snapshot.json`，其`request_rows`仅含请求ID、输入/输出计数与历史runtime_metrics，`pendings`不含Token数组，`checkpoints`不含当前Strict所需完整组件切片绑定。

- `rq3-agentx-g005-epoch8`：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/snap_61962299108912449a53f411ed3cf2d1c013_162.822000/snapshot.json`；矩阵引用原Formal JSONL第16行。
- `rq3-agentx-g006-epoch35`：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/snap_63d3960fcc384388d5e7e4df8287cd1be081_955.256000/snapshot.json`；矩阵引用原Formal JSONL第19行。
- `rq3-agentx-g010-epoch38`：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/snap_77cfef0760bda37fb6f8829cebd9a7871aa5_78085.736000/snapshot.json`；矩阵引用原Formal JSONL第31行。
- `rq3-agentx-g017-epoch12`：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/snap_cbbac1ac4e0a88cb6ee971ef166fe6a24222_3205.463000/snapshot.json`；矩阵引用原Formal JSONL第52行。
- `rq3-agentx-g018-epoch30`：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/snap_d77fa1e0115f7fca67ca88b95a053d3a41bc_5402.628000/snapshot.json`；矩阵引用原Formal JSONL第55行。
- `rq3-agentx-g021-epoch149`：`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/per_snapshot/snap_ecd3ea07309dd8a6b62b2f90fd8c60fa92b2_2077.167000/snapshot.json`；矩阵引用原Formal JSONL第64行。

其余直接来源：

- `evaluation/paper_workload_structure_final/legacy_recheck_scan.py`：冻结扫描公式、WFC/Greedy规则、Exact证书及统计分母；本轮只加载纯CPU定义，未运行其写盘main。
- `evaluation/paper_workload_structure_final/legacy573_recheck.json`：历史69行Greedy/WFC选集校准，23快照及原预算饱和证书。
- `evaluation/gate1_step3_compatibility_retention_contract_audit/{audit_report.md,compatibility_diff.csv}`：完整1,956关系，70条Formal-only、413条Raw-only；本轮对六例选集精确联接，没有把全部70条泛化到每一例。
- `/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614/FORMAL_CANDIDATE_POPULATION.json`：23个冻结边界。
- `rq3_agentx_policy_eval_20260906_214956/{per_case_results.jsonl,formal_snapshot_manifest.json,source_integrity.json}`：69个原预算条件、实际兼容编码及历史完整性记录。

|本轮读取文件|数据行数/身份|SHA-256|
|---|---|---|
|`evaluation/paper_workload_structure_final/agentx_budget_scan.csv`|737行（CSV含表头）|`c075687e78d1bc9e2244e175a000aea2a55c648bf624e4a521007d7a107fb9f4`|
|`evaluation/gate1_step1_allocation_equivalence/matrices.jsonl`|573条JSONL记录|`9aa81ef02257a01f45a9aedc1357a48b98d601ab13cc80e6ba76e9b9520a64e6`|
|`evaluation/gate1_step3_compatibility_retention_contract_audit/compatibility_diff.csv`|1957行（CSV含表头）|`4f5f14cd1c6e76e23bd28fff4f31306c11039cff67dbddb0f5d05093ca5ceca8`|
|`rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl`|69条JSONL记录|`a22e18fa42cdcaa8490ea98230c70398d98df877442f843cd5a1d9dc3bb14866`|
|`/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/runtime_snapshots.jsonl`|23条JSONL记录|`cb88015fb75d5ea9d0978c2d4b1af1f2469bdfe1ee887a84337a9ced64a958e8`|

本轮对`source_hashes.json`记录的stats/matrices两份源摘要重新核对一致；原Formal结果SHA与T1来源清单一致；原始语料SHA亦重新全量核验一致。上述五个工作文件读取前后SHA一致。736行复核只验证冻结集合的模型J、Exact目标及K_min，不改写矩阵，不运行新策略，不认证原生恢复。

**结论边界：** 可以保留六个Formal条件的覆盖数/恢复收益冲突和Greedy184/184最优等CPU观察；不能声称六例已证明真实在线Agentic共享、可公平落实的物理预算、Native Resume正确性或GPU TTFT优势。严格GPU物化门禁为BLOCKED。
