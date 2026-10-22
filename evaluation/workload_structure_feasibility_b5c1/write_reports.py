"""根据只读结构检查生成中文报告；不执行策略或运行时操作。"""
import json
import hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
root=P.parents[1]
a=json.loads((P/'audit_summary.json').read_text())
inv=json.loads((P/'openhands_session_inventory.json').read_text())
by={r['session_id']:r for r in inv}
tasks=a['openhands']['multi_session_task_groups']
examples=[]
for task,ids in sorted(tasks.items())[:2]:
    examples.append(f"- `{task}`：共{len(ids)}条来源记录；按Session ID排序前两条为 "+'、'.join('`'+x+'`' for x in sorted(ids)[:2])+'。')

def write(name,s):
    (P/name).write_text(s.strip()+'\n')

write('source_inventory.md', '''# 来源清单与检查范围

本轮只读现有本地数据，未联网、下载、分词、构建快照或执行策略。具体文件存在性、大小、SHA-256见source_manifest.json。以下将本轮直接检查与历史审计分开，历史数字不冒充新运行结果。

|来源|真实路径与处理入口|本轮检查及已有证据|结论|
|---|---|---|---|
|OpenHands 原始归档|`/home/wjg/data/agentic_coding_trajectories/sessions.parquet`；`evaluation/rq3_openhands_population.py:32–40,184–224`|全部5000条`source_dataset=nebius-swe-rebench-openhands`记录，完整消息对象前缀；保留session_id/source_id/instance_id；不是仅检查旧168组|完整消息可读；每条是一条串行记录，多次运行同任务可追溯；原模型recorded_model全部为空，缺原生父子/Ready及Token身份|
|严格OpenHands受控证据|`evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/`；B5-C0 structural_contract.json、verification.json|本轮复用B5-C0的四组合同和结论，不重新运行矩阵评估|四组单共享，不支持多共享推断；新组合不能继承旧物理身份|
|AgentX/Weka|`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl`；agentx_chain_detection.py、agentx_runtime_snapshot_collection.py|本轮全文件SHA与P2记录一致，重读首条schema。历史P2全量393 traces、98827 invocations、1697 subagent容器|真实容器归属及hash形状可用；缺原始消息/Token、Ready、父请求继承事件。不能把旧Formal边改名Strict|
|FreeInference|`evaluation/new_dataset_suitability_audit/raw/fi_2026-08-09/`、`raw/fi_2026-08-30/`；sample_audit.py、session_metrics.jsonl|本轮读完已有464行逐会话指标，保留全部165个至少两个block前缀分叉的线索，其中25个含subagent|仅block结构；历史报告明确无原始消息且渲染非无损。多分叉不等于多个当前候选|
|LMCache|`/tmp/lmcache_check/0.parquet`至`4.parquet`存在；`agentic_workflow_dataset_research/phase2_submission_inputs/`|本轮不重扫五分片；读取selection_lock与既有报告。历史767 sessions、629 tasks、120多运行任务、16对冻结目标Token输入|完整消息/目标Token受控输入可用；16对旧合同每对仅一个共享C0，其他中点候选是PROPOSED；跨对不能直接合并成原生工作流|
|TraceElephant|P2 source_inventory.json列出的五个step_records.json|本轮重读五个固定首轨迹，共9/21/47/48/18步；input.messages可读，均有多个agent_id|标签和工具转交不等于继承前缀；未完成跨步骤输入、时间和执行配置关联。其余轨迹UNKNOWN，不宣称全库不适用|
|TraceLab|`/home/wjg/data/tracelab/v0.0.2/syfi_coding_trace.duckdb`|本轮核对路径；复用P2只读schema：rounds、timing_events、tool_calls、trace_source|轮次/工具时序并非模型Ready或共享状态；原始输入完整性待核验|
|InferenceX/Weka|`/home/wjg/data/inferencex_weka/traces_first30.jsonl`|路径存在；复用P2首条schema|同类hash输入，不补齐Exact Token|
|GATC|`evaluation/gatc_real_session_sample_audit/final_report.md`与更新的D1 final_report.md|本地没有已取得的实际样本证据；较新D1记录访问受限401，较旧D2记录网络阻断|BLOCKED；不重复下载，也不据卡片认定存在合格案例|

## 证据时效与边界

P2和早期真实结构报告早于B1/B4，不能用它们“OpenHands未证实共享”的旧结论否定后来真实跨Workflow恢复。这里复用的是原始数据字段限制；后来的B4四组单共享证据以B5-C0为准。

原始归档不是原服务端Token日志。OpenHands的消息可构成新目标模型受控输入，但不能恢复已丢失的原模型Tokenizer/模板身份。LMCache冻结数组也只认证对应目标重分词输入，不是原模型生产Token。所有来源的旧/新身份必须分别记录。
''')
write('sharing_structure_audit.md', '''# 多共享结构审计

## 固定判据与统计单位

在检查数值之前采用以下内容判据，不读取收益、排名或胜负：

1. 以每个原始Session为来源叶子，原样规范化完整消息对象（保留role/content/tool_calls_json/tool_call_id），构建消息前缀树。
2. 仅检查最后一个assistant之前的历史，确保每个内容前缀之后至少存在一个记录中的请求。每个Session计一次支持，不把其连续轮次当多个同时Pending。
3. 保存全部支持至少两个不同Session的前缀，分别标记仅System、超越System以及祖先/后代支持集合变化。保存所有匹配项，不按策略优劣筛选。
4. 本轮“支持”仅指消息结构，不是正收益支持。只有新Runtime中实际物化、L≤T且严格准入的候选才能进入后续正收益判断。未指定新的候选位置、Planning Target或预算。

## OpenHands直接检查结果

全部5000个唯一Session：634988个不同完整消息前缀节点；1217个共享节点，其中1个深度1的System节点，1216个深度2的System＋完整用户消息节点。全部记录前三角色均为system/user/assistant。1216个深度2共享组均对应同一个真实instance_id内部的重复执行；完整清单见openhands_shared_message_prefixes.json及openhands_session_inventory.json。

存在1216对“System祖先支持→任务用户消息子集支持”的严格包含关系，但其祖先全部只是System；**没有超越System的共享祖先再嵌套另一个不同共享支持集合**。在本次完整对象匹配口径下，没有深度≥3的跨Session共享节点。不能据此排除第三条消息内部的Token共同片段，也不能排除工具ID规范化后出现更多相同文本；本轮没有丢弃字段制造一致性。

1216个同任务多Session组提供可追溯的重复尝试，不证明其中一个请求原生fork成另一个。若将两个任务各取两个Session作为四需求，只能称Trace-derived Controlled Composition。它在消息层具有两个互不相交的非单例任务前缀，及可能的共同System祖先；不是新的Runtime快照。

同一共享支持的多个不同深度候选不一定增加分配难度：当较深者均合法且恢复成本单调，较深者可支配较浅者。不能用重复共享节点数替代不同支持集合数。还需保留专属状态并核验目标T，才能讨论预算权衡。

## 其他来源

- AgentX原生嵌套subagent与hash分叉具有潜在多层结构；没有完整Token，不能确认哪些Checkpoint能支持哪个Pending。旧23个非严格矩阵的多共享计数不作为本轮认证结果。SHA与P2一致仅认证来源内容未变。
- FreeInference的全部165个“至少两个block分叉”线索见freeinference_structural_leads.json，25个含subagent。该筛选仅用于结构索引，两个分叉可能发生在不同时间、不同模型或不同输入重写阶段。没有同边界多共享资格计数。其余299行没有满足本判据，不代表不含可用共享。
- LMCache已冻结的16对输入有合法目标Token前缀，旧C0/C1/C2设计仍是单共享。不能在C0之前任意加切点后称为自然多共享；实际前缀恢复A5只证明受控个例。
- TraceElephant五个固定来源含真实多agent步骤及消息，但本轮尚未建立父请求到子请求完整输入继承；其他轨迹不作否定判断。TraceLab同样只有候选事件线索。

## 前缀树与精确DP的适用性

在同一执行配置、准入当且仅当真实Token祖先且L≤T、完整驻留、固定资格、纯基数预算的条件下，将每个目标截到T挂在树上，候选支持必为嵌套或互不相交（允许相等）。这是理论条件，不是本轮新增Token认证。位置单调收益成立时，可适用T1的“节点、外部最近保留祖先、子树预算”精确DP；本轮不实现或计时。

多个非单例支持并不要求非层次交叉。两个互不相交的共享子树可能各自满足B5-C0的单共享化简，再对分支预算组合精确求解；出现多共享并不能证明一般困难性或Greedy必要性。若仅有支持laminar、收益任意或配对资格有额外过滤，不能自动套用纯前缀DP。参见allocation_structural_properties/final_report.md:184–236。

## 预算证据

当前没有新边界，因此K_min、实际W/R/H占用、K_ret及原生容量压力均UNKNOWN。多个真实需求及共享/专属位置意味着潜在选择权衡，但不证明同时Ready或有限容量竞争。严禁看结构后设置K=K_min−1制造物理压力；未来可预注册逻辑K范围作模型研究，必须与真实物理保留合同分列。
''')
write('strict_reconstruction_feasibility.md', '''# 严格兼容重建可行性

|来源|历史内容|Token/配置身份|同边界Pending与物化|本轮资格|
|---|---|---|---|---|
|OpenHands全量原始记录|完整序列可读，重复任务来源明确|原始Runtime Token缺失；recorded_model全部空；可未来固定Qwen身份生成新的受控输入|新组合均未采集；自然Ready未知|PARTIAL：可准备新受控输入，不可补写旧Strict矩阵|
|B4 g001–g004|已有实际输入与严格准入证据|同配置身份和物理合同已有|已认证受控静止边界|已有单共享子类；不是本轮多共享达标样本|
|AgentX|hash块与容器结构|不能逆推出原Token或完整文本|旧replay是block形状，原生Ready缺失|无法直接重建严格多共享输入|
|FreeInference|block_ids和请求/子代理记录|非无损；无消息可重建|时间重叠不等于Ready；无Recurrent物化|只能结构研究|
|LMCache|真实消息、部分冻结目标模型Token数组|目标受控输入身份有据；原执行配置不等同目标配置|多共享候选及共同边界缺失|PARTIAL：旧单共享认证不能外推|
|TraceElephant/TraceLab|部分消息或轮次/工具事件|覆盖与模板完整性尚未证明|父子继承和Ready尚未闭合|UNKNOWN，需数据合同审查|
|GATC|未取得样本|UNKNOWN|UNKNOWN|BLOCKED|

## 最小证据链

新受控工作负载需要：来源Session/请求索引→当前准备好的完整目标Token→实际提交及接收摘要→历史请求真实完成→原生Checkpoint物化→同一静止边界全树Census→组件/所有权/引用/别名→实际FA与T→逐边准入。只用消息摘要相同不足；不能假设完整首次输入末尾恰好留下checkpoint，原生跟踪粒度、模板尾部和Radix分裂均需观测。

复用入口为rq3_openhands_neutral_collector.py:236–303的输入截断和既有strict_compatibility_b4b0/capture.py。准备Pending时不读取待生成assistant输出；先冻结Barrier证据，再做接收身份核对，不用之后生成的状态反填。历史工具输出只在受控已知输入回放中作为已提供记录使用，不能称自然闭环Agent执行。

新组合必须独立于B4已有矩阵和物理绑定。EPOCH_ADMISSIBLE仍不等于RESUME_VALIDATED；R2跨Workflow个例也不认证新组合每条边。是否存在候选外可恢复状态、是否能保持多个祖先与分支专属Checkpoint同时驻留、安全协调是否落实配额，均需后续独立Runtime证据。

目前可可靠认证的新多共享严格Snapshot数为0；这是尚未采集/证据不足，不是证明数据里绝无此类结构。
''')
write('candidate_workload_plan.md', '''# 下一轮候选工作负载方案（仅设计）

## 优先来源和命名

优先采用OpenHands同任务重复执行，因为具有本地完整消息、instance_id关联及可复用受控采集设施。名称必须为 **Trace-derived Controlled Composition**，不是原生在线Fork/Join或原模型重执行。任务归属只决定可追溯组合，兼容性仍由实际目标Token和Runtime决定。

## 固定、非胜负驱动的前置核验方案

1. 以当前sessions.parquet内容SHA固定来源。完整1216个同任务重复组全部保留在审计账本；本轮未创建正式分组或新快照。
2. 建议下一次仅作CPU输入可用性核验：按instance_id、session_id字典序，检查最早两个任务各前两个Session，不失败补位。固定核验前两次历史请求及第三次请求之前的输入；不读第三次assistant输出。若轮次不足直接报告，不换成功样本。
3. 冻结目标Qwen模型、Tokenizer、模板和normalization身份后，才生成新受控输入；同时核验原消息可解析性、文本截断、tools及模板支持。它们是新目标输入，绝不冒充原始Runtime Token。
4. 检查两个任务对各自有共享内容、不同对有专属分支，保留所有结构及无结构结果。多个共同深度只记为潜在内容前缀，不新增中点候选，不计算Phi或策略结果。
5. 只有CPU输入证据成立，才另行申请Runtime采集。按自然历史请求和已有缓存语义形成状态；不强制插入目标切点，不改变原生匹配。若最终仍仅一个共享项或没有共享Checkpoint，原样报告并停止。

以下仅为按固定来源顺序得到的待核验例，不是认证候选集合：
'''+ '\n'.join(examples)+'''

## 如未来进入Runtime的准入与停止规则

必须捕获两个不同共享位置及其实际支持集，并明确区分同支持冗余、互不相交共享子树、嵌套共享子树；同时记录全部专属状态、当前准备输入和全树未管理状态。不以是否优于WFC筛选，不排除可精确化简者。

预算必须在看到策略结果前锁定。若做逻辑容量分析，可预注册K区间并另报K_min；若声称物理竞争，须先定义真实持久保留配额、工作/COW空间、池容量与释放合同。当前不能指定一个已证实造成真实竞争的槽数，保持UNKNOWN。不沿用28槽有空闲这一事实宣称HBM压力。

身份不一致、输入只能来自尚未提供的未来输出、原生状态无法同时驻留、存在关键UNKNOWN、物理保护冲突或原生驱逐污染时停止，不临时增加切点或重组任务。多共享结构通过后也先核对纯前缀DP适用性，而不是假设必须用一般Greedy。

本文件不是执行授权。本轮没有分词、GPU采集、严格矩阵构建或预算/策略比较。
''')
write('final_report.md', '''# Phase B5-C1 最终报告

## 最终门禁：PARTIAL

发现有真实来源的多共享**潜在内容结构**，但尚没有新的多共享Recurrent Snapshot满足Token身份、同边界物化及需求证据。因此不能给READY，也不把现有缺证据写成共享不存在。

## 主要发现

- 本轮直接读取全部5000条OpenHands原始归档会话。完整消息前缀有1217个跨会话共享节点：1个公共System节点、1216个System＋完整用户消息节点；后者全部具有相同instance_id的重复执行来源。没有发现共享到第三条完整消息的跨会话前缀；不排除消息内部Token部分共享。
- 有1216个同任务多Session组，全部记录而非筛选胜例。两个任务各两个Session可提出Trace-derived Controlled Composition，具有两个潜在互不相交共享子树；不是原生共同祖先Fork。公共System祖先与任务子集的嵌套只是内容层，不证明Runtime恰有两个对应共享Checkpoint。
- AgentX来源文件SHA与P2一致；hash和原生subagent容器仍不能补齐Token输入，旧Formal关系不转为Strict。
- FreeInference既有464行逐会话指标中，165个至少有两个block前缀分叉，25个还含subagent。已完整列出这些线索，但没有消息/Token，不能认证同边界多共享。
- LMCache已有16对Token输入和单共享恢复证据，不证明新的多共享状态；TraceElephant五个固定样本有多agent和消息，父子输入继承尚未核验；TraceLab仅有部分事件schema；GATC没有本地可用样本。

## 对核心问题的回答

1. 是否存在自然多层共享？AgentX/FreeInference有block/容器线索；本轮没有完整的原生Token＋Pending＋Checkpoint证据链。
2. 是否存在可追溯受控组合？存在OpenHands同任务重复执行的完整消息依据，可制定有限前置核验计划。不能仅凭相同任务认定兼容；完整消息检查与后续目标Token检查必须分开。
3. 是否出现多个非单例正收益支持？现有B4四组仍只有一个；新来源只有内容支持集合，没有新T/L/物化合同，所以正收益计数UNKNOWN。
4. 是否有真实容量竞争？UNKNOWN。多个需求、逻辑K不足与物理池压力是不同命题，不能通过事后压低K认证。
5. 是否能避开特殊精确可解结构？不能保证。严格Token树天然嵌套/不交，可能适用T1精确DP；两个独立单共享子树也可能继续分解。应保留这些案例，而不是追求不可能的交叉前缀。

## 唯一下一步建议

先执行一次独立、有限的 **OpenHands同任务重复轨迹受控组合CPU输入身份核验**：按来源固定顺序核对最早两个任务各两个Session、前两次历史及第三次请求输入，固定目标配置并检查完整Token前缀及消息因果截断，失败不补位。暂不投入GPU、不构造Checkpoint或比较策略。通过后才能另行决定是否进行严格Runtime采集。

## 交付和边界

source_inventory.md、sharing_structure_audit.md、strict_reconstruction_feasibility.md、candidate_workload_plan.md分别保存来源、结构、严格重建缺口和条件性方案。所有匹配的消息前缀、来源Session、层次关系及block结构线索均保存JSON账本；audit.py是只读CPU复核入口，不导入优化器或Runtime。source_manifest.json记录读取文件摘要。

未修改旧冻结产物、Greedy、Phi或Baseline；未启动GPU、Engine或Replay；未生成新Strict矩阵；没有新增策略比较或性能结论。完成后停止，等待审查。
''')
print('报告已写入。')
