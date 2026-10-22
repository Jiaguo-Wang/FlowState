"""独立核验完整Token产物并生成中文报告，不调用Tokenizer或策略。"""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
def read(n):return json.loads((P/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,s):(P/n).write_text(s.strip()+'\n')
def save(n,x):write(n,json.dumps(x,ensure_ascii=False,indent=2))
def lcp(a,b):return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
s=read('source_selection_lock.json');t=read('token_prefix_matrix.json');c=read('causal_checks.json');config=read('input_configuration_lock.json');req=read('requests.json')
assert sha(Path(s['source_path']))==s['source_sha256']
old=ROOT/'evaluation/workload_structure_feasibility_b5c1'
for name,digest in json.loads((old/'artifact_manifest.json').read_text())['files'].items():assert sha(old/name)==digest,name
assert len(req)==12 and t['gate']=='READY'
arrays={}
for r in req:
 tokens=read(r['token_file']);assert len(tokens)==r['token_count']
 assert sha(P/r['token_file'])==r['file_sha256']
 assert hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()==r['canonical_token_sha256']
 if r['turn']==3:arrays[r['label']]=tokens
assert len(arrays)==4
for a in arrays:
 for b in arrays:assert lcp(arrays[a],arrays[b])==t['pairwise_lcp'][a][b]
assert min(t['pairwise_lcp'][a][b] for a in arrays for b in arrays)==t['four_way_lcp']
boundary_rows=[]
for label in 'ABCD':
 for entry in c['boundary_mapping'][label][:2]:
  assert entry['exact_token_prefix']
  pos=entry['tokenized_prefix_length'];prefix=arrays[label][:pos]
  boundary_rows.append({'origin':label,'after_messages':entry['after_message_count'],'position':pos,'support':[q for q in arrays if arrays[q][:pos]==prefix],'token_prefix_sha256':hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),'materialized':False})
save('message_boundary_supports.json',boundary_rows)
assert all(not x[k] for x in c['collector_audits'] for k in ('pending_turn_output_read','r_plus_2_message_consumed','r_plus_2_request_materialized','future_timing_read','future_checkpoint_read'))
assert sum(len(r['tool_results']) for r in c['source_rows'])==8
for r in c['source_rows']:
 assert all(v['call_message_index']<v['message_index']<r['cut_index'] for v in r['tool_results'])
selection='\n'.join(f"|{r['label']}|{r['instance_id']}|`{r['session_id']}`|{r['parquet_row_index_zero_based']}|{r['pending_cut_index']}|" for r in s['records'])
mat='\n'.join('|'+a+'|'+'|'.join(str(t['pairwise_lcp'][a][b]) for b in 'ABCD')+'|' for a in 'ABCD')
lengths='\n'.join(f"|{label}|"+'|'.join(str(next(r['token_count'] for r in req if r['label']==label and r['turn']==turn)) for turn in (1,2,3))+'|' for label in 'ABCD')
write('causal_cut_audit.md',f'''# 因果截断与工作负载身份

## 固定来源

本轮身份为 **Trace-derived Controlled Composition**。按B5-C1 candidate_workload_plan.md已预定的instance_id、session_id字典序，而不是Parquet行序、Token长度或胜负选择。两任务分别为0b01001001__spectree-64与12rambau__sepal_ui-411，四Session固定，没有替换。

|标签|任务|Session|Parquet零基行号|第三次请求截断消息索引|
|---|---|---|---:|---:|
{selection}

## 读取与时间证据

每个输入只消费原始索引0至5：System、User、Assistant1、Tool1、Assistant2、Tool2。索引6仅通过role定位第三次请求；其content不进入重建。准备脚本需要解析源JSON记录以读取列表，但分词脚本仅收到截断历史加一个只含role的assistant哨兵，无法访问第三次输出及后续消息。没有用后续内容选取Session或计算前缀。

八条Tool记录均有对应先前Assistant的tool_call_id：调用索引2→结果3、调用4→结果5。原始顺序证明结果已在第三次请求输入之前出现；没有原始wall-clock、Ready/Dispatch日志，绝不把顺序索引冒充真实时间戳或自然并发证明。工具输出是历史原记录，不是本轮Qwen生成或执行工具的结果。

完整调用ID、索引与核验保存在causal_checks.json。沿用neutral collector的五个未来信息检查均为false。四个来源均有完整两次历史与第三次输入，没有补位或截短输入。

## 与B4一致性及差异

原函数：evaluation/openhands_single_workflow_smoke.py:66–110、169–187；evaluation/rq3_openhands_neutral_collector.py:230–303。按B4执行锁逐文件SHA核对后，AST原样加载三个规范化函数、materialize_group_requests及依赖，仅避免模块顶层导入Runtime。函数体、消息规范化和apply_chat_template参数未变。

使用冻结Qwen3.5-9B Tokenizer及模板、Transformers 5.12.1，SGLang镜像版本0.5.17。tools参数未传入，原始System内工具说明、历史tool_calls和tool_call_id按B4规则保留；没有额外猜测或补写工具schema。enable_thinking未传入，使用相同模板默认值。未截断Token。

差异是来源组合和CPU范围：本轮有8个准备好的历史输入及4个Pending缓冲，没有实际回放、Runtime接收确认、FA观测或Strict Epoch。独立B5C2请求ID替换旧构造器的标签，Token保持相同；requests.json保留旧构造器返回ID供核对。新组合不属于B4冻结g000或g001。

复用工具按turn再A/B/C/D的顺序仅保存为未来回放计划；本轮没有执行任何历史请求。
''')
write('runtime_pilot_prerequisites.md','''# 后续Runtime前提（仅记录，不执行）

本轮只认证目标CPU输入和内容结构。四个Pending分别取固定Session第三次请求，前两次历史输入都已保存。未来如获授权，可按A1/B1/C1/D1/A2/B2/C2/D2准备历史回放；四份Pending在Barrier前准备并记录controlled_input_ready，不称agent_native_ready。

预期内容边界：System完整前缀2428；四方共同LCP2438；A/B完整System＋User边界3723，最大LCP3765；C/D完整System＋User边界3502，最大LCP3566。它们只是观测到的Token位置，**不构成Checkpoint创建列表，也不要求Runtime保留这些精确位置**。原生track interval为256时，位置对齐、实际history输入路径、插入/分裂及保留行为都可能使实际节点不同。

未来必须另外验证：

1. 模型权重、Tokenizer、模板、镜像及配置身份；本轮未加载模型权重或验证GPU环境。配置锁中的B4 Engine参数仅作来源参照。
2. 历史请求实际提交/接收完整Token与本轮数组一致；不能直接把CPU重建写成原始Runtime Token。
3. 在真实静止边界观察原生生成的所有Checkpoint，不人工插入切点；完整Conv/Temporal、Node/Slot、所有权、引用、别名和候选外可恢复状态。
4. 同边界真实FA frontier及Planning Target。所有内容位置均不超过对应完整输入长度，但本轮T为UNKNOWN，不能假设T=N或全FA命中；须逐边检查L≤T。
5. 多个共享位置是否同时真实驻留且关联不同Pending支持集，是否还有更深专属状态。内容支持不等于EPOCH_ADMISSIBLE，更不等于RESUME_VALIDATED。
6. 真实持久保留配额与工作/COW空间、保护和native eviction；不能拿逻辑K或28槽配置宣称真实容量竞争。本轮没有K、K_min或收益矩阵。
7. 不能用后续Pending执行生成的节点回填此前Barrier。若实际仍单共享或候选不完整，原样报告，不更换Session、预算或输入。

下一步是否启动Runtime由人工另行决定；本轮到此停止。
''')
write('final_report.md',f'''# Phase B5-C2 最终报告

## Gate：READY（仅CPU输入身份与Token内容结构）

固定四Session来源、因果截断、目标Tokenizer/模板身份和完整Token数组均通过。形成四请求公共前缀与两个任务内更深共享子集，未更换来源或修改消息。不代表新Strict Snapshot、Recurrent物化、实际恢复或分配收益。

## 输入规模与LCP

|标签|历史1 Token数|历史2 Token数|第三次Pending Token数|
|---|---:|---:|---:|
{lengths}

|LCP（tokens）|A|B|C|D|
|---|---:|---:|---:|---:|
{mat}

- 四方共同最大前缀：**2438**；支持集合{{A,B,C,D}}。
- System完整渲染前缀：**2428**；四方共同LCP还包括用户消息头部的10个Token，不能把2438全称System。
- 任务一A/B：**3765**；任务二C/D：**3566**。两者均严格大于2438，支持分别恰为{{A,B}}、{{C,D}}。
- 所有跨任务配对LCP均为2438；Token分叉发生在User的工作区路径，位于完整System之后。
- A/B完整System＋User结束于3723；3765处分叉位于首条历史Assistant文字。C/D完整System＋User结束于3502；3566处分叉位于首条历史Assistant的think工具参数文字。不是两个任务共享同一原生Fork。

实际渲染全文及逐Token字符offset均保存；causal_checks.json记录每个消息边界与完整Token数组一致性，message_boundary_supports.json记录System及完整User边界支持。token_prefix_matrix.json保留所有配对、分叉Token及字符范围。

## 身份、因果性与配置

来源SHA与B5-C1一致；固定选择顺序在分词前写入source_selection_lock.json。四条记录均在第3个Assistant标记（索引6）之前截断；8条工具结果可追溯到更早tool_call_id。原始墙钟时间、自然Ready及跨Session同步关系UNKNOWN；READY只针对顺序可核验的受控已知输入，不声称原生在线因果事件认证。

原B4镜像ID为`{config['image_id']}`；Transformers {config['versions']['transformers']}、Tokenizers {config['versions']['tokenizers']}、SGLang包{config['versions']['sglang']}。Tokenizer文件及模型config摘要、实际加载模板摘要与B4一致。工具参数、规范化、生成提示参数与冻结入口相同。模型权重未加载；没有SGLang导入、Engine、GPU设备或网络。

当前主机uv312的Transformers 4.57.6不同于B4，因此没有用它生成正式Token。运行采用冻结镜像CPU容器、runc、无GPU挂载、网络禁用、只读源码/模型，仅本目录可写。容器已自动删除。

## 必须保留的限制

Token位置都在对应输入范围内；实际Planning Target与FA frontier尚未观测，L≤T只能留待Runtime确认。System/任务消息边界和LCP都不是已物化Checkpoint，也不是建议强制创建的位置。多个非单例内容支持仍为层次/不交结构，不证明一般困难性或FlowState收益。本轮没有计算任何收益、预算或策略。

来源中recorded_model缺失，不影响本轮明确指定Qwen的受控输入协议，但不能声称恢复了原模型的生产输入身份。历史Assistant和Tool均来自原轨迹，不冒称由本轮目标模型产生。

## 执行异常及完整性

四次前置/诊断失败均保留cpu_attempt1–4.log：AST注入节点缺少行号、容器UID写权限、System-only诊断被模板拒绝、offset的tuple/list序列化比较差异。前两次未生成Token；后两次发生在完整12份Token生成后。修复仅涉及本轮脚本/诊断，最终运行逐项复核此前Token和配置相同，没有修改冻结源码、输入、模板或换Session。最后cpu_run.log完整输出结构通过；门禁不单靠退出码。

独立finalize.py重新验证12份数组长度与SHA、全部LCP、工具因果索引、B5-C1所有产物摘要与原Parquet摘要。verification.json及artifact_manifest.json保存结果和交付身份。

主要交付：source_selection_lock.json、input_configuration_lock.json、token_prefix_matrix.json、causal_cut_audit.md、runtime_pilot_prerequisites.md，以及12份完整Token数组、原始截断历史、渲染文本和复核脚本。

**本轮完成后停止，不自动进入Runtime Pilot。**
''')
save('verification.json',{'gate':'READY','source_sha':'PASS','prior_b5c1_artifacts':'PASS','token_files':12,'pending_inputs':4,'pairwise_lcp':'PASS','message_boundaries':'PASS','tool_result_causality':{'count':8,'ordered_links':'PASS','wall_clock':'UNKNOWN'},'planning_target':'UNKNOWN','runtime_checkpoint_count':'NOT_OBSERVED','source_or_production_modified':False,'cpu_container_cleanup':'PASS：docker ps -a按本轮名称过滤为空','gpu_used':False,'engine_started':False})
save('artifact_manifest.json',{'files':{p.name:sha(p) for p in P.iterdir() if p.is_file() and p.name!='artifact_manifest.json'},'说明':'只覆盖本轮独立CPU产物；旧来源摘要复核通过'})
print('READY：独立核验通过，报告和清单已生成。')
