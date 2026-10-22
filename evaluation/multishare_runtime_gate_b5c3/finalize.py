"""只依据真实采集和清理结果定门禁，不以期望位置补齐候选。"""
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
def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write(n,x):(P/n).write_text(x.strip()+'\n')
source=read('source_identity.json')
unchanged=all(sha(Path(p))==h for p,h in source['sha256'].items())
cleanup=read('gpu_cleanup.json')
if (P/'failure.json').exists() or (P/'launch_failure.json').exists():
 failure=read('failure.json') if (P/'failure.json').exists() else read('launch_failure.json')
 for name in ('historical_replay_log.json','physical_state_census.json','strict_epoch_snapshot.json','sharing_support_matrix.json'):
  if not (P/name).exists():save(name,{'status':'NOT_COMPLETED','reason':failure.get('error'),'说明':'失败后未填造观测'})
 write('state_integrity_report.md','# 状态完整性\n\nBLOCKED。生命周期发生异常，不能声明完整Strict观察通过。详见failure.json及runtime.log；没有自动重试。')
 write('final_report.md','# Phase B5-C3\n\n**BLOCKED**\n\n错误：'+str(failure.get('error'))+'\n\n清理：'+cleanup['status']+'。保留全部部分记录，不重试、不更换输入。')
else:
 result=read('result.json');state=read('physical_state_census.json');epoch=read('strict_epoch_snapshot.json');support=read('sharing_support_matrix.json');history=read('historical_replay_log.json');inputs=read('prepared_inputs.json')
 assert len(history)==8 and result['pending_dispatched']==0
 assert all(r['completed'] for r in history)
 assert state['before']==state['after'] and state['payload_before']==state['payload_after'] and state['unchanged']
 statuses={s:sum(e['status']==s for e in epoch['edges']) for s in ('EPOCH_ADMISSIBLE','REJECTED','UNKNOWN')}
 assert len(epoch['edges'])==len(epoch['candidates'])*4
 resident={n['node_id'] for n in state['before']['nodes'] if n['slots']}
 assert resident=={c['state_identity']['node_id'] for c in epoch['candidates']}
 slots=[s for n in state['before']['nodes'] for s in n['slots']]
 assert len(slots)==len(set(slots)) and len(slots)+len(state['before']['free_slots'])==28
 for i,row in enumerate(history):
  req=inputs['history'][i]
  assert row['request_id']==req['request_id'] and row['submitted_token_digest']==req['token_prefix_sha256']
  assert row['received_matches'] and all(m['input_token_digest']==req['token_prefix_sha256'] for m in row['received_matches'])
 origins=[]
 for c in epoch['candidates']:
  observations=[]
  for i in range(1,9):
   hist=read(f'history_{i:02d}.json')
   if any(n['node_id']==c['state_identity']['node_id'] and n['token_ids']==c['token_ids'] and n['slots'] for n in hist['state']['before']['nodes']):observations.append(hist['request']['request_id'])
  origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':observations[0] if observations else 'UNKNOWN','observed_after_requests':observations,'compatible_history_input_prefix_ids':[r['request_id'] for r in inputs['history'] if r['token_ids'][:c['token_position_L']]==c['token_ids']],'说明':'首次逐请求Census出现不是原生insert时间戳，也不以相同位置推断来源'})
 save('candidate_provenance.json',origins)
 from flowstate.recovery_model import RecoveryCostModel
 model=RecoveryCostModel()
 edges={(e['checkpoint_id'],e['continuation_id']):e for e in epoch['edges']}
 assert len(edges)==len(epoch['edges'])
 checked=0
 for c,row in zip(epoch['candidates'],support['rows']):
  assert c['checkpoint_id']==row['checkpoint_id']
  expected=[]
  for pending in epoch['pendings']:
   edge=edges[c['checkpoint_id'],pending['continuation_id']]
   prefix=pending['token_ids'][:c['token_position_L']]==c['token_ids']
   assert edge['conditions']['exact_token_prefix']==prefix
   assert edge['conditions']['L_le_T']==(c['token_position_L']<=pending['planning_target_T'])
   assert c['barrier_id']==pending['barrier_id']==epoch['epoch_id']
   assert c['configuration_digest']==pending['configuration_digest']==epoch['configuration_digest']
   assert (edge['status']=='EPOCH_ADMISSIBLE')==all(edge['conditions'].values())
   value=0
   if edge['status']=='EPOCH_ADMISSIBLE':
    t=pending['planning_target_T']
    value=model.estimate(t,t)-model.estimate(t-c['token_position_L'],t)
    if value>0:expected.append(pending['label'])
   assert abs(value-row['standalone_positive_evidence_ms'][pending['label']])<1e-9
   checked+=1
  assert expected==row['strict_positive_support']
 save('independent_validation.json',{'status':'PASS','checked_pairs':checked,'说明':'从完整Token数组独立核对前缀、位置、同边界及配置，并以冻结模型核对单边正收益；未执行任何分配策略'})
 sets={tuple(r['strict_positive_support']) for r in support['rows']}
 targets={('A','B','C','D'),('A','B'),('C','D')}
 gate='READY' if targets<=sets and statuses['UNKNOWN']==0 else 'PARTIAL'
 if cleanup['status']!='PASS' or not unchanged or not read('engine_cleanup.json')['shutdown_returned']:gate='BLOCKED'
 table='\n'.join(f"|{r['checkpoint_id']}|{r['L']}|{r['slots']}|{','.join(r['token_content_support']) or '无'}|{','.join(r['strict_positive_support']) or '无'}|" for r in support['rows'])
 requests='\n'.join(f"|{r['request_id']}|{r['output']['meta_info']['prompt_tokens']}|{r['output']['output_ids']}|{len(r['resident_node_ids_after'])}|{r['removed_recurrent_node_ids']}|" for r in history)
 pendings='\n'.join(f"|{p['label']}|{len(p['token_ids'])}|{p['attention_frontier']}|{p['planning_target_T']}|" for p in epoch['pendings'])
 write('state_integrity_report.md',f'''# 状态完整性与观察边界

同一Scheduler同步控制窗口：`{state['window']}`。scope检查无running/waiting/chunked请求、无传输/异步缓存功能；当前主线程同步调用不返回调度循环，故窗口内Admission不能推进。不是跨RPC租约。

前后PhysicalView、FA结构/索引摘要、LRU及引用等semantic内容一致；所有驻留Conv/Temporal实际数值逐槽SHA前后一致；逐层存储切片身份和无重叠检查通过。必要细节见physical_state_census.json的before/after、audit_before/audit_after、payload_before/payload_after。

Pool容量28，当前驻留槽{len(slots)}，空闲{len(state['before']['free_slots'])}；二者无交集且闭合。候选覆盖全部{len(resident)}个非哨兵驻留Recurrent节点，候选外有效Recurrent状态为空，不根据历史请求数假定候选数。FA-only节点完整保留在全树视图。

本轮Native eviction观察事件数{len(state['native_eviction_events'])}；每步节点增减、空闲槽和Native Match/COW保存在history_01–08.json。历史请求正常插入、分裂及访问元数据变化是回放副作用；最终观察窗口内无此变化。观察器包装原函数仅记录返回，不主动创建或驱逐状态。原生节点首次出现的逐请求溯源另存candidate_provenance.json，不冒称精确insert事件时间。

所有候选引用/保护和Host条件检查通过；Owner通过当前Node→Slot→组件字节切片确认。长期slot generation仍UNKNOWN，本轮只声明同一连续窗口内的身份，不建立永久Pin/Lease，也不认证未来请求Admission时仍驻留。

GPU清理：{cleanup['status']}；来源与既有冻结文件摘要：{'PASS' if unchanged else 'FAIL'}。四个Pending派发数0；Policy Reconciliation和主动Recurrent Eviction数0。
''')
 write('final_report.md',f'''# Phase B5-C3 最终报告

## Gate：{gate}

固定四Session的八条历史请求完成，四个Pending均未派发。实际捕获{len(resident)}个同边界驻留Recurrent候选；准入矩阵{len(epoch['edges'])}条：EPOCH_ADMISSIBLE={statuses['EPOCH_ADMISSIBLE']}、REJECTED={statuses['REJECTED']}、UNKNOWN={statuses['UNKNOWN']}。目标三个严格正收益支持集合是否全部出现：{targets<=sets}。

## 实际驻留结构

|Checkpoint|实际L|当前Slot|Token内容支持|严格正收益支持|
|---|---:|---|---|---|
{table}

位置均来自真实Cache节点，不预设为CPU LCP 2438/3765/3566。本轮没有插入候选、移动分叉边界、恢复/驱逐状态或更换Session。完整Conv/Temporal摘要、组件切片、Slot/Node、FA路径、引用及同窗口不变性均存于physical_state_census.json。

实际结构包含两个支持C/D的共享节点，但没有仅支持D的专属候选；八条历史请求不对应八个独占候选。D2的原生记录显示恢复前沿3456、分支跟踪位置3520，最终Census中的node27位于3520。此处只报告原始Match记录与最终物化观察，不把逐请求Census当作精确插入事件。未为补齐D专属状态采取任何操作。

## Pending前沿

|Pending|完整Token数|实际FA frontier|T=min(输入长度,FA)|
|---|---:|---:|---:|
{pendings}

FA由已验证的只读树遍历取得，不调用Prefix Match，不提交Pending。完整Token关系与严格准入分列；只有EPOCH_ADMISSIBLE且冻结Phi(T,T)−Phi(T−L,T)>0才计入正支持，REJECTED为0，UNKNOWN不视为合法。本轮只为正支持核验单边收益，不计算任何策略选集、预算扫描或总分配目标。

## 固定回放账本

|请求|实际接收Token数|原生生成Token|完成后驻留节点数|本步消失节点|
|---|---:|---|---:|---|
{requests}

实际提交与Native Match接收摘要逐请求一致；completion_tokens=1、无截断或回撤。原生生成输出逐项保存，只作为正常缓存副作用，不加入四个冻结Pending。历史Assistant和工具结果来自原始记录，不称目标模型原生闭环执行。没有请求级性能比较或TTFT改善结论。

## 合同与限制

Qwen3.5-9B / SGLang v0.5.17 / TP=1 / 单H100 PCIe / 28槽；完整输入与模板沿用B5-C2。全新独立生命周期，单次Scheduler同步窗口内冻结PhysicalView、FA、T和完整边矩阵。无分配器调用、无Reconciliation、无主动Eviction、无Pending执行。

本工作负载仍是Trace-derived Controlled Composition，不是自然同时Ready或原生Fork。EPOCH_ADMISSIBLE不等于逐边RESUME_VALIDATED；本轮不验证恢复输出。逻辑预算K未指定，不把当前池使用量解释为物理容量压力。多共享前缀支持仍可能满足树形精确可解结构，不证明任何策略优势。

## 完整性与结束

来源/既有冻结文件摘要：{'PASS' if unchanged else 'FAIL'}；Engine shutdown返回：{read('engine_cleanup.json')['shutdown_returned']}；GPU清理：{cleanup['status']}。三次清理观测与专用容器退出证据见gpu_cleanup.json，未终止其他GPU进程。

退出日志保留一条CUDA IPC警告：生产进程结束时部分共享CUDA张量尚未释放。Engine shutdown已返回、进程返回码为0；此后专用容器已退出，三次独立资源检查均无计算进程、显存为14 MiB。清理PASS依据这些实际检查，而不是仅依据退出码。警告未被删除，也未据此推断执行期状态损坏。

已交付全部九项文件及原始逐请求日志、源码锁、组件摘要、完整原始观测。完成后停止，不自动进入策略评估或恢复验证。
''')
 save('validation.json',{'gate':gate,'source_integrity':unchanged,'candidate_count':len(resident),'edge_counts':statuses,'three_support_sets':targets<=sets,'pending_dispatches':0,'slot_accounting':'PASS','state_noninterference':'PASS','gpu_cleanup':cleanup['status']})
save('evidence_manifest.json',{'files':{str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and 'triton_cache' not in p.parts and '__pycache__' not in p.parts and p.name!='evidence_manifest.json'},'说明':'本轮实际证据及实现，不含编译缓存'})
print('报告已完成。')
