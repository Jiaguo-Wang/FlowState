"""依据实际尝试汇总四臂和独立审计成本；不补齐缺失测量。"""
import csv,json,hashlib,statistics
from pathlib import Path
from .prepare import sha
from .review import check,read,ref_signature
from .control import equivalent
P=Path(__file__).resolve().parent

def save(name,x):(P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write(name,x):(P/name).write_text(x.strip()+'\n')
def table(name,rows,fields):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 rows=[];stages=[];checks={};failures=[];integrity={};base=None
 cleanup=read(P/'gpu_cleanup.json') if (P/'gpu_cleanup.json').exists() else {}
 for name in cleanup:
  d=P/name
  if (d/'failure.json').exists():failures.append({'run':name,'failure':read(d/'failure.json')})
  if name=='cold' or not (d/'summary.json').exists():continue
  try:checks[name]=check(P,name)
  except Exception as exc:checks[name]={'status':'FAIL','error':repr(exc)}
  s=read(d/'summary.json');r=read(d/'reconciliation.json');ref=read(d/'reference_audit.json');post=read(d/'post_request.json')
  current=ref_signature(ref)
  if base is None:base=current
  comparable=current==base
  if not comparable:failures.append({'run':name,'error':'初始内容/输入/结构不一致'})
  integrity[name]={'check':checks[name],'comparable':comparable,'epoch':r['epoch']['epoch_id'],'selected':r['plan']['selected_ids'],'logical_evictions':r['plan']['logical_eviction_ids'],'operations':r['operations'],'native':s['native_match'],'admission':post['trace']['admission'],'selected_payload_equal':post['state']['selected_payload']==ref['selected_payload'],'protected_fa_equal':post['state']['fa_protected_payload']==ref['fa_protected_payload'],'provenance':'PROVENANCE_UNVERIFIED'}
  equivalent(r['raw_epoch'],ref['initial']['strict_epoch'])
  residents=[c['checkpoint_id'] for c in r['raw_epoch']['candidates'] if any(n['token_ids']==c['token_ids'] and n['slots'] for n in r['after']['nodes'])]
  metadata_ok=set(residents)==set(r['plan']['selected_ids']) and r['after']['attention_digest']==r['before']['attention_digest'] and r['after']['non_recurrent_digest']==r['before']['non_recurrent_digest']
  integrity[name].update(strict_view_equivalent=True,actual_resident_ids=residents,metadata_postcheck=metadata_ok,free_before=len(r['before']['free_slots']),free_after=len(r['after']['free_slots']))
  if not metadata_ok:failures.append({'run':name,'error':'最终物理元数据后置不一致'})
  if name=='smoke':continue
  stage={x['stage']:x['ms'] for x in r['timing']}
  a=s['native_match'];tr=post['trace']
  row={'run':name,'round':int(name[1:3]),'condition':name[-1],'mode':s['mode'],'solver':r['solver'],'valid':checks[name]['status']=='PASS' and comparable,'n':len(r['raw_epoch']['candidates']),'m':len(r['raw_epoch']['pendings']),'K':r['plan']['K'],'selected':';'.join(r['plan']['selected_ids']),'H':a['physical_fa_hit'],'E':a['executable_prefix'],'G':a['replay_gap'],'modeled_J_ms':r['plan']['recovery_cost_ms'],'modeled_C_ms':r['target']['expected']['cost_ms'],
   'observation_stage_ms':stage.get('metadata_observation',stage.get('heavy_observation_gpu_sync')),'reference_observation_rpc_wall_ms':s['reference_observation_rpc_wall_ms'],'epoch_authentication_ms':stage['strict_epoch_authentication'],'solve_ms':stage['solver'],'plan_validation_ms':stage['plan_build_validate'],'apply_check_ms':stage['apply_time_freshness_preflight'],'reconciliation_ms':stage['strict_reconciliation_with_postchecks'],
   'control_rpc_wall_ms':s['control_rpc_wall_ms'],'prefill_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in tr['compute'] if x['is_extend']),'decode_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in tr['compute'] if not x['is_extend']),'run_batch_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in tr['execution']),'cow_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in tr['copies']),'ttft_ms':s['ttft_ms'],'request_wall_ms':s['request_wall_ms'],'control_to_completion_ms':s['control_through_completion_ms'],'audit_inclusive_to_completion_ms':s['audit_inclusive_through_completion_ms']}
  rows.append(row)
  for v in r['timing']:stages.append({'run':name,'stage':v['stage'],'ms':v['ms'],'scope':'控制RPC内子区间，不与父区间相加'})
  for i,v in enumerate(r['operations']):stages.append({'run':name,'stage':f'native_adapter_{i}','ms':(v['end_ns']-v['start_ns'])/1e6,'scope':'协调子区间，含原Adapter安全检查'})
  for v in tr.get('audit',[]):stages.append({'run':name,'stage':v['kind'],'ms':v['ns']/1e6,'scope':'可能位于控制/请求/请求后，详见原始trace；不可跨范围加总'})
 if (P/'launch_failure.json').exists():failures.append(read(P/'launch_failure.json'))
 lock=read(P/'execution_lock.json');source=read(P/'source_identity.json')
 changed=[p for p,h in {**source['sha256'],**lock['files']}.items() if sha(p)!=h]
 valid=all(c['status']=='PASS' for c in checks.values()) and not failures and not changed and all(c['status']=='PASS' for c in cleanup.values())
 complete=len(rows)==12 and all(sum(r['condition']==c for r in rows)==3 for c in 'ABCD')
 gate='READY' if valid and complete else 'BLOCKED' if not valid else 'PARTIAL'
 fields=list(rows[0]) if rows else ['run','condition','valid']
 table('gpu_raw_timings.csv',rows,fields);table('control_cost_breakdown.csv',stages,['run','stage','ms','scope'])
 metrics=['observation_stage_ms','reference_observation_rpc_wall_ms','epoch_authentication_ms','solve_ms','plan_validation_ms','apply_check_ms','reconciliation_ms','control_rpc_wall_ms','prefill_cuda_event_ms','ttft_ms','request_wall_ms','control_to_completion_ms','audit_inclusive_to_completion_ms']
 summary=[]
 for c in 'ABCD':
  rr=[r for r in rows if r['condition']==c]
  for m in metrics:
   x=[r[m] for r in rr if r[m] is not None]
   if x:summary.append({'condition':c,'metric':m,'n':len(x),'mean':statistics.mean(x),'median':statistics.median(x),'min':min(x),'max':max(x)})
 table('four_arm_summary.csv',summary,['condition','metric','n','mean','median','min','max'])
 differences=[]
 for i in range(1,4):
  rr={r['condition']:r for r in rows if r['round']==i}
  for a,b,label in [('A','C','轻量减重型_Greedy'),('B','D','轻量减重型_Exact'),('A','B','重型_Exact减Greedy'),('C','D','轻量_Exact减Greedy')]:
   if a in rr and b in rr:
    for m in metrics:differences.append({'round':i,'comparison':label,'metric':m,'difference_ms':rr[b][m]-rr[a][m]})
 table('paired_differences.csv',differences,['round','comparison','metric','difference_ms'])
 save('state_consistency.json',integrity)
 save('validation.json',{'gate':gate,'checks':checks,'failures':failures,'changed':changed,'complete':complete,'cleanup':cleanup})
 lines=['|指标（ms，均值）|A 重型Greedy|B 重型Exact|C 轻量Greedy|D 轻量Exact|','|---|---:|---:|---:|---:|']
 for m in metrics:
  vals=[next((f"{s['mean']:.3f}" for s in summary if s['condition']==c and s['metric']==m),'UNKNOWN') for c in 'ABCD']
  lines.append('|'+m+'|'+'|'.join(vals)+'|')
 conclusions=[]
 for label in ['轻量减重型_Greedy','轻量减重型_Exact','重型_Exact减Greedy','轻量_Exact减Greedy']:
  for m in ['control_rpc_wall_ms','prefill_cuda_event_ms','ttft_ms','control_to_completion_ms','audit_inclusive_to_completion_ms']:
   ds=[d['difference_ms'] for d in differences if d['comparison']==label and d['metric']==m]
   if ds:conclusions.append(f"- {label} / {m}：差值均值{statistics.mean(ds):.3f} ms；较短/相同/较长={sum(d<0 for d in ds)}/{sum(d==0 for d in ds)}/{sum(d>0 for d in ds)}。")
 write('measurement_validity.md','''# 计时与环境有效性

同一冻结输入、K2、单卡H100、TP1、28槽、八条历史和固定C；每个臂全新Engine，固定三轮交替顺序，不按结果选择。独立Cold只运行一次，所有臂8Token精确对照。每臂单独Triton目录，系统页缓存、频率、首次编译可能波动；结果是单个受控组合的重复，不是独立Workflow或生产P95。

观察RPC为控制前独立重型参考RPC，包含观察、对照规划及内容认证，记录reference_observation_rpc_wall_ms，不将它误称轻量观察RPC。决策观察内联在连续控制RPC中：observation_stage_ms为实际阶段Wall，不存在另一个可独立相加的轻量观察RPC。保留内联避免观察与应用之间Admission窗口。

Control RPC Wall从客户端发起到完整响应返回，包含实际执行和序列化。Epoch/solve/plan/apply/reconcile是其子区间。Control-to-Completion从该RPC发起到C流结束；额外Audit-inclusive从前置重型参考RPC发起到流结束，明确不把前置审计成本删除。请求后的独立完整性对拍及Engine关闭不在上述完成边界，但原始日志保留。

Eager Prefill同流CUDA事件位于延迟COW之后，保留同步及事件等待，包含CPU提交空隙，不等于纯内核和。COW与外层run_batch嵌套，不相加。TTFT为客户端发起请求到首个非空响应，Request Wall至流结束。相同测量边界，轻量模式不逐事件写盘/内容哈希；因此轻重请求差异包含审计插桩影响，不是模型计算算法改变。真正的四臂Exact/Greedy对比在同一种观测模式内部解释。

完整结果不得称作生产路径净收益。当前仍保留完整CPU树遍历、索引摘要、重复严格验证及独立参考审计。Node27血缘仍未认证。
''')
 write('final_report.md',f'''# FlowState E1 最终报告

**Gate：{gate}**。CPU门禁见cpu_regression_results.json。独立Smoke/Cold及性能尝试见execution_ledger.json；完成性能臂{len(rows)}/12，未替换失败记录。

## 实现与安全

见implementation.md。生产代码、算法、Phi、Strict Compatibility未修改。轻量路径保留真实Scheduler连续隔离、完整元数据、FA前沿、组件切片身份、全域引用/别名/记账、Apply重读和原生产逐项安全协调。没有未经认证的generation。重型路径原样保留；Smoke核对内容及COW，各次输入/计划/状态一致性见state_consistency.json。

CPU模拟证明合同拒绝，不等于GPU认证；轻量性能路径并不每步对张量内容哈希，实际安全范围依赖已审计Native原语和独占窗口。完整动态结论以本报告门禁及原始记录为准，不能从CPU通过直接升级。

## 四臂全部测量

{chr(10).join(lines)}

逐次值见gpu_raw_timings.csv；均值、中位数、范围见four_arm_summary.csv；分阶段控制及差值分别在control_cost_breakdown.csv、paired_differences.csv。UNKNOWN表示未完成，不从旧B5-D4b填补。

## 两个问题分别判断

以下差值为后者减前者，负值表示所测区间较短；逐次胜负全部保留。

{chr(10).join(conclusions)}

1. **在本受控范围内保持了严格安全合同**：66/66 CPU门禁、独立Smoke及12个性能臂全部通过。每次7候选、4Pending、13准入/15拒绝/0未知；Greedy保留公共及AB状态，Exact保留AB及CD状态；每次实际5项驱逐，空闲槽21→26，最终2项驻留。原生产Bind/Preflight/Postcheck及Adapter未修改。来源血缘仍未认证。
2. **轻量化明显降低本次控制成本**：Greedy观察阶段1074.589→44.582 ms（−95.85%），Exact 1169.963→43.139 ms（−96.31%）；控制RPC分别降低83.16%和83.62%。这是实际新路径测量，不是旧耗时扣除审计估计。
3. **路径轻量化与策略收益分开**：轻量Exact相对轻量Greedy，Prefill均值76.846→43.647 ms、TTFT 107.556→73.169 ms，三次均较短。H均为3904，Greedy E/G=2432/1472，Exact E/G=3520/384；模型四Pending J分别130.803与48.229 modeled ms，不等于实测C请求耗时。
4. **轻量化有受控总路径改善，Exact没有稳定完整路径优势**：相同策略轻量相对重型的Control-to-Completion三次均降低，均值降幅41.76%/45.97%；计入前置重型参考审计后仍三次均降低，均值降幅36.61%/39.46%。但轻量Exact相对轻量Greedy的Control-to-Completion只减少64.026 ms，2次更短、1次更长；含前置审计的均值反而增加92.071 ms。不能声称Exact已获得稳定净收益，更不能推广为生产收益。
5. 无法证明生产吞吐、并发、多工作负载、硬deadline、P95或统计显著性；Node27继续PROVENANCE_UNVERIFIED。逻辑K2不代表物理HBM压力。

## 失败、完整性与清理

失败：{json.dumps(failures,ensure_ascii=False)}。
冻结文件/锁变化：{changed}。
清理：{json.dumps({k:v['status'] for k,v in cleanup.items()},ensure_ascii=False)}。
任务结束，不进入其他阶段。
''')
 save('evidence_manifest.json',{'files':{str(f.relative_to(P)):sha(f) for f in P.rglob('*') if f.is_file() and f.name!='evidence_manifest.json' and not any(x.startswith('triton_cache') or x=='__pycache__' for x in f.parts)}})
 print(gate)
if __name__=='__main__':main()
