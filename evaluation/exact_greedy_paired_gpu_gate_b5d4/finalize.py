"""保留全部已执行臂，分开报告模型目标、设备事件区间与含审计完整成本。"""
import csv,hashlib,json,statistics
from pathlib import Path
from .review import read,check_pair,signature
from .protocol import compare_outputs
P=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,x):(P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write(name,s):(P/name).write_text(s.strip()+'\n')
def table(name,rows,fields=None):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def avg(xs):return statistics.mean(xs)

def main():
 cleanup=read(P/'gpu_cleanup.json');source=read(P/'source_identity.json');lock=read(P/'implementation_lock.json')
 changed=[p for p,h in {**source['sha256'],**lock['files']}.items() if sha(p)!=h]
 cold=read(P/'cold/output.json') if (P/'cold/output.json').exists() else None
 rows=[];costs=[];sources={};correct={};identities={};validations={};failures=[]
 for pair in range(1,6):
  if (P/f'p{pair:02d}_Greedy/summary.json').exists() and (P/f'p{pair:02d}_Exact/summary.json').exists() and cold:
   try:validations[pair]=check_pair(P,pair)
   except Exception as exc:validations[pair]={'status':'FAIL','error':repr(exc)}
  for solver in ['Greedy','Exact']:
   name=f'p{pair:02d}_{solver}';d=P/name
   if not d.exists():continue
   if (d/'failure.json').exists():failures.append({'run':name,'failure':read(d/'failure.json')})
   if not (d/'summary.json').exists():continue
   s=read(d/'summary.json');r=read(d/'reconciliation.json');post=read(d/'post_request.json');trace=post['trace'];a=s['native_match'];plan=r['plan'];t=r['target']
   try:correct[name]=compare_outputs(read(d/'output.json'),cold) if cold else {'status':'UNKNOWN'}
   except Exception as exc:correct[name]={'status':'FAIL','error':repr(exc)}
   compute=trace['compute'];outer=trace['execution'];copies=trace['copies']
   row={'pair':pair,'run':name,'solver':solver,'valid_pair':validations.get(pair,{}).get('status')=='PASS',
    'n':len(r['raw_epoch']['candidates']),'m':len(r['raw_epoch']['pendings']),'K':plan['K'],'selected_ids':';'.join(plan['selected_ids']),
    'H':a['physical_fa_hit'],'E':a['executable_prefix'],'G':a['replay_gap'],'input_tokens':a['input_token_count'],
    'prefill_tokens':sum(x.get('extend_token_count',0) for x in outer if x['is_extend']),
    'modeled_J_all_ms':plan['recovery_cost_ms'],'modeled_C_ms':t['expected']['cost_ms'],
    'prefill_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in compute if x['is_extend']),
    'decode_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in compute if not x['is_extend']),
    'run_batch_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in outer),'cow_cuda_event_ms':sum(x['cuda_elapsed_ms'] for x in copies),
    'native_match_host_ms':sum(x['native_call_ns']/1e6 for x in trace['matches']),
    'ttft_client_ms':s['ttft_ms'],'stream_tpot_ms':s['stream_tpot_ms'],'request_wall_ms':s['request_wall_ms'],
    'control_rpc_wall_ms':s['control_rpc_wall_ms'],'control_through_completion_ms':s['control_through_completion_ms']}
   rows.append(row)
   for v in r['timing']:
    kind='审计' if v['stage'] in ['heavy_observation_gpu_sync','independent_oracle'] or v['stage'].startswith('fa_content') else '必要控制与审计混合'
    costs.append({'run':name,'stage':v['stage'],'ms':v['ms'],'kind':kind,'containment':'控制RPC的子区间，不与RPC总量相加'})
   for i,op in enumerate(r['operations']):costs.append({'run':name,'stage':f'native_adapter_eviction_{i}','ms':(op['end_ns']-op['start_ns'])/1e6,'kind':'原Adapter原生操作含其安全检查','containment':'strict_reconciliation_with_postchecks内，非纯设备内核'})
   for v in trace.get('audit',[]):costs.append({'run':name,'stage':v['kind'],'ms':v['ns']/1e6,'kind':'内容/状态审计','containment':'可能在Admission、Match、COW或请求后；与父区间重叠'})
   costs.append({'run':name,'stage':'trace_json_write','ms':trace.get('audit_write_ns',0)/1e6,'kind':'证据写盘','containment':'包括嵌套写盘；非所有客户端日志写盘'})
   for k in ['control_rpc_wall_ms','request_wall_ms','control_through_completion_ms']:costs.append({'run':name,'stage':k,'ms':s[k],'kind':'实测总量','containment':'完整路径包含控制RPC和请求Wall，不得再次加总'})
   sources[name]={'target':t['pending']['continuation_id'],'source_history':t['source_history']['request_id'],'source_session':t['source_history']['workflow_id'],
    'target_session':t['pending']['workflow_id'],'source_checkpoint':t['source_checkpoint'],'plan_expected':t['expected'],'actual_match':a,'copies':copies,
    'selected_integrity':r['selected_payload_before']==r['selected_payload_after']==post['state']['selected_payload'],
    'fa_reconciliation_unchanged':r['fa_content_before']==r['fa_content_after'],'protected_fa_after_resume':post['state']['fa_protected_payload']==r['fa_protected_payload']}
   identities[name]={'gpu':read(P/f'{name}_gpu_before.json'),'engine':r['before']['engine_id'],'window':r['before']['barrier_id'],'epoch':r['epoch'],'configuration':r['raw_epoch']['configuration'],'comparable_signature_sha256':hashlib.sha256(json.dumps(signature(r),sort_keys=True).encode()).hexdigest()}
 if (P/'launch_failure.json').exists():failures.append(read(P/'launch_failure.json'))
 allpass=len(validations)==5 and all(x['status']=='PASS' for x in validations.values()) and len(rows)==10 and all(x['status']=='PASS' for x in correct.values())
 clean=len(cleanup)==11 and all(x['status']=='PASS' for x in cleanup.values())
 gate='READY' if allpass and clean and not failures and not changed else 'BLOCKED' if failures or changed or any(x['status']!='PASS' for x in cleanup.values()) else 'PARTIAL'
 table('paired_gpu_timings.csv',rows,fields=None if rows else ['pair','run','solver','valid_pair'])
 table('control_cost_breakdown.csv',costs,fields=['run','stage','ms','kind','containment'])
 save('resume_source_verification.json',sources);save('cold_reference_correctness.json',{'cold_path':'cold/output.json','comparisons':correct})
 save('paired_runtime_identity.json',{'source_identity':'source_identity.json','implementation_lock':'implementation_lock.json','identities':identities,'pair_validation':validations})
 differences=[]
 for pair in range(1,6):
  rr={r['solver']:r for r in rows if r['pair']==pair}
  if len(rr)!=2:continue
  g,e=rr['Greedy'],rr['Exact'];v={'pair':pair,'valid':validations.get(pair,{}).get('status')=='PASS'}
  for col in ['prefill_cuda_event_ms','ttft_client_ms','request_wall_ms','control_rpc_wall_ms','control_through_completion_ms']:v[col+'_exact_minus_greedy']=e[col]-g[col]
  differences.append(v)
 save('paired_differences.json',differences)
 lines=['|Pair|Greedy Prefill事件ms|Exact Prefill事件ms|Exact−Greedy完整路径ms|','|---|---:|---:|---:|']
 for d in differences:
  g=next(r for r in rows if r['pair']==d['pair'] and r['solver']=='Greedy');e=next(r for r in rows if r['pair']==d['pair'] and r['solver']=='Exact')
  lines.append(f"|{d['pair']}|{g['prefill_cuda_event_ms']:.3f}|{e['prefill_cuda_event_ms']:.3f}|{d['control_through_completion_ms_exact_minus_greedy']:.3f}|")
 summary=[]
 for solver in ['Greedy','Exact']:
  rs=[r for r in rows if r['solver']==solver]
  if not rs:continue
  for metric in ['prefill_cuda_event_ms','ttft_client_ms','request_wall_ms','control_rpc_wall_ms','control_through_completion_ms']:
   xs=[r[metric] for r in rs];summary.append({'solver':solver,'metric':metric,'n':len(xs),'mean':avg(xs),'median':statistics.median(xs),'min':min(xs),'max':max(xs)})
 save('descriptive_statistics.json',summary)
 write('reconciliation_correctness.md',f'''# 协调与恢复正确性

实际有完整摘要的策略臂{len(rows)}，配对核验结果：{json.dumps({k:v['status'] for k,v in validations.items()},ensure_ascii=False)}。

各臂独立八条历史，当前Epoch构建、当前计划、Apply-time重验、bind/preflight与生产reconcile_strict在同一次Scheduler同步调用内完成。逻辑互斥、组件、引用和完整驱逐补集经原生产检查，所选组件内容和FA全树索引/内容独立后验检查。实际操作记录见每臂reconciliation.json；没有复用其他生命周期物理绑定。

请求Admission再核对协调基线；实际Native Match来源、COW完成与源/目标内容、历史状态保护见resume_source_verification.json。只有固定Pending C被派发，其他三个Pending没有执行。Exact来源是否为D历史已作为运行门禁，而非结果之后重新命名来源。

已知失败：{json.dumps(failures,ensure_ascii=False)}。所有已执行臂保留，没有替换失败运行。
''')
 write('measurement_validity.md','''# 计时有效性与可比性

计时接口依据冻结Runtime真实源码，见runtime_timing_source_audit.json、observation.py、bridge.py。非重叠调度、禁用CUDA Graph；COW在ModelRunner._forward_raw里先于EagerRunner.execute。主Prefill指标为EagerRunner.execute的同流CUDA事件区间，包含输入缓冲/Attention初始化/模型执行及主机提交空隙，排除前面的COW内容哈希。首次编译可能让事件间设备空闲；不称其为内核时间之和或稳态生产Prefill。

外层run_batch事件包含COW及其审计，不能与内层Eager事件相加。COW事件与Host Match记录各有独立边界。事件前设备同步、事件后等待会扰动异步执行，两臂同样应用；完整Wall保留所有这些成本。没有用Host调用时间冒充GPU时间。

客户端TTFT从流式调用前到首非空Token响应；TPOT为首末响应间隔/7，仅8Token描述量。完整路径从控制RPC发起到目标流结束，含控制/后续准备/请求与内嵌审计，不包含Engine启动、八条历史或请求后的最终Census；这些仍有原始日志。未记录的写盘开销不推测为0；其影响仍在相应总Wall内。

配对一致性要求完整Token、严格条件、T/H、候选内容、FA索引和内容、池容量与占用一致；物理指针不能跨生命周期复用。每臂单独空Triton目录，但系统文件页缓存、设备温度、频率及其他库缓存未强制一致。首次编译与Host提交开销可能主导Decode或Prefill，故即使5对全部完成，也不是生产稳态或显著性证明。

所有均值、范围和逐对差值以实际记录为准，完整成本不通过减去审计项人为改成生产净收益。四Pending模型J、目标C模型成本、GPU事件时间和Wall分栏。单C结果不证明四Pending总吞吐、Goodput或整体收益。逻辑K2不等于真实物理内存压力。
''')
 numeric='\n'.join(f"- {x['solver']} {x['metric']}：均值{x['mean']:.3f} ms，中位{x['median']:.3f} ms，范围[{x['min']:.3f},{x['max']:.3f}]，n={x['n']}。" for x in summary)
 write('final_report.md',f'''# Phase B5-D4 最终报告

**Gate：{gate}**。有效配对{sum(v['status']=='PASS' for v in validations.values())}/5；已完成策略臂{len(rows)}；失败记录{len(failures)}。Gate不以Exact获胜为条件。

## 恢复与状态证据

固定目标B5C2_C_TURN_003；实际当前选集、来源、H/E/G、工作槽及8Token对照见paired_gpu_timings.csv、resume_source_verification.json及cold_reference_correctness.json。各臂由当前Epoch重新求解，Exact与当前Oracle核对，非选状态通过原生产严格协调移除。跨Workflow来源通过当前回放首次物化记录、Token前缀和Native匹配/COW执行串联，不仅以输出一致推断。

## 全部成对观察

{chr(10).join(lines)}

负值表示Exact在该实测区间更短，正值更长。以上Prefill为COW之后的Eager设备事件时间线，不等同恢复缺口G的纯计算。完整输入后缀还包含T之后的新输入，实际Prefill Token数单独保存。

{numeric}

## 解释范围

完整控制路径包含重型观察、GPU哈希、当前Epoch认证、Oracle、规划、Apply检查和协调；成本详见control_cost_breakdown.csv。独立求解时间不能直接与GPU恢复成本抵消。不得从剔除审计成本后的数字声称轻量生产E2E净收益。所有5对或失败臂保留，未按性能筛选。

本数据仅为一个固定四Session受控组合中单Pending C的成对实验，重复运行不是多个独立Workflow。即使实际恢复前沿更深且某个计时区间更短，也不能推广为四Pending全局吞吐或生产性能优势；首次编译、事件同步、重型审计和环境波动限制见measurement_validity.md。

## 完整性、失败与清理

冻结来源及执行锁变化：{changed}。清理：{json.dumps({k:v['status'] for k,v in cleanup.items()},ensure_ascii=False)}。
失败与异常：{json.dumps(failures,ensure_ascii=False)}。原始运行日志保留所有警告；不只依靠进程返回码判断成功。未修改生产算法、Phi、Compatibility及历史冻结证据。任务结束，不扩大工作负载、不进入下一阶段。
''')
 save('validation.json',{'gate':gate,'valid_pairs':sum(v['status']=='PASS' for v in validations.values()),'changed':changed,'failures':failures,'cleanup_pass':clean})
 save('evidence_manifest.json',{'files':{str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and not any(x.startswith('triton_cache') or x=='__pycache__' for x in p.parts) and p.name!='evidence_manifest.json'}})
 print('最终门禁',gate)
if __name__=='__main__':main()
