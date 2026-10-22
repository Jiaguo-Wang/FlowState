"""汇总全部尝试、每对差值及独立审计边界，不按性能筛选。"""
import csv,json,statistics
from pathlib import Path
from evaluation.native_resume_correctness_e4_phase1a.prepare import sha
from .review import read,check,ref_signature
P=Path(__file__).resolve().parent

def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(n,rows,fields):
 with (P/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 rows=[];checks=[];failures=[];stages=[];base=None
 cleanup=read(P/'gpu_cleanup.json') if (P/'gpu_cleanup.json').exists() else {}
 protocol=read(P/'experiment_protocol.json')['order']
 for item in protocol:
  name=item['name'];d=P/name
  if (d/'failure.json').exists():failures.append({'run':name,'failure':read(d/'failure.json')})
  if item['phase']=='cold' or not (d/'summary.json').exists():continue
  try:
   v=check(P,name,item['pending'].split('_')[1]+'_Cold',item['pending'],item['phase']=='smoke')
   current=ref_signature(read(d/'reference_audit.json'))
   if base is None:base=current
   assert base==current,'初始状态不可比'
   checks.append(v)
  except Exception as exc:
   failures.append({'run':name,'error':repr(exc)});continue
  if item['phase']=='smoke':continue
  s=read(d/'summary.json');r=read(d/'reconciliation.json');tr=read(d/'post_request.json')['trace'];stage={x['stage']:x['ms'] for x in r['timing']};a=s['native_match']
  row={'run':name,'pending':item['pending'],'pair':item['pair'],'solver':item['solver'],'H':a['physical_fa_hit'],'E':a['executable_prefix'],'G':a['replay_gap'],'actual_prefill_tokens':v['actual_prefill_tokens'],'selected':';'.join(r['plan']['selected_ids']),'J_model_ms':r['plan']['recovery_cost_ms'],'pending_model_ms':r['target']['expected']['cost_ms'],
   'observation_inline_ms':stage['metadata_observation'],'heavy_reference_rpc_ms':s['reference_observation_rpc_wall_ms'],'epoch_authentication_ms':stage['strict_epoch_authentication'],'benefit_matrix_ms':stage['benefit_matrix'],'solve_ms':stage['solver'],'plan_validation_ms':stage['plan_build_validate'],'apply_check_ms':stage['apply_time_freshness_preflight'],'reconciliation_ms':stage['strict_reconciliation_with_postchecks'],'control_rpc_ms':s['control_rpc_wall_ms'],
   'prefill_cuda_ms':sum(x['cuda_elapsed_ms'] for x in tr['compute'] if x['is_extend']),'cow_cuda_ms':sum(x['cuda_elapsed_ms'] for x in tr['copies']),'ttft_ms':s['ttft_ms'],'request_wall_ms':s['request_wall_ms'],'control_to_completion_ms':s['control_through_completion_ms'],'audit_inclusive_completion_ms':s['audit_inclusive_through_completion_ms']}
  rows.append(row)
  for x in r['timing']:stages.append({'run':name,'stage':x['stage'],'ms':x['ms'],'scope':'控制RPC内子区间，不与父区间相加'})
  for i,x in enumerate(r['operations']):stages.append({'run':name,'stage':f'native_adapter_{i}','ms':(x['end_ns']-x['start_ns'])/1e6,'scope':'协调内嵌区间，含Adapter安全检查'})
 if (P/'launch_failure.json').exists():failures.append(read(P/'launch_failure.json'))
 source=read(P/'source_identity.json');lock=read(P/'execution_lock.json') if (P/'execution_lock.json').exists() else {'sha256':{}}
 changed=[p for p,h in {**source['sha256'],**lock['sha256']}.items() if sha(p)!=h]
 good=not failures and not changed and all(x['status']=='PASS' for x in cleanup.values())
 complete=len(rows)==12 and len(checks)==16 and len(cleanup)==18
 gate='READY' if good and complete else 'BLOCKED' if not good else 'PARTIAL'
 fields=list(rows[0]) if rows else ['run','pending','solver'];table('gpu_raw_timings.csv',rows,fields)
 table('control_cost_breakdown.csv',stages,['run','stage','ms','scope'])
 metrics=[k for k in fields if k.endswith('_ms') and 'model' not in k]
 differences=[];summaries=[]
 for pending in ['IMW1_A_TURN_003','IMW1_C_TURN_003']:
  for pair in range(1,4):
   rr={r['solver']:r for r in rows if r['pending']==pending and r['pair']==pair}
   if len(rr)!=2:continue
   for m in metrics:differences.append({'pending':pending,'pair':pair,'metric':m,'greedy_ms':rr['Greedy'][m],'exact_ms':rr['Exact'][m],'exact_minus_greedy_ms':rr['Exact'][m]-rr['Greedy'][m]})
  for solver in ['Greedy','Exact']:
   for m in metrics:
    vals=[r[m] for r in rows if r['pending']==pending and r['solver']==solver]
    if vals:summaries.append({'pending':pending,'solver':solver,'metric':m,'n':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals),'min':min(vals),'max':max(vals)})
 table('paired_differences.csv',differences,['pending','pair','metric','greedy_ms','exact_ms','exact_minus_greedy_ms'])
 table('timing_summary.csv',summaries,['pending','solver','metric','n','mean','median','min','max'])
 save('validation.json',{'gate':gate,'complete':complete,'checks':checks,'failures':failures,'changed':changed,'cleanup':cleanup})
 lines=['# E4-Phase 1B：w1/K2轻量恢复性能Smoke',f'\n**Gate：{gate}**。有效测量臂{len(rows)}/12，已验证严格恢复（含Smoke）{len(checks)}/16，清理记录{len(cleanup)}/18。',
 '\n只使用固定w1/A与C，四个轻量Smoke通过后才测量。每个臂独立Engine、八条相同历史；每个Pending独立Cold，temperature=0/max_new_tokens=8/ignore_eos=True。没有复用被其他Pending污染的状态。',
 '\n## 实际恢复与计算区间', '\n|Pending|策略|H/E/G|实际Prefill Tokens|', '|---|---|---|---:|']
 for pending in ['IMW1_A_TURN_003','IMW1_C_TURN_003']:
  for solver in ['Greedy','Exact']:
   rr=[r for r in rows if r['pending']==pending and r['solver']==solver]
   if rr:lines.append(f"|{pending}|{solver}|{sorted(set((r['H'],r['E'],r['G']) for r in rr))}|{sorted(set(r['actual_prefill_tokens'] for r in rr))}|")
 lines+=['\n## 三对实测结果','\n所有值为毫秒；差值为Exact减Greedy，负值表示Exact较短。均值、中位数、范围及全部逐次原始数据分别见timing_summary.csv、gpu_raw_timings.csv、paired_differences.csv。','\n|Pending|指标|Greedy均值|Exact均值|配对差值均值|Exact较短/相同/较长|','|---|---|---:|---:|---:|---|']
 for pending in ['IMW1_A_TURN_003','IMW1_C_TURN_003']:
  for m in ['observation_inline_ms','solve_ms','reconciliation_ms','control_rpc_ms','prefill_cuda_ms','ttft_ms','control_to_completion_ms','audit_inclusive_completion_ms']:
   ds=[d for d in differences if d['pending']==pending and d['metric']==m]
   if ds:
    vals=[d['exact_minus_greedy_ms'] for d in ds]
    lines.append(f"|{pending}|{m}|{statistics.mean(d['greedy_ms'] for d in ds):.3f}|{statistics.mean(d['exact_ms'] for d in ds):.3f}|{statistics.mean(vals):.3f}|{sum(v<0 for v in vals)}/{sum(v==0 for v in vals)}/{sum(v>0 for v in vals)}|")
 lines+=['\n## 方向性判断','\nA的三对Prefill CUDA与TTFT均改善，支持本次更深Native前沿能够减少实际恢复计算。A的Control RPC三对均更慢；完整Control-to-completion与Audit-inclusive各仅2/3对改善，不能证明控制后的稳定净收益。C保持相同前沿和Prefill Tokens，时间变化属于负对照波动，不是分配恢复收益。','\n当前Control RPC包含客户端请求/响应证据写盘与fsync，不能解释为纯生产控制成本。前置全内容审计及请求后对拍分别保存；完整路径没有静默扣除这些区间内实际发生的审计工作。','\n## 正确性和结论边界','\n有效臂均要求当前Strict Epoch与冻结输入及重型参考一致、Apply-time同Scheduler调用、5项真实驱逐、最终恰为计划选集、FA与非目标元数据不变、实际Native来源匹配、COW返回、8 Token与独立Cold精确一致、请求后保留组件及受保护FA内容不变。四个Smoke还执行控制后全内容对拍与COW源/目标内容检查。性能臂不在关键路径逐槽哈希，保留原E1安全检查及请求后内容审计。',
 '\nA用于检验更深恢复位置，C为相同恢复位置负对照；C的任何时间变化不能归因于前沿改善。求解目标J为四Pending模型值，不能直接解释本次单请求CUDA或Wall差异。PROVENANCE_UNVERIFIED保持不变。',
 '\n必须分别解释CUDA计算、客户端TTFT、控制到完成及含前置审计的完成时间。三对只提供受控方向性描述，不能证明稳定生产净收益、P95或显著性。K=2是逻辑预算，不是HBM容量压力；未扩展其他工作负载。',
 '\n计时的父子区间与审计范围见measurement_validity.md。已保留全部尝试和失败；若未完成，不补造缺失数据。完成后停止。']
 (P/'final_report.md').write_text('\n'.join(lines)+'\n')
 files={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and p.name!='evidence_manifest.json' and 'triton_cache_' not in str(p)}
 save('evidence_manifest.json',{'files':files,'gate':gate})
 print(gate, len(rows), '有效测量臂',flush=True)
if __name__=='__main__':main()
