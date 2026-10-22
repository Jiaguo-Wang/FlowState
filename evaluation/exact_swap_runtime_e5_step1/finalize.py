"""汇总固定三组四Pending等权和，保留所有失败及控制成本。"""
import csv,json,statistics
from pathlib import Path
from evaluation.native_resume_correctness_e4_phase1a.prepare import sha
from .review import read,check,ref_signature
P=Path(__file__).resolve().parent

def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(n,rows,fields=None):
 fields=fields or (list(rows[0]) if rows else ['run'])
 with (P/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 protocol=read(P/'experiment_protocol.json')['order'];cleanup=read(P/'gpu_cleanup.json') if (P/'gpu_cleanup.json').exists() else {};rows=[];checks=[];failures=[];base=None;stages=[]
 for item in protocol:
  name=item['name'];d=P/name
  if (d/'failure.json').exists():failures.append({'run':name,'failure':read(d/'failure.json')})
  if item['phase']=='cold' or not (d/'summary.json').exists():continue
  try:
   v=check(P,name,item['pending'].split('_')[1]+'_Cold',item['pending'],item['phase']=='smoke')
   signature=ref_signature(read(d/'reference_audit.json'))
   if base is None:base=signature
   assert signature==base,'初始状态不可比'
   checks.append(dict(v,phase=item['phase']))
  except Exception as exc:failures.append({'run':name,'error':repr(exc)});continue
  if item['phase']=='smoke':continue
  s=read(d/'summary.json');r=read(d/'reconciliation.json');tr=read(d/'post_request.json')['trace'];a=s['native_match'];stage={x['stage']:x['ms'] for x in r['timing']}
  rows.append({'run':name,'pair':item['pair'],'pending':item['pending'],'solver':item['solver'],'H':a['physical_fa_hit'],'E':a['executable_prefix'],'G':a['replay_gap'],'actual_prefill_tokens':v['actual_prefill_tokens'],'selected':';'.join(r['plan']['selected_ids']),'model_J_ms':r['plan']['recovery_cost_ms'],'observation_ms':stage['metadata_observation'],'epoch_ms':stage['strict_epoch_authentication'],'matrix_ms':stage['benefit_matrix'],'solve_or_frozen_selection_ms':stage['solver'],'plan_validate_ms':stage['plan_build_validate'],'apply_ms':stage['apply_time_freshness_preflight'],'reconciliation_ms':stage['strict_reconciliation_with_postchecks'],'control_rpc_ms':s['control_rpc_wall_ms'],'reference_rpc_ms':s['reference_observation_rpc_wall_ms'],'prefill_cuda_ms':sum(x['cuda_elapsed_ms'] for x in tr['compute'] if x['is_extend']),'ttft_ms':s['ttft_ms'],'request_wall_ms':s['request_wall_ms'],'control_to_completion_ms':s['control_through_completion_ms'],'audit_inclusive_completion_ms':s['audit_inclusive_through_completion_ms']})
  for x in r['timing']:stages.append({'run':name,'stage':x['stage'],'ms':x['ms'],'scope':'控制RPC子区间；不可重复相加'})
  for i,x in enumerate(r['operations']):stages.append({'run':name,'stage':f'native_adapter_{i}','ms':(x['end_ns']-x['start_ns'])/1e6,'scope':'协调内部Native接口与检查'})
 if (P/'launch_failure.json').exists():failures.append(read(P/'launch_failure.json'))
 source=read(P/'source_identity.json');lock=read(P/'execution_lock.json') if (P/'execution_lock.json').exists() else {'sha256':{}}
 changed=[p for p,h in {**source['sha256'],**lock['sha256']}.items() if sha(p)!=h]
 good=not failures and not changed and all(c['status']=='PASS' for c in cleanup.values())
 complete=len(rows)==24 and len(checks)==32 and len(cleanup)==36
 gate='READY' if good and complete else 'BLOCKED' if not good else 'PARTIAL'
 table('gpu_raw_timings.csv',rows);table('control_cost_breakdown.csv',stages)
 metrics=['prefill_cuda_ms','ttft_ms','control_rpc_ms','request_wall_ms','control_to_completion_ms','audit_inclusive_completion_ms'];diff=[];totals=[];summary=[]
 for pair in range(1,4):
  for label in 'ABCD':
   rr={r['solver']:r for r in rows if r['pair']==pair and r['pending']==f'IMW1_{label}_TURN_003'}
   if len(rr)!=2:continue
   for m in metrics:diff.append({'pair':pair,'pending':label,'metric':m,'exact_ms':rr['Exact'][m],'swap_ms':rr['Swap'][m],'exact_minus_swap_ms':rr['Exact'][m]-rr['Swap'][m]})
  rr=[r for r in rows if r['pair']==pair]
  if len(rr)==8:
   for m in metrics:
    exact=sum(r[m] for r in rr if r['solver']=='Exact');swap=sum(r[m] for r in rr if r['solver']=='Swap');totals.append({'pair':pair,'metric':m,'exact_equal_weight_sum_ms':exact,'swap_equal_weight_sum_ms':swap,'exact_minus_swap_ms':exact-swap})
 for label in 'ABCD':
  for solver in ('Exact','Swap'):
   for m in metrics:
    vals=[r[m] for r in rows if r['solver']==solver and r['pending']==f'IMW1_{label}_TURN_003']
    if vals:summary.append({'pending':label,'solver':solver,'metric':m,'n':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals),'min':min(vals),'max':max(vals)})
 table('pending_pair_differences.csv',diff);table('four_pending_paired_sums.csv',totals);table('timing_summary.csv',summary)
 save('validation.json',{'gate':gate,'complete':complete,'checks':checks,'failures':failures,'changed_sources':changed,'cleanup':cleanup})
 primary=[r for r in totals if r['metric']=='prefill_cuda_ms'];verdict='样本不完整，不能回答主要性能问题'
 if len(primary)==3:
  differences=[r['exact_minus_swap_ms'] for r in primary]
  if all(v<0 for v in differences):verdict='Exact的四Pending CUDA Prefill等权总和在三组均更低；支持本固定实例和测量合同下的有限联合恢复收益，不证明稳定生产优势或显著性。'
  elif all(v>0 for v in differences):verdict='1-Swap的四Pending CUDA Prefill等权总和在三组均更低；本实例不支持Exact的实际联合计算收益，保留负面结果，不修改Phi或实例。'
  else:verdict='三组差值方向不一致，当前重复中的波动使方向结论不稳；未证明Exact具有实际联合恢复收益，也不将混合符号当作统计等价证明。'
 lines=['# E5-Step 1：Exact与固定1-Swap真实恢复取舍',f'\n**Gate：{gate}**。Gate 0为59项CPU测试通过；已认证Smoke {sum(v["phase"]=="smoke" for v in checks)}/8、正式测量 {len(rows)}/24、生命周期清理 {len(cleanup)}/36。', '\n'+verdict,
 '\n## 冻结与正确性边界','\n固定w1/K3。Exact {N18,N25,N27}，模型J=21.953374716；实验冻结首改善1-Swap {N21,N22,N24}，J=31.817384152。历史ID只对齐完整Token前缀，当前Node/Slot由每个Engine重新Observe/Bind/Preflight。1-Swap不称生产求解器，其计时项为冻结选择读取/映射，不是假定在线重算Swap耗时。',
 '\n全部有效臂要求当前严格矩阵/Plan不变、连续Scheduler控制、4项真实Recurrent-only驱逐、恰好3项保留、Native Match来源符合Plan、COW工作槽与保留状态隔离、8Token与独立Cold精确一致、FA及保留内容合同通过。Smoke额外检查控制后全内容及COW复制内容；正式轻量臂延续E4合同及请求后独立内容验证。PROVENANCE_UNVERIFIED不升级。',
 '\n|Pending|策略|实际H/E/G|实际Prefill Tokens|','|---|---|---|---|']
 for label in 'ABCD':
  for solver in ('Exact','Swap'):
   rr=[r for r in rows if r['pending']==f'IMW1_{label}_TURN_003' and r['solver']==solver]
   if rr:lines.append(f"|{label}|{solver}|{sorted(set((r['H'],r['E'],r['G']) for r in rr))}|{sorted(set(r['actual_prefill_tokens'] for r in rr))}|")
 lines+=['\n## 主要指标：四次独立恢复的等权和','\n|配对组|Exact CUDA Prefill总和ms|1-Swap总和ms|Exact−1-Swap ms|','|---|---:|---:|---:|']
 for r in primary:lines.append(f"|{r['pair']}|{r['exact_equal_weight_sum_ms']:.3f}|{r['swap_equal_weight_sum_ms']:.3f}|{r['exact_minus_swap_ms']:.3f}|")
 lines+=['\n## 单请求三次均值与控制路径','\n|Pending|指标ms|Exact|1-Swap|','|---|---|---:|---:|']
 for label in 'ABCD':
  for m in metrics:
   values={r['solver']:r['mean'] for r in summary if r['pending']==label and r['metric']==m}
   if len(values)==2:lines.append(f"|{label}|{m}|{values['Exact']:.3f}|{values['Swap']:.3f}|")
 lines+=['\n完整逐次、逐Pending差值、各组等权和、均值/中位数/范围见对应CSV。控制及完整路径和也是四次独立生命周期测量的描述性加总，不是一个自然四请求服务周期，更不是吞吐。不能只报告C/D收益而略去A/B代价。',
 '\n## 测量合同与不能证明的结论','\n沿用E4-1B客户端写盘、fsync及原生同流CUDA事件位置，不重构计时，不删首次Decode、不单侧预热。Control RPC包含客户端证据持久化；观察/认证/求解/协调为其子区间。Control-to-completion从控制RPC发起到目标流结束，Audit-inclusive另包含前置重型参考；请求后全内容检查和清理位于结束端点之外。CUDA事件含提交空隙，不称纯内核和。主要Prefill测量位于Native COW之后的原Eager路径，Smoke不纳入性能样本。',
 '\n三组是固定受控组合的重复；无统计显著性或生产P95结论。模型J差距不能证明CUDA优劣，G=0也不代表完整输入无需Prefill。逻辑K3不等于GPU容量压力。未扩展来源、未改变生产求解器/Phi/Strict合同。若失败，详情见launch_failure.json和validation.json，不自动重试。',
 '\n本轮完成即停止，不扩展到其他工作负载或性能优化。']
 (P/'final_report.md').write_text('\n'.join(lines)+'\n')
 save('evidence_manifest.json',{'gate':gate,'files':{str(f.relative_to(P)):sha(f) for f in P.rglob('*') if f.is_file() and f.name!='evidence_manifest.json' and 'triton_cache_' not in str(f)}})
 print(gate,len(rows),verdict,flush=True)
if __name__=='__main__':main()
