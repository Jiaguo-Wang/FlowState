"""汇总实际原生恢复证据；未执行项明确留空，不以预测替代Runtime结果。"""
import csv,json
from pathlib import Path
from evaluation.native_resume_correctness_e4_phase1a.prepare import read,sha,write
OUT=Path(__file__).resolve().parent

def main():
 inputs=read(OUT/'prepared_inputs.json');frozen=read(OUT/'frozen_snapshot.json');plans=read(OUT/'frozen_plans.json');cleanup=read(OUT/'gpu_cleanup.json') if (OUT/'gpu_cleanup.json').exists() else {}
 old={tuple(c['token_ids']):c['checkpoint_id'] for c in frozen['candidates']};rows=[];details=[];cold=[]
 for p in inputs['pendings']:
  label=p['label'];pid=p['continuation_id'];cn=label+'_Cold'
  cold.append({'pending':pid,'run':cn,'status':read(OUT/cn/'summary.json')['status'] if (OUT/cn/'summary.json').exists() else 'NOT_COMPLETED','cleanup':cleanup.get(cn,{}).get('status','NOT_EXECUTED')})
  for solver in ('Greedy','Exact'):
   name=label+'_'+solver;path=OUT/name;vfile=OUT/f'{name}_validation.json';pred=next(x for x in plans[solver]['per_pending'] if x['continuation_id']==pid)
   row={'pending':pid,'solver':solver,'gate':'NOT_COMPLETED','logical_selected':plans[solver]['selected_ids'],'model_J_ms':plans[solver]['J_ms'],'model_T':pred['T'],'model_E':pred['E'],'model_G':pred['G'],'actual_H':None,'actual_E':None,'actual_G':None,'source_current_node':None,'source_current_slots':None,'working_slots':None,'evictions':None,'reconciliation':'NOT_VALIDATED','resume':'NOT_VALIDATED','cold_output':'NOT_VALIDATED','cleanup':cleanup.get(name,{}).get('status','NOT_EXECUTED'),'provenance':'PROVENANCE_UNVERIFIED'}
   if vfile.exists():
    v=read(vfile);r=read(path/'reconciliation.json');post=read(path/'post_request.json');summary=read(path/'summary.json');trace=post['trace'];native=summary['native_match'];raw=r['raw_epoch']
    mapping={c['checkpoint_id']:old[tuple(c['token_ids'])] for c in raw['candidates']}
    assert set(mapping[c] for c in r['plan']['selected_ids'])==set(plans[solver]['selected_ids'])
    assert abs(r['plan']['recovery_cost_ms']-plans[solver]['J_ms'])<=1e-9
    resident={n['node_id'] for n in r['after']['nodes'] if n['slots']};keep={c['node']['node_id'] for c in r['reconciliation']['binding']['candidates'] if c['candidate_id'] in r['plan']['selected_ids']}
    assert resident==keep and len(keep)==2 and len(r['operations'])==5
    assert native['executable_prefix']==pred['E'] and native['physical_fa_hit']==pred['T'] and native['replay_gap']==pred['G']
    assert r['selected_payload_before']==r['selected_payload_after']==post['state']['selected_payload']
    assert r['fa_content_before']==r['fa_content_after'] and read(path/'post_control_audit.json')['ok']
    assert len(trace['admission'])==1 and len(trace['execution'])==8 and all(x['native_run_returned'] for x in trace['execution'])
    assert all(c['returned'] and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in trace['copies'])
    assert v['output']['exact_token_match'] and row['cleanup']=='PASS'
    row.update(gate='READY',actual_H=native['physical_fa_hit'],actual_E=native['executable_prefix'],actual_G=native['replay_gap'],source_current_node=native['best_match_node'],source_current_slots=native['cow_source_slots'],working_slots=native['working_slots'],evictions=len(r['operations']),reconciliation='RECONCILIATION_VALIDATED',resume='RESUME_VALIDATED',cold_output='EXACT_8_TOKEN_MATCH')
    details.append({'run':name,'source_validation':read(path/'native_source_validation.json'),'output':v['output'],'native':native,'copies':trace['copies'],'selected_current':r['plan']['selected_ids'],'selected_frozen':[mapping[c] for c in r['plan']['selected_ids']],'resident_after_control':sorted(resident),'free_slots_before':len(r['before']['free_slots']),'free_slots_after':len(r['after']['free_slots']),'FA_control_content_unchanged':True,'selected_payload_unchanged_after_request':True,'non_recomputed_FA_unchanged':post['state']['fa_protected_payload']==r['fa_protected_payload'],'execution_records':len(trace['execution']),'artifact_directory':str(path)})
   elif (path/'failure.json').exists():row['gate']='BLOCKED';row['failure']=read(path/'failure.json')['error']
   rows.append(row)
 # 所有行保留一致列，包括未执行和失败。
 columns=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/'per_pending_results.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
 write(OUT/'resume_correctness.json',{'rows':rows,'details':details,'cold':cold})
 failed=(OUT/'launch_failure.json').exists() or any(v['status']!='PASS' for v in cleanup.values())
 gate='BLOCKED' if failed else 'READY' if len(details)==8 and len(cleanup)==12 and all(c['status']=='PASS' for c in cold) else 'PARTIAL'
 report=['# E4-Phase 1A：w1/K2独立来源Native Resume门禁','',f'**Gate：{gate}**。本轮仅做正确性验证，不报告性能优劣。','',f'实际认证{len(details)}/8个策略—Pending恢复；Cold完成{sum(c["status"]=="PASS" for c in cold)}/4；生命周期清理记录{len(cleanup)}份。','', '旧BLOCKED目录完整保留。本次只改变输出目录，复用已通过10项新增CPU测试及54项Strict Epoch/Control测试的入口。每个Pending使用独立Cold、Greedy、Exact生命周期，严格臂各重放八条相同历史；没有在被前一请求污染的状态上验证后续Pending。','','## 冻结逻辑选集与实际执行范围','','| 策略 | 冻结候选 | Model J ms |','|---|---|---:|']
 for solver,p in plans.items():report.append(f'| {solver} | {p["selected_ids"]} | {p["J_ms"]:.9f} |')
 report+=['','Greedy+1-Swap在本条件与Exact同集，因此没有重复物理Arm。上述ID仅用于标识冻结候选；每个新生命周期按完整Token前缀双射重新映射，当前Node/Slot/组件由真实Observe、Bind、Preflight和Apply-time检查认证。','','## 逐Pending真实结果','','| Pending | 策略 | 模型T/E/G | 实际H/E/G | 当前来源Node/Slot | 工作Slot | Gate |','|---|---|---|---|---|---|---|']
 for r in rows:report.append(f'| {r["pending"]} | {r["solver"]} | {r["model_T"]}/{r["model_E"]}/{r["model_G"]} | {r["actual_H"]}/{r["actual_E"]}/{r["actual_G"]} | {r["source_current_node"]}/{r["source_current_slots"]} | {r["working_slots"]} | {r["gate"]} |')
 report+=['','未完成项不以模型预测填充Runtime值，None表示没有相应认证。完整输出Token与Cold逐Token对照见resume_correctness.json；全部原始Match、Admission、COW、执行和状态摘要保留在各Arm子目录。','','## 安全合同与证据等级','','已通过行的认证要求：单Scheduler同步控制窗口内重建/认证当前Epoch并确认冻结计划；Bind、Preflight和Apply-time内容/身份一致；正式StateController.reconcile_strict执行5项Recurrent-only Eviction；控制后7→2个候选驻留、空闲槽21→26；保留的Conv/Temporal内容不变；全部原FA索引与内容不变。请求实际Admission再次检查状态，不能把前一控制窗口称作跨RPC租约。','','Native实际匹配必须等于当前物理绑定且H/E符合计划；原生COW源/目的切片隔离，源前后与目的复制后内容摘要一致；实际Prefill/Decode完成，8个确定性Token及停止行为与独立Cold完全一致；请求后保留状态内容及非重算FA仍受保护。正常请求会重算E之后的目标FA并创建新状态，不能机械要求请求后整棵树不变。','','- EPOCH_ADMISSIBLE：当前完整严格准入合同，不单独证明实际恢复。\n- RECONCILIATION_VALIDATED：仅对应真实物理控制及后置检查已通过的Arm。\n- RESUME_VALIDATED：仅对应Native来源/COW/执行/Cold输出及状态保护均通过的Pending。\n- PROVENANCE_UNVERIFIED：全部维持原等级；Node匹配和Token共享不认证张量跨Workflow血缘。','','## 资源、完整性和结论边界','','每个生命周期启动前检查GPU0无其他计算任务；只清理本轮专用容器，保留逐次退出及显存/进程记录。CPU协议、输入、模型和Runtime源码身份逐项核对，不改生产算法、Phi、严格兼容矩阵或旧实验。日志可能包含既有CUDA事件与Wall计时，但本轮不据此进行性能比较，不将modelled ms解释为GPU时延；K=2仍是逻辑保留预算，不是HBM压力。']
 if failed:report+=['','## 停止原因','',json.dumps(read(OUT/'launch_failure.json') if (OUT/'launch_failure.json').exists() else cleanup,ensure_ascii=False,indent=2),'','出现门禁失败后未自动重试或修复Runtime。']
 report+=['','本轮完成后停止，不扩展w2–w4，不进入E4-Phase 1B或性能实验。']
 (OUT/'final_report.md').write_text('\n'.join(report)+'\n')
 source=read(OUT/'source_identity.json');assert all(sha(p)==h for p,h in source['sha256'].items())
 lock=read(OUT/'execution_lock.json');assert all(sha(p)==h for p,h in lock['sha256'].items())
 write(OUT/'final_integrity.json',{'status':'PASS','source_files_unchanged':len(source['sha256']),'execution_code_inputs_unchanged':True,'previous_blocked_preserved':True})
 write(OUT/'evidence_manifest.json',{'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and not any(q.startswith('triton_cache') or q=='__pycache__' for q in p.parts)}})
 print(gate,len(details),'个恢复认证')
if __name__=='__main__':main()
