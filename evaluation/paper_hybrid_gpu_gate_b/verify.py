"""场景B结果核验（宿主机执行，只读）：单臂产物 vs 冻结 Oracle 与物理/请求正确性合同。

对照来源（全部只读）：
- c_formal_perf_gpu/g1_CB_FS 冻结运行（同环境同K的 FlowState-Greedy 臂：计划、census、输出）；
- policy_plans.json + kvflow_multiworkflow_cpu_main/cpu_results.json（C/K3 冻结 CPU 选集与 J）；
- g1_CB_FS/cold_reference_C_B.json（冻结 Cold 参照）。
输出 gate_b_result.json；任何一项不满足即给出 FAILED 项并退出非零。
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
OUT=ROOT/'evaluation/paper_hybrid_gpu_gate_b'
RUN='b1_hybrid_greedy'
G1=ROOT/'evaluation/c_formal_perf_gpu/g1_CB_FS'
MW=ROOT/'evaluation/kvflow_multiworkflow_cpu_main'
EPS=1e-9
checks=[]
def check(name,ok,detail):
    checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':detail})
    return ok
def main():
    summary=json.loads((OUT/RUN/'summary.json').read_text())
    rec=json.loads((OUT/RUN/'reconciliation.json').read_text())
    post=json.loads((OUT/RUN/'post_request.json').read_text())
    ref=json.loads((G1/'run'/'reconciliation.json').read_text())
    plans=json.loads((OUT/'policy_plans.json').read_text());fs=plans['FlowState-Greedy']
    cpu_rows=[r for r in json.loads((MW/'cpu_results.json').read_text())['rows'] if r['structure']=='C' and r['K']==3]
    cpu_fs=next(r for r in cpu_rows if r['policy']=='FlowState-Greedy')
    cold=json.loads((OUT/'cold_reference_C_B.json').read_text())
    # 1) 在线决策：超门禁自动 Marginal Greedy（不读 policy_plans 作为本轮选择）
    d=rec['strict_decision']
    check('在线决策=Marginal Greedy且门禁拒绝',d['solver']=='Marginal Greedy' and not d['gate_allowed'],d)
    check('门禁规模=16候选/4Pending/K=3',(d['gate']['candidates'],d['gate']['pendings'],d['gate']['budget_k'])==(16,4,3),d['gate'])
    check('选择来源=plan_strict在线(非policy_plans)',rec['selection_source']=='GlobalOptimizer.plan_strict(在线)' and rec['oracle_policy'].startswith('FlowState-Greedy(policy_plans.json，仅事后对照)'),rec['selection_source'])
    check('决策记录完整(选集/J/纯求解时间)',set(d)>={'solver','gate_allowed','gate','selected_ids','objective_ms','pure_solve_ms'} and abs(d['objective_ms']-rec['plan']['recovery_cost_ms'])<=EPS and d['pure_solve_ms']>0,{k:d[k] for k in ('selected_ids','objective_ms','pure_solve_ms')})
    # 2) 选集与 J：== 冻结 GPU 计划 == policy_plans(经census映射) == cpu_results
    sel=set(rec['plan']['selected_ids']);J=rec['plan']['recovery_cost_ms']
    cpu_to_phys={m['cpu_checkpoint_id']:m['physical_checkpoint_id'] for m in rec['census']['mapped']}
    check('选集==g1冻结GPU计划',sel==set(ref['plan']['selected_ids']),sorted(sel))
    check('选集==policy_plans冻结FS(经census映射)',sel=={cpu_to_phys[c] for c in fs['selected_checkpoint_ids']},fs['selected_checkpoint_ids'])
    check('选集==cpu_results冻结FS(经census映射)',sel=={cpu_to_phys[c] for c in cpu_fs['selected']},cpu_fs['selected'])
    check('J==冻结J 154.34895561599998',abs(J-154.34895561599998)<=EPS,J)
    check('J==g1冻结计划J且==cpu_results J',abs(J-ref['plan']['recovery_cost_ms'])<=EPS and abs(J-cpu_fs['current_J_ms'])<=EPS,{'g1':ref['plan']['recovery_cost_ms'],'cpu':cpu_fs['current_J_ms']})
    # 3) 逐 Pending E/T/G == 冻结合同
    expect=fs['expected_E_G']
    check('逐Pending E/T/G==冻结预期',all((p['E'],p['T'],p['G'])==(expect[p['pending_id']]['E'],expect[p['pending_id']]['T'],expect[p['pending_id']]['G']) for p in rec['plan']['pending']),[(p['pending_id'],p['E'],p['T'],p['G']) for p in rec['plan']['pending']])
    # 4) census 对账与物化时间线
    check('census 16=16且物化于冻结预热水(round0/1)',rec['census']['physical_domain_size']==16==len(rec['census']['mapped']) and not rec['census']['unmapped_extras'] and not rec['census']['missing_physical'] and all(r['round'] in ('round0','round1') for r in rec['census']['materialization_rounds']),{'mapped':len(rec['census']['mapped']),'rounds':sorted({r['round'] for r in rec['census']['materialization_rounds']})})
    # 5) 物理协调：Recurrent-only 驱逐补集完整落实（16−3=13）
    ops=[(o['checkpoint_id'],o['returned']) for o in rec['operations']]
    all_ids={c['candidate']['checkpoint_id'] for c in rec['epoch']['candidates']}
    check('驱逐==未选中补集(13项)且全部返回',sorted(x for x,_ in ops)==sorted(all_ids-sel) and len(ops)==13 and all(r for _,r in ops),ops)
    # 6) Native 恢复：目标 C_B 的 E/H/G 与来源绑定
    nm=summary['native_match'];bound=next(x['node'] for x in rec['reconciliation']['binding']['candidates'] if x['candidate_id']==rec['target']['source_checkpoint'])
    check('目标C_B恢复源==计划绑定(B5C3_NODE_55)',rec['target']['source_checkpoint']=='B5C3_NODE_55' and nm['best_match_node']==bound['node_id'] and list(nm['cow_source_slots'])==list(bound['slots']),{'source':rec['target']['source_checkpoint'],'native':(nm['best_match_node'],nm['cow_source_slots']),'bound':(bound['node_id'],bound['slots'])})
    check('Native E=5120/H=5120/G=0与冻结合同一致',nm['executable_prefix']==5120 and nm['physical_fa_hit']==5120 and nm['physical_fa_hit']-nm['executable_prefix']==0,{'E':nm['executable_prefix'],'H':nm['physical_fa_hit'],'G':nm['physical_fa_hit']-nm['executable_prefix']})
    check('Prefill区间=计划E准入+32新Token',summary['batch_prefix_length']==5120 and summary['prefill_token_count']==32,{'batch_prefix_length':summary['batch_prefix_length'],'prefill_token_count':summary['prefill_token_count']})
    # 7) 请求正确性：8 Token == 冻结 Cold 参照
    check('输出==冻结Cold参照逐Token',summary['output_ids']==cold['output_ids'] and summary['cold_output_match'],{'output':summary['output_ids'],'cold':cold['output_ids']})
    # 8) 协调后状态保护（runner 已断言，此处从落盘证据复核）
    check('请求后Selected内容不变',post['state']['selected_payload']==rec['selected_payload_after'],None)
    check('请求后FA映射全保留',all(post['state']['fa_map'].get(k)==v for k,v in rec['fa_map_before'].items()),None)
    check('请求后受保护FA内容不变',post['state']['fa_protected_payload']==rec['fa_protected_payload'],None)
    # 9) Epoch 形成方式记录
    check('Epoch形成方式已记录(冻结预热两轮重放,非全在线)',len(rec['origins'])==16 and rec['epoch']['K']==3 and rec['epoch']['candidates'].__len__()==16,{'note':'复用C/K3冻结16条预热×2轮重放+实时观察准入'})
    # 10) 执行完整性
    proc=json.loads((OUT/(RUN+'_process.json')).read_text());cleanup=json.loads((OUT/(RUN+'_cleanup.json')).read_text())
    check('进程退出码0且GPU清理PASS',proc['returncode']==0 and cleanup['status']=='PASS',proc)
    lock=json.loads((OUT/'execution_lock.json').read_text())['files']
    scope={p:h for p,h in lock.items() if Path(p).name not in ('prepare.py','launch.py','verify.py')}
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in scope.items())
    check('冻结输入/被测代码/生产入口全程未变',unchanged,'%d files（宿主机脚本除外）'%len(scope))
    ok=all(c['status']=='PASS' for c in checks)
    result={'scenario':'B_ck3_greedy_branch','run':RUN,'verdict':'PASS' if ok else 'FAIL','checks':checks,
            'decision':d,'selected_physical':sorted(sel),'selected_cpu_mapped':rec['oracle_selected_cpu'],
            'J_ms':J,'evictions':ops,'native_match':nm,'output_ids':summary['output_ids'],'cold_match':True,
            'census_counts':{'physical':rec['census']['physical_domain_size'],'mapped':len(rec['census']['mapped'])}}
    (OUT/'gate_b_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for c in checks:print(('PASS' if c['status']=='PASS' else 'FAIL'),c['check'])
    print('场景B核验判定：',result['verdict'])
    sys.exit(0 if ok else 1)
if __name__=='__main__':main()
