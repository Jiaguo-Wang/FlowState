"""场景A结果核验（宿主机执行，只读）：单臂产物 vs 三重冻结 Oracle 与物理/请求正确性合同。

对照来源（全部只读）：
- b5d4b p01_Exact 冻结运行（同环境同K的 Exact 臂：计划、Oracle、Native 匹配、输出）；
- b5d4b p01_Greedy 冻结运行（旧 Greedy 选集，用于明确选集差异断言）；
- e3_step1/condition_results.csv 的 B5-C4 K=2 Exact 行（冻结 CPU 审计值）；
- b5d4b 冻结 Cold 参照（10 臂逐Token一致）。
输出 gate_a_result.json；任何一项不满足即给出 FAILED 项并退出非零。
"""
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
OUT=ROOT/'evaluation/paper_hybrid_gpu_gate_a'
RUN='a1_hybrid_exact'
B5D4B=ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4b'
E3CSV=ROOT/'evaluation/greedy_swap_structure_e3_step1/condition_results.csv'
EPS=1e-9
checks=[]
def check(name,ok,detail):
    checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':detail})
    return ok
def main():
    summary=json.loads((OUT/RUN/'summary.json').read_text())
    rec=json.loads((OUT/RUN/'reconciliation.json').read_text())
    post=json.loads((OUT/RUN/'post_request.json').read_text())
    cold_check=json.loads((OUT/RUN/'cold_reference_check.json').read_text())
    ref_exact=json.loads((B5D4B/'p01_Exact'/'reconciliation.json').read_text())
    ref_greedy=json.loads((B5D4B/'p01_Greedy'/'reconciliation.json').read_text())
    frozen_cold=json.loads((OUT/'frozen_cold_reference.json').read_text())
    # 1) 在线决策：门内自动 Exact
    d=rec['strict_decision']
    check('在线决策=Exact且门禁准许',d['solver']=='Exact' and d['gate_allowed'],d)
    check('门禁规模=7候选/4Pending/K=2/29子集',(d['gate']['candidates'],d['gate']['pendings'],d['gate']['budget_k'],d['gate']['enumerable_subsets'])==(7,4,2,29),d['gate'])
    check('决策记录完整(选集/J/纯求解时间)',set(d)>= {'solver','gate_allowed','gate','selected_ids','objective_ms','pure_solve_ms'} and d['selected_ids']==list(rec['plan']['selected_ids']) and abs(d['objective_ms']-rec['plan']['recovery_cost_ms'])<=EPS and d['pure_solve_ms']>0,{k:d[k] for k in ('selected_ids','objective_ms','pure_solve_ms')})
    # 2) 选集与 J：== 冻结 Exact Oracle（b5d4b 运行内 Decimal Oracle）== e3 冻结 Exact 行
    sel=set(rec['plan']['selected_ids']);J=rec['plan']['recovery_cost_ms']
    check('选集==b5d4b冻结Exact计划',sel==set(ref_exact['plan']['selected_ids']),sorted(sel))
    check('选集==本轮运行内Decimal Oracle',sel==set(rec['oracle']['selected']),rec['oracle']['selected'])
    check('J==b5d4b冻结Exact J(±1e-9)',abs(J-ref_exact['plan']['recovery_cost_ms'])<=EPS,J)
    check('J==本轮Decimal Oracle J(±1e-9)',abs(J-rec['oracle']['J_ms'])<=EPS,rec['oracle'])
    erow=next(r for r in csv.DictReader(open(E3CSV)) if r['case']=='B5-C4' and r['K']=='2' and r['policy']=='Exact')
    check('选集==e3冻结Exact行',sel==set(json.loads(erow['selected'])),erow['selected'])
    check('J==e3冻结Exact行(±1e-9)',abs(J-float(erow['J_ms']))<=EPS,erow['J_ms'])
    # 3) 与旧 Greedy 的明确选集差异
    gsel=set(ref_greedy['plan']['selected_ids'])
    check('与旧Greedy选集存在明确差异',sel!=gsel and bool(sel-gsel),{'exact':sorted(sel),'greedy_frozen':sorted(gsel),'difference':sorted(sel-gsel)})
    # 4) 物理协调：Recurrent-only 驱逐补集完整落实
    ops=[(o['checkpoint_id'],o['returned']) for o in rec['operations']]
    all_ids={c['candidate']['checkpoint_id'] for c in rec['epoch']['candidates']}
    check('驱逐==未选中补集且全部返回',sorted(x for x,_ in ops)==sorted(all_ids-sel) and all(r for _,r in ops),ops)
    check('逐Pending E/T/G==冻结Exact臂',[(p['pending_id'],p['E'],p['T'],p['G']) for p in rec['plan']['pending']]==[(p['pending_id'],p['E'],p['T'],p['G']) for p in ref_exact['plan']['pending']],[(p['pending_id'],p['E'],p['T'],p['G']) for p in rec['plan']['pending']])
    # 5) Native 恢复：E/FA命中/来源绑定
    nm=summary['native_match'];bound=next(x['node'] for x in rec['reconciliation']['binding']['candidates'] if x['candidate_id']==rec['target']['source_checkpoint'])
    check('Native来源==计划绑定Node/Slot',nm['best_match_node']==bound['node_id'] and list(nm['cow_source_slots'])==list(bound['slots']),{'native':(nm['best_match_node'],nm['cow_source_slots']),'bound':(bound['node_id'],bound['slots'])})
    check('Native E==计划E(3520)且FA命中==冻结边界(3904)',nm['executable_prefix']==rec['target']['expected']['E']==3520 and nm['physical_fa_hit']==rec['target']['pending']['attention_frontier']==3904,{'E':nm['executable_prefix'],'H':nm['physical_fa_hit'],'G':nm['replay_gap']})
    # 6) 请求正确性：8 Token == 冻结 Cold 参照（10 臂验证过）
    check('输出==冻结Cold参照逐Token',summary['output_ids']==frozen_cold['output_ids'] and cold_check['status']=='PASS' and cold_check['exact_token_match'],cold_check)
    # 7) 协调后状态保护（runner 已断言，此处从落盘证据复核）
    check('请求后Selected内容不变',post['state']['selected_payload']==rec['selected_payload_after'],None)
    check('请求后FA映射全保留',all(post['state']['fa_map'].get(k)==v for k,v in rec['fa_map_before'].items()),None)
    check('请求后受保护FA内容不变',post['state']['fa_protected_payload']==rec['fa_protected_payload'],None)
    # 8) Epoch 形成方式记录
    check('Epoch形成方式已记录(冻结历史重放,非全在线)',rec['origins'] and all(o['first_observed_after'] for o in rec['origins']) and len(rec['epoch']['candidates'])==7 and rec['epoch']['K']==2,{'origins':rec['origins'],'note':'复用B5C3冻结8条历史重放+实时观察准入'})
    # 9) 执行完整性
    proc=json.loads((OUT/(RUN+'_process.json')).read_text());cleanup=json.loads((OUT/(RUN+'_cleanup.json')).read_text())
    check('进程退出码0且GPU清理PASS',proc['returncode']==0 and cleanup['status']=='PASS',proc)
    # 10) 门禁期间冻结输入与被测代码未变（宿主机编排脚本 prepare/launch/verify 不属于被测路径；
    #     verify.py 曾在运行后修复自身键路径 bug，见 result['post_run_notes']）
    lock=json.loads((OUT/'execution_lock.json').read_text())['files']
    scope={p:h for p,h in lock.items() if Path(p).name not in ('prepare.py','launch.py','verify.py')}
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in scope.items())
    check('冻结输入/被测代码/生产入口全程未变',unchanged,'%d files（宿主机脚本除外）'%len(scope))
    ok=all(c['status']=='PASS' for c in checks)
    result={'scenario':'A_b5c4_k2_exact_branch','run':RUN,'verdict':'PASS' if ok else 'FAIL','checks':checks,
            'decision':d,'selected':sorted(sel),'J_ms':J,'oracle':rec['oracle'],
            'greedy_frozen_selection':sorted(gsel),'evictions':ops,
            'native_match':nm,'output_ids':summary['output_ids'],'cold_match':True,
            'post_run_notes':'verify.py 在 docker 运行完成后修复自身核验键路径（epoch.candidates[].candidate.checkpoint_id），不影响任何运行输入或被测代码；完整性检查范围已排除宿主机编排脚本'}
    (OUT/'gate_a_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for c in checks:print(('PASS' if c['status']=='PASS' else 'FAIL'),c['check'])
    print('场景A核验判定：',result['verdict'])
    sys.exit(0 if ok else 1)
if __name__=='__main__':main()
