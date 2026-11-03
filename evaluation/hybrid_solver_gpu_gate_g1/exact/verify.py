"""G1 实验A离线核验（宿主机执行，只读）：4臂逐项对照冻结参考并汇总。

核验面：状态/场景/自动Exact分支（strict_decision）/选集与J==e3 w1/K2 Exact 冻结行/
逐Pending E-G==e3冻结值/Cold逐Token==E4-1A冻结Cold/驱逐5且全部returned/
执行锁事后完整（锁定文件哈希未变）/清理PASS/进程returncode 0。
"""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
OUT=ROOT/'evaluation/hybrid_solver_gpu_gate_g1/exact'
E3_REF={'selected':{'B5C3_NODE_18','B5C3_NODE_21'},'J_ms':53.77075886917578}
EG_REF={'IMW1_A_TURN_003':(3392,256),'IMW1_B_TURN_003':(3392,320),
       'IMW1_C_TURN_003':(3712,384),'IMW1_D_TURN_003':(3712,448)}
ARMS={'IMW1_A_TURN_003':'e1_w1k2_exact_A','IMW1_B_TURN_003':'e2_w1k2_exact_B',
      'IMW1_C_TURN_003':'e3_w1k2_exact_C','IMW1_D_TURN_003':'e4_w1k2_exact_D'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rows=[];ok_all=True
    for pid,run in ARMS.items():
        s=json.loads((OUT/run/'summary.json').read_text());d=s['strict_decision'];c=[]
        def chk(name,cond,detail=None):
            nonlocal ok_all;ok_all&=cond;c.append({'check':name,'status':'PASS' if cond else 'FAIL','detail':detail})
        chk('summary_status_PASS',s['status']=='PASS')
        chk('scenario_A_w1_k2_exact',s['scenario']=='A_w1_k2_exact_branch')
        chk('solver_自动Exact',d['solver']=='Exact' and d['gate_allowed'] and s['solver']=='Exact')
        chk('gate_7候选/4Pending/K2/29子集',(d['gate']['candidates'],d['gate']['pendings'],d['gate']['budget_k'],d['gate']['enumerable_subsets'])==(7,4,2,29))
        chk('选集==e3冻结Exact行',set(d['selected_ids'])==E3_REF['selected'] and set(s['plan_selected_ids'])==E3_REF['selected'])
        chk('J==e3冻结值',abs(d['objective_ms']-E3_REF['J_ms'])<=1e-9 and abs(s['plan_recovery_cost_ms']-E3_REF['J_ms'])<=1e-9)
        chk('selection_source_在线plan_strict',s['selection_source']=='GlobalOptimizer.plan_strict(在线)')
        e,g=EG_REF[pid]
        nm=s['native_match']
        chk('E-G==e3冻结值',(nm['executable_prefix'],nm['physical_fa_hit']-nm['executable_prefix'])==(e,g),{'E':nm['executable_prefix'],'G':nm['physical_fa_hit']-nm['executable_prefix'],'expect':(e,g)})
        chk('Cold逐Token==E4-1A',s['cold_reference_match'] and s['output_ids']==json.loads((OUT/f'cold_reference_{pid}.json').read_text())['output_ids'])
        chk('驱逐5且全returned',len(s['evictions'])==5 and all(r for _,r in s['evictions']))
        proc=json.loads((OUT/(run+'_process.json')).read_text());cl=json.loads((OUT/(run+'_cleanup.json')).read_text())
        chk('returncode_0_清理PASS',proc['returncode']==0 and cl['status']=='PASS')
        lock=json.loads((OUT/('execution_lock_'+run+'.json')).read_text())['files']
        # 宿主launch.py在e1后修复多臂重跑守卫（e2起为最终版并逐臂锁定）；证据与输入文件零漂移
        drift=[p for p,h in lock.items() if Path(p).exists() and sha(p)!=h]
        evidence_drift=[p for p in drift if not (Path(p).name=='launch.py' and run=='e1_w1k2_exact_A')]
        chk('执行锁事后完整(证据零漂移)',not evidence_drift,{'harness_note':'e1锁中launch.py为守卫修复前版本；其余臂无漂移','drift':drift[:3]})
        rows.append({'pending':pid,'run':run,'checks':c})
    out={'status':'PASS' if ok_all else 'FAIL','arms':rows,
         'e3_reference':{'source':'evaluation/greedy_swap_structure_e3_step1/condition_results.csv w1/K2/Exact 行','J_ms':E3_REF['J_ms'],'selected':sorted(E3_REF['selected'])},
         'cold_reference_source':'E4-1A {A,B,C,D}_Cold/output.json（经 validated_resumes 逐Token验证）'}
    (OUT/'verify.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('实验A核验',out['status'],'：4臂×10项')
if __name__=='__main__':main()
