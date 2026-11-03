"""G1 实验B离线核验（宿主机执行，只读）：4臂逐项对照冻结参考并汇总。

核验面：状态/场景/自动Marginal Greedy分支（gate_allowed=False，16候选超门禁）/
census 16=16/在线选集==冻结FS三元集/J==154.34895561599998/逐Pending E-G==冻结合同
E=(0,5120,6144,7168)、G=(4096,0,0,0)/Prefill==G+32/Cold逐Token==冻结Cold/
驱逐13且全returned/执行锁事后完整/清理PASS/进程returncode 0；
另附失败尝试台账（g1/g2/g3/g4 首批与 r2，根因与证据路径）。
"""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
OUT=ROOT/'evaluation/hybrid_solver_gpu_gate_g1/greedy'
FS={'C_B_P05120','C_C_P06144','C_D_P07168'};J_REF=154.34895561599998
EG_REF={'C_A':(0,4096),'C_B':(5120,0),'C_C':(6144,0),'C_D':(7168,0)}
ARMS={'C_A':'g1_ck3_greedy_A_r3','C_B':'g2_ck3_greedy_B_r3','C_C':'g3_ck3_greedy_C_r3','C_D':'g4_ck3_greedy_D_r3'}
FAILED=[{'run':'g1_ck3_greedy_A','runner':'fix_b原版','error':'StopIteration：runner第94行假定目标必有绑定源，C_A(E=0)无源'},
        {'run':'g2_ck3_greedy_B','runner':'fix_b原版','error':'FileExistsError：g1首败遗留/b5out/target_request.json'},
        {'run':'g3_ck3_greedy_C','runner':'fix_b原版','error':'同g2（根目录x写跨臂冲突）'},
        {'run':'g4_ck3_greedy_D','runner':'fix_b原版','error':'同g2（根目录x写跨臂冲突）'},
        {'run':'g2_ck3_greedy_B_r2','runner':'pkg_greedy v1','error':'FileExistsError：g1_r3成功后在根写入target_request.json'},
        {'run':'g3_ck3_greedy_C_r2','runner':'pkg_greedy v1','error':'同上'},
        {'run':'g4_ck3_greedy_D_r2','runner':'pkg_greedy v1','error':'同上'}]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rows=[];ok_all=True
    for pid,run in ARMS.items():
        s=json.loads((OUT/run/'summary.json').read_text());d=s['strict_decision'];c=[]
        def chk(name,cond,detail=None):
            nonlocal ok_all;ok_all&=cond;c.append({'check':name,'status':'PASS' if cond else 'FAIL','detail':detail})
        chk('summary_status_PASS',s['status']=='PASS')
        chk('scenario_B_ck3_greedy',s['scenario']=='B_ck3_greedy_branch' and s['pending']==pid)
        chk('solver_自动MarginalGreedy',d['solver']=='Marginal Greedy' and not d['gate_allowed'] and d['gate']['candidates']==16)
        chk('census_16=16',s['census_counts']=={'physical':16,'mapped':16,'unmapped_extras':0})
        rec=json.loads((OUT/run/'reconciliation.json').read_text())
        cpu2phys={m['cpu_checkpoint_id']:m['physical_checkpoint_id'] for m in rec['census']['mapped']}
        chk('在线选集==冻结FS三元集',set(s['selected_cpu_mapped'])==FS and {cpu2phys[x] for x in s['selected_cpu_mapped']}==set(rec['plan']['selected_ids']))
        chk('J==冻结154.34895561599998',abs(s['plan_recovery_cost_ms']-J_REF)<=1e-9 and abs(d['objective_ms']-J_REF)<=1e-9)
        chk('selection_source_在线plan_strict',s['selection_source']=='GlobalOptimizer.plan_strict(在线)')
        chk('oracle_match',s['oracle_match'] and s['oracle_policy'].startswith('FlowState-Greedy(policy_plans.json，仅事后对照)'))
        e,g=EG_REF[pid];nm=s['native_match']
        chk('E-G==冻结合同',(s['E_executable_prefix'],s['G_replay_gap'])==(e,g) and (nm['executable_prefix'],nm['physical_fa_hit']-nm['executable_prefix'])==(e,g),{'E':s['E_executable_prefix'],'G':s['G_replay_gap'],'expect':(e,g)})
        chk('Prefill==G+32',s['prefill_token_count']==g+32 and s['batch_prefix_length']==e)
        chk('Cold逐Token==冻结Cold',s['cold_output_match'] and s['output_ids']==json.loads((OUT/f'cold_reference_{pid}.json').read_text())['output_ids'])
        chk('驱逐13且全returned',len(s['evictions'])==13 and all(r for _,r in s['evictions']))
        proc=json.loads((OUT/(run+'_process.json')).read_text());cl=json.loads((OUT/(run+'_cleanup.json')).read_text())
        chk('returncode_0_清理PASS',proc['returncode']==0 and cl['status']=='PASS')
        lock=json.loads((OUT/('execution_lock_'+run+'.json')).read_text())['files']
        # 证据与输入零漂移；豁免：pkg_greedy/*.py 在 g1_r3 后由 v1 重建为 v2（两处最小修复，见 build_record），
        # g2-g4_r3 均以 v2 锁定执行。g1_r3 锁中为 v1（其求解/协调语义与 v2 完全一致，差异仅E=0断言与留证路径）。
        drift=[p for p,h in lock.items() if Path(p).exists() and sha(p)!=h and '/pkg_greedy/' not in p]
        pkg_note='g1_r3以v1(=fix_b+E=0修复)执行并通过；v2仅再改留证写入路径' if run=='g1_ck3_greedy_A_r3' else 'v2锁定执行'
        chk('执行锁事后完整(pkg v1→v2重建豁免并另行核验)',not drift,{'drift':drift[:3],'pkg':pkg_note})
        rows.append({'pending':pid,'run':run,'checks':c,'pkg_lock_note':pkg_note})
    # 全局补验：当前 pkg_greedy 文件 == g4_r3 所锁 v2（证明现存代码即 g2-g4_r3 实际执行版）
    lock4=json.loads((OUT/'execution_lock_g4_ck3_greedy_D_r3.json').read_text())['files']
    pkg_now={p:h for p,h in lock4.items() if '/pkg_greedy/' in p}
    pkg_ok=all(sha(p)==h for p,h in pkg_now.items())
    ok_all&=pkg_ok
    rows.append({'pending':'(全局)','run':'pkg_greedy_v2_convergence','checks':[{'check':'当前pkg==g4_r3锁定版','status':'PASS' if pkg_ok else 'FAIL','detail':sorted(Path(p).name for p in pkg_now)}]})
    out={'status':'PASS' if ok_all else 'FAIL','arms':rows,
         'fs_reference':{'source':'c_formal_perf_gpu/g1_CB_FS policy_plans.json FlowState-Greedy + 冻结运行记录','J_ms':J_REF,'selected_cpu':sorted(FS),'E_G':{k:list(v) for k,v in EG_REF.items()}},
         'cold_reference_source':'c_formal_perf_gpu/g1_CB_FS（==根原始==m1{A,B,C,D}_FS 冻结输出）',
         'failed_attempts':FAILED,'failed_attempts_note':'全部失败证据（runtime.log/failure.json/execution_lock）原样保留于本目录；修复见 pkg_greedy/build_record.json 与 input_provenance_amendment1.json'}
    (OUT/'verify.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('实验B核验',out['status'],'：4臂×13项；失败尝试',len(FAILED),'条已入台账')
if __name__=='__main__':main()
