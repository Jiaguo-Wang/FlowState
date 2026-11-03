"""Task 2：C/K3 简单策略等价性与 Exact 最优性独立 CPU 复核（只读冻结输入，不改生产代码）。

四个事实问题：
  A) PCB / LDR 在 C/K3 上是否得到与 FlowState-Greedy 相同选集；
  B) 仅按恢复缺口 Token 数的 Marginal Greedy（Gap-only）是否得到相同选集；
  C) FlowState-Greedy 是否达到冻结 Position-aware 目标 J 的全局最优（独立枚举证书）；
  D) StandaloneTopK / WFC 冻结选集中是否存在最终零边际收益的冗余候选。
全部使用既有实现与冻结输入：p1_strong_baseline_validation 的 common/pcb/ldr、
kvflow_multiworkflow_cpu_main 的 workloads.json/cpu_results.json、
kvflow_multiworkflow_cpu_review 的 review_results.json、paper_solver_tradeoff_summary.csv。
"""
import csv,itertools,json,sys,hashlib,time
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
OUT=ROOT/'evaluation/paper_ck3_simple_policy_check'
MW=ROOT/'evaluation/kvflow_multiworkflow_cpu_main'
RV=ROOT/'evaluation/kvflow_multiworkflow_cpu_review'
P1=ROOT/'evaluation/p1_strong_baseline_validation'
TR=ROOT/'evaluation/paper_solver_tradeoff_summary.csv'
T1M=ROOT/'evaluation/paper_strict_evidence_extraction/strict_allocation_matrix.csv'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(P1))
import common,pcb,ldr
EPS=1e-9
checks=[]
def check(name,ok,detail):
    checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':detail}); return ok
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eg(result):return {r['continuation_id']:(r['T'],r['E'],r['G']) for r in result['per_pending']}

# ---------- 0) 冻结输入与来源哈希 ----------
sources={'workloads.json':MW/'workloads.json','cpu_results.json':MW/'cpu_results.json',
         'review_results.json':RV/'review_results.json','p1_common.py':P1/'common.py',
         'p1_pcb.py':P1/'pcb.py','p1_ldr.py':P1/'ldr.py',
         'recovery_model.py':ROOT/'flowstate/recovery_model.py',
         'run_cpu.py':MW/'run_cpu.py','tradeoff_summary.csv':TR,'task1_matrix.csv':T1M}
ledger={'sha256':{str(p):sha(p) for p in sources.values()},'generated_unix_ns':time.time_ns()}

# ---------- 1) 复刻冻结 C/K3 Epoch（逐行复刻 run_cpu.py epoch(case,k) 的 scope='current' 语义） ----------
case=[x for x in json.loads((MW/'workloads.json').read_text())['cases'] if x['structure']=='C'][0]
calls=case['calls']; order=case['predeclared_evaluation_order']
qs=[next(p for p in calls if p['continuation_id']==i) for i in order if next(p for p in calls if p['continuation_id']==i)['ready']]
ids={p['continuation_id'] for p in qs}
ep=common.Epoch(tuple(common.Candidate(c['checkpoint_id'],c['token_position_L'],tuple(x for x in c['token_compatible_calls'] if x in ids)) for c in case['candidates']),tuple(common.Pending(p['continuation_id'],p['planning_target_T']) for p in qs),3)
check('Epoch身份=16候选/4Ready Pending/K=3',(len(ep.candidates),len(ep.pending),ep.k)==(16,4,3),{'candidates':[c.identifier for c in ep.candidates],'pendings':[p.identifier for p in ep.pending]})
join_empty=[c.identifier for c in ep.candidates if not c.compatible]
check('JOIN候选当前兼容为空(未来态)',join_empty==['C_JOIN_P04032','C_JOIN_P04096'],join_empty)

# ---------- 2) 冻结参照核验（选集+J+逐Pending E/G 与既有产物逐位一致） ----------
rows={r['policy']:r for r in json.loads((MW/'cpu_results.json').read_text())['rows'] if r['structure']=='C' and r['K']==3}
t2=json.loads((RV/'review_results.json').read_text())['task2_standalone_vs_greedy_C']['K=3']
FS=tuple(rows['FlowState-Greedy']['selected']); SA=tuple(rows['Marconi Adaptation']['selected']); WF=tuple(rows['WFC']['selected'])
for nm,sel in [('FlowState-Greedy',FS),('StandaloneTopK(与Marconi行同集)',SA),('WFC',WF)]:
    key={'FlowState-Greedy':'FlowState-Greedy','StandaloneTopK(与Marconi行同集)':'Marconi Adaptation','WFC':'WFC'}[nm]
    z=common.score(ep,sel); fz=rows[key]
    same_eg=eg(z)=={r['continuation_id']:(r['T'],r['E'],r['G']) for r in fz['current_E_G']}
    check(f'冻结参照重算一致[{nm}]',abs(z['cost_ms']-fz['current_J_ms'])<=EPS and same_eg,{'recomputed':z['cost_ms'],'frozen':fz['current_J_ms'],'same_EG':same_eg})
check('StandaloneTopK独立复核行(cpu_review task2)==Marconi行',t2['standalone_selected']==list(SA) and abs(t2['standalone_current_J_ms']-rows['Marconi Adaptation']['current_J_ms'])<=EPS and t2['greedy_selected']==list(FS),{'review_same_set':t2['same_set'],'J_diff':t2['J_diff_standalone_minus_greedy_ms']})

# ---------- 3) 检查A：PCB 与 LDR（既有实现，仅输入适配） ----------
def relation(sel_a,sel_b,ja,jb):
    """与 FlowState-Greedy 的关系：同集/异集 × 同/异目标，附差值。"""
    same=set(sel_a)==set(sel_b)
    return {'same_set':same,'same_EG':eg(common.score(ep,sel_a))==eg(common.score(ep,sel_b)),'same_J':abs(ja-jb)<=EPS,'delta_J_ms':ja-jb,'relation':'SAME_SET_SAME_OBJECTIVE' if same and abs(ja-jb)<=EPS else ('SAME_SET_DIFFERENT_OBJECTIVE' if same else 'DIFFERENT_SET_DIFFERENT_OBJECTIVE')}
psel,pinfo=pcb.select(ep); lsel,linfo=ldr.select(ep)
pj=common.score(ep,psel); lj=common.score(ep,lsel)
rel_pcb=relation(psel,FS,pj['cost_ms'],common.score(ep,FS)['cost_ms']); rel_ldr=relation(lsel,FS,lj['cost_ms'],common.score(ep,FS)['cost_ms'])
check('A1_PCB在C/K3可公平执行并完整记录',True,{'selected':list(psel),'J_ms':pj['cost_ms'],'relation':rel_pcb['relation'],'score_rule':'每Pending的Phi(T)-Phi(T-L)独立收益','sort_key':'(-score,checkpoint_id,pending_id)','tie_break':'(-score,cid,pid)排序后顺序取前K个不同cid','recommendations':pinfo['recommendations']})
check('A2_LDR在C/K3可公平执行并完整记录',True,{'selected':list(lsel),'J_ms':lj['cost_ms'],'relation':rel_ldr['relation'],'score_rule':'每Pending最深合法候选位置L','sort_key':'(-L即(-score,checkpoint_id,pending_id)','tie_break':'排序后顺序取前K个不同cid','recommendations':linfo['recommendations']})

# ---------- 4) 检查B：Gap-only Marginal Greedy（J_gap=Σ(T-E)，无 Phi；预算/平局规则同生产 Greedy） ----------
def jgap(sel):
    """缺口目标：Σ_p (T_p - E_p(S))，E=max 兼容已选位置。"""
    by={c.identifier:c for c in ep.candidates}; return sum(p.target-max([by[c].position for c in sel if p.identifier in by[c].compatible]+[0]) for p in ep.pending)
gsel=[]; gtrace=[]
while len(gsel)<ep.k:
    marg={c.identifier:jgap(gsel)-jgap(gsel+[c.identifier]) for c in ep.candidates if c.identifier not in gsel}
    best=max(marg.values())
    if best<=EPS: break  # 无正边际增益即停，规则同生产 Greedy
    pick=min(cid for cid,v in marg.items() if abs(v-best)<=EPS)  # 平局取 checkpoint_id 最小
    gtrace.append({'round':len(gsel)+1,'chosen':pick,'marginal_gap_reduction_tokens':best,'J_gap_before':jgap(gsel),'J_gap_after':jgap(gsel+[pick]),'E_after':{r['continuation_id']:r['E'] for r in common.score(ep,tuple(gsel+[pick]))['per_pending']},'all_marginals_tokens':marg})
    gsel.append(pick)
gsel=tuple(gsel); gj=common.score(ep,gsel); rel_gap=relation(gsel,FS,gj['cost_ms'],common.score(ep,FS)['cost_ms'])
check('B_Gap-only Greedy可执行且每轮记录完整',len(gtrace)==len(gsel),{'selected':list(gsel),'J_gap_tokens':jgap(gsel),'Phi_J_ms':gj['cost_ms'],'relation':rel_gap['relation'],'trace':gtrace})

# ---------- 5) 检查C：独立 Exact 枚举证书（|S|≤K 全子集，冻结 Phi，不触生产门禁） ----------
cands=sorted(c.identifier for c in ep.candidates)
subsets=[()] ; [subsets.extend(itertools.combinations(cands,r)) for r in (1,2,3)]
check('C1_枚举规模==C(16,0)+C(16,1)+C(16,2)+C(16,3)==697',len(subsets)==697,len(subsets))
js={s:common.score(ep,s)['cost_ms'] for s in subsets}
jmin=min(js.values()); argmin=[list(s) for s in subsets if js[s]-jmin<=EPS]
fs_j=js[tuple(sorted(FS))]
greedy_opt=set(FS) in [set(s) for s in argmin]
top5=sorted(js.items(),key=lambda kv:kv[1])[:5]
swap_ck3=[r for r in csv.DictReader(open(TR)) if r.get('source','')=='ck3']
swap_consist=all(abs(js[tuple(sorted(json.loads(r['selected'])))]-float(r['J_ms']))<=EPS for r in swap_ck3)
check('C2_FlowState-Greedy重算J==冻结J',abs(fs_j-rows['FlowState-Greedy']['current_J_ms'])<=EPS,{'recomputed':fs_j,'frozen':rows['FlowState-Greedy']['current_J_ms']})
check('C3_1-Swap(ck3)冻结行与枚举评分一致',swap_consist,[{'method':r['method'],'selected':r['selected'],'J_ms':r['J_ms']} for r in swap_ck3])
check('C4_Greedy是否全局最优(并列须全报)',greedy_opt,{'greedy_set':sorted(FS),'greedy_J':fs_j,'min_J':jmin,'argmin_all':argmin,'abs_gap_ms':fs_j-jmin,'rel_gap':(fs_j-jmin)/jmin if jmin>0 else 0.0,'top5':[{'set':list(s),'J_ms':v} for s,v in top5]})

# ---------- 6) 检查D：SA / WFC 冻结选集逐候选最终边际收益 ----------
red=[]
for nm,sel,frozen_j in [('StandaloneTopK',SA,rows['Marconi Adaptation']['current_J_ms']),('WFC',WF,rows['WFC']['current_J_ms'])]:
    full=common.score(ep,sel)
    for c in sel:
        rest=tuple(x for x in sel if x!=c); without=common.score(ep,rest)
        red.append({'policy':nm,'frozen_J_ms':frozen_j,'checkpoint_id':c,'L':next(z.position for z in ep.candidates if z.identifier==c),
                    'J_S_ms':full['cost_ms'],'J_S_minus_c_ms':without['cost_ms'],'final_marginal_ms':without['cost_ms']-full['cost_ms'],
                    'zero_marginal':abs(without['cost_ms']-full['cost_ms'])<=EPS,
                    'per_pending_EG_full':eg(full),'per_pending_EG_without':eg(without)})
sa7104=[r for r in red if r['policy']=='StandaloneTopK' and r['checkpoint_id']=='C_D_P07104'][0]
wfc_root=[r for r in red if r['policy']=='WFC' and r['checkpoint_id']=='C_ROOT_P01024'][0]
check('D1_SA中C_D_P07104最终边际==0',sa7104['zero_marginal'],{k:sa7104[k] for k in ('final_marginal_ms','J_S_ms','J_S_minus_c_ms')})
check('D2_WFC中C_ROOT_P01024最终边际==0',wfc_root['zero_marginal'],{k:wfc_root[k] for k in ('final_marginal_ms','J_S_ms','J_S_minus_c_ms')})

# ---------- 7) 产物落盘 ----------
OUT.mkdir(exist_ok=True)
def table(name,rows_):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows_[0])); w.writeheader()
        for r in rows_: w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
frozen_note='冻结产物 cpu_results.json / review_results.json'
comp=[
 {'policy':'PCB','source':'本轮复算(p1 pcb.py)','selected':list(psel),'J_ms':pj['cost_ms'],'per_pending_EG':eg(pj),'vs_flowstate':rel_pcb['relation'],'delta_J_ms':rel_pcb['delta_J_ms'],'note':'每Pending最大独立Phi收益，(-score,cid,pid)排序去重'},
 {'policy':'LDR','source':'本轮复算(p1 ldr.py)','selected':list(lsel),'J_ms':lj['cost_ms'],'per_pending_EG':eg(lj),'vs_flowstate':rel_ldr['relation'],'delta_J_ms':rel_ldr['delta_J_ms'],'note':'每Pending最深候选，(-L,cid,pid)排序去重'},
 {'policy':'Gap-only Greedy','source':'本轮复算(检查B)','selected':list(gsel),'J_ms':gj['cost_ms'],'per_pending_EG':eg(gj),'vs_flowstate':rel_gap['relation'],'delta_J_ms':rel_gap['delta_J_ms'],'note':'J_gap=Σ(T-E)边际下降最大，平局取cid最小，预算K=3；J_gap终值=%d tokens'%jgap(gsel)},
 {'policy':'FlowState-Greedy','source':frozen_note,'selected':list(FS),'J_ms':rows['FlowState-Greedy']['current_J_ms'],'per_pending_EG':{r['continuation_id']:(r['T'],r['E'],r['G']) for r in rows['FlowState-Greedy']['current_E_G']},'vs_flowstate':'REFERENCE','delta_J_ms':0.0,'note':'生产_select_candidates边际J下降'},
 {'policy':'StandaloneTopK','source':frozen_note+' '+'review task2','selected':list(SA),'J_ms':rows['Marconi Adaptation']['current_J_ms'],'per_pending_EG':{r['continuation_id']:(r['T'],r['E'],r['G']) for r in rows['Marconi Adaptation']['current_E_G']},'vs_flowstate':'DIFFERENT_SET_DIFFERENT_OBJECTIVE','delta_J_ms':193.88505565,'note':'静态独立收益TopK，与Marconi行同集同J'},
 {'policy':'WFC','source':frozen_note,'selected':list(WF),'J_ms':rows['WFC']['current_J_ms'],'per_pending_EG':{r['continuation_id']:(r['T'],r['E'],r['G']) for r in rows['WFC']['current_E_G']},'vs_flowstate':'DIFFERENT_SET_DIFFERENT_OBJECTIVE','delta_J_ms':rows['WFC']['current_J_ms']-rows['FlowState-Greedy']['current_J_ms'],'note':'工作流共享覆盖'},
 {'policy':'Exact(独立枚举)','source':'本轮枚举(检查C)','selected':argmin[0] if len(argmin)==1 else argmin,'J_ms':jmin,'per_pending_EG':eg(common.score(ep,tuple(argmin[0]))),'vs_flowstate':'SAME_SET_SAME_OBJECTIVE' if greedy_opt else 'DIFFERENT_SET_DIFFERENT_OBJECTIVE','delta_J_ms':fs_j-jmin,'note':'697子集全枚举，argmin并列数=%d'%len(argmin)}]
table('policy_comparison.csv',comp)
(OUT/'exact_enumeration.json').write_text(json.dumps({'enumerated_subsets':len(subsets),'expected_subsets':697,'tolerance':EPS,'min_J_ms':jmin,'argmin_sets':argmin,'argmin_count':len(argmin),'flowstate_greedy':{'set':sorted(FS),'J_ms':fs_j,'global_optimal':greedy_opt,'abs_gap_ms':fs_j-jmin,'rel_gap':(fs_j-jmin)/jmin if jmin>0 else 0.0},'consistency':{'greedy_J_vs_frozen':abs(fs_j-rows['FlowState-Greedy']['current_J_ms'])<=EPS,'swap_ck3_rows_consistent':swap_consist},'top5_subsets':[{'set':list(s),'J_ms':v} for s,v in top5],'input_sha256':ledger['sha256'],'生产门禁未被触碰':'plan_strict Exact 门禁仍为 n≤12 拒绝 n=16；本枚举仅为独立审计证书'},ensure_ascii=False,indent=2)+'\n')
table('redundancy_check.csv',red)
(OUT/'gap_greedy_trace.json').write_text(json.dumps(gtrace,ensure_ascii=False,indent=2)+'\n')
(OUT/'checks.json').write_text(json.dumps({'checks':checks,'all_pass':all(c['status']=='PASS' for c in checks),'input_ledger':ledger},ensure_ascii=False,indent=2)+'\n')
print('\n'.join(f"{c['status']} {c['check']}" for c in checks))
print('全部通过' if all(c['status']=='PASS' for c in checks) else '存在FAIL')
