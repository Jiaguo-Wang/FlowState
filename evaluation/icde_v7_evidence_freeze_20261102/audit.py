"""只读冻结记录并独立复核统计；不运行求解器、模型拟合或GPU实验。"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('FLOWSTATE_AUDIT_OUTPUT', str(Path(__file__).resolve().parent)))
PAPER = Path('/home/wjg/.codex/attachments/148edb61-b7d9-46fa-94de-725f1eb9e315/Pasted text.txt')
NAT = ROOT/'evaluation/rq3_workload_structure/natural_analysis'
BUD = ROOT/'evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030'
E3 = ROOT/'evaluation/greedy_swap_structure_e3_step1'
EPS = 1e-9
POLICIES = ('FlowState','PCB','LDR','WFC')
SOURCES = {}
MASTER = []


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def track(path):
    p = Path(path)
    if not p.is_absolute(): p = ROOT/p
    if str(p) not in SOURCES:
        SOURCES[str(p)] = {'bytes':p.stat().st_size,'sha256_before':sha(p)}
    return p


def obj(path): return json.loads(track(path).read_text())
def rows(path):
    with track(path).open() as f: return list(csv.DictReader(f))
def lines(path):
    with track(path).open() as f: return [json.loads(x) for x in f if x.strip()]


def outcsv(name, values):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def dump(name, value):
    target=OUT/name
    if target.exists():
        assert json.loads(target.read_text())==value, '既有派生结果不同：'+name
        return
    with target.open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def metric(cid, section, label, value, unit, grain, evidence_type, source, locator, paper_value='', status='VERIFIED', limit=''):
    MASTER.append({'claim_id':cid,'paper_section':section,'paper_locator':locator,'metric':label,
                   'paper_value':paper_value,'audited_value':value,'unit':unit,'statistical_grain':grain,
                   'evidence_type':evidence_type,'source':str(source),'status':status,'limitation':limit})


def phi(g,t):
    return 37.828150*g/1024 + 0.345974143*(g/1024)*(t/1024) - 0.156201917*(g/1024)**2


def evaluate(matrix, selected):
    ids=matrix['candidate_ids']; assert len(selected)==len(set(selected)) and set(selected)<=set(ids)
    ix=[ids.index(x) for x in selected]
    es=[max([matrix['positions'][i] for i in ix if matrix['compatibility'][p][i]]+[0]) for p in range(len(matrix['pending_ids']))]
    gs=[t-e for t,e in zip(matrix['targets'],es)]
    assert all(g>=0 for g in gs)
    return sum(phi(g,t) for g,t in zip(gs,matrix['targets'])),es,gs


def structure(matrix):
    n,m=len(matrix['candidate_ids']),len(matrix['pending_ids'])
    support=[{p for p in range(m) if matrix['benefit_ms'][p][c]>EPS} for c in range(n)]
    for p in range(m):
        for c in range(n):
            expect=phi(matrix['targets'][p],matrix['targets'][p])-phi(matrix['targets'][p]-matrix['positions'][c],matrix['targets'][p]) if matrix['compatibility'][p][c] else 0
            assert abs(expect-matrix['benefit_ms'][p][c])<EPS
    best=[max([matrix['positions'][c] for c in range(n) if matrix['compatibility'][p][c]]+[0]) for p in range(m)]
    required=sum(1<<p for p in range(m) if best[p]>0)
    masks=[sum(1<<p for p in range(m) if best[p]>0 and matrix['compatibility'][p][c] and matrix['positions'][c]==best[p]) for c in range(n)]
    dp={0:0}
    for mask in masks:
        for covered,count in list(dp.items()):dp[covered|mask]=min(dp.get(covered|mask,n+1),count+1)
    kmin=dp[required]
    return {'shared':sum(len(s)>1 for s in support),'positive':sum(bool(s) for s in support),
            'overlap':sum(bool(support[a]&support[b]) for a in range(n) for b in range(a+1,n)),
            'K_min':kmin,'support':support,'best':best}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paper=track(PAPER).read_text()
    dump('paper_version.json',{'path':str(PAPER),'sha256':sha(PAPER),'lines':len(paper.splitlines()),
          'missing_table_placeholders':paper.count('暂时无法在飞书文档外展示此内容'),'说明':'用户本轮提供的v7.0全文；不修改正文。'})
    matrices=lines('evaluation/gate1_step1_allocation_equivalence/matrices.jsonl')
    original=lines('evaluation/p1_strong_baseline_validation/per_case_results.jsonl')
    naturals=rows(NAT/'natural_workload_per_condition.csv')
    assert len(matrices)==len(original)==len(naturals)==573
    matrix_by={x['instance_id']:x for x in matrices}
    natural_by={x['condition_id']:x for x in naturals}
    snapshots={}; structures={}; outputs=[]; original_deltas=[]
    for line,r in enumerate(original,1):
        cid=f"{r['snapshot_id']}::budget={r['budget_ratio']:g}"
        mat=matrix_by[cid]; nat=natural_by[cid]
        s=structure(mat); structures[r['snapshot_id']]=s
        assert s['K_min']==int(nat['K_min_frontier']) and s['shared']==int(nat['shared_positive_candidate_count'])
        snapshots[r['snapshot_id']]=mat
        costs={}; selections={}
        for p in POLICIES:
            sel=r['policies'][p]['selected']; val,es,gs=evaluate(mat,sel)
            assert len(sel)<=r['k'] and abs(val-r['policies'][p]['cost_ms'])<EPS
            for actual,e,g in zip(r['policies'][p]['per_pending'],es,gs):
                assert actual['E']==e and actual['G']==g
            costs[p]=val;selections[p]=set(sel)
        assert len({tuple(sorted(x)) for x in selections.values()})==1
        assert max(costs.values())-min(costs.values())<EPS
        assert len(set(mat['checkpoint_bytes']))==1
        outputs.append({'domain':'Formal比例预算','condition_id':cid,'population':r['population'],
            'snapshot_id':r['snapshot_id'],'budget_ratio':r['budget_ratio'],'K':r['k'],
            'candidate_count':len(mat['candidate_ids']),'pending_count':len(mat['pending_ids']),
            'positive_candidates':s['positive'],'shared_candidates':s['shared'],'overlap_pairs':s['overlap'],
            'K_min':s['K_min'],'tight_budget':r['k']<s['K_min'],'shared_and_tight':s['shared']>0 and r['k']<s['K_min'],
            'same_set_all':True,'same_J_all':True,'different_set_equal_J':False,
            'J_by_policy':json.dumps(costs),'oracle_status':'结构证书，非本轮枚举',
            'physical_shared_certification':'NOT_STRICT_CERTIFIED','source':str(ROOT/'evaluation/p1_strong_baseline_validation/per_case_results.jsonl'),'source_row':line})
    assert len(snapshots)==191
    fixed=rows(BUD/'budget_sensitivity_per_snapshot.csv')
    strategies=rows(BUD/'budget_sensitivity_strategy_results.csv')
    oracle=rows(BUD/'exact_oracle_results.csv')
    groups=defaultdict(dict)
    for x in strategies: groups[(x['snapshot_id'],int(x['K']))][x['policy']]=x
    actual_cases=[]; oracle_optimal=Counter()
    for lineno,r in enumerate(fixed,2):
        sid,k=r['snapshot_id'],int(r['K']);mat=snapshots[sid];s=structures[sid]
        assert s['K_min']==int(r['K_min'])
        got=groups[sid,k];costs={};sets={}
        for p in POLICIES:
            row=got[p];sel=json.loads(row['selected']);v,e,g=evaluate(mat,sel)
            assert len(sel)<=k and abs(v-float(row['J_ms']))<EPS
            assert e==json.loads(row['E']) and g==json.loads(row['G'])
            costs[p]=v;sets[p]=set(sel)
        same_set=len({tuple(sorted(s)) for s in sets.values()})==1
        same_J=max(costs.values())-min(costs.values())<=EPS
        assert same_set==(r['same_set_all_four']=='True') and same_J==(r['same_J_all_four']=='True')
        outputs.append({'domain':'Formal固定预算','condition_id':sid+f'::K={k}','population':r['population'],
            'snapshot_id':sid,'budget_ratio':'N/A','K':k,'candidate_count':len(mat['candidate_ids']),
            'pending_count':len(mat['pending_ids']),'positive_candidates':s['positive'],'shared_candidates':s['shared'],
            'overlap_pairs':s['overlap'],'K_min':s['K_min'],'tight_budget':k<s['K_min'],
            'shared_and_tight':s['shared']>0 and k<s['K_min'],'same_set_all':same_set,'same_J_all':same_J,
            'different_set_equal_J':not same_set and same_J,'J_by_policy':json.dumps(costs),
            'oracle_status':r['oracle_status'],'physical_shared_certification':'NOT_STRICT_CERTIFIED',
            'source':str(BUD/'budget_sensitivity_per_snapshot.csv'),'source_row':lineno})
        if not same_J:
            support={c:sorted(mat['pending_ids'][p] for p in sup) for c,sup in zip(mat['candidate_ids'],s['support'])}
            actual_cases.append({'snapshot_id':sid,'population':r['population'],'K':k,'K_min':s['K_min'],
                 'candidate_count':len(mat['candidate_ids']),'pending_count':len(mat['pending_ids']),
                 'shared_candidates':s['shared'],'shared_and_tight':s['shared']>0 and k<s['K_min'],
                 'FlowState_J_ms':costs['FlowState'],'PCB_J_ms':costs['PCB'],'LDR_J_ms':costs['LDR'],'WFC_J_ms':costs['WFC'],
                 'selections':json.dumps({p:json.loads(got[p]['selected']) for p in POLICIES}),
                 'support_of_selected':json.dumps({c:support[c] for c in set.union(*sets.values())}),
                 'E_by_policy':json.dumps({p:json.loads(got[p]['E']) for p in POLICIES}),
                 'G_by_policy':json.dumps({p:json.loads(got[p]['G']) for p in POLICIES}),
                 'oracle_status':r['oracle_status'],'explanation':'覆盖数偏置或共享累计价值竞争' if r['population']=='AgentX' else '无共享；相同L而不同T的收益排序及ID并列截断',
                 'source':str(BUD/'budget_sensitivity_per_snapshot.csv'),'source_row':lineno})
    assert len(fixed)==764 and len(actual_cases)==10
    for r in oracle:
        sid,k=r['snapshot_id'],int(r['K']);n,m=int(r['n']),int(r['m'])
        allowed=n<=12 and m<=4 and k<=4 and sum(math.comb(n,i) for i in range(min(k,n)+1))<=794
        assert (r['status']=='PASS')==allowed
        if allowed:
            v,_,_=evaluate(snapshots[sid],json.loads(r['selected']));assert abs(v-float(r['J_ms']))<EPS
            for p in POLICIES:
                if abs(float(groups[sid,k][p]['J_ms'])-v)<=EPS:oracle_optimal[p]+=1
    strict=rows(E3/'condition_results.csv');assert len(strict)==140
    strict_groups=defaultdict(dict)
    for r in strict:strict_groups[r['case'],int(r['K'])][r['policy']]=r
    strict_opt=Counter()
    for (sid,k),group in strict_groups.items():
        exact=float(group['Exact']['J_ms'])
        for p,r in group.items():
            val=sum(phi(x['G'],x['T']) for x in json.loads(r['per_pending']))
            assert abs(val-float(r['J_ms']))<=EPS
            assert abs(float(r['J_ms'])-exact-float(r['gap_exact_ms']))<=EPS
            if abs(float(r['J_ms'])-exact)<=EPS:strict_opt[p]+=1
    nesting=obj('evaluation/paper_workload_structure_final/strict20_optima_nesting.json')
    assert nesting['unique_optima_conditions']==20 and nesting['structural_nonnested']==11
    short=obj('evaluation/paper_recovery_model_short_context_audit/audit_data.json')
    modeldir=ROOT/'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020'
    train=rows(modeldir/'calibration_points.csv');held=rows(modeldir/'heldout_predictions.csv')
    heldraw=rows(modeldir/'heldout_raw.csv');selection=obj(modeldir/'model_selection.json')
    assert len(train)==20 and len(held)==12 and len(heldraw)==210 and not selection['heldout_used_for_fit']
    modelmetrics={}
    for name in ('M0','M1','M2'):
        modelmetrics[name]={'MAPE_percent':st.mean(abs(float(r[name+'_prediction_ms'])-float(r['measured_phi_ms']))/abs(float(r['measured_phi_ms'])) for r in held)*100,
                            'MAE_ms':st.mean(abs(float(r[name+'_prediction_ms'])-float(r['measured_phi_ms'])) for r in held)}
    pairs=rows('evaluation/paper_recovery_model_short_context_audit/partB_pairs.csv')
    absolute=rows('evaluation/paper_recovery_model_short_context_audit/partB_absolute.csv')
    assert len(pairs)==24 and sum(r['same_frontier']=='False' for r in pairs)==19
    assert all(r['direction_ttft_ok']=='True' and r['direction_prefill_ok']=='True' for r in pairs if r['same_frontier']=='False')
    # 主文五轮：直接读15份原始摘要，不与T2-G2或最新Serving两臂混合。
    main={}; mainpaths=[]
    for rep in range(1,6):
        for tag in ('fs','mc','kv'):
            path=ROOT/f'evaluation/p0_control_cycle_gpu/arms/m{rep}_ck3_{tag}/run/summary.json'
            r=obj(path);assert r['status']=='PASS' and r['oracle_match']
            assert len(r['pending_rows'])==4 and all(q['cold_output_match'] and len(q['output_ids'])==8 for q in r['pending_rows'])
            main[rep,tag]=r;mainpaths.append(str(path))
    mainstats={}
    for tag in ('fs','mc','kv'):
        mainstats[tag]={'sum_ttft_median_ms':st.median(sum(q['ttft_ms'] for q in main[i,tag]['pending_rows']) for i in range(1,6)),
            'sum_cuda_median_ms':st.median(sum(q['cuda_prefill_ms'] for q in main[i,tag]['pending_rows']) for i in range(1,6)),
            'control_median_ms':st.median(main[i,tag]['cycle']['control_cycle_ms'] for i in range(1,6)),
            'full_median_ms':st.median(main[i,tag]['cycle']['cycle_ms_full_incl_warm'] for i in range(1,6)),
            'selection_call_median_ms':st.median(main[i,tag]['selection']['selection_call_ms'] for i in range(1,6))}
    pairmain={}
    for tag in ('mc','kv'):
        tt=[sum(q['ttft_ms'] for q in main[i,'fs']['pending_rows'])-sum(q['ttft_ms'] for q in main[i,tag]['pending_rows']) for i in range(1,6)]
        frac=[(sum(q['ttft_ms'] for q in main[i,tag]['pending_rows'])-sum(q['ttft_ms'] for q in main[i,'fs']['pending_rows']))/sum(q['ttft_ms'] for q in main[i,tag]['pending_rows'])*100 for i in range(1,6)]
        full=[main[i,'fs']['cycle']['cycle_ms_full_incl_warm']-main[i,tag]['cycle']['cycle_ms_full_incl_warm'] for i in range(1,6)]
        pairmain[tag]={'ttft_fs_minus_baseline_each_ms':tt,'median_ttft_difference_ms':st.median(tt),'median_relative_improvement_percent':st.median(frac),'full_fs_minus_baseline_each_ms':full,'median_full_difference_ms':st.median(full)}
    # 受控分支GPU：仅读取40个正式记录，按轮和Pending配对。
    g2=rows('evaluation/b5c4_formal_gpu_g2/formal/raw_results.csv');assert len(g2)==40
    gmap={(int(r['round']),r['solver'],r['pending'].split('_')[1]):r for r in g2}
    gstats={};rounds=[]
    for pid in ('A','B','C','D'):
        delta=[float(gmap[i,'Greedy',pid]['ttft_ms'])-float(gmap[i,'Exact',pid]['ttft_ms']) for i in range(1,6)]
        cuda=[float(gmap[i,'Greedy',pid]['cuda_prefill_ms'])-float(gmap[i,'Exact',pid]['cuda_prefill_ms']) for i in range(1,6)]
        gstats[pid]={'paired_TTFT_median_ms':st.median(delta),'deltas':delta,'paired_CUDA_median_ms':st.median(cuda)}
    for i in range(1,6):rounds.append(st.mean(float(gmap[i,'Greedy',p]['ttft_ms'])-float(gmap[i,'Exact',p]['ttft_ms']) for p in ('A','B','C','D')))
    assert all(r['cold_output_match']=='1' and int(r['audit_entries'])==2 and int(r['request_window_new_kernels'])==0 for r in g2)
    # 历史24快照的请求级统计：从216个原运行记录中的864条请求重算。
    legacy=lines('rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/runs.jsonl')
    assert len(legacy)==216 and len({r['run_id'] for r in legacy})==216
    totals=defaultdict(list)
    for r in legacy:
        assert r['status']=='PASS' and r['worker_exit_code']==0 and len(r['pending_requests'])==4
        totals[r['snapshot_id'],r['policy']].append(st.mean(q['ttft_ms'] for q in r['pending_requests']))
    ids=sorted({x[0] for x in totals});assert len(ids)==24
    legacy_stats={}
    for policy in ('LRU','Marconi'):
        diffs=[(st.mean(totals[s,policy])-st.mean(totals[s,'FlowState']))/st.mean(totals[s,policy])*100 for s in ids]
        legacy_stats[policy]={'mean_snapshot_relative_improvement_percent':st.mean(diffs),'wins':sum(x>0 for x in diffs),'losses':sum(x<0 for x in diffs)}
    # 不复算Bootstrap、不重新执行任何旧实验，读取已冻结的区间和逐epoch统计。
    cp=obj('evaluation/control_path_optimization_output/control_20260929/analysis/comparison.json')
    kimi=obj('evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json')
    assert len(kimi)==5 and all(x['status']=='PASS' and all(x['checks'].values()) for x in kimi)
    kdelta=[x['B_resume_ms']-x['A_resume_ms'] for x in kimi]
    solverraw=rows('evaluation/paper_solver_tradeoff_raw.csv')
    solvervals={p:st.median(float(x['elapsed_ms']) for x in solverraw if x['source']=='strict20' and x['method']==p) for p in ('Exact','Greedy','Greedy+1-Swap')}
    fixedsolvetime=st.median(float(r['solve_median_ms']) for r in strategies if r['policy']=='FlowState')
    fixedstats={}
    for pop in ('OpenHands','AgentX'):
        for k in (1,2,3,4):
            group=[r for r in outputs if r['domain']=='Formal固定预算' and r['population']==pop and r['K']==k]
            fixedstats[f'{pop}/K{k}']={'conditions':len(group),'shared':sum(r['shared_candidates']>0 for r in group),'tight':sum(r['tight_budget'] for r in group),'shared_and_tight':sum(r['shared_and_tight'] for r in group),'different_J':sum(not r['same_J_all'] for r in group)}
    naturalstats={pop:{'snapshots':len({r['snapshot_id'] for r in outputs if r['domain']=='Formal比例预算' and r['population']==pop}), 'conditions':sum(r['domain']=='Formal比例预算' and r['population']==pop for r in outputs),'shared_conditions':sum(r['domain']=='Formal比例预算' and r['population']==pop and r['shared_candidates']>0 for r in outputs),'tight_conditions':sum(r['domain']=='Formal比例预算' and r['population']==pop and r['tight_budget'] for r in outputs)} for pop in ('OpenHands','AgentX')}
    audit={'natural':naturalstats,'fixed':fixedstats,'fixed_divergent_conditions':len(actual_cases),'fixed_same_J':sum(r['same_J_all'] for r in outputs if r['domain']=='Formal固定预算'),
       'fixed_different_set_equal_J':sum(r['different_set_equal_J'] for r in outputs if r['domain']=='Formal固定预算'),
       'oracle_conditions':sum(r['status']=='PASS' for r in oracle),'oracle_not_admitted':sum(r['status']!='PASS' for r in oracle),'oracle_optimal_counts':dict(oracle_optimal),
       'strict20_optimal_counts':dict(strict_opt),'strict20_unique_optima':nesting['unique_optima_conditions'],'strict20_structural_nonnested_transitions':nesting['structural_nonnested'],
       'model_heldout':modelmetrics,'main15':mainstats,'main15_paired':pairmain,'g2_40':gstats,'g2_round_equal_weight_TTFT_deltas_ms':rounds,
       'legacy24':legacy_stats,'kimi_paired_mean_ms':st.mean(kdelta),'kimi_paired_median_ms':st.median(kdelta),
       'strict20_CPU_pooled_median_ms':solvervals,'fixed_CPU_median_of_condition_medians_ms':fixedsolvetime}
    outcsv('workload_structure_summary.csv',outputs)
    outcsv('ten_difference_cases.csv',actual_cases)
    dump('audit_statistics.json',audit)
    metric('D01','VI-D','比例预算四策略同集同J',573,'条件','191快照×3预算；4策略','CPU_FORMAL','evaluation/p1_strong_baseline_validation/per_case_results.jsonl','附件306','573/573',limit='非Strict Runtime认证；非573个独立工作流')
    metric('D02','VI-D','固定预算差异',10,'条件','191快照×K1–4=764','CPU_FORMAL',str(BUD/'budget_sensitivity_per_snapshot.csv'),'附件308','10/764',limit='9个AX共享紧预算；1个OH无共享；全部J不同')
    for p,v in oracle_optimal.items():metric('D03_'+p,'VI-D',p+'门内最优',v,'条件','396门内；368域外N/A','CPU_FORMAL',str(BUD/'exact_oracle_results.csv'),'附件309',str(v)+'/396',limit='已存Oracle证据，不外推超门禁')
    for p,v in strict_opt.items():metric('E01_'+p,'VI-E',p+'达到Strict20最优',v,'条件','5快照×K1–4=20','CPU_STRICT',str(E3/'condition_results.csv'),'附件320–323',limit='Strict准入不等于逐边恢复认证')
    for key,v in solvervals.items():metric('E02_'+key,'VI-E',key+'CPU中位',v,'ms','Strict20 pooled 2400次/方法；输入准备在外','CPU_TIMING','evaluation/paper_solver_tradeoff_raw.csv','附件324',limit='不是在线延迟或P95')
    metric('E03','VI-E','固定预算Greedy求解',fixedsolvetime,'ms','764条件各75次批均测量中位的跨条件中位','CPU_TIMING',str(BUD/'budget_sensitivity_strategy_results.csv'),'附件327','0.1045')
    for p in ('M0','M2'):
        for k,v in modelmetrics[p].items():metric('C01_'+p+'_'+k,'VI-C',p+' '+k,v,'%' if 'percent' in k else 'ms','12个非零G held-out聚合点；210次含30预热','GPU_CALIBRATION',str(modeldir/'heldout_predictions.csv'),'附件299',limit='T 48K–112K；不代表短上下文精度')
    for tag,d in mainstats.items():
        for k,v in d.items():metric('B01_'+tag+'_'+k,'VI-B/VI-F',tag+' '+k,v,'ms','同一受控C/K3；5轮；每轮4串行请求','GPU_REQUEST_OR_CONTROL','evaluation/p0_control_cycle_gpu/arms/m{1..5}_ck3_'+tag+'/run/summary.json','附件280/287/339',limit='P0完整周期批次；不混T2-G2或20261101批次')
    for tag,d in pairmain.items():
        for k,v in d.items():
            if not isinstance(v,list):metric('B02_'+tag+'_'+k,'VI-B/VI-F',tag+' '+k,v,'%' if 'percent' in k else 'ms','先同轮四请求和配对，再5轮取中位','GPU_PAIRED','evaluation/p0_control_cycle_gpu/arms/m{1..5}_ck3_{fs,'+tag+'}/run/summary.json','附件280/340',limit='差值FS−基线；n5无显著性主张')
    for pid,r in gstats.items():metric('D04_'+pid,'VI-D','分支'+pid+' Greedy−Exact TTFT配对中位',r['paired_TTFT_median_ms'],'ms','1受控快照×2策略×4Pending×5轮=40臂','GPU_PAIRED','evaluation/b5c4_formal_gpu_g2/formal/raw_results.csv','附件315','34.71/36.71' if pid in ('C','D') else '',limit='不是五个新工作流；A/B负结果保留')
    for p,d in legacy_stats.items():metric('B03_'+p,'VI-B/VI-F','旧24快照TTFT对'+p,d['mean_snapshot_relative_improvement_percent'],'%','24快照；每策略3生命周期，每生命周期4请求，单输出Token','GPU_LEGACY','rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/runs.jsonl','附件291', '43.06' if p=='LRU' else '24.26',status='QUALIFIED',limit='旧门禁；无Cold与完整Strict组件切片认证；非控制含入')
    metric('F01','VI-F','Kimi恢复窗口配对均值',st.mean(kdelta),'ms','固定MLA16K；5对TP2；max-rank内部匹配到单输出','GPU_CROSS_ARCH','evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json','附件334','760.19',limit='非客户端TTFT；非策略加速；无Kimi Phi校准')
    metric('F02','VI-F','Kimi恢复窗口配对中位',st.median(kdelta),'ms','同上','GPU_CROSS_ARCH','evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json','附件334','759.08')
    metric('F03','VI-F','旧Resume-Epoch控制均值',cp['new_means']['FlowState']['total_control_latency_ms'],'ms','旧优化协议，24快照×3重复','GPU_LEGACY_CONTROL','evaluation/control_path_optimization_output/control_20260929/analysis/comparison.json','附件341','131.904',limit='不可与10.139秒P0或最新GPU1路径直接比较')
    for p in ('FlowState','LRU','Marconi'):
        metric('F04_'+p,'VI-F',p+'完整Epoch均值',cp['new_means'][p]['control_inclusive_epoch_latency_ms'],'ms','旧Resume-Epoch；每快照先3重复平均、再24快照平均','GPU_LEGACY_EPOCH','evaluation/control_path_optimization_output/control_20260929/analysis/comparison.json','附件342',limit='两基线配对CI都含0；不是最新Serving协议')
    outcsv('evidence_master.csv',MASTER)
    for path,v in SOURCES.items():
        v['sha256_after']=sha(Path(path));assert v['sha256_after']==v['sha256_before'],path
    dump('source_manifest.json',{'files':SOURCES,'说明':'只读聚合及必要确定性复核；没有新实验、求解器执行、模型拟合或GPU调用。'})
    print('冻结数据独立核对完成；未修改源数据。')


if __name__=='__main__': main()
