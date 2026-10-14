"""只读取冻结结果，生成统一矩阵与论文图表；不执行任何分配策略。"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/flowstate-p4-mpl')
import csv, json, hashlib, statistics as st
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EPS=1e-9
sources={}
def read(path):
    p=ROOT/path if not Path(path).is_absolute() else Path(path)
    raw=p.read_bytes();sources[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return raw.decode()
def jl(path): return [json.loads(s) for s in read(path).splitlines() if s.strip()]
def js(path): return json.loads(read(path))
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jout(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
p1path='evaluation/p1_strong_baseline_validation/per_case_results.jsonl'
cases=jl(p1path)
assert len(cases)==573
cache={}
long=[]; exact=[]
for line,c in enumerate(cases,1):
    for path in [c['source_file'],c['ablation_source']]:
        if path not in cache: cache[path]=jl(path)
    raw=cache[c['source_file']][c['source_line']-1]
    ab=cache[c['ablation_source']][c['ablation_line']-1]
    assert raw['snapshot_id']==ab['snapshot_id']==c['snapshot_id']
    assert raw['budget_ratio']==ab['budget_ratio']==c['budget_ratio']
    assert raw['snapshot_digest']==ab['snapshot_digest']==c['snapshot_digest']
    ex=raw['exact_opt']
    if ex['tractable']:
        exact.append(dict(dataset=c['population'],budget_ratio=c['budget_ratio'],snapshot_id=c['snapshot_id'],match=abs(raw['policies']['FlowState']['total_recovery_cost_ms']-ex['total_recovery_cost_ms'])<=EPS,source=c['source_file']+':'+str(c['source_line'])))
    for method,p in c['policies'].items():
        source=f'{p1path}:{line}';cost=p['cost_ms']; selected=p['selected']
        if method in raw['policies']:
            old=raw['policies'][method];cost=old['total_recovery_cost_ms']
            assert set(selected)==set(old['selected_checkpoint_ids'])
            source=c['source_file']+':'+str(c['source_line'])
        elif method=='StandaloneTopK':
            old=ab['variants']['without_set_dependent'];cost=old['common_cost_ms']
            assert set(selected)==set(old['selected_checkpoint_ids'])
            source=c['ablation_source']+':'+str(c['ablation_line'])
        assert abs(cost-p['cost_ms'])<=EPS
        original_cost=cost
        cost=p['cost_ms']
        coverage=all(x['G']==0 for x in p['per_pending'])
        assert coverage==(cost<=EPS)
        bs=c['budget_structure'];kmin=bs['minimum_checkpoints_for_all_frontiers']
        long.append(dict(dataset=c['population'],snapshot_id=c['snapshot_id'],budget_ratio=c['budget_ratio'],method=method,cost_ms=cost,original_frozen_cost_ms=original_cost,baseline_minus_flowstate_ms=cost-c['policies']['FlowState']['cost_ms'],full_coverage=int(coverage),same_selected_set=int(set(selected)==set(c['policies']['FlowState']['selected'])),K=c['k'],K_min_frontier=kmin,selected_ids=json.dumps(sorted(selected)),source=source,structural_source=f'{p1path}:{line}'))
assert len({(r['dataset'],r['snapshot_id'],r['budget_ratio'],r['method']) for r in long})==4584
csvout('per_case_evidence.csv',long);csvout('exact_opt_cases.csv',exact)
groups=defaultdict(list)
for r in long: groups[r['dataset'],r['budget_ratio'],r['method']].append(r)
matrix=[]
for (d,b,m),rows in sorted(groups.items()):
    n=len(rows);assert n=={'OpenHands':168,'AgentX':23}[d]
    delta=[r['baseline_minus_flowstate_ms'] for r in rows]
    matrix.append(dict(dataset=d,budget_ratio=b,method=m,snapshots=n,mean_modeled_cost_ms=st.mean(r['cost_ms'] for r in rows),median_modeled_cost_ms=st.median(r['cost_ms'] for r in rows),mean_baseline_minus_flowstate_ms=st.mean(delta),median_baseline_minus_flowstate_ms=st.median(delta),flowstate_wins=sum(x>EPS for x in delta),ties=sum(abs(x)<=EPS for x in delta),flowstate_losses=sum(x < -EPS for x in delta),full_coverage_count=sum(r['full_coverage'] for r in rows),full_coverage_fraction=st.mean(r['full_coverage'] for r in rows),same_selected_set_count=sum(r['same_selected_set'] for r in rows),same_selected_set_fraction=st.mean(r['same_selected_set'] for r in rows),K_min=min(r['K'] for r in rows),K_max=max(r['K'] for r in rows),K_min_frontier_min=min(r['K_min_frontier'] for r in rows),K_min_frontier_max=max(r['K_min_frontier'] for r in rows),K_lt_K_min_frontier=sum(r['K']<r['K_min_frontier'] for r in rows),K_eq_K_min_frontier=sum(r['K']==r['K_min_frontier'] for r in rows),K_gt_K_min_frontier=sum(r['K']>r['K_min_frontier'] for r in rows),evidence='per_case_evidence.csv'))
csvout('overall_allocation_matrix.csv',matrix)
exact_summary=[]
for d in ['OpenHands','AgentX','总体']:
    for b in [.25,.5,.75,'全部预算']:
        rows=[r for r in exact if (d=='总体' or r['dataset']==d) and (b=='全部预算' or r['budget_ratio']==b)]
        exact_summary.append(dict(dataset=d,budget_ratio=b,matches=sum(r['match'] for r in rows),tractable_cases=len(rows)))
csvout('exact_opt_subset_quality.csv',exact_summary)
strong=[]
for m in ['PCB','LDR','WFC']:
    comp=[c['comparisons'][m] for c in cases]
    strong.append(dict(method=m,cases=len(comp),same_objective=sum(abs(x['baseline_minus_flowstate_ms'])<=EPS for x in comp),same_set=sum(x['same_set'] for x in comp),same_EG=sum(x['same_EG'] for x in comp),different_order=sum(x['order_different'] for x in comp)))
csvout('strong_baseline_summary.csv',strong)
struct=[]
for d in ['OpenHands','AgentX']:
    for b in [.25,.5,.75]:
        sub=[c for c in cases if c['population']==d and c['budget_ratio']==b]
        struct.append(dict(dataset=d,budget_ratio=b,cases=len(sub),binding=sum(c['k']<c['budget_structure']['minimum_checkpoints_for_all_frontiers'] for c in sub),positive_shared=sum(any(sum(x>EPS for x in vals)>1 for vals in c['benefit_matrix_ms'].values()) for c in sub),unavoidable_residual=sum(c['budget_structure']['all_candidate_cost_ms']>EPS for c in sub)))
csvout('workload_structure.csv',struct)
ablation=[]
abrows=jl('evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726/per_case_results.jsonl')
for d in ['OpenHands','AgentX']:
    for b in [.25,.5,.75]:
        for v in ['without_position_aware','without_set_dependent']:
            rows=[r for r in abrows if r['population']==d and r['budget_ratio']==b]
            comp=[r['comparisons'][v] for r in rows]
            ablation.append(dict(dataset=d,budget_ratio=b,variant=v,cases=len(rows),flowstate_wins=sum(x['outcome']=='win' for x in comp),ties=sum(x['outcome']=='tie' for x in comp),flowstate_losses=sum(x['outcome']=='loss' for x in comp),different_sets=sum(x['selected_set_different'] for x in comp),different_orders=sum(x['selection_order_different'] for x in comp),mean_difference_ms=st.mean(x['ablation_minus_full_cost_ms'] for x in comp)))
csvout('ablation_summary.csv',ablation)

runtime=[];paired={};diagnostics={}
for batch,path in [('初始控制计时','evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928'),('优化后控制计时','evaluation/control_path_optimization_output/control_20260929/optimized_formal')]:
    summary=js(path+'/analysis/summary.json'); records=[js(str(p.relative_to(ROOT))) for p in sorted((ROOT/path/'runs').glob('*/record.json'))]
    assert len(records)==216 and sum(len(r['requests']) for r in records)==864
    paired[batch]=summary['paired']
    diagnostics[batch]={k:summary[k] for k in ['rounds','cv_summary'] if k in summary}
    for policy,expected in summary['means'].items():
        rs=[r for r in records if r['policy']==policy];assert len(rs)==72
        for metric,target in expected.items():
            bysnap=defaultdict(list)
            for r in rs: bysnap[r['snapshot_id']].append(r[metric])
            assert len(bysnap)==24 and all(len(v)==3 for v in bysnap.values())
            value=st.mean(st.mean(v) for v in bysnap.values());assert abs(value-target)<1e-6
            runtime.append(dict(batch=batch,policy=policy,metric=metric,mean_ms=value,snapshots=24,lifecycles=72,source=path+'/runs/*/record.json',frozen_summary=path+'/analysis/summary.json'))
    for r in records:
        assert abs(r['control_inclusive_epoch_latency_ms']-(r['epoch_end_ts']-r['epoch_start_ts']))<.01
        assert r['status']=='PASS'
        for q in r['requests']:
            assert abs(q['completion_ts']-q['submit_ts']-q['request_latency_ms'])<.01
            assert abs(q['first_token_ts']-q['submit_ts']-q['ttft_ms'])<.01
csvout('runtime_summary.csv',runtime);jout('runtime_paired_frozen.json',paired)
rq4='rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652'
rq4rows=[]
for p in sorted((ROOT/rq4/'runs').glob('*/pending_requests.jsonl')): rq4rows+=jl(str(p.relative_to(ROOT)))
assert len(rq4rows)==864
rq4means=[]
for policy in ['LRU','Marconi','FlowState']:
    grouped=defaultdict(list)
    for r in rq4rows:
        if r['policy']==policy:grouped[r['snapshot_id'],r['repetition']].append(r['ttft_ms'])
    assert len(grouped)==72 and all(len(v)==4 for v in grouped.values())
    snaps=defaultdict(list)
    for (sid,rep),v in grouped.items():snaps[sid].append(st.mean(v))
    rq4means.append(dict(policy=policy,mean_ttft_ms=st.mean(st.mean(v) for v in snaps.values()),snapshots=len(snaps),source=rq4+'/runs/*/pending_requests.jsonl'))
csvout('runtime_ttft_summary.csv',rq4means)
diagnostics['RQ4']={name:js(rq4+'/analysis_snapshot_level/'+name+'.json') for name in ['paired_ttft_results','bootstrap_results','per_round_results','repetition_stability']}
for row in rq4means:
    ps=diagnostics['RQ4']['paired_ttft_results']['comparisons']
    if row['policy']!='FlowState':
        fs=next(x['mean_ttft_ms'] for x in rq4means if x['policy']=='FlowState')
        assert abs(row['mean_ttft_ms']-fs-ps[row['policy']]['mean_absolute_reduction'])<1e-8
jout('runtime_batch_diagnostics.json',diagnostics)
reports=['evaluation/p0_allocation_mechanism_audit/final_report.md','evaluation/p0_allocation_mechanism_audit/case_studies.md','evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md','evaluation/p1_strong_baseline_validation/final_report.md','evaluation/p2_shared_cohort_feasibility_audit/final_report.md','evaluation/p3_physical_retention_budget_contract/final_report.md','evaluation/rq2_model_audit_output/20260909_055110/final_report.md','motivation/artifacts/replay_cost_20260819/findings.md','evaluation/rq6_batched_control_output/rq6e_20260911_085820/analysis/final_report.md','evaluation/serving_path_e2e_output/serving_20260929/final_report.md','evaluation/control_path_optimization_output/control_20260929/analysis/final_report.md','evaluation/sota_latency_runtime.py','evaluation/control_path_optimized_e2e.py','evaluation/resume_epoch_e2e.py','evaluation/rq4_snapshot_level_analysis.py','evaluation/rq4_unified_runtime_harness.py','evaluation/rq3_formal_policy_evaluation.py','evaluation/rq5_frozen_snapshot_ablation.py']
for path in reports: read(path)
lines=['# 完整数值表（从冻结逐案例文件汇总）','','单位为 modeled ms；差值为 baseline−FlowState，胜负为 FlowState 视角。完整精度与来源见CSV。','','| 数据集 | 预算 | 方法 | N | mean | median | mean差 | W/T/L | Full Coverage | 同集合 | K<K_min |','|---|---|---|---:|---:|---:|---:|---|---|---|---|']
for r in matrix:
    lines.append(f"| {r['dataset']} | {r['budget_ratio']:.0%} | {r['method']} | {r['snapshots']} | {r['mean_modeled_cost_ms']:.6f} | {r['median_modeled_cost_ms']:.6f} | {r['mean_baseline_minus_flowstate_ms']:.6f} | {r['flowstate_wins']}/{r['ties']}/{r['flowstate_losses']} | {r['full_coverage_count']}/{r['snapshots']} | {r['same_selected_set_count']}/{r['snapshots']} | {r['K_lt_K_min_frontier']}/{r['snapshots']} |")
(OUT/'allocation_tables.md').write_text('\n'.join(lines)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'pdf.fonttype':42,'axes.unicode_minus':False,'font.size':10})
from matplotlib import font_manager
font_manager.fontManager.addfont(font.get_file())
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,d in zip(axes,['OpenHands','AgentX']):
    for m,color,marker in [('LRU','#b9770e','s'),('LFU','#808080','v'),('Marconi','#7b3294','^'),('StandaloneTopK','#0571b0','D'),('FlowState','#00865a','o')]:
        data=sorted([r for r in matrix if r['dataset']==d and r['method']==m],key=lambda r:r['budget_ratio'])
        label='FlowState / PCB / LDR / WFC' if m=='FlowState' else ('Marconi Adaptation' if m=='Marconi' else m)
        ax.plot([25,50,75],[r['mean_modeled_cost_ms'] for r in data],label=label,color=color,marker=marker)
    ax.set(title=d+'（'+str(168 if d=='OpenHands' else 23)+' 个快照）',xlabel='逻辑保留预算比例',ylabel='平均可执行恢复成本（模型预测 ms）',xticks=[25,50,75],xticklabels=['25%','50%','75%'],ylim=(0,None));ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.08))
fig.suptitle('总体分配质量：两个子图使用不同纵轴范围');fig.tight_layout();save(fig,'overall_allocation_quality')
fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
table=ax.table(cellText=[[r['method'],f"{r['same_objective']}/573",f"{r['same_set']}/573",f"{r['same_EG']}/573",str(r['different_order'])] for r in strong],colLabels=['方法','目标值相同','最终集合相同','可执行前沿相同','选择顺序不同'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.8)
ax.set_title('强基线验证：全部最终收益持平');fig.text(.5,.03,'573 个快照–预算条件来自 191 个快照；顺序差异不构成最终收益改善。',ha='center');save(fig,'strong_baseline_validation')
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for d,color in [('OpenHands','#0571b0'),('AgentX','#00865a')]:
    sub=[r for r in struct if r['dataset']==d]
    axes[0].plot([25,50,75],[100*r['binding']/r['cases'] for r in sub],marker='o',label=d,color=color)
    axes[1].plot([25,50,75],[100*r['positive_shared']/r['cases'] for r in sub],marker='o',label=d,color=color)
for ax,title in zip(axes,['K 小于最深候选前沿所需数量','存在正收益共享候选']):
    ax.set(title=title,xlabel='逻辑保留预算比例',ylabel='该预算下快照占比（%）',xticks=[25,50,75],xticklabels=['25%','50%','75%'],ylim=(-2,102));ax.grid(alpha=.2);ax.legend()
fig.text(.5,.01,'基于冻结兼容矩阵的结构统计；不证明原生共享事件或物理容量竞争。三个预算不是独立样本。',ha='center',fontsize=9);fig.tight_layout(rect=(0,.06,1,1));save(fig,'budget_workload_discriminativeness')
for path,digest in sources.items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
jout('source_integrity.json',{'说明':'输入文件读取前后 SHA-256 一致；仅写入本交付目录。','source_count':len(sources),'sha256':sources})
jout('verification.json',{'说明':'只汇总既有结果，未运行策略或实验。','allocation_cases':len(cases),'matrix_rows':len(matrix),'per_case_method_rows':len(long),'exact_cases':len(exact),'strong_baselines':strong,'runtime_records_per_batch':216,'runtime_requests_per_batch':864,'source_integrity':'PASS'})
print(json.dumps({'exact':exact_summary,'strong':strong,'structure':struct,'ttft':rq4means},ensure_ascii=False,indent=2))
