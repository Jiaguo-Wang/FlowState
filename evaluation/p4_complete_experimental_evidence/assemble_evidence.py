"""读取既有证据，归档原图并汇总结果；不导入任何实验或策略模块。"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/flowstate-p4g2-mpl')
import csv,json,hashlib,statistics as st,re,io
from pathlib import Path
from collections import defaultdict,Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
S={}; checks={}; parts={}
def read(path):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 raw=p.read_bytes();key=str(p);digest=hashlib.sha256(raw).hexdigest()
 if key in S: assert S[key]==digest
 S[key]=digest;return raw
def txt(path):return read(path).decode()
def js(path):return json.loads(read(path))
def jl(path):return [json.loads(x) for x in txt(path).splitlines() if x.strip()]
def rows(path):return list(csv.DictReader(io.StringIO(txt(path))))
def savejson(path,data):(OUT/path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def csvout(path,data):
 with (OUT/path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def archive(src,dest):
 (OUT/dest).write_bytes(read(src));return str(OUT/dest)
def fmt(x):
 if isinstance(x,float):return f'{x:.6f}'
 return str(x).replace('|','/').replace('\n',' ')
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in data])
paths={
 '01_semantics':'motivation/artifacts/runtime_validation_gap_replay_20260819',
 '01_direct':'motivation/artifacts/wp3a_direct_eviction_20260819',
 '01_gate':'evaluation/runtime_artifacts/openhands_recurrent_only_causal_gate_20260831_052756_001767',
 '01_sota':'evaluation/runtime_artifacts/sota_correctness_20260825_104513_833972',
 '02_qwen':'motivation/artifacts/replay_cost_20260819',
 '02_long':'evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133712_596813',
 '02_failed':'evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133234_448476',
 '02_position':'evaluation/runtime_artifacts/recovery_position_audit_20260826_144654_852303',
 '02_kimi':'evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555',
 '03_model':'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020',
 '03_audit':'evaluation/rq2_model_audit_output/20260909_055110',
 '04_collection':'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017',
 '04_oh':'evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011',
 '04_ax':'rq3_agentx_policy_eval_20260906_214956',
 '04_g1':'evaluation/p4_evaluation_restructuring',
 '05_p0':'evaluation/p0_allocation_mechanism_audit',
 '05_p1b':'evaluation/p1_strong_baseline_validation',
 '05_ablation':'evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726',
 '06_ttft':'rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652',
 '07_fine':'evaluation/rq6_overhead_output/rq6_overhead_20260909_061500',
 '07_diagnostic':'evaluation/rq6_runtime_bottleneck_output/rq6d_20260910_01',
 '07_batch':'evaluation/rq6_batched_control_output/rq6e_20260911_085820',
 '07_profile':'evaluation/control_path_optimization_output/control_20260929/profiling',
 '08_initial':'evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928',
 '08_optimized':'evaluation/control_path_optimization_output/control_20260929/optimized_formal',
 '08_serving':'evaluation/serving_path_e2e_output/serving_20260929',
 '09_capacity':'evaluation/runtime_artifacts/openhands_4workflow_occupancy_calibration_20260831_050831_071744',
 '09_size':'motivation/artifacts/checkpoint_size_20260819',
 '09_multiround':'evaluation/runtime_artifacts/openhands_final_multi_round_20260903_025836_200373',
 '09_kimi1':'evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043',
 '09_kimi2':'evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527',
 '09_kimifreeze':'evaluation/rq5b_kimi_final_output/rq5b_frozen_20260923_090154',
 '09_p2':'evaluation/p2_shared_cohort_feasibility_audit',
 '09_p3g0':'evaluation/p3_native_instrumentation_feasibility',
 '09_p3g1':'evaluation/p3_physical_retention_budget_contract',
 '09_kvflow':'evaluation/rq3_kvflow_controlled_20260907_123928',
 '09_boundary':'motivation/artifacts/workflow_boundary_value_20260819',
 '09_legacy':'evaluation/runtime_artifacts/sota_latency_20260825_113526_839592',
 '09_rerun':'evaluation/runtime_artifacts/formal_model_latency_rerun_20260827_015154_313413',
 '09_v2':'evaluation/recovery_profiler_v2',
 '09_tracelab':'evaluation/public_agent_trace/formal_policy_results_20260827_075548_356403',
 '09_regression':'evaluation/formal_model_regression',
}
# 归档原图逐字节复制，不执行旧绘图脚本。
figures=[]
for key,names in [('04_g1',['overall_allocation_quality','strong_baseline_validation','budget_workload_discriminativeness']),('02_qwen',['plot_replay_latency']),('01_direct',['plot_fork_eviction_latency']),('09_boundary',['plot_boundary_value'])]:
 for name in names:
  for ext in ['png','pdf']:
   src=paths[key]+'/'+name+'.'+ext
   if (ROOT/src).exists():
    archive(src,'figures/'+name+'.'+ext);figures.append(dict(figure=name,kind='原图逐字节归档',source=str(ROOT/src),output='figures/'+name+'.'+ext))
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
def figsave(fig,name,source):
 for ext in ['png','pdf']:
  fig.savefig(OUT/'figures'/(name+'.'+ext),bbox_inches='tight',dpi=160)
  figures.append(dict(figure=name,kind='冻结数值CPU重绘',source=source,output='figures/'+name+'.'+ext))
 plt.close(fig)
q=rows(paths['02_qwen']+'/replay_cost.csv');archive(paths['02_qwen']+'/replay_cost.csv','tables/qwen_replay_summary.csv')
rawq=rows(paths['02_qwen']+'/raw_runs.csv');checks['qwen_replay_raw_rows']=len(rawq)
parts['QWEN_TABLE']=table(['G（tokens）','有效重复','TTFT均值（ms）','中位数','P25','P75','请求完成中位数'],[[r[k] for k in ['expected_replay_tokens','n_valid','recovery_latency_mean_ms','recovery_latency_median_ms','recovery_latency_p25_ms','recovery_latency_p75_ms','request_e2e_latency_median_ms']] for r in q])
for key,name in [('02_long','summary.csv'),('02_position','position_matrix_summary.csv')]:
 r=rows(paths[key]+'/'+name);archive(paths[key]+'/'+name,'tables/'+key+'.csv')
 parts[key.upper()+'_TABLE']=table(['T（tokens）','G（tokens）','测量次数','TTFT均值（ms）','中位数','标准差','相对同T基线成本（ms）'],[[x.get('target_position',131072),x['gap_tokens'],x['valid_measured_count'],x['mean_ms'],x['median_ms'],x['std_ms'],x['measured_phi_ms']] for x in r])
k=js(paths['02_kimi']+'/summary.json');assert k['valid_pairs']==5
parts['KIMI_TABLE']=table(['配对','顺序','A恢复（ms）','B恢复（ms）','B−A（ms）'],[[x['pair'],'→'.join(x['order']),x['A_resume_ms'],x['B_resume_ms'],x['paired_difference_ms']] for x in k['pairs']])
csvout('tables/kimi_pairs.csv',[{t:r[t] for t in ['pair','A_resume_ms','B_resume_ms','paired_difference_ms','status']} for r in k['pairs']])
fig,ax=plt.subplots(figsize=(7,4));ax.plot(range(1,6),[x['A_resume_ms'] for x in k['pairs']],'o-',label='A：G=0');ax.plot(range(1,6),[x['B_resume_ms'] for x in k['pairs']],'s-',label='B：G=8K');ax.set(xlabel='独立配对编号',ylabel='内部恢复路径时间（ms）',title='Kimi TP=2：每条件取两rank中较大值',xticks=list(range(1,6)),ylim=(0,None));ax.legend();ax.grid(alpha=.2);figsave(fig,'kimi_recovery_pairs',str(ROOT/paths['02_kimi']/'summary.json'))
model=js(paths['03_model']+'/model_selection.json');pred=rows(paths['03_model']+'/heldout_predictions.csv');cal=rows(paths['03_model']+'/calibration_points.csv');hr=rows(paths['03_model']+'/heldout_raw.csv')
assert len(pred)==12 and len(cal)==20 and len(hr)==210
for f in ['heldout_predictions.csv','heldout_summary.csv','calibration_points.csv']:archive(paths['03_model']+'/'+f,'tables/'+f)
parts['MODEL_TABLE']=table(['模型','MAE（ms）','MAPE（%）','最大绝对误差（ms）','最大相对误差（%）','R²','门禁'],[[m]+[model['heldout_metrics'][m]['overall'][z] for z in ['mae_ms','mape_percent','max_absolute_error_ms','max_relative_error_percent','r_squared']]+[model['model_statuses'][m]] for m in ['M0','M1','M2']])
parts['CALIBRATION_TABLE']=table(['来源','T（tokens）','G（tokens）','校准增量成本（ms）'],[[r['subset'],r['target_tokens'],r['gap_tokens'],r['measured_phi_ms']] for r in cal])
parts['HELDOUT_TABLE']=table(['T','G','实测增量（ms）','M0预测','M1预测','M2预测','M2相对误差（%）'],[[r['target_tokens'],r['gap_tokens'],r['measured_phi_ms'],r['M0_prediction_ms'],r['M1_prediction_ms'],r['M2_prediction_ms'],100*float(r['M2_relative_error'])] for r in pred])
for m in ['M0','M1','M2']:
 mae=st.mean(float(x[m+'_absolute_error_ms']) for x in pred);mape=100*st.mean(float(x[m+'_relative_error']) for x in pred)
 assert abs(mae-model['heldout_metrics'][m]['overall']['mae_ms'])<1e-8 and abs(mape-model['heldout_metrics'][m]['overall']['mape_percent'])<1e-8
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for m,color,mark in [('M0','#a6611a','s'),('M1','#0571b0','^'),('M2','#00865a','o')]:
 axs[0].scatter([float(r['measured_phi_ms']) for r in pred],[float(r[m+'_prediction_ms']) for r in pred],label=m,color=color,marker=mark)
 axs[1].plot(range(12),[100*float(r[m+'_relative_error']) for r in pred],marker=mark,label=m,color=color)
axs[0].plot([0,6000],[0,6000],'--',color='gray',label='预测等于实测');axs[0].set(xlabel='同T基线扣除后的实测增量（ms）',ylabel='模型预测（ms）',title='12个非零留出配置');axs[1].set(xlabel='T/G（Ki tokens）',ylabel='绝对相对误差（%）',title='逐配置误差，非逐请求样本');axs[1].set_xticks(range(12),[r['target_ki']+'/'+r['gap_ki'] for r in pred],rotation=65)
for a in axs:a.grid(alpha=.2);a.legend()
fig.tight_layout();figsave(fig,'cost_model_heldout',str(ROOT/paths['03_model']/'heldout_predictions.csv'))
# 复用P4-G1，不再次运行任何选择规则。
for f in ['overall_allocation_matrix.csv','per_case_evidence.csv','exact_opt_subset_quality.csv','strong_baseline_summary.csv','ablation_summary.csv','workload_structure.csv']:
 archive(paths['04_g1']+'/'+f,'tables/'+f)
parts['ALLOCATION_TABLE']=txt(paths['04_g1']+'/allocation_tables.md').split('\n',4)[-1]
parts['EXACT_TABLE']=table(['数据集','预算','匹配数','有Exact结果的条件数'],[[r['dataset'],r['budget_ratio'],r['matches'],r['tractable_cases']] for r in rows(paths['04_g1']+'/exact_opt_subset_quality.csv')])
case=txt(paths['05_p0']+'/case_studies.md')
case='\n'.join(line for line in case.splitlines() if not line.startswith('英文论文解释：'))
case=re.sub(r'\]\((case_[A-G]\.json)\)',lambda m:']('+str(ROOT/paths['05_p0']/m[1])+')',case)
case=case.replace('# 代表性案例：冻结分配结果机制审计','### 5.4 七个代表案例的完整记录（P0冻结表归档）').replace('\n## 案例','\n#### 案例')
parts['CASES']=case
rroot=ROOT/paths['06_ttft'];rr=[]
for p in sorted((rroot/'runs').glob('*/pending_requests.jsonl')):rr+=jl(p)
assert len(rr)==864
csvout('tables/rq4_raw_requests.csv',[{k:r.get(k) for k in ['snapshot_id','policy','repetition','rid','ttft_ms','request_latency_ms','status']} for r in rr])
paired=js(paths['06_ttft']+'/analysis_snapshot_level/paired_ttft_results.json')
rsummary=[]
for policy in ['LRU','Marconi','FlowState']:
 groups=defaultdict(list)
 for r in rr:
  if r['policy']==policy:groups[r['snapshot_id'],r['repetition']].append(r['ttft_ms'])
 assert len(groups)==72 and all(len(v)==4 for v in groups.values())
 snap=defaultdict(list)
 for (sid,rep),v in groups.items():snap[sid].append(st.mean(v))
 rsummary.append([policy,len(snap),len(groups),288,st.mean(st.mean(v) for v in snap.values())])
parts['TTFT_TABLE']=table(['方法','快照','生命周期','请求','TTFT均值（ms）'],rsummary)
parts['TTFT_PAIRED']=table(['对照','均值改善（ms）','中位改善（ms）','95% CI','W/T/L'],[[p,x['mean_absolute_reduction'],x['median_absolute_reduction'],x['mean_absolute_reduction_95_ci'],'/'.join(str(sum(r['outcome']==o for r in x['rows'])) for o in ['win','tie','loss'])] for p,x in paired['comparisons'].items()])
fig,ax=plt.subplots(figsize=(7,4));ax.bar([r[0] for r in rsummary],[r[-1] for r in rsummary],color=['#b9770e','#7b3294','#00865a']);ax.set(ylabel='客户端首令牌延迟（ms）',title='RQ4：24个快照等权均值；控制路径不在计时内');figsave(fig,'rq4_ttft',str(rroot/'analysis_snapshot_level/paired_ttft_results.json'))
# 原控制、批处理及E2E批次分别读取，绝不合并计时样本。
control=[]
for key in ['07_fine','07_batch']:
 data=jl(paths[key]+'/raw/per_epoch_overhead.jsonl');rs=[r for r in data if r.get('record_kind')=='runtime_control']
 assert len(rs)==72
 for name in ['introspection','construction','allocation','reconciliation','total_control']:
  vs=[r['timings_ns'][name]/1e6 for r in rs]
  control.append(dict(batch=key,metric=name,n=72,mean_ms=st.mean(vs),median_ms=st.median(vs),max_ms=max(vs)))
csvout('tables/control_raw_aggregate.csv',control)
parts['CONTROL_TABLE']=table(['批次','阶段','生命周期','均值（ms）','中位数（ms）','最大值（ms）'],[[r[k] for k in ['batch','metric','n','mean_ms','median_ms','max_ms']] for r in control])
e2etable=[];e2epaired=[];neg=[];summaries={};e2eraw=[]
for key,label in [('08_initial','初始E2E'),('08_optimized','优化E2E')]:
 summ=js(paths[key]+'/analysis/summary.json');summaries[label]=summ
 records=[js(p) for p in sorted((ROOT/paths[key]/'runs').glob('*/record.json'))];assert len(records)==216
 for r in records:
  for j,q in enumerate(r['requests'],1):
   e2eraw.append(dict(batch=label,snapshot_id=r['snapshot_id'],policy=r['policy'],repetition=r['repetition'],request_ordinal=j,submit_ts=q['submit_ts'],first_token_ts=q['first_token_ts'],completion_ts=q['completion_ts'],ttft_ms=q['ttft_ms'],request_latency_ms=q['request_latency_ms'],epoch_start_ts=r['epoch_start_ts'],epoch_end_ts=r['epoch_end_ts']))
 for policy,v in summ['means'].items():
  rs=[r for r in records if r['policy']==policy]
  assert len(rs)==72
  for m in v:assert abs(st.mean(r[m] for r in rs)-v[m])<1e-6
  e2etable.append([label,policy,v['control_inclusive_epoch_latency_ms'],v['mean_barrier_to_completion_ms'],v['mean_request_execution_ms'],v['total_control_latency_ms'],v['between_request_validation_and_dispatch_ms']])
 for base,x in summ['paired'].items():
  r=x['control_inclusive_epoch_latency_ms'];e2epaired.append([label,base,r['mean_difference_ms'],r['median_difference_ms'],r['bootstrap_95_ci_ms'],f"{r['win']}/{r['tie']}/{r['loss']}"])
  for c in r['non_winning_cases']:neg.append(dict(batch=label,baseline=base,**c))
parts['E2E_TABLE']=table(['批次','方法','epoch（ms）','平均屏障到完成','平均请求执行','总控制','剩余时间UNRESOLVED'],e2etable)
parts['E2E_PAIRED']=table(['批次','对照','均值改善（ms）','中位改善','95% CI','W/T/L'],e2epaired)
parts['E2E_NEGATIVE']=table(['批次','对照','快照','round','baseline−FlowState（ms）'],[[r['batch'],r['baseline'],r['snapshot_id'],r['round'],r['baseline_minus_flowstate_ms']] for r in neg])
csvout('tables/e2e_raw_request_timestamps.csv',e2eraw);csvout('tables/e2e_non_winning_cases.csv',neg);savejson('tables/e2e_frozen_summaries.json',summaries)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,(label,summ) in zip(axes,summaries.items()):
 for i,base in enumerate(['LRU','Marconi']):
  r=summ['paired'][base]['control_inclusive_epoch_latency_ms'];m=r['mean_difference_ms'];lo,hi=r['bootstrap_95_ci_ms'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=5)
 ax.axvline(0,color='gray',linestyle='--');ax.set(yticks=[0,1],yticklabels=['对 LRU','对 Marconi'],xlabel='基线−FlowState 的epoch差（ms）',title=label+'：冻结95% CI（独立横轴）',ylim=(-.6,1.6));ax.grid(alpha=.2)
fig.tight_layout();figsave(fig,'e2e_paired_intervals','两个E2E批次各自analysis/summary.json')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,(label,summ) in zip(axes,summaries.items()):
 policies=['LRU','Marconi','FlowState'];bottom=[0,0,0]
 for metric,title in [('common_observation_ms','公共观测'),('policy_observation_ms','专属观测'),('input_construction_ms','输入构造'),('allocation_ms','选择'),('reconciliation_ms','协调')]:
  vs=[summ['means'][p][metric] for p in policies];ax.bar(policies,vs,bottom=bottom,label=title);bottom=[a+b for a,b in zip(bottom,vs)]
 ax.set(title=label+'（独立纵轴）',ylabel='每epoch控制阶段均值（ms）')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5);fig.tight_layout(rect=(0,.07,1,1));figsave(fig,'e2e_control_stages','两个E2E批次各自analysis/summary.json')
legacy=rows(paths['09_rerun']+'/weighted_summary.csv')
parts['EARLY_RUNTIME_TABLE']=table(['受控场景','K','方法','真实测量次数','逻辑加权请求数','TTFT加权均值（ms）','加权中位数','加权P95'],[[r[k] for k in ['scenario','K','policy','measured_case_count','weighted_request_count','ttft_weighted_mean_ms','ttft_weighted_median_ms','ttft_weighted_p95_ms']] for r in legacy])
tlab=rows(paths['09_tracelab']+'/aggregate_by_budget.csv')
parts['TRACELAB_TABLE']=table(['预算','方法','snapshot数','mean modeled ms','median modeled ms','pending EHR'],[[r[k] for k in ['budget_ratio','policy','snapshot_count','mean_total_recovery_cost_ms_per_snapshot','median_total_recovery_cost_ms_per_snapshot','executable_hit_ratio']] for r in tlab])
for key,name in [('09_rerun','weighted_summary.csv'),('09_legacy','summary.csv'),('09_tracelab','aggregate_by_budget.csv'),('09_v2','validation_summary.csv')]:archive(paths[key]+'/'+name,'tables/'+key+'_'+name)
parts['TTFT_NEGATIVE']=table(['对照','快照','round','baseline−FlowState（ms）'],[[base,r['snapshot_id'],r['allocation_round'],r['absolute_reduction']] for base,v in paired['comparisons'].items() for r in v['rows'] if r['outcome']!='win'])
# 索引区分文件存在性与内容核对；遍历目录不意味着逐日志语义认证。
inventory=[];commands=[];provenance=[]
for key,root in paths.items():
 base=ROOT/root
 assert base.exists(),root
 for p in sorted(base.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts:continue
  inventory.append(dict(experiment=key,path=str(p),bytes=p.stat().st_size,inspection='内容读取并记录摘要' if str(p) in S else '存在性索引，未逐行认证'))
  if p.name in ['launch_command.txt','server_command.txt']:
   commands.append('### '+key+'：'+p.name+'\n\n来源：`'+str(p)+'`。以下是历史记录，不代表本轮执行。\n\n```text\n'+txt(p).strip()+'\n```\n')
  if p.parent==base or p.parent.name in ['analysis','correctness','logs','provenance']:
   if p.name.endswith(('.md','.json','.log','.txt')) and p.stat().st_size<200000:
    read(p)
 for typ,patterns in [('配置',['*config*.json','*protocol*.json','*plan*.json']),('原始结果',['*raw*.csv','*raw*.jsonl','*per_case*.jsonl','*per_snapshot*.jsonl','*per_epoch*.jsonl','*requests*.jsonl']),('报告',['*report.md','findings.md','README.md']),('日志',['*.log']),('图表',['*.pdf','*.png']),('完整性与正确性',['*integrity*.json','*manifest*.json','*correctness*.json'])]:
  matches=sorted({str(p) for pattern in patterns for p in base.rglob(pattern) if p.is_file()})
  provenance.append(dict(experiment=key,category=typ,count=len(matches),examples='; '.join(matches[:8]),status='存在，完整列表见artifact_inventory.csv' if matches else 'MISSING：该目录未找到此类独立文件'))
csvout('artifact_inventory.csv',inventory);csvout('evidence_categories.csv',provenance);csvout('figures/figure_sources.csv',figures)
(OUT/'historical_commands.md').write_text('# 历史命令记录\n\n仅转录文件中已有记录，未执行。容器路径保持原文；不能假定在宿主机直接可用。\n\n'+'\n'.join(commands))
parts['ROOTS']=table(['证据编号','完整artifact root'],[[k,str(ROOT/v)] for k,v in paths.items()])
fingerprint_path='/home/wjg/code/newfind/artifacts/exp2-b-PASS-20260812T130323Z/strict_L2048.json'
fingerprint=js(fingerprint_path)
code_paths=['flowstate/adapters/sglang.py','flowstate/optimizer.py','flowstate/recovery_model.py','flowstate/workflow.py','flowstate/state_catalog.py','flowstate/executable_state.py','evaluation/recovery_model_freeze.py','evaluation/recovery_profiler_128k.py','evaluation/recovery_position_audit.py','evaluation/rq3_openhands_neutral_collector.py','evaluation/rq3_formal_policy_evaluation.py','evaluation/rq3_agentx_formal_policy_evaluation.py','evaluation/rq3_frozen_snapshot_evaluator.py','evaluation/sota_policies.py','evaluation/controlled_multiworkflow_v1/policies.py','evaluation/rq4_unified_runtime_harness.py','evaluation/rq4_snapshot_level_analysis.py','evaluation/sota_latency_runtime.py','evaluation/rq6_system_overhead.py','evaluation/rq6_runtime_overhead.py','evaluation/rq6_batched_control.py','evaluation/rq6_bottleneck_diagnosis.py','evaluation/control_path_profile.py','evaluation/control_path_optimized_client.py','evaluation/control_path_codec.py','evaluation/control_path_optimized_e2e.py','evaluation/resume_epoch_e2e.py','evaluation/resume_epoch_e2e_analysis.py','evaluation/control_path_optimization_analysis.py','evaluation/serving_path_gate_qualification.py','evaluation/rq5b_kimi_phase1.py','evaluation/rq5b_kimi_phase2.py','evaluation/rq5b_kimi_phase3.py','evaluation/run_rq5b_kimi_phase3.py','evaluation/agentx_runtime_snapshot_collection.py','tests/runtime/rq6_batched_control_transport.py','tests/runtime/control_path_optimized_transport.py','tests/runtime/control_path_profile_transport.py']
source_rows=[]
for cp in code_paths:
    raw=txt(cp)
    functions=[str(i)+': '+line.strip() for i,line in enumerate(raw.splitlines(),1) if re.match(r'^(?:    )?def ',line)]
    source_rows.append([str(ROOT/cp),'<br>'.join(functions)])
(OUT/'source_code_index.md').write_text('# 源码函数与行号索引\n\n只读当前文件；不将当前代码逐字节等同于全部历史版本。历史运行身份优先使用对应provenance。\n\n'+table(['完整路径','函数及行号'],source_rows)+'\n')
(OUT/'experiment_evidence_paths.md').write_text('# 实验来源路径索引\n\n编号前两位对应手册实验。全部普通文件路径、大小及核对范围见[artifact_inventory.csv](artifact_inventory.csv)；分类存在性与缺项见[evidence_categories.csv](evidence_categories.csv)。不存在原始命令记录时，仅给脚本入口并标记RECONSTRUCTED_FROM_CODE，不伪造shell历史。\n\n'+parts['ROOTS']+'\n\n历史命令见[historical_commands.md](historical_commands.md)。源码函数及行号见[source_code_index.md](source_code_index.md)，指标口径和各批次缺口见手册；图表来源见[figures/figure_sources.csv](figures/figure_sources.csv)。\n\n`/workspace`、`/repo`与`/model`是历史容器内路径。对应仓库文件存在时以本索引宿主机路径为准；模型运行环境仅按冻结配置陈述。本轮未启动容器验证。\n')
specs=[
('1','组件语义','GPU正确性','H/E/G与EXTEND','tokens','4项断言','受控请求','01_semantics','driver_results.json','findings.md','PASS','plot_fork_eviction_latency.png'),
('1','直接驱逐','GPU局部干预','配对TTFT差','ms','fork/normal各5对','配对重复','01_direct','analysis_summary.json','findings.md','PASS；NORMAL差为负','plot_fork_eviction_latency.png'),
('1','OpenHands causal gate','GPU正确性','组件隔离与恢复','布尔','12请求/15census','单生命周期','01_gate','requests.jsonl','summary.json','PASS',''),
('1','SOTA correctness','GPU正确性','E/G与驻留一致性','tokens','16快照/69等价类','快照与实际请求','01_sota','e2e_cases.jsonl','summary.json','PASS',''),
('2','Qwen32K','GPU刻画','服务端TTFT','ms','6条件×5=30','每条件重复','02_qwen','raw_runs.csv','replay_cost.csv','PASS','plot_replay_latency.png'),
('2','Qwen128K失败前序','GPU资格失败','实际E不匹配','tokens','完成1/126','trial','02_failed','raw_trials.csv','config.json','FAIL',''),
('2','Qwen128K正式','GPU刻画','服务端TTFT/增量Phi','ms','9条件×14=126含预热','每条件12测量','02_long','raw_trials.csv','summary.csv','PASS',''),
('2','Qwen位置矩阵','GPU刻画','同T扣基线恢复成本','ms','20条件×14=280；legacy70另批','每条件12测量','02_position','position_matrix_raw.csv','position_matrix_summary.csv','PASS；gap-only假设FAIL',''),
('2','Kimi恢复','GPU配对','rank最大内部resume时间','ms','5对/10引擎','配对重复','02_kimi','paired_measurements.json','summary.json','READY；5/5正','kimi_recovery_pairs.png'),
('3','正式恢复模型','CPU拟合与GPU留出','MAE/MAPE/逐点误差','ms/%','20校准；210留出raw/12非零点','非零配置均值','03_model','heldout_raw.csv','model_selection.json','M0 FAIL/M1 WEAK/M2 PASS','cost_model_heldout.png'),
('4','OpenHands正式采集','GPU中立采集','准入/驻留','groups','200指定/168准入/32失败','4workflow组','04_collection','collection_summary.json','FORMAL_COLLECTION.json','168准入；32失败不补位',''),
('4','OpenHands allocation','CPU冻结评价','mean/median/W-T-L/Full Coverage','modeled ms','168快照×3预算','各预算snapshot配对','04_oh','per_snapshot_results.jsonl','aggregate_results.json','READY','overall_allocation_quality.png'),
('4','AgentX allocation','CPU冻结评价','mean/median/W-T-L/Full Coverage','modeled ms','23快照×3预算','各预算snapshot配对','04_ax','per_case_results.jsonl','aggregate_results.json','READY；原生事件证据不足','overall_allocation_quality.png'),
('5','P0案例','CPU只读机制审计','marginal/最终成本','modeled ms','573条件；A–G七例','快照–预算条件','05_p0','all_cases.jsonl','summary.json','已审计；非GPU性能',''),
('5','P1-B强基线','CPU基线验证','objective/集合/E-G等价','modeled ms/计数','573条件×3强基线','191快照的三预算','05_p1b','per_case_results.jsonl','summary.json','573/573全平','strong_baseline_validation.png'),
('5','Exact OPT','CPU穷举冻结子集','目标匹配','modeled ms','482/573可穷举条件','snapshot–budget条件','04_g1','exact_opt_cases.csv','exact_opt_subset_quality.csv','482/482匹配；91未运行',''),
('5','消融','CPU冻结评价','去位置/去集合依赖差','modeled ms','573条件','每预算snapshot','05_ablation','per_case_results.jsonl','aggregate_results.json','已有冻结结果',''),
('6','RQ4请求TTFT','GPU正式性能','提交至首token','ms','216引擎/864请求','24snapshot先聚合3重复','06_ttft','analysis_snapshot_level/snapshot_level_metrics.csv','analysis_snapshot_level/paired_ttft_results.json','correctness PASS；Marconi有2负例','rq4_ttft.png'),
('7','CPU决策计时','CPU测量','construction+allocation','ms','573条件×3=1719','snapshot–budget重复','07_fine','raw/per_epoch_overhead.jsonl','analysis/overhead_summary.json','PASS',''),
('7','细粒度控制','GPU状态控制计时','直接total及四分段','ms','72引擎','24snapshot×3','07_fine','raw/per_epoch_overhead.jsonl','analysis/overhead_summary.json','72/72 PASS',''),
('7','瓶颈诊断','GPU状态诊断','RPC/主机detach-free','ms','6最终runs；120no-op','6引擎；不把no-op独立化','07_diagnostic','final_diagnostic/diagnostic_summary.json','final_report.md','PARTIAL；初步及失败独立保留',''),
('7','批量控制','GPU状态控制计时','total/RPC','ms/次数','72引擎','24snapshot×3','07_batch','raw/per_epoch_overhead.jsonl','analysis/aggregate_results.json','72/72 PASS',''),
('7','E2E旧路径profiling','GPU诊断','子步骤/调用数','ms/次数','72引擎','24snapshot×3','07_profile','analysis/profile_summary.json','analysis/profiling_report.md','72/72 PASS；不回填旧内部时间','e2e_control_stages.png'),
('8','初始E2E','GPU正式性能','epoch/请求/控制/余量','ms','216引擎/864请求','24snapshot配对','08_initial','analysis/summary.json','analysis/final_report.md','correctness PASS；净收益未建立','e2e_paired_intervals.png'),
('8','优化E2E','GPU正式性能','epoch/请求/控制/余量','ms','216引擎/864请求','24snapshot配对','08_optimized','analysis/summary.json','analysis/final_report.md','correctness PASS；净收益未建立','e2e_paired_intervals.png'),
('8','Serving-path资格','CPU门禁资格','中间事件证据充分性','布尔','GPU 0/216；请求0/864','CPU资格反例','08_serving','gate_qualification.json','final_report.md','BLOCKED_BY_CORRECTNESS_GATE',''),
('9','完整state大小','既有报告及原始fingerprint','conv+temporal payload','bytes/MiB','一次allocation probe','物理slot','09_size','server_probe.log','findings.md','启动日志MISSING；另有原始fingerprint支持49.125MiB',''),
('9','物理headroom','GPU容量校准','驻留/空闲槽','slots','20请求/21census','单受控执行','09_capacity','census.jsonl','summary.json','PASS；非逻辑K强制',''),
('9','Kimi Phase1','TP2正确性','组件隔离/rank一致','布尔','smoke+3独立生命周期','生命周期','09_kimi1','summary.json','final_report.md','READY',''),
('9','Kimi Phase2','TP2接口验证','选择/批量实现','集合/布尔','smoke+3；4候选K2','生命周期','09_kimi2','summary.json','final_report.md','READY；模型成本非Kimi预测',''),
('9','P2','CPU可行性审计','原生证据/容量','非性能','393traces及23冻结快照','来源与结构','09_p2','feasibility_summary.json','final_report.md','NOT_FEASIBLE_FROM_EXISTING_DATA',''),
('9','P3-G0','设计审查','联合事件可行性','非性能','未运行Pilot','源码能力','09_p3g0','missing_primitives.md','final_report.md','PARTIAL_RUNTIME_OR_AGENT_GAP',''),
('9','P3-G1','设计审查','物理保留合同','非性能','未运行Pilot','容量合同','09_p3g1','retention_budget_contract.md','final_report.md','RESOURCE_CONTRACT_NEEDS_EXTENSION',''),
('9','旧受控latency','GPU历史性能','加权客户端TTFT','ms','69类/690测量+138预热','等价类/逻辑multiplicity','09_legacy','raw_samples.jsonl','summary.csv','PASS；两选择改变点不能复用',''),
('9','正式模型定向重跑','GPU独立重跑','加权客户端TTFT','ms','33类/330测量+66预热','受控场景等价类','09_rerun','raw_trials.jsonl','weighted_summary.csv','正式同批结果独立归档',''),
('9','Profiler v2','GPU历史校准','held-out误差','ms/%','9条件×14=126','4留出gap点','09_v2','raw_samples.jsonl','model_comparison.json','PASS；不替代正式位置模型',''),
('9','TraceLab','CPU逻辑快照评价','modeled成本/pending EHR','modeled ms/比例','105主快照/37次级切片','snapshot','09_tracelab','raw_policy_results.csv','aggregate_by_budget.csv','PASS；非真实驻留GPU',''),
('9','KVFlow受控例','CPU设计例','共同目标成本','modeled ms','3pending、K1','单解析受控设计','09_kvflow','final_report.md','final_report.md','READY；非完整KVFlow系统',''),
('9','边界价值','GPU观察加成本估计','reuse count/估计恢复成本','次数/估计ms','单工作流7请求','受控horizon','09_boundary','analysis_summary.json','findings.md','Resume右删失；估计非实测','plot_boundary_value.png')]
idx=[]
for num,name,kind,metric,unit,n,stat,key,raw,agg,status,fig in specs:
    rp=ROOT/paths[key]/raw;ap=ROOT/paths[key]/agg
    idx.append(dict(experiment_id=num,name=name,category=kind,metric=metric,unit=unit,sample_size=n,statistical_unit=stat,artifact_root=str(ROOT/paths[key]),raw_or_detail_path=str(rp),result_file=str(ap),verification_status=status,raw_exists=rp.exists(),result_exists=ap.exists(),figure_path=str(OUT/'figures'/fig) if fig else '不适用；见表格',evidence_scope='既有记录；未在本轮运行实验'))
csvout('experiment_results_index.csv',idx)

template=txt(OUT/'handbook_template.md')
for key,value in parts.items():template=template.replace('{{'+key+'}}',value)
assert not re.search(r'\{\{[A-Z_]+\}\}',template)
(OUT/'complete_experiment_handbook.md').write_text(template)
for p,h in S.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
savejson('source_integrity.json',{'说明':'仅证明本清单已读取文件在本次汇总期间未改变，不代表所有日志逐行通过审计。','status':'PASS','files':len(S),'sha256':S})
savejson('verification.json',{'说明':'仅CPU读数、归档和图表生成；无新策略或实验执行。','qwen_replay_rows':len(rawq),'heldout_raw_rows':len(hr),'heldout_points':len(pred),'calibration_points':len(cal),'kimi_pairs':len(k['pairs']),'rq4_requests':len(rr),'e2e_timestamp_rows':len(e2eraw),'input_integrity':'PASS','figure_files':len(figures),'indexed_files':len(inventory)})
print('证据汇总完成，输入文件数',len(S),'索引文件数',len(inventory))
