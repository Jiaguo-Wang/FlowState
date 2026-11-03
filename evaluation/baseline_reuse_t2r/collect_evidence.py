"""复用冻结CPU结果并提取既有物理映射；不重新运行实验。"""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MAIN=ROOT/'evaluation/kvflow_multiworkflow_cpu_main'
seen={}
def load(path):
    p=Path(path);b=p.read_bytes();seen[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return json.loads(b)
def save(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})

manifest=load(MAIN/'output_manifest.json');verified=[]
for name,h in manifest['files'].items():
    p=MAIN/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual!=h:raise ValueError('冻结CPU产物摘要不符：'+name)
    seen[str(p.relative_to(ROOT))]=actual;verified.append(name)
lock=load(MAIN/'input_lock.json');drift=[]
for name,h in lock['sha256'].items():
    p=Path(name);actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'MISSING'
    if actual!=h:drift.append({'path':name,'frozen_sha256':h,'current_sha256':actual})
    if p.exists():seen[str(p)]=actual
cases=load(MAIN/'workloads.json')['cases']
rows=list(csv.DictReader((MAIN/'cpu_comparison.csv').open()))
for i,r in enumerate(rows,2):
    r['original_csv_line']=i;r['result_origin']='REUSED_FROZEN';r['historical_decision_timing']='UNKNOWN_NOT_RECORDED'
table('cpu_comparison_reused.csv',rows)
priorities=list(csv.DictReader((MAIN/'candidate_priorities.csv').open()));table('candidate_priorities_reused.csv',priorities)
reg=load(OUT/'adapter_regression.json')['results'];t=[]
for r in reg:t.append({'epoch':r['epoch_id'],'K':r['K'],'policy':r['policy'],**r['timing']})
table('adapter_call_diagnostics.csv',t)
# 使用存储切片的独立字节范围，不把整池storage_bytes当成单槽成本。
base=ROOT/'evaluation/c_shared_gpu_pilot_k3/census/run'
census=load(base/'census.json');summary=load(base/'summary.json')
case=next(c for c in cases if c['structure']=='C')
cs={c['checkpoint_id']:c for c in case['candidates']}
phys={c['checkpoint_id']:c for c in census['candidates']}
slices=census['audit_before']['component_slices'];mapping=[]
for m in summary['mapping']:
    c=cs[m['cpu_checkpoint_id']];p=phys[m['physical_checkpoint_id']]
    if c['token_ids']!=p['token_ids'] or c['token_position_L']!=p['token_position_L']:raise ValueError('历史物理Token映射不一致')
    intervals=[(start,start+size) for comp in p['component_ids'] for start,size in slices[comp]['byte_ranges']]
    intervals.sort()
    if any(b>c for (a,b),(c,d) in zip(intervals,intervals[1:])):raise ValueError('组件切片重叠')
    mapping.append({**m,'independent_recurrent_bytes':sum(b-a for a,b in intervals),'physical_epoch':p['barrier_id'],'mapping_evidence':'逐Token相等、位置相等、独立切片范围','reuse_in_new_engine':'FORBIDDEN','provenance':'PROVENANCE_UNVERIFIED'})
assert len(mapping)==16 and len({r['independent_recurrent_bytes'] for r in mapping})==1
save('historical_physical_mapping.json',{'说明':'仅过去Census的身份对照，不是当前可应用绑定。实际字节分母可获得，FLOPs分子仍缺Qwen GDN证据。','rows':mapping,'source':str(base.relative_to(ROOT)),'before_after_unchanged':census['unchanged']})
# 记录原论文实现与现有适配，不移植Mamba-1公式到GatedDeltaNet。
for f in ['evaluation/sota_policies.py','evaluation/sota_metadata.py','evaluation/SOTA_BASELINES.md','evaluation/kvflow_inspired_retention_cpu/policy.py','evaluation/sota_baseline_audit_b5a/raw/marconi_repo/utils.py','evaluation/sota_baseline_audit_b5a/raw/marconi_repo/radix_cache_hybrid.py','evaluation/c_shared_gpu_pilot_k3/control.py','evaluation/c_shared_gpu_pilot_k3/final_report.md','evaluation/c_formal_perf_gpu/protocol.json','flowstate/optimizer.py','flowstate/strict_control.py']:
    seen[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
win={}
for policy in ['Marconi Adaptation','KVFlow-inspired']:
    counts={'win':0,'tie':0,'loss':0}
    for case in cases:
        for k in [3,5]:
            rr={r['policy']:r for r in rows if r['structure']==case['structure'] and int(r['K'])==k}
            diff=float(rr['FlowState-Greedy']['current_J_ms'])-float(rr[policy]['current_J_ms'])
            counts['tie' if abs(diff)<=1e-9 else 'win' if diff<0 else 'loss']+=1
    win[policy]=counts
save('reuse_summary.json',{'gate':'PARTIAL','frozen_cpu_rows':len(rows),'three_policy_rows':18,'regression_conditions':len(reg),'output_manifest_verified':len(verified),'input_lock_drift':drift,'flowstate_vs':win,'historical_physical_candidates':len(mapping),'bytes_per_recurrent_slot':mapping[0]['independent_recurrent_bytes']})
save('gpu_gate.json',{'gate':'BLOCKED','query':'nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv','observed_date':'2026-10-27 Asia/Shanghai','processes':[{'pid':3492611,'process_name':'sglang::scheduler_TP0_EP0','used_memory_MiB':65736}],'action':'未启动Engine、未等待重试、未干预进程','protocol_status':'没有启动新的三策略GPU臂；正式重复协议尚未提交审核'})
for f,h in seen.items():
    if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h:raise ValueError('源文件在整理期间改变：'+f)
save('source_identity.json',{'说明':'冻结输出清单逐项核验；旧输入锁漂移单列，不刷新旧锁。','sha256':seen,'unchanged_during_collection':True,'historical_input_lock_drift':drift})
print(json.dumps(load(OUT/'reuse_summary.json'),ensure_ascii=False,indent=2))
