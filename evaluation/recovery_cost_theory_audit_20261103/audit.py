"""独立复算冻结恢复成本数据和连续域性质，不拟合参数、不运行GPU。"""
import ast
from bisect import bisect_left
from collections import Counter, defaultdict
from decimal import Decimal
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FREEZE=ROOT/'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020'
POSITION=ROOT/'evaluation/runtime_artifacts/recovery_position_audit_20260826_144654_852303'
LONG=ROOT/'evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133712_596813'
SOURCES={}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def track(p):
    p=Path(p)
    if not p.is_absolute():p=ROOT/p
    if str(p) not in SOURCES:SOURCES[str(p)]={'bytes':p.stat().st_size,'sha256_before':sha(p)}
    return p


def rows(p):
    with track(p).open() as f:return list(csv.DictReader(f))
def obj(p):return json.loads(track(p).read_text())
def dump(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def outcsv(name,rr):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)


def summarize(raw):
    """所有非预热样本均须有效，不剔除不利点。"""
    ids=[r['case_id'] for r in raw];assert len(ids)==len(set(ids))
    groups=defaultdict(list);warm=Counter()
    for r in raw:
        assert r['status']=='PASS' and r['correctness_pass']=='True' and r['gap_match']=='True'
        assert r['fa_preserved']=='True' and r['safety_pass']=='True'
        t,g=int(r['target_H']),int(r['target_gap'])
        assert (int(r['runtime_H']),int(r['runtime_E']),int(r['runtime_G']))==(t,t-g,g)
        assert int(r['request_input_tokens'])==t+64
        if r['is_warmup']=='True':warm[t,g]+=1
        else:groups[t,g].append(float(r['ttft_ms']))
    result={}
    for (t,g),values in groups.items():
        assert len(values)==12 and warm[t,g]==2
        baseline=st.mean(groups[t,0]);value=st.mean(values)
        result[t,g]={'target_tokens':t,'gap_tokens':g,'baseline_ms':baseline,'mean_ttft_ms':value,
                     'measured_phi_ms':0.0 if not g else value-baseline,'n_measured':12,'n_warmup':2,
                     'std_ttft_ms':st.stdev(values)}
    return result


def predict(g,t,coef):
    gg,tt=g/1024,t/1024
    return coef['a']*gg+coef['b']*gg*tt+coef.get('c',0)*gg*gg


def m0(g,knots):
    x=[k['gap_ki'] for k in knots];y=[k['cost_ms'] for k in knots];g=g/1024
    i=bisect_left(x,g)
    if i<len(x) and x[i]==g:return y[i]
    assert 0<i<len(x)
    return y[i-1]+(y[i]-y[i-1])*(g-x[i-1])/(x[i]-x[i-1])


def main():
    # 从实际生产声明读取参数，不导入或调用拟合/实验入口。
    model_text=track('flowstate/recovery_model.py').read_text();tree=ast.parse(model_text)
    meta=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FORMAL_RECOVERY_MODEL_METADATA' for x in n.targets))
    production={k.arg:ast.literal_eval(k.value) for k in meta.keywords}
    coef={'a':production['coefficient_a'],'b':production['coefficient_b'],'c':production['coefficient_c']}
    models=obj(FREEZE/'candidate_models.json');selection=obj(FREEZE/'model_selection.json');config=obj(FREEZE/'config.json')
    assert sha(FREEZE/'candidate_models.json')==selection['candidate_models_hash_before_heldout']==selection['candidate_models_hash_after_heldout']
    assert models['heldout_used_for_fit'] is False and selection['heldout_used_for_fit'] is False
    train=rows(FREEZE/'calibration_points.csv');raw=rows(FREEZE/'heldout_raw.csv');stored=rows(FREEZE/'heldout_predictions.csv')
    held=summarize(raw);ps=summarize(rows(POSITION/'position_matrix_raw.csv'));ls=summarize(rows(LONG/'raw_trials.csv'))
    assert len(train)==20 and len(stored)==12 and len(raw)==210 and len(held)==15
    train_pairs={(int(r['target_tokens']),int(r['gap_tokens'])) for r in train}
    test_pairs={(int(r['target_tokens']),int(r['gap_tokens'])) for r in stored}
    assert len(train_pairs)==20 and len(test_pairs)==12 and not train_pairs&test_pairs
    assert {t for t,g in train_pairs}.isdisjoint({t for t,g in test_pairs})
    for r in train:
        t,g=int(r['target_tokens']),int(r['gap_tokens']);source=ps if r['subset']=='position_matrix' else ls
        assert abs(source[t,g]['measured_phi_ms']-float(r['measured_phi_ms']))<1e-8
    recorded_summary=rows(FREEZE/'heldout_summary.csv')
    for r in recorded_summary:
        actual=held[int(r['target_tokens']),int(r['gap_tokens'])]
        assert abs(actual['measured_phi_ms']-float(r['measured_phi_ms']))<1e-8
        assert abs(actual['baseline_ms']-float(r['baseline_mean_ms']))<1e-8
    # M0结点均值定义核对，不重新回归或产生新模型。
    grouped=defaultdict(list)
    for r in train:grouped[int(r['gap_tokens'])].append(float(r['measured_phi_ms']))
    for k in models['M0']['parameters']['knots']:
        g=round(k['gap_ki']*1024)
        assert abs(k['cost_ms']-(st.mean(grouped[g]) if g else 0))<1e-8
    predictions=[]
    for r in stored:
        t,g=int(r['target_tokens']),int(r['gap_tokens']);actual=held[t,g]['measured_phi_ms']
        p0=m0(g,models['M0']['parameters']['knots']);p1=predict(g,t,models['M1']['parameters']);p2=predict(g,t,models['M2']['parameters']);pp=predict(g,t,coef)
        assert abs(p0-float(r['M0_prediction_ms']))<1e-8 and abs(p2-float(r['M2_prediction_ms']))<1e-8
        predictions.append({'T_tokens':t,'G_tokens':g,'E_tokens':t-g,'measured_phi_ms':actual,
            'same_T_zero_G_baseline_ms':held[t,g]['baseline_ms'],'M0_ms':p0,'M1_ms':p1,'M2_fitted_ms':p2,'M2_production_ms':pp,
            'M0_signed_error_ms':p0-actual,'M2_signed_error_ms':p2-actual,
            'M0_absolute_error_ms':abs(p0-actual),'M2_absolute_error_ms':abs(p2-actual),
            'M0_APE_percent':abs(p0-actual)/abs(actual)*100,'M2_APE_percent':abs(p2-actual)/abs(actual)*100,
            'M2_minus_M0_prediction_ms':p2-p0,'absolute_error_improvement_ms':abs(p0-actual)-abs(p2-actual),
            'production_rounding_delta_ms':pp-p2})
    metrics={}
    for label,key in [('M0','M0_ms'),('M1','M1_ms'),('M2_fitted','M2_fitted_ms'),('M2_production','M2_production_ms')]:
        errors=[abs(r[key]-r['measured_phi_ms']) for r in predictions]
        relative=[abs(r[key]-r['measured_phi_ms'])/abs(r['measured_phi_ms'])*100 for r in predictions]
        metrics[label]={'MAE_ms':st.mean(errors),'MAPE_percent':st.mean(relative),'max_APE_percent':max(relative),
                        'n_test_configurations':12}
    # 全量固定G与固定T比较，不事后选择有利子集。
    position_pairs=[];gap_steps=[];curvature=[]
    for g in sorted({g for t,g in ps if g>0}):
        tt=sorted(t for t,gg in ps if gg==g)
        for t1,t2 in zip(tt,tt[1:]):
            v1,v2=ps[t1,g]['measured_phi_ms'],ps[t2,g]['measured_phi_ms']
            position_pairs.append({'G':g,'T_lower':t1,'T_upper':t2,'phi_lower_ms':v1,'phi_upper_ms':v2,
                                   'delta_ms':v2-v1,'cost_increases':v2>v1})
    for label,data in [('position_train_source',ps),('long_gap_train_source',ls),('heldout',held)]:
        for t in sorted({t for t,g in data}):
            gg=sorted(g for tt,g in data if tt==t);slopes=[]
            for g1,g2 in zip(gg,gg[1:]):
                v1,v2=data[t,g1]['measured_phi_ms'],data[t,g2]['measured_phi_ms'];slope=(v2-v1)/((g2-g1)/1024)
                gap_steps.append({'source':label,'T':t,'G_lower':g1,'G_upper':g2,'phi_lower_ms':v1,'phi_upper_ms':v2,
                                  'delta_ms':v2-v1,'slope_ms_per_Ki':slope,'cost_increases':v2>v1})
                slopes.append((g1,g2,slope))
            for x,y in zip(slopes,slopes[1:]):curvature.append({'source':label,'T':t,'first_G_interval':str(x[:2]),'second_G_interval':str(y[:2]),
                 'first_slope_ms_per_Ki':x[2],'second_slope_ms_per_Ki':y[2],'slope_increases':y[2]>x[2]})
    # 实际分配点覆盖：Formal573全部合法边/空集，Strict20合法边/空集和主GPU实际点。
    points=set()
    with track('evaluation/gate1_step1_allocation_equivalence/matrices.jsonl').open() as f:
        for line in f:
            m=json.loads(line)
            for p,t in enumerate(m['targets']):
                points.add((t,t))
                for i,pos in enumerate(m['positions']):
                    if m['compatibility'][p][i]:points.add((t-pos,t))
    roster=rows('evaluation/paper_strict_evidence_extraction/strict_allocation_matrix.csv')
    for r in roster:
        if r['condition_set']!='strict20' or r['policy']!='Exact':continue
        cs=json.loads(r['candidates_roster'])
        for p in json.loads(r['per_pending_EG']):
            points.add((p['T'],p['T']))
            for c in cs:
                if p['pid'] in c['compat']:points.add((p['T']-c['L'],p['T']))
    for tag in ('fs','mc','kv'):
        s=obj(f'evaluation/p0_control_cycle_gpu/arms/m1_ck3_{tag}/run/summary.json')
        for r in s['pending_rows']:points.add((r['G'],r['T']))
    assert all(0<=g<=t<=production['maximum_target_tokens'] for g,t in points)
    aa,bb,cc=[Decimal(str(coef[k])) for k in ('a','b','c')]
    tau_max=Decimal(production['maximum_target_tokens'])/1024
    corners=[(Decimal(0),Decimal(0)),(Decimal(0),tau_max),(tau_max,tau_max)]
    derivatives=[aa+bb*t+2*cc*g for g,t in corners]
    # 导数是线性函数，闭三角形极值在顶点；同时给出任意tau的解析下界。
    math_result={'coefficient_a':str(aa),'coefficient_b':str(bb),'coefficient_c':str(cc),
        'b_plus_2c':str(bb+2*cc),'b_plus_c':str(bb+cc),'tau_max':str(tau_max),
        'derivative_at_vertices':[{ 'g':str(g),'tau':str(t),'dPhi_dg':str(d)} for (g,t),d in zip(corners,derivatives)],
        'global_min_derivative_ms_per_Ki':str(min(derivatives)),
        'zero_gap_exact':True,'nonnegative_analytic':aa>0 and bb+cc>=0,
        'monotonic_analytic':aa>0 and bb+2*cc>=0,'actual_allocation_unique_points':len(points),
        'allocation_T_min':min(t for g,t in points),'allocation_T_max':max(t for g,t in points),
        'allocation_G_min':min(g for g,t in points),'allocation_G_max':max(g for g,t in points),
        'all_actual_points_inside_domain':True,'minus_two_c_over_b':float(-2*cc/bb)}
    native=obj(OUT/'native_sources.json');prior=obj('evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs/source_identity.json')['runtime_sources']
    compared=[]
    for f in native['files']:
        if f['path'] in prior:
            assert f['sha256']==prior[f['path']]
            compared.append(f['path'])
    # 只读记录一条Native实际执行的区间、源状态和延迟观测证据。
    trace=obj('evaluation/runtime_control_closure_r2c1b2/strict/native_trace.json')
    outcsv('heldout_prediction_recheck.csv',predictions)
    outcsv('heldout_baseline_recheck.csv',list(held.values()))
    outcsv('same_G_position_pairs.csv',position_pairs)
    outcsv('same_T_gap_steps.csv',gap_steps)
    outcsv('finite_slope_changes.csv',curvature)
    outcsv('allocation_domain_points.csv',[{'G_tokens':g,'T_tokens':t} for g,t in sorted(points)])
    dump('recheck_results.json',{'metrics':metrics,'production_metadata':production,'frozen_full_precision':models['M2']['parameters'],
        'rounding_prediction_max_abs_ms':max(abs(r['production_rounding_delta_ms']) for r in predictions),
        'train_count':20,'test_nonzero_configurations':12,'test_raw_trials':len(raw),'test_measured_trials':sum(r['is_warmup']=='False' for r in raw),
        'position_raw_trials':sum(x['n_measured']+x['n_warmup'] for x in ps.values()),
        'long_raw_trials':sum(x['n_measured']+x['n_warmup'] for x in ls.values()),
        'train_test_T_disjoint':True,'train_test_TG_disjoint':True,'parameter_hash_unchanged_across_heldout':True,
        'heldout_used_for_parameter_fit':False,'heldout_used_for_model_selection':True,
        'M2_absolute_error_better_count':sum(r['M2_absolute_error_ms']<r['M0_absolute_error_ms'] for r in predictions),
        'same_G_adjacent_position_pairs':len(position_pairs),'same_G_increasing_count':sum(x['cost_increases'] for x in position_pairs),
        'fixed_T_gap_steps':len(gap_steps),'fixed_T_gap_increasing_count':sum(x['cost_increases'] for x in gap_steps),
        'finite_slope_changes':len(curvature),'finite_slope_increases':sum(x['slope_increases'] for x in curvature),
        'native_source_hashes_matching_prior':compared,'mathematics':math_result,
        'verdict':'PARTIAL','说明':'预测和数学性质通过；未将TTFT位置效应隔离归因到特定组件或Kernel。'})
    # 其他只读代码、环境、结构与现代范围限制证据。
    for name in ['evaluation/recovery_model_freeze.py','evaluation/recovery_profiler_128k.py','evaluation/recovery_position_audit.py',
                 'evaluation/sota_latency_runtime.py','evaluation/controlled_multiworkflow_v1/runtime_gate.py',
                 'evaluation/runtime_control_closure_r2c1b2/observation.py','evaluation/runtime_control_closure_r2c1b2/req_api_audit.md',
                 'evaluation/runtime_control_closure_r2c1b2/strict/summary.json',
                 'evaluation/serving_control_gpu_gate_20261102/arms/A_shadow_validation/run/post_request.json',
                 'evaluation/paper_recovery_model_short_context_audit/partB_absolute.csv']:
        track(name)
    for name in ['config.json','environment.txt','server_command.txt','execution_order.json','structural_validation.json','theory_compatibility.md']:track(FREEZE/name)
    for path,s in SOURCES.items():s['sha256_after']=sha(path);assert s['sha256_before']==s['sha256_after'],path
    dump('source_manifest.json',{'files':SOURCES,'说明':'源文件只读且读取前后摘要一致；没有重新拟合或运行GPU。'})
    print('数值与数学检查通过；机理归因裁决PARTIAL。')


if __name__=='__main__':main()
