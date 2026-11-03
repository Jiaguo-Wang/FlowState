"""补齐原始运行复核、Strict20结构和论文逐项对照；只更新本轮派生清单。"""
import csv
import json
import statistics as st
from pathlib import Path
from collections import Counter
from evaluation.icde_v7_evidence_freeze_20261102 import audit as a

OUT=a.OUT
ROOT=a.ROOT


def writecsv(name, rows):
    """仅更新本轮新目录中的尚未最终封存派生表，不写源目录。"""
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    a.SOURCES=json.loads((OUT/'source_manifest.json').read_text())['files']
    master=list(csv.DictReader((OUT/'evidence_master.csv').open()))
    a.MASTER=master
    structures=list(csv.DictReader((OUT/'workload_structure_summary.csv').open()))
    assert len(structures)==1337, '此步骤只允许在本次新输出目录执行一次'
    strict=a.rows('evaluation/greedy_swap_structure_e3_step1/condition_results.csv')
    strictgroups={}
    for r in strict:strictgroups.setdefault((r['case'],int(r['K'])),{})[r['policy']]=r
    rosters=a.rows('evaluation/paper_strict_evidence_extraction/strict_allocation_matrix.csv')
    for r in rosters:
        if r['condition_set']!='strict20' or r['policy']!='Exact':continue
        case,k=r['case'],int(r['K']);cand=json.loads(r['candidates_roster']);ps=json.loads(r['per_pending_EG'])
        matrix={'candidate_ids':[x['id'] for x in cand],'positions':[x['L'] for x in cand],
                'pending_ids':[p['pid'] for p in ps],'targets':[p['T'] for p in ps]}
        matrix['compatibility']=[[p['pid'] in c['compat'] for c in cand] for p in ps]
        matrix['benefit_ms']=[[a.phi(p['T'],p['T'])-a.phi(p['T']-c['L'],p['T']) if p['pid'] in c['compat'] else 0 for c in cand] for p in ps]
        s=a.structure(matrix);assert s['K_min']==4
        group=strictgroups[case,k]
        for row in group.values():
            value,es,gs=a.evaluate(matrix,json.loads(row['selected']));assert abs(value-float(row['J_ms']))<a.EPS
            assert es==[p['E'] for p in json.loads(row['per_pending'])]
        sets={tuple(sorted(json.loads(x['selected']))) for x in group.values()}
        js=[float(x['J_ms']) for x in group.values()]
        structures.append({'domain':'Strict20受控共享分支','condition_id':case+f'::K={k}', 'population':case,
            'snapshot_id':case,'budget_ratio':'N/A','K':k,'candidate_count':7,'pending_count':4,
            'positive_candidates':s['positive'],'shared_candidates':s['shared'],'overlap_pairs':s['overlap'],
            'K_min':4,'tight_budget':k<4,'shared_and_tight':s['shared']>0 and k<4,
            'same_set_all':len(sets)==1,'same_J_all':max(js)-min(js)<=a.EPS,
            'different_set_equal_J':len(sets)>1 and max(js)-min(js)<=a.EPS,
            'J_by_policy':json.dumps({key:float(v['J_ms']) for key,v in group.items()}),
            'oracle_status':'冻结全子集枚举证书','physical_shared_certification':'EPOCH_ADMISSIBLE；未逐边RESUME认证；血缘未认证',
            'source':r['snapshot_source'],'source_row':f'case={case},K={k}'})
    for case in ['w1','w2','w3','w4']:
        raw=a.obj(f'evaluation/independent_runtime_allocation_e2/{case}/strict_epoch_snapshot.json')
        assert len(raw['candidates'])==7 and len(raw['pendings'])==4 and raw['pending_dispatched'] is False
        assert all(x['status']!='UNKNOWN' for x in raw['edges'])
    raw=a.obj('evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json');assert len(raw['candidates'])==7
    # 验证40个原始摘要与派生CSV一致，保留0.001ms导出舍入容差。
    g2=a.rows('evaluation/b5c4_formal_gpu_g2/formal/raw_results.csv')
    for r in g2:
        s=a.obj(f"evaluation/b5c4_formal_gpu_g2/formal/{r['run']}/summary.json")
        assert s['status']=='PASS' and s['cold_output_match'] and len(s['output_ids'])==8
        assert s['E_executable_prefix']==int(r['E']) and s['prefill_token_count']==int(r['prefill_tokens'])
        assert abs(s['ttft_ms']-float(r['ttft_ms']))<=0.000501
        if r['solver']=='Exact':assert s['strict_decision']['solver']=='Exact'
    # C/K3 的另一独立60臂批次，不与论文P0十五臂合并。
    ck=a.rows('evaluation/c_formal_perf_gpu/lifecycle_results.csv');main=[r for r in ck if r['name'].startswith('m')]
    assert len(main)==60 and len(ck)==62
    for r in main:
        s=a.obj(f"evaluation/c_formal_perf_gpu/{r['name']}/run/summary.json")
        assert s['status']=='PASS' and s['cold_output_match'] and len(s['output_ids'])==8
    ckstats={}
    for base in ('SA','WFC'):
        grouped={(int(r['rep']),r['pending'],r['pol']):r for r in main}
        ckstats[base]={}
        for pid in ['C_A','C_B','C_C','C_D']:
            dif=[float(grouped[i,pid,'FS']['ttft_ms'])-float(grouped[i,pid,base]['ttft_ms']) for i in range(1,6)]
            ckstats[base][pid]=st.median(dif)
    # 旧Resume-Epoch全部216份record，直接复核控制分项及完整周期。
    ep=[]
    epochdir=ROOT/'evaluation/control_path_optimization_output/control_20260929/optimized_formal/runs'
    for p in sorted(epochdir.glob('*/record.json')):
        r=a.obj(p);assert r['status']=='PASS' and len(r['requests'])==4
        epoch=r['control_inclusive_epoch_latency_ms'];reqsum=sum(q['request_latency_ms'] for q in r['requests'])
        assert abs(epoch-r['total_control_latency_ms']-reqsum-r['between_request_validation_and_dispatch_ms'])<1e-3
        ep.append(r)
    assert len(ep)==216
    epstats={p:{k:st.mean(r[k] for r in ep if r['policy']==p) for k in ('control_inclusive_epoch_latency_ms','total_control_latency_ms','common_observation_ms','policy_observation_ms','allocation_ms','reconciliation_ms','input_construction_ms')} for p in ('FlowState','LRU','Marconi')}
    # 216历史运行目录与其原正确性文件：不将旧PASS升级成Cold或Strict。
    legacy=a.lines('rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/runs.jsonl')
    missing=[]
    for r in legacy:
        d=ROOT/'rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/runs'/r['run_id']
        for name in ['correctness.json','selection.json','request_telemetry.jsonl','replay.json','runtime.log','result.json','pending_requests.jsonl','handle_mapping.json','reconcile.json']:
            if not (d/name).is_file() or (d/name).stat().st_size==0:missing.append(str(d/name))
        c=a.obj(d/'correctness.json')
    assert not missing
    # Kimi两rank各条件原始14项检查及实际工作量。
    for i in range(1,6):
        for side in ('A','B'):
            r=a.obj(f'evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/runs/pair{i:02d}_{side}/record.json')
            assert r['status']=='PASS' and all(r['checks'].values()) and len(r['per_rank_resume_ms'])==2
    # 读取源代码和既有报告，只核对调用语义，不执行它们。
    for p in ['evaluation/p0_control_cycle_gpu/runner.py','evaluation/p0_control_cycle_gpu/control.py',
              'evaluation/p0_control_cycle_gpu/five_round_R3_audit.md',
              'evaluation/legacy_runtime_evidence_p0_audit/final_report.md',
              'evaluation/agentx_real_trace_gpu_feasibility_audit.md',
              'evaluation/rq3_workload_structure/natural_analysis/scripts/analyze.py',
              'evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/scripts/run_budget.py',
              'evaluation/kvflow_multiworkflow_cpu_main/protocol.json',
              'evaluation/kvflow_multiworkflow_cpu_main/final_report.md',
              'evaluation/p0_control_cycle_gpu/arms/five_round_ledger.jsonl',
              'evaluation/paper_recovery_model_short_context_audit.md',
              'evaluation/paper_workload_structure_final/strict20_optima_nesting.py',
              'evaluation/control_path_separation_20261101/final_report.md',
              'evaluation/control_path_remaining_attribution_20261101/final_report.md',
              'evaluation/serving_control_gpu_gate_20261102/final_report.md']:
        a.track(p).read_text()
    live=a.obj('evaluation/serving_control_gpu_gate_20261102/correctness_evidence.json')
    assert live['status']=='GPU_CORRECTNESS_PASS' and len(live['arms'])==2
    # 本文明确的三个CUDA值与原始同批五轮中位数不一致。
    paper_cuda={'fs':233.5,'mc':700.4,'kv':648.5}
    for r in master:
        for tag,value in paper_cuda.items():
            if r['claim_id']==f'B01_{tag}_sum_cuda_median_ms':
                r['paper_value']=str(value);r['status']='NUMERIC_MISMATCH'
                r['limitation']='附件287；应按同批15臂summary逐轮求和后取中位，不匹配233.5/700.4/648.5'
    # 更多可复核指标，不混用来源或统计分母。
    a.metric('D05','VI-D','比例预算OpenHands共享条件',0,'条件','504 Formal条件；168受控组合','CPU_FORMAL','evaluation/gate1_step1_allocation_equivalence/matrices.jsonl','附件307','0/504',limit='无跨Pending正收益支持不等于真实轨迹没有公共Token历史')
    a.metric('D06','VI-D','比例预算AgentX共享条件',66,'条件','23快照×3比例；共享快照22个','CPU_FORMAL','evaluation/gate1_step1_allocation_equivalence/matrices.jsonl','附件307','66/69',limit='逻辑矩阵共享，非Strict物理复用')
    a.metric('D07','VI-D','固定预算shared且K<K_min',9,'条件','AgentX K1的9快照；其余0','CPU_FORMAL','evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/budget_sensitivity_per_snapshot.csv','附件308','9/10差异来自AgentX',limit='OH g051/K3不共享但J不同0.047745076924ms')
    for name,key in [('观察共用','common_observation_ms'),('观察策略','policy_observation_ms'),('Allocation','allocation_ms'),('Reconciliation','reconciliation_ms'),('输入准备','input_construction_ms')]:
        a.metric('F05_'+key,'VI-F','旧Resume-Epoch '+name,epstats['FlowState'][key],'ms','24快照×3重复；72个FS record均值','GPU_LEGACY_CONTROL','evaluation/control_path_optimization_output/control_20260929/optimized_formal/runs/*/record.json','附件341',limit='观察共用+策略=91.210；不能套用到当前Strict路径')
    a.metric('F06','VI-F','Serving精简接口GPU正确性',2,'生命周期','A/B各1；26驱逐；8请求×8Token','GPU_CORRECTNESS','evaluation/serving_control_gpu_gate_20261102/correctness_evidence.json','正文尚未引用','',limit='仅正确性；两个单次不支持性能收益；全部SHA保留')
    a.metric('C02','VI-C','短上下文预测方向一致',19,'比较','24同Pending比较中19个非零预测差','GPU_DERIVED','evaluation/paper_recovery_model_short_context_audit/partB_pairs.csv','附件301','19/19',limit='三套历史独立协议的分类比较，不是19个工作流或当前P0的19轮')
    a.metric('C03','VI-C','M0/M2 Greedy与Exact选集改变',0,'条件','Strict20+C/K3=21条件','CPU_MODEL_SENSITIVITY','evaluation/paper_recovery_model_short_context_audit/partC_conditions.csv','附件302','0/21',limit='不能从预测精度改善宣称本实例的状态选择收益')
    # 正文隐藏表不能从周围文字猜测单元格。
    a.metric('T01','VI各表','飞书导出隐藏表格',8,'表','全文8个占位；均在Evaluation','DOCUMENT_GAP',str(a.PAPER),'附件278/282/289/297/311/319/337等','',status='UNKNOWN_TABLE_CELLS',limit='需导出表格内容后才能对全部单元格签署一致性')
    writecsv('evidence_master.csv',master)
    writecsv('workload_structure_summary.csv',structures)
    extra={'raw_g2_summaries_verified':40,'raw_ck3_summaries_verified':60,'ck3_separate_paired_medians_FS_minus_baseline_ms':ckstats,
           'raw_legacy_epoch_records_verified':216,'raw_epoch_means':epstats,'raw_legacy_runtime_correctness_files_verified':216,
           'legacy_missing_files':missing,'kimi_raw_rank_records_verified':10,
           'paper_cuda_expected':paper_cuda,'paper_hidden_tables':8,'verdict':'NEEDS_TARGETED_FIX'}
    with (OUT/'supplementary_verification.json').open('x') as f:json.dump(extra,f,ensure_ascii=False,indent=2)
    for path,v in a.SOURCES.items():
        v['sha256_after']=a.sha(Path(path));assert v['sha256_after']==v['sha256_before'],path
    with (OUT/'source_manifest.json').open('w') as f:json.dump({'files':a.SOURCES,'说明':'原文件只读；本轮派生表已补齐Strict20及原始运行交叉核对。'},f,ensure_ascii=False,indent=2)
    print('补充复核完成；主要数字可复现，但三个CUDA数值、隐藏表格与范围限定需定向修复。')


if __name__=='__main__':main()
