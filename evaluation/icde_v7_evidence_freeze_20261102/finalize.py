"""生成论文对照清单与审计报告；不修改论文或任何实验来源。"""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path
from evaluation.icde_v7_evidence_freeze_20261102 import audit as a

OUT=a.OUT
ROOT=a.ROOT


def savecsv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    stats=json.loads((OUT/'audit_statistics.json').read_text())
    extra=json.loads((OUT/'supplementary_verification.json').read_text())
    a.SOURCES=json.loads((OUT/'source_manifest.json').read_text())['files']
    master=list(csv.DictReader((OUT/'evidence_master.csv').open()));a.MASTER=master
    pred=a.rows('evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv')
    for model in ('M0','M2'):
        maximum=max(abs(float(r[model+'_prediction_ms'])-float(r['measured_phi_ms']))/abs(float(r['measured_phi_ms'])) for r in pred)*100
        a.metric('C04_'+model,'VI-C',model+'最大相对误差',maximum,'%', '12非零G测试聚合点','GPU_CALIBRATION','evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv','附件299','28.036' if model=='M0' else '8.179')
    absrows=a.rows('evaluation/paper_recovery_model_short_context_audit/partB_absolute.csv')
    err=[float(r['abs_err_prefill_ms'])/float(r['meas_prefill_ms'])*100 for r in absrows]
    a.metric('C05','VI-C','短上下文10格相对误差分母',[min(err),max(err)],'%', '10个完整CUDA Prefill配置；分母为完整实测Prefill','GPU_DERIVED','evaluation/paper_recovery_model_short_context_audit/partB_absolute.csv','附件301','额外恢复成本高估10.7%–34.9%',status='DENOMINATOR_MISMATCH',limit='该范围数值能复现，但分母不是单独的额外恢复成本')
    for side in ('a','b'):
        p=f'evaluation/paper_hybrid_gpu_gate_{side}/gate_{side}_result.json';r=a.obj(p)
        assert r['verdict']=='PASS' and r['cold_match']
        a.metric('F07_'+side,'VI-F','生产混合路径GPU门禁 '+side,len(r['checks']),'检查','各一独立生命周期','GPU_CORRECTNESS',p,'附件332',limit='小Exact/大Greedy分支的局部正确性，不是所有请求与架构全覆盖')
    # 纸面表格不可见，另提供同源参考值，避免猜测原表格单元格。
    references=[]
    for tag,policy in [('fs','FlowState-Greedy'),('mc','Marconi-Evict Adaptation'),('kv','KVFlow-inspired')]:
        runs=[a.obj(f'evaluation/p0_control_cycle_gpu/arms/m{i}_ck3_{tag}/run/summary.json') for i in range(1,6)]
        for s in runs:
            for r in s['pending_rows']:assert abs(r['ttft_ms']-(r['first_token_ns']-r['start_ns'])/1e6)<1e-12
        for j in range(4):
            pid=runs[0]['pending_rows'][j]['pending_id'];pr=[s['pending_rows'][j] for s in runs]
            assert all((r['E'],r['G'],r['T'])==(pr[0]['E'],pr[0]['G'],pr[0]['T']) for r in pr)
            row={'policy':policy,'pending':pid,'T':pr[0]['T'],'E':pr[0]['E'],'G':pr[0]['G'],
                 'prefill_tokens':pr[0]['prefill_token_count'],'TTFT_median_ms':st.median(r['ttft_ms'] for r in pr),
                 'CUDA_Prefill_median_ms':st.median(r['cuda_prefill_ms'] for r in pr)}
            references.append(row)
            a.metric('B04_'+tag+'_'+pid,'VI-B','逐Pending前沿 '+policy+' '+pid,[row[k] for k in ('T','E','G','prefill_tokens')],'Tokens','每臂原生实际观测，五轮一致','GPU_NATIVE',f'evaluation/p0_control_cycle_gpu/arms/m{{1..5}}_ck3_{tag}/run/summary.json','附件281–287',limit='表Y未导出，本行为同源参考，不猜测原表单元格')
    savecsv('gpu_main15_reference.csv',references)
    savecsv('evidence_master.csv',master)
    for p in ['evaluation/gate1_step1_allocation_equivalence/gate1_step1_allocation_equivalence_audit.md',
              'evaluation/paper_final_evidence_gates_20261031/gate_b_shared_branch_gpu.md',
              'evaluation/greedy_swap_structure_e3_step1/final_report.md',
              'evaluation/greedy_swap_structure_e3_step1/local_trap_diagnostics.json',
              'evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/protocol.json',
              'evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/source_manifest.json',
              'evaluation/evaluation_data_package/gpu_per_round.csv',
              'evaluation/evaluation_data_package/gpu_overhead.csv']:
        a.track(p)
    for path,v in a.SOURCES.items():
        v['sha256_after']=a.sha(Path(path));assert v['sha256_after']==v['sha256_before'], path
    (OUT/'source_manifest.json').write_text(json.dumps({'files':a.SOURCES,'说明':'源文件读取前后SHA一致；不修改源实验、论文或生产代码。'},ensure_ascii=False,indent=2))
    print('最终指标清单',len(master),'行；只读来源',len(a.SOURCES),'份；三个CUDA数值和短上下文分母需修复。')


if __name__=='__main__':main()
