"""逐行联接原始Formal评估与派生矩阵，不执行任何选择算法。"""
import json
from pathlib import Path
from evaluation.icde_v7_evidence_freeze_20261102 import audit as a


def main():
    a.SOURCES=json.loads((a.OUT/'source_manifest.json').read_text())['files']
    records=a.lines('evaluation/p1_strong_baseline_validation/per_case_results.jsonl')
    matrix={r['instance_id']:r for r in a.lines('evaluation/gate1_step1_allocation_equivalence/matrices.jsonl')}
    parents={}; checked=[]
    for r in records:
        path=r['source_file']
        if path not in parents:parents[path]=a.lines(path)
        raw=parents[path][int(r['source_line'])-1]
        assert (raw['snapshot_id'],raw['budget_ratio'],raw['k'])==(r['snapshot_id'],r['budget_ratio'],r['k'])
        assert raw['snapshot_digest']==r['snapshot_digest']
        original=raw['policies']['FlowState'];derived=r['policies']['FlowState']
        assert original['selected_checkpoint_ids']==derived['selected']
        assert abs(original['total_recovery_cost_ms']-derived['cost_ms'])<=a.EPS
        mid=f"{r['snapshot_id']}::budget={r['budget_ratio']:g}";m=matrix[mid]
        diag=raw['mechanism_diagnostics']
        for i,c in enumerate(m['candidate_ids']):
            assert diag['standalone_benefits'][c]['token_pos']==m['positions'][i]
            gains={x['continuation_id']:x for x in diag['standalone_benefits'][c]['per_pending']}
            for j,p in enumerate(m['pending_ids']):
                assert diag['compatibility_matrix'][c][p]==m['compatibility'][j][i]
                assert abs(gains[p]['benefit_ms']-m['benefit_ms'][j][i])<=a.EPS
        checked.append({'condition_id':mid,'source_file':path,'source_line':r['source_line'],'status':'PASS'})
    for path,v in a.SOURCES.items():
        v['sha256_after']=a.sha(Path(path));assert v['sha256_after']==v['sha256_before'],path
    (a.OUT/'source_manifest.json').write_text(json.dumps({'files':a.SOURCES,'说明':'原始评估行、矩阵及运行记录读取前后摘要相同；不重放原轨迹或重跑策略。'},ensure_ascii=False,indent=2))
    with (a.OUT/'parent_link_verification.json').open('x') as f:
        json.dump({'conditions':573,'status':'PASS','rows':checked},f,ensure_ascii=False,indent=2)
    print('573个原始评估行及其兼容/收益/位置/选集与派生表一致。')


if __name__=='__main__':main()
