"""只读核对冻结输入、成本模型和历史 A5 证据，不构造新的候选或预算。"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'agentic_workflow_dataset_research/phase2_submission_inputs'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
manifest=[json.loads(s) for s in (SOURCE/'runtime_candidate_manifest.jsonl').read_text().splitlines()]
checks=[];hashes={}
for record in sorted(manifest,key=lambda r:(r['case_id']!='lmcache_10',r['case_id'])):
    paths=[Path(p) for p in record['input_artifact_paths']]
    actual=[sha(p) for p in paths]
    assert actual==record['input_sha256'],record['case_id']
    arrays=[read(p)['input_ids'] for p in paths]
    a,b=arrays
    lcp=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    assert (len(a),len(b),lcp)==(record['p1_total_tokens'],record['p2_total_tokens'],record['prefix_tokens'])
    assert all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for ids in arrays for x in ids)
    checks.append(dict(case_id=record['case_id'],input_sha256=actual,p1_tokens=len(a),p2_tokens=len(b),exact_lcp=lcp,status='PASS',input_paths=list(map(str,paths))))
    hashes.update(zip(map(str,paths),actual))
assert len(checks)==16
baseline=ROOT/'evaluation/p1_strong_baseline_validation'
for name,expected in read(baseline/'implementation_lock.json').items():
    assert sha(baseline/name)==expected,name
    hashes[str(baseline/name)]=expected
sources=[ROOT/p for p in ['flowstate/optimizer.py','flowstate/recovery_model.py','flowstate/state_catalog.py','evaluation/FORMAL_RECOVERY_MODEL.md','evaluation/rq3_sanity_structure_audit.py','evaluation/submission_p0_lmcache_recovery/input_identity.json','evaluation/submission_p0_lmcache_recovery/runtime_evidence.json','evaluation/submission_p0_lmcache_recovery/lmcache_runtime_report.md']]
sources.extend([SOURCE/'runtime_candidate_manifest.jsonl',SOURCE/'selection_lock.json'])
for p in sources:hashes[str(p)]=sha(p)
a5=read(ROOT/'evaluation/submission_p0_lmcache_recovery/input_identity.json')
for v in a5['input_files'].values():assert sha(Path(v['path']))==v['sha256']
result=dict(status='INPUT_IDENTITY_READY',cases=checks,baseline_lock='PASS',a5_input_identity='PASS',a5_recorded_status=read(ROOT/'evaluation/submission_p0_lmcache_recovery/runtime_evidence.json')['status'],original_a6_candidate_rule='MISSING',original_a6_target_rule='MISSING',original_a6_budget_rule='MISSING',original_a6_remaining_cases_rule='MISSING',gpu_executed=False)
(OUT/'input_identity_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(OUT/'source_integrity_before.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
print('16 组输入、冻结 Baseline 摘要和 A5 输入身份核验通过；尚未执行分配比较。')
