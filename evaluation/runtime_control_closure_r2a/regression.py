"""记录生产严格入口与冻结参考的一致性，不执行物理协调。"""
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.strict_epoch import StrictAllocationEpoch
OUT=Path(__file__).resolve().parent
RAW=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01'
FROZEN=ROOT/'evaluation/strict_compatibility_b4c'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks=[]
    for p in [RAW/'manifest.json',FROZEN/'evidence_manifest.json']:
        for name,h in json.loads(p.read_text())['files'].items():
            target=Path(name) if Path(name).is_absolute() else p.parent/name
            assert sha(target)==h
            checks.append({'path':str(target),'sha256':h})
    refs=json.loads((FROZEN/'results.json').read_text())
    source=json.loads((FROZEN/'input_integrity.json').read_text())['sha256']
    rows=[];plans=[]
    for snapshot in refs['snapshots']:
        path=RAW/snapshot['group']/'snapshot.json'
        for k in range(1,5):
            epoch=StrictAllocationEpoch.from_snapshot_bytes(path.read_bytes(),expected_sha256=source[str(path)],K=k)
            plan=GlobalOptimizer(RecoveryCostModel()).plan_strict(epoch)
            ref=next(r for r in snapshot['rows'] if r['K']==k and r['policy']=='FlowState')
            assert list(plan.selected_ids)==ref['selected']
            assert abs(plan.recovery_cost_ms-ref['J_ms'])<1e-9
            plans.append(asdict(plan))
            rows.append(dict(group=snapshot['group'],K=k,status='PASS',selected_order=json.dumps(plan.selected_ids),J_ms=plan.recovery_cost_ms,frozen_J_ms=ref['J_ms'],selected_count=plan.selected_count,logical_eviction_ids=json.dumps(plan.logical_eviction_ids)))
    with (OUT/'regression.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'plans.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
    changed={str(ROOT/p):sha(ROOT/p) for p in ['flowstate/optimizer.py','flowstate/strict_epoch.py','tests/test_strict_epoch.py']}
    (OUT/'integrity.json').write_text(json.dumps(dict(frozen_artifact_checks=checks,current_implementation=changed,说明='历史产物摘要不变；旧执行锁内optimizer摘要属于旧版本，未改写旧锁，不宣称当前源码通过旧运行锁。'),ensure_ascii=False,indent=2)+'\n')
    print('严格生产回归16/16通过，历史产物摘要引用核对',len(checks))
if __name__=='__main__':main()
