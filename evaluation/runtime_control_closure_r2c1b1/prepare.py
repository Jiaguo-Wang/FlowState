"""只读核验既有证据、目标边及严格计划，建立本轮来源锁。"""
from pathlib import Path
from evaluation.runtime_control_closure_r2c0.prepare import ROOT,read,sha,write
from .bridge import load,TARGET,SOURCE
OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'evaluation/runtime_control_closure_r2c1a'
def main():
    for directory in (PRIOR,ROOT/'evaluation/runtime_control_closure_r2c0'):
        for rel,h in read(directory/'evidence_manifest.json')['files'].items():assert sha(directory/rel)==h,rel
    identity=read(PRIOR/'source_identity.json')
    for row in identity['files']:assert sha(row['path'])==row['current_sha256'],row['path']
    for p,h in read(PRIOR/'implementation_lock.json')['sha256'].items():assert sha(p)==h,p
    raw,epoch,plan=load();edge=next(e for e in raw['edges'] if e['checkpoint_id']==SOURCE and e['continuation_id']==TARGET)
    assert edge['status']=='EPOCH_ADMISSIBLE'
    pending=next(p for p in plan.pending if p.pending_id==TARGET);assert pending.E==2432
    admiss={e['checkpoint_id'] for e in raw['edges'] if e['continuation_id']==TARGET and e['status']=='EPOCH_ADMISSIBLE'}
    assert set(plan.selected_ids)&admiss=={SOURCE}
    identity['scope']='R2-C1b1：一个目标Pending的Strict恢复及独立Cold对照；不改冻结输入'
    write(OUT/'source_identity.json',identity)
    write(OUT/'target_contract.json',{'edge':edge,'T':pending.T,'E':pending.E,'G':pending.G,'selected_compatible_ids':sorted(set(plan.selected_ids)&admiss),'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True},'protocol_source':'evaluation/strict_compatibility_b1c/runner.py:279-301','logits_protocol':'既有协议未要求Logits，当前不新增数值容差'})
    print('PASS：目标边及来源身份通过')
if __name__=='__main__':main()
