"""读取五个冻结严格快照；缺ASG或固定边界时记录拒绝，不伪造性能结果。"""
import csv
import hashlib
import io
import json
import unittest
from pathlib import Path
from dataclasses import replace
from .policy import select, priorities, IncompleteEvidence
from . import test_policy
from evaluation.greedy_swap_structure_e3_step1 import audit as frozen

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HASH={}

def read(path):
    data=path.read_bytes();HASH[str(path)]=hashlib.sha256(data).hexdigest()
    return json.loads(data)

def save(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def table(name, rows):
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})

def main():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_policy))
    (OUT/'cpu_tests.log').write_text(stream.getvalue())
    if not result.wasSuccessful():raise RuntimeError('CPU单测未通过')
    cases=frozen.load()
    rows=[];candidate_rows=[];pending_rows=[];availability=[];graphs=[]
    for name, base, prior in cases:
        root=ROOT/'evaluation'/('multishare_runtime_gate_b5c3' if name=='B5-C4' else 'independent_runtime_allocation_e2/'+name)
        snapshot=read(root/'strict_epoch_snapshot.json');prepared=read(root/'prepared_inputs.json')
        graph=snapshot.get('agent_step_graph',[]);declarations=snapshot.get('fixed_prefix_declarations',[])
        # 当前格式没有这些元数据；不把受控Ready或Session名称转换为Agent声明。
        reason=[]
        if not graph:reason.append('缺少决策时ASG调用ID、前驱与AND/OR声明')
        if not declarations:reason.append('缺少固定提示Token末端与动态后缀声明')
        for p in snapshot['pendings']:
            pending_rows.append({'snapshot':name,'pending_id':p['continuation_id'],'controlled_input_ready':p.get('controlled_input_ready'),'agent_native_ready':p.get('agent_native_ready'),'agent_invocation_id':'UNKNOWN','STE':'UNKNOWN','reason':'输入已准备不等于具有声明的Agent调用身份'})
        diagnostics=priorities(snapshot['candidates'],graph,declarations,snapshot['epoch_id'],snapshot['configuration_digest'])
        for r in diagnostics:
            candidate_rows.append({'snapshot':name,**r,'priority':'UNKNOWN' if r['priority'] is None else ('DYNAMIC' if r['priority']==float('inf') else r['priority']),'evidence_path':str(root/'strict_epoch_snapshot.json')})
        graphs.append({'snapshot':name,'epoch_id':snapshot['epoch_id'],'asg':graph,'fixed_prefix_declarations':declarations,'status':'UNKNOWN' if reason else 'AVAILABLE','source_fields':list(snapshot),'prepared_fields':list(prepared)})
        availability.append({'snapshot':name,'candidates':len(snapshot['candidates']),'pendings':len(snapshot['pendings']),'pairs':len(snapshot['edges']),'status':'UNADAPTABLE' if reason else 'AVAILABLE','reason':'；'.join(reason),'evidence_path':str(root/'strict_epoch_snapshot.json')})
        for k in (1,2,3,4):
            epoch=replace(base,k=k)
            try:
                selected,_=select(snapshot['candidates'],graph,declarations,snapshot['epoch_id'],snapshot['configuration_digest'],k)
                score=frozen.common.score(epoch,selected);status='COMPUTED'
            except IncompleteEvidence as exc:
                selected=None;score=None;status='UNKNOWN';failure=str(exc)
            rows.append({'snapshot':name,'K':k,'policy':'KVFlow-inspired','status':status,'selected':selected,'J_ms':None if score is None else score['cost_ms'],'per_pending':None if score is None else score['per_pending'],'set_vs_kvflow':'UNKNOWN' if selected is None else 'SAME','WTL_vs_kvflow':'UNKNOWN' if score is None else 'TIE','reason':'；'.join(reason)+('；'+failure if score is None else ''),'source':str(root/'strict_epoch_snapshot.json')})
            for policy, aliases in [('FlowState-Greedy',('FlowState-Greedy','FlowState')),('WFC',('WFC',)),('LRU',('LRU',)),('Marconi Adaptation',('Marconi Adaptation','Marconi'))]:
                old=next((r for r in prior if r['K']==k and r['policy'] in aliases),None)
                if old:
                    val=frozen.common.score(epoch,old['selected']);assert abs(val['cost_ms']-old['J_ms'])<1e-9
                    same='UNKNOWN' if selected is None else ('SAME' if set(old['selected'])==set(selected) else 'DIFFERENT')
                    delta=None if score is None else val['cost_ms']-score['cost_ms']
                    wtl='UNKNOWN' if delta is None else ('WIN' if delta< -1e-9 else 'LOSS' if delta>1e-9 else 'TIE')
                else:val=None;same='UNKNOWN';wtl='UNKNOWN'
                rows.append({'snapshot':name,'K':k,'policy':policy,'status':'FROZEN_RECHECKED' if old else 'UNAVAILABLE','selected':None if old is None else old['selected'],'J_ms':None if val is None else val['cost_ms'],'per_pending':None if val is None else val['per_pending'],'set_vs_kvflow':same,'WTL_vs_kvflow':wtl,'reason':'KVFlow输入缺证据，不能判断同集或胜负' if old else '该严格候选域无冻结结果；不能跨旧实验借用访问元数据','source':str(root)})
    HASH.update(frozen.SOURCES)
    for p,h in HASH.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    table('candidate_priorities.csv',candidate_rows);table('pending_ste.csv',pending_rows);table('cpu_comparison.csv',rows);table('snapshot_availability.csv',availability)
    save('asg_extraction.json',graphs)
    save('source_integrity.json',{'unchanged':True,'sha256':HASH})
    save('test_results.json',{'unit_tests':result.testsRun,'passed':result.wasSuccessful(),'real_snapshots':len(cases),'real_conditions':20,'computed_kvflow_conditions':sum(r['policy']=='KVFlow-inspired' and r['status']=='COMPUTED' for r in rows),'unknown_priority_candidates':sum(r['priority']=='UNKNOWN' for r in candidate_rows),'baseline_objectives_rechecked':sum(r['status']=='FROZEN_RECHECKED' for r in rows)})
    print(json.dumps(availability,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
