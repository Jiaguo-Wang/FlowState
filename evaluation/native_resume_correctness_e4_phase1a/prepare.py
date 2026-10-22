"""核对E2/E3及既有恢复实现，冻结w1四Pending独立生命周期协议。"""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
E2=ROOT/'evaluation/independent_runtime_allocation_e2';E3=ROOT/'evaluation/greedy_swap_structure_e3_step1';E1=ROOT/'evaluation/lightweight_runtime_e1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 assert not (OUT/'prepared_inputs.json').exists(),'不覆盖本轮输入'
 source=read(E2/'source_identity.json')
 for base in (E2,E3,E1,ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4b'):
  manifest=base/'evidence_manifest.json';source['sha256'][str(manifest)]=sha(manifest)
  for n,h in read(manifest)['files'].items():
   p=Path(n) if Path(n).is_absolute() else base/n
   assert sha(p)==h,str(p);source['sha256'][str(p)]=h
 for p,h in source['sha256'].items():assert sha(p)==h,p
 inputs=read(E2/'w1/prepared_inputs.json');snap=read(E2/'w1/strict_epoch_snapshot.json');rows=read(E2/'w1/allocation_results.json')['rows'];plans={}
 for solver in ('Greedy','Exact'):
  row=next(r for r in rows if r['K']==2 and r['policy']=='FlowState-'+solver)
  assert len(row['selected'])==2
  plans[solver]={'selected_ids':row['selected'],'J_ms':row['J_ms'],'per_pending':row['per_pending']}
 swap=next(r for r in read(E3/'results.json')['rows'] if r['case']=='w1' and r['K']==2 and r['policy']=='Greedy+1-Swap')
 assert set(swap['selected'])==set(plans['Exact']['selected_ids']) and abs(swap['J_ms']-plans['Exact']['J_ms'])<1e-9
 write(OUT/'prepared_inputs.json',inputs);write(OUT/'frozen_snapshot.json',snap);write(OUT/'frozen_plans.json',plans)
 write(OUT/'source_identity.json',source);write(OUT/'cpu_gate.json',read(E1/'cpu_gate.json'))
 write(OUT/'experiment_protocol.json',{'workload':'w1','K':2,'pending_order':[p['continuation_id'] for p in inputs['pendings']],'lifecycle_order':'每个Pending先独立Cold，再独立Greedy、独立Exact；最多12个生命周期，各严格臂重放同一八条历史','generation':{'temperature':0,'max_new_tokens':8,'ignore_eos':True},'failure':'任何安全、恢复或输出失败立即停止后续生命周期；不重试','control':'复用E1重型路径和B5-D4a安全协调；同Scheduler调用Apply-time重验；所有臂均执行完整内容后验','identity':'按完整Token前缀双射映射冻结逻辑候选，不复用旧Node/Slot；当前矩阵、T/H及计划必须等价','provenance':'PROVENANCE_UNVERIFIED','timing':'不以性能为门禁，不作时延比较','swap_arm':'w1/K2与Exact同集，不另执行'})
 print('来源及四Pending冻结通过',len(source['sha256']))
if __name__=='__main__':main()
