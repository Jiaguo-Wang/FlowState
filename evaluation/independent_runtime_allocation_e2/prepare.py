"""核验已有输入与版本，按原顺序装配四个独立回放包；不重新分词。"""
import hashlib,json
from pathlib import Path
from evaluation.strict_compatibility_b1c.protocol import token_digest
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SRC=ROOT/'evaluation/independent_multishare_workloads';C3=ROOT/'evaluation/multishare_runtime_gate_b5c3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 assert not (OUT/'source_identity.json').exists(),'不覆盖准备记录'
 manifest=read(SRC/'artifact_manifest.json');assert read(SRC/'verification.json')['gate']=='READY'
 hashes={str(SRC/n):h for n,h in manifest['files'].items()};hashes[str(SRC/'artifact_manifest.json')]=sha(SRC/'artifact_manifest.json')
 # 旧清单先于最终完成打印生成；只接受逐字节可证明的固定后缀。
 log=SRC/'cpu_run.log';suffix='READY：独立核验通过，数据交付已生成。\n'.encode();data=log.read_bytes()
 if sha(log)!=hashes[str(log)]:
  assert data.endswith(suffix) and hashlib.sha256(data[:-len(suffix)]).hexdigest()==hashes[str(log)],'日志差异无法解释'
  hashes[str(log)]=sha(log)
 source=read(C3/'source_identity.json');hashes.update(source['sha256'])
 for n in ('runner.py','bridge.py'):hashes[str(C3/n)]=sha(C3/n)
 old=read(C3/'implementation_lock.json')
 for name,h in old['sha256'].items():assert sha(name)==h,name
 lock=read(SRC/'input_configuration_lock.json');hashes.update(lock['files'])
 selected=read(SRC/'source_selection_lock.json');hashes[selected['source_path']]=selected['source_sha256']
 b4=read(ROOT/'evaluation/strict_compatibility_b4c/input_integrity.json')['sha256']
 for n in ['common.py','pcb.py','ldr.py','wfc.py']:
  p=ROOT/'evaluation/p1_strong_baseline_validation'/n;assert sha(p)==b4[str(p)];hashes[str(p)]=b4[str(p)]
 p=ROOT/'flowstate/recovery_model.py';assert sha(p)==b4[str(p)];hashes[str(p)]=b4[str(p)]
 for name in ['flowstate/strict_epoch.py','flowstate/strict_control.py','flowstate/optimizer.py','evaluation/exact_planner_contract_gate_b5d1/planner.py','evaluation/multishare_allocation_gate_b5c4/evaluate.py','evaluation/multishare_allocation_gate_b5c4/results.json']:
  p=ROOT/name;hashes[str(p)]=sha(p)
 for p,h in hashes.items():assert sha(p)==h,p
 assert lock['image_id']==source['image_id']
 config=read(C3/'prepared_inputs.json')['configuration'];assert config==lock['b4_identity']
 records=read(SRC/'requests.json');report=[]
 for group in read(OUT/'protocol.json')['workload_order']:
  out=OUT/group;out.mkdir(exist_ok=False);history=[];pending=[]
  def row(label,turn):
   r=next(r for r in records if (r['group'],r['label'],r['turn'])==(group,label,turn));tokens=read(SRC/r['token_file'])
   assert sha(SRC/r['token_file'])==r['file_sha256'] and len(tokens)==r['token_count']
   origin=next(x for x in selected['records'] if (x['group'],x['label'])==(group,label));assert origin['failure'] is None
   return {'request_id':r['request_id'],'continuation_id':r['request_id'],'workflow_id':origin['session_id'],'label':label,'turn':turn,'token_ids':tokens,'token_prefix_sha256':token_digest(tokens),'source_token_file':str(SRC/r['token_file']),'source_file_sha256':r['file_sha256'],'source':origin}
  for t in (1,2):
   for l in 'ABCD':history.append(row(l,t))
  for l in 'ABCD':pending.append(row(l,3))
  inputs={'configuration':config,'history':history,'pendings':pending,'sampling':{'temperature':0,'max_new_tokens':1,'ignore_eos':True},'workload_type':'Trace-derived Controlled Composition','pending_dispatch_allowed':False}
  write(out/'prepared_inputs.json',inputs);report.append({'group':group,'history':len(history),'pending':len(pending),'sha256':sha(out/'prepared_inputs.json')})
 source['sha256']=hashes;source['status']='PASS'
 write(OUT/'source_identity.json',source)
 for group in read(OUT/'protocol.json')['workload_order']:write(OUT/group/'source_identity.json',source)
 write(OUT/'input_checks.json',{'status':'PASS','groups':report,'source_files':len(hashes),'token_regeneration':False})
 print('四组冻结输入核验通过',len(hashes))
if __name__=='__main__':main()
