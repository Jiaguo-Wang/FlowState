"""只读校验冻结输入及既有观察实现，冻结本轮八历史与四Pending。"""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
C2=ROOT/'evaluation/multishare_cpu_input_gate_b5c2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 hashes={}
 for name,h in read(C2/'artifact_manifest.json')['files'].items():
  p=C2/name;assert sha(p)==h,str(p);hashes[str(p)]=h
 assert read(C2/'verification.json')['gate']=='READY'
 selection=read(C2/'source_selection_lock.json');assert sha(selection['source_path'])==selection['source_sha256']
 hashes[selection['source_path']]=selection['source_sha256']
 for path,h in read(ROOT/'evaluation/runtime_control_closure_r2c1b2/implementation_lock.json')['sha256'].items():
  assert sha(path)==h,path;hashes[path]=h
 old=read(ROOT/'evaluation/runtime_control_closure_r2c1b2/source_identity.json')
 for row in old['files']:
  p=row['path'];assert sha(p)==row['current_sha256'],p;hashes[p]=row['current_sha256']
 for rel in ('evaluation/strict_compatibility_b4b0/contracts.py','evaluation/barrier_fa_frontier_control.py','tests/runtime/wp3b_end_to_end_transport.py','motivation/artifacts/wp3b_gate_20260820/targeted_probe.py'):
  p=ROOT/rel;hashes[str(p)]=sha(p)
 config=read(C2/'input_configuration_lock.json')['b4_identity']
 req=read(C2/'requests.json');history=[];pending=[]
 for turn in (1,2,3):
  for label in 'ABCD':
   row=next(r for r in req if r['label']==label and r['turn']==turn);ids=read(C2/row['token_file'])
   assert len(ids)==row['token_count'] and sha(C2/row['token_file'])==row['file_sha256']
   from evaluation.strict_compatibility_b1c.protocol import token_digest
   src=next(r for r in selection['records'] if r['label']==label)
   obj={'request_id':row['request_id'],'continuation_id':row['request_id'],'label':label,'turn':turn,'workflow_id':src['session_id'],'token_ids':ids,'token_prefix_sha256':token_digest(ids),'source':src}
   (history if turn<3 else pending).append(obj)
 assert [r['request_id'] for r in history]==[f'B5C2_{label}_TURN_{turn:03d}' for turn in (1,2) for label in 'ABCD']
 sampling={'temperature':0,'max_new_tokens':1,'ignore_eos':True}
 assert config['engine_configuration']['tp_size']==1 and config['engine_configuration']['max_mamba_cache_size']==28
 for p in OUT.glob('*.py'):compile(p.read_text(),str(p),'exec')
 write(OUT/'prepared_inputs.json',{'configuration':config,'history':history,'pendings':pending,'sampling':sampling,'workload_type':'Trace-derived Controlled Composition','pending_dispatch_allowed':False})
 write(OUT/'source_identity.json',{'status':'PASS','sha256':hashes,'runtime_sources':old['runtime_sources'],'image_id':read(C2/'input_configuration_lock.json')['image_id'],'说明':'全部C2内容及R2既有观察源码/冻结模型内容复核；镜像内部源码在Engine创建前再次核验'})
 print('PASS：输入、历史顺序、配置和既有实现摘要一致。')
if __name__=='__main__':main()
