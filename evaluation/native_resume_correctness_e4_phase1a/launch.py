"""每个Pending三次独立生命周期，任何安全或正确性失败即停且保留证据。"""
import subprocess,socket,time,json,traceback
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write
from .review import check,ref_signature

def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def main():
 assert not (OUT/'execution_started.json').exists(),'禁止覆盖或重试'
 source=read(OUT/'source_identity.json');assert read(OUT/'cpu_regression_results.json')['status']=='PASS'
 assert '\nOK\n' in (OUT/'production_contract_unittest.log').read_text(),'生产CPU合同未通过'
 for p,h in source['sha256'].items():assert sha(p)==h,p
 image=json.loads(run(['docker','image','inspect',source['image_id']]))[0];assert image['Id']==source['image_id']
 lock={str(p):sha(p) for p in list(OUT.glob('*.py'))+list(OUT.glob('*.json'))}
 write(OUT/'execution_lock.json',{'sha256':lock,'image':image['Id'],'solver_scope':'冻结w1/K2；每个Pending独立严格臂与Cold'})
 write(OUT/'execution_started.json',{'unix_ns':time.time_ns()});ledger=[];cleanup={};valid=[];initial=None
 def one(name,solver,pid):
  for p,h in lock.items():assert sha(p)==h,p
  apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv']);assert len(apps.strip().splitlines())==1,'GPU存在其他任务'
  gpu=run(['nvidia-smi','-i','0','--query-gpu=uuid,name,memory.used,memory.total','--format=csv']);assert 'H100 PCIe' in gpu
  assert int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())<1000
  container='flowstate-e4p1a-'+name.lower();assert not run(['docker','ps','-a','--filter','name='+container,'--format','{{.Names}}']).strip()
  with socket.socket() as sock:sock.bind(('127.0.0.1',49998))
  write(OUT/f'{name}_gpu_before.json',{'gpu':gpu,'apps':apps,'unix_ns':time.time_ns()})
  cmd=['docker','run','--rm','--pull','never','--name',container,'--gpus','device=0','--network','host','--shm-size','32g']
  for src,dst,ro in [(ROOT,ROOT,True),(Path('/home/wjg/models/qwen3.5-9b'),Path('/home/wjg/models/qwen3.5-9b'),True),(OUT,Path('/b5out'),False)]:cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
  env={'E1_MODE':'heavy','E1_SMOKE':'1','E4_PENDING':pid,'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','FLOWSTATE_STEP5D_PORT':'49998','TRITON_CACHE_DIR':'/b5out/triton_cache_'+name,'B5D4_ARM':'cold' if solver=='Cold' else 'strict','B5D4_RUN':name,'B5D4_SOLVER':solver,'PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
  for k,v in env.items():cmd+=['--env',k+'='+v]
  cmd+=['--workdir',str(ROOT),'--entrypoint','python3',image['Id'],'-B','-m','evaluation.native_resume_correctness_e4_phase1a.runner'];write(OUT/f'{name}_command.json',cmd)
  ledger.append({'run':name,'solver':solver,'pending':pid,'started_ns':time.time_ns()});(OUT/'execution_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2));result=None
  print('开始',name,flush=True)
  try:
   with (OUT/f'{name}.runtime.log').open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
   write(OUT/f'{name}_process.json',{'returncode':result.returncode})
  finally:
   removed=subprocess.run(['docker','rm','-f',container],capture_output=True,text=True);obs=[]
   for _ in range(3):
    time.sleep(3);mem=int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip());apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv']);obs.append({'memory_mib':mem,'processes':apps})
   own=run(['docker','ps','-a','--filter','name='+container,'--format','{{.Names}}']).strip();cleanup[name]={'status':'PASS' if not own and all(x['memory_mib']<100 and len(x['processes'].strip().splitlines())==1 for x in obs) else 'FAIL','observations':obs,'own_container':own,'remove_stderr':removed.stderr}
   write(OUT/f'{name}_cleanup.json',cleanup[name]);(OUT/'gpu_cleanup.json').write_text(json.dumps(cleanup,ensure_ascii=False,indent=2))
  assert cleanup[name]['status']=='PASS' and result and result.returncode==0,'运行或清理失败'
  assert read(OUT/name/'summary.json')['status']=='PASS','执行摘要不完整'
  print('完成',name,flush=True)
 try:
  for p in read(OUT/'prepared_inputs.json')['pendings']:
   label=p['label'];pid=p['continuation_id'];cold=label+'_Cold';one(cold,'Cold',pid)
   for solver in ('Greedy','Exact'):
    name=label+'_'+solver;one(name,solver,pid)
    current=ref_signature(read(OUT/name/'reference_audit.json'))
    if initial is None:initial=current
    assert initial==current,'独立臂初始状态不等价'
    row=check(OUT,name,cold,pid);write(OUT/f'{name}_validation.json',row);valid.append(row)
    (OUT/'validated_resumes.json').write_text(json.dumps(valid,ensure_ascii=False,indent=2));print('恢复认证通过',name,flush=True)
 except BaseException as exc:
  write(OUT/'launch_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'attempted':ledger,'validated':len(valid)});raise
 finally:
  (OUT/'gpu_cleanup.json').write_text(json.dumps(cleanup,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
