"""冻结顺序执行最多五对；首对和独立Cold全部通过后才继续，任意失败即停。"""
import json,subprocess,time,socket
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write
from .review import check_pair

def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def main():
 assert not (OUT/'execution_started.json').exists(),'禁止重跑已启动实验'
 source=read(OUT/'source_identity.json');assert read(OUT/'cpu_gate.json')['status']=='PASS'
 for p,h in source['sha256'].items():assert sha(p)==h,p
 image=json.loads(run(['docker','image','inspect',source['image_id']]))[0];assert image['Id']==source['image_id']
 protocol=read(OUT/'protocol_lock.json')
 lock={str(p):sha(p) for p in list(OUT.glob('*.py'))+[OUT/'experiment_protocol.md',OUT/'protocol_lock.json',OUT/'cpu_gate.json',OUT/'source_identity.json',OUT/'prepared_inputs.json']}
 write(OUT/'implementation_lock.json',{'files':lock,'image':image['Id']})
 write(OUT/'execution_started.json',{'unix_ns':time.time_ns(),'protocol':protocol})
 cleanup={};executed=[]
 def one(name,solver):
  for p,h in lock.items():assert sha(p)==h,p
  apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv']);assert len(apps.strip().splitlines())==1,'GPU0有其他任务'
  gpu=run(['nvidia-smi','-i','0','--query-gpu=uuid,name,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.limit','--format=csv']);assert 'H100 PCIe' in gpu
  assert int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())<1000
  container='flowstate-b5d4-'+name.lower();assert not run(['docker','ps','-a','--filter','name='+container,'--format','{{.Names}}']).strip()
  with socket.socket() as sock:sock.bind(('127.0.0.1',49995))
  write(OUT/f'{name}_gpu_before.json',{'gpu':gpu,'apps':apps,'unix_ns':time.time_ns()})
  cmd=['docker','run','--rm','--pull','never','--name',container,'--gpus','device=0','--network','host','--shm-size','32g']
  for src,dst,ro in [(ROOT,ROOT,True),(Path('/home/wjg/models/qwen3.5-9b'),Path('/home/wjg/models/qwen3.5-9b'),True),(OUT,Path('/b5out'),False)]:cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
  env={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','FLOWSTATE_STEP5D_PORT':'49995','TRITON_CACHE_DIR':'/b5out/triton_cache_'+name,'B5D4_ARM':'cold' if solver=='Cold' else 'strict','B5D4_RUN':name,'B5D4_SOLVER':solver,'PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
  for k,v in env.items():cmd+=['--env',k+'='+v]
  cmd+=['--workdir',str(ROOT),'--entrypoint','python3',image['Id'],'-B','-m','evaluation.exact_greedy_paired_gpu_gate_b5d4.runner'];write(OUT/f'{name}_command.json',cmd)
  executed.append(name);write(OUT/'execution_ledger.json',{'started':executed});r=None
  print('启动',name,flush=True)
  try:
   with (OUT/f'{name}.runtime.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
   write(OUT/f'{name}_process.json',{'returncode':r.returncode})
  finally:
   removed=subprocess.run(['docker','rm','-f',container],capture_output=True,text=True);obs=[]
   for _ in range(3):
    time.sleep(3);mem=int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip());apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv']);obs.append({'memory_mib':mem,'processes':apps})
   own=run(['docker','ps','-a','--filter','name='+container,'--format','{{.Names}}']).strip()
   cleanup[name]={'status':'PASS' if not own and all(x['memory_mib']<100 and len(x['processes'].strip().splitlines())==1 for x in obs) else 'FAIL','observations':obs,'own_container':own,'remove_stderr':removed.stderr}
   write(OUT/f'{name}_cleanup.json',cleanup[name]);(OUT/'gpu_cleanup.json').write_text(json.dumps(cleanup,ensure_ascii=False,indent=2))
  assert cleanup[name]['status']=='PASS' and r and r.returncode==0,'执行或清理失败'
  assert read(OUT/name/'summary.json')['status']=='PASS','缺少完整正确性摘要'
  print('本臂完成',name,flush=True)
 try:
  for pair,order in enumerate(protocol['order'],1):
   for solver in order:one(f'p{pair:02d}_{solver}',solver)
   if pair==1:one('cold','Cold')
   checked=check_pair(OUT,pair);write(OUT/f'pair_{pair:02d}_validation.json',checked);print('配对门禁PASS',pair,flush=True)
 except BaseException as exc:
  write(OUT/'launch_failure.json',{'error':repr(exc),'started':executed});raise
 finally:
  (OUT/'gpu_cleanup.json').write_text(json.dumps(cleanup,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
