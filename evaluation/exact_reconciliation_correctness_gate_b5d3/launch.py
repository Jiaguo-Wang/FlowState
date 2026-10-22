"""每臂仅一次；Strict失败不执行Cold，不操作其他容器或进程。"""
import json,subprocess,time,socket
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write

def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def main():
 assert not (OUT/'execution_started.json').exists(),'禁止重跑已启动生命周期'
 source=read(OUT/'source_identity.json');assert read(OUT/'cpu_gate.json')['status']=='PASS'
 for p,h in source['sha256'].items():assert sha(p)==h,p
 image=json.loads(run(['docker','image','inspect',source['image_id']]))[0];assert image['Id']==source['image_id']
 lock={str(p):sha(p) for p in OUT.glob('*.py')};write(OUT/'implementation_lock.json',{'files':lock,'image':image['Id'],'source_identity_sha256':sha(OUT/'source_identity.json')})
 write(OUT/'execution_started.json',{'unix_ns':time.time_ns(),'arms':['strict','cold'],'target':'由当前Exact选集及来源按Pending标签和候选ID固定排序选第一条合法跨Workflow边'})
 cleanup={}
 try:
  for arm in ['strict','cold']:
   for p,h in lock.items():assert sha(p)==h,p
   apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv']);assert len(apps.strip().splitlines())==1,'GPU0存在其他任务'
   gpu=run(['nvidia-smi','-i','0','--query-gpu=uuid,name,memory.used,memory.total','--format=csv']);assert 'H100 PCIe' in gpu
   assert int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())<1000
   name='flowstate-b5d3-'+arm
   assert not run(['docker','ps','-a','--filter','name='+name,'--format','{{.Names}}']).strip(),'专用容器已存在'
   with socket.socket() as sock:sock.bind(('127.0.0.1',49994))
   write(OUT/f'{arm}_gpu_before.json',{'gpu':gpu,'processes':apps})
   cmd=['docker','run','--rm','--pull','never','--name',name,'--gpus','device=0','--network','host','--shm-size','32g']
   for src,dst,ro in [(ROOT,ROOT,True),(Path('/home/wjg/models/qwen3.5-9b'),Path('/home/wjg/models/qwen3.5-9b'),True),(OUT,Path('/b5out'),False)]:cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
   env={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','FLOWSTATE_STEP5D_PORT':'49994','TRITON_CACHE_DIR':'/b5out/triton_cache_'+arm,'B5D3_ARM':arm,'PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
   for k,v in env.items():cmd+=['--env',k+'='+v]
   cmd+=['--workdir',str(ROOT),'--entrypoint','python3',image['Id'],'-B','-m','evaluation.exact_reconciliation_correctness_gate_b5d3.runner'];write(OUT/f'{arm}_command.json',cmd)
   try:
    with (OUT/f'{arm}.runtime.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    write(OUT/f'{arm}_process.json',{'returncode':r.returncode})
   finally:
    removed=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True);obs=[]
    for _ in range(3):
     time.sleep(3);mem=int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip());apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv']);obs.append({'memory_mib':mem,'processes':apps})
    own=run(['docker','ps','-a','--filter','name='+name,'--format','{{.Names}}']).strip()
    cleanup[arm]={'status':'PASS' if not own and all(x['memory_mib']<100 and len(x['processes'].strip().splitlines())==1 for x in obs) else 'FAIL','observations':obs,'own_container':own,'remove_stderr':removed.stderr};write(OUT/f'{arm}_cleanup.json',cleanup[arm])
   assert cleanup[arm]['status']=='PASS','清理失败';assert r.returncode==0,'本臂失败，不继续'
   assert read(OUT/arm/'summary.json')['status']=='PASS','缺少实际正确性摘要'
 except BaseException as exc:
  write(OUT/'launch_failure.json',{'error':repr(exc)});raise
 finally:
  write(OUT/'gpu_cleanup.json',cleanup)
  (OUT/'runtime.log').write_text('\n'.join('[生命周期 '+arm+']\n'+(OUT/f'{arm}.runtime.log').read_text() for arm in ['strict','cold'] if (OUT/f'{arm}.runtime.log').exists()))
if __name__=='__main__':main()
