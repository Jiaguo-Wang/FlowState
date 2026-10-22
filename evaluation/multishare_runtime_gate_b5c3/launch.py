"""固定一次生命周期；失败不重试，只清理本轮专用容器。"""
import json,subprocess,time,socket
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write
NAME='flowstate-b5c3-multishare'
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def main():
 started=False
 source=read(OUT/'source_identity.json');assert source['status']=='PASS'
 for p,h in source['sha256'].items():assert sha(p)==h,p
 image=json.loads(run(['docker','image','inspect',source['image_id']]))
 assert image[0]['Id']==source['image_id']
 write(OUT/'image_identity.json',image)
 apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv'])
 gpu=run(['nvidia-smi','-i','0','--query-gpu=uuid,name,memory.used,memory.total','--format=csv'])
 assert len(apps.strip().splitlines())==1,'GPU0有其他任务，不启动'
 assert 'H100 PCIe' in gpu and int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())<1000
 assert not run(['docker','ps','-a','--filter','name='+NAME,'--format','{{.Names}}']).strip(),'专用容器已存在，禁止覆盖'
 with socket.socket() as s:s.bind(('127.0.0.1',49997))
 write(OUT/'gpu_before.json',{'gpu':gpu,'processes':apps})
 write(OUT/'implementation_lock.json',{'sha256':{str(p):sha(p) for p in OUT.glob('*.py')},'source_identity_sha256':sha(OUT/'source_identity.json'),'prepared_inputs_sha256':sha(OUT/'prepared_inputs.json')})
 write(OUT/'execution_started.json',{'unix_ns':time.time_ns(),'single_lifecycle':True,'history_requests':8,'pending_submissions':0})
 (OUT/'preflight_report.md').write_text('# CPU与资源预检查\n\nPASS：B5-C2全部摘要、固定12份输入、八请求顺序及既有R2观察源码一致。原模型权重/配置按既有来源锁检查。GPU0为H100 PCIe、无计算进程，显存低于1000 MiB；镜像ID与B4/C2一致。\n\n真实SGLang源码、版本及模板在runner创建Engine之前逐项核验；未通过则不创建Engine。历史参数沿用temperature=0、max_new_tokens=1、ignore_eos=True。\n\n使用R2-C0已验证scope/observe和B1-C原始match/COW只读记录；FA直接只读遍历，前后完整物理视图与组件内容复核。没有调用R2控制计划、bind、reconcile或驱逐入口。新适配只负责全树枚举和证据序列化。\n')
 cmd=['docker','run','--rm','--pull','never','--name',NAME,'--gpus','device=0','--network','host','--shm-size','32g']
 for src,dst,ro in ((ROOT,ROOT,True),(Path('/home/wjg/models/qwen3.5-9b'),Path('/home/wjg/models/qwen3.5-9b'),True),(OUT,Path('/b5out'),False)):
  cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
 env={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','FLOWSTATE_STEP5D_PORT':'49997','TRITON_CACHE_DIR':'/b5out/triton_cache','PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
 for k,v in env.items():cmd+=['--env',k+'='+v]
 cmd+=['--workdir',str(ROOT),'--entrypoint','python3',source['image_id'],'-B','-m','evaluation.multishare_runtime_gate_b5c3.runner']
 write(OUT/'command.json',cmd)
 try:
  with (OUT/'runtime.log').open('x') as log:
   started=True;r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
  write(OUT/'process_result.json',{'returncode':r.returncode})
 except BaseException as exc:
  write(OUT/'launch_failure.json',{'error':repr(exc),'started':started});raise
 finally:
  if started:
   rm=subprocess.run(['docker','rm','-f',NAME],capture_output=True,text=True)
   observations=[]
   for _ in range(3):
    time.sleep(3)
    mem=run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip()
    apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv'])
    observations.append({'memory_mib':int(mem),'processes':apps,'unix_ns':time.time_ns()})
   own=run(['docker','ps','-a','--filter','name='+NAME,'--format','{{.Names}}'])
   passed=not own.strip() and all(o['memory_mib']<100 and len(o['processes'].strip().splitlines())==1 for o in observations)
   write(OUT/'gpu_cleanup.json',{'status':'PASS' if passed else 'FAIL','own_container':own,'remove_returncode':rm.returncode,'remove_stderr':rm.stderr,'observations':observations,'other_processes_terminated':False})
if __name__=='__main__':main()
