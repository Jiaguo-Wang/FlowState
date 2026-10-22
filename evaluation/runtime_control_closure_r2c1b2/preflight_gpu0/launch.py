"""两个独立生命周期顺序执行；失败不重试，只清理本轮容器。"""
import json
import subprocess
import time
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write
IMAGE='lmsysorg/sglang:v0.5.17-cu129-runtime'
GPU='0'
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def main():
    write(OUT/'execution_started.json',{'unix_ns':time.time_ns(),'arms':['strict','cold'],'说明':'每臂一次，Strict失败不继续Cold'})
    cleanup={}
    try:
        identity=read(OUT/'source_identity.json')
        assert read(OUT/'cpu_gate.json')['status']=='PASS'
        for p,h in read(OUT/'cpu_gate.json')['sha256'].items():assert sha(p)==h,p
        image=json.loads(run(['docker','image','inspect',IMAGE]));assert image[0]['Id']==read(ROOT/'evaluation/runtime_control_closure_r2c0/image_identity.json')[0]['Id']
        write(OUT/'image_identity.json',image)
        paths=list(OUT.glob('*.py'))+[OUT/'source_identity.json',OUT/'target_contract.json',OUT/'cpu_tests.log',OUT/'cpu_gate.json',OUT/'req_api_source.json']
        for d in ('runtime_control_closure_r2c0','runtime_control_closure_r2c1a'):paths+=list((ROOT/'evaluation'/d).glob('*.py'))
        paths+=list((ROOT/'flowstate').glob('*.py'))+[ROOT/'flowstate/adapters/sglang.py',ROOT/'evaluation/strict_compatibility_b1c/runner.py']
        write(OUT/'implementation_lock.json',{'sha256':{str(p):sha(p) for p in paths},'image_id':image[0]['Id'],'host_gpu_index':GPU,'container_gpu':'cuda:0','scope':'单目标Strict与独立Cold，非性能比较'})
        for arm in ('strict','cold'):
            for row in identity['files']:assert sha(row['path'])==row['current_sha256'],row['path']
            for p,h in read(OUT/'implementation_lock.json')['sha256'].items():assert sha(p)==h,p
            apps=run(['nvidia-smi','-i',GPU,'--query-compute-apps=pid,process_name,used_memory','--format=csv'])
            assert len(apps.strip().splitlines())==1,'GPU存在其他进程'
            assert int(run(['nvidia-smi','-i',GPU,'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())<1000
            device=run(['nvidia-smi','-i',GPU,'--query-gpu=uuid,name,memory.used,memory.total','--format=csv']);assert 'H100 PCIe' in device
            write(OUT/f'{arm}_gpu_before.json',{'device':device,'processes':apps})
            name='flowstate-r2c1b2-'+arm
            command=['docker','run','--rm','--pull','never','--name',name,'--gpus','device='+GPU,'--network','host','--shm-size','32g']
            model=Path('/home/wjg/models/qwen3.5-9b')
            for src,dst,ro in ((ROOT,ROOT,True),(model,model,True),(OUT,Path('/r2out'),False)):
                command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
            env={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','FLOWSTATE_STEP5D_PORT':'49995','TRITON_CACHE_DIR':'/r2out/triton_cache_'+arm,'R2_ARM':arm,'PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
            for k,v in env.items():command+=['--env',k+'='+v]
            command+=['--workdir',str(ROOT),'--entrypoint','python3',IMAGE,'-B','-m','evaluation.runtime_control_closure_r2c1b2.runner']
            write(OUT/f'{arm}_command.json',command)
            try:
                with (OUT/f'{arm}.runtime.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                write(OUT/f'{arm}_process.json',{'returncode':result.returncode})
            finally:
                removed=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True);time.sleep(3)
                apps=run(['nvidia-smi','-i',GPU,'--query-compute-apps=pid,process_name,used_memory','--format=csv'])
                memory=int(run(['nvidia-smi','-i',GPU,'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())
                own=run(['docker','ps','-a','--filter','name='+name,'--format','{{.Names}}'])
                cleanup[arm]={'processes':apps,'memory_used_mib':memory,'own_container':own,'remove_stderr':removed.stderr,'status':'PASS' if not own and len(apps.strip().splitlines())==1 and memory<1000 else 'FAIL'}
                write(OUT/f'{arm}_cleanup.json',cleanup[arm])
            assert cleanup[arm]['status']=='PASS','GPU清理未通过'
            assert result.returncode==0,'本臂失败，停止后续生命周期'
            assert read(OUT/arm/'summary.json')['status']=='PASS'
    except BaseException as exc:
        write(OUT/'launch_failure.json',{'error':repr(exc),'completed_cleanup':list(cleanup)})
        raise
    finally:
        write(OUT/'gpu_cleanup.json',cleanup)
        with (OUT/'runtime.log').open('x') as f:
            for arm in ('strict','cold'):
                p=OUT/f'{arm}.runtime.log'
                if p.exists():f.write('\n[生命周期：'+arm+']\n'+p.read_text())
            f.write('\n[宿主机清理]\n'+json.dumps(cleanup,ensure_ascii=False))
if __name__=='__main__':main()
