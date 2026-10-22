"""只启动一次g001只读生命周期；任何失败不重试，只清理本轮容器。"""
import json
import subprocess
import time
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write
IMAGE='lmsysorg/sglang:v0.5.17-cu129-runtime'
NAME='flowstate-r2c0-g001-readonly'

def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def main():
    write(OUT/'execution_started.json',{'unix_ns':time.time_ns(),'说明':'只允许一次生命周期，不自动重试'})
    started=False
    try:
        source=read(OUT/'source_identity.json');assert source['status']=='PASS'
        for row in source['files']:assert sha(row['path'])==row['current_sha256'],row['path']
        assert '59 passed' in (OUT/'cpu_tests_final.log').read_text()
        current_image=json.loads(run(['docker','image','inspect',IMAGE]))
        expected=read(ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/image_identity.json')
        assert current_image[0]['Id']==expected[0]['Id']
        write(OUT/'image_identity.json',current_image)
        apps=run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv'])
        gpu=run(['nvidia-smi','-i','0','--query-gpu=name,memory.used,memory.total','--format=csv'])
        assert len(apps.strip().splitlines())==1,'GPU0存在计算进程'
        assert 'H100 PCIe' in gpu
        assert int(run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())<1000
        write(OUT/'gpu_before.json',{'gpu':gpu,'processes':apps})
        paths=[OUT/n for n in ('bridge.py','runner.py','launch.py','prepare.py','test_bridge.py','cpu_tests_final.log','source_identity.json')]
        paths.extend(ROOT/n for n in ('flowstate/strict_control.py','flowstate/strict_epoch.py','flowstate/optimizer.py','flowstate/recovery_model.py'))
        write(OUT/'implementation_lock.json',{'sha256':{str(p):sha(p) for p in paths}})
        command=['docker','run','--rm','--pull','never','--name',NAME,'--gpus','device=0','--network','host','--shm-size','32g']
        model=Path('/home/wjg/models/qwen3.5-9b')
        for src,dst,ro in ((ROOT,ROOT,True),(model,model,True),(OUT,Path('/r2out'),False)):
            command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
        env={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','FLOWSTATE_STEP5D_PORT':'49993','TRITON_CACHE_DIR':'/r2out/triton_cache','PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
        for k,v in env.items():command+=['--env',k+'='+v]
        command+=['--workdir',str(ROOT),'--entrypoint','python3',IMAGE,'-B','-m','evaluation.runtime_control_closure_r2c0.runner']
        write(OUT/'command.json',command)
        with (OUT/'runtime.log').open('x') as log:
            started=True
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        write(OUT/'process_result.json',{'returncode':result.returncode})
    except BaseException as exc:
        write(OUT/'launch_failure.json',{'error':repr(exc),'started':started})
        raise
    finally:
        if started:
            cleanup=subprocess.run(['docker','rm','-f',NAME],capture_output=True,text=True)
            # 只查询本轮容器和GPU，不终止任何其他进程。
            time.sleep(3)
            write(OUT/'gpu_cleanup.json',{'remove_returncode':cleanup.returncode,'remove_stderr':cleanup.stderr,'gpu':run(['nvidia-smi','-i','0','--query-gpu=name,memory.used,memory.total','--format=csv']),'processes':run(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_memory','--format=csv']),'own_container':run(['docker','ps','-a','--filter','name='+NAME,'--format','{{.Names}}'])})
if __name__=='__main__':main()
