"""下一阶段批次启动器；只有显式执行参数才调用Docker。"""
import argparse
import subprocess
import traceback
from pathlib import Path
from .contracts import *
IMAGE='lmsysorg/sglang:v0.5.17-cu129-runtime'
MODEL=Path('/home/wjg/models/qwen3.5-9b')

def execute_plan(out,run_group):
    """固定顺序且不补位；失败停止并记录其余未执行组。"""
    counts={'A':0,'B':0,'C':0};completed=[]
    append_event(out,'发现案例仅引用',group=0,source_sessions_charged=4)
    for group in GROUPS:
        append_event(out,'计划执行',group=group,source_sessions_charged=4)
        try:
            summary=run_group(group)
            primary=summary.get('structure',{}).get('primary')
            if summary['status']!='CAPTURED':reason='关键证据UNKNOWN'
            elif primary is None:reason='没有结构标签'
            elif counts[primary]>=4:reason='主层配额已满，不转派'
            else:reason='入选';counts[primary]+=1
            append_event(out,'处理完成',group=group,decision=reason,summary=summary)
            completed.append({'group':group,'decision':reason,'summary':summary})
        except BaseException as exc:
            append_event(out,'批次停止',group=group,error=repr(exc),traceback=traceback.format_exc())
            for later in GROUPS:
                if later>group:append_event(out,'未执行',group=later,reason='前序失败，无补位或自动重试')
            raise
    return {'groups':completed,'accepted_by_stratum':counts,'source_sessions':28,'unused_source_session_quota':2}

def gpu_check():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],check=True,capture_output=True,text=True)
    mem=subprocess.run(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    require(int(mem.stdout.strip())<1000,'GPU0未清理或被占用')
    return result.stdout

def main():
    p=argparse.ArgumentParser(description='固定g001至g006独立生命周期受控采集')
    p.add_argument('--execute-b4b1',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require(a.execute_b4b1,'缺少下一阶段显式执行参数')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        lock=verify_lock();write_new(out/'execution_lock.json',lock)
        image=json.loads(subprocess.run(['docker','image','inspect',IMAGE],check=True,capture_output=True,text=True).stdout)
        write_new(out/'image_identity.json',image)
        expected=read(ROOT/'evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2/image_identity.json')
        require(image[0]['Id']==expected[0]['Id'],'容器镜像身份变化')
        def run_group(group):
            group_contract(group);name=f'flowstate-b4b1-g{group:03d}'
            (out/f'g{group:03d}.gpu_before.csv').write_text(gpu_check())
            command=['docker','run','--rm','--pull','never','--name',name,'--gpus','device=0','--network','host','--shm-size','32g']
            for src,dst,ro in [(ROOT,ROOT,True),(MODEL,MODEL,True),(DATA.parent,DATA.parent,True),(out,Path('/b4out'),False)]:
                command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
            env={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','FLOWSTATE_STEP5D_PORT':'49986','TRITON_CACHE_DIR':'/b4out/triton_cache','PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
            for k,v in env.items():command+=['--env',k+'='+v]
            command+=['--workdir',str(ROOT),'--entrypoint','python3',IMAGE,'-B','-m','evaluation.strict_compatibility_b4b0.capture','--execute-b4b1','--group',str(group),'--out',f'/b4out/g{group:03d}']
            write_new(out/f'g{group:03d}.command.json',command)
            try:
                with (out/f'g{group:03d}.runtime.log').open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            finally:
                cleanup=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True)
                append_event(out,'容器清理',group=group,returncode=cleanup.returncode,stderr=cleanup.stderr)
                (out/f'g{group:03d}.gpu_after.csv').write_text(gpu_check())
            return read(out/f'g{group:03d}'/'summary.json')
        write_new(out/'batch_summary.json',execute_plan(out,run_group))
    except BaseException as exc:
        append_event(out,'启动或执行异常',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        finalize_manifest(out)
if __name__=='__main__':main()
