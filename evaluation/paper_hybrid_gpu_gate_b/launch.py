"""场景B单臂受控启动（宿主机执行）：GPU 空闲检查→冻结镜像→docker 运行→落盘命令/日志/清理证据。

单臂 b1_hybrid_greedy（C/K3 冻结环境，K=3，strict；Cold 参照复用 g1_CB_FS 冻结 cold_reference_C_B.json，
不另起 Cold 生命周期）。已启动即拒绝重跑；任何失败立即停止并保留证据。
"""
import json,subprocess,time,socket
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
OUT=ROOT/'evaluation/paper_hybrid_gpu_gate_b'
FIX=ROOT/'evaluation/paper_hybrid_gpu_fix_b'
RUN='b1_hybrid_greedy';PORT=49998;GPU=0;TIMEOUT_S=2400
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout
def write(name,x):
    p=OUT/name;assert not p.exists(),'拒绝覆盖：'+name
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
    import hashlib;return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (OUT/'execution_started.json').exists(),'禁止重跑已启动验证'
    source=json.loads((OUT/'source_identity.json').read_text())
    image=json.loads(run(['docker','image','inspect',source['image_id']]))[0];assert image['Id']==source['image_id']
    lock={str(p):sha(p) for p in list(OUT.glob('*.py'))+list((OUT).glob('*.json'))}
    lock.update({str(p):sha(p) for p in FIX.glob('*.py')})
    lock.update(source['hybrid_entry_sources'])
    write('execution_lock.json',{'files':lock,'image':image['Id'],'runner':'evaluation.paper_hybrid_gpu_fix_b.runner','scenario':'B_ck3_greedy_branch'})
    write('execution_started.json',{'unix_ns':time.time_ns(),'run':RUN})
    for p,h in lock.items():assert sha(p)==h,'锁定文件已变化：'+p
    apps=run(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid,process_name,used_memory','--format=csv']);assert len(apps.strip().splitlines())==1,'GPU有其他任务'
    gpu=run(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid,name,memory.used,memory.total,temperature.gpu,clocks.sm,clocks.mem,power.limit','--format=csv']);assert 'H100 PCIe' in gpu
    assert int(run(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())<1000
    container='flowstate-hybgate-b';assert not run(['docker','ps','-a','--filter','name='+container,'--format','{{.Names}}']).strip()
    with socket.socket() as sock:sock.bind(('127.0.0.1',PORT))
    write(RUN+'_gpu_before.json',{'gpu':gpu,'apps':apps,'unix_ns':time.time_ns()})
    cmd=['docker','run','--rm','--pull','never','--name',container,'--gpus','device=%d'%GPU,'--network','host','--shm-size','32g']
    for src,dst,ro in [(ROOT,ROOT,True),(Path('/home/wjg/models/qwen3.5-9b'),Path('/home/wjg/models/qwen3.5-9b'),True),(OUT,Path('/b5out'),False)]:cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    env={'CUDA_VISIBLE_DEVICES':str(GPU),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
         'FLOWSTATE_STEP5D_PORT':str(PORT),'TRITON_CACHE_DIR':'/b5out/triton_cache_'+RUN,'B5D4_ARM':'strict','B5D4_RUN':RUN,'PILOT_PENDING':'C_B',
         'PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
    for k,v in env.items():cmd+=['--env',k+'='+v]
    cmd+=['--workdir',str(ROOT),'--entrypoint','python3',image['Id'],'-B','-m','evaluation.paper_hybrid_gpu_fix_b.runner']
    write(RUN+'_command.json',cmd)
    print('启动',RUN,flush=True)
    r=None
    try:
        with (OUT/(RUN+'.runtime.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=TIMEOUT_S)
        write(RUN+'_process.json',{'returncode':r.returncode})
    finally:
        removed=subprocess.run(['docker','rm','-f',container],capture_output=True,text=True);obs=[]
        for _ in range(3):
            time.sleep(3);mem=int(run(['nvidia-smi','-i',str(GPU),'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip());apps=run(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid,process_name,used_memory','--format=csv']);obs.append({'memory_mib':mem,'processes':apps})
        own=run(['docker','ps','-a','--filter','name='+container,'--format','{{.Names}}']).strip()
        cleanup={'status':'PASS' if not own and all(x['memory_mib']<100 and len(x['processes'].strip().splitlines())==1 for x in obs) else 'FAIL','observations':obs,'own_container':own,'remove_stderr':removed.stderr}
        write(RUN+'_cleanup.json',cleanup);write('gpu_cleanup.json',cleanup)
    assert cleanup['status']=='PASS' and r and r.returncode==0,'执行或清理失败'
    assert json.loads((OUT/RUN/'summary.json').read_text())['status']=='PASS','缺少完整正确性摘要'
    print('场景B单臂完成',RUN,flush=True)
if __name__=='__main__':main()
