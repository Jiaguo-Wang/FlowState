"""按冻结E2E协议测量FlowState控制路径的同步等待及内部子步骤。"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from time import perf_counter_ns
from evaluation import resume_epoch_e2e as frozen


class ClientProfile:
    """按原有每调用一连接协议记录编码、连接、排队与响应边界。"""
    def __init__(self):
        self.rows=[]

    def install(self):
        """仅对当前worker的分配边界之后的调用启用计时。"""
        from targeted_probe import ControlClient
        original=ControlClient._call
        rows=self.rows
        active=[False]
        def call(client,request):
            nonce=str(request.get('nonce',''))
            if nonce.endswith(':common'):
                active[0]=True
            if not active[0]:
                return original(client,request)
            phase=('common' if nonce.endswith(':common') else 'reconciliation' if nonce.endswith(':reconcile')
                   else 'policy' if ':frontier:' in nonce else 'request_validation')
            request={**request,'_control_profile':True}
            ts={'client_start_ns':perf_counter_ns()}
            with socket.create_connection(('127.0.0.1',client.port),timeout=client.timeout_s) as sock:
                ts['connected_ns']=perf_counter_ns()
                sock.settimeout(client.timeout_s)
                encode_start=perf_counter_ns()
                payload=(json.dumps(request)+'\n').encode()
                ts['encode_ns']=perf_counter_ns()-encode_start
                sock.sendall(payload)
                ts['sent_ns']=perf_counter_ns()
                line=sock.makefile('rb').readline()
                ts['received_ns']=perf_counter_ns()
            decode_start=perf_counter_ns()
            response=json.loads(line.decode())
            ts['decode_ns']=perf_counter_ns()-decode_start
            ts['client_end_ns']=perf_counter_ns()
            if not response.get('ok'):
                raise RuntimeError('原控制调用失败：'+str(response))
            rows.append({'phase':phase,'action':request.get('action',request['op']),
                         'request_bytes':len(payload),'response_bytes':len(line),**ts,
                         **response.get('_control_profile',{})})
            return response
        ControlClient._call=call


def worker(output,run_id):
    """直接调用冻结worker，仅替换为有计时探针的同语义入口。"""
    import rq6_batched_control_transport as batch
    from control_path_profile_transport import ProfileEngine
    batch.RQ6BatchedControlEngine=ProfileEngine
    profile=ClientProfile()
    profile.install()
    code=frozen.worker(output,run_id)
    frozen.write(output/'runs'/run_id/'profile.json',profile.rows)
    return code


def initialize(output):
    """全部24快照、三次重复均保留，只选择FlowState做瓶颈诊断。"""
    output.mkdir(parents=True,exist_ok=False)
    baseline=frozen.ROOT/'evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928'
    plan=[p for p in json.loads((baseline/'plan.json').read_text()) if p['policy']=='FlowState']
    frozen.require(len(plan)==72,'诊断计划不完整')
    frozen.write(output/'plan.json',plan)
    frozen.write(output/'protocol.json',{'说明':'仅计时探针；不修改冻结实现、状态操作、检查或请求。24快照乘3次FlowState独立生命周期。',
        'baseline':str(baseline),'统计口径':'每子步骤先累计每epoch，再报告72个epoch的均值/P50/P95；嵌套包含时间不重复相加。',
        '限制':'探针自身增加计时与响应字节；新测量不能伪称历史293.093ms的精确内部重建。'})


def run(output):
    """独立运行诊断计划，关键路径只在内存收集日志。"""
    plan=json.loads((output/'plan.json').read_text())
    (output/'logs').mkdir(exist_ok=True)
    frozen.rq4.wait_gpu_stable(gpu_index=0)
    for index,p in enumerate(plan,1):
        path=output/'runs'/p['run_id']/'record.json'
        if path.exists():
            frozen.require(json.loads(path.read_text())['status']=='PASS','已有失败，禁止自动重试')
            continue
        proc=subprocess.Popen([sys.executable,'-m','evaluation.control_path_profile','--worker','--output',str(output),'--run-id',p['run_id']],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            log,_=proc.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            log,_=proc.communicate()
        (output/'logs'/(p['run_id']+'.log')).write_bytes(log)
        cleanup=frozen.rq4.wait_gpu_stable(gpu_index=0)
        record=json.loads(path.read_text()) if path.exists() else {'status':'FAIL','correctness':{}}
        record['correctness']['gpu_cleanup']=cleanup['stable']
        record['gpu_cleanup']=cleanup
        record['status']='PASS' if proc.returncode==0 and record['status']=='PASS_PENDING_CLEANUP' and all(record['correctness'].values()) else 'FAIL'
        frozen.write(path,record)
        frozen.write(output/'progress.json',{'completed':index,'planned':72,'status':record['status'],'latest':p['run_id']})
        print(index,p['run_id'],record['status'],flush=True)
        frozen.require(record['status']=='PASS','诊断失败，停止')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='保持原实现的控制路径计时诊断')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.initialize:initialize(args.output)
    elif args.worker:raise SystemExit(worker(args.output,args.run_id))
    else:run(args.output)
