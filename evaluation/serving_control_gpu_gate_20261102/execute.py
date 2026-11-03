"""仅执行两个已授权生命周期；任一门禁失败不重试、不回退。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
SOURCE = ROOT/'evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs'
OLD = ROOT/'evaluation/control_path_separation_20261101'
GPU = 'GPU-051c9abc-864a-9f31-c4f7-a737aed997ff'
IMAGE = 'sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926'
MODEL = '/home/wjg/models/qwen3.5-9b'
NAMES = ['prepared_inputs.json','cpu_candidate_prefix_mapping.json','cpu_gate.json',
         'cold_reference_C_A.json','cold_reference_C_B.json','cold_reference_C_C.json',
         'cold_reference_C_D.json','t2g1_plans.json','workloads_case_C.json','source_identity.json']
SCHEDULE = [('A_shadow_validation','shadow_validation'), ('B_serving','serving')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
        f.flush()
        os.fsync(f.fileno())


def query(args):
    return subprocess.check_output(args,text=True).strip()


def resources():
    free = shutil.disk_usage(ROOT).free
    assert free > 20*1024**3, '磁盘余量不足20GiB'
    rows = query(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
    row = next(r for r in rows.splitlines() if r.split(',')[1].strip()==GPU)
    assert 'H100 PCIe' in row and int(row.split(',')[3])<100, 'GPU型号或显存门禁失败：'+row
    processes = query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
    assert not any(r.split(',')[0].strip()==GPU for r in processes.splitlines()), 'GPU1存在计算进程'
    containers = query(['docker','ps','--format','{{.Names}}'])
    assert not any(r.startswith(('flowstate-serving-gate-', 'flowstate-sep-', 'flowstate-p04-')) for r in containers.splitlines()), '存在相关实验容器'
    ports = query(['ss','-ltn','sport = :50087'])
    assert len(ports.splitlines())==1, '控制端口被占用'
    return {'time_ns':time.time_ns(),'gpu':rows,'compute_processes':processes,'containers':containers,
            'free_bytes':free,'port':ports,'说明':'利用率仅记录；独占以计算进程和显存门禁判定，不操作GPU0'}


def check_identity(lock):
    for p,h in lock['code_sha256'].items():
        assert sha(Path(p))==h, '代码身份变化：'+p
    for p,h in lock['source_sha256'].items():
        assert sha(Path(p))==h, '原始证据/源码变化：'+p
    for n,h in lock['input_sha256'].items():
        assert sha(SOURCE/n)==h, '冻结输入变化：'+n


def freeze():
    prior = json.loads((OLD/'execution_lock_gpu1.json').read_text())
    assert prior['image']==IMAGE and prior['gpu_uuid']==GPU
    for n in NAMES:
        assert sha(SOURCE/n)==prior['input_sha256'][n], n
    cpu = (PKG/'cpu_tests.log').read_text()
    assert 'Ran 85 tests' in cpu and cpu.rstrip().endswith('OK'), 'CPU门禁尚未通过'
    cfg = json.loads((SOURCE/'prepared_inputs.json').read_text())['configuration']
    assert cfg['engine_configuration']['tp_size']==1 and cfg['sglang_version']=='0.5.17'
    assert cfg['engine_configuration']['max_mamba_cache_size']==28
    for entry in cfg['model_files']+cfg['tokenizer_files']:
        if 'sha256' in entry:
            assert sha(Path(entry['path']))==entry['sha256'], '模型或Tokenizer身份变化'
    cpu_lock = json.loads((ROOT/'evaluation/serving_control_interface_gate_20261101/implementation_lock.json').read_text())
    original = json.loads((OLD/'original_source_manifest.json').read_text())['files']
    sources = dict(original)
    sources.update(cpu_lock['sha256'])
    for path,h in sources.items():
        assert sha(Path(path))==h, '此前冻结证据变化：'+path
    save(PKG/'execution_lock.json',{'image':IMAGE,'model':MODEL,'gpu_uuid':GPU,'schedule':SCHEDULE,
        'policy':'FlowState-Greedy','K':3,'max_gpu_lifecycles':2,
        'code_sha256':{str(p):sha(p) for p in PKG.glob('*.py')},'source_sha256':sources,
        'input_sha256':{n:sha(SOURCE/n) for n in NAMES},'cpu_tests_sha256':sha(PKG/'cpu_tests.log'),
        '预检适配':'仅Token/Slot列表元组逐元素归一化及新增计时，不改变内容、安全或计时边界',
        '禁止事项':'不重试失败臂，不启动性能矩阵，不从两臂差值推断性能收益'})


def command(directory, name, selected_mode):
    cmd = ['docker','run','--rm','--pull','never','--name',name,'--gpus','device='+GPU,
           '--network','host','--shm-size','32g','--mount',f'type=bind,src={ROOT},dst={ROOT},readonly',
           '--mount',f'type=bind,src={MODEL},dst={MODEL},readonly',
           '--mount',f'type=bind,src={directory},dst=/b5out']
    env = {'CUDA_VISIBLE_DEVICES':GPU,'EXPECTED_GPU_UUID':GPU,'HF_HUB_OFFLINE':'1',
           'TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
           'FLOWSTATE_STEP5D_PORT':'50087','B5D4_RUN':'run','B5D4_ARM':'strict',
           'PILOT_MODE':'strict_warm','PILOT_POLICY':'FlowState-Greedy','FS_MEASURE_MODE':'1',
           'FS_SEPARATION_MODE':'serving','FS_ADMISSION_BASELINE_MODE':selected_mode,
           'FS_SEPARATION_VERIFY':'0','TRITON_CACHE_DIR':'/b5out/triton_cache',
           'PYTHONPATH':f'{ROOT}:{ROOT}/tests/runtime:{ROOT}/motivation/artifacts/wp3b_gate_20260820'}
    for k,v in env.items():
        cmd += ['--env',k+'='+v]
    return cmd+['--workdir',str(ROOT),'--entrypoint','python3',IMAGE,'-B',str(PKG/'device_entry.py')]


def validate(directory, selected_mode):
    from evaluation.serving_control_gpu_gate_20261102 import interface
    rdir = directory/'run'
    assert not (rdir/'failure.json').exists() and not (rdir/'failure_after_summary.json').exists()
    s = json.loads((rdir/'summary.json').read_text())
    r = json.loads((rdir/'reconciliation.json').read_text())
    p = json.loads((rdir/'post_request.json').read_text())
    assert s['status']=='PASS' and s['oracle_match'] and s['cycle']['identity_exact']
    assert s['selection']['selected_cpu']==['C_B_P05120','C_C_P06144','C_D_P07168']
    assert s['separation']['admission_baseline_mode']==selected_mode
    interface.validate_response(r,selected_mode)
    assert r['reconciliation']['status']=='PASS' and len(r['operations'])==13
    assert all(x['returned'] for x in r['operations'])
    assert len(r['after']['free_slots'])==25 and sum(bool(n['slots']) for n in r['after']['nodes'])==3
    assert p['state']['selected_components']==r['admission_baseline']['selected_components']
    assert p['state']['selected_payload']==r['selected_payload_after']
    assert p['state']['fa_protected_payload']==r['fa_protected_payload']
    assert all(p['state']['fa_map'].get(k)==v for k,v in r['fa_map_before'].items())
    assert s['audit_entries_total']==2 and len(p['trace']['admission'])==4
    assert len(s['pending_rows'])==4
    for row in s['pending_rows']:
        cold = json.loads((directory/f"cold_reference_{row['pending_id']}.json").read_text())
        assert len(row['output_ids'])==8 and row['output_ids']==cold['output_ids']
        assert row['cold_output_match']
    assert json.loads((rdir/'engine_cleanup.json').read_text())['shutdown_returned']
    assert json.loads((directory/'host_exit.json').read_text())['returncode']==0
    assert (directory/'gpu_cleanup.json').exists()
    assert len(s['separation']['transfers'])==1 and s['separation']['transfers'][0]['digest_verified']
    save(directory/'independent_validation.json',{'status':'PASS','mode':selected_mode,
         'selected_cpu':s['selection']['selected_cpu'],'E':[r['E'] for r in s['pending_rows']],
         'G':[r['G'] for r in s['pending_rows']],'evictions':13,'cold_matches':4,
         'provenance':'PROVENANCE_UNVERIFIED','说明':'各臂只用当前生命周期绑定，不比较跨生命周期指针'})


def run_arm(arm, selected_mode, lock):
    check_identity(lock)
    before = resources()
    directory = PKG/'arms'/arm
    directory.mkdir(parents=True,exist_ok=False)
    for n in NAMES:
        shutil.copyfile(SOURCE/n,directory/n)
    name = 'flowstate-serving-gate-'+arm.lower().replace('_','-')
    cmd = command(directory,name,selected_mode)
    save(directory/'command.json',cmd)
    save(directory/'preflight.json',before)
    save(directory/'experiment_identity.json',{'arm':arm,'mode':selected_mode,'image':IMAGE,'gpu_uuid':GPU,
         'input_sha256':lock['input_sha256'],'code_sha256':lock['code_sha256']})
    start = time.monotonic_ns()
    rc = None
    print('启动',arm,selected_mode,flush=True)
    try:
        with (directory/'runtime.log').open('x') as f:
            process = subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            rc = process.wait(timeout=5400)
        assert rc==0, 'GPU生命周期异常退出'
    except BaseException:
        # 只能清理本轮唯一命名容器，不影响任何其他容器或GPU任务。
        subprocess.run(['docker','stop','--time','30',name],capture_output=True)
        raise
    finally:
        save(directory/'host_exit.json',{'returncode':rc,'start_ns':start,'end_ns':time.monotonic_ns()})
        try:
            save(directory/'gpu_cleanup.json',resources())
        except BaseException as exc:
            save(directory/'gpu_cleanup_failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
    validate(directory,selected_mode)
    check_identity(lock)
    print('完成',arm,'PASS',flush=True)


def main():
    lock_path = PKG/'execution_lock.json'
    try:
        if not lock_path.exists():
            freeze()
        lock = json.loads(lock_path.read_text())
        assert lock['schedule']==[list(x) for x in SCHEDULE], '调度变化'
        assert not (PKG/'arms').exists(), '已有执行尝试；禁止自动重试或覆盖'
        save(PKG/'gpu_preflight.json',resources())
        for arm,selected_mode in SCHEDULE:
            run_arm(arm,selected_mode,lock)
        save(PKG/'gpu_cleanup.json',resources())
        save(PKG/'execution_result.json',{'status':'GPU_CORRECTNESS_PASS','gpu_lifecycles':2})
    except BaseException as exc:
        save(PKG/'execution_failure.json',{'status':'GPU_CORRECTNESS_FAIL','error':repr(exc),
             'traceback':traceback.format_exc(),'说明':'停止，不重试、不继续后续臂'})
        raise


if __name__=='__main__':
    main()
