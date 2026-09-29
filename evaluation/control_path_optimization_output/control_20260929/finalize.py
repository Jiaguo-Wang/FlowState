"""正式复测全部通过后生成并列报告及完整性清单。"""
import hashlib,json,time
from pathlib import Path
from evaluation.control_path_optimization_analysis import compare
from evaluation.rq3_openhands_neutral_collector import wait_gpu_stable
root=Path('/home/wjg/code/FlowState')
output=Path(__file__).resolve().parent
while True:
    p=output/'optimized_formal/progress.json'
    if p.exists():
        try:state=json.loads(p.read_text())
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        if state['status']=='FAIL':raise RuntimeError('正式运行失败，停止汇总')
        if state['completed']==216:break
    time.sleep(30)
result=compare(output)
protected=json.loads((output/'integrity_before.json').read_text())
sources=json.loads((output/'provenance/source_manifest.json').read_text())
changed=[]
for name,value in {**protected,**sources}.items():
    p=root/name
    actual={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} if p.exists() else None
    if actual!=value:changed.append(name)
gpu=wait_gpu_stable(gpu_index=0)
integrity={'status':'PASS' if not changed and gpu['stable'] else 'FAIL','protected_files':len(protected),
           'source_files':len(sources),'changed_files':changed,'gpu_cleanup':gpu,'旧冻结结果修改':False if not changed else '需要核查'}
(output/'integrity_summary.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
files={}
for p in output.rglob('*'):
    if p.is_file() and p.name not in ['frozen_manifest.json','completion.json','finalize.log']:
        files[str(p.relative_to(output))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(output/'frozen_manifest.json').write_text(json.dumps({'说明':'覆盖profiling、预验证、正式复测、源码副本与报告；排除清单自身、完成标记和汇总日志。','files':files},ensure_ascii=False,indent=2)+'\n')
(output/'completion.json').write_text(json.dumps({'status':integrity['status'],'formal_runs':216,'profiling_runs':72,'stable_net_benefit':result['statistically_stable_net_benefit_vs_both']},ensure_ascii=False,indent=2)+'\n')
print('正式复测、并列统计与完整性核验完成',integrity['status'],flush=True)
