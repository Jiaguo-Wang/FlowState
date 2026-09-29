"""正式采集结束后生成统计与完整性报告，不新增GPU工作。"""
import hashlib
import json
from pathlib import Path
import time
from evaluation.resume_epoch_e2e_analysis import analyze
from evaluation.rq3_openhands_neutral_collector import wait_gpu_stable
root=Path('/home/wjg/code/FlowState')
output=Path(__file__).resolve().parent
while True:
    progress=output/'progress.json'
    if progress.exists():
        try:
            state=json.loads(progress.read_text())
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        if state.get('status')=='FAIL':
            raise RuntimeError('正式采集失败，停止汇总')
        if state.get('completed')==216:
            break
    time.sleep(30)
result=analyze(output)
before=json.loads((output/'integrity_before.json').read_text())
changed=[]
for name,expected in before.items():
    path=root/name
    actual={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} if path.exists() else None
    if actual!=expected:
        changed.append(name)
gpu=wait_gpu_stable(gpu_index=0)
integrity={'status':'PASS' if not changed and gpu['stable'] else 'FAIL','checked_files':len(before),
           'changed_files':changed,'frozen_artifacts_modified':bool(changed),'gpu_cleanup':gpu}
(output/'integrity_summary.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
files={}
for path in output.rglob('*'):
    if path.is_file() and path.name not in ('frozen_manifest.json','finalize.log','completion.json'):
        files[str(path.relative_to(output))]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(output/'frozen_manifest.json').write_text(json.dumps({'说明':'本轮正式结果与源码副本的完整性清单。清单、完成标记和后台汇总日志不参与自引用。','files':files},ensure_ascii=False,indent=2)+'\n')
(output/'completion.json').write_text(json.dumps({'status':integrity['status'],'runs':216,'requests':864,
    'analysis':'analysis/final_report.md','integrity':'integrity_summary.json'},ensure_ascii=False,indent=2)+'\n')
print('正式216次采集、统计与完整性验证完成',integrity['status'],flush=True)
