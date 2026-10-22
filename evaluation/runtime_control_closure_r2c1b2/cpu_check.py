"""在无GPU容器中核对源码、执行真实接口与既有协议测试，并锁定CPU门禁。"""
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/r2out')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import sglang.srt.managers.schedule_batch as module
    from sglang.srt.managers.scheduler import Scheduler
    rows={}
    for obj in (module.Req,module.ScheduleBatch.prepare_for_extend,Scheduler.run_batch):
        obj=inspect.unwrap(obj)
        lines,line=inspect.getsourcelines(obj);path=inspect.getsourcefile(obj)
        rows[obj.__qualname__]={'path':path,'line':line,'source':''.join(lines),'file_sha256':sha(path)}
    (OUT/'req_api_source.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    identity=json.loads((OUT/'source_identity.json').read_text())
    for path,digest in identity['runtime_sources'].items():assert sha(path)==digest,path
    args=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','evaluation/runtime_control_closure_r2c1b2/test_real_api.py','evaluation/runtime_control_closure_r2c1b2/test_protocol.py','evaluation/runtime_control_closure_r2c1a/test_bridge.py','tests/test_strict_control.py']
    result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    (OUT/'cpu_tests.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    paths=list((ROOT/'evaluation/runtime_control_closure_r2c1b2').glob('*.py'))+[OUT/'source_identity.json',OUT/'cpu_tests.log',OUT/'req_api_source.json']
    (OUT/'cpu_gate.json').write_text(json.dumps({'status':'PASS','sha256':{str(ROOT/'evaluation/runtime_control_closure_r2c1b2'/p.name) if p.parent==OUT else str(p):sha(p) for p in paths},'runtime_sources':identity['runtime_sources'],'说明':'真实Req/Batch及Scheduler源码，CPU合同，不启动Engine'},ensure_ascii=False,indent=2))
    print(result.stdout)
if __name__=='__main__':main()
