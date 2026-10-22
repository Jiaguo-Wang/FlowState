"""归档 CPU 模拟链、测试结果与新版本身份，不调用真实 Runtime。"""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tests.test_strict_epoch import load
from tests.test_strict_control import FakeRuntime
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.controller import StateController
from flowstate.strict_control import StrictReconciliationError
OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    epoch=load(1,2);plan=GlobalOptimizer(RecoveryCostModel()).plan_strict(epoch)
    success=FakeRuntime(epoch);before=asdict(success.view)
    result=StateController.reconcile_strict(epoch,plan,success.bridge())
    failed=FakeRuntime(epoch);failed.fail_at=2
    try:StateController.reconcile_strict(epoch,plan,failed.bridge())
    except StrictReconciliationError as error:failure=error.ledger
    else:raise AssertionError('模拟故障未拒绝')
    write('integration_trace.json',dict(evidence_level='CPU_MOCK_ONLY',说明='当前模拟Node从10000开始，slot为独立构造，绝非历史物理身份认证。',plan=asdict(plan),before=before,success=result,after=asdict(success.view),operations=success.operations,partial_failure=failure,failure_operations=failed.operations))
    log=(OUT/'tests_final.log').read_text();assert '118 passed, 16 subtests passed' in log
    names=[n.name for n in ast.walk(ast.parse((ROOT/'tests/test_strict_control.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
    write('safety_tests.json',dict(status='PASS',evidence_level='CPU_MOCK_ONLY',passed=118,frozen_regression_subtests=16,safety_tests=[dict(name=n,status='PASS') for n in names],test_log_sha256=sha(OUT/'tests_final.log'),说明='28项新增安全测试；其余包含R2-A及Legacy回归，不是118次GPU验证。'))
    paths=[ROOT/p for p in ['flowstate/controller.py','flowstate/strict_control.py','flowstate/strict_epoch.py','flowstate/optimizer.py','flowstate/recovery_model.py','flowstate/adapters/sglang.py','flowstate/state_catalog.py','tests/test_strict_control.py','tests/test_strict_epoch.py','tests/runtime/rq6_batched_control_transport.py','tests/runtime/control_path_optimized_transport.py','evaluation/FORMAL_RECOVERY_MODEL.md','evaluation/runtime_control_closure_r2a/integrity.json']]
    raw=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01'
    paths += [raw/f'g{i:03d}/snapshot.json' for i in range(1,5)]
    paths += [raw/'execution_lock.json',ROOT/'evaluation/strict_compatibility_b4c/results.json',ROOT/'evaluation/strict_compatibility_b4c/evidence_manifest.json',Path(__file__)]
    protected=[]
    for manifest in [raw/'manifest.json',ROOT/'evaluation/strict_compatibility_b4c/evidence_manifest.json']:
        for name,h in json.loads(manifest.read_text())['files'].items():
            p=Path(name) if Path(name).is_absolute() else manifest.parent/name
            assert sha(p)==h;protected.append(dict(path=str(p),sha256=h))
    prior=json.loads((ROOT/'evaluation/runtime_control_closure_r2a/integrity.json').read_text())
    for name,h in prior['current_implementation'].items():assert sha(Path(name))==h
    write('source_identity.json',dict(schema='flowstate.r2b.cpu_contract.v1',scope='仅CPU合同，无真实Runtime执行锁或GPU后置条件认证',python=sys.version,sha256={str(p):sha(p) for p in paths},frozen_checks=protected,r2a_implementation_unchanged=True,mock_configuration=dict(pool_capacity=28,first_node_id=10000,engine='模拟新Engine',generation='UNKNOWN，依赖连续禁止Admission合同'),说明='新身份记录不覆盖旧锁；真实R2-C需锁定实际Runtime桥接及模型执行环境。'))
if __name__=='__main__':main()
