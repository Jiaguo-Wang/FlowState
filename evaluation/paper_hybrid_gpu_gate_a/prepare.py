"""场景A门禁装配（宿主机执行）：从 b5d4b 冻结门禁复制输入并锁定被测生产入口源码身份。

只读复制，不修改任何冻结目录；重复执行拒绝覆盖。冻结链：
- prepared_inputs.json / cpu_gate.json ← b5d4b（先按其 execution_lock 哈希核验源文件未变）；
- frozen_cold_reference.json ← b5d4b cold/output.json（其 8 Token 已被 10 个严格臂逐Token一致验证，
  再按 cold_reference_correctness.json 的 cold_output_ids 复核）；
- source_identity.json：镜像ID + 镜像内Runtime源码哈希（沿用 b5d4b，运行时在容器内逐项核验）
  + hybrid_entry_sources（本轮被测生产入口 flowstate/*.py 的当前哈希）。
"""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
SRC=ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4b'
OUT=ROOT/'evaluation/paper_hybrid_gpu_gate_a'
FIX=ROOT/'evaluation/paper_hybrid_gpu_fix_a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_once(name,data):
    p=OUT/name
    assert not p.exists(),'拒绝覆盖既有文件：'+name
    p.write_bytes(data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,indent=2).encode()+b'\n')
def main():
    lock=json.loads((SRC/'execution_lock.json').read_text())['files']
    # 冻结输入：先核验 b5d4b 源文件与其执行锁一致，再逐字节复制
    for name in ('prepared_inputs.json','cpu_gate.json'):
        src=SRC/name;h=lock[str(src)]
        assert sha(src)==h,'b5d4b 冻结输入与其执行锁不一致：'+name
        write_once(name,src.read_bytes())
    # 冻结 Cold 参照：cold/output.json 的 8 Token 与 cold_reference_correctness.json 记录复核
    cold=json.loads((SRC/'cold'/'output.json').read_text())
    ref=json.loads((SRC/'cold_reference_correctness.json').read_text())['comparisons']
    cold_ids=[v['cold_output_ids'] for v in ref.values()]
    assert cold_ids and all(x==cold['output_ids'] for x in cold_ids),'Cold参照与b5d4b记录不一致'
    write_once('frozen_cold_reference.json',(SRC/'cold'/'output.json').read_bytes())
    # 源身份：镜像内 Runtime 源码沿用 b5d4b 冻结哈希；生产入口按当前源码锁定
    b5d4b_source=json.loads((SRC/'source_identity.json').read_text())
    identity={'image_id':b5d4b_source['image_id'],
              'runtime_sources':b5d4b_source['runtime_sources'],
              'hybrid_entry_sources':{str(p):sha(p) for p in sorted((ROOT/'flowstate').rglob('*.py'))},
              'note':'runtime_sources 为镜像内 sglang 源码（容器内运行时核验）；hybrid_entry_sources 为被测生产入口 flowstate/*.py 当前哈希；镜像ID 与 b5d4b 一致'}
    write_once('source_identity.json',identity)
    # 语法预检查（不执行）
    for p in sorted(FIX.glob('*.py')):compile(p.read_text(),str(p),'exec')
    provenance={'prepared_inputs':{'source':str(SRC/'prepared_inputs.json'),'sha256':sha(SRC/'prepared_inputs.json'),'verified_against':'b5d4b execution_lock.json'},
                'cpu_gate':{'source':str(SRC/'cpu_gate.json'),'sha256':sha(SRC/'cpu_gate.json'),'verified_against':'b5d4b execution_lock.json'},
                'frozen_cold_reference':{'source':str(SRC/'cold'/'output.json'),'sha256':sha(SRC/'cold'/'output.json'),'verified_against':'b5d4b cold_reference_correctness.json（10臂cold_output_ids）'},
                'environment':'B5C3 冻结环境（8历史+4Pending，K=2），Epoch 形成为冻结历史重放+实时观察，非全在线构建'}
    write_once('input_provenance.json',provenance)
    print('场景A门禁装配PASS：prepared_inputs/cpu_gate/cold参照已冻结复制；flowstate 生产入口哈希已锁定（%d 个文件）'%len(identity['hybrid_entry_sources']))
if __name__=='__main__':main()
