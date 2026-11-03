"""场景B门禁装配（宿主机执行）：从 c_formal_perf_gpu/g1_CB_FS 冻结臂复制输入并锁定被测生产入口源码身份。

只读复制，不修改任何冻结目录；重复执行拒绝覆盖。冻结链：
- prepared_inputs.json / cpu_candidate_prefix_mapping.json / cpu_gate.json / policy_plans.json /
  cold_reference_C_B.json ← g1_CB_FS（C/K3 正式实验臂的冻结输入；policy_plans 本轮仅作事后对照 Oracle）；
- source_identity.json：镜像ID + 镜像内Runtime源码哈希（沿用 g1_CB_FS，运行时在容器内逐项核验）
  + hybrid_entry_sources（本轮被测生产入口 flowstate/*.py 的当前哈希）
  + frozen_input_hashes（复制时逐文件记录，供审计追溯）。
"""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
SRC=ROOT/'evaluation/c_formal_perf_gpu/g1_CB_FS'
OUT=ROOT/'evaluation/paper_hybrid_gpu_gate_b'
FIX=ROOT/'evaluation/paper_hybrid_gpu_fix_b'
NAMES=('prepared_inputs.json','cpu_candidate_prefix_mapping.json','cpu_gate.json','policy_plans.json','cold_reference_C_B.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_once(name,data):
    p=OUT/name
    assert not p.exists(),'拒绝覆盖既有文件：'+name
    p.write_bytes(data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,indent=2).encode()+b'\n')
def main():
    # 冻结输入复制：policy_plans.json 在本轮仅作求解完成后的事后对照 Oracle（控制路径不用于选择）
    frozen_hashes={}
    for name in NAMES:
        src=SRC/name;frozen_hashes[str(src)]=sha(src)
        write_once(name,src.read_bytes())
    # 语义预核验：冻结 FS 选集/E-G 合同/Cold 参照与 g1_CB_FS 冻结运行记录一致
    rec=json.loads((SRC/'run'/'reconciliation.json').read_text())
    plans=json.loads((SRC/'policy_plans.json').read_text());fs=plans['FlowState-Greedy']
    cpu_to_phys={m['cpu_checkpoint_id']:m['physical_checkpoint_id'] for m in rec['census']['mapped']}
    assert {cpu_to_phys[c] for c in fs['selected_checkpoint_ids']}==set(rec['plan']['selected_ids']),'冻结FS选集与g1运行记录不一致'
    assert abs(rec['plan']['recovery_cost_ms']-154.34895561599998)<=1e-9,'冻结J与公开冻结值不一致'
    cold=json.loads((SRC/'cold_reference_C_B.json').read_text())
    assert cold['output_ids']==json.loads((SRC/'run'/'output.json').read_text())['output_ids'],'Cold参照与g1运行输出不一致'
    # 源身份：镜像内 Runtime 源码沿用 g1_CB_FS 冻结哈希；生产入口按当前源码锁定
    g1_source=json.loads((SRC/'source_identity.json').read_text())
    identity={'image_id':g1_source['image_id'],
              'runtime_sources':g1_source['runtime_sources'],
              'hybrid_entry_sources':{str(p):sha(p) for p in sorted((ROOT/'flowstate').rglob('*.py'))},
              'note':'runtime_sources 为镜像内 sglang 源码（容器内运行时核验）；hybrid_entry_sources 为被测生产入口 flowstate/*.py 当前哈希；镜像ID 与 g1_CB_FS 一致'}
    write_once('source_identity.json',identity)
    for p in sorted(FIX.glob('*.py')):compile(p.read_text(),str(p),'exec')
    provenance={'frozen_inputs':frozen_hashes,
                'policy_plans_role':'仅事后对照Oracle：控制路径在 plan_strict 求解完成后才读入，不参与本轮选择',
                'environment':'C/K3 冻结环境（16预热×2轮+4Pending，K=3），Epoch 形成为冻结预热重放+实时观察，非全在线构建',
                'cold_reference_role':'正式请求输出的逐Token对照基准（g1_CB_FS 冻结 cold_reference_C_B.json）'}
    write_once('input_provenance.json',provenance)
    print('场景B门禁装配PASS：5个冻结输入已复制并记录哈希；flowstate 生产入口哈希已锁定（%d 个文件）'%len(identity['hybrid_entry_sources']))
if __name__=='__main__':main()
