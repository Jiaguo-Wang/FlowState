"""G1 实验B（C/K3 Marginal Greedy）门禁装配（宿主机执行）：只读复制冻结输入并锁定身份。

复用生产验证程序 evaluation/paper_hybrid_gpu_fix_b（零改动：PILOT_PENDING 已参数化、
Cold 参照按 cold_reference_{continuation_id}.json 命名）。冻结链（全部只读，拒绝覆盖）：
- prepared_inputs / cpu_candidate_prefix_mapping / cpu_gate / policy_plans /
  cold_reference_C_{A,B,C,D}.json ← c_formal_perf_gpu/g1_CB_FS（C/K3 正式臂冻结输入；
  policy_plans 本轮仅作求解完成后的事后对照 Oracle，控制路径不用于选择）；
- 语义预核验：g1_CB_FS 运行记录的 census 16=16、FS 物理选集、J=154.34895561599998；
  policy_plans FS 选集/E-G 合同 E=(0,5120,6144,7168)、G=(4096,0,0,0)；4 份 Cold 参照
  与 g1_CB_FS 运行输出及 m1{A,B,C,D}_FS 正式臂冻结输出逐 Token 一致；
- source_identity.json：镜像ID + 镜像内 Runtime 源码哈希（沿用 g1_CB_FS）+ hybrid_entry_sources
  （flowstate/*.py 当前哈希）。
"""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
SRC=ROOT/'evaluation/c_formal_perf_gpu/g1_CB_FS'
CF=ROOT/'evaluation/c_formal_perf_gpu'
OUT=ROOT/'evaluation/hybrid_solver_gpu_gate_g1/greedy'
FIX=ROOT/'evaluation/paper_hybrid_gpu_fix_b'
NAMES=('prepared_inputs.json','cpu_candidate_prefix_mapping.json','cpu_gate.json','policy_plans.json',
       'cold_reference_C_A.json','cold_reference_C_B.json','cold_reference_C_C.json','cold_reference_C_D.json')
FROZEN_FS_J_MS=154.34895561599998
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_once(name,data):
    p=OUT/name
    assert not p.exists(),'拒绝覆盖既有文件：'+name
    p.write_bytes(data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,indent=2).encode()+b'\n')
def main():
    # ── 冻结输入复制（含四 Pending 全套 Cold 参照）──
    frozen_hashes={}
    for name in NAMES:
        src=SRC/name;frozen_hashes[str(src)]=sha(src)
        write_once(name,src.read_bytes())
    # ── 语义预核验 1：g1_CB_FS 冻结运行记录（census/选集/J）──
    rec=json.loads((SRC/'run'/'reconciliation.json').read_text())
    assert len(rec['census']['mapped'])==16 and not rec['census'].get('unmapped'),'census非16=16'
    plans=json.loads((SRC/'policy_plans.json').read_text());fs=plans['FlowState-Greedy']
    cpu_to_phys={m['cpu_checkpoint_id']:m['physical_checkpoint_id'] for m in rec['census']['mapped']}
    assert {cpu_to_phys[c] for c in fs['selected_checkpoint_ids']}==set(rec['plan']['selected_ids']),'冻结FS选集与g1运行记录不一致'
    assert abs(rec['plan']['recovery_cost_ms']-FROZEN_FS_J_MS)<=1e-9,'冻结J与公开冻结值不一致'
    # ── 语义预核验 2：FS 选集/E-G 合同（E=(0,5120,6144,7168)，G=(4096,0,0,0)）──
    assert set(fs['selected_checkpoint_ids'])=={'C_D_P07168','C_C_P06144','C_B_P05120'},'FS选集非冻结三元集'
    expect=fs['expected_E_G']
    for x,e,g in zip('ABCD',(0,5120,6144,7168),(4096,0,0,0)):
        row=expect['C_'+x]
        assert (row['E'],row['G'])==(e,g),'E-G合同不符：C_'+x
    # ── 语义预核验 3：四 Cold 参照（g1_CB_FS 副本==c_formal 根原始；逐 Token==冻结正式臂输出）──
    cold_ok={}
    for x in 'ABCD':
        pid='C_'+x
        assert (SRC/f'cold_reference_{pid}.json').read_bytes()==(CF/f'cold_reference_{pid}.json').read_bytes(),'Cold副本与根原始不一致：'+pid
        cold=json.loads((SRC/f'cold_reference_{pid}.json').read_text())['output_ids']
        arm=json.loads((CF/f'm1{x}_FS'/'run'/'summary.json').read_text())
        assert arm['cold_output_match'] and arm['output_ids']==cold,'Cold参照与m1%s_FS冻结输出不一致'%x
        cold_ok[pid]=len(cold)
    assert json.loads((SRC/'cold_reference_C_B.json').read_text())['output_ids']==json.loads((SRC/'run'/'output.json').read_text())['output_ids'],'Cold参照与g1_CB_FS运行输出不一致'
    # ── 源身份：镜像内 Runtime 源码沿用 g1_CB_FS 冻结哈希；生产入口按当前源码锁定 ──
    g1_source=json.loads((SRC/'source_identity.json').read_text())
    identity={'image_id':g1_source['image_id'],
              'runtime_sources':g1_source['runtime_sources'],
              'hybrid_entry_sources':{str(p):sha(p) for p in sorted((ROOT/'flowstate').rglob('*.py'))},
              'note':'runtime_sources 为镜像内 sglang 源码（容器内运行时核验）；hybrid_entry_sources 为被测生产入口 flowstate/*.py 当前哈希；镜像ID 与 g1_CB_FS/c_formal 一致'}
    write_once('source_identity.json',identity)
    # ── 被复用程序语法检查（零改动合同：fix_b 源码未修改）──
    for p in sorted(FIX.glob('*.py')):compile(p.read_text(),str(p),'exec')
    write_once('input_provenance.json',{
        'frozen_inputs':frozen_hashes,
        'reused_program':{str(p):sha(p) for p in sorted(FIX.glob('*.py'))},
        'reused_program_note':'evaluation/paper_hybrid_gpu_fix_b 零改动复用：PILOT_PENDING 环境变量选择目标 Pending，Cold 参照按 cold_reference_{continuation_id}.json 命名',
        'policy_plans_role':'仅事后对照Oracle：控制路径在 plan_strict 求解完成后才读入，不参与本轮选择',
        'environment':'C/K3 冻结环境（16预热×2轮+4Pending，K=3），Epoch 形成为冻结预热重放+实时观察，非全在线构建',
        'cold_reference_validation':'4份Cold与m1{A,B,C,D}_FS正式臂冻结输出逐Token一致；C_B另与g1_CB_FS运行输出一致',
        'pending_scope':'4臂：PILOT_PENDING=C_A/C_B/C_C/C_D'})
    print('实验B门禁装配PASS：8个冻结输入（含4 Cold参照）已复制；FS合同E=(0,5120,6144,7168)/G=(4096,0,0,0)核验通过；fix_b零改动复用')
if __name__=='__main__':main()
