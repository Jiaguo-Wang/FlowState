"""G1 实验A（w1/K2 Exact）门禁装配（宿主机执行）：只读复制冻结输入并锁定被测生产入口身份。

冻结链（全部只读复制，不修改任何冻结目录；重复执行拒绝覆盖）：
- prepared_inputs.json ← E4-1A（native_resume_correctness_e4_phase1a_run01；w1 冻结 8 历史+4 Pending），
  复制前按其 execution_lock.json 的 sha256 台账核验源文件未变；
- cold_reference_IMW1_{A,B,C,D}_TURN_003.json ← E4-1A {X}_Cold/output.json（其 8 Token 已在该实验
  与 Greedy/Exact 恢复臂逐 Token 一致验证，见 validated_resumes.json / resume_correctness.json）；
- cpu_gate.json：镜像内 Runtime 源码哈希（沿用 E4-1A source_identity，运行时容器内核验）
  + 宿主 CPU 冒烟（E2/w1 快照 → 生产 plan_strict → 与 e3 w1/K2 Exact 冻结行对照）；
- source_identity.json：镜像ID + 镜像内 Runtime 源码哈希 + hybrid_entry_sources（flowstate/*.py 当前哈希）。
"""
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
E41A=ROOT/'evaluation/native_resume_correctness_e4_phase1a_run01'
E2W1=ROOT/'evaluation/independent_runtime_allocation_e2/w1'
E3=ROOT/'evaluation/greedy_swap_structure_e3_step1/condition_results.csv'
OUT=ROOT/'evaluation/hybrid_solver_gpu_gate_g1/exact'
PKG=ROOT/'evaluation/hybrid_solver_gpu_gate_g1/pkg_exact'
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_once(name,data):
    p=OUT/name
    assert not p.exists(),'拒绝覆盖既有文件：'+name
    p.write_bytes(data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,indent=2).encode()+b'\n')
def main():
    # ── 冻结输入核验与复制（prepared_inputs 与 E4-1A 执行锁逐哈希核验）──
    lock=json.loads((E41A/'execution_lock.json').read_text())['sha256']
    src_pi=E41A/'prepared_inputs.json'
    assert lock[str(src_pi)]==sha(src_pi),'E4-1A prepared_inputs 与其执行锁不一致'
    write_once('prepared_inputs.json',src_pi.read_bytes())
    # ── Cold 参照：4 Pending 各一份；复核与 E4-1A 冻结验证记录一致 ──
    vr={r['pending']:r for r in json.loads((E41A/'validated_resumes.json').read_text()) if r['status']=='PASS' and r['resume']=='RESUME_VALIDATED'}
    cold_ok={}
    for x in ('A','B','C','D'):
        pid=f'IMW1_{x}_TURN_003'
        cold=json.loads((E41A/f'{x}_Cold'/'output.json').read_text())
        assert cold['output_ids']==vr[pid]['output']['cold_output_ids'] and vr[pid]['output']['exact_token_match'],f'E4-1A Cold 与冻结恢复验证输出不一致：{pid}'
        cold_ok[pid]=cold['output_ids']
        write_once(f'cold_reference_{pid}.json',(E41A/f'{x}_Cold'/'output.json').read_bytes())
    # ── 宿主 CPU 冒烟：E2/w1 冻结快照 → 生产 plan_strict（应走 Exact）→ 对照 e3 冻结行 ──
    from evaluation.paper_hybrid_solver_integration_gate import build_epoch
    from flowstate.optimizer import GlobalOptimizer
    from flowstate.recovery_model import RecoveryCostModel
    epoch,evidence=build_epoch(E2W1,2)
    opt=GlobalOptimizer(RecoveryCostModel());plan=opt.plan_strict(epoch);d=dict(opt.last_strict_decision)
    erow=next(r for r in csv.DictReader(open(E3)) if r['case']=='w1' and r['K']=='2' and r['policy']=='Exact')
    assert d['solver']=='Exact' and d['gate_allowed'],'CPU冒烟未走Exact：'+repr(d)
    assert set(d['selected_ids'])==set(json.loads(erow['selected'])),'CPU冒烟选集与e3不一致'
    assert abs(d['objective_ms']-float(erow['J_ms']))<=1e-9,'CPU冒烟J与e3不一致'
    smoke={'status':'PASS','solver':d['solver'],'gate':d['gate'],'selected_ids':d['selected_ids'],
           'objective_ms':d['objective_ms'],'e3_reference':{'selected':erow['selected'],'J_ms':erow['J_ms']},
           'epoch_content_digest':epoch.content_digest}
    # ── 源身份与 CPU 门禁文件 ──
    e41a_source=json.loads((E41A/'source_identity.json').read_text())
    identity={'image_id':e41a_source['image_id'],'runtime_sources':e41a_source['runtime_sources'],
              'hybrid_entry_sources':{str(p):sha(p) for p in sorted((ROOT/'flowstate').rglob('*.py'))},
              'note':'runtime_sources 为镜像内 sglang 源码（容器内运行时核验）；hybrid_entry_sources 为被测生产入口 flowstate/*.py 当前哈希；镜像ID 与 E4-1A/b5d4b/c_formal 一致'}
    write_once('source_identity.json',identity)
    write_once('cpu_gate.json',{'runtime_sources':e41a_source['runtime_sources'],'cpu_smoke_w1_k2':smoke})
    for p in sorted(PKG.glob('*.py')):compile(p.read_text(),str(p),'exec')
    write_once('input_provenance.json',{
        'prepared_inputs':{'source':str(src_pi),'sha256':sha(src_pi),'verified_against':'E4-1A execution_lock.json'},
        'cold_references':{'source':str(E41A)+'/{A,B,C,D}_Cold/output.json','verified_against':'E4-1A validated_resumes.json（8臂 RESUME_VALIDATED 逐Token一致）'},
        'cpu_smoke':{'source':str(E2W1)+' 快照 + '+str(E3)+' w1/K2 Exact 行','result':smoke['status']},
        'environment':'E2/w1 冻结环境（8历史+4Pending，K=2），Epoch 形成为冻结历史重放+实时观察，非全在线构建',
        'pending_scope':'4臂：G1_PENDING=IMW1_A/B/C/D_TURN_003'})
    print('实验A门禁装配PASS：w1输入+4Cold参照已冻结复制；CPU冒烟',smoke['status'],'选集',d['selected_ids'],'J',d['objective_ms'])
if __name__=='__main__':main()
