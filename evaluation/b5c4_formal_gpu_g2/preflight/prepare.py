"""G2 门禁装配（宿主机执行）：只读冻结 B5-C4 输入并锁定被测生产入口身份。

冻结链（全部只读复制，不修改冻结目录；重复执行拒绝覆盖）：
- prepared_inputs.json / cpu_gate.json ← b5d4b（先按其 execution_lock.json 哈希核验源文件未变；
  8 历史+4 Pending，K=2，B5C2_*×B5C3_* 结构）；
- g2_reference.json ← 由 e3 condition_results.csv 的 B5-C4/K=2 Exact 与 Greedy 行全精度抽取
  （选集/J/逐Pending E-T-G），仅作求解完成后的事后对照 Oracle，控制路径不读取；
- cold_reference_B5C2_C_TURN_003.json ← gate_a frozen_cold_reference.json（b5d4b cold 输出，
  10 严格臂逐 Token 一致验证）；A/B/D 的 Cold 参照由 preflight/cold/ 三臂按同协议补采后落位；
- source_identity.json：镜像ID + 镜像内 Runtime 源码哈希（沿用 b5d4b）+ hybrid_entry_sources
  （flowstate/*.py 当前哈希，容器内运行时核验）。
"""
import csv,hashlib,json
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
SRC=ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4b'
GATE_A=ROOT/'evaluation/paper_hybrid_gpu_gate_a'
E3=ROOT/'evaluation/greedy_swap_structure_e3_step1/condition_results.csv'
OUT=ROOT/'evaluation/b5c4_formal_gpu_g2/preflight'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_once(name,data):
    p=OUT/name
    assert not p.exists(),'拒绝覆盖既有文件：'+name
    p.write_bytes(data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,indent=2).encode()+b'\n')
def main():
    # ── 冻结输入：按 b5d4b 执行锁逐哈希核验后复制 ──
    lock=json.loads((SRC/'execution_lock.json').read_text())['files']
    for name in ('prepared_inputs.json','cpu_gate.json'):
        src=SRC/name;assert lock[str(src)]==sha(src),'b5d4b 冻结输入与其执行锁不一致：'+name
        write_once(name,src.read_bytes())
    # ── e3 全精度参照抽取（事后对照 Oracle；控制路径不读取）──
    rows={r['policy']:r for r in csv.DictReader(open(E3)) if r['case']=='B5-C4' and r['K']=='2'}
    ref={}
    for policy,solver in (('Exact','Exact'),('Greedy','Marginal Greedy')):
        r=rows[policy];per={x['continuation_id']:{'T':x['T'],'E':x['E'],'G':x['G']} for x in json.loads(r['per_pending'])}
        ref[solver]={'source':str(E3)+' case=B5-C4 K=2 policy='+policy,'selected':json.loads(r['selected']),
                     'J_ms':float(r['J_ms']),'per_pending':per}
    assert ref['Exact']['selected']==['B5C3_NODE_18','B5C3_NODE_27'] and ref['Marginal Greedy']['selected']==['B5C3_NODE_20','B5C3_NODE_18']
    assert abs(ref['Exact']['J_ms']-48.228728759089606)<1e-12 and abs(ref['Marginal Greedy']['J_ms']-130.8030838298517)<1e-12
    write_once('g2_reference.json',ref)
    # ── 冻结 Cold 参照（C）：gate_a 已复核 b5d4b cold 输出 ──
    cold=json.loads((GATE_A/'frozen_cold_reference.json').read_text())
    assert len(cold['output_ids'])==8
    write_once('cold_reference_B5C2_C_TURN_003.json',(GATE_A/'frozen_cold_reference.json').read_bytes())
    # ── 源身份：镜像内 Runtime 源码沿用 b5d4b 冻结哈希；生产入口按当前源码锁定 ──
    b5d4b_source=json.loads((SRC/'source_identity.json').read_text())
    identity={'image_id':b5d4b_source['image_id'],
              'runtime_sources':b5d4b_source['runtime_sources'],
              'hybrid_entry_sources':{str(p):sha(p) for p in sorted((ROOT/'flowstate').rglob('*.py'))},
              'note':'runtime_sources 为镜像内 sglang 源码（容器内运行时核验）；hybrid_entry_sources 为被测生产入口 flowstate/*.py 当前哈希；镜像ID 与 b5d4b/gate_a/G1 一致'}
    write_once('source_identity.json',identity)
    write_once('input_provenance.json',{
        'prepared_inputs':{'source':str(SRC/'prepared_inputs.json'),'sha256':sha(SRC/'prepared_inputs.json'),'verified_against':'b5d4b execution_lock.json'},
        'cpu_gate':{'source':str(SRC/'cpu_gate.json'),'sha256':sha(SRC/'cpu_gate.json'),'verified_against':'b5d4b execution_lock.json'},
        'g2_reference':{'source':str(E3),'role':'仅事后对照Oracle：控制路径在求解完成后才由runner读入，不参与本轮选择','Exact':{'selected':ref['Exact']['selected'],'J_ms':ref['Exact']['J_ms']},'Marginal Greedy':{'selected':ref['Marginal Greedy']['selected'],'J_ms':ref['Marginal Greedy']['J_ms']}},
        'cold_reference_C':{'source':str(GATE_A/'frozen_cold_reference.json'),'verified_against':'b5d4b cold_reference_correctness.json（10臂cold_output_ids 逐Token一致）'},
        'cold_reference_A_B_D':'待 preflight/cold/ 三臂按 b5d4b 同协议补采（全新Engine、空缓存断言、无控制、直接请求）后核验落位',
        'environment':'B5-C4 冻结环境（8历史+4Pending，K=2），Epoch 形成为冻结历史重放+实时观察，非全在线构建'})
    print('G2 门禁装配PASS：prepared_inputs/cpu_gate/g2_reference/Cold(C)已冻结；Exact J=%r Greedy J=%r'%(ref['Exact']['J_ms'],ref['Marginal Greedy']['J_ms']))
if __name__=='__main__':main()
