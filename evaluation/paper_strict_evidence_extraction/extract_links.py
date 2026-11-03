"""Baseline 实现合同与 CPU→GPU 证据链提取（Task 3/4/5，只读）。

来源见 extract.py 头注；本脚本额外读取：
- b5d4b 10 臂（B5-C4 K=2 Exact/Greedy）+ 冷参照 + 描述统计；
- e45_unified_timing（E4-1B：w1/K2 Exact vs Greedy；E5-1：w1/K3 Exact vs 1-Swap）；
- c_formal_perf_gpu 主矩阵 m* 60 臂 + g1/g2 门禁臂 + finalize 汇总；
- paper_hybrid_gpu_gate_a/b（Spec 8）；
- e3 legacy_573_structure.csv（旧 573 条组织）。
输出：baseline_input_contract.csv、gpu_evidence_links.csv，并把新增检查追加进 checks.json。
"""
import csv, hashlib, json, statistics
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/paper_strict_evidence_extraction'
E3 = ROOT / 'evaluation/greedy_swap_structure_e3_step1'
B5D4B = ROOT / 'evaluation/exact_greedy_paired_gpu_gate_b5d4b'
E45 = ROOT / 'evaluation/e45_unified_timing'
E4B = ROOT / 'evaluation/native_resume_performance_e4_phase1b'
CF = ROOT / 'evaluation/c_formal_perf_gpu'
GA = ROOT / 'evaluation/paper_hybrid_gpu_gate_a'
GB = ROOT / 'evaluation/paper_hybrid_gpu_gate_b'
POL = {'FS': 'FlowState-Greedy', 'SA': 'StandaloneTopK', 'WFC': 'WFC'}
checks = json.loads((OUT / 'checks.json').read_text())['checks']

def check(name, ok, detail):
    checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

def jload(p):
    return json.loads(Path(p).read_text())

def jrows(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))

def put(name, rows, fields):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v)
                        for k, v in r.items()})

# ---------- Task 3：Baseline 实现合同（源自代码与既有审计，逐字段可溯源） ----------
contract = [
 {'method': 'Greedy(FlowState生产核心)',
  'implementation_source': 'flowstate/optimizer.py GlobalOptimizer._select_candidates；审计驱动=evaluation/p1_strong_baseline_validation/common.py score()（e3 audit.py:60 同法）',
  'selection_rule': '每轮选择使总恢复成本J下降最大的候选（边际=Σ Phi(T_p−E_p) 下降量），动态更新各Pending可执行前沿E_p，直至填满预算K或无正边际',
  'decision_inputs': '候选ID、token位置L、严格兼容Pending集（EPOCH_ADMISSIBLE边）、Pending目标T、预算K；不读访问时钟/工作流步数',
  'tie_break': '边际收益并列(1e-9)按 checkpoint_id 字典序（paper_solver_tradeoff_gate.json tolerance_tiebreak）',
  'frozen_params': 'EPS=1e-9；Phi=position_aware_quadratic_v1 冻结二次系数（37.828150g+0.345974143gt−0.156201917g²）',
  'workloads_compared': 'Strict20、C/K3 CPU、C/K3 GPU 60臂(FS)、B5-C4 K2 GPU(b5d4b Greedy臂)、w1/K2 GPU(E4-1B Greedy臂)',
  'strict20_status': '已运行（e3 140行之一，独立复算一致）', 'ck3_status': '已运行（cpu_results + GPU FS 20臂）',
  'selection_protocol': 'CPU离线审计 + 运行内求解（fix harness，Scheduler进程内）+ 生产plan_strict Greedy分支（Spec8 b1 GPU实测）',
  'adaptation_vs_original': 'FlowState 自身方法，无外部对应',
  'missing_metadata': ''},
 {'method': 'Exact(全子集枚举)',
  'implementation_source': '审计=e3 audit.py:61-63；生产同语义=flowstate/optimizer.py _enumerate_exact_subsets（plan_strict Exact分支）',
  'selection_rule': '枚举全部 |S|≤K 子集取最小J',
  'decision_inputs': '同Greedy（同一Epoch与评分）',
  'tie_break': '容差1e-9内取最少项数，再按ID字典序',
  'frozen_params': 'EPS=1e-9；门禁 n≤12∧|P|≤4∧K≤4∧子集数≤794',
  'workloads_compared': 'Strict20（20/20证书）、B5-C4 K2 GPU(b5d4b Exact臂)、w1/K2 GPU(E4-1B)、w1/K3 GPU(E5-1)',
  'strict20_status': '已运行：20/20 全子集枚举最优证书（e3+e5_step0+tradeoff三处一致）',
  'ck3_status': 'NOT_AVAILABLE：n=16>12 门禁拒绝，未绕过（paper_solver_tradeoff_gate.json）',
  'selection_protocol': 'CPU离线枚举（审计/运行内求解）+ 生产plan_strict Exact分支（Spec8 a1 GPU实测）',
  'adaptation_vs_original': '内部最优参照，非外部基线',
  'missing_metadata': 'C/K3 无独立Exact证书（不启动新枚举）'},
 {'method': 'Greedy+1-Swap(固定首改善)',
  'implementation_source': 'e3 audit.py:42-56（swap()）；ck3=paper_solver_tradeoff_bench.py',
  'selection_rule': '以Greedy解为起点；按(移出ID升序,移入ID升序)遍历1换1；接受首个使J严格下降>1e-9的交换；接受后重扫；完整扫描无改善即停',
  'decision_inputs': '同Greedy', 'tie_break': '固定字典序首改善（不接受等值交换）', 'frozen_params': 'EPS=1e-9',
  'workloads_compared': 'Strict20（18/20达Exact，w1/K3、w2/K3为2项局部陷阱需≥3项同步替换）、C/K3（同集同J=Greedy）、w1/K3 GPU(E5-1 Swap臂)',
  'strict20_status': '已运行（实验性对照，非生产策略）', 'ck3_status': '已运行（tradeoff ck3行）',
  'selection_protocol': 'CPU离线审计 + GPU冻结选集执行（E5-1）',
  'adaptation_vs_original': '实验对照方法，规则路径敏感（difference_cases.md）',
  'missing_metadata': ''},
 {'method': 'WFC',
  'implementation_source': 'evaluation/p1_strong_baseline_validation/wfc.py:8-24（P1 Section B3）',
  'selection_rule': '动态更新前沿；每轮选 (q 最大, r 最大, ID 最小)：q=前沿严格推进的Pending数，r=总Token推进；最优q=0即停',
  'decision_inputs': '候选ID/位置L/兼容集/T/K；不读Phi或任何恢复成本',
  'tie_break': '(-q,-r,checkpoint_id)', 'frozen_params': '无成本参数',
  'workloads_compared': 'Strict20（20/20与Greedy同集同前沿）、C/K3 CPU、C/K3 GPU(WFC 20臂)',
  'strict20_status': '已运行（e5_step0 420对分类：同集同前沿）', 'ck3_status': '已运行（J=471.874343021，选集含C_ROOT_P01024浅共享）',
  'selection_protocol': 'CPU离线 + GPU冻结选集执行（c_formal PILOT_POLICY读policy_plans.json）',
  'adaptation_vs_original': '内部定义的Workflow-aware强基线（P1合同），非外部系统复现；20/20持平不能外推为与Greedy一般等价（e5_step0 §3）',
  'missing_metadata': ''},
 {'method': 'StandaloneTopK',
  'implementation_source': 'e3 audit.py solvers()（静态独立收益）；C/K3=kvflow_multiworkflow_cpu_review/review_cpu.py',
  'selection_rule': '按各Pending独立收益之和静态降序取Top-K，仅保留正收益；不更新集合边际（重复覆盖收益不扣除）',
  'decision_inputs': '独立收益矩阵（Phi按单候选单独保留计算）、K', 'tie_break': '(-收益和,checkpoint_id)',
  'frozen_params': 'EPS=1e-9',
  'workloads_compared': 'Strict20、C/K3 CPU(复核)、C/K3 GPU(SA 20臂)',
  'strict20_status': '已运行：17/20 J==Greedy；w3/K3、w3/K4、w4/K4、B5-C4/K4 存在静态冗余（e5_step0）',
  'ck3_status': '已运行（J=348.234011266，选集与Marconi Adaptation同集；C_B无恢复源E=0）',
  'selection_protocol': 'CPU离线 + GPU冻结选集执行（c_formal SA臂）',
  'adaptation_vs_original': '内部基线：模拟"逐请求独立TopK"缓存价值聚合', 'missing_metadata': ''},
 {'method': 'PCB',
  'implementation_source': 'evaluation/p1_strong_baseline_validation/pcb.py（P1 Section B1）',
  'selection_rule': '每Pending推荐独立Phi收益最大候选；按(-收益,候选ID,请求ID)排序去重后取前K',
  'decision_inputs': '独立Phi收益、兼容集、K', 'tie_break': '(-score,checkpoint_id,pending_id)', 'frozen_params': '正式Phi',
  'workloads_compared': 'Strict20、旧573（历史）、E2/B5-C4原门禁',
  'strict20_status': '已运行：5/20达Exact（全部为K=4饱和条件）；K1–3显著劣于Greedy', 'ck3_status': '未运行（无C/K3记录）',
  'selection_protocol': 'CPU离线审计', 'adaptation_vs_original': '内部基线（P1合同）',
  'missing_metadata': ''},
 {'method': 'LDR',
  'implementation_source': 'evaluation/p1_strong_baseline_validation/ldr.py（P1 Section B2）',
  'selection_rule': '每Pending推荐位置最深候选；按(-位置,候选ID,请求ID)排序去重后取前K',
  'decision_inputs': '位置L、兼容集、K', 'tie_break': '(-位置,checkpoint_id,pending_id)', 'frozen_params': '无',
  'workloads_compared': 'Strict20（与PCB 20/20同集）', 'ck3_status': '未运行',
  'selection_protocol': 'CPU离线审计', 'adaptation_vs_original': '内部基线；非全候选按深度直接TopK（e5_step0 §3）',
  'missing_metadata': ''},
 {'method': 'LRU',
  'implementation_source': 'evaluation/kvflow_multiworkflow_cpu_main/run_cpu.py:66',
  'selection_rule': '按冻结合成访问序 last_access_order 降序（最近优先）取Top-K',
  'decision_inputs': 'last_access_order（合成声明）、K', 'tie_break': '(-last_access_order,checkpoint_id)',
  'frozen_params': '访问序来自 workloads.json synthetic_access_events 派生的冻结字段',
  'workloads_compared': 'C/K3 CPU（J=582.034711402，选集含2个JOIN未来状态）',
  'strict20_status': '未运行（Strict快照无访问序元数据）', 'ck3_status': '已运行（CPU）',
  'selection_protocol': 'CPU离线（合成输入）', 'adaptation_vs_original': '合成访问序近似，非真实时钟',
  'missing_metadata': 'GPU未测量；Strict20无输入字段'},
 {'method': 'Marconi Adaptation',
  'implementation_source': 'evaluation/sota_policies.py:96-140,216-258（MarconiStylePolicy）；CPU批调用=run_cpu.py:34',
  'selection_rule': 'utility=归一化recency + alpha×归一化(flop_saved/memory_bytes)，降序取Top-K（仅recurrent_resident候选）',
  'decision_inputs': 'last_access_by_checkpoint、flop_saved_by_checkpoint、memory_bytes、alpha',
  'tie_break': '(-utility,checkpoint_id)', 'frozen_params': 'CPU批：alpha=1.0，flop_saved代理=token_position_L，memory_bytes=1，recency=last_access_order',
  'workloads_compared': 'C/K3 CPU（J=348.234011266，与StandaloneTopK同集同J）',
  'strict20_status': 'UNVERIFIED：w1–w4/B5-C4缺last_access/flop_saved评分元数据，适配公平性未过门槛（e5_step0 §3；sota_baseline_gate_b5b0；strict_compatibility_b4c）',
  'ck3_status': '仅CPU合成输入运行，非公平复现',
  'selection_protocol': 'CPU离线（代理输入）', 
  'adaptation_vs_original': '与Marconi原系统差异：非原生访问时钟/真实FLOPs/树父节点跨度/资源分母/独立Bootstrap（b5a审计+b5b0门禁在案）',
  'missing_metadata': '原生last_access_order、marconi_flop_saved、原生驱逐资格、独立Bootstrap锁'},
 {'method': 'KVFlow-inspired',
  'implementation_source': 'evaluation/kvflow_inspired_retention_cpu/policy.py:18-92',
  'selection_rule': '按图STE（关联未来调用的最小steps-to-execution）升序取Top-K；固定前缀祖先priority=0优先保留',
  'decision_inputs': '调用图、declarations、epoch_id、configuration_digest、K', 'tie_break': '(priority,checkpoint_id)升序',
  'frozen_params': 'STE=关联调用最小步数',
  'workloads_compared': 'C/K3 CPU（J=627.6149768490118，三种结构均运行）',
  'strict20_status': '未运行（Strict快照无调用图声明）', 'ck3_status': '已运行（CPU）',
  'selection_protocol': 'CPU离线（合成输入）', 'adaptation_vs_original': '步数优先保留抽象，非官方KVFlow调度器复现',
  'missing_metadata': 'GPU未测量；Strict20无输入字段'},
 {'method': 'plan_strict(生产混合入口)',
  'implementation_source': 'flowstate/optimizer.py GlobalOptimizer.plan_strict（Spec7集成，Spec8 GPU验证）',
  'selection_rule': '_exact_admission门禁(n≤12∧|P|≤4∧K≤4∧子集≤794)：门内走Exact枚举，超域走未改动的Greedy核心；异常一律上抛无回退',
  'decision_inputs': '同Greedy/Exact（同一Epoch）', 'tie_break': '两分支各自继承',
  'frozen_params': '门禁阈值与容差冻结；last_strict_decision记录solver/gate/选集/J/纯求解时间',
  'workloads_compared': 'Strict20（CPU集成门禁20/20）、C/K3（CPU回退==冻结FS）、B5-C4 K2 GPU(a1在线Exact)、C/K3 GPU(b1在线Greedy)',
  'strict20_status': '已运行（paper_hybrid_solver_integration_gate：门内自动Exact）',
  'ck3_status': '已运行（n=16>12自动Greedy==冻结FS；b1 GPU在线实测）',
  'selection_protocol': '生产在线：Scheduler进程内实际调用（strict_decision在案）',
  'adaptation_vs_original': 'FlowState 自身生产入口', 'missing_metadata': ''},
]
put('baseline_input_contract.csv', contract,
    ['method', 'implementation_source', 'selection_rule', 'decision_inputs', 'tie_break', 'frozen_params',
     'workloads_compared', 'strict20_status', 'ck3_status', 'selection_protocol', 'adaptation_vs_original',
     'missing_metadata'])

# ---------- Task 4：GPU 证据链 ----------
links = []
def link(**kw):
    row = {'experiment': '', 'condition': '', 'policy_cpu': '', 'pending': '', 'cpu_selection': '', 'cpu_J_ms': '',
           'mapping_evidence': '', 'execution_mode': '', 'gpu_arms': '', 'n_lifecycles': '', 'pending_coverage': '',
           'native_E': '', 'native_H': '', 'native_G': '', 'prefill_tokens': '', 'cuda_prefill_ms_median': '',
           'ttft_ms_median': '', 'resume_vs_cold': '', 'stat_method': '', 'artifacts': '', 'notes': ''}
    row.update(kw); links.append(row)

e3rows = jrows(E3 / 'condition_results.csv')
e3idx = {(r['case'], int(r['K']), r['policy']): r for r in e3rows}

# --- B5-C4 K=2：b5d4b 10 臂 ---
armJ = {}
for arm in [f'p0{i}_{s}' for i in range(1, 6) for s in ('Exact', 'Greedy')]:
    rec = jload(B5D4B / arm / 'reconciliation.json'); summ = jload(B5D4B / arm / 'summary.json')
    armJ[arm] = (rec['plan']['selected_ids'], rec['plan']['recovery_cost_ms'], summ['ttft_ms'],
                 summ['native_match']['executable_prefix'], summ['native_match']['physical_fa_hit'],
                 summ['native_match']['replay_gap'], summ['output_ids'])
ok_exact = all(set(v[0]) == {'B5C3_NODE_18', 'B5C3_NODE_27'} and abs(v[1] - 48.228728759089606) <= 1e-9
               for k, v in armJ.items() if k.endswith('Exact'))
ok_greedy = all(set(v[0]) == {'B5C3_NODE_20', 'B5C3_NODE_18'} and abs(v[1] - 130.8030838298517) <= 1e-9
                for k, v in armJ.items() if k.endswith('Greedy'))
check('b5d4b 10臂 == B5-C4 K2 冻结Exact/Greedy选集与J', ok_exact and ok_greedy,
      {'Exact臂': ok_exact, 'Greedy臂': ok_greedy})
cold_ids = jload(B5D4B / 'cold' / 'output.json')['output_ids']
ok_cold = all(v[6] == cold_ids for v in armJ.values())
coldchk = jload(B5D4B / 'cold_reference_correctness.json')
ok_coldchk = all(c['status'] == 'PASS' and c['exact_token_match'] for c in coldchk['comparisons'].values())
check('b5d4b 10臂输出==冷参照逐Token', ok_cold and ok_coldchk,
      {'cold_output_ids': cold_ids, 'per_arm_comparisons': len(coldchk['comparisons']),
       'all_exact_token_match': ok_coldchk})
ds = jload(B5D4B / 'descriptive_statistics.json')
dsidx = {(d['solver'], d['metric']): d for d in ds}
ex_e = next(v for k, v in armJ.items() if k.endswith('Exact')); gr_e = next(v for k, v in armJ.items() if k.endswith('Greedy'))
for pol, fr, nm, ex in (('Exact', armJ['p01_Exact'], 'Exact', ex_e), ('Greedy', armJ['p01_Greedy'], 'Greedy', gr_e)):
    r = e3idx[('B5-C4', 2, pol)]
    link(experiment='E4配对门禁(b5d4b)', condition='B5-C4/K2(n=7)', policy_cpu=pol, pending='B5C2_C_TURN_003(目标)',
         cpu_selection=json.loads(r['selected']), cpu_J_ms=float(r['J_ms']),
         mapping_evidence='同ID命名空间：CPU与GPU均为B5C3_NODE_*（B5C3快照内直接绑定）',
         execution_mode='运行内求解：Scheduler进程内对冻结历史重放+实时观察的Epoch求解（solver由B5D4_SOLVER指定）+运行内Decimal Oracle复核',
         gpu_arms=f'p01–p05_{pol}', n_lifecycles='5臂+1 Cold(10臂全过)', pending_coverage='目标Pending=C（其余3个Pending未单独测）',
         native_E=ex[3], native_H=ex[4], native_G=ex[5],
         prefill_tokens=ex_e[2] and (jload(B5D4B / f'p01_{pol}' / 'summary.json')['native_match']['input_token_count'] - ex[3]),
         cuda_prefill_ms_median=dsidx.get((pol, 'prefill_cuda_event_ms'), {}).get('median'),
         ttft_ms_median=dsidx.get((pol, 'ttft_client_ms'), {}).get('median'),
         resume_vs_cold='10/10臂8Token==Cold参照（cold_reference_correctness PASS）',
         stat_method='descriptive_statistics.json（n=5，中位）',
         artifacts='evaluation/exact_greedy_paired_gpu_gate_b5d4b/{p01–p05_*}/ {{summary,reconciliation,output}}.json + cold/ + descriptive_statistics.json',
         notes='计时含研究仪表，非生产口径（measurement_validity.md）')

# --- Spec 8：a1 / b1 ---
ga = jload(GA / 'gate_a_result.json'); gb = jload(GB / 'gate_b_result.json')
link(experiment='Spec8 Hybrid Solver GPU闭环(a1)', condition='B5-C4/K2(n=7)', policy_cpu='plan_strict→Exact(在线)',
     pending='B5C2_C_TURN_003(目标)', cpu_selection=ga['selected'], cpu_J_ms=ga['J_ms'],
     mapping_evidence='同ID命名空间B5C3_NODE_*；Native绑定Node27/slots[18]==计划绑定',
     execution_mode='生产plan_strict在线（Scheduler进程内，last_strict_decision在案；pure_solve 0.222861ms）',
     gpu_arms='a1_hybrid_exact', n_lifecycles='1(最小正确性验证,非性能实验)', pending_coverage='目标Pending=C',
     native_E=3520, native_H=3904, native_G=384, prefill_tokens=915,
     cuda_prefill_ms_median='', ttft_ms_median='',
     resume_vs_cold='8Token==b5d4b冻结Cold参照（10臂验证过的ids）',
     stat_method='单次正确性验证；计时字段仅执行记录',
     artifacts='evaluation/paper_hybrid_gpu_gate_a/{a1_hybrid_exact,gate_a_result}.json',
     notes='==冻结Exact==运行内Decimal Oracle==e3；与旧Greedy选集差异N27')
cf_plans = jload(CF / 'policy_plans.json')
link(experiment='Spec8 Hybrid Solver GPU闭环(b1)', condition='C/K3(n=16)', policy_cpu='plan_strict→Marginal Greedy(在线)',
     pending='C_B(目标)', cpu_selection=sorted(gb['decision']['selected_ids']), cpu_J_ms=gb['J_ms'],
     mapping_evidence='census 16=16：CPU C_*↔物理B5C3_NODE_*（reconciliation.census.mapped）',
     execution_mode='生产plan_strict在线（门禁16>12拒→自动Greedy；policy_plans仅事后对照；pure_solve 0.402763ms）',
     gpu_arms='b1_hybrid_greedy', n_lifecycles='1(最小正确性验证)', pending_coverage='目标Pending=C_B',
     native_E=5120, native_H=5120, native_G=0, prefill_tokens=32,
     cuda_prefill_ms_median='', ttft_ms_median='',
     resume_vs_cold='8Token==g1_CB_FS冻结Cold参照C_B',
     stat_method='单次正确性验证',
     artifacts='evaluation/paper_hybrid_gpu_gate_b/{b1_hybrid_greedy,gate_b_result}.json',
     notes='选集==g1冻结GPU计划==policy_plans(经census映射)==cpu_results；J=154.34895561599998')

# --- E4-1B / E5-1：e45 统一计时 ---
e45rows = jrows(E45 / 'unified_timing_rows.csv')
e45stats = jrows(E45 / 'unified_timing_stats.csv')
e45ver = jload(E45 / 'unified_timing_verification.json')
check('e45 统一计时 54/54 生命周期 PASS（E4-1B 18 + E5-1 36）',
      e45ver['pass_rows'] == e45ver['lifecycle_rows'] == 54, {k: e45ver[k] for k in ('gate', 'lifecycle_rows', 'pass_rows')})
fp4 = jload(E4B / 'frozen_plans.json'); fp5 = jload(E5S1FZ := (ROOT / 'evaluation/exact_swap_runtime_e5_step1' / 'frozen_plans.json'))
check('E4-1B 冻结计划==e3 w1/K2 Exact/Greedy',
      set(fp4['Exact']['selected_ids']) == set(json.loads(e3idx[('w1', 2, 'Exact')]['selected'])) and
      set(fp4['Greedy']['selected_ids']) == set(json.loads(e3idx[('w1', 2, 'Greedy')]['selected'])),
      {'E4-1B Exact': fp4['Exact']['selected_ids'], 'Greedy': fp4['Greedy']['selected_ids']})
check('E5-1 冻结计划==e3 w1/K3 Exact/1-Swap',
      set(fp5['Exact']['selected_ids']) == set(json.loads(e3idx[('w1', 3, 'Exact')]['selected'])) and
      set(fp5['Swap']['selected_ids']) == set(json.loads(e3idx[('w1', 3, 'Greedy+1-Swap')]['selected'])),
      {'E5-1 Exact': fp5['Exact']['selected_ids'], 'Swap': fp5['Swap']['selected_ids']})
# 每臂 E/H/G 跨重复一致性
byk = {}
for r in e45rows:
    byk.setdefault((r['experiment'], r['pending'], r['solver']), []).append(
        (r['H_physical_fa_hit'], r['E_executable_prefix'], r['G_replay_gap'], r['actual_prefill_tokens']))
bad = [k for k, v in byk.items() if len({(x[0], x[1], x[2], x[3]) for x in v}) != 1]
check('E4-1B/E5-1 各臂 E/H/G/Prefill 跨重复一致', not bad, {'不一致': bad})
STAT = {(r['experiment'], r['pending'], r['solver'], r['metric']): r for r in e45stats}
COND = {'E4-1B': 'w1/K2', 'E5-1': 'w1/K3'}
SOLVER = {'Exact': 'Exact', 'Greedy': 'Greedy', 'Swap': 'Greedy+1-Swap'}
for (exp, pend, solver), vals in sorted(byk.items()):
    if solver == 'Cold': continue
    pol = SOLVER[solver]; cond = COND[exp]
    r = e3idx[('w1', int(cond.split('/')[1][1:]), pol)]
    tt = STAT.get((exp, pend, solver, 'request_ttft_ms'), {})
    cp = STAT.get((exp, pend, solver, 'cuda_prefill_ms'), {})
    link(experiment=f'{exp}({cond})', condition=f'{cond}(n=7)', policy_cpu=pol, pending=pend,
         cpu_selection=json.loads(r['selected']), cpu_J_ms=float(r['J_ms']),
         mapping_evidence='同ID命名空间B5C3_NODE_*（w1快照内）',
         execution_mode='冻结选集执行（frozen_plans.json；无运行内求解）',
         gpu_arms=f'r1–r3_{pend.split("_")[0]}_{solver}', n_lifecycles='3正式+smoke+独立Cold',
         pending_coverage=f'{pend}（{"A,C" if exp == "E4-1B" else "A,B,C,D"}中之一）',
         native_E=int(vals[0][1]), native_H=int(vals[0][0]), native_G=int(vals[0][2]), prefill_tokens=int(vals[0][3]),
         cuda_prefill_ms_median=cp.get('median'), ttft_ms_median=tt.get('median'),
         resume_vs_cold='8Token==独立Cold（validated_resumes/validation.json；e45 54/54 PASS）',
         stat_method='e45 unified_timing_stats（n=3，中位）',
         artifacts=f'evaluation/{ "native_resume_performance_e4_phase1b" if exp == "E4-1B" else "exact_swap_runtime_e5_step1" } + evaluation/e45_unified_timing/',
         notes='每Pending独立恢复生命周期；E5-1三组Exact−Swap总和方向不一致（−13.4/+1.9/−3.1ms）')

# --- C/K3：c_formal m* 60 臂主矩阵 ---
fin = jload(CF / 'finalize_results.json')
armsm = jload(CF / 'arm_status_main.json')
check('c_formal 生命周期=62（主矩阵60 m* + 门禁2 g*），62/62 PASS',
      fin['lifecycles_total'] == 62 and fin['lifecycles_passed'] == 62 and len(armsm) == 60,
      {'lifecycles_total': fin['lifecycles_total'], 'main_arms': len(armsm)})
agg = {}
census_ok = True; cold_ok = True
for rep in (1, 2, 3, 4, 5):
    for pl in 'ABCD':
        for pn, pol in POL.items():
            s = jload(CF / f'm{rep}{pl}_{pn}' / 'run' / 'summary.json')
            rec = jload(CF / f'm{rep}{pl}_{pn}' / 'run' / 'reconciliation.json')
            m = {x['cpu_checkpoint_id']: x['physical_checkpoint_id'] for x in rec['census']['mapped']}
            want = {m[c] for c in cf_plans[pol]['selected_checkpoint_ids']}
            if set(rec['plan']['selected_ids']) != want or len(m) != 16:
                census_ok = False
            if not s['cold_output_match']:
                cold_ok = False
            agg.setdefault((pol, f'C_{pl}'), []).append(
                (s['ttft_ms'], s['cuda_prefill_ms'], s['E_executable_prefix'], s['H_physical_fa_hit'],
                 s['G_replay_gap'], s['prefill_token_count']))
check('c_formal 60臂 census映射选集==policy_plans(每臂16=16)', census_ok, {})
check('c_formal 60臂 cold_output_match 全True', cold_ok, {})
eg_expect = {p: cf_plans[p]['expected_E_G'] for p in cf_plans}
eg_ok = all(int(v[2]) == eg_expect[pol][pend]['E'] and int(v[4]) == eg_expect[pol][pend]['G']
            for (pol, pend), vs in agg.items() for v in vs)
check('c_formal 60臂 Native E/G==policy_plans expected_E_G 逐臂', eg_ok, {})
PDS = fin['pair_delta_stats']
for (pol, pend), vs in sorted(agg.items()):
    link(experiment='C/K3 正式GPU主矩阵(c_formal m*)', condition='C/K3(n=16)', policy_cpu=pol, pending=pend,
         cpu_selection=sorted(cf_plans[pol]['selected_checkpoint_ids']),
         cpu_J_ms=(154.34895561599998 if pol == 'FlowState-Greedy' else
                   {'StandaloneTopK': 348.234011266, 'WFC': 471.874343021}[pol]),
         mapping_evidence='census 16=16（每臂reconciliation.census，CPU C_*↔物理B5C3_NODE_*）',
         execution_mode='冻结选集执行（PILOT_POLICY读policy_plans.json；无运行内求解）',
         gpu_arms=f'm1–m5{pend[-1]}_{ {v:k for k,v in POL.items()}[pol] }', n_lifecycles='5(主矩阵60臂；门禁g1/g2另计不混入)',
         pending_coverage=pend,
         native_E=int(statistics.median([v[2] for v in vs])), native_H=int(statistics.median([v[3] for v in vs])),
         native_G=int(statistics.median([v[4] for v in vs])), prefill_tokens=int(statistics.median([v[5] for v in vs])),
         cuda_prefill_ms_median=round(statistics.median([v[1] for v in vs]), 3),
         ttft_ms_median=round(statistics.median([v[0] for v in vs]), 3),
         resume_vs_cold='每臂8Token==该Pending冻结Cold参照（cold_reference_C_*.json）',
         stat_method='60臂run/summary.json 中位数（n=5；正式统计仅m*主矩阵，协议v1.2）',
         artifacts=f'evaluation/c_formal_perf_gpu/m1–m5{ {v:k for k,v in POL.items()}[pol] and pend[-1]}_{ {v:k for k,v in POL.items()}[pol] }/run/ + finalize_results.json',
         notes=(('FS−SA配对TTFT中位差 ' + str(PDS.get(f'{pend}|FS-SA', {}).get('ttft_ms', {}).get('median')) + 'ms；FS−WFC ' +
                str(PDS.get(f'{pend}|FS-WFC', {}).get('ttft_ms', {}).get('median')) + 'ms') if pol == 'FlowState-Greedy' else
               (('SA−FS配对TTFT中位差 ' + str(PDS.get(f'{pend}|FS-SA', {}).get('ttft_ms', {}).get('median')) + 'ms（负=FS更慢）') if pol == 'StandaloneTopK' else
                ('WFC−FS配对TTFT中位差 ' + str(PDS.get(f'{pend}|FS-WFC', {}).get('ttft_ms', {}).get('median')) + 'ms（负=FS更慢）'))))
for g in ('g1_CB_FS', 'g1_CB_SA', 'g2_CB_FS', 'g2_CB_SA'):
    d = CF / g
    if not (d / 'run' / 'summary.json').exists():
        continue
    s = jload(d / 'run' / 'summary.json')
    pol = POL[g.split('_')[-1]]
    link(experiment='C/K3 门禁臂(不混入主统计)', condition='C/K3(n=16)', policy_cpu=pol, pending='C_B',
         cpu_selection=sorted(cf_plans[pol]['selected_checkpoint_ids']),
         cpu_J_ms={'FlowState-Greedy': 154.34895561599998, 'StandaloneTopK': 348.234011266, 'WFC': 471.874343021}[pol],
         mapping_evidence='census 16=16', execution_mode='冻结选集执行（门禁复核算径）',
         gpu_arms=g, n_lifecycles='1(门禁)', pending_coverage='C_B',
         native_E=s['E_executable_prefix'], native_H=s['H_physical_fa_hit'], native_G=s['G_replay_gap'],
         prefill_tokens=s['prefill_token_count'], cuda_prefill_ms_median='', ttft_ms_median=round(s['ttft_ms'], 3),
         resume_vs_cold=str(s['cold_output_match']),
         stat_method='门禁臂，正式统计不含（finalize按gate_arms+main_arms分列）',
         artifacts=f'evaluation/c_formal_perf_gpu/{g}/run/', notes='协议v1.1→v1.2门禁复核臂')

# --- NOT_MEASURED 汇总行（Strict20 无GPU证据的选集 + C/K3 其余策略） ---
measured = {('B5-C4', 2, 'Exact'), ('B5-C4', 2, 'Greedy'), ('w1', 2, 'Exact'), ('w1', 2, 'Greedy'),
            ('w1', 3, 'Exact'), ('w1', 3, 'Greedy+1-Swap')}
for case in ('w1', 'w2', 'w3', 'w4', 'B5-C4'):
    for k in (1, 2, 3, 4):
        for pol in ('Exact', 'Greedy', 'Greedy+1-Swap', 'WFC', 'StandaloneTopK', 'PCB', 'LDR'):
            if (case, k, pol) in measured: continue
            r = e3idx[(case, k, pol)]
            link(experiment='—', condition=f'{case}/K{k}', policy_cpu=pol, pending='ALL',
                 cpu_selection=json.loads(r['selected']), cpu_J_ms=float(r['J_ms']),
                 mapping_evidence='', execution_mode='', gpu_arms='', n_lifecycles='0',
                 pending_coverage='', native_E='', native_H='', native_G='', prefill_tokens='',
                 cuda_prefill_ms_median='', ttft_ms_median='', resume_vs_cold='', stat_method='',
                 artifacts='', notes='NOT_MEASURED（不根据模型化J推测实际TTFT）')
ck3_extra = {'LRU': 582.034711402, 'Marconi Adaptation': 348.234011266, 'KVFlow-inspired': 627.6149768490118,
             'Greedy+1-Swap': 154.34895561599998}
ck3_selmap = {'LRU': ['C_D_P07168', 'C_JOIN_P04096', 'C_JOIN_P04032'],
              'Marconi Adaptation': ['C_C_P06144', 'C_D_P07104', 'C_D_P07168'],
              'KVFlow-inspired': ['C_A_P04032', 'C_AB_P02048', 'C_AB_P01984'],
              'Greedy+1-Swap': ['C_B_P05120', 'C_C_P06144', 'C_D_P07168']}
for pol, j in ck3_extra.items():
    link(experiment='—', condition='C/K3', policy_cpu=pol, pending='ALL', cpu_selection=ck3_selmap[pol],
         cpu_J_ms=j, mapping_evidence='', execution_mode='', gpu_arms='', n_lifecycles='0', pending_coverage='',
         native_E='', native_H='', native_G='', prefill_tokens='', cuda_prefill_ms_median='', ttft_ms_median='',
         resume_vs_cold='', stat_method='', artifacts='', notes='NOT_MEASURED')
link(experiment='—', condition='C/K3', policy_cpu='PCB/LDR', pending='ALL', cpu_selection=[], cpu_J_ms='',
     mapping_evidence='', execution_mode='', gpu_arms='', n_lifecycles='0', pending_coverage='', native_E='',
     native_H='', native_G='', prefill_tokens='', cuda_prefill_ms_median='', ttft_ms_median='', resume_vs_cold='',
     stat_method='', artifacts='', notes='NOT_RUN：C/K3 无PCB/LDR运行记录')
link(experiment='—', condition='C/K3', policy_cpu='Exact', pending='ALL', cpu_selection=[], cpu_J_ms='',
     mapping_evidence='', execution_mode='', gpu_arms='', n_lifecycles='0', pending_coverage='', native_E='',
     native_H='', native_G='', prefill_tokens='', cuda_prefill_ms_median='', ttft_ms_median='', resume_vs_cold='',
     stat_method='', artifacts='', notes='NOT_AVAILABLE：无C/K3独立Exact证书（门禁拒绝n=16>12）')

put('gpu_evidence_links.csv', links,
    ['experiment', 'condition', 'policy_cpu', 'pending', 'cpu_selection', 'cpu_J_ms', 'mapping_evidence',
     'execution_mode', 'gpu_arms', 'n_lifecycles', 'pending_coverage', 'native_E', 'native_H', 'native_G',
     'prefill_tokens', 'cuda_prefill_ms_median', 'ttft_ms_median', 'resume_vs_cold', 'stat_method',
     'artifacts', 'notes'])

# --- gpu_ck3.csv n 列观测（归因审计产物是否存在门禁臂混入） ---
gck = jrows(ROOT / 'evaluation/paper_allocation_structure_gpu_ck3.csv')
nobs = {(r['policy'], r['pending']): r['n'] for r in gck}
check('观测：paper_allocation_structure_gpu_ck3.csv 的 n 列（C_B=6 含门禁臂，其余=5）',
      all(int(v) == (6 if pd == 'C_B' and pol in ('FlowState-Greedy', 'StandaloneTopK') else 5) for (pol, pd), v in nobs.items()),
      {'n': {f'{a}|{b}': v for (a, b), v in nobs.items()}, 'note': '该CSV为归因审计产物：FS/SA的C_B行n=6含g1/g2门禁臂（WFC无门禁臂n=5）；正式统计须用m*主矩阵（本表已用60臂单独重算）'})

# --- 旧 573 条组织（只读复核，不重跑策略） ---
lg = jrows(E3 / 'legacy_573_structure.csv')
from collections import Counter
cnt = Counter(r['dataset'] for r in lg)
certs = Counter(r['optimality_certificate'] for r in lg)
strict_rt = Counter(r['strict_runtime_certification'] for r in lg)
check('旧573=OpenHands504+AgentX69；证书类型与Strict认证状态',
      cnt == Counter({'OpenHands': 504, 'AgentX': 69}),
      {'conditions': dict(cnt), 'certificates': dict(certs), 'strict_runtime_certified(0=未认证)': dict(strict_rt),
       'greedy_optimal_in_legacy_model': dict(Counter(r['greedy_optimal_in_legacy_model'] for r in lg))})

json.dump({'checks': checks}, (OUT / 'checks.json').open('w'), ensure_ascii=False, indent=2)
print('contract rows:', len(contract), 'links rows:', len(links))
for c in checks:
    print(c['status'], c['check'])
