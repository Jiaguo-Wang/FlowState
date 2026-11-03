"""Solver 质量–延迟权衡审计：正确性门禁 + 纯 CPU 求解计时（只读冻结输入，不启动 GPU、不修改任何算法实现）。

求解器实现来源（全部为既有冻结实现，本脚本零改动，仅做包装计时）：
  1) Marginal Greedy  —— 生产入口 GlobalOptimizer._select_candidates（flowstate/optimizer.py:78-174），
     经 e3 审计闭包 greedy_swap_structure_e3_step1/audit.py:60 调用（候选等大小 size=1，预算=K 项）；
  2) Exact            —— e3 审计闭包 audit.py:61-63（B4-C 容差/最少项/字典序同分规则，与
     exact_planner_contract_gate_b5d1/planner.py choose_exact 同语义）；准入门禁调用正式
     planner.allowed（planner.py:121-122：n≤12、|P|≤4、K≤4、枚举子集数≤794），门禁失败一律留空；
  3) Greedy+1-Swap    —— e3 审计 swap()（audit.py:42-56）：初始集=Greedy 解，1-1 邻域，固定
     (removed,added) 字典序首改善，严格下降才接受（不跨等值平台），无改善即终止；
  4) WFC              —— p1_strong_baseline_validation/wfc.py select()（哈希锁实现）；
  5) Swap-only        —— 同 swap()，从 Greedy 解出发仅执行交换阶段，用于报告 1-Swap 增量成本。

冻结输入（全部只读并做哈希核对）：
  - Strict 20 条件：e3 audit.load()（w1–w4 来自 independent_runtime_allocation_e2/*/strict_epoch_snapshot.json，
    B5-C4 来自 multishare_allocation_gate_b5c4/results.json；e2/b5c4 evidence_manifest + b4c input_integrity
    由 audit.load() 自行验证）；交叉核对 e3 condition_results.csv（Greedy/Exact/Greedy+1-Swap/WFC 选集与 J）。
  - C/K3：kvflow_multiworkflow_cpu_main/workloads.json 结构 C 情景（16 物理候选），Epoch 构造逐字复刻
    run_cpu.py epoch(case,k,'current')；对 input_lock.json 子集哈希核对；交叉核对 cpu_results.json 的
    FlowState-Greedy/WFC 行（选集 + current_J_ms）。n=16>12 → Exact 门禁失败，该列留空，不绕过。

计时协议：perf_counter_ns；先热身（每条件每方法 5 轮不计入），随后每条件 120 轮、轮内按 repeat 轮转
方法次序交错执行；单次调用单独计时；每次输出与冻结选集断言一致。P95 仅为本地 CPU Microbenchmark
经验分位数，不是生产在线 P95；同一快照的四个 K 不视为独立工作负载（source/case 列保留快照标识）。
输入加载、Epoch 构造、收益矩阵构建另行单独计时（各 5 次取中位），与纯求解时间分开报告。
"""
import csv
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
E3 = EVAL / 'greedy_swap_structure_e3_step1'
MW = EVAL / 'kvflow_multiworkflow_cpu_main'
B5D1 = EVAL / 'exact_planner_contract_gate_b5d1'
EPS = 1e-9
WARMUP_ROUNDS = 5
TIMED_ROUNDS = 120            # ≥100 次独立计时调用
SETUP_REPS = 5                # 输入加载/Epoch构造/矩阵构建的重复次数（取中位）
QUALITY_METHODS = ('Greedy', 'Exact', 'Greedy+1-Swap', 'WFC')


def load_module(name, path):
    """按路径加载既有冻结模块（不改名导入，避免与同名模块冲突）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e3audit = load_module('e3_audit_module', E3 / 'audit.py')
planner = load_module('b5d1_planner_module', B5D1 / 'planner.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def out_csv(path, rows):
    """新文件一次性写出（x 模式，禁止覆盖既有产物）。"""
    with open(path, 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                        for k, v in r.items()})


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def pct95(xs):
    """线性插值经验分位数（与 statistics.quantiles(method='inclusive') 一致口径）。"""
    return statistics.quantiles(xs, n=100, method='inclusive')[95]


def allowed_gate(ep):
    """调用正式 planner.allowed（planner.py:121-122）做 Exact 准入判定；适配 common.Epoch 属性名。"""
    return planner.allowed(SimpleNamespace(candidates=ep.candidates, pendings=ep.pending, K=ep.k)), \
        {'n': len(ep.candidates), 'n_pending': len(ep.pending), 'K': ep.k,
         'subsets': sum(math.comb(len(ep.candidates), i)
                        for i in range(min(len(ep.candidates), ep.k) + 1))}


def gate_strict20():
    """Strict 20 条件：重建输入 → 复算四方法选集/J → 与 condition_results.csv 逐条核对。"""
    frozen = list(csv.DictReader(open(E3 / 'condition_results.csv')))
    ref = {(r['case'], int(r['K']), r['policy']): r for r in frozen}
    cases = e3audit.load()
    conditions = []
    gate_notes = []
    for name, base, _old in cases:
        for k in (1, 2, 3, 4):
            ep = replace(base, k=k)
            funcs = e3audit.solvers(ep)
            sel = {p: funcs[p]() for p in QUALITY_METHODS}
            ok, gate_info = allowed_gate(ep)
            assert ok, f'Exact 门禁在 n=7 应放行：{name} K={k}'
            ids = {c.identifier for c in ep.candidates}
            for p, s in sel.items():
                assert len(s) == len(set(s)) and len(s) <= ep.k and set(s) <= ids, (name, k, p, '预算或域违规')
            score = {p: e3audit.common.score(ep, s) for p, s in sel.items()}
            for p in QUALITY_METHODS:
                r = ref[(name, k, p)]
                assert set(sel[p]) == set(json.loads(r['selected'])), (name, k, p, '选集与冻结审计不一致')
                assert abs(score[p]['cost_ms'] - float(r['J_ms'])) <= EPS, (name, k, p, 'J 与冻结审计不一致')
            # 区分"选集不同但 J 相同"：对四方法两两登记
            for a in QUALITY_METHODS:
                for b in QUALITY_METHODS:
                    if a < b and abs(score[a]['cost_ms'] - score[b]['cost_ms']) <= EPS \
                            and set(sel[a]) != set(sel[b]):
                        gate_notes.append({'case': name, 'K': k, 'pair': f'{a}|{b}',
                                           'J_ms': score[a]['cost_ms'],
                                           'set_a': sorted(sel[a]), 'set_b': sorted(sel[b])})
            conditions.append({'case': name, 'K': k, 'ep': ep, 'funcs': funcs, 'sel': sel,
                               'score': score, 'gate': gate_info})
    assert len(conditions) == 20
    return conditions, gate_notes


def build_ck3_epoch(case, k):
    """逐字复刻 run_cpu.py epoch(case,k,'current')：ready 调用为 Pending，兼容集限定当前 Pending 域。"""
    calls = case['calls']
    order = case['predeclared_evaluation_order']
    qs = [next(p for p in calls if p['continuation_id'] == i) for i in order if
          next(p for p in calls if p['continuation_id'] == i)['ready']]
    ids = {q['continuation_id'] for q in qs}
    return e3audit.common.Epoch(
        tuple(e3audit.common.Candidate(c['checkpoint_id'], c['token_position_L'],
                                       tuple(x for x in c['token_compatible_calls'] if x in ids))
              for c in case['candidates']),
        tuple(e3audit.common.Pending(p['continuation_id'], p['planning_target_T']) for p in qs), k)


def gate_ck3():
    """C/K3：input_lock 子集哈希核对 → 复刻 run_cpu.epoch 构造 → 与 cpu_results.json 核对 Greedy/WFC。"""
    lock = json.loads((MW / 'input_lock.json').read_text())['sha256']
    used = [MW / 'workloads.json', EVAL / 'p1_strong_baseline_validation' / 'common.py',
            EVAL / 'p1_strong_baseline_validation' / 'wfc.py', ROOT / 'flowstate' / 'optimizer.py',
            ROOT / 'flowstate' / 'recovery_model.py']
    for p in used:
        assert lock.get(str(p)) == sha(p), f'冻结输入哈希变化：{p}'
    case = next(c for c in json.loads((MW / 'workloads.json').read_text())['cases'] if c['structure'] == 'C')
    ep = build_ck3_epoch(case, 3)
    assert len(ep.candidates) == 16 and len(ep.pending) == 4 and ep.k == 3
    ok, gate_info = allowed_gate(ep)
    assert not ok and gate_info['n'] == 16, 'C/K3 n=16 应被 Exact 门禁拒绝'
    funcs = e3audit.solvers(ep)
    sel = {'Greedy': funcs['Greedy'](), 'Greedy+1-Swap': funcs['Greedy+1-Swap'](), 'WFC': funcs['WFC']()}
    rows = json.loads((MW / 'cpu_results.json').read_text())['rows']
    ref = {r['policy']: r for r in rows if r['structure'] == 'C' and r['K'] == 3}
    score = {}
    for p, name in (('Greedy', 'FlowState-Greedy'), ('WFC', 'WFC')):
        s = sel[p]
        assert set(s) == set(ref[name]['selected']), ('C/K3', p, '选集与冻结 cpu_results 不一致')
        score[p] = e3audit.common.score(ep, s)
        assert abs(score[p]['cost_ms'] - ref[name]['current_J_ms']) <= EPS, ('C/K3', p, 'J 与冻结 cpu_results 不一致')
    score['Greedy+1-Swap'] = e3audit.common.score(ep, sel['Greedy+1-Swap'])  # 无冻结参照，仅报告相对 Greedy
    return {'case': 'C/K3', 'K': 3, 'ep': ep, 'funcs': funcs, 'sel': sel, 'score': score,
            'gate': gate_info, 'workload_case': case, 'frozen_rows': ref}


def time_setup(conditions, ck3):
    """单独计时输入加载 / Epoch 构造 / 收益矩阵构建（每条件 5 次取中位，与求解计时分离）。"""
    rows = []
    e2 = ROOT / 'evaluation/independent_runtime_allocation_e2'
    c4 = ROOT / 'evaluation/multishare_allocation_gate_b5c4'
    for cond in conditions:
        name, k, ep = cond['case'], cond['K'], cond['ep']
        raw_path = e2 / name / 'strict_epoch_snapshot.json' if name != 'B5-C4' else c4 / 'results.json'
        load_ts, build_ts, bm_ts = [], [], []
        for _ in range(SETUP_REPS):
            t0 = time.perf_counter_ns()
            data = json.loads(raw_path.read_text())
            load_ts.append((time.perf_counter_ns() - t0) / 1e6)
            t0 = time.perf_counter_ns()
            if name != 'B5-C4':
                rebuilt = e3audit.common.Epoch(
                    tuple(e3audit.common.Candidate(c['checkpoint_id'], c['token_position_L'],
                           tuple(e['continuation_id'] for e in data['edges'] if e['checkpoint_id'] == c['checkpoint_id']
                                 and e['status'] == 'EPOCH_ADMISSIBLE')) for c in data['candidates']),
                    tuple(e3audit.common.Pending(p['continuation_id'], p['planning_target_T']) for p in data['pendings']), k)
            else:
                z = data['input']
                rebuilt = e3audit.common.Epoch(
                    tuple(e3audit.common.Candidate(c['identifier'], c['position'], tuple(c['compatible']))
                          for c in z['candidates']),
                    tuple(e3audit.common.Pending(**p) for p in z['pending']), k)
            build_ts.append((time.perf_counter_ns() - t0) / 1e6)
            t0 = time.perf_counter_ns()
            e3audit.common.benefit_matrix(rebuilt)
            bm_ts.append((time.perf_counter_ns() - t0) / 1e6)
            assert rebuilt == ep, (name, k, '重建 Epoch 与门禁用 Epoch 不一致')
        rows.append({'source': 'strict20', 'case': name, 'K': k,
                     'setup_load_ms': statistics.median(load_ts),
                     'setup_epoch_ms': statistics.median(build_ts),
                     'setup_matrix_ms': statistics.median(bm_ts)})
    case = ck3['workload_case']
    load_ts, build_ts, bm_ts = [], [], []
    for _ in range(SETUP_REPS):
        t0 = time.perf_counter_ns()
        data = json.loads((MW / 'workloads.json').read_text())['cases']
        load_ts.append((time.perf_counter_ns() - t0) / 1e6)
        cse = next(c for c in data if c['structure'] == 'C')
        t0 = time.perf_counter_ns()
        _ = build_ck3_epoch(cse, 3)   # 与门禁同一构造路径，仅计时
        build_ts.append((time.perf_counter_ns() - t0) / 1e6)
        t0 = time.perf_counter_ns()
        e3audit.common.benefit_matrix(ck3['ep'])
        bm_ts.append((time.perf_counter_ns() - t0) / 1e6)
    rows.append({'source': 'ck3', 'case': 'C/K3', 'K': 3,
                 'setup_load_ms': statistics.median(load_ts),
                 'setup_epoch_ms': statistics.median(build_ts),
                 'setup_matrix_ms': statistics.median(bm_ts)})
    return rows


def bench_condition(source, case, k, ep, funcs, sel, exact_allowed):
    """单条件计时：热身后 120 轮轮转交错；Swap-only 单独计时（从冻结 Greedy 解出发）。"""
    methods = ['Greedy', 'Greedy+1-Swap', 'WFC', 'Swap-only'] + (['Exact'] if exact_allowed else [])
    # Swap-only 的输出 = 从 Greedy 解出发的交换终集，与 Greedy+1-Swap 选集相同（同一 swap 实现）
    expected = {m: tuple(sorted(sel['Greedy+1-Swap'] if m == 'Swap-only' else sel[m])) for m in methods}

    def run_once(m):
        if m == 'Swap-only':
            return e3audit.swap(ep, sel['Greedy'])[0]
        return funcs[m]()

    for rep in range(-WARMUP_ROUNDS, 0):     # 热身不计入
        for m in methods[(-rep) % len(methods):] + methods[:(-rep) % len(methods)]:
            assert tuple(sorted(run_once(m))) == expected[m]
    raw = []
    for rep in range(TIMED_ROUNDS):
        order = methods[rep % len(methods):] + methods[:rep % len(methods)]
        for oi, m in enumerate(order):
            t0 = time.perf_counter_ns()
            out = run_once(m)
            elapsed = time.perf_counter_ns() - t0
            assert tuple(sorted(out)) == expected[m], (case, k, m, '计时中输出变化')
            raw.append({'source': source, 'case': case, 'K': k, 'n': len(ep.candidates),
                        'n_pending': len(ep.pending), 'method': m, 'repeat': rep,
                        'order_index': oi, 'elapsed_ms': elapsed / 1e6})
    return raw


def main():
    t_start = time.perf_counter()
    conditions, gate_notes = gate_strict20()
    ck3 = gate_ck3()
    gate_report = {
        'strict20': {
            'conditions': len(conditions),
            'cross_checked': 'e3_step1/condition_results.csv：Greedy/Exact/Greedy+1-Swap/WFC 选集与 J 全部一致（EPS=1e-9）',
            'exact_gate': '全部 20 条件 n=7、|P|=4、K≤4、枚举子集 99 ≤ 794，planner.allowed 放行',
            'tolerance_tiebreak': {
                'EPS': EPS,
                'Greedy': '边际收益并列时按 checkpoint_id 字典序（optimizer.py tie-break）',
                'Exact': '容差 1e-9 内取最少项、再字典序（audit.py:63 与 planner.choose_exact 同语义）',
                'Greedy+1-Swap': '初始集=Greedy 解；(removed,added) 固定字典序首改善；仅严格下降接受（不跨等值平台）；无改善终止（audit.py:42-56）',
                'WFC': '排序键 (-q, -r, checkpoint_id)（wfc.py:14）',
            },
            'same_J_different_set': gate_notes,
        },
        'ck3': {
            'input_lock_verified': True,
            'candidates': 16, 'pending': 4, 'K': 3,
            'exact_gate': 'n=16 > 12 → planner.allowed 拒绝；Exact 列留空，未绕过门禁，不宣称任何最优性',
            'cross_checked': 'Greedy/WFC 选集与 current_J_ms 对 kvflow_multiworkflow_cpu_main/cpu_results.json C/K3 行一致',
            'one_swap_reference': 'C/K3 无冻结 1-Swap 参照（e3 未覆盖该输入），仅报告相对 Greedy 的 J 变化',
        },
        'sources': {
            'greedy': 'flowstate/optimizer.py GlobalOptimizer._select_candidates（经 e3 audit.py:60 闭包）',
            'exact': 'e3 audit.py:61-63（choose_exact 语义 = b5d1 planner.py:123-126）',
            'one_swap': 'e3 audit.py:42-56 swap()',
            'wfc': 'evaluation/p1_strong_baseline_validation/wfc.py select()',
            'allowed_gate': 'b5d1 planner.py:121-122 allowed()',
        },
    }
    write_json(EVAL / 'paper_solver_tradeoff_gate.json', gate_report)

    # 环境记录（本地 CPU Microbenchmark 口径；非生产在线计时）
    gate_report['environment'] = {
        'python': sys.version, 'platform': platform.platform(),
        'cpu': Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(),
        'clock': str(time.get_clock_info('perf_counter')),
        'note': '共享机器上的本地微基准，可能受后台负载干扰；P95 仅为微基准经验分位数',
    }
    write_json(EVAL / 'paper_solver_tradeoff_gate.json', gate_report)

    setup_rows = time_setup(conditions, ck3)
    raw = []
    for cond in conditions:
        raw += bench_condition('strict20', cond['case'], cond['K'], cond['ep'],
                               cond['funcs'], cond['sel'], True)
    raw += bench_condition('ck3', 'C/K3', 3, ck3['ep'], ck3['funcs'], ck3['sel'], False)
    out_csv(EVAL / 'paper_solver_tradeoff_raw.csv', raw)

    # ── 汇总：逐条件×方法 质量 + 计时；质量指标用门禁阶段冻结选集，不再重复求解 ──
    setup_by = {(r['source'], r['case'], r['K']): r for r in setup_rows}
    summary = []
    for source, conds in (('strict20', conditions), ('ck3', [ck3])):
        for cond in conds:
            case, k, ep = cond['case'], cond['K'], cond['ep']
            j0 = e3audit.common.score(ep, ())['cost_ms']
            has_exact = 'Exact' in cond['score']
            j_exact = cond['score']['Exact']['cost_ms'] if has_exact else None
            f_exact = j0 - j_exact if has_exact else None
            su = setup_by[(source, case, k)]
            timed_methods = ['Greedy', 'Greedy+1-Swap', 'WFC', 'Swap-only'] + (['Exact'] if has_exact else [])
            for m in timed_methods:
                ts = [r['elapsed_ms'] for r in raw
                      if r['source'] == source and r['case'] == case and r['K'] == k and r['method'] == m]
                # Swap-only 的质量字段与 Greedy+1-Swap 相同（同一交换终集），计时单独报告
                q_ref = 'Greedy+1-Swap' if m == 'Swap-only' else m
                sel_m = cond['sel'][q_ref]
                j_m = cond['score'][q_ref]['cost_ms'] if q_ref in cond['score'] else cond['score']['Greedy']['cost_ms']
                f_m = j0 - j_m
                row = {
                    'source': source, 'case': case, 'K': k, 'n': len(ep.candidates),
                    'n_pending': len(ep.pending), 'method': m,
                    'J_ms': j_m, 'J0_ms': j0, 'F_ms': f_m,
                    'gap_vs_exact_ms': (j_m - j_exact) if has_exact else '',
                    'benefit_ratio_vs_exact': (f_m / f_exact) if (has_exact and f_exact > EPS) else '',
                    'is_exact_optimal': (has_exact and abs(j_m - j_exact) <= EPS),
                    'same_set_as_greedy': set(sel_m) == set(cond['sel']['Greedy']),
                    'selected': sorted(sel_m),
                    'median_ms': statistics.median(ts), 'p95_ms': pct95(ts),
                    'min_ms': min(ts), 'max_ms': max(ts), 'iqr_ms': (statistics.quantiles(ts, n=4)[2] - statistics.quantiles(ts, n=4)[0]),
                    'setup_load_ms': su['setup_load_ms'], 'setup_epoch_ms': su['setup_epoch_ms'],
                    'setup_matrix_ms': su['setup_matrix_ms'],
                }
                summary.append(row)
    out_csv(EVAL / 'paper_solver_tradeoff_summary.csv', summary)

    # ── 门禁后再次核对源码哈希（证明只读）──
    for p, h in e3audit.SOURCES.items():
        assert sha(p) == h, f'运行后哈希变化：{p}'
    lock = json.loads((MW / 'input_lock.json').read_text())['sha256']
    for p in [MW / 'workloads.json', EVAL / 'p1_strong_baseline_validation' / 'common.py',
              EVAL / 'p1_strong_baseline_validation' / 'wfc.py', ROOT / 'flowstate' / 'optimizer.py',
              ROOT / 'flowstate' / 'recovery_model.py']:
        assert lock.get(str(p)) == sha(p), f'运行后哈希变化：{p}'

    # ── 控制台摘要（供撰写审计报告核对）──
    print('门禁：Strict20 20/20 一致；C/K3 Greedy/WFC 一致；Exact 门禁 99≤794 放行(Strict20)/n=16 拒绝(C/K3)')
    print('same-J-different-set 记录数：', len(gate_notes))
    for m in ['Greedy', 'Greedy+1-Swap', 'WFC', 'Swap-only', 'Exact']:
        ts = [r['elapsed_ms'] for r in raw if r['method'] == m and r['source'] == 'strict20']
        if ts:
            print('strict20 %-14s median=%.4f p95=%.4f min=%.4f max=%.4f ms' % (
                m, statistics.median(ts), pct95(ts), min(ts), max(ts)))
    for m in ['Greedy', 'Greedy+1-Swap', 'WFC', 'Swap-only']:
        ts = [r['elapsed_ms'] for r in raw if r['method'] == m and r['source'] == 'ck3']
        print('C/K3    %-14s median=%.4f p95=%.4f min=%.4f max=%.4f ms' % (
            m, statistics.median(ts), pct95(ts), min(ts), max(ts)))
    print('总耗时 %.1f s；原始计时行 %d；汇总行 %d' % (time.perf_counter() - t_start, len(raw), len(summary)))


if __name__ == '__main__':
    main()
