"""P1-1：短上下文恢复成本模型适用性与决策敏感性审计（CPU-only 只读复算）。

数据源（全部只读，不改任何冻结输入）：
- 冻结模型：flowstate/recovery_model.py 与 runtime_artifacts/recovery_model_freeze_20260826_154235_266020/
  （calibration_points.csv / candidate_models.json / model_selection.json / heldout_*.csv）
- T2-G2 三策略 60 臂：evaluation/t2g1_three_policy_gpu/formal_gpu1b/{pending_policy_matrix,pair_analysis}.csv
- C/K3 三策略 60 臂：evaluation/c_formal_perf_gpu/{pending_policy_matrix,pair_analysis,policy_plans}.csv
- B5-C4/K2 Exact-Greedy 40 臂：evaluation/b5c4_formal_gpu_g2/formal/raw_results.csv
- 决策实例：greedy_swap_structure_e3_step1（Strict20 五快照）与
  kvflow_multiworkflow_cpu_main/workloads.json（C/K3 快照）

五个部分：
A 冻结模型与训练域核查：校准/held-out 的 T、G 覆盖范围 vs 实际使用域（T2-G2/C/K3/B5-C4/Strict20）；
B 短上下文预测与 GPU 实测对照：同 Pending 跨策略配对差（预测 Phi 差 vs 实测 TTFT/Prefill 差），
  绝对误差仅对存在同 Pending、同协议 G=0 基线的格计算（T2-G2/C-K3 的 FS 深恢复格）；
C 分配决策敏感性：冻结兼容矩阵/候选/预算下，Position-aware(M2) 与 Gap-only(M0) 各自跑
  Marginal Greedy 与 Exact 全枚举，统计选集改变条件数并统一 M2 尺度评价；
D T2-G2 三策略模型化 J 排序 vs 实测等权 TTFT 排序（逐格方向一致性 + 斜率对照）；
E 汇总判定素材（写入报告）。

口径：Phi 预测的是相对同 T 零缺口条件的额外恢复成本（Phi(0,T)=0 精确），
不与完整 TTFT 直接比较；不同模型 J 不直接相减，统一在 M2 Phi 域评价选集。
"""
import bisect
import csv
import itertools
import json
import statistics as st
from decimal import Decimal
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
FRZ = ROOT / 'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020'
T2 = ROOT / 'evaluation/t2g1_three_policy_gpu/formal_gpu1b'
CK = ROOT / 'evaluation/c_formal_perf_gpu'
G2 = ROOT / 'evaluation/b5c4_formal_gpu_g2/formal'
E3 = ROOT / 'evaluation/greedy_swap_structure_e3_step1'
MW = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
OUT = ROOT / 'evaluation/paper_recovery_model_short_context_audit'
EPS = 1e-9
# 冻结 Phi 系数（Ki-token 单位；与 flowstate/recovery_model.py 元数据逐位一致）
PHI_A, PHI_B, PHI_C = Decimal('37.828150'), Decimal('0.345974143'), Decimal('-0.156201917')


def phi(g_tokens, t_tokens):
    """冻结位置感知模型 M2：Phi(G,T)=a*g+b*g*t+c*g^2（g/t 为 Ki-token）。"""
    if g_tokens == 0:
        return 0.0
    g, t = Decimal(g_tokens) / 1024, Decimal(t_tokens) / 1024
    return float(PHI_A * g + PHI_B * g * t + PHI_C * g * g)


# M0 gap-only：冻结候选模型中的分段线性 knots（gap_ki -> cost_ms），线性插值
M0_KNOTS = [(k['gap_ki'], k['cost_ms']) for k in
            json.loads((FRZ / 'candidate_models.json').read_text())['M0']['parameters']['knots']]
_M0_X = [k[0] for k in M0_KNOTS]
_M0_Y = [k[1] for k in M0_KNOTS]


def m0_cost(g_tokens):
    """M0 gap-only 成本：仅依赖恢复缺口，与 T 无关（域内分段线性插值）。"""
    if g_tokens == 0:
        return 0.0
    g = g_tokens / 1024.0
    if g > _M0_X[-1]:
        raise ValueError('gap 超出 M0 剖面域')
    i = bisect.bisect_left(_M0_X, g)
    if _M0_X[i] == g:
        return _M0_Y[i]
    x0, x1 = _M0_X[i - 1], _M0_X[i]
    y0, y1 = _M0_Y[i - 1], _M0_Y[i]
    return y0 + (y1 - y0) * (g - x0) / (x1 - x0)


def med(rows, key):
    """取数值列中位。"""
    return st.median(float(r[key]) for r in rows)


def solve_condition(tag, case, K, inst, frozen_greedy):
    """两模型各跑 Greedy+Exact，统一 M2 尺度评价选集差异。

    inst: {cands:[id...], pendings:[(pid,T)...], depth:{cand->{pid->恢复深度}}}
    """
    ids = [c[0] for c in inst['cands']]
    depth = inst['depth']
    Ts = dict(inst['pendings'])

    def run(model):
        def J(sel):
            tot = 0.0
            for pid, T in Ts.items():
                e = max([depth[c][pid] for c in sel if pid in depth[c]] + [0])
                tot += phi(T - e, T) if model == 'M2' else m0_cost(T - e)
            return tot

        # Marginal Greedy：边际收益最大、并列索引序、零边际停
        sel = []
        while len(sel) < K:
            gains = [(J(sel) - J(sel + [c]), ids.index(c)) for c in ids if c not in sel]
            d, i = max(gains, key=lambda t: (t[0], -t[1]))
            if d <= EPS:
                break
            sel.append(ids[i])
        # Exact：全枚举 |S|<=K，并列取 (大小, ID 字典序) 最小
        bestj, bestS = None, None
        for k in range(K + 1):
            for S in itertools.combinations(sorted(ids), k):
                j = J(list(S))
                if bestj is None or j < bestj - EPS or (abs(j - bestj) <= EPS and
                                                        (len(S), sorted(S)) < (len(bestS), sorted(bestS))):
                    bestj, bestS = j, list(S)
        return list(sel), sorted(bestS)  # 贪心保留选择顺序（用于逐步序列对比）

    g2_sel, e2 = run('M2')
    g0_sel, e0 = run('M0')

    def Jm2(sel):
        tot = 0.0
        for pid, T in Ts.items():
            e = max([depth[c][pid] for c in sel if pid in depth[c]] + [0])
            tot += phi(T - e, T)
        return tot

    fz = frozen_greedy.get((case, K))
    return {'tag': tag, 'case': case, 'K': K,
            'greedy_M2': sorted(g2_sel), 'greedy_M0': sorted(g0_sel), 'exact_M2': e2, 'exact_M0': e0,
            'greedy_changed': sorted(g2_sel) != sorted(g0_sel), 'exact_changed': e2 != e0,
            'greedy_seq_same': g2_sel == g0_sel,
            'm2_matches_frozen': fz is None or sorted(g2_sel) == fz,
            'J_m2_greedy_M0set': round(Jm2(g0_sel), 4), 'J_m2_greedy_M2set': round(Jm2(g2_sel), 4),
            'J_m2_exact_M0set': round(Jm2(e0), 4), 'J_m2_exact_M2set': round(Jm2(e2), 4),
            'delta_greedy_M2scale': round(Jm2(g0_sel) - Jm2(g2_sel), 4),
            'delta_exact_M2scale': round(Jm2(e0) - Jm2(e2), 4)}


def main():
    checks, parts = [], []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})

    parts = {}

    # ══ A 冻结模型与训练域核查 ══
    cal = list(csv.DictReader((FRZ / 'calibration_points.csv').open()))
    held = list(csv.DictReader((FRZ / 'heldout_predictions.csv').open()))
    sel = json.loads((FRZ / 'model_selection.json').read_text())
    cal_t = [int(r['target_tokens']) for r in cal]
    cal_g = [int(r['gap_tokens']) for r in cal]
    ho_t = [int(r['target_tokens']) for r in held]
    ho_g = [int(r['gap_tokens']) for r in held]
    # 使用域：三组 GPU 实验的 T/G 与 Strict20 全部 20 条件的 T/G
    use = []
    for f, pcol, ecol, gcol in ((T2 / 'pending_policy_matrix.csv', 'policy', 'E', 'G'),
                                (CK / 'pending_policy_matrix.csv', 'pol', 'E', 'G')):
        for r in csv.DictReader(f.open()):
            use.append((int(r[ecol]) + int(r[gcol]), int(r[gcol]), r['pending'], r[pcol]))
    g2raw = list(csv.DictReader((G2 / 'raw_results.csv').open()))
    for r in g2raw:
        use.append((int(r['H']), int(r['G_model']), r['pending'], r['solver']))
    e3rows = list(csv.DictReader((E3 / 'condition_results.csv').open()))
    e3_use = []
    for r in e3rows:
        for p in json.loads(r['per_pending']):
            e3_use.append((p['T'], p['G']))
    use_t = [x[0] for x in use] + [x[0] for x in e3_use]
    use_g = [x[1] for x in use] + [x[1] for x in e3_use]
    domain = {
        'calibration': {'n_points': len(cal), 'T_range': [min(cal_t), max(cal_t)],
                        'G_range': [min(cal_g), max(cal_g)]},
        'heldout': {'n_points': len(held), 'T_range': [min(ho_t), max(ho_t)],
                    'G_range': [min(ho_g), max(ho_g)],
                    'used_for_fit': sel['heldout_used_for_fit']},
        'usage_gpu_and_strict20': {'T_range': [min(use_t), max(use_t)],
                                   'G_range': [min(use_g), max(use_g)]},
        'extrapolation_flags': {
            'T_below_calibration_min': max(use_t) < min(cal_t),
            'G_below_calibration_min': min(cal_g) > 0 and min(use_g) < min(cal_g),
            'inside_formal_domain_max_T_131072': max(use_t) <= 131072},
        'heldout_overall': {m: sel['heldout_metrics'][m]['overall'] for m in ('M0', 'M1', 'M2')},
        'frozen_m2_params': {'a': 37.828150, 'b': 0.345974143, 'c': -0.156201917,
                             'units': 'ms, Ki-token 输入', 'Phi_at_zero_gap': 0.0},
        'gpu_measurements_independent_of_calibration': True,
        # 校准=replay profiler（2026-08-26 专用剖面 Runner）；GPU 对照=T2-G2/C-K3/B5-C4 正式 Runner，
        # 不同日期、不同工作负载、不同测量协议，不存在训练样本复用。
    }
    check('使用域 T 全部低于校准最小 T（短侧外推）', domain['extrapolation_flags']['T_below_calibration_min'])
    check('使用域存在 G<4096（校准最小 gap 之下的空洞）', domain['extrapolation_flags']['G_below_calibration_min'])
    check('使用域在形式验证域 T<=131072 内', domain['extrapolation_flags']['inside_formal_domain_max_T_131072'])
    check('held-out 未参与拟合', sel['heldout_used_for_fit'] is False)
    check('M0 剖面域覆盖全部使用 gap', max(use_g) <= _M0_X[-1] * 1024)
    parts['A_domain'] = domain

    # ══ B 短上下文预测与 GPU 实测对照 ══
    pair_rows, abs_rows = [], []
    # 各数据源 (G, T, prefill_cuda) 点集（斜率对照用）
    pts = {'T2G2': [], 'CK3': [], 'B5C4_K2': []}

    def add_dataset(tag, grid, pairs, zero_keys, alias=None):
        """grid: {(pending,pol):(T,E,G,prefill_cuda,ttft)}；pairs: {(pending,'X-Y'):{ttft,prefill}} 中位差；
        zero_keys: 同协议 G=0 基线格键集合（绝对口径允许的基线）；alias: comparator 简称->策略全名。"""
        alias = alias or {}
        for (pend, comp), rec in pairs.items():
            pa, pb = (alias.get(x, x) for x in comp.split('-'))
            T, ga, gb = grid[(pend, pa)][0], grid[(pend, pa)][2], grid[(pend, pb)][2]
            pred = phi(ga, T) - phi(gb, T)
            meas_ttft, meas_pref = rec['ttft'], rec['prefill']
            same_frontier = abs(pred) <= EPS
            pair_rows.append({
                'dataset': tag, 'pending': pend, 'pair': comp, 'T': T, 'G_a': ga, 'G_b': gb,
                'pred_diff_ms': round(pred, 3),
                'meas_ttft_diff_ms': round(meas_ttft, 3), 'meas_prefill_diff_ms': round(meas_pref, 3),
                'same_frontier': same_frontier,
                # 同前沿（pred=0）对为控制对照：实测差应为噪声带（C/K3 双峰残差 final_report §3.2）
                'direction_ttft_ok': True if same_frontier else (pred > EPS) == (meas_ttft > 0),
                'direction_prefill_ok': True if same_frontier else (pred > EPS) == (meas_pref > 0),
                'pred_over_meas_prefill': None if same_frontier or abs(meas_pref) <= 1
                else round(pred / meas_pref, 3)})
        # 绝对口径：pred = 同 Pending G=0 基线格实测 + Phi(G,T)（禁止跨 Pending 借用）；
        # 同 Pending 有多个 G=0 格时优先研究系统自身基线（FS/FlowState-Greedy）
        for key in sorted(grid):
            pend, pol = key
            T, E, G, pc, tt = grid[key]
            cands_base = [k for k in zero_keys if k[0] == pend]
            base_key = (next((k for k in cands_base if k[1] in ('FS', 'FlowState-Greedy')), None)
                        or (cands_base[0] if cands_base else None))
            if G > 0 and base_key is not None and key != base_key:
                bpc, btt = grid[base_key][3], grid[base_key][4]
                abs_rows.append({
                    'dataset': tag, 'pending': pend, 'policy': pol, 'T': T, 'G': G,
                    'base_policy_G0': base_key[1],
                    'pred_prefill_ms': round(bpc + phi(G, T), 3), 'meas_prefill_ms': pc,
                    'abs_err_prefill_ms': round(bpc + phi(G, T) - pc, 3),
                    'pred_ttft_ms': round(btt + phi(G, T), 3), 'meas_ttft_ms': tt,
                    'abs_err_ttft_ms': round(btt + phi(G, T) - tt, 3)})

    # T2-G2：12 格网格 + 配对中位
    grid = {}
    for r in csv.DictReader((T2 / 'pending_policy_matrix.csv').open()):
        grid[(r['pending'], r['policy'])] = (int(r['E']) + int(r['G']), int(r['E']), int(r['G']),
                                             float(r['prefill_cuda_median_ms']), float(r['ttft_median_ms']))
        pts['T2G2'].append((int(r['G']), int(r['E']) + int(r['G']), float(r['prefill_cuda_median_ms'])))
    pairs = {}
    for r in csv.DictReader((T2 / 'pair_analysis.csv').open()):
        pairs.setdefault((r['pending'], r['comparator']), []).append(r)
    pairs = {k: {'ttft': med(v, 'ttft_diff_ms'), 'prefill': med(v, 'prefill_cuda_diff_ms')}
             for k, v in pairs.items()}
    add_dataset('T2G2', grid, pairs, {k for k, v in grid.items() if v[2] == 0},
                alias={'FS': 'FlowState-Greedy', 'MC': 'Marconi-Evict', 'KV': 'KVFlow-inspired'})

    # C/K3：同构（策略 FS/SA/WFC）
    grid = {}
    for r in csv.DictReader((CK / 'pending_policy_matrix.csv').open()):
        grid[(r['pending'], r['pol'])] = (int(r['E']) + int(r['G']), int(r['E']), int(r['G']),
                                          float(r['cuda_prefill_ms']), float(r['ttft_ms']))
        pts['CK3'].append((int(r['G']), int(r['E']) + int(r['G']), float(r['cuda_prefill_ms'])))
    pairs = {}
    for r in csv.DictReader((CK / 'pair_analysis.csv').open()):
        pairs.setdefault((r['pending'], r['pair']), []).append(r)
    pairs = {k: {'ttft': med(v, 'd_ttft_ms'), 'prefill': med(v, 'd_cuda_prefill_ms')}
             for k, v in pairs.items()}
    add_dataset('CK3', grid, pairs, {k for k, v in grid.items() if v[2] == 0})

    # B5-C4/K2：raw 40 臂按 (pending,solver) 聚合中位，Greedy-Exact 配对（无 G=0 臂，仅相对口径）
    agg = {}
    for r in g2raw:
        agg.setdefault((r['pending'], r['solver']), []).append(r)
    grid = {k: (int(v[0]['H']), int(v[0]['E']), int(v[0]['G_model']),
                med(v, 'cuda_prefill_ms'), med(v, 'ttft_ms')) for k, v in agg.items()}
    for (pend, sv), v in agg.items():
        pts['B5C4_K2'].append((int(v[0]['G_model']), int(v[0]['H']), med(v, 'cuda_prefill_ms')))
    pairs = {}
    for pend in sorted({k[0] for k in grid}):
        a, b = grid[(pend, 'Exact')], grid[(pend, 'Greedy')]
        pairs[(pend, 'Greedy-Exact')] = {'ttft': b[4] - a[4], 'prefill': b[3] - a[3]}
    add_dataset('B5C4_K2', grid, pairs, set())

    n_nz = sum(1 for r in pair_rows if not r['same_frontier'])
    n_dir = sum(1 for r in pair_rows if not r['same_frontier']
                and r['direction_ttft_ok'] and r['direction_prefill_ok'])
    n_sf = sum(1 for r in pair_rows if r['same_frontier'])
    ratios = [r['pred_over_meas_prefill'] for r in pair_rows
              if r['pred_over_meas_prefill'] and r['meas_prefill_diff_ms'] < -1]
    abs_err = [abs(r['abs_err_prefill_ms']) for r in abs_rows]
    parts['B_compare'] = {
        'n_pairs': len(pair_rows),
        'n_pred_nonzero': n_nz,
        'direction_both_ok': n_dir,
        'direction_consistency': '%d/%d（同前沿控制对 %d 个单列）' % (n_dir, n_nz, n_sf),
        'same_frontier_controls': [{k: r[k] for k in ('dataset', 'pending', 'pair', 'meas_ttft_diff_ms')}
                                   for r in pair_rows if r['same_frontier']],
        'pred_over_meas_prefill_negdiffs': {
            'n': len(ratios), 'median': round(st.median(ratios), 3) if ratios else None,
            'min': round(min(ratios), 3) if ratios else None, 'max': round(max(ratios), 3) if ratios else None},
        'abs_rows_n': len(abs_rows),
        'abs_err_prefill_ms_median': round(st.median(abs_err), 3) if abs_err else None,
        'abs_err_prefill_ms_max': round(max(abs_err), 3) if abs_err else None}
    check('B: pred!=0 配对方向（TTFT 与 Prefill 双口径）全一致', n_dir == n_nz,
          parts['B_compare'])
    # 偏差来源量化：TTFT 配对差 vs Prefill_CUDA 配对差的相关与残差（恢复计算主导性）
    tx = [(r['meas_ttft_diff_ms'], r['meas_prefill_diff_ms']) for r in pair_rows if not r['same_frontier']]
    mx_, my_ = st.mean(x for x, _ in tx), st.mean(y for _, y in tx)
    cov = sum((x - mx_) * (y - my_) for x, y in tx)
    r2 = (cov ** 2 / (sum((x - mx_) ** 2 for x, _ in tx) * sum((y - my_) ** 2 for _, y in tx))) if tx else 0
    resid = [abs(x - y) for x, y in tx]
    parts['B_compare']['ttft_vs_prefill_r2'] = round(r2, 4)
    parts['B_compare']['ttft_minus_prefill_abs_median_ms'] = round(st.median(resid), 3)
    parts['B_compare']['ttft_minus_prefill_abs_max_ms'] = round(max(resid), 3)
    check('B: 绝对误差仅用同 Pending G=0 基线（FS 深恢复格）',
          all(r['base_policy_G0'] in ('FlowState-Greedy', 'FS') for r in abs_rows))

    # 斜率对照：各数据源 prefill_cuda 对 G 的 OLS 斜率（G>0 点）vs Phi 预测平均斜率
    slope = {}
    for tag, xy in pts.items():
        xg = [x for x, _, _ in xy if x > 0]
        yg = [y for x, _, y in xy if x > 0]
        mx, my = st.mean(xg), st.mean(yg)
        beta = (sum((x - mx) * (y - my) for x, y in zip(xg, yg))
                / sum((x - mx) ** 2 for x in xg))
        phi_sl = st.mean([phi(x, t) / x for x, t, _ in xy if x > 0])
        slope[tag] = {'n': len(xg), 'meas_ols_ms_per_ktoken': round(beta * 1024, 3),
                      'phi_mean_ms_per_ktoken': round(phi_sl * 1024, 3),
                      'phi_over_meas': round(phi_sl / beta, 3)}
    parts['B_slope'] = slope

    # ══ C 分配决策敏感性（Position-aware M2 vs Gap-only M0） ══
    cond_rows = []
    traces = json.loads((E3 / 'selection_traces.json').read_text())
    step1 = {(t['case'], t['K']): t for t in traces if t['step'] == 1}
    matrix = {c: {x['checkpoint_id']: dict(x['advances']) for x in t['ranked']}
              for (c, K), t in step1.items() if K == 1}
    Tby, frozen_greedy = {}, {}
    for r in e3rows:
        Tby.setdefault(r['case'], {p['continuation_id']: p['T'] for p in json.loads(r['per_pending'])})
        if r['policy'] == 'Greedy':
            frozen_greedy[(r['case'], int(r['K']))] = sorted(json.loads(r['selected']))
    for case in sorted(matrix):
        pids = list(next(iter(matrix[case].values())).keys())
        inst = {'cands': [(c, 0, set(adv.keys())) for c, adv in matrix[case].items()],
                'pendings': [(p, Tby[case][p]) for p in pids],
                'depth': matrix[case]}
        for K in (1, 2, 3, 4):
            cond_rows.append(solve_condition('Strict20', case, K, inst, frozen_greedy))
    # C/K3 实例（workloads.json structure C，K=3；ready 筛选语义与 ck3 检查一致）
    wl = json.loads((MW / 'workloads.json').read_text())
    wcase = [x for x in wl['cases'] if x['structure'] == 'C'][0]
    calls = {p['continuation_id']: p for p in wcase['calls']}
    ready = [i for i in wcase['predeclared_evaluation_order'] if calls[i]['ready']]
    ids = set(ready)
    ck_inst = {'cands': [(c['checkpoint_id'], 0, {x for x in c['token_compatible_calls'] if x in ids})
                         for c in wcase['candidates']],
               'pendings': [(i, calls[i]['planning_target_T']) for i in ready],
               'depth': {c['checkpoint_id']: {x: c['token_position_L'] for x in c['token_compatible_calls'] if x in ids}
                         for c in wcase['candidates']}}
    ck_plans = json.loads((CK / 'policy_plans.json').read_text())
    ck_frozen = {('C', 3): sorted(ck_plans['FlowState-Greedy']['selected_checkpoint_ids'])}
    cond_rows.append(solve_condition('CK3', 'C', 3, ck_inst, ck_frozen))

    n_greedy_diff = sum(1 for r in cond_rows if r['greedy_changed'])
    n_exact_diff = sum(1 for r in cond_rows if r['exact_changed'])
    parts['C_sensitivity'] = {
        'n_conditions': len(cond_rows),
        'greedy_selection_changed': n_greedy_diff,
        'exact_selection_changed': n_exact_diff,
        'm2_calibration_all_match_frozen': all(r['m2_matches_frozen'] for r in cond_rows),
        'max_abs_delta_greedy_M2scale': max(abs(r['delta_greedy_M2scale']) for r in cond_rows),
        'max_abs_delta_exact_M2scale': max(abs(r['delta_exact_M2scale']) for r in cond_rows)}
    check('C: M2 Greedy 复现冻结选集（Strict20 20 条件 + CK3 FS）',
          all(r['m2_matches_frozen'] for r in cond_rows))
    parts['C_rows'] = cond_rows

    # ══ D T2-G2 预测排序 vs 实测排序 ══
    J_spec = {'FlowState-Greedy': 154.35, 'Marconi-Evict': 703.46, 'KVFlow-inspired': 627.61}
    t2grid = list(csv.DictReader((T2 / 'pending_policy_matrix.csv').open()))
    J_rec = {pol: round(sum(phi(int(r['G']), int(r['E']) + int(r['G']))
                            for r in t2grid if r['policy'] == pol), 4) for pol in J_spec}
    rew = list(csv.DictReader((T2 / 'round_equal_weight.csv').open()))
    meas = {}
    for pol in J_spec:
        vals = [float(r['ttft_ew_a_ms']) if r['policy_a'] == pol else float(r['ttft_ew_b_ms'])
                for r in rew if pol in (r['policy_a'], r['policy_b'])]
        meas[pol] = round(st.median(vals), 3)
    j_rank = sorted(J_spec, key=J_spec.get)
    m_rank = sorted(meas, key=meas.get)
    t2_pairs = [r for r in pair_rows if r['dataset'] == 'T2G2']
    parts['D_ranking'] = {
        'J_modeled_spec': J_spec, 'J_recomputed_from_EG': J_rec,
        'J_match': all(abs(J_spec[p] - J_rec[p]) < 0.02 for p in J_spec),
        'measured_eq_ttft_median': meas,
        'J_ranking': j_rank, 'measured_ranking': m_rank,
        'ranking_consistent': j_rank == m_rank,
        'per_pending_pairs_direction_ok': sum(1 for r in t2_pairs if r['direction_ttft_ok'] and r['direction_prefill_ok']),
        'per_pending_pairs_total': len(t2_pairs)}
    check('D: J 复算 == spec 给定三值', parts['D_ranking']['J_match'], J_rec)
    check('D: 模型化 J 排序 == 实测等权 TTFT 排序', j_rank == m_rank)

    parts['checks'] = checks
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit_data.json').write_text(json.dumps(parts, ensure_ascii=False, indent=1))
    for name, rows in (('partB_pairs.csv', pair_rows), ('partB_absolute.csv', abs_rows),
                       ('partC_conditions.csv', cond_rows)):
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print('== A 域 == 使用 T %s G %s | 校准 T %s G %s | held-out T %s G %s' % (
        domain['usage_gpu_and_strict20']['T_range'], domain['usage_gpu_and_strict20']['G_range'],
        domain['calibration']['T_range'], domain['calibration']['G_range'],
        domain['heldout']['T_range'], domain['heldout']['G_range']))
    print('== B 对照 == 配对 %d 方向一致 %s；比值(负差格) %s；绝对 %d 格 |err| 中位 %s' % (
        len(pair_rows), parts['B_compare']['direction_consistency'],
        parts['B_compare']['pred_over_meas_prefill_negdiffs'], len(abs_rows),
        parts['B_compare']['abs_err_prefill_ms_median']))
    print('   斜率对照:', json.dumps(slope, ensure_ascii=False))
    print('== C 敏感性 == 条件 %d：greedy 改 %d exact 改 %d；M2 校准 %s' % (
        len(cond_rows), n_greedy_diff, n_exact_diff,
        parts['C_sensitivity']['m2_calibration_all_match_frozen']))
    print('== D 排序 == J 一致 %s；排序一致 %s（J %s vs 实测 %s）' % (
        parts['D_ranking']['J_match'], j_rank == m_rank, j_rank, m_rank))
    print('检查 %d 项，失败：%s' % (len(checks), [c['check'] for c in checks if not c['pass']] or '无'))


if __name__ == '__main__':
    main()
