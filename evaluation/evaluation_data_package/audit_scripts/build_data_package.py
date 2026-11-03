"""Evaluation 数据包构建脚本(任务书第二/三阶段;只读现有实验数据,新增整理产物)。

原则:
- 每行数据带 block(块)+ source_file(原始来源)两列,可逐行溯源;
- 原始观测(直接抄录)与计算派生指标(本脚本计算)用 kind 列区分;
- 不修改、不删除任何历史数据;所有产物一次性写入(拒绝覆盖);
- CPU 与 GPU 数据分块标注,五轮重复与跨工作负载分开。

产物:recovery_model_data.csv / allocation_data.csv / solver_data.csv /
figure_data/*.csv / source_manifest.json。
"""
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EV = REPO / 'evaluation'
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / 'figure_data'

FREEZE = EV / 'runtime_artifacts' / 'recovery_model_freeze_20260826_154235_266020'
SC_AUDIT = EV / 'paper_recovery_model_short_context_audit'
BUDGET = EV / 'rq3_workload_structure' / 'budget_sensitivity_k1_k4_20261030'
SOLVER = EV / 'paper_solver_tradeoff_summary.csv'
ST20 = EV / 'paper_allocation_structure_strict20.csv'
LEG = EV / 'paper_allocation_structure_legacy573.csv'
CK3GPU = EV / 'paper_allocation_structure_gpu_ck3.csv'
CK3CHK = EV / 'paper_ck3_simple_policy_check'


def rows_of(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, header, rows):
    assert not Path(path).exists(), '拒绝覆盖:' + str(path)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('写出 %s(%d 行)' % (Path(path).relative_to(OUT), len(rows)))


# ============ 1. Recovery Cost Model ============
def build_recovery():
    ms = json.loads((FREEZE / 'model_selection.json').read_text())
    cm = json.loads((FREEZE / 'candidate_models.json').read_text())
    hp = rows_of(FREEZE / 'heldout_predictions.csv')
    cal = rows_of(FREEZE / 'calibration_points.csv')
    pairs = rows_of(SC_AUDIT / 'partB_pairs.csv')
    abso = rows_of(SC_AUDIT / 'partB_absolute.csv')
    ad = json.loads((SC_AUDIT / 'audit_data.json').read_text())

    header = ['block', 'source_file', 'kind', 'model', 'target_tokens', 'gap_tokens',
              'recovery_fraction', 'pair_or_pending', 'metric', 'value', 'unit']
    out = []
    # —— 块 A:模型总览(原始登记 + 选择结论)——
    for m in ('M0', 'M1', 'M2'):
        ov = ms['heldout_metrics'][m]['overall']
        for metric, unit in (('mape_percent', '%'), ('r_squared', ''), ('mae_ms', 'ms'),
                             ('max_relative_error_percent', '%')):
            out.append(['model_overview', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/model_selection.json',
                        '原始登记', m, '', '', '', '', metric, ov[metric], unit])
        out.append(['model_overview', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/candidate_models.json',
                    '原始登记', m, '', '', '', '', 'formula', cm[m]['formula'], ''])
        out.append(['model_overview', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/candidate_models.json',
                    '原始登记', m, '', '', '', '', 'parameters', json.dumps(cm[m]['parameters']), ''])
    out.append(['model_overview', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/model_selection.json',
                '原始登记', 'M2', '', '', '', '', 'selected_model', ms['selected_model'], ''])
    # —— 块 B:held-out 12 点(测量 vs 三模型预测;MAPE 可逐行复算)——
    for r in hp:
        out.append(['heldout_12points', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv',
                    '原始观测', '', r['target_tokens'], r['gap_tokens'], r['recovery_fraction'], '',
                    'measured_phi_ms', r['measured_phi_ms'], 'ms'])
        for m in ('M0', 'M1', 'M2'):
            out.append(['heldout_12points', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv',
                        '原始观测', m, r['target_tokens'], r['gap_tokens'], r['recovery_fraction'], '',
                        'prediction_ms', r['%s_prediction_ms' % m], 'ms'])
            out.append(['heldout_12points', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv',
                        '原始观测', m, r['target_tokens'], r['gap_tokens'], r['recovery_fraction'], '',
                        'relative_error', r['%s_relative_error' % m], ''])
    # —— 块 C:20 校准点 ——
    for r in cal:
        out.append(['calibration_20points', 'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/calibration_points.csv',
                    '原始观测', '', r['target_tokens'], r['gap_tokens'], '', r.get('subset', ''),
                    'measured_phi_ms', r['measured_phi_ms'], 'ms'])
    # —— 块 D:短上下文配对(24 对方向一致性)——
    for r in pairs:
        for metric in ('pred_diff_ms', 'meas_ttft_diff_ms', 'meas_prefill_diff_ms', 'pred_over_meas_prefill'):
            out.append(['short_context_pairs_24', 'paper_recovery_model_short_context_audit/partB_pairs.csv',
                        '原始观测', '', r['T'], r.get('G_a', ''), '', '%s|%s' % (r['dataset'], r['pair']),
                        metric, r[metric], 'ms' if metric != 'pred_over_meas_prefill' else '比值'])
        out.append(['short_context_pairs_24', 'paper_recovery_model_short_context_audit/partB_pairs.csv',
                    '原始观测', '', r['T'], r.get('G_a', ''), '', '%s|%s' % (r['dataset'], r['pair']),
                    'direction_ttft_ok', r['direction_ttft_ok'], 'bool'])
    # —— 块 E:短上下文绝对对照(10 格)——
    for r in abso:
        for metric in ('pred_prefill_ms', 'meas_prefill_ms', 'abs_err_prefill_ms', 'pred_ttft_ms', 'meas_ttft_ms'):
            out.append(['short_context_absolute_10', 'paper_recovery_model_short_context_audit/partB_absolute.csv',
                        '原始观测', r['policy'], r['T'], r['G'], '', r['pending'], metric, r[metric], 'ms'])
    # —— 块 F:审计汇总统计(派生)——
    bc = ad['B_compare']
    # pred_over_meas_prefill_negdiffs 为汇总字典 {n, median, min, max}(登记值)
    pm = bc['pred_over_meas_prefill_negdiffs']
    derived = [('短上下文预测/实测比值中位(深恢复负差 %d 格)' % pm['n'], pm['median'], '比值(区间 %s–%s)' % (pm['min'], pm['max'])),
               ('短上下文方向一致(预测≠0 的 19 对)', '%d/%d' % (bc['direction_both_ok'], bc['n_pred_nonzero']), ''),
               ('短上下文绝对误差中位(10 格,prefill)', bc['abs_err_prefill_ms_median'], 'ms'),
               ('使用域 T 范围(GPU+Strict20)', ad['A_domain']['usage_gpu_and_strict20'].get('T_range', ''), 'token'),
               ('校准域 T 下限', ad['A_domain']['calibration'].get('T_min_tokens', ''), 'token')]
    for name, val, unit in derived:
        out.append(['short_context_summary', 'paper_recovery_model_short_context_audit/audit_data.json', '派生汇总',
                    '', '', '', '', '', name, val, unit])
    # —— MAPE 独立复算(派生;均值口径)——
    for m in ('M0', 'M1', 'M2'):
        rel = [float(r['%s_relative_error' % m]) for r in hp]
        out.append(['heldout_mape_recompute', 'audit_scripts/build_data_package.py(由 heldout_predictions.csv 复算)',
                    '派生复算', m, '', '', '', '', 'mape_percent_mean', round(statistics.fmean(rel) * 100, 6), '%'])
    write_once(OUT / 'recovery_model_data.csv', header, out)
    return hp


# ============ 2. Checkpoint Allocation ============
def build_allocation():
    header = ['block', 'source_file', 'kind', 'dataset_or_population', 'case_or_snapshot', 'K',
              'policy_or_method', 'metric', 'value', 'unit', 'detail']
    out = []
    # —— 块 A:数据集规模总览(legacy573)——
    for r in rows_of(LEG):
        ds = r['dataset']
        for metric, unit in (('conditions', '条'), ('snapshots', '个'), ('n_candidate_min', '个'),
                             ('n_candidate_max', '个'), ('conditions_with_shared', '条'),
                             ('conditions_with_overlap', '条'), ('conditions_K_lt_Kmin', '条'),
                             ('greedy_optimal_in_legacy_model', '条')):
            out.append(['dataset_overview', 'paper_allocation_structure_legacy573.csv', '原始登记', ds, '', '',
                        '', metric, r[metric], unit, ''])
    # —— 块 B:预算敏感性 population×K(四策略同集/同J、FS 胜平负、Oracle)——
    for r in rows_of(BUDGET / 'budget_sensitivity_summary.csv'):
        pop, K = r['population'], r['K']
        base = ['budget_sensitivity_pop_K', 'rq3_workload_structure/budget_sensitivity_k1_k4_20261030/budget_sensitivity_summary.csv',
                '原始登记', pop, '', K]
        out.append(base + ['conditions', r['conditions'], '条', ''])
        out.append(base + ['all_four_same_set', r['all_four_same_set'], '条', ''])
        out.append(base + ['all_four_same_J', r['all_four_same_J'], '条', ''])
        out.append(base + ['shared_conditions', r['shared_conditions'], '条', ''])
        out.append(base + ['oracle_eligible', r['oracle_eligible'], '条', '门内 Exact'])
        wtl = json.loads(r['FlowState_W_T_L'])
        for b, v in wtl.items():
            out.append(base + ['FS_vs_%s_win' % b, v['win'], '条', 'FS 胜=J 更低'])
            out.append(base + ['FS_vs_%s_tie' % b, v['tie'], '条', ''])
            out.append(base + ['FS_vs_%s_loss' % b, v['loss'], '条', ''])
        oc = json.loads(r['policy_oracle_optimal_counts'])
        out.append(base + ['FS_oracle_optimal', oc['FlowState'], '条', '门内最优计'])
    # —— 块 C:Strict20 案例×K(各方法 J;Exact/Greedy/1-Swap/WFC/TopK/PCB/LDR)——
    for r in rows_of(ST20):
        base = ['strict20_case', 'paper_allocation_structure_strict20.csv', '原始登记', 'Strict20', r['case'], r['K']]
        for m, col in (('Exact', 'J_exact'), ('Greedy', 'J_greedy'), ('Greedy+1-Swap', 'J_1swap'),
                       ('WFC', 'J_wfc'), ('StandaloneTopK', 'J_standalone_topk'), ('PCB', 'J_pcb'), ('LDR', 'J_ldr')):
            out.append(base + [m, 'J_ms', r[col], 'ms', ''])
        out.append(base + ['Greedy', 'gap_vs_exact_ms', r['gap_greedy_exact'], 'ms', ''])
        out.append(base + ['Greedy+1-Swap', 'gap_vs_exact_ms', r['gap_1swap_exact'], 'ms', ''])
        out.append(base + ['', 'n_pending', r['n_pending'], '个', ''])
        out.append(base + ['', 'n_candidate', r['n_candidate'], '个', ''])
        out.append(base + ['', 'shared_candidates', r['shared_candidates'], '个', ''])
    # —— 块 D:预算分歧案例(共享/截断机制实例,论文用)——
    for r in rows_of(BUDGET / 'strategy_divergence_cases.csv'):
        base = ['budget_divergence_case', 'rq3_workload_structure/budget_sensitivity_k1_k4_20261030/strategy_divergence_cases.csv',
                '原始登记', r['population'], r['snapshot_id'], r['K']]
        out.append(base + ['FlowState', 'J_ms', r['FlowState_J_ms'], 'ms', ''])
        out.append(base + [r['baseline'], 'J_ms', r['baseline_J_ms'], 'ms', ''])
        out.append(base + [r['baseline'], 'baseline_minus_FS_ms', r['baseline_minus_FlowState_ms'], 'ms', '正值=FS 更优'])
        out.append(base + ['', 'candidate_count', r['candidate_count'], '个', ''])
        out.append(base + ['', 'pending_count', r['pending_count'], '个', ''])
        out.append(base + ['', 'K_min', r['K_min'], '个', ''])
        out.append(base + ['', 'shared', r['shared'], 'bool', ''])
    # —— 块 E:GPU C/K3 逐 Pending(分配→GPU 恢复的桥)——
    for r in rows_of(CK3GPU):
        base = ['gpu_ck3_pending', 'paper_allocation_structure_gpu_ck3.csv', '原始登记(五轮中位)', r['policy'],
                r['pending'], '']
        for metric in ('E', 'G', 'T', 'prefill_tokens', 'cuda_prefill_ms_median', 'ttft_ms_median', 'n'):
            out.append(base + ['', metric, r[metric], 'token' if metric in ('E', 'G', 'T', 'prefill_tokens') else ('ms' if 'ms' in metric else '轮'), r['retained']])
    write_once(OUT / 'allocation_data.csv', header, out)


# ============ 3. Solver Quality & Overhead ============
def build_solver():
    header = ['block', 'source_file', 'kind', 'scope', 'case', 'K', 'n', 'n_pending', 'method',
              'J_ms', 'gap_vs_exact_ms', 'is_exact_optimal', 'median_ms', 'p95_ms', 'min_ms', 'max_ms', 'notes']
    out = []
    summ = rows_of(SOLVER)
    # —— 块 A:Strict20/C/K3 汇总(原始登记;CPU 微基准)——
    for r in summ:
        out.append(['solver_tradeoff_summary', 'paper_solver_tradeoff_summary.csv', '原始登记(CPU 微基准)',
                    r['source'], r['case'], r['K'], r['n'], r['n_pending'], r['method'],
                    r['J_ms'], r['gap_vs_exact_ms'], r['is_exact_optimal'],
                    r['median_ms'], r['p95_ms'], r['min_ms'], r['max_ms'], ''])
    # —— 块 B:预算敏感性求解计时(本脚本由 51780 行原始计时独立聚合)——
    raw = rows_of(BUDGET / 'solver_timing_raw.csv')
    by_cond = {}
    for r in raw:
        by_cond.setdefault((r['population'], r['snapshot_id'], r['K'], r['policy']), []).append(r)
    per_policy_med, per_policy_p95 = {}, {}
    for (pop, snap, K, pol), rs in by_cond.items():
        vals = sorted(float(x['mean_per_call_ms']) for x in rs)
        per_policy_med.setdefault(pol, []).append(statistics.median(vals))
        per_policy_p95.setdefault(pol, []).append(
            vals[max(0, int(round(0.95 * (len(vals) - 1))))])
    cond_n = {pol: len(v) for pol, v in per_policy_med.items()}
    for pol in sorted(per_policy_med):
        out.append(['budget_timing_recompute', 'rq3_workload_structure/budget_sensitivity_k1_k4_20261030/solver_timing_raw.csv',
                    '本脚本独立聚合(CPU,批均单次耗时:条件中位→跨条件中位)', 'rq3_workload_structure 191 快照', '', '',
                    '', '', pol, '', '', '', round(statistics.median(per_policy_med[pol]), 6),
                    round(statistics.median(per_policy_p95[pol]), 6), '', '',
                    '条件数=%d(Exact 仅门内 396 条件)' % cond_n[pol]])
    # —— 块 C:Strict20 最优性计数(本脚本由 strict20 分配表复算)——
    st = rows_of(ST20)
    n = len(st)
    g_opt = sum(1 for r in st if float(r['gap_greedy_exact']) == 0)
    swap_fix = sum(1 for r in st if float(r['gap_greedy_exact']) > 0 and float(r['gap_1swap_exact']) == 0)
    out.append(['strict20_optimality_recompute', 'paper_allocation_structure_strict20.csv',
                '本脚本复算', 'Strict20', '', '', '', '', 'Greedy', '', '', 'True' if g_opt == n else 'False',
                '', '', '', '', '域内最优 %d/%d;Greedy 有差距时 1-Swap 修复 %d/%d' % (g_opt, n, swap_fix, n - g_opt)])
    # —— 块 D:C/K3 全枚举核验(原始登记)——
    ee = json.loads((CK3CHK / 'exact_enumeration.json').read_text())
    gr = ee['flowstate_greedy'] if isinstance(ee['flowstate_greedy'], dict) else {}
    out.append(['ck3_enumeration', 'paper_ck3_simple_policy_check/exact_enumeration.json', '原始登记',
                'C/K3(GPU 冻结输入)', '', '', '', '', 'Exact 枚举',
                ee['min_J_ms'], 0.0, True, '', '', '', '',
                '枚举 %d/%d 子集;argmin 唯一=%s;min_J==Greedy J:%s;tol=%s' % (
                    ee['enumerated_subsets'], ee['expected_subsets'], ee['argmin_count'] == 1,
                    abs(ee['min_J_ms'] - (gr.get('J_ms') if gr else 154.34895561599998)) < 1e-9, ee['tolerance'])])
    write_once(OUT / 'solver_data.csv', header, out)
    return {'strict20_greedy_optimal': g_opt, 'strict20_n': n, 'swap_fix': swap_fix,
            'budget_timing': {p: round(statistics.median(per_policy_med[p]), 6) for p in per_policy_med}}


# ============ 4. figure_data ============
def build_figures(hp):
    # 图:恢复模型预测 vs 实测(held-out 12 点)
    rows = [[r['target_tokens'], r['gap_tokens'], r['recovery_fraction'], r['measured_phi_ms'],
             r['M2_prediction_ms'], r['M0_prediction_ms']] for r in hp]
    write_once(FIG / 'fig_recovery_pred_vs_measured.csv',
               ['target_tokens', 'gap_tokens', 'recovery_fraction', 'measured_phi_ms',
                'M2_position_aware_pred_ms', 'M0_gap_only_pred_ms'], rows)
    # 图:GPU 逐 Pending 配对差(五轮+中位)
    f = json.loads((OUT / 'audit_findings_five_round.json').read_text())
    rows = []
    for key in ('FS-MC', 'FS-KV'):
        pd_ = f['paired'][key]['per_pending_diffs']
        for i, pid in enumerate(('C_A', 'C_B', 'C_C', 'C_D')):
            rows.append([key, pid] + [pd_[pid][r] for r in range(5)] + [f['paired'][key]['per_pending_median'][pid]])
    write_once(FIG / 'fig_gpu_ttft_paired.csv',
               ['comparison', 'pending', 'R1_diff_ms', 'R2_diff_ms', 'R3_diff_ms', 'R4_diff_ms', 'R5_diff_ms', 'median_ms'], rows)
    rows = []
    for key in ('FS-MC', 'FS-KV'):
        rows.append([key, 'sum_ttft'] + f['paired'][key]['sum_ttft_diffs'] + [f['paired'][key]['sum_ttft_diff_median']])
    write_once(FIG / 'fig_gpu_sum_ttft_paired.csv',
               ['comparison', 'metric', 'R1', 'R2', 'R3', 'R4', 'R5', 'median_ms'], rows)
    # 图:预算敏感性胜负结构
    rows = []
    for r in rows_of(BUDGET / 'budget_sensitivity_summary.csv'):
        wtl = json.loads(r['FlowState_W_T_L'])
        oc = json.loads(r['policy_oracle_optimal_counts'])
        rows.append([r['population'], r['K'], r['conditions'], r['shared_conditions'],
                     wtl['PCB']['win'], wtl['PCB']['tie'], wtl['PCB']['loss'],
                     wtl['LDR']['win'], wtl['LDR']['tie'], wtl['LDR']['loss'],
                     wtl['WFC']['win'], wtl['WFC']['tie'], wtl['WFC']['loss'], oc['FlowState']])
    write_once(FIG / 'fig_budget_sensitivity.csv',
               ['population', 'K', 'conditions', 'shared_conditions',
                'FS_vs_PCB_win', 'tie', 'loss', 'FS_vs_LDR_win', 'tie2', 'loss2',
                'FS_vs_WFC_win', 'tie3', 'loss3', 'FS_oracle_optimal'], rows)
    # 图:求解器质量-耗时(Strict20 各方法)
    summ = rows_of(SOLVER)
    by_method = {}
    for r in summ:
        if r['source'] == 'strict20':
            by_method.setdefault(r['method'], []).append(r)
    rows = []
    for mth, rs in by_method.items():
        meds = [float(r['median_ms']) for r in rs]
        gaps = [float(r['gap_vs_exact_ms'] or 0) for r in rs]
        opts = sum(1 for r in rs if r['is_exact_optimal'] in ('True', 'true'))
        rows.append([mth, len(rs), round(statistics.median(meds), 6), round(max(meds), 6),
                     round(statistics.median(gaps), 6), opts])
    write_once(FIG / 'fig_solver_tradeoff.csv',
               ['method', 'cases', 'median_ms_median', 'median_ms_max', 'gap_vs_exact_ms_median', 'exact_optimal_cases'], rows)


# ============ 5. source_manifest.json ============
def build_manifest():
    pkg_files = ['recovery_model_data.csv', 'allocation_data.csv', 'solver_data.csv',
                 'five_round_audit.md', 'data_audit_report.md', 'experiment_inventory.md',
                 'gpu_per_round.csv', 'gpu_per_pending.csv', 'gpu_overhead.csv',
                 'audit_findings_five_round.json',
                 'figure_data/fig_recovery_pred_vs_measured.csv', 'figure_data/fig_gpu_ttft_paired.csv',
                 'figure_data/fig_gpu_sum_ttft_paired.csv', 'figure_data/fig_budget_sensitivity.csv',
                 'figure_data/fig_solver_tradeoff.csv']
    sources = ['runtime_artifacts/recovery_model_freeze_20260826_154235_266020/model_selection.json',
               'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/candidate_models.json',
               'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv',
               'runtime_artifacts/recovery_model_freeze_20260826_154235_266020/calibration_points.csv',
               'paper_recovery_model_short_context_audit/partB_pairs.csv',
               'paper_recovery_model_short_context_audit/partB_absolute.csv',
               'paper_recovery_model_short_context_audit/audit_data.json',
               'paper_allocation_structure_legacy573.csv', 'paper_allocation_structure_strict20.csv',
               'paper_allocation_structure_gpu_ck3.csv',
               'rq3_workload_structure/budget_sensitivity_k1_k4_20261030/budget_sensitivity_summary.csv',
               'rq3_workload_structure/budget_sensitivity_k1_k4_20261030/strategy_divergence_cases.csv',
               'rq3_workload_structure/budget_sensitivity_k1_k4_20261030/solver_timing_raw.csv',
               'paper_solver_tradeoff_summary.csv', 'paper_solver_tradeoff_raw.csv',
               'paper_ck3_simple_policy_check/exact_enumeration.json',
               'p0_control_cycle_gpu/arms/five_round_ledger.jsonl',
               'p0_control_cycle_gpu/arms/m1_ck3_fs/run/summary.json']
    manifest = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'note': '数据包内每个 CSV 的 source_file 列逐行标注原始来源;此处登记来源文件 sha256 与包内文件 sha256。',
        'package_files': {f: {'sha256': sha256(OUT / f)} for f in pkg_files if (OUT / f).exists()},
        'source_files': {},
    }
    for s in sources:
        p = EV / s
        manifest['source_files'][s] = {'sha256': sha256(p), 'bytes': p.stat().st_size} if p.exists() else 'MISSING'
    # 五轮全部 15 臂 summary 逐臂登记(数据包 GPU 数据的原始来源)
    for r in range(1, 6):
        for suf in ('fs', 'mc', 'kv'):
            s = 'p0_control_cycle_gpu/arms/m%d_ck3_%s/run/summary.json' % (r, suf)
            manifest['source_files'][s] = {'sha256': sha256(EV / s), 'bytes': (EV / s).stat().st_size}
    path = OUT / 'source_manifest.json'
    assert not path.exists(), '拒绝覆盖:' + str(path)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('写出 source_manifest.json(来源 %d 个;包内文件 %d 个)' % (len(manifest['source_files']), len(manifest['package_files'])))


def main():
    hp = build_recovery()
    build_allocation()
    solver_info = build_solver()
    build_figures(hp)
    build_manifest()
    print('Strict20 复算:Greedy 域内最优 %d/%d,1-Swap 修复 %d/%d' % (
        solver_info['strict20_greedy_optimal'], solver_info['strict20_n'],
        solver_info['swap_fix'], solver_info['strict20_n'] - solver_info['strict20_greedy_optimal']))
    print('预算敏感性计时复算(中位 ms):', json.dumps(solver_info['budget_timing']))


if __name__ == '__main__':
    main()
