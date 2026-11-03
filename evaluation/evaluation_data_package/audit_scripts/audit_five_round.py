"""P0-4 五轮 GPU 实验·独立复算审计(任务书第一阶段)。

独立性声明:本脚本不导入 five_round_stats.py、不读取其产出的 CSV/JSON,
直接从 15 个实验臂的原始 run/summary.json、source_identity.json、装配输入
与追加式总账重新计算全部指标;所有断言 fail-loud。

审计范围(任务书 §二):
1. 正确性与身份:15 臂 PASS/summary 唯一/零 failure;镜像与 Runtime 16 文件、
   Runner/Observation 哈希(以原始 identity 文件为准,并核对报告历史笔误);
   五轮 9 装配输入逐字节一致;在线选择内核身份;无未来信息泄漏(请求全部在
   T3 之后启动);cold/E/T/G/prefill/COW/FA/恒等式;R1 Pilot 排除;R3 公平性。
2. 性能独立复算:逐轮逐 Pending TTFT/ΣTTFT/Σprefill/配对差/中位数/逐 Pending
   配对中位/相对改善%(显式分母)/J/J vs 实测 prefill 差异(显式分母)/
   selection_call/control_cycle/full_cycle/request_exec。
3. R3 敏感性:含 R3 五轮 vs 排除 R3 四轮。
4. 控制开销分解:六级 apply 阶段+段间隙+未分解余量;不定义净收益。

产物(一次性写入,拒绝覆盖):
- ../gpu_per_round.csv   逐轮×策略复算明细
- ../gpu_per_pending.csv 逐轮×策略×Pending 复算明细(含配对差)
- ../gpu_overhead.csv    控制周期与开销分解
- ../audit_findings_five_round.json  全部审计发现(供撰写 five_round_audit.md)
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / 'evaluation' / 'p0_control_cycle_gpu'
ARMS = PKG / 'arms'
OUT = Path(__file__).resolve().parents[1]

ROUNDS = [1, 2, 3, 4, 5]
POLICY_SUFFIX = {'FlowState-Greedy': 'fs', 'Marconi-Evict': 'mc', 'KVFlow-inspired': 'kv'}
POLICIES = list(POLICY_SUFFIX)
# 9 个装配输入(与 prepare.NAMES+t2g1_plans+workloads 对应;device_binding 为运行时输出不入列)
INPUT_FILES = ['cold_reference_C_A.json', 'cold_reference_C_B.json', 'cold_reference_C_C.json',
               'cold_reference_C_D.json', 'cpu_candidate_prefix_mapping.json', 'cpu_gate.json',
               'prepared_inputs.json', 't2g1_plans.json', 'workloads_case_C.json']
EXPECT_OBS = '7c1dcefcb5b456144a70908d7b48cba2b815584a5f0332c3e9f2586445392837'
EXPECT_RUNNER = 'b6cee2d3ffa04e8f85e9658bf166a37cfb063ad21983c9488b37fb8b4eaf9f84'
EXPECT_IMAGE = 'sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926'
EXPECT_E = {'fs': [0, 5120, 6144, 7168], 'mc': [1024] * 4, 'kv': [4032, 2048, 0, 0]}
EXPECT_COW = {'fs': [0, 1, 1, 1], 'mc': [4, 4, 4, 4], 'kv': [1, 1, 0, 0]}
EXPECT_J = {'fs': 154.34895561599998, 'mc': 703.45464601, 'kv': 627.6149768490118}
EXPECT_SEL = {'fs': {'C_B_P05120', 'C_C_P06144', 'C_D_P07168'},
              'mc': {'C_JOIN_P04032', 'C_JOIN_P04096', 'C_ROOT_P01024'},
              'kv': {'C_AB_P01984', 'C_AB_P02048', 'C_A_P04032'}}


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def median(v):
    return statistics.median(v)


def build(round_no, suffix):
    """读取单臂原始记录并做逐臂断言;返回复算指标字典。"""
    arm = 'm%d_ck3_%s' % (round_no, suffix)
    d = ARMS / arm
    run = d / 'run'
    summaries = sorted(run.glob('summary.json'))
    assert len(summaries) == 1, 'summary 不唯一:%s(%d 个)' % (arm, len(summaries))
    fails = sorted(p.name for p in run.glob('failure*'))
    assert not fails, '存在 failure 文件:%s %s' % (arm, fails)
    s = json.loads((run / 'summary.json').read_text())
    assert s['status'] == 'PASS' and s['policy'] == [p for p in POLICIES if POLICY_SUFFIX[p] == suffix][0]

    # —— 身份(以原始 identity 文件为准)——
    ident = json.loads((d / 'source_identity.json').read_text())
    hes = ident['hybrid_entry_sources']
    assert ident['image_id'] == EXPECT_IMAGE, '镜像漂移:' + arm
    assert hes['/home/wjg/code/FlowState/evaluation/p0_control_cycle_gpu/observation.py'] == EXPECT_OBS
    assert hes['/home/wjg/code/FlowState/evaluation/p0_control_cycle_gpu/runner.py'] == EXPECT_RUNNER

    # —— 正确性九项 ——
    rows = s['pending_rows']
    fz = json.loads((d / 't2g1_plans.json').read_text())['plans'][s['policy']]
    eg = fz['expected_E_G']
    assert all(r['cold_output_match'] for r in rows), 'cold 失配:' + arm
    assert [r['E'] for r in rows] == EXPECT_E[suffix], 'E 向量漂移:' + arm
    assert [r['cow_copy_count'] for r in rows] == EXPECT_COW[suffix], 'COW 漂移:' + arm
    for r in rows:
        exp = eg[r['pending_id']]
        assert (r['T'], r['G']) == (exp['T'], exp['G']), 'T/G 漂移:' + arm
        assert r['prefill_token_count'] == r['T'] - r['E'] + 32, 'prefill 物理不一致:' + arm
        assert r['batch_prefix_length'] == r['E'] and r['physical_fa_hit'] == r['T'], '前缀物理不一致:' + arm
    assert s['cycle']['identity_exact'], '计时恒等式不成立:' + arm
    assert set(s['selection']['selected_cpu']) == EXPECT_SEL[suffix], '选集(集合语义)漂移:' + arm
    assert s['selection']['plan_recovery_cost_ms'] == EXPECT_J[suffix], 'J 漂移:' + arm
    assert s['oracle_match'] is True, 'oracle 校验失败:' + arm
    tri = {w['window']: w['new_entries'] for w in s['triton_new_kernels']}
    assert tri.get('post_warm_block→post_request') == 0, '请求窗口存在新编译:' + arm
    assert s['census_counts']['unmapped_extras'] == 0, '存在未映射物理量:' + arm
    # 无未来信息泄漏:4 个请求全部在 T3(控制周期结束)之后才启动 → 选择不可能用到请求结果
    assert min(r['start_ns'] for r in rows) > s['cycle']['t3_ns'], '请求先于 T3 启动:' + arm
    # 在线内核身份(真实内核而非桩;三内核入口为实测值,见 m1 三臂 summary)
    kern = {'fs': 'flowstate.optimizer.GlobalOptimizer.plan_strict',
            'mc': 'evaluation.marconi_evict_t2m.evict_policy.MarconiEvictPolicy.select',
            'kv': 'evaluation.kvflow_inspired_retention_cpu.policy.select'}
    assert s['selection']['kernel_entry'] == kern[suffix], '内核入口异常:' + arm

    cyc = s['cycle']
    sel = s['selection']
    m = {
        'arm': arm, 'round': round_no, 'policy': s['policy'],
        'solver': sel['solver'], 'kernel_entry': sel['kernel_entry'],
        'selection_call_ms': sel['selection_call_ms'],
        'control_cycle_ms': cyc['control_cycle_ms'],
        'cycle_ms_full_incl_warm': cyc['cycle_ms_full_incl_warm'],
        'request_exec_total_ms': cyc['request_exec_total_ms'],
        'J_ms': sel['plan_recovery_cost_ms'],
        'sum_ttft_ms': sum(r['ttft_ms'] for r in rows),
        'sum_cuda_prefill_ms': sum(r['cuda_prefill_ms'] for r in rows),
        'sum_stream_tpot_ms': sum(r['stream_tpot_ms'] for r in rows),
        'selection_ids': sorted(sel['selected_cpu']),
        'pendings': [{
            'pending_id': r['pending_id'], 'E': r['E'], 'T': r['T'], 'G': r['G'],
            'prefill_token_count': r['prefill_token_count'],
            'batch_prefix_length': r['batch_prefix_length'], 'physical_fa_hit': r['physical_fa_hit'],
            'cow_copy_count': r['cow_copy_count'], 'ttft_ms': r['ttft_ms'],
            'ttft_check_ms': r['timing_decomposition']['ttft_ms_check'],
            'cuda_prefill_ms': r['cuda_prefill_ms'], 'stream_tpot_ms': r['stream_tpot_ms'],
            'request_wall_ms': r['request_wall_ms'],
        } for r in rows],
        # 开销分解原始字段
        'apply_stages': {st['stage']: st['ms'] for st in cyc['apply_stages_ms']},
        'apply_stage_gaps_sum_ms': sum(cyc['apply_stage_gaps_ms']),
        'rpc_tail_ms': cyc['rpc_tail_ms'], 'control_rpc_wall_ms': cyc['control_rpc_wall_ms'],
        'audit_entries_total': s['audit_entries_total'],
        'journal_after_window_flush_ms': s['journal_io']['after_window_flush_ms'],
        'journal_after_window_serialize_ms': s['journal_io']['after_window_serialize_ms'],
        'journal_in_window_serialize_ms': s['journal_io']['in_window_serialize_ms'],
        'warmup1_wall_ms': s['warmup1_wall_ms'], 'warmup2_wall_ms': s['warmup2_wall_ms'],
        'engine_boot_ms': s['engine_boot_ms'],
        'identity': {'image_id': ident['image_id'],
                     'runner': EXPECT_RUNNER, 'observation': EXPECT_OBS,
                     'runtime_sources': ident['runtime_sources'],
                     'input_sha': {f: sha256(d / f) for f in INPUT_FILES}},
    }
    return m


def main():
    data = {}       # (round, policy) -> metrics
    input_hashes = {}   # 输入哈希跨臂一致性:文件 -> {arm: sha}
    runtime_sources_variants = []
    for rnd in ROUNDS:
        for p in POLICIES:
            m = build(rnd, POLICY_SUFFIX[p])
            data[(rnd, p)] = m
            for f, h in m['identity']['input_sha'].items():
                input_hashes.setdefault(f, {})[m['arm']] = h
            runtime_sources_variants.append(json.dumps(m['identity']['runtime_sources'], sort_keys=True))
    # 五轮输入逐字节一致(15 臂 × 9 文件哈希两两相同),并与 r1 Pilot 对照
    for f, per_arm in input_hashes.items():
        assert len(set(per_arm.values())) == 1, '输入不一致:' + f
        r1 = ARMS / ('r1_ck3_%s' % POLICY_SUFFIX[p])  # 任一 r1 臂同名文件对照
        # 逐字节对照 r1 三臂
        for suf in ('fs', 'mc', 'kv'):
            assert sha256(ARMS / ('r1_ck3_%s' % suf) / f) == list(per_arm.values())[0], \
                '与 r1 Pilot 输入不一致:%s' % f
    assert len(set(runtime_sources_variants)) == 1, 'runtime_sources 跨臂漂移'

    # —— 总账交叉核对(原始账本,只读)——
    led = [json.loads(x) for x in (ARMS / 'five_round_ledger.jsonl').read_text().splitlines() if x.strip()]
    assert len(led) == 15 and all(r['status'] == 'PASS' for r in led), '总账非 15 行全 PASS'
    assert all(r['arm'].startswith('m') for r in led), '总账混入非 m* 臂'
    for r in led:
        m = data[(r['round'], r['policy'])]
        assert r['arm'] == m['arm'], '总账臂名不一致'
        assert abs(r['sum_ttft_ms'] - m['sum_ttft_ms']) < 0.01, '总账 ΣTTFT 与复算不一致:' + r['arm']
        assert abs(r['J_ms'] - m['J_ms']) < 1e-9, '总账 J 与复算不一致:' + r['arm']

    # —— TTFT 内部恒等(pending 行自带 ttft_ms_check)——
    for m in data.values():
        for pd_ in m['pendings']:
            assert abs(pd_['ttft_ms'] - pd_['ttft_check_ms']) < 1e-9, 'ttft 双记录不一致:' + m['arm']

    # —— 配对统计(逐轮差→中位;百分比显式分母)——
    paired = {}
    for base in ('Marconi-Evict', 'KVFlow-inspired'):
        key = 'FS-%s' % ('MC' if base.startswith('Marconi') else 'KV')
        d_sum, d_pre, d_tpot = [], [], []
        per_pending = {pd_id: [] for pd_id in ('C_A', 'C_B', 'C_C', 'C_D')}
        imp_round = []          # 逐轮改善%(分母=同轮基线 ΣTTFT)
        for rnd in ROUNDS:
            fs, b = data[(rnd, 'FlowState-Greedy')], data[(rnd, base)]
            d_sum.append(round(fs['sum_ttft_ms'] - b['sum_ttft_ms'], 3))
            d_pre.append(round(fs['sum_cuda_prefill_ms'] - b['sum_cuda_prefill_ms'], 3))
            d_tpot.append(round(fs['sum_stream_tpot_ms'] - b['sum_stream_tpot_ms'], 3))
            imp_round.append(round((fs['sum_ttft_ms'] - b['sum_ttft_ms']) / b['sum_ttft_ms'] * 100, 2))
            for a, c in zip(fs['pendings'], b['pendings']):
                assert a['pending_id'] == c['pending_id']
                per_pending[a['pending_id']].append(round(a['ttft_ms'] - c['ttft_ms'], 3))
        b_med = median([data[(r, base)]['sum_ttft_ms'] for r in ROUNDS])
        paired[key] = {
            'sum_ttft_diffs': d_sum, 'sum_ttft_diff_median': round(median(d_sum), 3),
            'sum_prefill_diffs': d_pre, 'sum_prefill_diff_median': round(median(d_pre), 3),
            'sum_tpot_diffs': d_tpot, 'sum_tpot_diff_median': round(median(d_tpot), 3),
            'per_pending_diffs': per_pending,
            'per_pending_median': {k: round(median(v), 3) for k, v in per_pending.items()},
            'per_round_improvement_pct': imp_round,                 # 分母=同轮基线
            'median_improvement_pct_round_first': round(median(imp_round), 2),
            'aggregate_improvement_pct': round(median(d_sum) / b_med * 100, 2),  # 分母=基线五轮中位
            'baseline_sum_ttft_median': round(b_med, 3),
        }

    # J vs 实测 Σprefill:两种分母显式给出
    j_vs = {}
    for base, key in (('Marconi-Evict', 'FS-MC'), ('KVFlow-inspired', 'FS-KV')):
        jd = EXPECT_J['fs'] - EXPECT_J[POLICY_SUFFIX[base]]          # J 差(恒定)
        real = [data[(r, 'FlowState-Greedy')]['sum_cuda_prefill_ms'] - data[(r, base)]['sum_cuda_prefill_ms'] for r in ROUNDS]
        j_vs[key] = {
            'J_diff': round(jd, 3), 'realized_diffs': [round(x, 3) for x in real],
            'realized_median': round(median(real), 3),
            # 分母一:|J 差| → 实测比模型小多少 %(保守度)
            'gap_over_J_pct_per_round': [round((abs(jd) - abs(x)) / abs(jd) * 100, 2) for x in real],
            'gap_over_J_pct_median': round(median([(abs(jd) - abs(x)) / abs(jd) * 100 for x in real]), 2),
            # 分母二:|实测差| → 模型比实测大多少 %
            'J_over_realized_pct_median': round(median([(abs(jd) - abs(x)) / abs(x) * 100 for x in real]), 2),
            'order_consistent_all_rounds': all(
                (EXPECT_J['fs'] < EXPECT_J[POLICY_SUFFIX[base]]) == (x < 0) for x in real),
        }

    # —— R3 敏感性(排除 R3 的四轮)——
    r3s = {}
    for key in ('FS-MC', 'FS-KV'):
        base = 'Marconi-Evict' if key == 'FS-MC' else 'KVFlow-inspired'
        ex3 = [round(data[(r, 'FlowState-Greedy')]['sum_ttft_ms'] - data[(r, base)]['sum_ttft_ms'], 3)
               for r in ROUNDS if r != 3]
        r3s[key] = {'with_R3_median': paired[key]['sum_ttft_diff_median'],
                    'excl_R3_diffs': ex3, 'excl_R3_median': round(median(ex3), 3)}

    # —— 40 格口径:逐策略×轮×pending 配对符号统计 ——
    cells = {}
    for key in ('FS-MC', 'FS-KV'):
        base = 'Marconi-Evict' if key == 'FS-MC' else 'KVFlow-inspired'
        pos = neg = 0
        pos_cells = []
        for rnd in ROUNDS:
            for a, c in zip(data[(rnd, 'FlowState-Greedy')]['pendings'], data[(rnd, base)]['pendings']):
                if a['ttft_ms'] - c['ttft_ms'] > 0:
                    pos += 1
                    pos_cells.append('R%d/%s' % (rnd, a['pending_id']))
                else:
                    neg += 1
        cells[key] = {'positive_cells': pos, 'negative_cells': neg, 'positive_where': pos_cells}

    # —— 控制开销分解(各级中位;未分解余量=control_cycle−Σstages−Σgaps)——
    overhead = {}
    for p in POLICIES:
        ms = [data[(r, p)] for r in ROUNDS]
        st_names = list(ms[0]['apply_stages'])
        ov = {'selection_call_ms': round(median([m['selection_call_ms'] for m in ms]), 3),
              'control_cycle_ms': round(median([m['control_cycle_ms'] for m in ms]), 3),
              'cycle_full_ms': round(median([m['cycle_ms_full_incl_warm'] for m in ms]), 3),
              'request_exec_total_ms': round(median([m['request_exec_total_ms'] for m in ms]), 3)}
        for st in st_names:
            ov['stage_' + st] = round(median([m['apply_stages'][st] for m in ms]), 3)
        ov['stage_gaps_sum_ms'] = round(median([m['apply_stage_gaps_sum_ms'] for m in ms]), 3)
        ov['unclassified_residual_ms'] = round(median(
            [m['control_cycle_ms'] - sum(m['apply_stages'].values()) - m['apply_stage_gaps_sum_ms'] for m in ms]), 3)
        overhead[p] = ov

    # —— 产出 CSV/JSON(一次性写入)——
    per_round_csv = OUT / 'gpu_per_round.csv'
    assert not per_round_csv.exists(), '拒绝覆盖:' + str(per_round_csv)
    with per_round_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['round', 'policy', 'arm', 'sum_ttft_ms', 'sum_cuda_prefill_ms', 'sum_stream_tpot_ms',
                    'J_ms', 'selection_call_ms', 'control_cycle_ms_T3_T0', 'cycle_ms_full_incl_warm_T4_T0',
                    'request_exec_total_ms', 'ttft_per_pending_ms', 'E_per_pending', 'solver', 'selection_ids'])
        for rnd in ROUNDS:
            for p in POLICIES:
                m = data[(rnd, p)]
                w.writerow([rnd, p, m['arm'], round(m['sum_ttft_ms'], 3), round(m['sum_cuda_prefill_ms'], 3),
                            round(m['sum_stream_tpot_ms'], 3), m['J_ms'], round(m['selection_call_ms'], 3),
                            m['control_cycle_ms'], m['cycle_ms_full_incl_warm'], m['request_exec_total_ms'],
                            '|'.join('%.3f' % x['ttft_ms'] for x in m['pendings']),
                            '|'.join(str(x['E']) for x in m['pendings']), m['solver'],
                            '|'.join(m['selection_ids'])])

    per_pending_csv = OUT / 'gpu_per_pending.csv'
    assert not per_pending_csv.exists(), '拒绝覆盖:' + str(per_pending_csv)
    with per_pending_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['round', 'policy', 'arm', 'pending_id', 'E_tokens', 'T_tokens', 'G_tokens',
                    'prefill_token_count', 'batch_prefix_length', 'physical_fa_hit', 'cow_copy_count',
                    'ttft_ms', 'cuda_prefill_ms', 'stream_tpot_ms', 'request_wall_ms',
                    'paired_diff_vs_FS_ttft_ms'])
        for rnd in ROUNDS:
            for p in POLICIES:
                m = data[(rnd, p)]
                for x, fsrow in zip(m['pendings'], data[(rnd, 'FlowState-Greedy')]['pendings']):
                    diff = '' if p == 'FlowState-Greedy' else round(x['ttft_ms'] - fsrow['ttft_ms'], 3)
                    w.writerow([rnd, p, m['arm'], x['pending_id'], x['E'], x['T'], x['G'],
                                x['prefill_token_count'], x['batch_prefix_length'], x['physical_fa_hit'],
                                x['cow_copy_count'], round(x['ttft_ms'], 3), round(x['cuda_prefill_ms'], 3),
                                round(x['stream_tpot_ms'], 3), round(x['request_wall_ms'], 3), diff])

    overhead_csv = OUT / 'gpu_overhead.csv'
    assert not overhead_csv.exists(), '拒绝覆盖:' + str(overhead_csv)
    with overhead_csv.open('w', newline='') as f:
        w = csv.writer(f)
        keys = list(overhead['FlowState-Greedy'])
        w.writerow(['policy'] + keys)
        for p in POLICIES:
            w.writerow([p] + [overhead[p][k] for k in keys])

    findings = {
        'arms_total': 15, 'all_pass': True,
        'identity': {'image_id': EXPECT_IMAGE, 'runner': EXPECT_RUNNER, 'observation': EXPECT_OBS,
                     'runtime_sources_files': len(data[(1, 'FlowState-Greedy')]['identity']['runtime_sources']),
                     'inputs_byte_identical_across_15_arms_and_r1': True,
                     'observation_note': '最终审计报告 §2 第 9 条曾误写 7c1dcef5…(应为 7c1dcefc…),以臂内 source_identity.json 为准',
                     'selection_note': '最终审计报告 §2 第 8 条把 MC/KV 选集误写为 {C_A..C_D}/{C_A,C_B};实测(以 t2g1_plans+15 臂 summary 为准)为 MC {C_JOIN_P04032,C_JOIN_P04096,C_ROOT_P01024}、KV {C_AB_P01984,C_AB_P02048,C_A_P04032},均 3 枚==K'},
        'leakage_check': '15 臂 4 请求 start_ns 全部 > t3_ns(选择先于全部请求);oracle_match=True;选集==冻结计划(集合语义)',
        'policy_metrics': {p: {k: round(median([data[(r, p)][k] for r in ROUNDS]), 3)
                               for k in ('sum_ttft_ms', 'sum_cuda_prefill_ms', 'sum_stream_tpot_ms',
                                         'selection_call_ms', 'control_cycle_ms', 'cycle_ms_full_incl_warm',
                                         'request_exec_total_ms')}
                           for p in POLICIES},
        'policy_metrics_minmax': {p: {k: [round(min(data[(r, p)][k] for r in ROUNDS), 3),
                                          round(max(data[(r, p)][k] for r in ROUNDS), 3)]
                                      for k in ('sum_ttft_ms', 'control_cycle_ms', 'cycle_ms_full_incl_warm')}
                                  for p in POLICIES},
        'paired': paired, 'J_vs_realized': j_vs, 'R3_sensitivity': r3s, 'cells_40': cells,
        'overhead': overhead,
    }
    fj = OUT / 'audit_findings_five_round.json'
    assert not fj.exists(), '拒绝覆盖:' + str(fj)
    fj.write_text(json.dumps(findings, ensure_ascii=False, indent=2))

    print('独立复算完成:15/15 臂断言全过(正确性九项/身份/输入逐字节/泄漏检查/总账交叉核对)')
    print('配对中位 ΣTTFT:FS-MC %.1f  FS-KV %.1f' % (paired['FS-MC']['sum_ttft_diff_median'],
                                                    paired['FS-KV']['sum_ttft_diff_median']))
    print('改善%%(逐轮先/分母=同轮基线):FS-MC %.2f%%  FS-KV %.2f%%' % (
        paired['FS-MC']['median_improvement_pct_round_first'], paired['FS-KV']['median_improvement_pct_round_first']))
    print('改善%%(聚合/分母=基线中位):FS-MC %.2f%%  FS-KV %.2f%%' % (
        paired['FS-MC']['aggregate_improvement_pct'], paired['FS-KV']['aggregate_improvement_pct']))
    print('Jvs实测(分母=|J差|):FS-MC 小 %.2f%%  FS-KV 小 %.2f%%' % (
        j_vs['FS-MC']['gap_over_J_pct_median'], j_vs['FS-KV']['gap_over_J_pct_median']))
    print('R3 敏感性:', json.dumps(r3s))
    print('40 格符号:', json.dumps(cells))
    print('开销分解中位:', json.dumps(overhead, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
