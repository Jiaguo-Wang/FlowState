"""先复现原预算，再固定全部输入扫描K；只执行CPU真实策略内核。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[1]
NAT = ROOT / 'evaluation/rq3_workload_structure/natural_analysis'
P1 = ROOT / 'evaluation/p1_strong_baseline_validation'
sys.path[:0] = [str(ROOT), str(P1), str(NAT / 'scripts')]
from flowstate.optimizer import GlobalOptimizer, _exact_admission
from flowstate.recovery_model import RecoveryCostModel
import common
import pcb
import ldr
import wfc
from analyze import cover_certificate

EPS = 1e-9
POLICIES = ('FlowState', 'PCB', 'LDR', 'WFC')
ALL = POLICIES + ('Exact Oracle',)


def enc(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    need(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= EPS, f'数值不一致：{a} 与 {b}')


def phi(gap, target):
    need(0 <= gap <= target <= 131072, '目标或恢复缺口不合法')
    g, t = gap / 1024, target / 1024
    return 37.828150 * g + 0.345974143 * g * t - 0.156201917 * g * g


def put(path, x):
    with path.open('x') as f:
        f.write(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    need(bool(rows), '结果表为空')
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: enc(v) if isinstance(v, (list, dict, tuple)) else 'N/A' if v is None else v
                        for k, v in row.items()})


def csv_read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class KernelCandidate:
    """只暴露生产Greedy循环真实使用的三个字段；不捏造Workflow或谱系。"""
    checkpoint_id: str
    memory_bytes: int
    recurrent_resident: bool = True


class Instance:
    """固定快照输入，预算仅在调用时提供。"""
    def __init__(self, row, matrix):
        self.population = row['population']
        self.sid = row['snapshot_id']
        self.ids = tuple(json.loads(row['candidate_ids']))
        self.positions = tuple(json.loads(row['positions']))
        self.pids = tuple(json.loads(row['pending_ids']))
        self.targets = tuple(json.loads(row['targets']))
        self.support = json.loads(row['positive_supports'])
        self.index = dict(zip(self.ids, range(len(self.ids))))
        self.byte_size = matrix['checkpoint_bytes'][0]
        need(len(set(matrix['checkpoint_bytes'])) == 1, '仅支持原等大小预算')
        self.compat = tuple(frozenset(self.pids[p] for p in range(len(self.pids))
                                     if matrix['compatibility'][matrix['pending_ids'].index(self.pids[p])]
                                     [matrix['candidate_ids'].index(c)]) for c in self.ids)
        self.epoch_candidates = tuple(common.Candidate(c, l, tuple(sorted(s)))
                                      for c, l, s in zip(self.ids, self.positions, self.compat))
        self.pendings = tuple(common.Pending(p, t) for p, t in zip(self.pids, self.targets))
        self.kernel_candidates = tuple(KernelCandidate(c, self.byte_size) for c in self.ids)
        self.optimizer = GlobalOptimizer(RecoveryCostModel())
        self.input_sha = row['input_without_budget_sha256']
        self.row = row
        comp = [{p for p, pid in enumerate(self.pids) if pid in s} for s in self.compat]
        self.full_frontiers, self.kmin, _, _, _, _ = cover_certificate(self.positions, comp, self.targets)
        need(self.kmin == int(row['K_min_frontier']), '独立K_min与自然审计不同')
        b = [[phi(t, t) - phi(t - l, t) if pid in s else 0.0
              for pid, t in zip(self.pids, self.targets)] for l, s in zip(self.positions, self.compat)]
        for i, c in enumerate(self.ids):
            for p, pid in enumerate(self.pids):
                close(b[i][p], matrix['benefit_ms'][matrix['pending_ids'].index(pid)][matrix['candidate_ids'].index(c)])
        self.shared = any(sum(v > EPS for v in vv) > 1 for vv in b)
        invariant = {'ids': list(self.ids), 'positions': list(self.positions), 'pending_ids': list(self.pids),
                     'targets': list(self.targets), 'compatibility': [sorted(s) for s in comp],
                     'benefits': b, 'sizes': matrix['checkpoint_bytes']}
        need(hashlib.sha256(enc(invariant).encode()).hexdigest() == self.input_sha, '去预算输入摘要不一致')

    def evaluate(self, ids):
        need(len(ids) == len(set(ids)) and set(ids) <= set(self.ids), '选集重复或超出候选域')
        es = tuple(max([self.positions[self.index[c]] for c in ids if pid in self.compat[self.index[c]]] + [0])
                   for pid in self.pids)
        gs = tuple(t - e for t, e in zip(self.targets, es))
        costs = tuple(phi(g, t) for g, t in zip(gs, self.targets))
        return sum(costs), es, gs, costs

    def call(self, policy, k):
        """调用已有策略；FlowState仅接原Greedy核心，不经过Strict混合路由。"""
        if policy == 'FlowState':
            allocation = self.optimizer._select_candidates(
                self.kernel_candidates, k * self.byte_size,
                lambda selected: self.evaluate(tuple(c.checkpoint_id for c in selected))[0])
            return tuple(c.checkpoint_id for c in allocation.selected)
        epoch = common.Epoch(self.epoch_candidates, self.pendings, k)
        return {'PCB': pcb, 'LDR': ldr, 'WFC': wfc}[policy].select(epoch)[0]

    def exact(self, k):
        """仅在既定门禁内枚举全部可行子集；不把证书交给选择策略。"""
        allowed, gate = _exact_admission(len(self.ids), len(self.pids), k)
        need(allowed, 'Exact超出门禁，禁止枚举')
        choices = [(self.evaluate(s)[0], s) for size in range(min(k, len(self.ids)) + 1)
                   for s in itertools.combinations(sorted(self.ids), size)]
        need(len(choices) == gate['enumerable_subsets'] <= 794, 'Exact枚举范围不一致')
        minimum = min(v for v, _ in choices)
        opt = [(v, s) for v, s in choices if abs(v - minimum) <= EPS]
        selected = min((s for _, s in opt), key=lambda s: (len(s), s))
        return selected, minimum, len(opt), len(choices)


def percentile(vs, q):
    vs = sorted(vs)
    z = (len(vs) - 1) * q
    a, b = math.floor(z), math.ceil(z)
    return vs[a] + (z - a) * (vs[b] - vs[a])


def main():
    parser = argparse.ArgumentParser(description='原预算复现通过后才扫描K1至4；只使用CPU')
    parser.add_argument('--out', type=Path, default=BASE)
    args = parser.parse_args()
    out = args.out.resolve()
    need(out == BASE or BASE in out.parents, '新增结果只能写入本轮独立目录')
    out.mkdir(parents=True, exist_ok=True)
    need(not any(out.glob('*.json')) and not any(out.glob('*.csv')) and not any(out.glob('*.md')),
         '拒绝覆盖既有实验目录；请指定新的独立输出子目录')
    phase = 'source_preflight'
    checks = []
    files = {}

    def source(path, role):
        p = Path(path).resolve()
        files[str(p)] = {'path': str(p), 'role': role, 'sha256_before': sha(p), 'bytes': p.stat().st_size}
        return p

    def check(name, cond):
        checks.append({'check': name, 'pass': bool(cond)})
        need(cond, name)

    try:
        protocol = {'date': '2026-10-30', 'scope': '冻结Formal输入，CPU离线预算敏感性；不是Strict GPU',
                    'K_values': [1, 2, 3, 4], 'policies': POLICIES,
                    'flowstate_mode': '原Marginal Greedy核心，不调用plan_strict混合入口',
                    'exact_gate': {'n_max': 12, 'pending_max': 4, 'K_max': 4, 'subsets_max': 794},
                    'epsilon_ms': EPS, 'exact_tie_rule': '最优容差1e-9内最少项，再按ID字典序',
                    'timing': {'clock': 'perf_counter_ns', 'warmups_per_condition_policy': 3,
                               'measurement_batches': 15, 'calls_per_batch': 5,
                               'policy_order': '按快照预算序号加测量批次轮转',
                               'quantile': '15个批均单次时间的中位与线性插值P95，不是单调用或线上P95',
                               'included': '真实选择函数及其内部诊断轨迹构建、评分和返回',
                               'excluded': '文件读取、JSON解析、输入对象准备、外部结果校验、报告生成'}}
        put(out / 'protocol.json', protocol)
        natural = json.loads(source(NAT / 'source_manifest.json', '前一阶段读取源与内容锁').read_text())
        for item in natural['source_files']:
            p = source(item['path'], '自然审计原始来源复核')
            check('自然来源锁：' + str(p), files[str(p)]['sha256_before'] == item['sha256_after'])
        for name in ['natural_workload_per_condition.csv', 'natural_workload_structure_summary.csv',
                     'strategy_agreement_analysis.csv', 'budget_replacement_analysis.csv', 'source_manifest.md', 'scripts/README.md']:
            source(NAT / name, '自然审计交接产物')
        for name in ['flowstate/optimizer.py', 'flowstate/recovery_model.py',
                     'evaluation/rq3_frozen_snapshot_evaluator.py', 'evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md']:
            source(ROOT / name, '策略调用链及冻结公式')
        source(__file__, '本轮实验脚本')
        lock = json.loads(source(P1 / 'implementation_lock.json', '原策略实现锁').read_text())
        for name, expected in lock.items():
            p = source(P1 / name, '原PCB/LDR/WFC及输入合同实现')
            check('策略实现锁：' + name, sha(p) == expected)
        frozen = [json.loads(x) for x in source(P1 / 'per_case_results.jsonl', '原573条件与四策略冻结输出').read_text().splitlines()]
        rows = csv_read(NAT / 'natural_workload_per_condition.csv')
        matrices = {v['instance_id']: v for v in map(json.loads, (ROOT / 'evaluation/gate1_step1_allocation_equivalence/matrices.jsonl').read_text().splitlines())}
        originals = {}
        for path in {r['source_file'] for r in frozen}:
            originals[path] = [json.loads(x) for x in source(path, '原始逐预算评估').read_text().splitlines()]
        instances, by_condition, fs_raw_order = {}, {}, {}
        original_index = {(r['population'], r['snapshot_id'], float(r['budget_ratio'])): r for r in frozen}
        for r in rows:
            sid = r['snapshot_id']
            x = Instance(r, matrices[r['condition_id']])
            if sid in instances:
                check('三预算输入一致：' + r['condition_id'], instances[sid].input_sha == x.input_sha)
            else:
                instances[sid] = x
            by_condition[r['condition_id']] = r
            saved = original_index[(r['population'], sid, float(r['budget_ratio']))]
            raw = originals[saved['source_file']][saved['source_line'] - 1]
            check('原始评估行身份：' + r['condition_id'], raw['snapshot_id'] == sid and raw['k'] == int(r['K'])
                  and raw['snapshot_digest'] == saved['snapshot_digest'])
            fs_raw_order[r['condition_id']] = raw['mechanism_diagnostics']['flowstate_greedy_trace']['selection_order']
        check('唯一Snapshot域完整', len(instances) == 191 and Counter(x.population for x in instances.values()) == {'OpenHands': 168, 'AgentX': 23})
        environment = {'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                       'cpu_model': next((v.split(':', 1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')), 'UNKNOWN'),
                       'cpu_affinity': sorted(os.sched_getaffinity(0)), 'timer': str(time.get_clock_info('perf_counter')),
                       'solver_entry': 'flowstate.optimizer.GlobalOptimizer._select_candidates',
                       'strict_entry_separate': 'plan_strict先验证Strict Epoch，再按12/4/4/794门禁选择Exact或Greedy；本次不是此路径'}
        put(out / 'environment.json', environment)
        phase = 'original_573_reproduction'
        reproduced = []
        for saved in frozen:
            iid = f"{saved['snapshot_id']}::budget={saved['budget_ratio']}"
            x = instances[saved['snapshot_id']]
            for policy in POLICIES:
                selected = x.call(policy, saved['k'])
                result = x.evaluate(selected)
                ref = saved['policies'][policy]
                check('复现选集：' + iid + '/' + policy, set(selected) == set(ref['selected']) and len(selected) == len(ref['selected']))
                close(result[0], ref['cost_ms'])
                check('复现E/G：' + iid + '/' + policy, list(result[1]) == [p['E'] for p in ref['per_pending']]
                      and list(result[2]) == [p['G'] for p in ref['per_pending']])
                for cost, p in zip(result[3], ref['per_pending']):
                    close(cost, p['cost_ms'])
                if policy == 'FlowState':
                    check('复现原Greedy选择顺序：' + iid, list(selected) == fs_raw_order[iid])
                reproduced.append({'condition_id': iid, 'population': x.population, 'snapshot_id': x.sid, 'K': saved['k'],
                                   'policy': policy, 'selected': selected, 'selected_count': len(selected), 'J_ms': result[0],
                                   'E': result[1], 'G': result[2], 'same_set': True, 'same_J': True,
                                   'same_order': list(selected) == ref['selected'], 'input_sha256': x.input_sha})
        check('573条件四策略复现通过', len(reproduced) == 2292)
        csv_write(out / 'original_budget_reproduction.csv', reproduced)
        put(out / 'reproduction_gate.json', {'status': 'PASS', 'conditions': 573, 'policy_evaluations': 2292,
                                            'set_count_E_G_J': '全通过', 'FlowState_original_step_order': '573/573一致'})
        print('第一道门禁PASS：573条件、2292策略结果均复现，开始固定K扫描', flush=True)
        phase = 'budget_scan'
        strategy_rows, pending_rows, oracle_rows, condition_rows, raw_timings, divergence = [], [], [], [], [], []
        computed = {}
        for ordinal, x in enumerate(instances.values()):
            for k in [1, 2, 3, 4]:
                allowed, gate = _exact_admission(len(x.ids), len(x.pids), k)
                entries = {}
                oracle = x.exact(k) if allowed else None
                if oracle:
                    close(x.evaluate(oracle[0])[0], oracle[1])
                oracle_rows.append({'population': x.population, 'snapshot_id': x.sid, 'K': k,
                                    'input_sha256': x.input_sha, 'n': len(x.ids), 'm': len(x.pids), 'K_min': x.kmin,
                                    'shared': x.shared, 'status': 'PASS' if allowed else 'N/A_SIZE_GATE',
                                    'gate': gate, 'selected': oracle[0] if oracle else None, 'J_ms': oracle[1] if oracle else None,
                                    'optimal_set_count_within_tolerance': oracle[2] if oracle else None,
                                    'enumerated_subsets': oracle[3] if oracle else 0})
                for policy in ALL:
                    if policy == 'Exact Oracle' and not allowed:
                        entries[policy] = None
                        continue
                    selected = oracle[0] if policy == 'Exact Oracle' else x.call(policy, k)
                    need(len(selected) <= k, '逻辑预算超限')
                    j, es, gs, costs = x.evaluate(selected)
                    gap = j - oracle[1] if oracle else None
                    if gap is not None:
                        need(gap >= -EPS, '策略成本低于完整枚举最优值')
                    entries[policy] = {'selected': tuple(selected), 'J': j, 'E': es, 'G': gs, 'costs': costs, 'gap': gap}
                active = [policy for policy in ALL if entries.get(policy) is not None]
                timers = {policy: [] for policy in active}

                def invoke(policy):
                    return x.exact(k)[0] if policy == 'Exact Oracle' else x.call(policy, k)

                for policy in active:
                    for _ in range(3):
                        need(tuple(invoke(policy)) == entries[policy]['selected'], '预热结果不确定')
                for rep in range(15):
                    shift = (ordinal * 4 + k + rep) % len(active)
                    order = active[shift:] + active[:shift]
                    for policy in order:
                        t0 = time.perf_counter_ns()
                        values = [invoke(policy) for _ in range(5)]
                        elapsed = time.perf_counter_ns() - t0
                        need(all(tuple(v) == entries[policy]['selected'] for v in values), '重复计时选择结果不确定')
                        per_call = elapsed / 5 / 1e6
                        timers[policy].append(per_call)
                        raw_timings.append({'population': x.population, 'snapshot_id': x.sid, 'K': k,
                                            'policy': policy, 'batch_index': rep + 1, 'calls': 5,
                                            'elapsed_batch_ns': elapsed, 'mean_per_call_ms': per_call})
                for policy in ALL:
                    v = entries.get(policy)
                    common_row = {'population': x.population, 'snapshot_id': x.sid, 'K': k, 'input_sha256': x.input_sha,
                                  'candidate_count': len(x.ids), 'pending_count': len(x.pids), 'K_min': x.kmin,
                                  'shared': x.shared, 'policy': policy, 'evidence': 'FROZEN_FORMAL_CPU_NOT_RUNTIME'}
                    strategy_rows.append({**common_row, 'status': 'PASS' if v else 'N/A_SIZE_GATE',
                                          'selected': v['selected'] if v else None, 'selected_count': len(v['selected']) if v else None,
                                          'J_ms': v['J'] if v else None, 'oracle_gap_ms': v['gap'] if v else None,
                                          'E': v['E'] if v else None, 'G': v['G'] if v else None,
                                          'solve_median_ms': statistics.median(timers[policy]) if v else None,
                                          'solve_P95_batchmean_ms': percentile(timers[policy], .95) if v else None})
                    if v:
                        for pid, t, e, g, cost in zip(x.pids, x.targets, v['E'], v['G'], v['costs']):
                            pending_rows.append({**common_row, 'pending_id': pid, 'T': t, 'E': e, 'G': g, 'cost_ms': cost,
                                                 'selected': v['selected'], 'aggregate_J_ms': v['J']})
                compare = [entries[p] for p in POLICIES]
                same_set = len({tuple(sorted(v['selected'])) for v in compare}) == 1
                same_j = max(v['J'] for v in compare) - min(v['J'] for v in compare) <= EPS
                condition_rows.append({'population': x.population, 'snapshot_id': x.sid, 'K': k, 'input_sha256': x.input_sha,
                                       'candidate_count': len(x.ids), 'pending_count': len(x.pids), 'K_min': x.kmin,
                                       'shared': x.shared, 'tight_frontier_budget': k < x.kmin,
                                       'same_set_all_four': same_set, 'same_J_all_four': same_j,
                                       'oracle_status': 'PASS' if allowed else 'N/A_SIZE_GATE',
                                       'selections': {p: entries[p]['selected'] for p in POLICIES},
                                       'J_by_policy': {p: entries[p]['J'] for p in POLICIES},
                                       'source_file': x.row['source_file'], 'source_line': x.row['source_line'],
                                       'snapshot_digest': x.row['snapshot_digest']})
                for baseline in ['PCB', 'LDR', 'WFC']:
                    fs, b = entries['FlowState'], entries[baseline]
                    if set(fs['selected']) != set(b['selected']) or abs(fs['J'] - b['J']) > EPS:
                        divergence.append({'population': x.population, 'snapshot_id': x.sid, 'K': k,
                                           'input_sha256': x.input_sha, 'candidate_count': len(x.ids), 'pending_count': len(x.pids),
                                           'K_min': x.kmin, 'shared': x.shared, 'baseline': baseline,
                                           'FlowState_selected': fs['selected'], 'baseline_selected': b['selected'],
                                           'FlowState_J_ms': fs['J'], 'baseline_J_ms': b['J'], 'baseline_minus_FlowState_ms': b['J'] - fs['J'],
                                           'FlowState_E': fs['E'], 'baseline_E': b['E'], 'FlowState_G': fs['G'], 'baseline_G': b['G'],
                                           'oracle_gap_FlowState_ms': fs['gap'], 'oracle_gap_baseline_ms': b['gap'],
                                           'same_J': abs(fs['J'] - b['J']) <= EPS})
                computed[(x.sid, k)] = entries
            if (ordinal + 1) % 25 == 0:
                print(f'固定输入扫描已完成 {ordinal + 1}/191 个快照', flush=True)
        phase = 'final_validation'
        replacements = []
        for x in instances.values():
            for k in [1, 2, 3]:
                for policy in ALL:
                    a, b = computed[(x.sid, k)].get(policy), computed[(x.sid, k + 1)].get(policy)
                    if not a or not b:
                        continue
                    sa, sb = set(a['selected']), set(b['selected'])
                    replacements.append({'population': x.population, 'snapshot_id': x.sid, 'policy': policy,
                                         'input_sha256': x.input_sha, 'K_low': k, 'K_high': k + 1,
                                         'selected_low': sorted(sa), 'selected_high': sorted(sb), 'removed': sorted(sa - sb),
                                         'added': sorted(sb - sa), 'nested': sa <= sb, 'J_low_ms': a['J'], 'J_high_ms': b['J'],
                                         'J_improvement_ms': a['J'] - b['J'], 'different_set_equal_J': sa != sb and abs(a['J'] - b['J']) <= EPS,
                                         'E_low': a['E'], 'E_high': b['E'], 'G_low': a['G'], 'G_high': b['G']})
        summary = []
        for pop in ['OpenHands', 'AgentX', 'ALL']:
            for k in [1, 2, 3, 4]:
                group = [r for r in condition_rows if r['K'] == k and (pop == 'ALL' or r['population'] == pop)]
                summary.append({'population': pop, 'K': k, 'conditions': len(group),
                                'all_four_same_set': sum(r['same_set_all_four'] for r in group),
                                'all_four_same_J': sum(r['same_J_all_four'] for r in group),
                                'shared_conditions': sum(r['shared'] for r in group),
                                'tight_frontier_budget_conditions': sum(r['tight_frontier_budget'] for r in group),
                                'oracle_eligible': sum(r['oracle_status'] == 'PASS' for r in group),
                                'policy_oracle_optimal_counts': {p: sum(computed[(r['snapshot_id'], k)][p]['gap'] is not None and abs(computed[(r['snapshot_id'], k)][p]['gap']) <= EPS for r in group) for p in POLICIES},
                                'FlowState_W_T_L': {p: {'win': sum(computed[(r['snapshot_id'], k)][p]['J'] - computed[(r['snapshot_id'], k)]['FlowState']['J'] > EPS for r in group),
                                                       'tie': sum(abs(computed[(r['snapshot_id'], k)][p]['J'] - computed[(r['snapshot_id'], k)]['FlowState']['J']) <= EPS for r in group),
                                                       'loss': sum(computed[(r['snapshot_id'], k)][p]['J'] - computed[(r['snapshot_id'], k)]['FlowState']['J'] < -EPS for r in group)} for p in ['PCB', 'LDR', 'WFC']}})
        check('完成191乘4条件', len(condition_rows) == 764)
        check('全部策略及门禁占位记录完整', len(strategy_rows) == 3820 and len(oracle_rows) == 764)
        for item in files.values():
            item['sha256_after'] = sha(item['path'])
            check('源文件保持不变：' + item['path'], item['sha256_before'] == item['sha256_after'])
        for name, table in [('budget_sensitivity_per_snapshot.csv', condition_rows),
                            ('budget_sensitivity_strategy_results.csv', strategy_rows),
                            ('budget_sensitivity_pending_results.csv', pending_rows),
                            ('budget_sensitivity_summary.csv', summary),
                            ('budget_replacement_analysis.csv', replacements),
                            ('strategy_divergence_cases.csv', divergence), ('exact_oracle_results.csv', oracle_rows),
                            ('solver_timing_raw.csv', raw_timings)]:
            csv_write(out / name, table)
        put(out / 'source_manifest.json', {'files': list(files.values()), 'natural_manifest': str(NAT / 'source_manifest.json'),
                                           'deduplicated_snapshots': {sid: {'input_sha256': x.input_sha, 'population': x.population,
                                                                           'source_snapshot_file': x.row['source_snapshot_file'],
                                                                           'state_identity': json.loads(x.row['candidate_node_map'])} for sid, x in instances.items()},
                                           'data_level': '旧Formal逻辑实例，不是Strict GPU认证'})
        text = '# 来源与内容身份\n\n固定全部候选、Pending、兼容、T/L及Phi，仅改变K。原始573条件复现通过后才扫描。\n\n|路径|用途|字节数|前后相同SHA256|\n|---|---|---:|---|\n'
        for r in files.values():
            text += f"|{r['path']}|{r['role']}|{r['bytes']}|`{r['sha256_before']}`|\n"
        with (out / 'source_manifest.md').open('x') as f:
            f.write(text)
        put(out / 'validation.json', {'status': 'PASS', 'phase': 'complete', 'original_reproduction_conditions': 573,
                                      'original_policy_evaluations': 2292, 'scan_snapshots': 191, 'scan_conditions': 764,
                                      'oracle_eligible_conditions': sum(r['status'] == 'PASS' for r in oracle_rows),
                                      'oracle_NA_conditions': sum(r['status'] != 'PASS' for r in oracle_rows),
                                      'pending_rows': len(pending_rows), 'timing_batches': len(raw_timings), 'checks': checks,
                                      'gpu_started': False, 'scope': '仅固定Formal矩阵的CPU结果；非物理执行兼容认证'})
        print(enc({'status': 'PASS', 'output': str(out), 'conditions': len(condition_rows),
                   'oracle_eligible': sum(r['status'] == 'PASS' for r in oracle_rows), 'divergence_rows': len(divergence)}), flush=True)
    except Exception as exc:
        if not (out / 'validation.json').exists():
            put(out / 'validation.json', {'status': 'BLOCKED', 'phase': phase, 'error': repr(exc),
                                          'traceback': traceback.format_exc(), 'checks': checks, 'gpu_started': False})
        raise


if __name__ == '__main__':
    main()
