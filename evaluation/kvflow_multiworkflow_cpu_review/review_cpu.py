"""只读复核：C K=5 反事实双零集合、StandaloneTopK 对照、未来候选决策时合法性。

不修改任何冻结数据、不覆盖既有 CSV；复核结果单独保存在本目录。
三个检查均为只读计算，输入摘要先校验，任何变化立即终止。
"""
import csv, json, sys, hashlib
from pathlib import Path

MAIN = Path('/home/wjg/code/FlowState/evaluation/kvflow_multiworkflow_cpu_main')
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'evaluation/p1_strong_baseline_validation'))
import common  # noqa: E402
from flowstate.optimizer import GlobalOptimizer  # noqa: E402
from flowstate.state_catalog import CheckpointCandidate  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


# ── 第 0 步：只读校验冻结输入锁，摘要变化即终止 ──
lock = read(MAIN / 'input_lock.json')
for p, h in lock['sha256'].items():
    assert sha(Path(p)) == h, '冻结输入或策略源码摘要变化：' + p
results = read(MAIN / 'cpu_results.json')


# ── 与 run_cpu.py 逐字一致的 epoch 构造（只读复用，保证同口径） ──
def epoch(case, k, scope='current'):
    calls = case['calls']
    order = case['predeclared_evaluation_order']
    qs = [next(p for p in calls if p['continuation_id'] == i) for i in order]
    qs = [p for p in qs if scope == 'all' or (p['ready'] if scope == 'current' else not p['ready'])]
    ids = {p['continuation_id'] for p in qs}
    return common.Epoch(
        tuple(common.Candidate(c['checkpoint_id'], c['token_position_L'],
                               tuple(x for x in c['token_compatible_calls'] if x in ids))
              for c in case['candidates']),
        tuple(common.Pending(p['continuation_id'], p['planning_target_T']) for p in qs), k)


# 校准范围（与 run_cpu.py 同一来源，仅用于状态标注）
cal = list(csv.DictReader((ROOT / 'evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/calibration_points.csv').open()))
MIN_T = min(int(x['target_tokens']) for x in cal)
MAX_T = max(int(x['target_tokens']) for x in cal)
MAX_G = max(int(x['gap_tokens']) for x in cal)


def tag_pending(score):
    """为逐 Pending 结果补校准状态标注，规则与 run_cpu.py 相同。"""
    out = []
    for p in score['per_pending']:
        p = dict(p)
        p['calibration_status'] = ('EXACT_ZERO_BY_DEFINITION' if p['G'] == 0 else
                                   ('OUTSIDE_CALIBRATION_TARGET_RANGE' if not MIN_T <= p['T'] <= MAX_T
                                    else 'INSIDE_CALIBRATION_BOUNDS_NOT_POINTWISE_CERTIFIED'))
        assert p['G'] <= MAX_G and p['T'] <= common.MODEL.metadata.maximum_target_tokens
        out.append(p)
    return out


def greedy_selection(case, k):
    """按 run_cpu.py 同样方式复现 FlowState 边际贪心（用于交叉核对冻结输出）。"""
    ep = epoch(case, k)
    core = tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'], 1, True)
                 for c in case['candidates'])
    sel = GlobalOptimizer(common.MODEL)._select_candidates(
        core, k, lambda chosen: common.score(ep, tuple(c.checkpoint_id for c in chosen))['cost_ms'])
    return tuple(c.checkpoint_id for c in sel.selected)


def standalone_topk(case, k):
    """冻结 StandaloneTopK 定义：empty-set 独立收益降序、id 升序兜底、仅保留正收益，取前 K。"""
    ep = epoch(case, k)
    empty = common.score(ep, ())['cost_ms']
    ben = {c['checkpoint_id']: empty - common.score(ep, (c['checkpoint_id'],))['cost_ms']
           for c in case['candidates']}
    ranking = [cid for cid, b in sorted(ben.items(), key=lambda x: (-x[1], x[0])) if b > 1e-9]
    selected = tuple(ranking[:min(k, len(ranking))])
    return selected, ranking, ben


cases = {c['structure']: c for c in read(MAIN / 'workloads.json')['cases']}
C, B = cases['C'], cases['B']
out = {'input_lock_verified': True, 'calibration_bounds': {'min_T': MIN_T, 'max_T': MAX_T, 'max_G': MAX_G}}

# ══ 任务 1：C K=5 反事实集合 {四叶 + JOIN} 的双零检验 ══
CF = ['C_A_P04096', 'C_B_P05120', 'C_C_P06144', 'C_D_P07168', 'C_JOIN_P04096']
byid = {c['checkpoint_id']: c for c in C['candidates']}
cur = common.score(epoch(C, 5, 'current'), tuple(CF))
fut = common.score(epoch(C, 5, 'future'), tuple(CF))
fs_row = next(r for r in results['rows'] if r['structure'] == 'C' and r['K'] == 5 and r['policy'] == 'FlowState-Greedy')
out['task1_counterfactual_C_K5'] = {
    'set': CF,
    'all_ids_in_frozen_domain': all(cid in byid for cid in CF),
    'per_candidate': {cid: {'L': byid[cid]['token_position_L'],
                            'compat': byid[cid]['token_compatible_calls'],
                            'state_status': byid[cid]['state_status'],
                            'runtime_materialized': byid[cid]['runtime_materialized'],
                            'logical_cost': byid[cid]['logical_cost'],
                            'expected_independent_slot_cost': byid[cid]['expected_independent_slot_cost'],
                            'parent_not_in_set': (byid[cid]['parent_candidate'] not in CF)}
                      for cid in CF},
    'total_slot_cost': sum(byid[cid]['expected_independent_slot_cost'] for cid in CF),
    'budget_K': 5, 'within_budget': len(CF) <= 5,
    'current_J_ms': cur['cost_ms'], 'current_per_pending': tag_pending(cur),
    'future_J_ms': fut['cost_ms'], 'future_per_pending': tag_pending(fut),
    'dual_zero': abs(cur['cost_ms']) < 1e-9 and abs(fut['cost_ms']) < 1e-9,
    'flowstate_frozen_reference': {'selected': list(fs_row['selected']),
                                   'current_J_ms': fs_row['current_J_ms'],
                                   'future_static_J_ms': fs_row['future_static_J_ms']},
}

# ══ 任务 2：C K=3/5 上 StandaloneTopK 与 FlowState Greedy 对照 ══
t2 = {}
for k in (3, 5):
    st, ranking, ben = standalone_topk(C, k)
    gr = greedy_selection(C, k)
    fs_frozen = next(r for r in results['rows'] if r['structure'] == 'C' and r['K'] == k and r['policy'] == 'FlowState-Greedy')
    assert sorted(gr) == sorted(fs_frozen['selected']), '贪心复现与冻结输出不一致'
    sc_st = common.score(epoch(C, k, 'current'), st)
    sc_gr = common.score(epoch(C, k, 'current'), gr)
    t2[f'K={k}'] = {
        'standalone_selected': list(st), 'standalone_ranking_top8': ranking[:8],
        'standalone_benefits_ms': {cid: round(ben[cid], 4) for cid in ranking[:8]},
        'greedy_selected': list(gr),
        'same_set': sorted(st) == sorted(gr),
        'standalone_current_J_ms': sc_st['cost_ms'], 'greedy_current_J_ms': sc_gr['cost_ms'],
        'J_diff_standalone_minus_greedy_ms': sc_st['cost_ms'] - sc_gr['cost_ms'],
        'standalone_per_pending': tag_pending(sc_st), 'greedy_per_pending': tag_pending(sc_gr),
        'standalone_slots_used': len(st), 'greedy_slots_used': len(gr),
        'greedy_matches_frozen': True,
    }
out['task2_standalone_vs_greedy_C'] = t2

# ══ 任务 3：未来候选（B 的 S1、C 的 JOIN）决策时合法性分级 ══
t3 = {'per_structure': {}}
for s in ('A', 'B', 'C'):
    case = cases[s]
    cand_fields = [{k: c[k] for k in ('checkpoint_id', 'state_status', 'runtime_materialized',
                                      'runtime_handle', 'control_domain', 'logical_cost',
                                      'expected_independent_slot_cost')}
                   for c in case['candidates']]
    events = case['synthetic_access_events']
    future_ids = [p['continuation_id'] for p in case['calls'] if not p['ready']]
    future_events = [e for e in events if e['checkpoint_id'].split('_P0')[0].replace('_P', '') and
                     any(cid.startswith(f + '_P') for f in future_ids for cid in [e['checkpoint_id']])]
    t3['per_structure'][s] = {
        'candidate_count': len(case['candidates']),
        'state_status_values': sorted({c['state_status'] for c in case['candidates']}),
        'runtime_materialized_values': sorted({c['runtime_materialized'] for c in case['candidates']}),
        'runtime_handle_values': sorted({json.dumps(c['runtime_handle']) for c in case['candidates']}),
        'control_domain_values': sorted({c['control_domain'] for c in case['candidates']}),
        'calls_runtime_received': sorted({p['runtime_received'] for p in case['calls']}),
        'runtime_actual_candidates': case['runtime_actual_candidates'],
        'common_retained_state_ids': case['common_retained_state_ids'],
        'access_event_count': len(events),
        'access_events_runtime_observed': sorted({e['runtime_observed'] for e in events}),
        'planned_warmup_rounds': sorted({w['request_id'].rsplit('_', 2)[0] + '_' + w['request_id'].rsplit('_', 2)[1] for w in case['planned_warmup']}),
        'planned_warmup_request_count': len(case['planned_warmup']),
        'future_call_ids': future_ids,
        'future_state_access_events': [{'event_order': e['event_order'], 'checkpoint_id': e['checkpoint_id'],
                                        'runtime_observed': e['runtime_observed']} for e in future_events],
        'candidate_fields_sample': cand_fields[:2],
    }
# LRU 证据：C 的 JOIN 访问事件是否排在最末（最近），从而被 LRU 优先保留
c_ev = C['synthetic_access_events']
join_events = [e for e in c_ev if e['checkpoint_id'].startswith('C_JOIN')]
lru_c_k3 = next(r for r in results['rows'] if r['structure'] == 'C' and r['K'] == 3 and r['policy'] == 'LRU')
t3['lru_evidence_C'] = {
    'join_event_orders': [e['event_order'] for e in join_events],
    'max_event_order': max(e['event_order'] for e in c_ev),
    'lru_C_K3_selected': list(lru_c_k3['selected']),
    'note': 'JOIN 的合成访问事件排在声明序列最末（=最近访问），LRU 按此将其排在保留首位；该事件 runtime_observed=False，对应调用 ready=False 且从未执行',
}
out['task3_decision_time_legality'] = t3

assert not (OUT / 'review_results.json').exists(), '复核结果已存在，禁止覆盖'
with (OUT / 'review_results.json').open('x') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)

# 控制台摘要（供报告引用）
t1 = out['task1_counterfactual_C_K5']
print('任务1: 域内=', t1['all_ids_in_frozen_domain'], '槽位=', t1['total_slot_cost'],
      'CurrentJ=', round(t1['current_J_ms'], 6), 'FutureJ=', round(t1['future_J_ms'], 6), '双零=', t1['dual_zero'])
for p in t1['current_per_pending'] + t1['future_per_pending']:
    print('  ', p['continuation_id'], 'E=', p['E'], 'G=', p['G'], p['calibration_status'])
for k, v in t2.items():
    print(f"任务2 {k}: 同集={v['same_set']} StandaloneJ={v['standalone_current_J_ms']:.4f} GreedyJ={v['greedy_current_J_ms']:.4f} "
          f"差={v['J_diff_standalone_minus_greedy_ms']:+.4f} 槽位={v['standalone_slots_used']}/{v['greedy_slots_used']}")
    print('   standalone排序前8:', v['standalone_ranking_top8'])
    print('   standalone收益:', v['standalone_benefits_ms'])
for s, v in t3['per_structure'].items():
    print(f"任务3 {s}: status={v['state_status_values']} materialized={v['runtime_materialized_values']} "
          f"events_observed={v['access_events_runtime_observed']} calls_received={v['calls_runtime_received']} warmup轮={v['planned_warmup_rounds']}")
print('任务3 LRU证据:', t3['lru_evidence_C']['join_event_orders'], 'max=', t3['lru_evidence_C']['max_event_order'],
      'LRU选中=', t3['lru_evidence_C']['lru_C_K3_selected'])
print('复核完成：输入锁校验通过，结果写入 review_results.json')
