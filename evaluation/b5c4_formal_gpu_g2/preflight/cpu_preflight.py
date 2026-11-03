"""G2 宿主机 CPU 预检（不触 GPU）：两求解臂在 B5-C4 冻结 Epoch 上与 e3 参照全精度一致。

验证链（全部只读）：
1. 基线固定：b5d3 strict/reconciliation.json 的 epoch/raw_epoch（B5-C4，7 候选/4 Pending/K=2）；
   origins 由 b5d4 历史 state 重推（与 b5d4a test_regression 同法）；
2. Exact 臂：生产 GlobalOptimizer.plan_strict 在线求解 → 决策记录应为门禁内自动 Exact，
   选集/J/逐Pending E-T-G 与 preflight/g2_reference.json（e3 全精度抽取）逐项相等；
3. Greedy 臂：pkg_g2.greedy_adapter.plan_greedy_strict（生产 Marginal Greedy 内部路径）
   → 选集/J/逐Pending 与 e3 Greedy 参照逐项相等；validate_plan 通过；
   _exact_admission(7,4,2) 允许（证明 Greedy 臂未通过篡改规模进入 Greedy）；
4. 来源合同：G2_PENDING 4 取值 × 2 求解臂共 8 组合，choose_source 均返回唯一来源
   （Exact: A/B→N18, C/D→N27；Greedy: A/B→N18, C/D→N20），全部 PROVENANCE_UNVERIFIED。
"""
import copy, hashlib, json, os, sys
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
sys.path.insert(0, str(ROOT))
from flowstate.optimizer import GlobalOptimizer, _exact_admission
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.strict_epoch import StrictAllocationEpoch, StrictCandidate, StrictEdge, StrictPending
from flowstate.strict_control import validate_plan
from evaluation.b5c4_formal_gpu_g2.pkg_g2.greedy_adapter import plan_greedy_strict

B5D3 = ROOT / 'evaluation/exact_reconciliation_correctness_gate_b5d3/strict/reconciliation.json'
B5D4 = ROOT / 'evaluation/exact_greedy_paired_gpu_gate_b5d4'
OUT = ROOT / 'evaluation/b5c4_formal_gpu_g2/preflight'
PENDING_IDS = ['B5C2_A_TURN_003', 'B5C2_B_TURN_003', 'B5C2_C_TURN_003', 'B5C2_D_TURN_003']
EXPECT_SOURCE = {'Exact': {'B5C2_A_TURN_003': 'B5C3_NODE_18', 'B5C2_B_TURN_003': 'B5C3_NODE_18',
                           'B5C2_C_TURN_003': 'B5C3_NODE_27', 'B5C2_D_TURN_003': 'B5C3_NODE_27'},
                 'Greedy': {'B5C2_A_TURN_003': 'B5C3_NODE_18', 'B5C2_B_TURN_003': 'B5C3_NODE_18',
                            'B5C2_C_TURN_003': 'B5C3_NODE_20', 'B5C2_D_TURN_003': 'B5C3_NODE_20'}}


def epoch_from(x):
    x = copy.deepcopy(x)
    for c in x['candidates']:
        c['candidate']['lineage_path'] = tuple(c['candidate']['lineage_path'])
        c['candidate'] = CheckpointCandidate(**c['candidate'])
        c['token_ids'] = tuple(c['token_ids'])
    x['candidates'] = tuple(StrictCandidate(**c) for c in x['candidates'])
    for p in x['pendings']:
        p['token_ids'] = tuple(p['token_ids'])
    x['pendings'] = tuple(StrictPending(**p) for p in x['pendings'])
    for e in x['edges']:
        e['conditions'] = tuple(tuple(v) for v in e['conditions'])
    x['edges'] = tuple(StrictEdge(**e) for e in x['edges'])
    return StrictAllocationEpoch(**x)


def main():
    r = json.loads(B5D3.read_text())
    epoch = epoch_from(r['epoch'])
    epoch.validate()
    raw = r['raw_epoch']
    assert len(raw['candidates']) == 7 and len(raw['pendings']) == 4, 'b5d3 基线不是 B5-C4 结构'
    ref = json.loads((OUT / 'g2_reference.json').read_text())
    histories = [json.loads((B5D4 / 'p01_Greedy' / f'history_{i:02d}.json').read_text()) for i in range(1, 9)]
    origins = []
    for c in raw['candidates']:
        seen = [h['request']['request_id'] for h in histories
                if any(n['token_ids'] == c['token_ids'] and n['slots'] for n in h['state']['before']['nodes'])]
        assert seen, '候选缺少历史来源'
        origins.append({'checkpoint_id': c['checkpoint_id'], 'first_observed_after': seen[0]})
    inputs = json.loads((B5D4 / 'prepared_inputs.json').read_text())

    optimizer = GlobalOptimizer(RecoveryCostModel())
    plans = {}
    # ── Exact 臂：生产 plan_strict，应门禁内自动 Exact ──
    plans['Exact'] = optimizer.plan_strict(epoch)
    decision = dict(optimizer.last_strict_decision)
    assert decision['solver'] == 'Exact' and decision['gate_allowed'], 'Exact 臂预检失败：' + repr(decision)
    # ── Greedy 臂：生产 Marginal Greedy 内部路径适配 ──
    allowed, gate = _exact_admission(len(epoch.candidates), len(epoch.pendings), epoch.K)
    assert allowed, 'B5-C4 域应满足 Exact 门禁（Greedy 臂合法性前提）'
    plans['Greedy'] = plan_greedy_strict(optimizer, epoch)
    checks = []
    for solver, plan in plans.items():
        validate_plan(epoch, plan)
        row = ref['Exact' if solver == 'Exact' else 'Marginal Greedy']
        assert set(plan.selected_ids) == set(row['selected']), solver + ' 预检选集与 e3 不一致：' + repr(plan.selected_ids)
        assert abs(plan.recovery_cost_ms - row['J_ms']) <= 1e-9, solver + ' 预检 J 与 e3 不一致'
        for prow in plan.pending:
            e = row['per_pending'][prow.pending_id]
            assert (prow.E, prow.T, prow.G) == (e['E'], e['T'], e['G']), solver + ' 预检 E-T-G 不符：' + prow.pending_id
        checks.append({'solver': solver, 'selected': list(plan.selected_ids), 'J_ms': plan.recovery_cost_ms,
                       'e3_match': True, 'validate_plan': 'PASS'})
    # ── 来源合同：8 组合唯一来源 ──
    from evaluation.b5c4_formal_gpu_g2.pkg_g2.source_contract import choose_source
    source_checks = []
    for solver, plan in plans.items():
        for pid in PENDING_IDS:
            os.environ['G2_PENDING'] = pid
            t = choose_source(raw, epoch, plan, origins, inputs, solver)
            assert t['pending']['continuation_id'] == pid and t['source_checkpoint'] == EXPECT_SOURCE[solver][pid], \
                pid + ' 来源合同不符：' + t['source_checkpoint']
            assert t['provenance']['status'] == 'PROVENANCE_UNVERIFIED', '来源血统不得升级'
            assert t['expected']['E'] > 0 and tuple(t['pending']['token_ids'][:t['expected']['E']]) == tuple(
                next(c['token_ids'] for c in raw['candidates'] if c['checkpoint_id'] == t['source_checkpoint'])), '恢复前缀不合法'
            source_checks.append({'solver': solver, 'pending': pid, 'source': t['source_checkpoint'],
                                  'E': t['expected']['E'], 'support_ok': True})
    os.environ.pop('G2_PENDING', None)
    result = {'status': 'PASS', 'baseline': str(B5D3), 'gate': {'allowed': allowed, 'detail': gate},
              'exact_decision': decision, 'solver_checks': checks, 'source_checks': source_checks,
              'note': '宿主机 CPU 预检：两求解臂与 e3 冻结行全精度一致；来源合同 8 组合唯一；GPU 侧仍须每生命周期重验 Token/物理身份'}
    out = OUT / 'cpu_preflight.json'
    assert not out.exists(), '拒绝覆盖 cpu_preflight.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('CPU 预检 PASS：Exact', checks[0]['selected'], checks[0]['J_ms'], '/ Greedy', checks[1]['selected'], checks[1]['J_ms'],
          '/ 来源合同 8/8 唯一')


if __name__ == '__main__':
    main()
