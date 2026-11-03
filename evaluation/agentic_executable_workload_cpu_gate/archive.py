"""只读复核既有四Session物理证据；绝不标记为当前A/B-only运行状态。"""
from dataclasses import asdict
from .adapter import ROOT, read, sha, require, classify_aliases

BASE = ROOT / 'evaluation/multishare_runtime_gate_b5c3'


def load_archival_epoch(K=1):
    from evaluation.exact_planner_contract_gate_b5d1 import planner
    snapshot = BASE / 'strict_epoch_snapshot.json'
    physical = BASE / 'physical_state_census.json'
    evidence = planner.FrozenEvidence(snapshot.read_bytes(), sha(snapshot), physical_snapshot=physical.read_bytes(), physical_sha256=sha(physical))
    epoch = planner.from_frozen(evidence, K, (read(physical), read(BASE/'candidate_provenance.json'), read(BASE/'prepared_inputs.json')))
    planner.prepare(epoch, evidence)
    return epoch


def archival_audit():
    from flowstate.optimizer import GlobalOptimizer
    from flowstate.recovery_model import RecoveryCostModel
    from flowstate.strict_control import PhysicalNode, PhysicalView, bind, preflight, validate_plan
    raw = read(BASE/'physical_state_census.json')
    view_data = dict(raw['before'])
    view_data['nodes'] = tuple(PhysicalNode(**{**n, 'token_ids': tuple(n['token_ids']), 'slots': tuple(n['slots']), 'component_ids': tuple(n['component_ids'])}) for n in view_data['nodes'])
    view_data['free_slots'] = tuple(view_data['free_slots'])
    view = PhysicalView(**view_data)
    history = read(BASE/'prepared_inputs.json')['history']
    expected = ['B5C2_A_TURN_001', 'B5C2_B_TURN_001', 'B5C2_A_TURN_002', 'B5C2_B_TURN_002']
    observed = [h['request_id'] for h in history]
    rows = []
    for K in [1, 2]:
        epoch = load_archival_epoch(K)
        plans = [GlobalOptimizer(RecoveryCostModel()).plan_strict(epoch) for _ in range(2)]
        require(plans[0] == plans[1], '历史合法Epoch规划不确定')
        plan = plans[0]
        validate_plan(epoch, plan)
        binding = bind(epoch, plan, view, view.lease_id)
        preflight(epoch, plan, binding, view, view.lease_id)
        rows.append({'scope': 'ARCHIVAL_ABCD_ONLY', 'K': K, 'deterministic': True, 'selected': list(plan.selected_ids), 'pending': [asdict(p) for p in plan.pending], 'J_ms': plan.recovery_cost_ms, 'historical_binding_check': 'PASS', 'current_physical_binding': None})
    source_to_node = {}
    submitted = []
    for i, request in enumerate(history, 1):
        state = read(BASE/f'history_{i:02d}.json')['state']['before']['nodes']
        tokens = request['token_ids']
        matches = [n for n in state if n['slots'] and len(n['token_ids']) <= len(tokens) and n['token_ids'] == tokens[:len(n['token_ids'])]]
        if matches:
            best = max(matches, key=lambda n: len(n['token_ids']))
            source_to_node[request['request_id']] = best['node_id']
            submitted.append({'request_id': request['request_id'], 'observed_node_id': best['node_id'], 'L': len(best['token_ids']), 'source': f'history_{i:02d}.json/state/before', 'meaning': '该时刻最深驻留前缀，不是逐张量创建血缘'})
    aliases = classify_aliases(source_to_node)
    return {'status': 'PASS', 'scope': 'ARCHIVAL_ABCD_ONLY', 'epoch_id': raw['window'], 'candidate_count': len(raw['strict_epoch']['candidates']), 'pending_count': len(raw['strict_epoch']['pendings']), 'recorded_history': observed, 'expected_AB_only_history': expected, 'reusable_as_current_AB_only': observed == expected and False, 'live': False, 'raw_mapping_rows': submitted, 'source_alias_diagnostic': aliases, 'actual_candidate_budget_unit': '全树唯一驻留物理Node；不按历史Request ID计数', 'allocation_determinism': rows, 'new_AB_only': {'state': 'BLOCKED', 'candidates': None, 'strict_edges': None, 'T': None, 'K': None, 'reason': '缺少仅A/B四条历史后的完整同边界物理观察；不能复用四Session快照或旧Node/Slot'}, 'sha256': {str(BASE/n): sha(BASE/n) for n in ['strict_epoch_snapshot.json', 'physical_state_census.json', 'prepared_inputs.json', 'candidate_provenance.json']}}
