"""P0-4 三策略完整控制周期：统一在线 Select 入口（CPU 可运行，GPU 阶段同代码复用）。

设计原则（对应 P0-4 审查 §2/§7 与 Task 1 spec §二/§三）：
- FlowState-Greedy 必须走正式生产入口 GlobalOptimizer.plan_strict()（与 G1 pkg_greedy 同一入口），
  在已认证的 StrictAllocationEpoch 上在线求解；16 候选超 Exact 门禁应自动 Marginal Greedy。
- Marconi-Evict 调用真实内核 MarconiEvictPolicy.select（T2-Mini），KVFlow-inspired 调用真实内核
  kvflow.select（ASG/STE，known_at_epoch 门控）。两内核评分/驱逐/保留规则零修改。
- 三策略产出统一换算到物理候选域后，经同一生产构造链 planner.finish(prepared, selected) 生成
  StrictAllocationPlan 并 validate_plan——与正式 b5d4_apply 为冻结选集构建计划的路径完全相同，
  后续 Recurrent-only Reconciliation 由 plan.logical_eviction_ids 驱动，策略无关。
- 冻结选集仅作事后 oracle 对照（求解完成后才读取；比对不一致即失败停止，绝不回退、
  绝不把冻结加载当在线 Select）。oracle 读取耗时单独计时并标记为窗口外研究审计。
- Epoch 构造链逐字复刻 paper_hybrid_solver_integration_gate.py B 段（已证复现冻结 J=154.348…）；
  census 对账复用 G1 pkg_greedy.physical_cpu_census 纯函数，CPU 侧可复算冻结 16=16 映射。
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter_ns

from evaluation.hybrid_solver_gpu_gate_g1.pkg_greedy.control import physical_cpu_census
from evaluation.kvflow_inspired_retention_cpu import policy as kvflow
from evaluation.marconi_evict_t2m.evict_policy import REAL_CHECKPOINT_BYTES, MarconiEvictPolicy
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.strict_control import validate_plan

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / 'evaluation/c_formal_perf_gpu'          # C/K3 正式实验冻结目录（只读）
T2G1 = ROOT / 'evaluation/t2g1_three_policy_gpu'      # 三策略冻结 oracle（只读）
MAIN = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
PILOT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'
OUT = Path(__file__).resolve().parent / 'out'         # 本 harness 唯一写入目录
K = 3
THREE_POLICIES = ('FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired')

# planner 模块按 integration gate 的同款 importlib 方式加载（避免包路径副作用）
_spec = importlib.util.spec_from_file_location(
    'b5d1_planner_gate', ROOT / 'evaluation/exact_planner_contract_gate_b5d1/planner.py')
planner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(planner)


def _encode(x) -> bytes:
    """与 b5c3/c_formal 控制路径相同的确定性序列化（sort_keys）。"""
    return json.dumps(x, sort_keys=True, ensure_ascii=False).encode()


def _read(p: Path):
    return json.loads(p.read_text())


def build_ck3_epoch():
    """按已冻结验证的构造链构建 C/K3 StrictAllocationEpoch（含 prepared 与 census 复算）。

    返回 ctx 字典：epoch/prepared（生产认证对象）、census 复算行、CPU↔物理映射、
    声明 case 与 GPU 四 Pending 域、冻结 oracle 路径等。GPU 阶段同结构由实时观察填充。
    """
    rec = _read(CFG / 'g1_CB_FS' / 'run' / 'reconciliation.json')
    raw = rec['raw_epoch']
    data = _encode(raw)
    assert hashlib.sha256(data).hexdigest() == rec['epoch']['snapshot_sha256'], \
        'raw_epoch 字节与冻结 snapshot_sha256 不一致'
    # physical 视图按冻结运行推导（prepare 当时已通过 ⇒ unchanged=True）
    initial_view = {'before': rec['before'], 'audit_before': rec['audit_before'],
                    'unchanged': True, 'window': rec['apply_validation']['window']}
    inputs = _read(CFG / 'g1_CB_FS' / 'prepared_inputs.json')
    origins = rec['origins']
    evidence = planner.FrozenEvidence(
        data, hashlib.sha256(data).hexdigest(),
        physical_snapshot=_encode(initial_view),
        physical_sha256=hashlib.sha256(_encode(initial_view)).hexdigest())
    epoch = planner.from_frozen(evidence, K, (initial_view, origins, inputs))
    prepared = planner.prepare(epoch, evidence)
    assert len(epoch.candidates) == 16 and len(epoch.pendings) == 4 and epoch.K == K, 'C/K3 域合同失败'
    # census 纯函数复算：CPU 声明域 ↔ 物理候选域 16=16 对账必须复现冻结结果
    mapping = _read(CFG / 'cpu_candidate_prefix_mapping.json')['mapping']
    rows, unmapped = physical_cpu_census(raw, inputs, mapping)
    assert rows == rec['census']['mapped'] and not unmapped, 'census 复算与冻结映射不一致'
    cpu_to_phys = {r['cpu_checkpoint_id']: r['physical_checkpoint_id'] for r in rows}
    phys_to_cpu = {r['physical_checkpoint_id']: r['cpu_checkpoint_id'] for r in rows}
    assert len(cpu_to_phys) == 16 == len(phys_to_cpu), 'census 映射非 16=16'
    # 声明 case（两 Baseline 内核的冻结输入域）与 GPU 四 Pending
    case = next(c for c in _read(MAIN / 'workloads.json')['cases'] if c['structure'] == 'C')
    gpu_pendings = [p['continuation_id'] for p in _read(PILOT / 'prepared_inputs.json')['pendings']]
    return {'epoch': epoch, 'prepared': prepared, 'raw': raw, 'inputs': inputs,
            'cpu_to_phys': cpu_to_phys, 'phys_to_cpu': phys_to_cpu, 'case': case,
            'gpu_pendings': gpu_pendings, 'physical_ids': sorted(cpu_to_phys.values())}


# 各策略决策时实际可见输入的如实描述（元数据公平记录，P0-4 §3.2）
VISIBILITY = {
    'FlowState-Greedy': 'Strict Epoch 冻结快照域：16 物理候选（含 L/槽位/物化时间线）、4 Pending 的 '
                        'T 与 FA 前沿、Φ 恢复成本模型；GPU 阶段为实时观察 + census 映射后同域',
    'Marconi-Evict': '声明 case 候选（token_position_L、等大小 51,511,296B）、last_access_order '
                     '（声明访问序）、parent 链；α=1.0；不读恢复成本',
    'KVFlow-inspired': '声明 candidates + graph + declarations（ASG/STE，known_at_epoch 门控，'
                       '无未来信息）；不读恢复成本',
}


def select_flowstate(ctx):
    """正式路径：GlobalOptimizer.plan_strict 在线求解（生产入口，非内部内核）。"""
    t0 = perf_counter_ns()
    optimizer = GlobalOptimizer(RecoveryCostModel())
    plan = optimizer.plan_strict(ctx['epoch'])
    call_ms = (perf_counter_ns() - t0) / 1e6
    decision = dict(optimizer.last_strict_decision)
    validate_plan(ctx['epoch'], plan)
    return plan, {'kernel_entry': 'flowstate.optimizer.GlobalOptimizer.plan_strict',
                  'selection_call_ms': call_ms, 'decision': decision}


def select_marconi(ctx, k=K):
    """真实内核：MarconiEvictPolicy.select 逐轮动态驱逐（规则零修改）。"""
    cs = ctx['case']['candidates']
    core = tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (),
                                     c['token_position_L'], REAL_CHECKPOINT_BYTES, True) for c in cs)
    recency = {c['checkpoint_id']: c['last_access_order'] for c in cs}
    parent = {c['checkpoint_id']: c['parent_candidate'] for c in cs}
    t0 = perf_counter_ns()
    result = MarconiEvictPolicy().select(core, k, recency, parent, 1.0)
    call_ms = (perf_counter_ns() - t0) / 1e6
    return result, {'kernel_entry': 'evaluation.marconi_evict_t2m.evict_policy.MarconiEvictPolicy.select',
                    'selection_call_ms': call_ms,
                    'eviction_order_cpu': list(result.evicted_checkpoint_ids),
                    'rounds': len(result.rounds)}


def select_kvflow(ctx, k=K):
    """真实内核：kvflow.select（ASG/STE 保留策略，known_at_epoch 门控）。"""
    case = ctx['case']
    t0 = perf_counter_ns()
    selected_cpu, trace = kvflow.select(case['candidates'], case['graph'], case['declarations'],
                                        case['epoch_id'], case['configuration_digest'], k)
    call_ms = (perf_counter_ns() - t0) / 1e6
    return tuple(selected_cpu), {'kernel_entry': 'evaluation.kvflow_inspired_retention_cpu.policy.select',
                                 'selection_call_ms': call_ms}


def run_three_policy_select(ctx):
    """三策略统一在线 Select；每策略产出完整记录 + 生产构造的 StrictAllocationPlan。

    全流程 fail-loud：任一内核异常直接传播；预算/域/映射不符立即 AssertionError；
    不存在任何"读冻结选集兜底"分支。oracle 对照在选集产生并构建 Plan 之后才执行。
    """
    records, plans = {}, {}

    # ── FlowState-Greedy：生产入口在线求解 ──
    plan, meta = select_flowstate(ctx)
    sel_phys = tuple(plan.selected_ids)
    records['FlowState-Greedy'] = {
        'policy': 'FlowState-Greedy', **meta,
        'selected_physical_ids': list(sel_phys),
        'selected_cpu_ids': sorted(ctx['phys_to_cpu'][c] for c in sel_phys),
        'evicted_physical_ids': list(plan.logical_eviction_ids),
        'recovery_cost_ms': plan.recovery_cost_ms,
        'inputs_visibility': VISIBILITY['FlowState-Greedy']}
    plans['FlowState-Greedy'] = plan

    # ── Marconi-Evict：真实驱逐内核，选集（CPU 域）换算物理域后走生产构造链 ──
    ev, meta = select_marconi(ctx)
    sel_cpu = tuple(sorted(ev.selected_checkpoint_ids))
    absent = [c for c in sel_cpu if c not in ctx['cpu_to_phys']]
    assert not absent, 'Marconi-Evict 选集未在物理域物化：%r' % (absent,)
    sel_phys = tuple(ctx['cpu_to_phys'][c] for c in sel_cpu)
    plan = planner.finish(ctx['prepared'], sel_phys)
    validate_plan(ctx['epoch'], plan)
    records['Marconi-Evict'] = {
        'policy': 'Marconi-Evict', **meta,
        'selected_physical_ids': list(sel_phys), 'selected_cpu_ids': list(sel_cpu),
        'evicted_physical_ids': list(plan.logical_eviction_ids),
        'eviction_order_physical_ids': [ctx['cpu_to_phys'][c] for c in meta['eviction_order_cpu']],
        'recovery_cost_ms': plan.recovery_cost_ms,
        'inputs_visibility': VISIBILITY['Marconi-Evict']}
    plans['Marconi-Evict'] = plan

    # ── KVFlow-inspired：真实保留内核，同生产构造链 ──
    sel_cpu_kv, meta = select_kvflow(ctx)
    sel_cpu = tuple(sorted(sel_cpu_kv))
    absent = [c for c in sel_cpu if c not in ctx['cpu_to_phys']]
    assert not absent, 'KVFlow-inspired 选集未在物理域物化：%r' % (absent,)
    sel_phys = tuple(ctx['cpu_to_phys'][c] for c in sel_cpu)
    plan = planner.finish(ctx['prepared'], sel_phys)
    validate_plan(ctx['epoch'], plan)
    records['KVFlow-inspired'] = {
        'policy': 'KVFlow-inspired', **meta,
        'selected_physical_ids': list(sel_phys), 'selected_cpu_ids': list(sel_cpu),
        'evicted_physical_ids': list(plan.logical_eviction_ids),
        'recovery_cost_ms': plan.recovery_cost_ms,
        'inputs_visibility': VISIBILITY['KVFlow-inspired']}
    plans['KVFlow-inspired'] = plan

    # 预算与域合同（三策略同尺度）
    for policy, r in records.items():
        assert len(r['selected_physical_ids']) == K and len(set(r['selected_physical_ids'])) == K, \
            '%s 预算违反 K=%d' % (policy, K)
        assert set(r['selected_physical_ids']) <= set(ctx['physical_ids']), '%s 选集超出 16 候选域' % policy
        assert set(r['selected_physical_ids']) | set(r['evicted_physical_ids']) == set(ctx['physical_ids']) \
            and not set(r['selected_physical_ids']) & set(r['evicted_physical_ids']), \
            '%s 选集/驱逐集未划分全体候选' % policy
    return records, plans


def oracle_compare(records, ctx):
    """事后 oracle 对照（仅研究审计；必须在全部在线选择完成后调用，耗时不计入 Select）。"""
    t0 = perf_counter_ns()
    t2g1_plans = _read(T2G1 / 't2g1_plans.json')['plans']
    pilot = _read(CFG / 'policy_plans.json')
    rec = _read(CFG / 'g1_CB_FS' / 'run' / 'reconciliation.json')
    results = {}
    for policy in THREE_POLICIES:
        # 选集为集合语义：按排序后比较（内核输出序与冻结记录序可能不同，元素一致即复现）
        ok_cpu = sorted(records[policy]['selected_cpu_ids']) == sorted(t2g1_plans[policy]['selected_checkpoint_ids'])
        results[policy] = {'matches_t2g1_frozen': ok_cpu}
        if not ok_cpu:
            raise AssertionError('在线 %s 选集与冻结 oracle 不一致：%r vs %r' % (
                policy, records[policy]['selected_cpu_ids'],
                t2g1_plans[policy]['selected_checkpoint_ids']))
    # FS 额外物理域/J 合同（G1 已证在线==冻结）
    fs = records['FlowState-Greedy']
    assert set(fs['selected_physical_ids']) == set(rec['plan']['selected_ids']), 'FS 物理选集与 G1 冻结不一致'
    assert abs(fs['recovery_cost_ms'] - rec['plan']['recovery_cost_ms']) <= 1e-9, 'FS J 与冻结不一致'
    assert sorted(fs['selected_cpu_ids']) == sorted(pilot['FlowState-Greedy']['selected_checkpoint_ids'])
    results['FlowState-Greedy'].update(
        {'matches_g1_physical_plan': True, 'matches_frozen_J': True, 'matches_pilot_plans': True})
    return {'per_policy': results, 'oracle_check_ms': (perf_counter_ns() - t0) / 1e6,
            'note': 'oracle 读取与比对发生在全部在线选择之后；属窗口外研究审计，不计入 Select 耗时'}


def main():
    """构造 epoch → 三策略在线选择 → oracle 事后对照 → 落盘记录（仅写本包 out/ 目录）。"""
    OUT.mkdir(exist_ok=True)
    ctx = build_ck3_epoch()
    records, _plans = run_three_policy_select(ctx)
    oracle = oracle_compare(records, ctx)
    payload = {'K': K, 'gpu_pendings': ctx['gpu_pendings'],
               'selection_records': records, 'oracle': oracle,
               'note': 'CPU 门禁产物；Select 为真实内核在线耗时；oracle 为事后对照'}
    (OUT / 'selection_records.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print('三策略在线选择完成：')
    for policy in THREE_POLICIES:
        r = records[policy]
        print('  %-16s call=%.3fms  selected_cpu=%s  J=%.3f' % (
            policy, r['selection_call_ms'], r['selected_cpu_ids'], r['recovery_cost_ms']))
    print('oracle 对照：', oracle['per_policy'])


if __name__ == '__main__':
    main()
