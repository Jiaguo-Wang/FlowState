"""T2-G1：三策略（FlowState-Greedy / Marconi-Evict / KVFlow-inspired）在 C/K3 冻结域的统一接入。

复用 C/K3 GPU Pilot 的合同：16 个 CPU 候选 ↔ 物理候选（T2-R 逐 Token 映射证据）、
四个 GPU Pending（C_A..C_D，与 CPU case 的 ready 调用逐 Token 全等）、K=3。
本模块只做选择与 CPU 侧合同核验，产出与 pilot policy_plans.json 同格式的
t2g1_plans.json（供 GPU runner 以 PILOT_POLICY 消费）；不运行 GPU、不产生物理操作凭据。

三策略决策输入的合法决策时刻核验：
- 全体策略输入来自冻结声明（validate_declared_case 复核：Token 关系/访问事件/ASG/固定前缀），
  评分域只含 4 个 ready Pending（C_JOIN 为 CPU 独有 future，不进入决策）；
- Marconi-Evict：last_access_order 全有限（DECLARED_SYNTHETIC_ACCESS 声明事件）、父子链合法；
- KVFlow-inspired：STE 无 UNKNOWN（T2-R 验证合同）。
Recurrent-only 落实性：选集 ⊆ 16 个已物化物理状态，控制路径按 logical_eviction_ids 驱逐
其余状态并保护 FA——Marconi 的逻辑父子驱逐只决定最终保留集，不需要原版的 Attention KV
合并操作（每个候选本就是独立物理状态）。
"""
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

from evaluation.baseline_reuse_t2r.interface import validate_declared_case
from evaluation.kvflow_inspired_retention_cpu import policy as kvflow
from evaluation.kvflow_multiworkflow_cpu_main import run_cpu as legacy
from evaluation.marconi_evict_t2m.evict_policy import REAL_CHECKPOINT_BYTES, MarconiEvictPolicy
from flowstate.optimizer import GlobalOptimizer
from flowstate.state_catalog import CheckpointCandidate

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
PILOT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'
T2R = ROOT / 'evaluation/baseline_reuse_t2r'
OUT = Path(__file__).resolve().parent
K = 3
THREE_POLICIES = ('FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired')


def load_case():
    """载入 C 结构冻结 case 与 GPU 四 Pending 域。"""
    case = next(c for c in json.loads((MAIN / 'workloads.json').read_text())['cases'] if c['structure'] == 'C')
    prepared = json.loads((PILOT / 'prepared_inputs.json').read_text())
    gpu_pendings = {p['continuation_id']: p for p in prepared['pendings']}
    # GPU 四 Pending 必须与 CPU ready 调用逐 Token 全等（决策域对齐）
    ready = {p['continuation_id']: p for p in case['calls'] if p['ready']}
    assert set(gpu_pendings) <= set(ready), 'GPU Pending 不在 CPU ready 域'
    for pid, g in gpu_pendings.items():
        assert ready[pid]['token_ids'] == g['token_ids'] and ready[pid]['planning_target_T'] == g['planning_target_T']
    return case, gpu_pendings


def select_three(case, k=K):
    """三策略统一选择；各策略复用既有内核，评分/驱逐算法零修改。"""
    validate_declared_case(case, k)  # 冻结声明输入合法性（Token/访问/ASG/前缀）
    cs = case['candidates']
    ep = legacy.epoch(case, k)  # current 域=4 个 ready Pending，与 GPU 决策问题一致
    core = tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'], 1, True) for c in cs)
    recency = {c['checkpoint_id']: c['last_access_order'] for c in cs}
    parent = {c['checkpoint_id']: c['parent_candidate'] for c in cs}
    results = {}
    # FlowState-Greedy：生产 Greedy 内核 + 共同评分闭包（与 run_cpu/pilot 同路径）
    t0 = perf_counter_ns()
    fs = GlobalOptimizer(legacy.common.MODEL)._select_candidates(
        core, k, lambda chosen: legacy.common.score(ep, tuple(c.checkpoint_id for c in chosen))['cost_ms'])
    results['FlowState-Greedy'] = (tuple(c.checkpoint_id for c in fs.selected), (perf_counter_ns() - t0) / 1e6)
    # Marconi-Evict：T2-Mini 增量收益 + 动态驱逐（实测等大小字节）
    t0 = perf_counter_ns()
    ev = MarconiEvictPolicy().select(
        tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'], REAL_CHECKPOINT_BYTES, True) for c in cs),
        k, recency, parent, 1.0)
    results['Marconi-Evict'] = (tuple(ev.selected_checkpoint_ids), (perf_counter_ns() - t0) / 1e6)
    # KVFlow-inspired：既有 ASG/STE 保留策略
    t0 = perf_counter_ns()
    kv, _ = kvflow.select(case['candidates'], case['graph'], case['declarations'], case['epoch_id'], case['configuration_digest'], k)
    results['KVFlow-inspired'] = (tuple(kv), (perf_counter_ns() - t0) / 1e6)
    return results, ep


def expected_E_G(case, gpu_pendings, selected):
    """按 pilot policy_plans.json 格式计算四 Pending 的 E/T/G 预期。"""
    by_id = {c['checkpoint_id']: c for c in case['candidates']}
    out = {}
    for pid, p in gpu_pendings.items():
        e = max((by_id[c]['token_position_L'] for c in selected
                 if kvflow.prefix(by_id[c]['token_ids'], p['token_ids'])), default=0)
        t = p['planning_target_T']
        out[pid] = {'E': e, 'T': t, 'G': t - e}
    return out


def main():
    case, gpu_pendings = load_case()
    snapshot_sha = hashlib.sha256((MAIN / 'workloads.json').read_bytes()).hexdigest()
    results, ep = select_three(case)
    pilot_plans = json.loads((PILOT / 'policy_plans.json').read_text())
    with (MAIN / 'cpu_comparison.csv').open() as f:
        frozen = {(r['structure'], int(r['K']), r['policy']): r for r in csv.DictReader(f)}
    physical = {r['cpu_checkpoint_id'] for r in json.loads((T2R / 'historical_physical_mapping.json').read_text())['rows']}
    plans, checks = {}, []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})
        if not ok:
            raise AssertionError('T2-G1 合同失败：%s %s' % (name, detail))

    for policy, (selected, ms) in results.items():
        sel = list(selected)
        check('budget_%s' % policy, len(sel) == K and len(set(sel)) == K, sel)
        check('domain_%s' % policy, set(sel) <= {c['checkpoint_id'] for c in case['candidates']}, '超出 16 候选域')
        check('physical_%s' % policy, set(sel) <= physical, '所选状态无物理物化证据')
        eg = expected_E_G(case, gpu_pendings, sel)
        j = legacy.common.score(ep, sel)['cost_ms']
        plans[policy] = {'selected_checkpoint_ids': sel, 'expected_E_G': eg,
                         'J_ms_current_scope': j, 'selection_call_ms': round(ms, 6),
                         'selection_source': 'T2-G1 统一接口（FS=生产Greedy内核；Marconi-Evict=T2-Mini；KVFlow=既有保留策略）'}
    # 交叉核对：FS 必须复现 GPU 已验证的 pilot 计划；KVFlow 必须复现冻结 CPU 行
    check('fs_matches_pilot', plans['FlowState-Greedy']['selected_checkpoint_ids'] == pilot_plans['FlowState-Greedy']['selected_checkpoint_ids']
          and plans['FlowState-Greedy']['expected_E_G'] == pilot_plans['FlowState-Greedy']['expected_E_G'],
          '统一接口未复现 pilot FS 计划')
    check('kvflow_matches_frozen', plans['KVFlow-inspired']['selected_checkpoint_ids'] == json.loads(frozen['C', K, 'KVFlow-inspired']['selected']))
    with (ROOT / 'evaluation/marconi_evict_t2m/evict_vs_legacy.csv').open() as f:
        evict_row = next(r for r in csv.DictReader(f)
                         if r['structure'] == 'C' and int(r['K']) == K and r['policy'] == 'Marconi Adaptation (evict)')
    check('marconi_matches_t2mini', plans['Marconi-Evict']['selected_checkpoint_ids'] == json.loads(evict_row['selected']))
    check('frozen_unchanged', hashlib.sha256((MAIN / 'workloads.json').read_bytes()).hexdigest() == snapshot_sha)
    (OUT / 't2g1_plans.json').write_text(json.dumps(
        {'K': K, 'gpu_pendings': sorted(gpu_pendings), 'plans': plans,
         'checks': checks, 'note': '与 c_shared_gpu_pilot_k3/policy_plans.json 同格式；GPU runner 经 PILOT_POLICY 消费；本轮未运行 GPU'},
        ensure_ascii=False, indent=2))
    print('三策略统一接入完成；检查 %d/%d 通过' % (sum(c['pass'] for c in checks), len(checks)))
    for p, v in plans.items():
        print('%-16s selected=%s' % (p, v['selected_checkpoint_ids']))
        print('   E/G：', {k: (e['E'], e['G']) for k, e in v['expected_E_G'].items()}, 'J=%.3f' % v['J_ms_current_scope'])


if __name__ == '__main__':
    main()
