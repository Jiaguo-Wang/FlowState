"""Hybrid Solver 集成的 CPU 正确性门禁：A) Strict 20 全 Exact；B) C/K3 超域回退 Greedy；C) 失败路径 Fail-stop。

被测对象：flowstate/optimizer.py GlobalOptimizer.plan_strict（本轮已集成的混合求解入口）。
冻结输入构造：复用 evaluation/exact_planner_contract_gate_b5d1/planner.py 的 from_frozen/prepare/benefits
仅用于构造 StrictAllocationEpoch 与独立 Oracle 参照——这是评估侧工具，运行时不依赖它（依赖方向仍为
evaluation → flowstate）。

五个快照来源（全部只读）：
  w1–w4  ← evaluation/independent_runtime_allocation_e2/<w>/strict_epoch_snapshot.json（e2 冻结链）
  B5-C4  ← evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json
           （已核实与 b5c4/results.json input 的位置/兼容/目标逐项相等，是 B5-C4 条件的严格来源）
C/K3 来源：evaluation/c_formal_perf_gpu/g1_CB_FS/run/reconciliation.json 内冻结的 raw_epoch/origins/plan
  （GPU 正式实验实际认证过的同一 Epoch 字节，sha256 与记录的 snapshot_sha256 核对）。

交叉参照：
  - e3_step1/condition_results.csv 的 Exact/Greedy 行（选集与 J）；
  - b5d1 planner.solve('Exact')（同门禁枚举、独立实现路径）；
  - c_formal_perf_gpu/policy_plans.json + kvflow_multiworkflow_cpu_main/cpu_results.json（C/K3 冻结值）。
不启动 GPU；不修改任何冻结输入；脚本首尾对全部读取文件与被测源码做哈希一致性断言。
"""
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVAL = Path(__file__).resolve().parent
E2 = EVAL / 'independent_runtime_allocation_e2'
B5C3 = EVAL / 'multishare_runtime_gate_b5c3'
CFG = EVAL / 'c_formal_perf_gpu'
E3CSV = EVAL / 'greedy_swap_structure_e3_step1' / 'condition_results.csv'
MW = EVAL / 'kvflow_multiworkflow_cpu_main'
EPS = 1e-9

sys.path.insert(0, str(ROOT))
from flowstate.optimizer import GlobalOptimizer, _exact_admission  # noqa: E402
from flowstate.recovery_model import RecoveryCostModel  # noqa: E402
from flowstate.strict_control import validate_plan  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'b5d1_planner_gate', EVAL / 'exact_planner_contract_gate_b5d1' / 'planner.py')
planner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(planner)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(x) -> bytes:
    """与 b5c3/c_formal 控制路径相同的确定性序列化（sort_keys）。"""
    return json.dumps(x, sort_keys=True, ensure_ascii=False).encode()


def read_json(p: Path):
    return json.loads(p.read_text())


def build_epoch(directory: Path, k: int):
    """按 e2 evaluate.py 的冻结构造链构建 StrictAllocationEpoch（含来源 origins）。"""
    snapshot_bytes = (directory / 'strict_epoch_snapshot.json').read_bytes()
    census_bytes = (directory / 'physical_state_census.json').read_bytes()
    s = json.loads(snapshot_bytes)
    physical = json.loads(census_bytes)
    assert 'before' in physical and 'audit_before' in physical, directory
    inputs = read_json(directory / 'prepared_inputs.json')
    histories = [read_json(p) for p in sorted(directory.glob('history_*.json'))]
    origins = []
    for c in s['candidates']:
        matches = [h for h in histories if any(
            n['node_id'] == c['state_identity']['node_id']
            and n['token_ids'] == c['token_ids']
            and n['slots'] == c['state_identity']['target_mamba_slots']
            for n in h['state']['before']['nodes'])]
        assert matches, f'缺少候选来源记录：{c["checkpoint_id"]}'
        origins.append({'checkpoint_id': c['checkpoint_id'],
                        'first_observed_after': matches[0]['request']['request_id'],
                        'provenance': 'PROVENANCE_UNVERIFIED'})
    evidence = planner.FrozenEvidence(
        snapshot_bytes, sha_bytes(snapshot_bytes),
        physical_snapshot=census_bytes, physical_sha256=sha_bytes(census_bytes))
    epoch = planner.from_frozen(evidence, k, (physical, origins, inputs))
    return epoch, evidence


def strict_score_ids(epoch):
    """与 plan_strict 内部同一评分语义的独立复刻（frontiers + 冻结 Phi），供 Greedy 行为对照。"""
    model = RecoveryCostModel()

    def score_ids(selected_ids):
        frontiers = epoch.frontiers(selected_ids)
        return sum(model.estimate(p.target - e, p.target)
                   for p, e in zip(epoch.pendings, frontiers))
    return score_ids


def main():
    watched = {}
    for p in [ROOT / 'flowstate' / 'optimizer.py', E3CSV,
              CFG / 'g1_CB_FS' / 'run' / 'reconciliation.json',
              CFG / 'policy_plans.json',
              MW / 'cpu_results.json',
              B5C3 / 'strict_epoch_snapshot.json', B5C3 / 'physical_state_census.json',
              B5C3 / 'prepared_inputs.json'] + \
             [d / f for d in [E2 / w for w in ('w1', 'w2', 'w3', 'w4')]
              for f in ('strict_epoch_snapshot.json', 'physical_state_census.json', 'prepared_inputs.json')]:
        watched[str(p)] = sha_bytes(p.read_bytes())

    results = {'A_strict20': [], 'B_ck3': {}, 'C_failures': [], 'boundary': []}

    # ── A. Strict 20：五个快照 × K1–4，全部应走 Exact 且与冻结 Oracle 一致 ──
    ref = {(r['case'], int(r['K']), r['policy']): r
           for r in csv.DictReader(open(E3CSV))}
    snapshots = [('w1', E2 / 'w1'), ('w2', E2 / 'w2'), ('w3', E2 / 'w3'),
                 ('w4', E2 / 'w4'), ('B5-C4', B5C3)]
    for name, directory in snapshots:
        for k in (1, 2, 3, 4):
            epoch, evidence = build_epoch(directory, k)
            opt = GlobalOptimizer(RecoveryCostModel())
            plan = opt.plan_strict(epoch)
            decision = opt.last_strict_decision
            validate_plan(epoch, plan)  # 生产只读校验：预算、frontiers 复算、驱逐补集
            assert decision['solver'] == 'Exact' and decision['gate_allowed'], (name, k, decision)
            assert decision['gate']['candidates'] == 7 and decision['gate']['pendings'] == 4
            assert decision['gate']['budget_k'] == k
            assert decision['gate']['enumerable_subsets'] == sum(
                math.comb(7, i) for i in range(k + 1))
            assert len(plan.selected_ids) <= k and len(set(plan.selected_ids)) == len(plan.selected_ids)
            assert decision['objective_ms'] == plan.recovery_cost_ms
            # e3 冻结 Exact 参照：选集与 J
            erow = ref[(name, k, 'Exact')]
            assert set(plan.selected_ids) == set(json.loads(erow['selected'])), (name, k, '选集≠e3 Exact')
            assert abs(plan.recovery_cost_ms - float(erow['J_ms'])) <= EPS, (name, k, 'J≠e3 Exact')
            # b5d1 独立 Oracle（同门禁枚举的另一实现路径）
            prepared = planner.prepare(epoch, evidence)
            matrix = planner.benefits(prepared)
            assert tuple(plan.selected_ids) == planner.solve(prepared, matrix, 'Exact'), (name, k, '≠b5d1 Oracle')
            # Greedy 行为不变性：直接调用原 _select_candidates（同一评分）应复现 e3 Greedy 行
            score_ids = strict_score_ids(epoch)
            size = epoch.candidates[0].candidate.memory_bytes
            greedy = opt._select_candidates(
                tuple(c.candidate for c in epoch.candidates), epoch.K * size,
                lambda sel: score_ids(tuple(c.checkpoint_id for c in sel)))
            grow = ref[(name, k, 'Greedy')]
            assert {c.checkpoint_id for c in greedy.selected} == set(json.loads(grow['selected'])), (name, k, 'Greedy 行为改变')
            results['A_strict20'].append({
                'case': name, 'K': k, 'solver': decision['solver'], 'gate': decision['gate'],
                'selected': sorted(plan.selected_ids), 'J_ms': plan.recovery_cost_ms,
                'pure_solve_ms': decision['pure_solve_ms'],
                'greedy_unchanged': True, 'matches_e3_exact': True, 'matches_b5d1_oracle': True})
    assert len(results['A_strict20']) == 20

    # ── B. C/K3：n=16 超门禁 → 自动 Marginal Greedy，复现冻结 FlowState-Greedy 选集与 J ──
    rec = read_json(CFG / 'g1_CB_FS' / 'run' / 'reconciliation.json')
    raw = rec['raw_epoch']
    data = encode(raw)
    assert sha_bytes(data) == rec['epoch']['snapshot_sha256'], 'raw_epoch 字节与冻结 snapshot_sha256 不一致'
    # physical 视图按冻结运行推导：prepare 当时已通过 ⇒ unchanged=True；window 与 bind 同值（记录于 apply_validation）。
    initial_view = {'before': rec['before'], 'audit_before': rec['audit_before'],
                    'unchanged': True, 'window': rec['apply_validation']['window']}
    inputs = read_json(CFG / 'g1_CB_FS' / 'prepared_inputs.json')
    origins = rec['origins']
    physical = encode(initial_view)
    evidence = planner.FrozenEvidence(data, sha_bytes(data),
                                      physical_snapshot=physical, physical_sha256=sha_bytes(physical))
    epoch = planner.from_frozen(evidence, 3, (initial_view, origins, inputs))
    assert len(epoch.candidates) == 16 and len(epoch.pendings) == 4 and epoch.K == 3
    opt = GlobalOptimizer(RecoveryCostModel())
    plan = opt.plan_strict(epoch)
    decision = opt.last_strict_decision
    validate_plan(epoch, plan)
    assert decision['solver'] == 'Marginal Greedy' and not decision['gate_allowed']
    assert decision['gate']['candidates'] == 16 and decision['gate']['pendings'] == 4 \
        and decision['gate']['budget_k'] == 3
    plans = read_json(CFG / 'policy_plans.json')
    cpu_rows = [r for r in read_json(MW / 'cpu_results.json')['rows']
                if r['structure'] == 'C' and r['K'] == 3]
    cpu_fs = next(r for r in cpu_rows if r['policy'] == 'FlowState-Greedy')
    # policy_plans/cpu_results 存 CPU 侧 ID；用冻结 census 的 cpu↔physical 映射换算后比对
    cpu_to_phys = {r['cpu_checkpoint_id']: r['physical_checkpoint_id']
                   for r in rec['census']['mapped']}
    assert len(cpu_to_phys) == 16
    plans_phys = {cpu_to_phys[c] for c in plans['FlowState-Greedy']['selected_checkpoint_ids']}
    cpu_phys = {cpu_to_phys[c] for c in cpu_fs['selected']}
    assert set(plan.selected_ids) == set(rec['plan']['selected_ids']) \
        == plans_phys == cpu_phys
    assert abs(plan.recovery_cost_ms - rec['plan']['recovery_cost_ms']) <= EPS
    assert abs(plan.recovery_cost_ms - cpu_fs['current_J_ms']) <= EPS
    assert abs(plan.recovery_cost_ms - 154.34895561599998) <= EPS
    results['B_ck3'] = {'solver': decision['solver'], 'gate': decision['gate'],
                        'selected': sorted(plan.selected_ids), 'J_ms': plan.recovery_cost_ms,
                        'pure_solve_ms': decision['pure_solve_ms'],
                        'matches_frozen_plan': True, 'matches_policy_plans': True,
                        'matches_cpu_results_J': True}

    # ── C. 失败路径：合同错误/求解器异常一律 Fail-stop，不得回退、不得产出部分计划 ──
    base_dir = E2 / 'w1'
    snapshot = read_json(base_dir / 'strict_epoch_snapshot.json')
    census_bytes = (base_dir / 'physical_state_census.json').read_bytes()
    physical_dict = json.loads(census_bytes)
    inputs_w1 = read_json(base_dir / 'prepared_inputs.json')

    def epoch_from_dict(s):
        """用损坏快照字典重建 Epoch（字节自洽，不含外部期望摘要），使错误由 validate 暴露。"""
        data = encode(s)
        evidence = planner.FrozenEvidence(data, sha_bytes(data),
                                          physical_snapshot=census_bytes,
                                          physical_sha256=sha_bytes(census_bytes))
        # 来源仅影响 from_frozen 取 history 字段，不做兼容判断；给一个真实存在的请求 ID 即可
        first = inputs_w1['history'][0]['request_id']
        origins = [{'checkpoint_id': c['checkpoint_id'], 'first_observed_after': first}
                   for c in s['candidates']]
        return planner.from_frozen(evidence, 2, (physical_dict, origins, inputs_w1))

    # C1 不完整矩阵：删除一条边 → 完整 n×m 矩阵断裂
    incomplete = dict(snapshot)
    incomplete['edges'] = snapshot['edges'][:-1]
    opt = GlobalOptimizer(RecoveryCostModel())
    try:
        opt.plan_strict(epoch_from_dict(incomplete))
        raise AssertionError('C1: 不完整 Epoch 未被拒绝')
    except AssertionError:
        raise
    except Exception as exc:
        results['C_failures'].append({'case': 'incomplete_matrix', 'error': repr(exc),
                                      'fallback_to_greedy': False, 'plan_returned': False})
        assert opt.last_strict_decision is None, 'C1: 失败路径不得记录成功决策'

    # C2 UNKNOWN 边：入口即拒，不得回退
    unknown = dict(snapshot)
    unknown['edges'] = [dict(snapshot['edges'][0], status='UNKNOWN')] + snapshot['edges'][1:]
    opt = GlobalOptimizer(RecoveryCostModel())
    try:
        opt.plan_strict(epoch_from_dict(unknown))
        raise AssertionError('C2: UNKNOWN 边未被拒绝')
    except AssertionError:
        raise
    except Exception as exc:
        results['C_failures'].append({'case': 'unknown_edge', 'error': repr(exc),
                                      'fallback_to_greedy': False, 'plan_returned': False})
        assert opt.last_strict_decision is None, 'C2: 失败路径不得记录成功决策'

    # C3 求解器内部异常：注入故障，断言异常上抛、无部分计划、无成功决策
    epoch_ok, _ = build_epoch(base_dir, 2)
    opt = GlobalOptimizer(RecoveryCostModel())
    with patch.object(GlobalOptimizer, '_enumerate_exact_subsets',
                      side_effect=RuntimeError('注入求解器异常')):
        try:
            opt.plan_strict(epoch_ok)
            raise AssertionError('C3: 求解器异常未上抛')
        except AssertionError:
            raise
        except RuntimeError as exc:
            results['C_failures'].append({'case': 'solver_exception_injected', 'error': repr(exc),
                                          'fallback_to_greedy': False, 'plan_returned': False})
    assert opt.last_strict_decision is None, 'C3: 异常路径不得记录成功决策'

    # ── 门禁边界纯函数判定 + 全体运行中"门禁判定↔实际求解器"一致性 ──
    for n, p, k, expect in [(12, 4, 4, True), (13, 4, 4, False), (12, 5, 4, False),
                            (12, 4, 5, False), (11, 4, 4, True), (16, 4, 3, False)]:
        allowed, gate = _exact_admission(n, p, k)
        assert allowed is expect, (n, p, k, gate)
        assert gate['enumerable_subsets'] == sum(
            math.comb(n, i) for i in range(min(n, k) + 1))
        results['boundary'].append({'n': n, 'pendings': p, 'K': k,
                                    'allowed': allowed, 'subsets': gate['enumerable_subsets']})
    for row in results['A_strict20']:
        assert row['solver'] == 'Exact' and row['gate']['candidates'] <= 12
    assert results['B_ck3']['solver'] == 'Marginal Greedy'

    # ── 首尾完整性：被测源码与全部冻结输入未变 ──
    for p, h in watched.items():
        assert sha_bytes(Path(p).read_bytes()) == h, f'文件在门禁期间被修改：{p}'

    out = {'verdict': 'PASS', 'tested_entry': 'flowstate.optimizer.GlobalOptimizer.plan_strict',
           'results': results,
           'integrity': {'files_verified': len(watched), 'unchanged': True}}
    (EVAL / 'paper_hybrid_solver_integration_gate.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    exact_times = [r['pure_solve_ms'] for r in results['A_strict20']]
    print('A Strict20：20/20 走 Exact、J/选集==e3 与 b5d1 双参照；Greedy 行为 20/20 不变')
    print('   Exact 纯求解时间 ms：min=%.4f median=%.4f max=%.4f' % (
        min(exact_times), sorted(exact_times)[len(exact_times) // 2], max(exact_times)))
    print('B C/K3：门禁拒绝(n=16)→Marginal Greedy；选集==冻结 FS；J=%.6f' % results['B_ck3']['J_ms'])
    print('C 失败路径：%d/3 Fail-stop（无回退、无部分计划、无成功决策记录）' % len(results['C_failures']))
    print('边界判定：', [(b['n'], b['K'], b['allowed']) for b in results['boundary']])
    print('判定：PASS（CPU 门禁；不等于在线 GPU 端到端验证）')


if __name__ == '__main__':
    main()
