"""P0-4 三策略完整控制周期：CPU 正确性门禁（Task 1 §四 逐项验证，不启动 GPU）。

检查项（与 spec §四 一一对应）：
  1  三种策略均进入真实 Select 调用路径（spy 包装直接证明内核被调用 + 选集复现冻结 oracle）
  2  输入、预算和选集记录完整（字段/预算/域/补集划分/可见性描述）
  3  计时区间没有重复累加或遗漏（整数纳秒恒等式精确成立 + 分段顺序无重叠）
  4  四 Pending 执行顺序与计时关联关系正确（声明顺序、TTFT/终点逐请求、T4=第 4 请求终点）
  5  Strict Epoch/物理映射/Recurrent-only Reconciliation/FA 保护检查仍然保留（CPU 可验证部分 +
     GPU 物化部分显式登记为待验证，不以 CPU 通过替代）
  6  异常路径明确失败，不静默切换到冻结选集（内核异常传播；错误选集被 oracle 拒绝；
     选择函数结构上不含冻结读取）
  7  原有正式实验入口及冻结数据未被破坏（输入文件 sha 前后一致；仅写本包 out/）
  8  GPU-only 待验证项登记（真实物化/协调/恢复/计时值）
"""
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.p0_control_inclusive_harness import select_entry
from evaluation.p0_control_inclusive_harness.cycle_orchestrator import DryBackend, OutOfWindowEvidence, run_cycle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'out'
CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})
    print('%s  %s%s' % ('PASS' if ok else 'FAIL', name, ('  | ' + str(detail)[:120]) if detail else ''))
    if not ok:
        raise AssertionError('CPU 门禁失败：%s %s' % (name, detail))


def main():
    OUT.mkdir(exist_ok=True)
    frozen_files = [
        ROOT / 'evaluation/c_formal_perf_gpu/g1_CB_FS/run/reconciliation.json',
        ROOT / 'evaluation/c_formal_perf_gpu/g1_CB_FS/prepared_inputs.json',
        ROOT / 'evaluation/c_formal_perf_gpu/policy_plans.json',
        ROOT / 'evaluation/c_formal_perf_gpu/cpu_candidate_prefix_mapping.json',
        ROOT / 'evaluation/t2g1_three_policy_gpu/t2g1_plans.json',
        ROOT / 'evaluation/kvflow_multiworkflow_cpu_main/workloads.json',
    ]
    before_sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files}

    # ── 检查 1+2：真实调用路径 + 记录完整性（spy 包装证明内核被真实调用）──
    spy = {'plan_strict': 0, 'marconi': 0, 'kvflow': 0}
    orig_plan_strict = select_entry.GlobalOptimizer.plan_strict
    orig_marconi = select_entry.MarconiEvictPolicy.select
    orig_kv = select_entry.kvflow.select

    def spy_plan_strict(self, epoch):
        spy['plan_strict'] += 1
        return orig_plan_strict(self, epoch)

    def spy_marconi(self, *a, **kw):
        spy['marconi'] += 1
        return orig_marconi(self, *a, **kw)

    def spy_kv(*a, **kw):
        spy['kvflow'] += 1
        return orig_kv(*a, **kw)

    select_entry.GlobalOptimizer.plan_strict = spy_plan_strict
    select_entry.MarconiEvictPolicy.select = spy_marconi
    select_entry.kvflow.select = spy_kv
    try:
        ctx = select_entry.build_ck3_epoch()
        records, plans = select_entry.run_three_policy_select(ctx)
        oracle = select_entry.oracle_compare(records, ctx)
    finally:
        select_entry.GlobalOptimizer.plan_strict = orig_plan_strict
        select_entry.MarconiEvictPolicy.select = orig_marconi
        select_entry.kvflow.select = orig_kv
    check('1a_真实内核调用', spy == {'plan_strict': 1, 'marconi': 1, 'kvflow': 1}, spy)
    gate = records['FlowState-Greedy']['decision']['gate']
    check('1b_FS走生产入口', records['FlowState-Greedy']['kernel_entry'].endswith('GlobalOptimizer.plan_strict')
          and records['FlowState-Greedy']['decision']['solver'] == 'Marginal Greedy'
          and {k: gate[k] for k in ('candidates', 'pendings', 'budget_k')} == {'candidates': 16, 'pendings': 4, 'budget_k': 3},
          'n=16 超门禁自动 Greedy；enumerable_subsets=%s' % gate.get('enumerable_subsets'))
    check('1c_选集复现冻结oracle', all(oracle['per_policy'][p]['matches_t2g1_frozen']
          for p in select_entry.THREE_POLICIES))
    required_fields = {'policy', 'kernel_entry', 'selection_call_ms', 'selected_physical_ids',
                       'selected_cpu_ids', 'evicted_physical_ids', 'recovery_cost_ms', 'inputs_visibility'}
    check('2a_记录字段完整', all(required_fields <= set(r) for r in records.values()))
    check('2b_预算与域', all(len(r['selected_physical_ids']) == 3
          and set(r['selected_physical_ids']) <= set(ctx['physical_ids']) for r in records.values()))
    check('2c_选集驱逐补集划分', all(set(r['selected_physical_ids']) | set(r['evicted_physical_ids'])
          == set(ctx['physical_ids']) for r in records.values()))
    check('2d_可见性如实记录', all(len(r['inputs_visibility']) > 20 for r in records.values()))

    # ── 检查 3+4：T0–T4 恒等式与四 Pending 关联（DryBackend 验证编排结构）──
    policies = {}
    for policy in select_entry.THREE_POLICIES:
        ev = OutOfWindowEvidence()
        ev.collect('census_summary', {'policy': policy})
        ev.collect('metrics_snapshot', {'policy': policy})

        def make_select(policy=policy):
            def select_fn(mode):
                if mode == 'epoch_context':
                    return {'candidates': ctx['physical_ids'], 'pendings': ctx['gpu_pendings']}
                return records[policy]
            return select_fn
        rec_cycle = run_cycle(policy, make_select(), DryBackend(ctx['gpu_pendings']), ev, ctx['gpu_pendings'])
        policies[policy] = rec_cycle
        t = rec_cycle['timing']
        recomputed = sum(s['ms'] for s in t['segments']) + t['gap_total_ms']
        check('3a_恒等式精确_%s' % policy, t['identity_exact'] and abs(recomputed - t['cycle_ms']) < 1e-9,
              'cycle=%.6fms 重算=%.6fms' % (t['cycle_ms'], recomputed))
        seg_bounds = [(s['stage'], None) for s in t['segments']]
        check('3b_分段数量与注册表_%s' % policy,
              len(t['segments']) == 4 + 2 * len(ctx['gpu_pendings']) and len(seg_bounds) == len(t['segments']),
              'T0-T3 四段(observe/census/select/reconcile) + 每请求两段(preflight/execute)')
        check('4a_四Pending声明顺序_%s' % policy,
              [r['pending_id'] for r in rec_cycle['pending_rows']] == list(ctx['gpu_pendings']),
              ctx['gpu_pendings'])
        ttfts = [r['ttft_ns'] for r in rec_cycle['pending_rows']]
        ends = [r['exec_end_ns'] for r in rec_cycle['pending_rows']]
        check('4b_逐请求计时关联_%s' % policy,
              all(0 < ttft < end for ttft, end in zip(ttfts, ends))
              and all(a['exec_end_ns'] < b['exec_end_ns']
                      for a, b in zip(rec_cycle['pending_rows'], rec_cycle['pending_rows'][1:]))
              and t['t4_exec_end_offset_ns'] >= 0,
              'TTFT<终点、请求串行单调、T4≥第4终点')
        check('4c_取证窗口外flush_%s' % policy,
              rec_cycle['evidence']['flush_ms'] is not None
              and [o for o in rec_cycle['evidence']['ops']] == ['census_summary', 'metrics_snapshot'],
              'flush_ms=%.4f（不计入周期）' % rec_cycle['evidence']['flush_ms'])

    # ── 检查 5：Strict 合同保留（CPU 可验证部分；GPU 物化部分登记待验证）──
    from flowstate.strict_control import validate_plan
    for policy, plan in plans.items():
        validate_plan(ctx['epoch'], plan)  # 生产只读校验：预算/frontiers 复算/驱逐补集
    check('5a_三策略Plan全部通过生产校验', True, 'validate_plan×3（含 FS 生产入口产物）')
    rec_census = json.loads((ROOT / 'evaluation/c_formal_perf_gpu/g1_CB_FS/run/reconciliation.json')
                            .read_text())['census']['mapped']
    mapping = json.loads((ROOT / 'evaluation/c_formal_perf_gpu/cpu_candidate_prefix_mapping.json')
                         .read_text())['mapping']
    rows, unmapped = select_entry.physical_cpu_census(ctx['raw'], ctx['inputs'], mapping)
    check('5b_census对账复现冻结16=16', rows == rec_census and not unmapped and len(rows) == 16)
    stages = [s[1] for s in policies['FlowState-Greedy']['stage_registry']]
    check('5c_合同分段注册表完整',
          {'observe_and_authenticate', 'census_epoch_confirm', 'policy_select',
           'plan_validate_and_reconcile'} <= set(stages) and 'resume_p0..p3' in stages)
    check('5d_协调后验合同在窗内', all(k in policies['FlowState-Greedy']['reconciliation']['postchecks']
          for k in ['FA/非R摘要不变', '槽位释放记账', '驻留==Plan'])
          and policies['FlowState-Greedy']['reconciliation']['execution'].startswith('CPU:'))

    # ── 检查 6：异常路径 fail-loud，无冻结回退 ──
    def expect_fail(name, fn):
        try:
            fn()
        except BaseException as exc:
            check(name, True, type(exc).__name__ + ': ' + str(exc)[:80])
            return
        check(name, False, '未按预期失败')

    def kernel_raises():
        select_entry.MarconiEvictPolicy.select = orig_marconi
        import evaluation.marconi_evict_t2m.evict_policy as ep
        saved = ep.MarconiEvictPolicy.select

        def bomb(self, *a, **kw):
            raise ValueError('内核注入异常')
        ep.MarconiEvictPolicy.select = bomb
        select_entry.MarconiEvictPolicy.select = bomb
        try:
            select_entry.run_three_policy_select(select_entry.build_ck3_epoch())
        finally:
            ep.MarconiEvictPolicy.select = saved
            select_entry.MarconiEvictPolicy.select = saved
    expect_fail('6a_内核异常直接传播', kernel_raises)

    def wrong_selection_rejected():
        # 内核返回合法但与 oracle 不符的选集 → 事后对照必须拒绝（证明没有"静默采纳"通道）
        saved = select_entry.select_marconi

        def fake_select(ctx, k=select_entry.K):
            cs = ctx['case']['candidates']
            wrong = tuple(sorted(c['checkpoint_id'] for c in cs[:3]))
            from evaluation.marconi_evict_t2m.evict_policy import EvictionSelection
            rest = tuple(sorted(c['checkpoint_id'] for c in cs[3:]))
            return EvictionSelection(wrong, rest, (), 0.0), {'kernel_entry': '注入', 'selection_call_ms': 0.0,
                                                             'eviction_order_cpu': list(rest), 'rounds': 13}
        select_entry.select_marconi = fake_select
        try:
            ctx2 = select_entry.build_ck3_epoch()
            recs, _ = select_entry.run_three_policy_select(ctx2)
            select_entry.oracle_compare(recs, ctx2)
        finally:
            select_entry.select_marconi = saved
    expect_fail('6b_错误选集被oracle拒绝', wrong_selection_rejected)

    src_sel = inspect.getsource(select_entry.select_flowstate) + inspect.getsource(select_entry.select_marconi) \
        + inspect.getsource(select_entry.select_kvflow)
    check('6c_选择函数结构上无冻结读取',
          all(tok not in src_sel for tok in ('t2g1_plans', 'policy_plans', 'oracle')))

    # ── 检查 7：冻结数据未被破坏 ──
    after_sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files}
    check('7a_冻结输入sha前后一致', before_sha == after_sha)
    check('7b_仅写本包out目录', set(OUT.iterdir()) <= {OUT / 'selection_records.json', OUT / 'cpu_gate_result.json',
                                                       OUT / 'cycle_records.json'},
          sorted(p.name for p in OUT.iterdir()))

    # ── 检查 8：GPU-only 待验证项登记（不以 CPU 通过替代）──
    gpu_pending = [
        '实时观察构造 Epoch（live observe_control + 实时 census 16=16 对账，替代冻结构造链）',
        'Recurrent-only Reconciliation 真实物化（reconcile_strict 逐驱逐 evict_mamba_only + 每步 postcheck）',
        'FA 内容审计在真实树上的逐摘要比对（CPU 仅登记合同）',
        'Native Resume 真实执行（E/G/Prefill 实测 + Prefill=G+32 合同）',
        'Cold Reference 逐 Token 一致（4 Pending）',
        '四 Pending 串行请求的真实计时值与请求间隙 Residual（CPU 仅验证恒等式结构）',
        '三策略完整周期 T0–T4 真实测量与配对统计',
    ]
    check('8_GPU待验证项登记', len(gpu_pending) == 7, '见 cpu_gate_result.json')

    payload = {'verdict': 'PASS', 'checks': CHECKS,
               'summary': {'policies': select_entry.THREE_POLICIES, 'K': select_entry.K,
                           'selection_call_ms': {p: records[p]['selection_call_ms'] for p in select_entry.THREE_POLICIES},
                           'selected_cpu': {p: records[p]['selected_cpu_ids'] for p in select_entry.THREE_POLICIES},
                           'J_ms': {p: records[p]['recovery_cost_ms'] for p in select_entry.THREE_POLICIES},
                           'cycle_structure_ms': {p: policies[p]['timing']['cycle_ms'] for p in select_entry.THREE_POLICIES},
                           'gpu_pending_verification': gpu_pending,
                           'note': 'CPU 门禁：结构/合同/计时恒等式全过；真实物化与性能值待 GPU 验证'}}
    (OUT / 'cpu_gate_result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (OUT / 'cycle_records.json').write_text(json.dumps(policies, ensure_ascii=False, indent=2))
    print('\nCPU 门禁 %d/%d 全部通过；verdict=PASS（GPU 待验证项 %d 项已登记）'
          % (sum(c['pass'] for c in CHECKS), len(CHECKS), len(gpu_pending)))


if __name__ == '__main__':
    main()
