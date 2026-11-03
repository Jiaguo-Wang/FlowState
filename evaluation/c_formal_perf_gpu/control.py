"""C Shared/Branched K=3 GPU Pilot 控制路径：把冻结 CPU 策略选择映射到真实物化状态并执行严格协调。

与 kvflow_native_recovery_diagnostic/control.py 的差别只在选择来源：
- 诊断轮按位置挑选全集/祖先做语义诊断；
- 本轮按 policy_plans.json 的冻结 CPU 选择（FlowState-Greedy/StandaloneTopK/WFC）
  经 (预热水前缀逐Token相等, L) 唯一映射到物理 B5C3_NODE_* 候选，其余合同不变。
不调用 planner.allowed：该上限只约束 Exact 枚举求解的合法输入规模；
本轮使用 CPU 冻结选集，仍完整经过 from_frozen/prepare/finish 的全部合同检查。
"""
from evaluation.exact_reconciliation_correctness_gate_b5d3.control import *
import os


def physical_cpu_census(raw, inputs, mapping):
    """物理驻留候选逐个映射回 CPU checkpoint_id：按 (预热水截断前缀逐Token相等, L) 唯一匹配。

    返回 (已映射行, 未能映射的额外物理状态)。A 与 JOIN 同为 4096/4032 长，
    但 Token 内容不同，按 (预热请求, L) 匹配天然区分。
    """
    bywarm = {h['request_id']: h['token_ids'] for h in inputs['history']}
    index = {}
    for cpu_id, row in mapping.items():
        index.setdefault((row['warmup_round0'], row['L']), []).append(cpu_id)
    rows = []
    unmapped = []
    for c in raw['candidates']:
        toks = c['token_ids']
        L = c['token_position_L']
        hits = [(rid, cpu) for (rid, l), ids in index.items() if l == L and bywarm[rid][:L] == toks for cpu in ids]
        if len(hits) == 1:
            rid, cpu = hits[0]
            rows.append({'physical_checkpoint_id': c['checkpoint_id'],
                         'node_id': c['state_identity']['node_id'],
                         'L': L, 'slots': c['state_identity']['target_mamba_slots'],
                         'cpu_checkpoint_id': cpu, 'first_materialized_request': rid})
        else:
            unmapped.append({'physical_checkpoint_id': c['checkpoint_id'],
                             'node_id': c['state_identity']['node_id'], 'L': L,
                             'match_count': len(hits)})
    return rows, unmapped


def apply(scheduler, request):
    result = {'ok': False, 'status': 'BLOCKED', 'operations': [], 'timing': [], 'observations': [], 'phase': 'start'}

    def timed(stage, fn):
        t = time.monotonic_ns()
        v = fn()
        result['timing'].append({'stage': stage, 'ms': (time.monotonic_ns() - t) / 1e6})
        return v

    try:
        initial = timed('final_heavy_observation', lambda: capture.observe_control(scheduler, {'final': True}))
        require(initial.get('ok'), '当前观察失败')
        scheduler._r2c0_owner = threading.get_ident()
        window = initial['window']
        prior.scope(scheduler)
        raw = initial['strict_epoch']
        result['raw_epoch'] = raw  # e1 bridge b5d4_apply 成功路径要求（attempt2 教训：协调全过但缺此键）
        inputs = json.loads((OUT / 'prepared_inputs.json').read_text())
        mapping = json.loads((OUT / 'cpu_candidate_prefix_mapping.json').read_text())['mapping']
        plans = json.loads((OUT / 'policy_plans.json').read_text())
        policy = os.environ['PILOT_POLICY']
        require(policy in plans, '未知策略：' + policy)

        # ── CPU 候选域 ↔ 物理候选域 对账：所有保留候选必须是决策前真实产生的状态 ──
        rows, unmapped = timed('cpu_physical_mapping', lambda: physical_cpu_census(raw, inputs, mapping))
        cpu_to_phys = {r['cpu_checkpoint_id']: r['physical_checkpoint_id'] for r in rows}
        missing_physical = sorted(set(mapping) - set(cpu_to_phys))
        result['census'] = {'mapped': rows, 'unmapped_extras': unmapped,
                            'cpu_domain_size': len(mapping), 'physical_domain_size': len(raw['candidates']),
                            'missing_physical': missing_physical,
                            'note': '未映射的额外物理状态一律视为未选中并在协调中真实驱逐，不免费保留任何祖先'}

        selected_cpu = tuple(plans[policy]['selected_checkpoint_ids'])
        require(len(selected_cpu) == 3, '冻结选择数与 K=3 不符')
        absent = [c for c in selected_cpu if c not in cpu_to_phys]
        require(not absent, '所选 CPU 状态未在物理域物化：' + repr(absent))
        selected = tuple(cpu_to_phys[c] for c in selected_cpu)

        # ── 来源证据：本轮 16 条预热的真实观察记录（runner 写入 history_XX.json）──
        histories = [json.loads((OUT / os.environ['B5D4_RUN'] / f'history_{i:02d}.json').read_text())
                     for i in range(1, len(inputs['history']) + 1)]
        origins = []
        phys_to_cpu = {r['physical_checkpoint_id']: r for r in rows}
        rounds = []
        for c in raw['candidates']:
            seen = [h['request']['request_id'] for h in histories
                    if any(n['node_id'] == c['state_identity']['node_id'] and n['token_ids'] == c['token_ids']
                           and n['slots'] for n in h['state']['before']['nodes'])]
            require(bool(seen), '缺少本生命周期来源线索')
            # 冻结时间线合同：候选必须最早物化于其 CPU 声明预热水（round0 或 round1，均属冻结 16 请求）。
            # attempt1 证实：末端(L)物化于 round0，祖先(L-64)由 round1 重复插入产生——两者都在决策前，
            # 不利用任何未来请求；只记录实测轮次，不把 round0==末端 写成硬断言。
            if c['checkpoint_id'] in phys_to_cpu:
                m = phys_to_cpu[c['checkpoint_id']]
                w0 = mapping[m['cpu_checkpoint_id']]['warmup_round0']
                w1 = mapping[m['cpu_checkpoint_id']]['warmup_round1']
                require(seen[0] in {w0, w1}, '物化时间线超出冻结预热声明：' + c['checkpoint_id'])
                wlen = next(len(h['token_ids']) for h in inputs['history'] if h['request_id'] == w0)
                rounds.append({'cpu_checkpoint_id': m['cpu_checkpoint_id'], 'L': m['L'],
                               'observed_first_request': seen[0],
                               'round': 'round0' if seen[0] == w0 else 'round1',
                               'state_kind': 'endpoint' if m['L'] == wlen else
                                             ('ancestor_L_minus_64' if m['L'] == wlen - 64 else 'unexpected_position')})
            origins.append({'checkpoint_id': c['checkpoint_id'], 'first_observed_after': seen[0]})
        result['census']['materialization_rounds'] = rounds

        def authenticate():
            data = encode(raw)
            physical = encode(initial)
            evidence = planner.FrozenEvidence(data, hashlib.sha256(data).hexdigest(),
                                              physical_snapshot=physical,
                                              physical_sha256=hashlib.sha256(physical).hexdigest())
            epoch = planner.from_frozen(evidence, len(selected), (initial, origins, inputs))
            prepared = planner.prepare(epoch, evidence)
            plan = planner.finish(prepared, selected)
            # 冻结选集直接构建计划；不调用 planner.allowed（Exact 枚举规模门禁，见模块说明）
            return epoch, prepared, plan

        epoch, prepared, plan = timed('strict_epoch_authentication_and_plan', authenticate)
        result.update(policy=policy, budget_K=3, policy_selected_cpu_ids=list(selected_cpu),
                      selected_physical_ids=list(selected), epoch=asdict(epoch), plan=asdict(plan), origins=origins)

        # ── CPU↔GPU 前沿合同：四个 Pending 的 E/T/G 必须与 CPU 冻结预期完全一致 ──
        expect = plans[policy]['expected_E_G']
        for prow in plan.pending:
            e = expect[prow.pending_id]
            require((prow.E, prow.T, prow.G) == (e['E'], e['T'], e['G']),
                    'GPU 实测前沿与 CPU 冻结预期不符：' + prow.pending_id)
        for p in raw['pendings']:
            declared = next(x for x in inputs['pendings'] if x['continuation_id'] == p['continuation_id'])
            require(p['planning_target_T'] == declared['planning_target_T'] and
                    p['attention_frontier'] == declared['attention_frontier'],
                    '活观察 T/FA前沿 与 CPU 声明不一致：' + p['continuation_id'])

        # ── 目标 Pending：按环境变量选择；E>0 须存在唯一合法恢复源 ──
        target_id = os.environ['PILOT_PENDING']
        tp = next(p for p in raw['pendings'] if p['continuation_id'] == target_id)
        expected = next(x for x in plan.pending if x.pending_id == target_id)
        if expected.E > 0:
            cands = [c for c in raw['candidates'] if c['checkpoint_id'] in selected and
                     c['token_ids'] == tp['token_ids'][:c['token_position_L']] and
                     c['token_position_L'] == expected.E]
            require(len(cands) == 1, '目标没有唯一合法恢复源')
            target = {'pending': tp, 'source_checkpoint': cands[0]['checkpoint_id'], 'expected': asdict(expected)}
        else:
            target = {'pending': tp, 'source_checkpoint': None, 'expected': asdict(expected),
                      'note': '该策略下目标无兼容保留状态，E=0，无恢复源；FA 命中照常认证'}
        result['target'] = target

        def check():
            before, audit = prior.observe(scheduler, epoch, window)
            require(asdict(before) == initial['before'] and audit['semantic'] == initial['audit_before']['semantic'], 'Apply前状态变化')
            require(capture.payload(scheduler.tree_cache, before) == initial['payload_before'], 'Apply前组件内容变化')
            from evaluation.barrier_fa_frontier_control import inspect_resident_fa_frontier
            for p in raw['pendings']:
                require(inspect_resident_fa_frontier(scheduler.tree_cache, p['token_ids'])['resident_fa_frontier']
                        == p['attention_frontier'], 'Apply前FA前沿变化')
            binding = bind(epoch, plan, before, window)
            preflight(epoch, plan, binding, before, window)
            return before, audit, binding

        before, audit, binding = timed('apply_time_freshness_preflight', check)
        result.update(before=asdict(before), audit_before=audit,
                      apply_validation={'status': 'PASS', 'binding': asdict(binding), 'window': window, 'same_scheduler_call': True})

        t = time.monotonic_ns()
        old_fa = fa_map(scheduler.tree_cache)
        indices = set(old_fa.values())
        full_hash = fa_payload(scheduler.tree_cache, indices)
        result['timing'].append({'stage': 'fa_content_audit_before', 'ms': (time.monotonic_ns() - t) / 1e6})
        result.update(selected_payload_before=selected_payload(scheduler.tree_cache, before, plan, epoch))

        @contextmanager
        def held():
            prior.scope(scheduler)
            yield window
            prior.scope(scheduler)

        def observe():
            view, a = prior.observe(scheduler, epoch, window)
            require(a['state']['tree']['structure_sha256'] == audit['state']['tree']['structure_sha256'], '协调过程FA树结构变化')
            result['observations'].append(asdict(view))
            return view

        def invalidate(ledger):
            scheduler._r2c0_failed = True

        t = time.monotonic_ns()
        runtime = StrictRuntimeBridge(RecordedAdapter(scheduler.tree_cache, result['operations']), held, observe, invalidate)
        ledger = StateController.reconcile_strict(epoch, plan, runtime)
        result['timing'].append({'stage': 'strict_reconciliation_with_postchecks', 'ms': (time.monotonic_ns() - t) / 1e6})
        after, last = prior.observe(scheduler, epoch, window)
        source_after = selected_payload(scheduler.tree_cache, after, plan, epoch)
        require(source_after == result['selected_payload_before'], '所选组件内容变化')
        require([x['checkpoint_id'] for x in result['operations']] == list(plan.logical_eviction_ids)
                and all(x['returned'] for x in result['operations']), '驱逐未完整落实')

        t = time.monotonic_ns()
        require(fa_map(scheduler.tree_cache) == old_fa and fa_payload(scheduler.tree_cache, indices) == full_hash, '协调改变FA索引或内容')
        # 目标请求可能重算 E 之后区间的 FA；只保护其余原始 FA 内容
        digest = hashlib.sha256()
        recomputed = set()
        for pos, token in enumerate(target['pending']['token_ids']):
            digest.update(int(token).to_bytes(8, 'little'))
            if pos >= target['expected']['E'] and digest.hexdigest() in old_fa:
                recomputed.add(old_fa[digest.hexdigest()])
        protected = sorted(indices - recomputed)
        protected_hash = fa_payload(scheduler.tree_cache, protected)
        result['timing'].append({'stage': 'fa_content_postcheck_and_resume_baseline', 'ms': (time.monotonic_ns() - t) / 1e6})
        result.update(ok=True, status='READY', phase='complete', reconciliation=ledger, after=asdict(after),
                      audit_after=last, selected_payload_after=source_after, fa_map_before=old_fa,
                      fa_protected_indices=protected, fa_protected_payload=protected_hash,
                      fa_expected_recomputed_indices=sorted(recomputed))
        prior.scope(scheduler)
        return result, epoch, plan
    except BaseException as exc:
        if isinstance(exc, StrictReconciliationError):
            result['reconciliation'] = exc.ledger
        scheduler._r2c0_failed = True
        result.update(error=repr(exc), traceback=traceback.format_exc(), status='BLOCKED')
        return result, None, None
    finally:
        scheduler._r2c0_owner = None
