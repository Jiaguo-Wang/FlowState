"""四 Pending 完整控制周期·控制路径：Scheduler 进程内三策略真实内核在线 Select + 生产协调。

与 G1 pkg_greedy/control.py（FS 在线 plan_strict 先例）的差别：
1. 三策略分派（PILOT_POLICY）：FlowState-Greedy 走生产入口 GlobalOptimizer.plan_strict；
   Marconi-Evict / KVFlow-inspired 走真实内核（evaluation.p0_control_inclusive_harness.select_entry，
   Task 1 已 CPU 门禁验证的同一调用方式），选集（CPU 域）经 census 映射换算物理域后由生产构造链
   planner.finish(prepared, selected) 建计划——与正式 b5d4_apply 为冻结选集建计划同一生产路径；
2. 不读取任何冻结选集/oracle（policy_plans/t2g1_plans 均不进控制路径；对照全部由 runner 在
   T4 之后窗口外执行）；E/G 合同仅做活观察与声明输入的一致性自检；
3. 四 Pending 全量 targets：每个 E>0 的 Pending 各自验证唯一合法恢复源；
4. FA 重算区域按四请求并集计算（各 Pending 按计划 E 起算的重算区间都可能被触碰）；
5. 全部分段记录绝对单调纳秒边界（result['clock']），供 runner 复核 T0–T4 整数恒等式。
census 对账、物化时间线合同、生产 Bind/Preflight、Recurrent-only Reconciliation、FA 审计
与正式实验逐项相同。Epoch 形成方式：复用冻结 16 条预热请求两轮重放 + 实时观察准入。
"""
import os
from evaluation.exact_reconciliation_correctness_gate_b5d3.control import *
from flowstate.strict_control import validate_plan
from evaluation.p0_control_inclusive_harness import select_entry
from evaluation.exact_greedy_paired_gpu_fix_b5d4a.persistence import save_new

# 三策略合法域（与 Task 1 harness 同名）
POLICIES = ('FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired')


class DeferredJournal:
    """取证落盘出窗版 Journal（论文证据审计 §5.2 风险一修复）。

    窗口内（T0–T3）record() 只做同步序列化（保留各阶段快照语义——延迟到 T3 后再序列化
    会读到被后续阶段改写的 dict，丢失阶段证据），写盘+flush+fsync 全部延迟到
    flush_after_window()（T3 之后、响应返回前，计入 RPC 尾段间隙，单独计时入账）。
    耗时双向记账：in_window_serialize_ms（窗内序列化，计入段间隙）与
    after_window_flush_ms（出窗落盘）；报告不得声称"窗口内零取证序列化"，
    只能声称"窗口内零取证磁盘 I/O"。
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.stage = 'start'
        self.index = 0
        self._pending = []
        self.in_window_serialize_ms = 0.0
        self.after_window_serialize_ms = 0.0
        self.window_closed = False   # apply 设定 t3_ns 后置 True：此后序列化计入出窗
        self.flush_ms = None

    def mark_window_closed(self):
        self.window_closed = True

    def record(self, stage, result):
        self.stage = stage
        t = time.monotonic_ns()
        blob = json.dumps(result, ensure_ascii=False, indent=2)
        elapsed = (time.monotonic_ns() - t) / 1e6
        if self.window_closed:
            self.after_window_serialize_ms += elapsed
        else:
            self.in_window_serialize_ms += elapsed
        self._pending.append((self.index, stage, blob))
        self.index += 1

    def flush_after_window(self):
        """T3 之后统一写盘+fsync；耗时单独记账，绝不计入控制周期分段。"""
        t = time.monotonic_ns()
        for i, stage, blob in self._pending:
            with (self.directory / f'{i:03d}_{stage}.json').open('x') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
        self.flush_ms = (time.monotonic_ns() - t) / 1e6
        self._pending = []
        return self.flush_ms

    def fail(self, result, exc):
        record = dict(result, failed_stage=result.get('phase', self.stage), error=repr(exc),
                      traceback=traceback.format_exc())
        self.record('failure', record)
        return record


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


def load_declared_case(inputs):
    """载入 CPU 声明 case（两 Baseline 内核的冻结输入域）并做 GPU 决策域对齐断言。

    断言与 t2g1 adapter.load_case 逐条相同：GPU 四 Pending 必须在 CPU ready 域内且
    token/T 逐项全等。case 文件来自臂目录（prepare 已做 sha 身份锁定）。
    """
    case = json.loads((OUT / 'workloads_case_C.json').read_text())
    ready = {p['continuation_id']: p for p in case['calls'] if p['ready']}
    gpu_pendings = {p['continuation_id']: p for p in inputs['pendings']}
    require(set(gpu_pendings) <= set(ready), 'GPU Pending 不在 CPU ready 域')
    for pid, g in gpu_pendings.items():
        require(ready[pid]['token_ids'] == g['token_ids'] and
                ready[pid]['planning_target_T'] == g['planning_target_T'],
                'CPU ready 声明与 GPU 输入不一致：' + pid)
    return case


def apply(scheduler, request):
    result = {'ok': False, 'status': 'BLOCKED', 'operations': [], 'timing': [], 'observations': [],
              'phase': 'start', 'clock': {'stages': [], 't0_ns': None, 't3_ns': None}}
    journal = None

    def timed(stage, fn):
        s = time.monotonic_ns()
        v = fn()
        e = time.monotonic_ns()
        result['timing'].append({'stage': stage, 'ms': (e - s) / 1e6})
        result['clock']['stages'].append({'stage': stage, 'start_ns': s, 'end_ns': e})   # 绝对边界
        if result['clock']['t0_ns'] is None:
            result['clock']['t0_ns'] = s   # T0 = 第一段（final_heavy_observation）开始
        return v

    try:
        journal = DeferredJournal(OUT / os.environ['B5D4_RUN'] / 'control_evidence')
        result['phase'] = 'observe'
        initial = timed('final_heavy_observation', lambda: capture.observe_control(scheduler, {'final': True}))
        require(initial.get('ok'), '当前观察失败')
        scheduler._r2c0_owner = threading.get_ident()
        window = initial['window']
        prior.scope(scheduler)
        raw = initial['strict_epoch']
        result['raw_epoch'] = raw  # 桥接 b5d4_apply 成功路径要求（attempt2 教训：协调全过但缺此键）
        journal.record('observed_epoch', result)
        inputs = json.loads((OUT / 'prepared_inputs.json').read_text())
        mapping = json.loads((OUT / 'cpu_candidate_prefix_mapping.json').read_text())['mapping']

        # ── CPU 候选域 ↔ 物理候选域 对账（16=16 合同；不含任何选择）──
        result['phase'] = 'census'
        rows, unmapped = timed('cpu_physical_mapping', lambda: physical_cpu_census(raw, inputs, mapping))
        cpu_to_phys = {r['cpu_checkpoint_id']: r['physical_checkpoint_id'] for r in rows}
        missing_physical = sorted(set(mapping) - set(cpu_to_phys))
        result['census'] = {'mapped': rows, 'unmapped_extras': unmapped,
                            'cpu_domain_size': len(mapping), 'physical_domain_size': len(raw['candidates']),
                            'missing_physical': missing_physical,
                            'note': '未映射的额外物理状态一律视为未选中并在协调中真实驱逐，不免费保留任何祖先'}
        require(len(rows) == 16 and len(raw['candidates']) == 16 and not unmapped and not missing_physical,
                'census 16=16 对账失败：physical=%d mapped=%d extras=%d missing=%d' % (
                    len(raw['candidates']), len(rows), len(unmapped), len(missing_physical)))

        # ── 来源证据：本轮 16 条二次预热的真实观察记录（runner 写入 history_XX.json）──
        result['phase'] = 'origins'
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

        # ── Epoch 认证（from_frozen/prepare 全合同；K=3 为 C/K3 冻结预算）──
        result['phase'] = 'epoch_authentication'

        def authenticate():
            data = encode(raw)
            physical = encode(initial)
            evidence = planner.FrozenEvidence(data, hashlib.sha256(data).hexdigest(),
                                              physical_snapshot=physical,
                                              physical_sha256=hashlib.sha256(physical).hexdigest())
            epoch = planner.from_frozen(evidence, 3, (initial, origins, inputs))
            prepared = planner.prepare(epoch, evidence)
            return epoch, prepared
        epoch, prepared = timed('strict_epoch_authentication', authenticate)
        result.update(epoch=asdict(epoch), origins=origins)
        journal.record('strict_epoch', result)

        # ── 本轮被测对象：三策略真实内核在线 Select（不读环境变量选择、不读任何冻结选集）──
        result['phase'] = 'policy_select'
        policy = os.environ['PILOT_POLICY']
        require(policy in POLICIES, '未知策略：' + policy)

        def select_online():
            """三策略统一在线选择：FS=生产入口 plan_strict；MC/KV=真实内核+census 映射+生产 finish。"""
            if policy == 'FlowState-Greedy':
                plan, meta = select_entry.select_flowstate({'epoch': epoch})
                return plan, meta, sorted(phys_to_cpu[c]['cpu_checkpoint_id'] for c in plan.selected_ids)
            case = load_declared_case(inputs)
            if policy == 'Marconi-Evict':
                ev, meta = select_entry.select_marconi({'case': case})
                sel_cpu = tuple(sorted(ev.selected_checkpoint_ids))
                return_dict = {'eviction_order_cpu': list(ev.evicted_checkpoint_ids), 'rounds': len(ev.rounds)}
            else:
                sel_cpu_kv, meta = select_entry.select_kvflow({'case': case})
                sel_cpu = tuple(sorted(sel_cpu_kv))
                return_dict = {}
            absent = [c for c in sel_cpu if c not in cpu_to_phys]
            require(not absent, '%s 选集未在物理域物化：%r' % (policy, absent))
            sel_phys = tuple(cpu_to_phys[c] for c in sel_cpu)
            plan = planner.finish(prepared, sel_phys)   # 与正式 b5d4_apply 冻结选集同一生产构造链
            meta = dict(meta, **return_dict)
            return plan, meta, list(sel_cpu)
        plan, meta, selected_cpu_now = timed('policy_online_select', select_online)
        validate_plan(epoch, plan)
        selected = plan.selected_ids
        decision = {'policy': policy, 'solver': meta.get('decision', {}).get('solver', policy + ' 真实内核'),
                    'kernel_entry': meta['kernel_entry'],
                    'gate': meta.get('decision', {}).get('gate'),
                    'selection_call_ms': meta['selection_call_ms']}
        result.update(selection_source=decision['kernel_entry'] + '(Scheduler进程内在线)',
                      budget_K=3, policy=policy, plan=asdict(plan),
                      strict_decision=decision, policy_selected_cpu_ids=selected_cpu_now,
                      selected_physical_ids=list(selected))
        journal.record('plan', result)

        # ── E/G 活观察一致性自检（不读冻结预期；活观察 T/FA 前沿与声明输入一致即可）──
        result['phase'] = 'epoch_consistency'
        expect_live = {p.pending_id: p for p in plan.pending}
        for p in raw['pendings']:
            declared = next(x for x in inputs['pendings'] if x['continuation_id'] == p['continuation_id'])
            require(p['planning_target_T'] == declared['planning_target_T'] and
                    p['attention_frontier'] == declared['attention_frontier'],
                    '活观察T/FA前沿与CPU声明不一致：' + p['continuation_id'])
            require(expect_live[p['continuation_id']].T == p['planning_target_T'],
                    'Plan前沿与活观察T不一致：' + p['continuation_id'])

        # ── 四 Pending targets：每个 E>0 的 Pending 各自验证唯一合法恢复源 ──
        result['phase'] = 'source_selection'
        targets = {}
        for tp in raw['pendings']:
            pid = tp['continuation_id']
            expected = next(x for x in plan.pending if x.pending_id == pid)
            if expected.E > 0:
                cands = [c for c in raw['candidates'] if c['checkpoint_id'] in selected and
                         c['token_ids'] == tp['token_ids'][:c['token_position_L']] and
                         c['token_position_L'] == expected.E]
                require(len(cands) == 1, '目标没有唯一合法恢复源：%s' % pid)
                targets[pid] = {'pending': tp, 'source_checkpoint': cands[0]['checkpoint_id'],
                                'expected': asdict(expected)}
            else:
                targets[pid] = {'pending': tp, 'source_checkpoint': None, 'expected': asdict(expected),
                                'note': '该策略下目标无兼容保留状态，E=0，无恢复源；FA 命中照常认证'}
        result['targets'] = targets
        journal.record('source_expectation', result)

        result['phase'] = 'apply_preflight'

        def check():
            before, audit = prior.observe(scheduler, epoch, window)
            require(asdict(before) == initial['before'] and audit['semantic'] == initial['audit_before']['semantic'],
                    'Apply前状态变化')
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
                      apply_validation={'status': 'PASS', 'binding': asdict(binding), 'window': window,
                                        'same_scheduler_call': True})

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
            require(a['state']['tree']['structure_sha256'] == audit['state']['tree']['structure_sha256'],
                    '协调过程FA树结构变化')
            result['observations'].append(asdict(view))
            return view

        def invalidate(ledger):
            scheduler._r2c0_failed = True

        result['phase'] = 'reconciliation'
        journal.record('before_reconciliation', result)
        t = time.monotonic_ns()
        runtime = StrictRuntimeBridge(RecordedAdapter(scheduler.tree_cache, result['operations']), held, observe,
                                      invalidate)
        ledger = StateController.reconcile_strict(epoch, plan, runtime)
        result['timing'].append({'stage': 'strict_reconciliation_with_postchecks', 'ms': (time.monotonic_ns() - t) / 1e6})
        after, last = prior.observe(scheduler, epoch, window)
        source_after = selected_payload(scheduler.tree_cache, after, plan, epoch)
        require(source_after == result['selected_payload_before'], '所选组件内容变化')
        require([x['checkpoint_id'] for x in result['operations']] == list(plan.logical_eviction_ids)
                and all(x['returned'] for x in result['operations']), '驱逐未完整落实')

        t = time.monotonic_ns()
        require(fa_map(scheduler.tree_cache) == old_fa and fa_payload(scheduler.tree_cache, indices) == full_hash,
                '协调改变FA索引或内容')
        # 四个正式请求可能各自重算其计划 E 之后区间的 FA；只保护其余原始 FA 内容（并集口径）
        recomputed = set()
        for pid, trow in targets.items():
            d2 = hashlib.sha256()
            e = trow['expected']['E']
            for pos, token in enumerate(trow['pending']['token_ids']):
                d2.update(int(token).to_bytes(8, 'little'))
                if pos >= e and d2.hexdigest() in old_fa:
                    recomputed.add(old_fa[d2.hexdigest()])
        protected = sorted(indices - recomputed)
        protected_hash = fa_payload(scheduler.tree_cache, protected)
        result['timing'].append({'stage': 'fa_content_postcheck_and_resume_baseline',
                                 'ms': (time.monotonic_ns() - t) / 1e6})
        # T3 = 协调+必要后验完成（apply() 内部最后一段结束）
        result['clock']['stages'].append({'stage': 'fa_content_postcheck_and_resume_baseline',
                                          'start_ns': t, 'end_ns': time.monotonic_ns()})
        result['clock']['t3_ns'] = result['clock']['stages'][-1]['end_ns']
        journal.mark_window_closed()   # 此后（含 complete 记录）序列化计入出窗
        result.update(ok=True, status='READY', phase='complete', reconciliation=ledger, after=asdict(after),
                      audit_after=last, selected_payload_after=source_after, fa_map_before=old_fa,
                      fa_protected_indices=protected, fa_protected_payload=protected_hash,
                      fa_expected_recomputed_indices=sorted(recomputed))
        prior.scope(scheduler)
        journal.record('complete', result)
        journal.flush_after_window()   # 全部取证写盘+fsync 在 T3 之后（RPC 尾段），单独计时
        result['journal_io'] = {'records': journal.index,
                                'in_window_serialize_ms': journal.in_window_serialize_ms,
                                'after_window_serialize_ms': journal.after_window_serialize_ms,
                                'after_window_flush_ms': journal.flush_ms,
                                'note': 'T0–T3 窗口内零取证磁盘 I/O（仅同步序列化，计入段间隙）；'
                                        '写盘+fsync 全部在 T3 之后'}
        return result, epoch, plan
    except BaseException as exc:
        if isinstance(exc, StrictReconciliationError):
            result['reconciliation'] = exc.ledger
        scheduler._r2c0_failed = True
        result.update(error=repr(exc), traceback=traceback.format_exc(), status='BLOCKED')
        if journal is not None:
            journal.fail(result, exc)
            journal.flush_after_window()   # 失败路径同样先落盘证据再返回（窗口已失效）
            result['journal_io'] = {'records': journal.index,
                                    'in_window_serialize_ms': journal.in_window_serialize_ms,
                                    'after_window_serialize_ms': journal.after_window_serialize_ms,
                                    'after_window_flush_ms': journal.flush_ms,
                                    'note': '失败路径：全部取证已落盘后返回'}
        return result, None, None
    finally:
        scheduler._r2c0_owner = None
