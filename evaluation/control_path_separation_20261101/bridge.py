"""四 Pending 完整控制周期桥接（测量模式）：请求窗口零 A 类审计，全量观察移至请求后。

基于 c_formal_perf_gpu/bridge.py（测量模式原版）的改造，差别只在四 Pending 串行：
- 单值 TARGET 改为集合 TARGETS（四个 continuation_id 一次写入；同一时刻仅一个活跃批次，
  独占单请求断言不变）；
- Admission 断言按 rid 各一次（四请求依次 Native Resume，各自只允许一次 Admission），
  Token 比对按 BASELINE['targets'][rid]['pending']['token_ids'] 分派；
- match/copy 观测与 execution/compute 行均带 rid，请求后按 Pending 归因验收；
- apply 后基线（A 类全量哈希）在 dispatch 内单独计时（t3 之后的 RPC 尾段，不冒充控制周期）。
A 类审计不删除：apply 段合同逐字节不变；请求后 b5d4_state 做全量状态/内容观察；
输出正确性由 Cold 逐 Token 一致与请求级合同共同把守。
"""
import os
import threading
import time
from types import SimpleNamespace

from evaluation.runtime_control_closure_r2c1a import bridge as reconcile
from evaluation.runtime_control_closure_r2c0 import bridge as view_bridge
from evaluation.strict_compatibility_b1c import runner as native
from evaluation.lightweight_runtime_e1 import control as original_control
from . import control
from . import observation

require = reconcile.require
# 本桥接只服务测量模式：环境变量必须显式设置，防止误以 strict 语义跑正式矩阵
MEASURE = os.environ.get('FS_MEASURE_MODE') == '1'
require(MEASURE, '本轮桥接要求 FS_MEASURE_MODE=1（测量模式：请求窗口零A类审计）')
CURRENT = None
TARGETS = frozenset()   # 四 Pending rid（协调完成后由 dispatch 写入）
ADMITTED = set()        # 已 Admission 的 rid（每 Pending 恰一次）
TRACE = {'admission': [], 'matches': [], 'copies': [], 'execution': [], 'housekeeping': [],
         'compute': [], 'shape_warm_matches': []}
BASELINE = None
AUDIT = False
ARM = os.environ.get('B5D4_ARM', 'strict')
RUN = os.environ.get('B5D4_RUN', 'run')
require(ARM == 'strict', '本轮只使用 strict 臂语义（apply 合同不变）')


def load():
    require(CURRENT is not None, '缺少当前实时Epoch')
    return CURRENT


def save():
    """测量模式下不落盘：请求窗口内零磁盘 I/O；TRACE 经请求后 b5d4_state 返回由 runner 持久化。"""
    return


def _cache_state(cache):
    """全量状态观察（A 类，仅用于 apply 段与请求后；与 e1 原版逐字节同逻辑）。"""
    import targeted_probe as probe
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
    from evaluation.barrier_fa_frontier_control import semantic_cache_snapshot
    nodes = []
    for nid, node in probe._tree_items(cache.tree_core):
        parts = []
        cur = node
        while cur.parent is not None:
            require(not cur.key.is_bigram, '不支持bigram')
            parts.append(list(cur.key))
            cur = cur.parent
        tokens = tuple(t for part in reversed(parts) for t in part)
        cd = node.component_data[CT.MAMBA]
        full = node.component_data[CT.FULL]
        nodes.append({'node_id': nid, 'token_ids': list(tokens), 'slots': native.tensor_ids(cd.value),
                      'full_present': full.value is not None,
                      'references': [int(cd.lock_ref), int(cd.host_lock_ref), int(cd.session_ref)]})
    tree = probe._global_maps(cache)
    accounting = probe._accounting_snapshot(cache)
    semantic = semantic_cache_snapshot(cache, component_type=CT, tree_snapshot=tree, accounting_snapshot=accounting)
    result = {'nodes': nodes, 'semantic': semantic, 'ns': time.monotonic_ns()}
    if ARM == 'strict' and BASELINE is not None:
        _, epoch, plan = load()
        view = SimpleNamespace(nodes=[SimpleNamespace(token_ids=tuple(n['token_ids']), slots=tuple(n['slots']))
                                      for n in nodes])
        result['selected_payload'] = reconcile.selected_payload(cache, view, plan, epoch)
        result['selected_components'] = {}
        tensors = cache.req_to_token_pool.mamba_pool.mamba_cache
        for c in epoch.candidates:
            cid = c.candidate.checkpoint_id
            if cid not in plan.selected_ids:
                continue
            n = next(n for n in view.nodes if n.token_ids == c.token_ids)
            result['selected_components'][cid] = [view_bridge.slice_identity(t, s)[0]
                                                  for s in n.slots for t in list(tensors.conv) + [tensors.temporal]]
    if ARM == 'strict' and BASELINE is not None:
        from evaluation.lightweight_runtime_e1.control import fa_map, fa_payload
        result['fa_map'] = fa_map(cache)
        result['fa_protected_payload'] = fa_payload(cache, BASELINE['fa_protected_indices'])
    return result


def cache_state(cache):
    """全量状态观察入口（含计时记录，供 apply 段与请求后审计归因；请求窗口内不调用）。"""
    start = time.monotonic_ns()
    result = _cache_state(cache)
    TRACE.setdefault('audit', []).append({'kind': 'cache_state_full_hash', 'ns': time.monotonic_ns() - start})
    return result


def install_observation(cache):
    """原生记录（MATCHES/COPIES/COPY_OWNERS，B 类轻量）+ 测量模式 match/copy 钩子。"""
    native.install_observation(cache)
    original_match = cache.match_prefix
    pool = cache.req_to_token_pool.mamba_pool
    original_copy = pool.copy_from

    def match(params):
        req = params.req
        rid = str(getattr(req, 'rid', '')) if req is not None else None
        target = rid in TARGETS
        record = None
        if target:
            # 只记时间戳与 COW 请求标志；匹配前全量哈希（A 类）已移至请求后
            record = {'rid': rid, 'ns_before': time.monotonic_ns(), 'cow_requested': bool(params.cow_mamba)}
            TRACE['matches'].append(record)
        call_start = time.monotonic_ns()
        result = original_match(params)
        call_end = time.monotonic_ns()
        if target:
            record.update(ns_after=time.monotonic_ns(), native_call_ns=call_end - call_start,
                          native=native.MATCHES[rid][-1])
        elif rid is not None and rid.startswith('SHAPE_WARM'):
            # 形状预热请求：记录原生匹配行用于验证“计划保留集下准入恰为计划 E”，零哈希零落盘
            TRACE['shape_warm_matches'].append(native.MATCHES[rid][-1])
        return result

    def copy(src, dst):
        ss = native.tensor_ids(src)
        ds = native.tensor_ids(dst)
        target = any(native.COPY_OWNERS.get((s, d)) in TARGETS for s, d in zip(ss, ds))
        row = None
        if target:
            row = {'source_slots': ss, 'destination_slots': ds, 'ns_before': time.monotonic_ns(),
                   'content_audit': '移至请求后全量观察+Cold逐Token一致'}
            TRACE['copies'].append(row)
        if target:
            # CUDA 事件保留：B 类 COW 的设备时间（事件开销微秒级，计入测量口径 A）
            import torch
            torch.cuda.synchronize()
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
        call_start = time.monotonic_ns()
        value = original_copy(src, dst)
        call_end = time.monotonic_ns()
        if target:
            event_end.record()
            event_end.synchronize()
            row.update(ns_after=time.monotonic_ns(), native_host_ns=call_end - call_start,
                       cuda_elapsed_ms=event_start.elapsed_time(event_end), returned=True)
        return value

    cache.match_prefix = match
    pool.copy_from = copy


def scheduler_process(*args, **kwargs):
    """专用进程：apply 一次（三策略在线 Select + 完整合同）+ 四请求测量钩子 + 请求后全量观察；失败保持隔离。"""
    import wp3b_end_to_end_transport as transport
    import targeted_probe as probe
    # 控制路径替换：b5d4_apply 分发到本轮 control.py（三策略在线 Select 版）
    original_control.apply = control.apply
    cls = transport._scheduler_module.Scheduler
    old_idle = cls.on_idle
    old_input = cls.process_input_requests
    old_run = cls.run_batch

    def idle(self):
        if getattr(self, '_r2c0_failed', False):
            print('[P04CYCLE] 失败隔离，等待本轮清理', flush=True)
            threading.Event().wait()
        return old_idle(self)

    def inputs(self, requests):
        targets = [r for r in requests if str(getattr(r, 'rid', '')) in TARGETS]
        if targets:
            try:
                require(len(targets) == 1, '同一批次出现多个目标Pending')
                rid = str(getattr(targets[0], 'rid', ''))
                require(rid not in ADMITTED, '目标重复Admission：%s' % rid)
                probe._validate_runtime_scope(self, self.tree_cache)
                t0 = time.monotonic_ns()
                received = list(targets[0].input_ids)
                # 轻量正确性检查：协调已完成 + 冻结 Token 逐项一致（O(len)）；
                # 全量状态/FA 内容比对（A 类）移至请求后 b5d4_state，由 runner 断言
                require(BASELINE is not None, '未完成协调')
                from .protocol import admission_gate
                admission_gate(BASELINE, None, received, BASELINE['targets'][rid]['pending']['token_ids'])
                ADMITTED.add(rid)
                TRACE['admission'].append({'rid': rid, 'ns_before': t0, 'ns_after': time.monotonic_ns(),
                                           'input_token_count': len(received), 'measure_mode': True})
            except BaseException as exc:
                TRACE['admission_error'] = repr(exc)
                self._r2c0_failed = True
                raise
        return old_input(self, requests)

    observation.TARGETS = TARGETS
    observation.COMPUTE = TRACE['compute']
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    EagerRunner.execute = observation.wrap_eager(EagerRunner.execute, save)
    cls.on_idle = idle
    cls.process_input_requests = inputs
    cls.run_batch = observation.wrap_run_batch(old_run, TRACE['execution'], save)

    def dispatch(scheduler, request):
        global BASELINE, CURRENT, TARGETS, AUDIT
        require(not getattr(scheduler, '_r2c0_failed', False), '生命周期已失效')
        action = request.get('action')
        if action == 'b5d4_apply':
            require(ARM == 'strict' and BASELINE is None, '禁止重复协调')
            result, epoch, plan = original_control.apply(scheduler, request)
            if result.get('ok'):
                CURRENT = (result['raw_epoch'], epoch, plan)
                TARGETS = frozenset(result['targets'])          # 四 Pending 全域
                observation.TARGETS = TARGETS
                result['observer_api_preflight'] = observation.real_api_preflight()
                BASELINE = result
                # apply 后全量基线（A 类，t3 之后的 RPC 尾段；单独计时，不冒充控制周期内分段）
                t = time.monotonic_ns()
                result['admission_baseline'] = cache_state(scheduler.tree_cache)
                result['post_apply_baseline_ns'] = {'start_ns': t, 'end_ns': time.monotonic_ns()}
            return result
        if action == 'b5d4_state':
            probe._validate_runtime_scope(scheduler, scheduler.tree_cache)
            AUDIT = True
            try:
                return {'ok': True, 'state': cache_state(scheduler.tree_cache), 'trace': TRACE,
                        'observer_api_preflight': observation.real_api_preflight()}
            finally:
                AUDIT = False
        if action == 'b5c3_observe':
            require(BASELINE is None, '协调后不重新构造分配快照')
            from evaluation.multishare_runtime_gate_b5c3.bridge import observe_control
            return observe_control(scheduler, request)
        raise ValueError('未授权动作：' + str(action))

    if os.environ.get('FS_SEPARATION_MODE') == 'serving':
        def wire_dispatch(scheduler, request):
            try:
                value = dispatch(scheduler, request)
                if request.get('action') == 'b5d4_apply' and value.get('ok'):
                    from .codec import pack
                    return pack(value, verify=os.environ.get('FS_SEPARATION_VERIFY') == '1')
                return value
            except BaseException:
                scheduler._r2c0_failed = True
                raise
        transport._run_checkpoint_control = wire_dispatch
    else:
        transport._run_checkpoint_control = dispatch
    transport._install_match_instrumentation = install_observation
    return transport._wrapped_run_scheduler_process(*args, **kwargs)
