"""仅供 RQ6-D 控制路径瓶颈诊断使用的调度器传输层。"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import os
from time import perf_counter_ns
from typing import Callable

from sglang.srt.entrypoints.engine import Engine as _SGLangEngine
from sglang.srt.managers import scheduler as _scheduler_module

import targeted_probe as _probe
import wp3b_end_to_end_transport as _base_transport
import sequential_eviction_trace_transport as _trace_transport

from flowstate.adapters.sglang import RuntimeCheckpointHandle, SGLangAdapter


_NOOP_ACTION = "flowstate_rq6d_noop"
_TIMED_CENSUS_ACTION = "flowstate_rq6d_timed_census"
_TIMED_INSPECT_ACTION = "flowstate_rq6d_timed_inspect"
_TIMED_EVICT_ACTION = "flowstate_rq6d_timed_evict"
_ORIGINAL_RUN_SCHEDULER_PROCESS = _scheduler_module.run_scheduler_process
_ORIGINAL_CENSUS = _probe._run_census


class _CallProfiler:
    """在不改变被调函数返回值的前提下累计嵌套阶段耗时。"""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.inclusive_ns: dict[str, int] = defaultdict(int)
        self._section = "外层"

    @contextmanager
    def section(self, name: str):
        """临时标记叶子操作所属的上层阶段。"""
        previous = self._section
        self._section = name
        try:
            yield
        finally:
            self._section = previous

    def wrap(self, name: str, function: Callable):
        """构造保留调用签名弹性的纳秒计时包装器。"""
        def timed(*args, **kwargs):
            started = perf_counter_ns()
            try:
                with self.section(name):
                    return function(*args, **kwargs)
            finally:
                self.counts[name] += 1
                self.inclusive_ns[name] += perf_counter_ns() - started

        return timed

    def wrap_leaf(self, name: str, function: Callable):
        """按当前上层阶段累计摘要或槽位读取时间。"""
        def timed(*args, **kwargs):
            section = self._section
            started = perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                key = f"{section}:{name}"
                self.counts[key] += 1
                self.inclusive_ns[key] += perf_counter_ns() - started

        return timed

    def row(self) -> dict[str, object]:
        """返回可序列化的阶段计数与耗时。"""
        return {
            "counts": dict(sorted(self.counts.items())),
            "inclusive_ns": dict(sorted(self.inclusive_ns.items())),
        }


@contextmanager
def _profile_probe_calls(profiler: _CallProfiler):
    """临时包裹正式快照函数，并在退出时完整恢复。"""
    names = (
        "_validate_runtime_scope",
        "_path_snapshot",
        "_accounting_snapshot",
        "_global_maps",
        "_find_exact_node",
    )
    originals = {name: getattr(_probe, name) for name in names}
    leaf_names = ("_tensor_sha256", "_tensor_ids")
    originals.update({name: getattr(_probe, name) for name in leaf_names})
    try:
        for name in names:
            setattr(_probe, name, profiler.wrap(name, originals[name]))
        for name in leaf_names:
            setattr(_probe, name, profiler.wrap_leaf(name, originals[name]))
        yield
    finally:
        for name, function in originals.items():
            setattr(_probe, name, function)


def _timing_envelope(
    request: dict,
    started_ns: int,
    *,
    profiler: _CallProfiler | None = None,
) -> dict[str, object]:
    """构造跨进程单调时钟可核对的服务端计时字段。"""
    ended_ns = perf_counter_ns()
    return {
        "client_sent_ns": int(request.get("client_sent_ns") or 0),
        "server_started_ns": started_ns,
        "server_ended_ns": ended_ns,
        "server_total_ns": ended_ns - started_ns,
        "profile": None if profiler is None else profiler.row(),
    }


def _timed_noop(_scheduler: object, request: dict) -> dict[str, object]:
    """只返回响应，不读取或修改运行时状态。"""
    started = perf_counter_ns()
    result = {"ok": True, "nonce": request.get("nonce")}
    result["diagnostic_timing"] = _timing_envelope(request, started)
    return result


def _timed_census(scheduler: object, request: dict) -> dict[str, object]:
    """按正式 census 语义执行一次分阶段只读计时。"""
    started = perf_counter_ns()
    profiler = _CallProfiler()
    forwarded = dict(request)
    forwarded["op"] = "census"
    with _profile_probe_calls(profiler):
        result = _probe._run_census(scheduler, forwarded)
    result["diagnostic_timing"] = _timing_envelope(
        request, started, profiler=profiler
    )
    return result


def _timed_inspect(scheduler: object, request: dict) -> dict[str, object]:
    """按正式 inspect 语义执行一次分阶段只读计时。"""
    started = perf_counter_ns()
    profiler = _CallProfiler()
    forwarded = dict(request)
    forwarded["action"] = "inspect"
    with _profile_probe_calls(profiler):
        result = _base_transport._run_checkpoint_control(scheduler, forwarded)
    result["diagnostic_timing"] = _timing_envelope(
        request, started, profiler=profiler
    )
    return result


def _timed_evict(scheduler: object, request: dict) -> dict[str, object]:
    """计时一次真实适配器驱逐，并保留前后正确性快照。"""
    started = perf_counter_ns()
    profiler = _CallProfiler()
    cache = scheduler.tree_cache
    with _profile_probe_calls(profiler):
        scope_started = perf_counter_ns()
        scope = _probe._validate_runtime_scope(scheduler, cache)
        scope_ns = perf_counter_ns() - scope_started

        token_ids = tuple(int(value) for value in request.get("token_ids") or ())
        extra_key = request.get("extra_key")
        before_started = perf_counter_ns()
        before = _probe._snapshot(cache, token_ids, extra_key)
        before_snapshot_ns = perf_counter_ns() - before_started
        handle = RuntimeCheckpointHandle(
            checkpoint_id=str(request["checkpoint_id"]),
            token_ids=token_ids,
            extra_key=extra_key,
            expected_node_id=int(request["expected_node_id"]),
            expected_prefix_digest=str(request["expected_prefix_sha256"]),
        )
        adapter = SGLangAdapter(cache)
        primitive_ns = 0
        original_primitive = adapter._evict_mamba_component_only

        def timed_primitive(node: object) -> None:
            nonlocal primitive_ns
            primitive_started = perf_counter_ns()
            try:
                original_primitive(node)
            finally:
                primitive_ns += perf_counter_ns() - primitive_started

        adapter._evict_mamba_component_only = timed_primitive
        adapter_started = perf_counter_ns()
        try:
            adapter.evict_mamba_only(handle)
        finally:
            adapter._evict_mamba_component_only = original_primitive
        adapter_total_ns = perf_counter_ns() - adapter_started

        after_started = perf_counter_ns()
        after = _probe._snapshot(cache, token_ids, extra_key)
        after_snapshot_ns = perf_counter_ns() - after_started

    changed = _probe._changed_mamba_nodes(
        before["tree"]["mamba_rows"], after["tree"]["mamba_rows"]
    )
    target_node_id = int(before["path"]["node_id"])
    proof = {
        "same_node": target_node_id == int(after["path"]["node_id"]),
        "fa_unchanged": (
            before["tree"]["full_tree_sha256"]
            == after["tree"]["full_tree_sha256"]
            and before["path"]["path_full_sha256"]
            == after["path"]["path_full_sha256"]
            and before["accounting"]["full_allocator"]
            == after["accounting"]["full_allocator"]
        ),
        "tree_unchanged": (
            before["tree"]["structure_sha256"]
            == after["tree"]["structure_sha256"]
        ),
        "only_target_mamba_changed": changed == [target_node_id],
        "target_recurrent_removed": (
            before["path"]["target_mamba_present"]
            and not after["path"]["target_mamba_present"]
        ),
        "cascade_called": False,
    }
    result = {
        "ok": True,
        "nonce": request.get("nonce"),
        "scope": scope,
        "before": before,
        "after": after,
        "proof": proof,
        "operation_timing_ns": {
            "scope_validation": scope_ns,
            "before_snapshot": before_snapshot_ns,
            "adapter_total": adapter_total_ns,
            "recurrent_remove_free": primitive_ns,
            "adapter_validation": adapter_total_ns - primitive_ns,
            "after_snapshot": after_snapshot_ns,
        },
        "formal_primitive": (
            "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
        ),
    }
    result["diagnostic_timing"] = _timing_envelope(
        request, started, profiler=profiler
    )
    return result


def _run_checkpoint_control(scheduler: object, request: dict) -> dict:
    """处理 RQ6-D 诊断动作，其余动作转交冻结端到端传输。"""
    action = request.get("action")
    if action == _NOOP_ACTION:
        return _timed_noop(scheduler, request)
    if action == _TIMED_CENSUS_ACTION:
        return _timed_census(scheduler, request)
    if action == _TIMED_INSPECT_ACTION:
        return _timed_inspect(scheduler, request)
    if action == _TIMED_EVICT_ACTION:
        return _timed_evict(scheduler, request)
    if not request.get("rq6d_measure"):
        return _trace_transport._run_checkpoint_control(scheduler, request)
    started = perf_counter_ns()
    profiler = _CallProfiler()
    adapter_original = SGLangAdapter._evict_mamba_component_only
    with _profile_probe_calls(profiler):
        SGLangAdapter._evict_mamba_component_only = profiler.wrap(
            "adapter_primitive", adapter_original
        )
        cache = scheduler.tree_cache
        targets = (
            (cache.tree_core, "_evict_component_and_detach_lru"),
            (cache.tree_core, "_update_evictable_leaf_sets"),
            (cache, "_free_values"),
            (cache, "sanity_check"),
        )
        originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
        try:
            for owner, name, function in originals:
                setattr(owner, name, profiler.wrap(name, function))
            result = _trace_transport._run_checkpoint_control(scheduler, request)
        finally:
            SGLangAdapter._evict_mamba_component_only = adapter_original
            for owner, name, function in originals:
                setattr(owner, name, function)
    result["diagnostic_timing"] = _timing_envelope(request, started, profiler=profiler)
    return result


def _profiled_census(scheduler: object, request: dict) -> dict:
    """只在显式诊断阶段记录正式 census 的原样执行时间。"""
    if not request.get("rq6d_measure"):
        return _ORIGINAL_CENSUS(scheduler, request)
    started = perf_counter_ns()
    profiler = _CallProfiler()
    with _profile_probe_calls(profiler):
        result = _ORIGINAL_CENSUS(scheduler, request)
    result["diagnostic_timing"] = _timing_envelope(request, started, profiler=profiler)
    return result


def _wrapped_run_scheduler_process(*args, **kwargs):
    """安装独立诊断动作，并保持正式 scheduler 空闲安全点。"""
    from evaluation.rq6_wait_diagnosis import install_wait_timing

    install_wait_timing(_probe, object)
    scheduler_class = _scheduler_module.Scheduler
    original_init = scheduler_class.__init__
    if not getattr(original_init, "_flowstate_rq6d_patched", False):

        def patched_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            _base_transport._install_match_instrumentation(self.tree_cache)
            _probe._run_checkpoint_control = _run_checkpoint_control
            _probe._run_census = _profiled_census
            port = _probe.install_control_server(self, requested_control_port())
            print(f"[RQ6D-TRANSPORT] 控制端口已就绪：{port}", flush=True)

        patched_init._flowstate_rq6d_patched = True
        scheduler_class.__init__ = patched_init

    original_on_idle = scheduler_class.on_idle
    if not getattr(original_on_idle, "_flowstate_rq6d_patched", False):

        def patched_on_idle(self):
            state = getattr(self, "_wp3d_probe_state", None)
            if state is not None and self.is_fully_idle():
                state.drain_one(self)
            return original_on_idle(self)

        patched_on_idle._flowstate_rq6d_patched = True
        scheduler_class.on_idle = patched_on_idle

    return _ORIGINAL_RUN_SCHEDULER_PROCESS(*args, **kwargs)


def requested_control_port() -> int:
    """读取 RQ6-D 独立诊断端口。"""
    return int(os.environ.get("FLOWSTATE_RQ6D_PORT", "49949"))


class RQ6BottleneckGateEngine(_SGLangEngine):
    """安装 RQ6-D 诊断传输层的冻结引擎包装器。"""

    run_scheduler_process_func = staticmethod(_wrapped_run_scheduler_process)
