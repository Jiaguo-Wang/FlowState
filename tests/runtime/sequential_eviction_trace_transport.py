"""仅供连续循环状态驱逐重驻留诊断使用的调度器传输层。"""

from __future__ import annotations

import os

from sglang.srt.entrypoints.engine import Engine as _SGLangEngine
from sglang.srt.managers import scheduler as _scheduler_module

import targeted_probe as _probe
import wp3b_end_to_end_transport as _base_transport

from evaluation.openhands_sequential_eviction_rematerialization_audit import (
    SequentialTraceRecorder,
    TraceSnapshot,
    allocator_state_from_accounting,
    checkpoint_state_from_path,
    instrument_evict_mamba_only,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle, SGLangAdapter


_TRACE_EVICTION_ACTION = "flowstate_trace_evict_mamba_only"
_TRACE_SNAPSHOT_ACTION = "flowstate_trace_snapshot"
_ORIGINAL_RUN_SCHEDULER_PROCESS = (
    _scheduler_module.run_scheduler_process
)


def _handles(request: dict) -> dict[str, RuntimeCheckpointHandle]:
    """解析并严格验证全量追踪句柄。"""
    result = {}
    for row in request.get("tracked_handles") or ():
        handle = RuntimeCheckpointHandle(
            checkpoint_id=str(row["checkpoint_id"]),
            token_ids=tuple(int(value) for value in row["token_ids"]),
            extra_key=row.get("extra_key"),
            expected_node_id=(
                None
                if row.get("expected_node_id") is None
                else int(row["expected_node_id"])
            ),
            expected_prefix_digest=row.get("expected_prefix_sha256"),
        )
        if handle.checkpoint_id in result:
            raise RuntimeError("追踪请求含重复检查点句柄")
        result[handle.checkpoint_id] = handle
    if not result:
        raise RuntimeError("追踪请求缺少检查点句柄")
    return result


def _snapshot_provider(cache: object, handles: dict[str, RuntimeCheckpointHandle]):
    """构造只调用现有精确路径与分配器快照的提供器。"""
    def provide() -> TraceSnapshot:
        paths = {
            checkpoint_id: _probe._path_snapshot(
                cache,
                handle.token_ids,
                handle.extra_key,
            )
            for checkpoint_id, handle in handles.items()
        }
        accounting = _probe._accounting_snapshot(cache)
        return TraceSnapshot(
            checkpoints={
                checkpoint_id: checkpoint_state_from_path(
                    checkpoint_id,
                    path,
                )
                for checkpoint_id, path in paths.items()
            },
            allocator=allocator_state_from_accounting(accounting),
        )

    return provide


def _target_handle(request: dict) -> RuntimeCheckpointHandle:
    """解析本次真实驱逐目标。"""
    return RuntimeCheckpointHandle(
        checkpoint_id=str(request["checkpoint_id"]),
        token_ids=tuple(int(value) for value in request["token_ids"]),
        extra_key=request.get("extra_key"),
        expected_node_id=(
            None
            if request.get("expected_node_id") is None
            else int(request["expected_node_id"])
        ),
        expected_prefix_digest=request.get("expected_prefix_sha256"),
    )


def _trace_snapshot(scheduler: object, request: dict) -> dict[str, object]:
    """在独立 scheduler safe point 记录目标的 S4。"""
    cache = scheduler.tree_cache
    scope = _probe._validate_runtime_scope(scheduler, cache)
    handles = _handles(request)
    recorder = SequentialTraceRecorder(
        tuple(sorted(handles)),
        _snapshot_provider(cache, handles),
    )
    target = str(request["target_checkpoint_id"])
    next_target = request.get("next_target_checkpoint_id")
    operation = (
        "结束连续驱逐序列前"
        if next_target is None
        else f"进入 {next_target} 驱逐前"
    )
    recorder.record(
        target_checkpoint_id=target,
        boundary="S4",
        operation=operation,
    )
    return {
        "ok": True,
        "nonce": request.get("nonce"),
        "scope": scope,
        "trace_rows": list(recorder.rows),
    }


def _trace_eviction(scheduler: object, request: dict) -> dict[str, object]:
    """调用原适配器一次，并在其真实内部边界采集 S0 至 S3。"""
    cache = scheduler.tree_cache
    scope = _probe._validate_runtime_scope(scheduler, cache)
    handles = _handles(request)
    handle = _target_handle(request)
    if handle.checkpoint_id not in handles:
        raise RuntimeError("真实驱逐目标不在追踪集合中")
    provider = _snapshot_provider(cache, handles)
    recorder = SequentialTraceRecorder(tuple(sorted(handles)), provider)
    before = _probe._snapshot(cache, handle.token_ids, handle.extra_key)
    adapter = SGLangAdapter(cache)
    instrument_evict_mamba_only(adapter, handle, recorder)
    after = _probe._snapshot(cache, handle.token_ids, handle.extra_key)
    changed_mamba_nodes = _probe._changed_mamba_nodes(
        before["tree"]["mamba_rows"],
        after["tree"]["mamba_rows"],
    )
    proof = {
        "same_node": before["path"]["node_id"] == after["path"]["node_id"],
        "fa_unchanged": (
            before["tree"]["full_tree_sha256"]
            == after["tree"]["full_tree_sha256"]
            and before["path"]["path_full_sha256"]
            == after["path"]["path_full_sha256"]
            and before["accounting"]["full_allocator"]
            == after["accounting"]["full_allocator"]
        ),
        "path_unchanged": (
            before["path"]["path_node_ids"]
            == after["path"]["path_node_ids"]
            and before["path"]["prefix_sha256"]
            == after["path"]["prefix_sha256"]
        ),
        "tree_unchanged": (
            before["tree"]["structure_sha256"]
            == after["tree"]["structure_sha256"]
        ),
        "only_target_mamba_changed": changed_mamba_nodes
        == [before["path"]["node_id"]],
        "sanity_check": True,
        "cascade_called": False,
        "fa_identity_unchanged": True,
    }
    recorder.record(
        target_checkpoint_id=handle.checkpoint_id,
        boundary="S3",
    )
    return {
        "ok": True,
        "nonce": request.get("nonce"),
        "scope": scope,
        "before": before,
        "after": after,
        "proof": proof,
        "formal_primitive": (
            "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
        ),
        "trace_rows": list(recorder.rows),
    }


def _run_checkpoint_control(scheduler: object, request: dict) -> dict:
    """处理诊断动作，其余动作完整转交既有端到端传输层。"""
    action = request.get("action")
    if action == _TRACE_EVICTION_ACTION:
        return _trace_eviction(scheduler, request)
    if action == _TRACE_SNAPSHOT_ACTION:
        return _trace_snapshot(scheduler, request)
    return _base_transport._run_checkpoint_control(scheduler, request)


def _wrapped_run_scheduler_process(*args, **kwargs):
    """安装诊断控制队列并保留既有请求前缀记录能力。"""
    scheduler_class = _scheduler_module.Scheduler
    original_init = scheduler_class.__init__
    if not getattr(original_init, "_flowstate_step12h9a_patched", False):

        def patched_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            _base_transport._install_match_instrumentation(self.tree_cache)
            _probe._run_checkpoint_control = _run_checkpoint_control
            port = _probe.install_control_server(
                self,
                requested_control_port(),
            )
            print(
                f"[STEP12H9A-TRANSPORT] 控制端口已就绪：{port}",
                flush=True,
            )

        patched_init._flowstate_step12h9a_patched = True
        scheduler_class.__init__ = patched_init

    original_on_idle = scheduler_class.on_idle
    if not getattr(original_on_idle, "_flowstate_step12h9a_patched", False):

        def patched_on_idle(self):
            state = getattr(self, "_wp3d_probe_state", None)
            if state is not None and self.is_fully_idle():
                state.drain_one(self)
            return original_on_idle(self)

        patched_on_idle._flowstate_step12h9a_patched = True
        scheduler_class.on_idle = patched_on_idle

    return _ORIGINAL_RUN_SCHEDULER_PROCESS(*args, **kwargs)


def requested_control_port() -> int:
    """读取连续驱逐诊断使用的本机控制端口。"""
    return int(os.environ.get("FLOWSTATE_STEP12H9A_PORT", "49948"))


class SequentialEvictionTraceGateEngine(_SGLangEngine):
    """安装连续驱逐边界诊断传输层的冻结引擎。"""

    run_scheduler_process_func = staticmethod(_wrapped_run_scheduler_process)
