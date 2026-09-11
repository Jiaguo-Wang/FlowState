"""RQ6-E 批量读取与批量循环状态协调的调度器传输层。"""

from __future__ import annotations

import hashlib
import json
import os
from time import perf_counter_ns

from sglang.srt.entrypoints.engine import Engine as _SGLangEngine
from sglang.srt.managers import scheduler as _scheduler_module

import targeted_probe as _probe
import wp3b_end_to_end_transport as _base_transport

from flowstate.adapters.sglang import RuntimeCheckpointHandle, SGLangAdapter


_BATCH_INTROSPECTION = "flowstate_batch_introspection"
_BATCH_RECONCILIATION = "flowstate_batch_reconciliation"
_ORIGINAL_RUN_SCHEDULER_PROCESS = _scheduler_module.run_scheduler_process


def _canonical_digest(value: object) -> str:
    """对批量状态视图生成稳定摘要。"""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_handles(request: dict) -> dict[str, RuntimeCheckpointHandle]:
    """解析完整候选句柄并拒绝重复或缺失身份字段。"""
    result: dict[str, RuntimeCheckpointHandle] = {}
    for row in request.get("handles") or ():
        handle = RuntimeCheckpointHandle(
            checkpoint_id=str(row["checkpoint_id"]),
            token_ids=tuple(int(value) for value in row["token_ids"]),
            extra_key=row.get("extra_key"),
            expected_node_id=int(row["expected_node_id"]),
            expected_prefix_digest=str(row["expected_prefix_digest"]),
        )
        if handle.checkpoint_id in result:
            raise RuntimeError("批量请求含重复 checkpoint_id")
        result[handle.checkpoint_id] = handle
    if not result:
        raise RuntimeError("批量请求缺少候选句柄")
    expected = request.get("candidate_ids")
    if expected is not None and list(result) != [str(value) for value in expected]:
        raise RuntimeError("批量句柄顺序或候选集合不一致")
    return result


def _batch_view(
    scheduler: object,
    handles: dict[str, RuntimeCheckpointHandle],
) -> dict[str, object]:
    """在同一个调度器安全时点生成一致且无副作用的全量状态视图。"""
    cache = scheduler.tree_cache
    scope = _probe._validate_runtime_scope(scheduler, cache)
    paths = {
        checkpoint_id: _probe._path_snapshot(
            cache, handle.token_ids, handle.extra_key
        )
        for checkpoint_id, handle in handles.items()
    }
    view = {
        "scope": scope,
        "tree": _probe._global_maps(cache),
        "accounting": _probe._accounting_snapshot(cache),
        "paths": paths,
    }
    return {"view": view, "view_digest": _canonical_digest(view)}


def _validate_identity_and_residency(
    handles: dict[str, RuntimeCheckpointHandle],
    view: dict[str, object],
) -> None:
    """在任何状态变更前验证精确身份、FA 路径与循环状态驻留。"""
    paths = view["paths"]
    for checkpoint_id, handle in handles.items():
        path = paths[checkpoint_id]
        if int(path["node_id"]) != handle.expected_node_id:
            raise RuntimeError(f"{checkpoint_id} 的节点身份不一致")
        if str(path["prefix_sha256"]) != handle.expected_prefix_digest:
            raise RuntimeError(f"{checkpoint_id} 的前缀摘要不一致")
        if not path["target_full_present"] or not path["path_full_all_present"]:
            raise RuntimeError(f"{checkpoint_id} 的 FA 路径未完整驻留")
        if not path["target_mamba_present"]:
            raise RuntimeError(f"{checkpoint_id} 的循环状态未驻留")


def _proof(
    before: dict[str, object],
    after: dict[str, object],
    selected_ids: tuple[str, ...],
    evicted_ids: tuple[str, ...],
) -> dict[str, object]:
    """一次统一验证批量协调后的驻留、结构、FA 与分配器变化。"""
    before_paths = before["paths"]
    after_paths = after["paths"]
    expected_nodes = sorted(
        int(before_paths[checkpoint_id]["node_id"])
        for checkpoint_id in evicted_ids
    )
    changed_nodes = _probe._changed_mamba_nodes(
        before["tree"]["mamba_rows"], after["tree"]["mamba_rows"]
    )
    selected_resident = all(
        after_paths[checkpoint_id]["target_mamba_present"]
        for checkpoint_id in selected_ids
    )
    evicted_absent = all(
        not after_paths[checkpoint_id]["target_mamba_present"]
        for checkpoint_id in evicted_ids
    )
    identities_unchanged = all(
        before_paths[checkpoint_id]["node_id"]
        == after_paths[checkpoint_id]["node_id"]
        and before_paths[checkpoint_id]["prefix_sha256"]
        == after_paths[checkpoint_id]["prefix_sha256"]
        and before_paths[checkpoint_id]["path_node_ids"]
        == after_paths[checkpoint_id]["path_node_ids"]
        for checkpoint_id in before_paths
    )
    fa_preserved = bool(
        before["tree"]["full_tree_sha256"]
        == after["tree"]["full_tree_sha256"]
        and before["tree"]["structure_sha256"]
        == after["tree"]["structure_sha256"]
        and before["accounting"]["full_allocator"]
        == after["accounting"]["full_allocator"]
        and all(
            before_paths[checkpoint_id]["path_full_sha256"]
            == after_paths[checkpoint_id]["path_full_sha256"]
            and after_paths[checkpoint_id]["target_full_present"]
            and after_paths[checkpoint_id]["path_full_all_present"]
            for checkpoint_id in before_paths
        )
    )
    released_slots = sum(
        len(before_paths[checkpoint_id]["target_mamba_slots"])
        for checkpoint_id in evicted_ids
    )
    available_delta = (
        int(after["accounting"]["mamba_available"])
        - int(before["accounting"]["mamba_available"])
    )
    recurrent_exact = bool(
        changed_nodes == expected_nodes
        and selected_resident
        and evicted_absent
        and available_delta == released_slots
        and int(before["tree"]["mamba_node_count"])
        - int(after["tree"]["mamba_node_count"])
        == len(evicted_ids)
    )
    passed = bool(
        recurrent_exact and identities_unchanged and fa_preserved
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "selected_resident": selected_resident,
        "evicted_absent": evicted_absent,
        "identities_unchanged": identities_unchanged,
        "fa_preserved": fa_preserved,
        "expected_changed_mamba_node_ids": expected_nodes,
        "changed_mamba_node_ids": changed_nodes,
        "expected_released_slots": released_slots,
        "mamba_available_delta": available_delta,
        "recurrent_change_exact": recurrent_exact,
        "native_recurrent_eviction": changed_nodes != expected_nodes,
        "unexpected_rematerialization": not evicted_absent,
        "fa_cascade": not fa_preserved,
    }


def _batch_introspection(scheduler: object, request: dict) -> dict[str, object]:
    """用一次控制消息返回当前 epoch 的一致性只读状态视图。"""
    started = perf_counter_ns()
    handles = _parse_handles(request)
    result = _batch_view(scheduler, handles)
    return {
        "ok": True,
        "nonce": request.get("nonce"),
        "candidate_ids": list(handles),
        **result,
        "worker_ns": perf_counter_ns() - started,
        "state_mutated": False,
    }


def _batch_reconciliation(scheduler: object, request: dict) -> dict[str, object]:
    """用一次控制消息执行本 epoch 的所有循环状态协调与统一后验验证。"""
    started = perf_counter_ns()
    handles = _parse_handles(request)
    candidate_ids = tuple(handles)
    selected_ids = tuple(str(value) for value in request.get("selected_ids") or ())
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("selected set 含重复检查点")
    if not set(selected_ids).issubset(handles):
        raise RuntimeError("selected set 含非候选检查点")
    evicted_ids = tuple(sorted(set(candidate_ids) - set(selected_ids)))

    before_result = _batch_view(scheduler, handles)
    expected_digest = str(request.get("expected_view_digest") or "")
    if expected_digest != before_result["view_digest"]:
        raise RuntimeError("批量协调前状态与只读决策视图不一致")
    before = before_result["view"]
    _validate_identity_and_residency(handles, before)

    adapter = SGLangAdapter(scheduler.tree_cache)
    adapter.validate_runtime_capabilities()
    completed = []
    operation_ns = []
    for checkpoint_id in evicted_ids:
        operation_start = perf_counter_ns()
        adapter.evict_mamba_only(handles[checkpoint_id])
        operation_ns.append(perf_counter_ns() - operation_start)
        completed.append(checkpoint_id)

    scheduler.tree_cache.sanity_check()
    after_result = _batch_view(scheduler, handles)
    proof = _proof(before, after_result["view"], selected_ids, evicted_ids)
    if proof["status"] != "PASS":
        raise RuntimeError(f"批量协调统一验证失败：{proof}")
    return {
        "ok": True,
        "nonce": request.get("nonce"),
        "candidate_ids": list(candidate_ids),
        "selected_ids": list(selected_ids),
        "evicted_ids": list(evicted_ids),
        "completed_eviction_ids": completed,
        "before": before,
        "after": after_result["view"],
        "before_view_digest": before_result["view_digest"],
        "after_view_digest": after_result["view_digest"],
        "proof": proof,
        "operation_ns": operation_ns,
        "worker_ns": perf_counter_ns() - started,
        "post_validation_count": 1,
    }


def _run_checkpoint_control(scheduler: object, request: dict) -> dict:
    """处理 RQ6-E 批量动作，其余动作交给原冻结传输实现。"""
    action = request.get("action")
    if action == _BATCH_INTROSPECTION:
        return _batch_introspection(scheduler, request)
    if action == _BATCH_RECONCILIATION:
        return _batch_reconciliation(scheduler, request)
    return _base_transport._run_checkpoint_control(scheduler, request)


def _wrapped_run_scheduler_process(*args, **kwargs):
    """安装独立批量控制入口并保留原请求前缀记录能力。"""
    scheduler_class = _scheduler_module.Scheduler
    original_init = scheduler_class.__init__
    if not getattr(original_init, "_flowstate_rq6e_patched", False):

        def patched_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            _base_transport._install_match_instrumentation(self.tree_cache)
            _probe._run_checkpoint_control = _run_checkpoint_control
            port = _probe.install_control_server(self, requested_control_port())
            print(f"[RQ6-E] 批量控制端口已就绪：{port}", flush=True)

        patched_init._flowstate_rq6e_patched = True
        scheduler_class.__init__ = patched_init

    original_on_idle = scheduler_class.on_idle
    if not getattr(original_on_idle, "_flowstate_rq6e_patched", False):

        def patched_on_idle(self):
            state = getattr(self, "_wp3d_probe_state", None)
            if state is not None and self.is_fully_idle():
                state.drain_one(self)
            return original_on_idle(self)

        patched_on_idle._flowstate_rq6e_patched = True
        scheduler_class.on_idle = patched_on_idle
    return _ORIGINAL_RUN_SCHEDULER_PROCESS(*args, **kwargs)


def requested_control_port() -> int:
    """读取 RQ6-E 独立批量控制端口。"""
    return int(os.environ.get("FLOWSTATE_RQ6E_PORT", "49951"))


class RQ6BatchedControlEngine(_SGLangEngine):
    """安装 RQ6-E 批量控制传输层的冻结引擎。"""

    run_scheduler_process_func = staticmethod(_wrapped_run_scheduler_process)
