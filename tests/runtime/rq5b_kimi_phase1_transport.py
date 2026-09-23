"""Kimi 双张量并行卡的只读状态探针与单组件驱逐入口。"""

from __future__ import annotations

import os

from sglang.srt.entrypoints.engine import Engine as SGLangEngine
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

import targeted_probe as probe

from flowstate.adapters.sglang import RuntimeCheckpointHandle, SGLangAdapter


ORIGINAL_RUN = scheduler_module.run_scheduler_process


def runtime_scope(scheduler: object, cache: object) -> dict:
    """要求双卡调度器空闲，并确认前缀状态相关开关仍然生效。"""
    args = scheduler.server_args
    rank = int(scheduler.ps.tp_rank)
    checks = {
        "tp_size": int(args.tp_size) == 2,
        "tp_rank": rank in (0, 1),
        "scheduler_idle": bool(scheduler.is_fully_idle()),
        "overlap_disabled": bool(args.disable_overlap_schedule),
        "radix_enabled": not bool(args.disable_radix_cache),
        "hybrid_prefix_enabled": bool(args.uses_mamba_radix_cache),
        "tracking_enabled": int(args.mamba_track_interval) > 0,
        "cache_type": type(cache).__name__ == "UnifiedRadixCache",
        "no_hicache": not bool(getattr(cache.tree_core, "enable_hicache", False)),
        "no_int8_pool": getattr(cache.req_to_token_pool, "mamba_ckpt_pool", None) is None,
        "full_component": ComponentType.FULL in cache.components,
        "recurrent_component": ComponentType.MAMBA in cache.components,
        "no_pending_insert": not cache.tree_core.has_ongoing_insert(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Kimi 运行时门禁失败：{checks}")
    return {
        "tp_rank": rank,
        "tp_size": int(args.tp_size),
        "cache_strategy": str(args.mamba_radix_cache_strategy),
        "track_interval": int(args.mamba_track_interval),
        "attention_backend": str(args.attention_backend),
        "linear_attn_backend": str(args.linear_attn_backend),
        "checks": checks,
    }


def control(scheduler: object, request: dict) -> dict:
    """在调度器安全时点检查双组件，或仅驱逐指定 KDA 检查点。"""
    cache = scheduler.tree_cache
    scope = runtime_scope(scheduler, cache)
    action = request.get("action")
    if action == "runtime_gate":
        return {
            "ok": True,
            "scope": scope,
            "tree": probe._global_maps(cache),
            "accounting": probe._accounting_snapshot(cache),
        }
    if action == "inspect":
        paths = {
            str(row["checkpoint_id"]): probe._path_snapshot(
                cache, tuple(int(x) for x in row["token_ids"]), row.get("extra_key")
            )
            for row in request["handles"]
        }
        return {
            "ok": True,
            "scope": scope,
            "paths": paths,
            "tree": probe._global_maps(cache),
            "accounting": probe._accounting_snapshot(cache),
        }
    if action == "evict_target":
        rows = {str(row["checkpoint_id"]): row for row in request["handles"]}
        target_id = str(request["target_id"])
        if target_id not in rows or len(rows) != 2:
            raise RuntimeError("目标与两个检查点句柄不一致")
        before = control(scheduler, {**request, "action": "inspect"})
        target = rows[target_id]
        handle = RuntimeCheckpointHandle(
            checkpoint_id=target_id,
            token_ids=tuple(int(x) for x in target["token_ids"]),
            extra_key=target.get("extra_key"),
            expected_node_id=int(target["expected_node_id"]),
            expected_prefix_digest=str(target["expected_prefix_digest"]),
        )
        SGLangAdapter(cache).evict_mamba_only(handle)
        cache.sanity_check()
        after = control(scheduler, {**request, "action": "inspect"})
        target_before = before["paths"][target_id]
        target_after = after["paths"][target_id]
        other_id = next(name for name in rows if name != target_id)
        other_before = before["paths"][other_id]
        other_after = after["paths"][other_id]
        changed = probe._changed_mamba_nodes(
            before["tree"]["mamba_rows"], after["tree"]["mamba_rows"]
        )
        checks = {
            "target_removed": target_before["target_mamba_present"] and not target_after["target_mamba_present"],
            "other_recurrent_unchanged": other_before["target_mamba_slots"] == other_after["target_mamba_slots"] and other_after["target_mamba_present"],
            "attention_unchanged": before["tree"]["full_tree_sha256"] == after["tree"]["full_tree_sha256"] and before["accounting"]["full_allocator"] == after["accounting"]["full_allocator"],
            "structure_unchanged": before["tree"]["structure_sha256"] == after["tree"]["structure_sha256"],
            "recurrent_change_exact": changed == [int(target_before["node_id"])],
            "no_unexpected_rematerialization": not target_after["target_mamba_present"],
        }
        return {
            "ok": all(checks.values()),
            "scope": scope,
            "before": before,
            "after": after,
            "changed_recurrent_nodes": changed,
            "checks": checks,
        }
    raise RuntimeError(f"不支持的阶段一控制动作：{action}")


def wrapped_run_scheduler_process(*args, **kwargs):
    """为两个调度器 rank 分别建立只在空闲时处理请求的本地端口。"""
    scheduler_class = scheduler_module.Scheduler
    original_init = scheduler_class.__init__

    def patched_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        probe._run_checkpoint_control = control
        probe._validate_runtime_scope = runtime_scope
        rank = int(self.ps.tp_rank)
        port = int(os.environ.get("FLOWSTATE_RQ5B_PORT", "49961")) + rank
        probe.install_control_server(self, port)
        print(f"[RQ5-B] rank={rank} 控制端口={port}", flush=True)

    original_on_idle = scheduler_class.on_idle

    def patched_on_idle(self):
        state = getattr(self, "_wp3d_probe_state", None)
        if state is not None and self.is_fully_idle():
            state.drain_one(self)
        return original_on_idle(self)

    scheduler_class.__init__ = patched_init
    scheduler_class.on_idle = patched_on_idle
    return ORIGINAL_RUN(*args, **kwargs)


class KimiPhase1Engine(SGLangEngine):
    """仅用于 Kimi 阶段一双 rank 状态观测和单组件隔离。"""

    run_scheduler_process_func = staticmethod(wrapped_run_scheduler_process)
