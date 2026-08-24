"""仅供单节点图形处理器验证使用的调度器进程传输层。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIRECTORY = (
    _REPOSITORY_ROOT
    / "motivation"
    / "artifacts"
    / "wp3b_gate_20260820"
)
sys.path.insert(0, str(_ARTIFACT_DIRECTORY))

import targeted_probe as _probe
from sglang.srt.entrypoints.engine import Engine as _SglangEngine
from sglang.srt.managers import scheduler as _scheduler_module

from flowstate.adapters.sglang import RuntimeCheckpointHandle, SGLangAdapter


_FORMAL_ACTION = "flowstate_evict_mamba_only"
_ORIGINAL_CHECKPOINT_CONTROL = _probe._run_checkpoint_control
_ORIGINAL_RUN_SCHEDULER_PROCESS = _scheduler_module.run_scheduler_process


def _run_formal_checkpoint_control(scheduler, request: dict) -> dict:
    """把正式适配器调用放到调度器线程，并返回测试审计结果。"""
    if request.get("action") != _FORMAL_ACTION:
        return _ORIGINAL_CHECKPOINT_CONTROL(scheduler, request)

    cache = scheduler.tree_cache
    scope = _probe._validate_runtime_scope(scheduler, cache)
    token_ids = tuple(int(value) for value in request.get("token_ids") or ())
    extra_key = request.get("extra_key")
    before = _probe._snapshot(cache, token_ids, extra_key)
    handle = RuntimeCheckpointHandle(
        checkpoint_id=str(request["checkpoint_id"]),
        token_ids=token_ids,
        extra_key=extra_key,
        expected_node_id=int(request["expected_node_id"]),
        expected_prefix_digest=str(request["expected_prefix_sha256"]),
    )

    SGLangAdapter(cache).evict_mamba_only(handle)

    after = _probe._snapshot(cache, token_ids, extra_key)
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
    }
    result = {
        "ok": True,
        "nonce": request.get("nonce"),
        "scope": scope,
        "before": before,
        "after": after,
        "proof": proof,
        "formal_primitive": (
            "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
        ),
    }
    print(
        "[STEP5C-TRANSPORT] RESULT="
        + json.dumps(
            {
                "node": before["path"]["node_id"],
                "proof": proof,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _wrapped_run_scheduler_process(*args, **kwargs):
    """安装测试队列，并在调度器完全空闲时执行一条正式调用。"""
    scheduler_class = _scheduler_module.Scheduler
    original_init = scheduler_class.__init__
    if not getattr(original_init, "_flowstate_step5c_patched", False):

        def patched_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            _probe._run_checkpoint_control = _run_formal_checkpoint_control
            port = _probe.install_control_server(
                self,
                int(requested_control_port()),
            )
            print(
                f"[STEP5C-TRANSPORT] 控制端口已就绪：{port}",
                flush=True,
            )

        patched_init._flowstate_step5c_patched = True
        scheduler_class.__init__ = patched_init

    original_on_idle = scheduler_class.on_idle
    if not getattr(original_on_idle, "_flowstate_step5c_patched", False):

        def patched_on_idle(self):
            state = getattr(self, "_wp3d_probe_state", None)
            if state is not None and self.is_fully_idle():
                state.drain_one(self)
            return original_on_idle(self)

        patched_on_idle._flowstate_step5c_patched = True
        scheduler_class.on_idle = patched_on_idle

    return _ORIGINAL_RUN_SCHEDULER_PROCESS(*args, **kwargs)


def requested_control_port() -> int:
    """读取测试传输层使用的本机控制端口。"""
    import os

    return int(os.environ.get("FLOWSTATE_STEP5C_PORT", "49936"))


class FormalEvictionGateEngine(_SglangEngine):
    """仅用于单次图形处理器验证的引擎包装器。"""

    run_scheduler_process_func = staticmethod(_wrapped_run_scheduler_process)
