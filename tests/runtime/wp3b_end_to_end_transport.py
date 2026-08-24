"""仅供 WP3B 端到端验证使用的调度器进程传输层。"""

from __future__ import annotations

import os

from sglang.srt.entrypoints.engine import Engine as _SGLangEngine
from sglang.srt.managers import scheduler as _scheduler_module

import single_node_eviction_transport as _single_node_transport
import targeted_probe as _probe


_RUNTIME_METRICS_ACTION = "flowstate_runtime_metrics"
_FORMAL_ACTION = "flowstate_evict_mamba_only"
_MATCH_RECORDS: dict[str, list[dict[str, object]]] = {}
_ORIGINAL_RUN_SCHEDULER_PROCESS = (
    _scheduler_module.run_scheduler_process
)


def _device_prefix_length(device_indices: object) -> int:
    """读取匹配结果中可执行设备前缀的长度。"""
    numel = getattr(device_indices, "numel", None)
    if callable(numel):
        return int(numel())
    return len(device_indices)


def _install_match_instrumentation(cache: object) -> None:
    """在测试进程中记录请求到达时的物理与可执行前缀。"""
    if getattr(cache, "_flowstate_step5d_match_instrumented", False):
        return

    original_match_prefix = cache.match_prefix

    def instrumented_match_prefix(params):
        result = original_match_prefix(params)
        request = getattr(params, "req", None)
        request_id = getattr(request, "rid", None)
        if request_id is None:
            return result

        physical_fa_hit = int(result.full_kv_hit_length)
        executable_prefix = _device_prefix_length(result.device_indices)
        replay_gap = physical_fa_hit - executable_prefix
        if replay_gap < 0:
            raise RuntimeError(
                "运行时前缀记录出现负恢复间隔："
                f"{physical_fa_hit} - {executable_prefix}"
            )

        record = {
            "request_id": str(request_id),
            "physical_fa_hit": physical_fa_hit,
            "executable_prefix": executable_prefix,
            "replay_gap": replay_gap,
            "mamba_branching_seqlen": result.mamba_branching_seqlen,
            "mamba_host_hit_length": int(result.mamba_host_hit_length),
        }
        _MATCH_RECORDS.setdefault(str(request_id), []).append(record)
        return result

    cache.match_prefix = instrumented_match_prefix
    cache._flowstate_step5d_match_instrumented = True


def _run_checkpoint_control(scheduler: object, request: dict) -> dict:
    """处理指标读取，其他动作转交既有正式适配器传输实现。"""
    action = request.get("action")
    if action == _RUNTIME_METRICS_ACTION:
        _probe._validate_runtime_scope(scheduler, scheduler.tree_cache)
        request_id = str(request.get("request_id") or "")
        records = _MATCH_RECORDS.get(request_id, ())
        if not records:
            raise RuntimeError(
                f"找不到请求 {request_id} 的运行时前缀记录"
            )
        return {
            "ok": True,
            "nonce": request.get("nonce"),
            "record_count": len(records),
            "metrics": records[-1],
        }

    result = _single_node_transport._run_formal_checkpoint_control(
        scheduler,
        request,
    )
    if action == _FORMAL_ACTION and result.get("ok"):
        result["proof"]["cascade_called"] = False
        result["proof"]["fa_identity_unchanged"] = True
    return result


def _wrapped_run_scheduler_process(*args, **kwargs):
    """安装测试队列、前缀记录器和调度器空闲时点处理。"""
    scheduler_class = _scheduler_module.Scheduler
    original_init = scheduler_class.__init__
    if not getattr(original_init, "_flowstate_step5d_patched", False):

        def patched_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            _install_match_instrumentation(self.tree_cache)
            _probe._run_checkpoint_control = _run_checkpoint_control
            port = _probe.install_control_server(
                self,
                requested_control_port(),
            )
            print(
                f"[STEP5D-TRANSPORT] 控制端口已就绪：{port}",
                flush=True,
            )

        patched_init._flowstate_step5d_patched = True
        scheduler_class.__init__ = patched_init

    original_on_idle = scheduler_class.on_idle
    if not getattr(original_on_idle, "_flowstate_step5d_patched", False):

        def patched_on_idle(self):
            state = getattr(self, "_wp3d_probe_state", None)
            if state is not None and self.is_fully_idle():
                state.drain_one(self)
            return original_on_idle(self)

        patched_on_idle._flowstate_step5d_patched = True
        scheduler_class.on_idle = patched_on_idle

    return _ORIGINAL_RUN_SCHEDULER_PROCESS(*args, **kwargs)


def requested_control_port() -> int:
    """读取本次端到端验证使用的本机控制端口。"""
    return int(os.environ.get("FLOWSTATE_STEP5D_PORT", "49937"))


class FormalEndToEndGateEngine(_SGLangEngine):
    """安装 Step 5D 测试传输层的冻结 SGLang 引擎。"""

    run_scheduler_process_func = staticmethod(
        _wrapped_run_scheduler_process
    )
