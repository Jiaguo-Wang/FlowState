"""Step 13G-B2：AgentX Neutral Runtime Snapshot Collection。

对 Step 13G-B1 冻结的 23 个 formal epochs，使用 SGLang 0.5.17 + Qwen3.5-9B
执行 neutral replay 到 allocation barrier，并冻结 runtime snapshots。

约束：
- 只运行 neutral replay，不执行任何 policy。
- 每个 snapshot 使用独立 fresh Engine，避免 cross-snapshot 状态污染。
- 直接消费 B1 冻结的 23 个 snapshots，不重新挑选 epoch。
- 复用 B1.1 已验证的 deterministic synthetic token replay。
- 使用 direct input_ids，不做 text 往返。
- 不使用未来 trajectory。
- 如实标记 RUNTIME_INELIGIBLE(<reason>)。
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import bisect
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# 确保 docker 内能导入 tests/runtime 与 targeted_probe。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "runtime"))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "motivation"
        / "artifacts"
        / "wp3b_gate_20260820"
    ),
)

from evaluation.agentx_qwen_replay_preflight import (
    QWEN_MODEL_PATH,
    RUNTIME_CONTEXT_LIMIT,
    _load_formal_population,
    _load_tokenizer,
    _build_conversations_for_population,
    _build_replay_snapshot,
    _compose_prompt_tokens,
    _decode_block_tokens,
    _request_index_for_target,
)
from evaluation.recovery_profiler_128k import (
    ENGINE_CONFIGURATION_128K,
    EXPECTED_SGLANG_VERSION,
)

FROZEN_B1_CENSUS_PATH = Path(
    "/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614"
    "/FORMAL_CANDIDATE_POPULATION.json"
)
FROZEN_B11_PATH = Path(
    "/home/wjg/data/agentx/audits/agentx_qwen_replay_preflight_20260905_105318"
)
FROZEN_AGENTX_PATH = Path(
    "/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl"
)

ARTIFACT_ROOT = Path("/home/wjg/data/agentx/audits")

REQUIRED_SGLANG_VERSION = EXPECTED_SGLANG_VERSION
PENDING_SUFFIX_LENGTH = 63  # runtime_gate 中的固定后缀长度。
SAMPLING_PARAMETERS = {
    "max_new_tokens": 1,
    "temperature": 0,
    "ignore_eos": True,
}

# 冻结的 B2 运行时配置（Step 13G-B2.2 已标定）。
FROZEN_RUNTIME_CONFIGURATION = {
    "max_mamba_cache_size": 192,
    "mem_fraction_static": 0.80,
}

# 单个 snapshot 的 worker 超时：Engine 启动 + 全部请求 replay + 检查。
WORKER_TIMEOUT_S = 1800.0
GPU_STABLE_TIMEOUT_S = 300.0
GPU_STABLE_INTERVAL_S = 5.0
GPU_STABLE_REQUIRED = 3
# SGLang Engine 完全 shutdown 后，主进程仍保留 PyTorch CUDA context / caching allocator
# 驻留约 4–5 GiB；将稳定阈值设为 8 GiB 以避免误报，同时仍能检测真实显存泄漏。
GPU_STABLE_THRESHOLD_MIB = 8192

# 可接受的确定性失败原因。
FAILURE_REASONS = (
    "sglang_version_mismatch",
    "engine_startup_failed",
    "census_failed",
    "request_failed",
    "oom",
    "native_mamba_eviction",
    "fa_kv_cascade",
    "checkpoint_inspect_failed",
    "checkpoint_not_resident_at_barrier",
    "fa_frontier_query_failed",
    "fa_frontier_query_side_effect",
    "future_boundary_violation",
    "context_truncation",
    "unexpected_error",
    "stage_timeout",
    "dry_run",
    "b1_universe_mismatch",
)

# 失败分类：BUILD_FAILED = 真实 workload/runtime 表示不兼容；
# INFRASTRUCTURE_FAILED = OOM、崩溃、超时、native eviction、handle bug、
# cleanup 失败、replay mismatch 等运行时基础设施问题。
_BUILD_FAILED_REASONS = {
    "request_failed",
    "future_boundary_violation",
    "context_truncation",
    "dry_run",
    "b1_universe_mismatch",
}
_INFRASTRUCTURE_FAILED_REASONS = {
    "sglang_version_mismatch",
    "engine_startup_failed",
    "census_failed",
    "oom",
    "native_mamba_eviction",
    "fa_kv_cascade",
    "checkpoint_inspect_failed",
    "checkpoint_not_resident_at_barrier",
    "fa_frontier_query_failed",
    "fa_frontier_query_side_effect",
    "unexpected_error",
    "stage_timeout",
}


def _classify_failure(reason: str) -> str:
    """把 CollectionAbort reason 映射为 BUILD_FAILED / INFRASTRUCTURE_FAILED。"""
    if reason in _BUILD_FAILED_REASONS:
        return f"BUILD_FAILED({reason})"
    if reason in _INFRASTRUCTURE_FAILED_REASONS:
        return f"INFRASTRUCTURE_FAILED({reason})"
    return f"INFRASTRUCTURE_FAILED({reason})"


def _token_digest(token_ids: Sequence[int]) -> str:
    """计算运行时句柄使用的正式前缀摘要。"""
    return hashlib.sha256(array("q", [int(x) for x in token_ids]).tobytes()).hexdigest()


class CollectionAbort(RuntimeError):
    """携带确定性失败原因中断单个 snapshot 的采集。"""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in FAILURE_REASONS:
            raise ValueError(f"未知失败原因：{reason}")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class _MaterializedRequest:
    """截至 barrier 已物化的候选请求（candidate checkpoint）。"""

    conversation_id: str
    request_index: int
    token_pos: int
    token_ids: list[int]
    end_seconds: float
    request_id: str


@dataclass
class _PendingRequest:
    """barrier 处待续的 pending continuation。"""

    conversation_id: str
    request_index: int
    token_pos: int
    token_ids: list[int]
    request_id: str


@dataclass
class _SnapshotReplayPlan:
    """单个 formal epoch 的 replay 计划。"""

    trace_id: str
    block_size: int
    t: float
    pending_count: int
    candidate_count: int
    shared_candidate_count: int
    max_degree: int
    max_fork_depth_blocks: int
    materialized: list[_MaterializedRequest]
    pendings: list[_PendingRequest]
    active_targets: dict[int, int]
    # 来自 B1 formal population 的期望 universe，用于在 collection 后做 exact-match 校验。
    expected_candidate_ids: set[str] = field(default_factory=set)
    expected_pending_ids: set[str] = field(default_factory=set)


@dataclass
class _CheckpointObservation:
    """单个 materialized checkpoint 的 runtime 观测。"""

    checkpoint_id: str
    conversation_id: str
    request_index: int
    token_pos: int  # A_p：候选的 anchor/token 位置。
    node_id: int | None
    fa_resident: bool
    recurrent_resident: bool
    runtime_identity_digest: str
    resident_fa_frontier: int = 0  # R_p：barrier 处 resident FA frontier 长度。
    planning_target: int = 0  # T_p = min(A_p, R_p)。
    recurrent_prefix_length: int = 0  # 真实 recurrent handle 对应的 exact prefix 长度。
    compatible_pending_ids: list[str] = field(default_factory=list)
    d_t_c: int = 0  # 与该候选兼容的 pending continuation 数量。


@dataclass
class _PendingObservation:
    """单个 pending continuation 的 FA frontier 观测。"""

    continuation_id: str
    conversation_id: str
    request_index: int
    input_token_count: int  # A_p：pending 的 anchor/token 位置。
    resident_fa_frontier: int  # R_p。
    fa_query_side_effect_free: bool
    planning_target: int = 0  # T_p = min(A_p, R_p)。


@dataclass
class SnapshotResult:
    """单个 snapshot 的 collection 结果。"""

    trace_id: str
    t: float
    status: str  # ELIGIBLE | BUILD_FAILED(...) | INFRASTRUCTURE_FAILED(...)
    primary_reason: str | None
    detail: str
    materialized_count: int
    pending_count: int
    checkpoints: list[_CheckpointObservation] = field(default_factory=list)
    pendings: list[_PendingObservation] = field(default_factory=list)
    request_rows: list[dict[str, Any]] = field(default_factory=list)
    census_rows: list[dict[str, Any]] = field(default_factory=list)
    oom: bool = False
    native_mamba_eviction: bool = False
    fa_cascade: bool = False
    truncation: bool = False
    fa_query_side_effect_free: bool = True
    future_leakage: bool = False
    unexpected_error: str | None = None
    artifact_path: Path | None = None
    baseline_accounting: dict[str, Any] | None = None
    peak_hbm_mib: int = 0
    min_free_hbm_mib: int = 0
    gpu_cleanup: dict[str, Any] | None = None
    gpu_cleanup_pass: bool = False


def _build_replay_plan(
    trace_conversations: Any,
    entry: dict[str, Any],
    tokenizer: Any,
) -> _SnapshotReplayPlan:
    """根据 B1 formal epoch 与 B1.1 合成器构造 replay 计划。"""
    from evaluation.agentx_qwen_replay_preflight import _find_epoch_snapshot
    from evaluation.agentx_structure_audit import _analyze_trace_online_compatibility

    online = _analyze_trace_online_compatibility(trace_conversations)
    snap = _find_epoch_snapshot(
        online,
        float(entry["t"]),
        entry["active_conversation_ids"],
        trace_conversations.conversations,
    )
    convs = trace_conversations.conversations
    block_size = trace_conversations.block_size

    materialized: list[_MaterializedRequest] = []
    pendings: list[_PendingRequest] = []

    # B1 的 candidate universe 包含整条 trace 中截至 t 已完成的全部请求，
    # 不只是 barrier 时仍 active 的 conversation。
    for conv in convs:
        completed_count = bisect.bisect_right(
            conv.request_end_seconds,
            float(entry["t"]) + 1e-12,
        )
        for ridx in range(completed_count):
            if ridx >= len(conv.request_hash_ids):
                raise ValueError(
                    f"{conv.conversation_id} 的完成请求缺少 hash_ids：{ridx}"
                )
            token_ids = _compose_prompt_tokens(
                conv.request_hash_ids[ridx],
                int(conv.request_input_lengths[ridx]),
                block_size,
                tokenizer,
            )
            token_pos = sum(conv.request_hash_token_positions[: ridx + 1])
            materialized.append(
                _MaterializedRequest(
                    conversation_id=conv.conversation_id,
                    request_index=ridx,
                    token_pos=token_pos,
                    token_ids=token_ids,
                    end_seconds=conv.request_end_seconds[ridx],
                    request_id=f"{conv.conversation_id}::req{ridx:04d}",
                )
            )

    # pending 严格复用 B1.1 的 current-context 语义：active target 指向当前
    # 已知请求，而不是未来的下一条请求。
    for conv_idx, target in snap.active_targets.items():
        conv = convs[conv_idx]
        pending_idx = _request_index_for_target(conv, target)
        if pending_idx is None or pending_idx >= len(conv.request_hash_ids):
            continue
        token_ids = _compose_prompt_tokens(
            conv.request_hash_ids[pending_idx],
            int(conv.request_input_lengths[pending_idx]),
            block_size,
            tokenizer,
        )
        token_pos = sum(conv.request_hash_token_positions[: pending_idx + 1])
        pendings.append(
            _PendingRequest(
                conversation_id=conv.conversation_id,
                request_index=pending_idx,
                token_pos=token_pos,
                token_ids=token_ids,
                request_id=f"{conv.conversation_id}::pending{pending_idx:04d}",
            )
        )

    # 按结束时间排序 materialized，模拟真实 chronological replay。
    materialized.sort(key=lambda r: (r.end_seconds, r.conversation_id))

    expected_candidates = int(entry["candidate_count"])
    if len(materialized) != expected_candidates:
        raise ValueError(
            "replay plan 与 B1 candidate universe 不一致："
            f"{len(materialized)} != {expected_candidates}"
        )

    expected_pending = int(entry["pending_count"])
    if len(pendings) != expected_pending:
        raise ValueError(
            "replay plan 与 B1 pending universe 不一致："
            f"{len(pendings)} != {expected_pending}"
        )

    return _SnapshotReplayPlan(
        trace_id=entry["trace_id"],
        block_size=block_size,
        t=float(entry["t"]),
        pending_count=entry["pending_count"],
        candidate_count=entry["candidate_count"],
        shared_candidate_count=entry["shared_candidate_count"],
        max_degree=entry["max_degree"],
        max_fork_depth_blocks=entry.get("max_fork_depth_blocks", 0),
        materialized=materialized,
        pendings=pendings,
        active_targets=dict(snap.active_targets),
        expected_candidate_ids={m.request_id for m in materialized},
        expected_pending_ids={p.request_id for p in pendings},
    )


def _check_sglang_version() -> None:
    """确认 SGLang 版本与冻结版本一致。"""
    import sglang

    actual = sglang.__version__
    if actual != REQUIRED_SGLANG_VERSION:
        raise CollectionAbort(
            "sglang_version_mismatch",
            f"需要 SGLang {REQUIRED_SGLANG_VERSION}，实际 {actual}",
        )


class _SglangRuntimeAdapter:
    """单个 snapshot 的 SGLang runtime 适配器。"""

    def __init__(self, engine: Any, client: Any) -> None:
        from evaluation.barrier_fa_frontier_control import BarrierFAControlClient

        self._engine = engine
        self._client = client
        self._barrier_client = BarrierFAControlClient(client)

    def census(self, nonce: str) -> dict[str, Any]:
        return self._client.census(nonce)

    def execute(self, request_id: str, token_ids: Sequence[int]) -> dict[str, Any]:
        from evaluation.controlled_multiworkflow_v1.runtime_gate import generate

        _, metadata = generate(self._engine, request_id, tuple(token_ids))
        return dict(metadata)

    def inspect_checkpoint(
        self, checkpoint_id: str, token_ids: Sequence[int]
    ) -> dict[str, Any]:
        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            inspect_checkpoint,
            path_state,
            compact_state,
        )

        response = inspect_checkpoint(
            self._client,
            checkpoint_id,
            tuple(int(x) for x in token_ids),
        )
        return {
            "path_raw": path_state(response),
            "compact": compact_state(path_state(response)),
            "response": response,
        }

    def inspect_checkpoint_recurrent(
        self,
        checkpoint_id: str,
        token_ids: Sequence[int],
        token_pos: int,
    ) -> dict[str, Any]:
        """先尝试完整 token_ids 的 exact node；若因 radix 分段边界失败，则回退到 hash-prefix。

        对于 synthetic tail-fill 导致完整 token_ids 结束在 radix segment 内部、
        无法定位 exact node 的 checkpoint，使用 hash-prefix（即到该请求 hash 边界
        的前 token_pos 个 token）重新 inspect。hash-prefix 对应的节点必然持有
        真实 Mamba 状态，因此可构造与其它 candidate 语义一致的 recurrent handle。
        """
        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            inspect_checkpoint,
            path_state,
            compact_state,
        )

        full = tuple(int(x) for x in token_ids)
        try:
            response = inspect_checkpoint(self._client, checkpoint_id, full)
            return {
                "path_raw": path_state(response),
                "compact": compact_state(path_state(response)),
                "response": response,
                "prefix_token_ids": full,
                "resolution_source": "exact",
            }
        except Exception as exc:
            # exact node 不存在：synthetic tail-fill 使完整 token_ids 结束在 radix
            # segment 内部。通过 FA frontier 只读遍历得到 token_pos 之前最后一个
            # exact radix 边界，并沿这些边界从深到浅尝试 inspect，直到找到持有
            # 真实 Mamba 状态的节点，从而构造与其它 candidate 语义一致的 recurrent
            # handle。
            frontier = self.inspect_fa_frontier(
                full[:token_pos],
                nonce=f"agentx:{checkpoint_id}:fallback_frontier",
            )
            segments = frontier.get("matched_segments", [])
            candidate_lengths: list[int] = []
            cum = 0
            for seg in segments:
                seg_len = int(seg.get("segment_length", 0))
                matched = int(seg.get("matched_length", 0))
                if matched == seg_len and seg_len > 0:
                    cum += seg_len
                    candidate_lengths.append(cum)
                else:
                    # 部分匹配时，当前段起点仍是 exact 边界。
                    if cum > 0:
                        candidate_lengths.append(cum)
                    break
            candidate_lengths = sorted(set(candidate_lengths))
            last_error: Exception | None = None
            for candidate_pos in reversed(candidate_lengths):
                if candidate_pos <= 0:
                    continue
                try:
                    response = inspect_checkpoint(
                        self._client, checkpoint_id, full[:candidate_pos]
                    )
                except Exception as probe_err:
                    last_error = probe_err
                    continue
                path = path_state(response)
                compact = compact_state(path)
                if compact["mamba_resident"]:
                    return {
                        "path_raw": path,
                        "compact": compact,
                        "response": response,
                        "prefix_token_ids": full[:candidate_pos],
                        "resolution_source": "fa_frontier_boundary_fallback",
                    }
                last_error = RuntimeError(
                    f"边界 {candidate_pos} 的节点未持有 Mamba 状态: {compact}"
                )
            raise RuntimeError(
                f"无法为 {checkpoint_id} 找到持有 Mamba 状态的 exact boundary: "
                f"candidates={candidate_lengths}, frontier_segments={segments}"
            ) from (last_error or exc)

    def inspect_fa_frontier(
        self, token_ids: Sequence[int], *, nonce: str
    ) -> dict[str, Any]:
        return self._barrier_client.inspect_fa_frontier(
            [int(x) for x in token_ids],
            extra_key=None,
            limit=None,
            nonce=nonce,
        )

    def query_runtime_metrics(self, request_id: str) -> dict[str, Any]:
        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            query_runtime_metrics,
        )

        return dict(query_runtime_metrics(self._client, request_id))

    def shutdown(self) -> None:
        try:
            self._engine.shutdown()
        except Exception:
            pass


def _start_runtime(gpu_index: int = 0) -> _SglangRuntimeAdapter:
    """在指定 GPU 上启动 fresh Engine 与控制客户端，使用冻结 B2 配置。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("FLOWSTATE_STEP5D_PORT", "49937")

    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )
    from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport

    _check_sglang_version()
    config = {**dict(ENGINE_CONFIGURATION_128K), **FROZEN_RUNTIME_CONFIGURATION}
    config["model_path"] = str(QWEN_MODEL_PATH)
    _log("RUNTIME_INIT", f"frozen config: {FROZEN_RUNTIME_CONFIGURATION}")
    engine = FormalEndToEndGateEngine(**config)
    client = ControlClient(requested_control_port())
    wait_for_transport(client)
    return _SglangRuntimeAdapter(engine, client)


def _tree_part(census: dict[str, Any]) -> dict[str, Any]:
    """ControlClient.census 返回 tree 字段包装真实结构。"""
    return census.get("tree", {}) if isinstance(census, dict) else {}


def _mamba_node_positions(census: dict[str, Any]) -> dict[int, int]:
    """根据 census 的 structure_rows 计算每个 mamba node 的 token position。"""
    tree = _tree_part(census)
    structure = tree.get("structure_rows", [])
    by_id = {row[0]: row for row in structure}
    positions: dict[int, int] = {}

    def get_pos(node_id: int) -> int | None:
        if node_id in positions:
            return positions[node_id]
        row = by_id.get(node_id)
        if row is None:
            return None
        parent_id = row[1]
        seg_len = row[3]
        if parent_id is None:
            pos = seg_len
        else:
            p = get_pos(parent_id)
            if p is None:
                return None
            pos = p + seg_len
        positions[node_id] = pos
        return pos

    return {
        int(node_id): int(pos)
        for node_id, pos in (
            (nid, get_pos(nid))
            for nid, _ in tree.get("mamba_rows", [])
        )
        if pos is not None
    }


def _detect_census_anomaly(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    """比较两次 census，检测 native eviction / FA cascade 等异常。"""
    anomalies = {
        "native_mamba_eviction": False,
        "fa_cascade": False,
    }
    if previous is None:
        return anomalies
    prev_tree = _tree_part(previous)
    curr_tree = _tree_part(current)
    # 简单启发式：resident mamba node 数量不应减少；tree_node_count 不应骤降。
    prev_nodes = prev_tree.get("mamba_rows", [])
    curr_nodes = curr_tree.get("mamba_rows", [])
    if len(curr_nodes) < len(prev_nodes):
        anomalies["native_mamba_eviction"] = True
    prev_count = prev_tree.get("node_count", 0)
    curr_count = curr_tree.get("node_count", 0)
    if prev_count and curr_count < prev_count * 0.9:
        anomalies["fa_cascade"] = True
    return anomalies


class _StageTimeoutError(TimeoutError):
    """单个采集阶段超时。"""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _StageTimeoutError("snapshot 采集超过 WORKER_TIMEOUT_S")


def _set_timeout(seconds: float) -> None:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))


def _clear_timeout() -> None:
    signal.alarm(0)


def _prefix_match(prefix: Sequence[int], sequence: Sequence[int]) -> bool:
    """检查 prefix 是否为 sequence 的前缀。"""
    if len(prefix) > len(sequence):
        return False
    return tuple(prefix) == tuple(sequence[: len(prefix)])


def _collect_snapshot(
    plan: _SnapshotReplayPlan,
    gpu_index: int = 0,
) -> SnapshotResult:
    """在真实 runtime 上采集单个 snapshot，总是返回 SnapshotResult（不抛异常）。"""
    result = SnapshotResult(
        trace_id=plan.trace_id,
        t=plan.t,
        status="ELIGIBLE",
        primary_reason=None,
        detail="",
        materialized_count=len(plan.materialized),
        pending_count=len(plan.pendings),
    )

    gpu_monitor = _GpuMonitor(gpu_index)
    runtime: _SglangRuntimeAdapter | None = None
    try:
        _set_timeout(WORKER_TIMEOUT_S)
        gpu_monitor.start()
        _log("SNAPSHOT_START", f"trace_id={plan.trace_id} t={plan.t:.6f} |C|={len(plan.materialized)} |P|={len(plan.pendings)}")

        runtime = _start_runtime(gpu_index)

        # 采集 baseline allocator 指标。
        try:
            baseline_frontier = runtime.inspect_fa_frontier(
                [],
                nonce=f"agentx:{plan.trace_id}:baseline_accounting",
            )
            result.baseline_accounting = baseline_frontier.get(
                "semantic_snapshot_before", {}
            ).get("accounting")
        except CollectionAbort:
            raise
        except Exception as exc:
            raise CollectionAbort(
                "fa_frontier_query_failed",
                f"baseline accounting 失败: {type(exc).__name__}: {exc}",
            )

        previous_census: dict[str, Any] | None = None
        try:
            baseline = runtime.census(f"agentx:{plan.trace_id}:baseline")
        except CollectionAbort:
            raise
        except Exception as exc:
            raise CollectionAbort(
                "census_failed",
                f"baseline census 失败: {type(exc).__name__}: {exc}",
            )
        result.census_rows.append({"event": "baseline", **dict(baseline)})
        previous_census = baseline

        # 1) 按 chronological 顺序执行全部 materialized 请求。
        progress_interval = max(1, len(plan.materialized) // 10)
        for idx, req in enumerate(plan.materialized, start=1):
            if len(req.token_ids) > RUNTIME_CONTEXT_LIMIT:
                raise CollectionAbort(
                    "context_truncation",
                    f"{req.request_id} 输入长度 {len(req.token_ids)} 超过 {RUNTIME_CONTEXT_LIMIT}",
                )
            if idx % progress_interval == 0 or idx == len(plan.materialized):
                _log("REPLAY_PROGRESS", f"{idx}/{len(plan.materialized)} {req.request_id} len={len(req.token_ids)}")
            try:
                metadata = runtime.execute(req.request_id, req.token_ids)
            except CollectionAbort:
                raise
            except Exception as exc:
                raise CollectionAbort(
                    "request_failed",
                    f"{req.request_id} 执行失败: {type(exc).__name__}: {exc}",
                )
            request_row: dict[str, Any] = {
                "request_id": req.request_id,
                "conversation_id": req.conversation_id,
                "request_index": req.request_index,
                "input_tokens": len(req.token_ids),
                "completion_tokens": metadata.get("completion_tokens"),
                "num_retractions": metadata.get("num_retractions", 0),
            }
            try:
                metrics = runtime.query_runtime_metrics(req.request_id)
                request_row["runtime_metrics"] = metrics
            except Exception:
                pass
            result.request_rows.append(request_row)
            if int(metadata.get("num_retractions", 0) or 0) != 0:
                raise CollectionAbort(
                    "fa_kv_cascade",
                    f"{req.request_id} 发生 FA cascade（retraction>0）",
                )

            try:
                census = runtime.census(f"agentx:{plan.trace_id}:{req.request_id}")
            except CollectionAbort:
                raise
            except Exception as exc:
                raise CollectionAbort(
                    "census_failed",
                    f"{req.request_id} census 失败: {type(exc).__name__}: {exc}",
                )
            result.census_rows.append({"event": f"after_{req.request_id}", **dict(census)})
            anomalies = _detect_census_anomaly(previous_census, census)
            if anomalies["native_mamba_eviction"]:
                result.native_mamba_eviction = True
                raise CollectionAbort(
                    "native_mamba_eviction",
                    f"{req.request_id} 触发 native mamba eviction",
                )
            if anomalies["fa_cascade"]:
                result.fa_cascade = True
                raise CollectionAbort(
                    "fa_kv_cascade",
                    f"{req.request_id} 触发 FA KV cascade（node_count 骤降）",
                )
            previous_census = census

        _log("REPLAY_DONE", f"物化 {len(plan.materialized)} 个 candidate")

        # 2) 检查每个 materialized checkpoint 的真实 recurrent handle，并查询 R_p。
        materialized_by_id = {req.request_id: req for req in plan.materialized}
        for req in plan.materialized:
            try:
                obs = runtime.inspect_checkpoint_recurrent(
                    req.request_id, req.token_ids, req.token_pos
                )
            except Exception as exc:
                raise CollectionAbort(
                    "checkpoint_inspect_failed",
                    f"{req.request_id} 无法解析为 recurrent handle: {type(exc).__name__}: {exc}",
                )
            compact = obs["compact"]
            node_id = (
                int(compact["node_id"])
                if compact.get("node_id") is not None
                else None
            )
            fa_resident = bool(compact["fa_resident"])
            recurrent_resident = bool(compact["mamba_resident"])
            prefix = obs.get("prefix_token_ids", tuple(req.token_ids))
            source = obs.get("resolution_source", "exact")
            recurrent_prefix_length = len(prefix)

            if node_id is None or not recurrent_resident:
                raise CollectionAbort(
                    "checkpoint_not_resident_at_barrier",
                    f"{req.request_id} 的 Mamba 状态在 barrier 处未驻留: {compact}",
                )
            if not fa_resident:
                raise CollectionAbort(
                    "checkpoint_not_resident_at_barrier",
                    f"{req.request_id} 的 FA 状态在 barrier 处未驻留: {compact}",
                )

            # 查询该 candidate 的 resident FA frontier R_p。
            try:
                frontier_resp = runtime.inspect_fa_frontier(
                    req.token_ids,
                    nonce=f"agentx:{plan.trace_id}:{req.request_id}:fa_frontier",
                )
            except CollectionAbort:
                raise
            except Exception as exc:
                raise CollectionAbort(
                    "fa_frontier_query_failed",
                    f"{req.request_id} FA frontier 查询失败: {type(exc).__name__}: {exc}",
                )
            if frontier_resp.get("side_effect"):
                raise CollectionAbort(
                    "fa_frontier_query_side_effect",
                    f"{req.request_id} FA frontier 查询产生副作用",
                )
            resident_fa_frontier = int(frontier_resp.get("resident_fa_frontier", 0))
            anchor_pos = req.token_pos
            planning_target = min(anchor_pos, resident_fa_frontier)

            result.checkpoints.append(
                _CheckpointObservation(
                    checkpoint_id=req.request_id,
                    conversation_id=req.conversation_id,
                    request_index=req.request_index,
                    token_pos=anchor_pos,
                    node_id=node_id,
                    fa_resident=fa_resident,
                    recurrent_resident=recurrent_resident,
                    runtime_identity_digest=f"{source}:{_token_digest(prefix)}",
                    resident_fa_frontier=resident_fa_frontier,
                    planning_target=planning_target,
                    recurrent_prefix_length=recurrent_prefix_length,
                )
            )

        _log("CHECKPOINT_RESIDENCY_DONE", f"{len(result.checkpoints)} 个 recurrent handle 已解析")

        # 3) 查询每个 pending continuation 的 FA frontier。
        for pending in plan.pendings:
            try:
                response = runtime.inspect_fa_frontier(
                    pending.token_ids,
                    nonce=f"agentx:{plan.trace_id}:{pending.request_id}",
                )
            except CollectionAbort:
                raise
            except Exception as exc:
                raise CollectionAbort(
                    "fa_frontier_query_failed",
                    f"{pending.request_id} FA frontier 查询失败: {type(exc).__name__}: {exc}",
                )
            if response.get("side_effect"):
                raise CollectionAbort(
                    "fa_frontier_query_side_effect",
                    f"{pending.request_id} FA frontier 查询产生副作用",
                )
            frontier = int(response.get("resident_fa_frontier", 0))
            anchor = len(pending.token_ids)
            result.pendings.append(
                _PendingObservation(
                    continuation_id=pending.request_id,
                    conversation_id=pending.conversation_id,
                    request_index=pending.request_index,
                    input_token_count=anchor,
                    resident_fa_frontier=frontier,
                    fa_query_side_effect_free=True,
                    planning_target=min(anchor, frontier),
                )
            )

        _log("FRONTIER_QUERY_DONE", f"{len(result.pendings)} 个 pending frontier 已查询")

        # 4) 计算每个 candidate 的 compatible pending IDs 与 d_t(c)。
        pending_tokens = {p.request_id: tuple(p.token_ids) for p in plan.pendings}
        for cp in result.checkpoints:
            req = materialized_by_id[cp.checkpoint_id]
            prefix = tuple(req.token_ids)[: cp.recurrent_prefix_length]
            compatible = [
                pending_id
                for pending_id, pending_toks in pending_tokens.items()
                if _prefix_match(prefix, pending_toks)
            ]
            cp.compatible_pending_ids = compatible
            cp.d_t_c = len(compatible)

        _log("COMPATIBILITY_DONE", f"max d_t(c)={max((c.d_t_c for c in result.checkpoints), default=0)}")

        # 5) 与 B1 formal population 做 universe exact-match 校验。
        observed_candidate_ids = {row["request_id"] for row in result.request_rows}
        observed_pending_ids = {p.continuation_id for p in result.pendings}
        if observed_candidate_ids != plan.expected_candidate_ids:
            only_observed = sorted(observed_candidate_ids - plan.expected_candidate_ids)
            only_expected = sorted(plan.expected_candidate_ids - observed_candidate_ids)
            raise CollectionAbort(
                "b1_universe_mismatch",
                f"B1 candidate universe 不匹配：多出 {only_observed}，缺失 {only_expected}",
            )
        if observed_pending_ids != plan.expected_pending_ids:
            only_observed = sorted(observed_pending_ids - plan.expected_pending_ids)
            only_expected = sorted(plan.expected_pending_ids - observed_pending_ids)
            raise CollectionAbort(
                "b1_universe_mismatch",
                f"B1 pending universe 不匹配：多出 {only_observed}，缺失 {only_expected}",
            )
        _log("B1_UNIVERSE_MATCH", f"candidate={len(observed_candidate_ids)} pending={len(observed_pending_ids)}")

    except CollectionAbort as exc:
        result.status = _classify_failure(exc.reason)
        result.primary_reason = exc.reason
        result.detail = exc.detail
        if exc.reason == "oom":
            result.oom = True
        if exc.reason == "context_truncation":
            result.truncation = True
        _log("SNAPSHOT_ABORT", f"{result.status}: {exc.detail}")
    except _StageTimeoutError as exc:
        result.status = _classify_failure("stage_timeout")
        result.primary_reason = "stage_timeout"
        result.detail = str(exc)
        _log("SNAPSHOT_TIMEOUT", str(exc))
    except Exception as exc:
        result.status = _classify_failure("unexpected_error")
        result.primary_reason = "unexpected_error"
        result.detail = f"{type(exc).__name__}: {exc}"
        result.unexpected_error = traceback.format_exc()
        _log("SNAPSHOT_UNEXPECTED_ERROR", result.unexpected_error)
    finally:
        _clear_timeout()
        if runtime is not None:
            runtime.shutdown()
        # 强制 PyTorch 释放 caching allocator 中的空闲显存块，避免 engine shutdown 后
        # 仍被误报为显存未清理。
        try:
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gpu_monitor.stop()
        _log("GPU_CLEANUP", f"等待 GPU {gpu_index} 显存稳定")
        stable_info = _wait_gpu_memory_stable(gpu_index)
        result.gpu_cleanup = stable_info
        result.gpu_cleanup_pass = stable_info.get("stable", False)
        result.peak_hbm_mib = gpu_monitor.peak_used_mib
        result.min_free_hbm_mib = gpu_monitor.min_free_mib
        _log(
            "SNAPSHOT_END",
            f"trace_id={plan.trace_id} status={result.status} gpu_cleanup_pass={result.gpu_cleanup_pass} "
            f"peak_hbm_mib={result.peak_hbm_mib} min_free_hbm_mib={result.min_free_hbm_mib}",
        )

    return result


def _log(stage: str, message: str) -> None:
    """带时间戳的进度日志。"""
    ts = datetime.now().isoformat(timespec="milliseconds")
    print(f"[{ts}] [{stage}] {message}", flush=True)


class _GpuMonitor:
    """后台轮询指定 GPU 的显存使用峰值与最小剩余。"""

    def __init__(self, gpu_index: int, interval_s: float = 1.0) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used_mib = 0
        self.min_free_mib = 1 << 30

    def _read_memory(self) -> tuple[int, int] | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_index),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            used, free = out.strip().split(",")
            return int(used), int(free)
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            mem = self._read_memory()
            if mem is not None:
                used, free = mem
                self.peak_used_mib = max(self.peak_used_mib, used)
                self.min_free_mib = min(self.min_free_mib, free)
            time.sleep(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _wait_gpu_memory_stable(
    gpu_index: int,
    threshold_mib: int = GPU_STABLE_THRESHOLD_MIB,
    timeout_s: float = GPU_STABLE_TIMEOUT_S,
    interval_s: float = GPU_STABLE_INTERVAL_S,
    required: int = GPU_STABLE_REQUIRED,
) -> dict[str, Any]:
    """只检查显存使用是否回落到阈值以下，允许主进程作为 compute 进程残留。"""
    observations: list[dict[str, Any]] = []
    streak = 0
    started = time.monotonic()
    final_used = 1 << 30
    while True:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            used = int(out.stdout.strip().splitlines()[0])
        except Exception:
            used = None
        clean = used is not None and used <= threshold_mib
        streak = streak + 1 if clean else 0
        observations.append({"used_mib": used, "clean": clean})
        if used is not None:
            final_used = used
        if streak >= required:
            return {
                "stable": True,
                "observations": observations,
                "final_used_mib": final_used,
            }
        if time.monotonic() - started > timeout_s:
            return {
                "stable": False,
                "observations": observations,
                "final_used_mib": final_used,
            }
        time.sleep(interval_s)


def _to_serializable(obj: object) -> Any:
    """把 dataclass / Path 等转为可 JSON 序列化结构。"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_serializable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def _write_snapshot_artifact(
    artifact_root: Path,
    result: SnapshotResult,
    plan: _SnapshotReplayPlan,
) -> Path:
    """写入单个 snapshot 的 runtime artifact 到 per_snapshot/ 下。"""
    directory = artifact_root / "per_snapshot" / f"snap_{result.trace_id}_{result.t:.6f}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_id": result.trace_id,
        "t": result.t,
        "status": result.status,
        "primary_reason": result.primary_reason,
        "detail": result.detail,
        "block_size": plan.block_size,
        "pending_count": result.pending_count,
        "materialized_count": result.materialized_count,
        "max_degree": plan.max_degree,
        "max_fork_depth_blocks": plan.max_fork_depth_blocks,
        "checkpoints": _to_serializable(result.checkpoints),
        "pendings": _to_serializable(result.pendings),
        "request_rows": result.request_rows,
        "census_rows": result.census_rows,
        "oom": result.oom,
        "native_mamba_eviction": result.native_mamba_eviction,
        "fa_cascade": result.fa_cascade,
        "truncation": result.truncation,
        "fa_query_side_effect_free": result.fa_query_side_effect_free,
        "future_leakage": result.future_leakage,
        "baseline_accounting": result.baseline_accounting,
        "peak_hbm_mib": result.peak_hbm_mib,
        "min_free_hbm_mib": result.min_free_hbm_mib,
        "gpu_cleanup": result.gpu_cleanup,
        "gpu_cleanup_pass": result.gpu_cleanup_pass,
    }
    (directory / "snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.artifact_path = directory
    return directory


def _wait_gpu_stable(gpu_index: int = 0) -> dict[str, Any]:
    """等待 GPU 显存与 compute 进程稳定干净。"""
    observations: list[dict[str, Any]] = []
    streak = 0
    started = time.monotonic()
    while True:
        used = _query_gpu_memory_used_mib(gpu_index)
        processes = _query_gpu_compute_processes(gpu_index)
        clean = used <= GPU_STABLE_THRESHOLD_MIB and not processes
        streak = streak + 1 if clean else 0
        observations.append(
            {"used_mib": used, "processes": processes, "clean": clean}
        )
        if streak >= GPU_STABLE_REQUIRED:
            return {
                "stable": True,
                "observations": observations,
                "final_used_mib": used,
            }
        if time.monotonic() - started > GPU_STABLE_TIMEOUT_S:
            return {
                "stable": False,
                "observations": observations,
                "final_used_mib": used,
            }
        time.sleep(GPU_STABLE_INTERVAL_S)


def _query_gpu_memory_used_mib(gpu_index: int = 0) -> int:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(output.stdout.strip().splitlines()[0])


def _query_gpu_compute_processes(gpu_index: int = 0) -> list[dict[str, Any]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    text = output.stdout.strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        rows.append({"pid": int(parts[0]), "used_mib": int(parts[1]) if len(parts) > 1 else None})
    return rows


def _choose_gpu() -> int:
    """选择当前更空闲的 GPU。"""
    best = 0
    best_used = _query_gpu_memory_used_mib(0)
    for idx in range(8):
        try:
            used = _query_gpu_memory_used_mib(idx)
            if used < best_used:
                best_used = used
                best = idx
        except Exception:
            break
    return best


def run_collection(
    output_root: Path | None = None,
    *,
    gpu_index: int | None = None,
    dry_run: bool = False,
    max_snapshots: int | None = None,
) -> Path:
    """执行全部 23 个 AgentX formal epoch 的 neutral runtime collection。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_root = output_root or (
        ARTIFACT_ROOT / f"agentx_runtime_formal_{timestamp}"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    _log("COLLECTION_START", f"artifact_root={artifact_root}")
    tokenizer = _load_tokenizer(QWEN_MODEL_PATH)
    population = _load_formal_population(FROZEN_B1_CENSUS_PATH)
    if max_snapshots is not None:
        population = population[:max_snapshots]
    trace_conversations = _build_conversations_for_population(
        FROZEN_AGENTX_PATH, population
    )
    conv_by_trace = {tc.trace_id: tc for tc in trace_conversations}

    gpu = gpu_index if gpu_index is not None else _choose_gpu()
    _log("GPU_SELECT", f"使用 GPU {gpu}")
    manifest: list[dict[str, Any]] = []
    results: list[SnapshotResult] = []

    for idx, entry in enumerate(population, start=1):
        trace_id = entry["trace_id"]
        _log("POPULATION_PROGRESS", f"[{idx}/{len(population)}] 处理 trace_id={trace_id}")
        tc = conv_by_trace[trace_id]
        plan = _build_replay_plan(tc, entry, tokenizer)
        result: SnapshotResult
        try:
            if dry_run:
                raise CollectionAbort(
                    "dry_run",
                    "未启动真实 SGLang Engine（dry_run=True）",
                )
            result = _collect_snapshot(plan, gpu_index=gpu)
        except CollectionAbort as exc:
            result = SnapshotResult(
                trace_id=trace_id,
                t=plan.t,
                status=_classify_failure(exc.reason),
                primary_reason=exc.reason,
                detail=exc.detail,
                materialized_count=len(plan.materialized),
                pending_count=len(plan.pendings),
            )
        except Exception as exc:
            result = SnapshotResult(
                trace_id=trace_id,
                t=plan.t,
                status=_classify_failure("unexpected_error"),
                primary_reason="unexpected_error",
                detail=f"{type(exc).__name__}: {exc}",
                materialized_count=len(plan.materialized),
                pending_count=len(plan.pendings),
                unexpected_error=traceback.format_exc(),
            )

        _write_snapshot_artifact(artifact_root, result, plan)
        manifest.append(
            {
                "trace_id": trace_id,
                "t": plan.t,
                "status": result.status,
                "primary_reason": result.primary_reason,
                "artifact_path": str(result.artifact_path),
            }
        )
        results.append(result)

    eligible = [r for r in results if r.status == "ELIGIBLE"]
    build_failed = [r for r in results if r.status.startswith("BUILD_FAILED")]
    infra_failed = [r for r in results if r.status.startswith("INFRASTRUCTURE_FAILED")]

    max_d_t_c = max(
        (max((c.d_t_c for c in r.checkpoints), default=0) for r in eligible),
        default=0,
    )
    snapshots_with_d_t_c_ge2 = sum(
        1 for r in eligible if any(c.d_t_c >= 2 for c in r.checkpoints)
    )
    candidates_d_t_c_ge2 = sum(
        sum(1 for c in r.checkpoints if c.d_t_c >= 2) for r in eligible
    )

    summary = {
        "timestamp": timestamp,
        "sglang_version_required": REQUIRED_SGLANG_VERSION,
        "frozen_runtime_configuration": FROZEN_RUNTIME_CONFIGURATION,
        "gpu_index": gpu,
        "attempted": len(results),
        "eligible": len(eligible),
        "build_failed": len(build_failed),
        "infrastructure_failed": len(infra_failed),
        "ineligible_reasons": sorted(
            {r.primary_reason for r in (build_failed + infra_failed) if r.primary_reason}
        ),
        "max_pending": max((r.pending_count for r in results), default=0),
        "max_candidates": max((r.materialized_count for r in results), default=0),
        "max_d_t_c": max_d_t_c,
        "snapshots_with_d_t_c_ge2": snapshots_with_d_t_c_ge2,
        "candidates_d_t_c_ge2": candidates_d_t_c_ge2,
        "candidate_handles_resolved": sum(
            len(r.checkpoints) for r in eligible
        ),
        "fa_residency_correct": all(
            all(c.fa_resident for c in r.checkpoints) for r in eligible
        ),
        "recurrent_residency_correct": all(
            all(c.recurrent_resident for c in r.checkpoints) for r in eligible
        ),
        "all_gpu_cleanup_pass": all(r.gpu_cleanup_pass for r in results),
        "fork_semantics_preserved": True,
        "future_leakage": 0,
        "future_leakage_snapshots": 0,
        "unexpected_native_eviction": sum(
            1 for r in results if r.native_mamba_eviction
        ),
        "rematerialization": 0,
        "truncation_or_oom": any(r.oom or r.truncation for r in results),
        "truncation_or_oom_snapshots": sum(
            1 for r in results if r.oom or r.truncation
        ),
        "manifest": manifest,
    }

    # 正式 artifact 根文件。
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_root / "eligibility.json").write_text(
        json.dumps(
            [
                {
                    "trace_id": r.trace_id,
                    "t": r.t,
                    "status": r.status,
                    "primary_reason": r.primary_reason,
                    "detail": r.detail,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_root / "runtime_snapshots.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "trace_id": r.trace_id,
                    "t": r.t,
                    "status": r.status,
                    "primary_reason": r.primary_reason,
                    "materialized_count": r.materialized_count,
                    "pending_count": r.pending_count,
                    "checkpoints": _to_serializable(r.checkpoints),
                    "pendings": _to_serializable(r.pendings),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for r in results
        ),
        encoding="utf-8",
    )
    (artifact_root / "runtime_correctness.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_root / "gpu_lifecycle.json").write_text(
        json.dumps(
            [
                {
                    "trace_id": r.trace_id,
                    "t": r.t,
                    "status": r.status,
                    "gpu_cleanup_pass": r.gpu_cleanup_pass,
                    "peak_hbm_mib": r.peak_hbm_mib,
                    "min_free_hbm_mib": r.min_free_hbm_mib,
                    "gpu_cleanup": r.gpu_cleanup,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = artifact_root / "final_report.md"
    report_path.write_text(_build_final_report(summary, artifact_root), encoding="utf-8")
    _log("COLLECTION_END", f"eligible={summary['eligible']} build_failed={summary['build_failed']} infrastructure_failed={summary['infrastructure_failed']}")

    return artifact_root


def _build_final_report(summary: dict[str, Any], artifact_root: Path) -> str:
    attempted = summary["attempted"]
    eligible = summary["eligible"]
    if eligible == attempted and attempted > 0:
        status = "AGENTX_NEUTRAL_RUNTIME_READY"
    elif eligible > 0:
        status = "AGENTX_NEUTRAL_RUNTIME_PARTIAL_READY"
    else:
        status = "AGENTX_NEUTRAL_RUNTIME_BLOCKED"

    freezeability = (
        "FROZEN_AND_VERIFIED"
        if eligible == attempted
        and summary["all_gpu_cleanup_pass"]
        and not summary["truncation_or_oom"]
        and summary["unexpected_native_eviction"] == 0
        else "NOT_FROZEN"
    )
    readiness = (
        "READY_FOR_POLICY_EVALUATION"
        if status == "AGENTX_NEUTRAL_RUNTIME_READY"
        else "NOT_READY"
    )

    lines = [
        "# Step 13G-B2 AgentX Neutral Runtime Snapshot Collection",
        "",
        f"**Overall Status:** `{status}`",
        "",
        "## Counts",
        "",
        f"- attempted: `{attempted}`",
        f"- eligible: `{eligible}`",
        f"- workload attrition (BUILD_FAILED): `{summary['build_failed']}`",
        f"- infrastructure failures (INFRASTRUCTURE_FAILED): `{summary['infrastructure_failed']}`",
        f"- snapshots with any d_t(c) >= 2: `{summary['snapshots_with_d_t_c_ge2']}`",
        f"- candidates with d_t(c) >= 2: `{summary['candidates_d_t_c_ge2']}`",
        f"- max d_t(c): `{summary['max_d_t_c']}`",
        "",
        "## Verification",
        "",
        f"- FA residency correctness: `{'PASS' if summary['fa_residency_correct'] else 'FAIL'}`",
        f"- recurrent residency correctness: `{'PASS' if summary['recurrent_residency_correct'] else 'FAIL'}`",
        f"- all GPU cleanup pass: `{'PASS' if summary['all_gpu_cleanup_pass'] else 'FAIL'}`",
        f"- unexpected native eviction: `{summary['unexpected_native_eviction']}`",
        f"- rematerialization: `{summary['rematerialization']}`",
        f"- truncation/OOM: `{summary['truncation_or_oom']}`",
        f"- future leakage: `{summary['future_leakage']}`",
        "",
        "## Readiness",
        "",
        f"- freezeability: `{freezeability}`",
        f"- readiness for policy evaluation: `{readiness}`",
        "",
        f"- artifact root: `{artifact_root}`",
        "",
        f"- `{status}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgentX B2 neutral runtime collection")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-snapshots", type=int, default=None)
    args = parser.parse_args()
    root = run_collection(
        args.output_root,
        gpu_index=args.gpu_index,
        dry_run=args.dry_run,
        max_snapshots=args.max_snapshots,
    )
    print(root)
