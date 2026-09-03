#!/usr/bin/env python3
"""为连续循环状态驱逐提供无修复语义的边界追踪与离线诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from flowstate.adapters.sglang import RuntimeCheckpointHandle


TRACE_BOUNDARIES = ("S0", "S1", "S2", "S3", "S4")
LRU_SECOND_EVICTION_ORDER = (
    "OPENHANDS_BARRIER_A_TURN_001",
    "OPENHANDS_BARRIER_B_TURN_001",
    "OPENHANDS_BARRIER_C_TURN_001",
    "OPENHANDS_BARRIER_C_TURN_002",
)
LRU_TRACKED_CHECKPOINTS = (
    "OPENHANDS_BARRIER_A_TURN_001",
    "OPENHANDS_BARRIER_B_TURN_001",
    "OPENHANDS_BARRIER_C_TURN_001",
    "OPENHANDS_BARRIER_C_TURN_002",
    "OPENHANDS_BARRIER_D_TURN_001",
    "OPENHANDS_BARRIER_D_TURN_002",
)
BOUNDARY_OPERATIONS = {
    "S0": "开始目标驱逐前",
    "S1": "精确句柄与目标查找后",
    "S2": "循环状态驱逐原语完成后",
    "S3": "驱逐后证明与检查完成后",
    "S4": "进入下一目标或结束序列前",
}


@dataclass(frozen=True)
class CheckpointTraceState:
    """保存一个检查点在单个追踪边界上的循环状态事实。"""

    checkpoint_id: str
    recurrent_present: bool
    host_present: bool
    in_mamba_lru: bool
    node_id: int
    token_position: int
    mamba_slots: tuple[int, ...]

    @property
    def mamba_status(self) -> str:
        """以互斥状态描述设备与主机副本。"""
        if self.recurrent_present and self.host_present:
            return "设备与主机均驻留"
        if self.recurrent_present:
            return "设备驻留"
        if self.host_present:
            return "仅主机驻留"
        return "完全缺失"

    def row(self) -> dict[str, object]:
        """转换为可直接写入诊断产物的字典。"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "recurrent_present": self.recurrent_present,
            "host_present": self.host_present,
            "in_mamba_lru": self.in_mamba_lru,
            "node_id": self.node_id,
            "token_position": self.token_position,
            "mamba_slots": list(self.mamba_slots),
            "mamba_status": self.mamba_status,
        }


@dataclass(frozen=True)
class AllocatorTraceState:
    """保存追踪边界上的 Mamba 分配器与可驱逐性计数。"""

    available_slots: int
    evictable_slots: int
    protected_slots: int

    def row(self) -> dict[str, int]:
        """转换为稳定的产物表示。"""
        return {
            "available_slots": self.available_slots,
            "evictable_slots": self.evictable_slots,
            "protected_slots": self.protected_slots,
        }


@dataclass(frozen=True)
class TraceSnapshot:
    """保存一次只读全局循环状态快照。"""

    checkpoints: Mapping[str, CheckpointTraceState]
    allocator: AllocatorTraceState


@dataclass(frozen=True)
class RematerializationEvent:
    """描述一个检查点首次由缺失变为设备驻留的事件。"""

    checkpoint_id: str
    first_rematerialization_target: str
    boundary: str
    triggering_operation: str
    previous_state: CheckpointTraceState
    new_state: CheckpointTraceState

    def row(self) -> dict[str, object]:
        """转换为诊断汇总使用的字典。"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "first_rematerialization_target": (
                self.first_rematerialization_target
            ),
            "boundary": self.boundary,
            "triggering_operation": self.triggering_operation,
            "previous_state": self.previous_state.row(),
            "new_state": self.new_state.row(),
        }


SnapshotProvider = Callable[[], TraceSnapshot]


class SequentialTraceRecorder:
    """按确定顺序采集边界快照，不参与任何选择或驱逐决策。"""

    def __init__(
        self,
        tracked_checkpoint_ids: Sequence[str],
        snapshot_provider: SnapshotProvider,
    ) -> None:
        tracked = tuple(str(value) for value in tracked_checkpoint_ids)
        if not tracked or len(set(tracked)) != len(tracked):
            raise ValueError("追踪检查点必须非空且互不重复")
        self._tracked_checkpoint_ids = tracked
        self._snapshot_provider = snapshot_provider
        self._rows: list[dict[str, object]] = []

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        """返回不可变顺序视图。"""
        return tuple(self._rows)

    def record(
        self,
        *,
        target_checkpoint_id: str,
        boundary: str,
        operation: str | None = None,
    ) -> dict[str, object]:
        """在当前位置调用只读快照提供器并追加一行。"""
        if boundary not in TRACE_BOUNDARIES:
            raise ValueError(f"未知追踪边界：{boundary}")
        snapshot = self._snapshot_provider()
        actual_ids = set(snapshot.checkpoints)
        expected_ids = set(self._tracked_checkpoint_ids)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            unexpected = sorted(actual_ids - expected_ids)
            raise RuntimeError(
                "追踪快照检查点集合不完整："
                f"缺失 {missing}，额外 {unexpected}"
            )
        row = {
            "sequence": len(self._rows),
            "target_checkpoint_id": target_checkpoint_id,
            "boundary": boundary,
            "runtime_operation": operation or BOUNDARY_OPERATIONS[boundary],
            "checkpoints": {
                checkpoint_id: snapshot.checkpoints[checkpoint_id].row()
                for checkpoint_id in self._tracked_checkpoint_ids
            },
            "mamba_allocator": snapshot.allocator.row(),
        }
        self._rows.append(row)
        return row


def checkpoint_state_from_path(
    checkpoint_id: str,
    path: Mapping[str, object],
) -> CheckpointTraceState:
    """把现有精确路径快照转换为统一追踪状态。"""
    slots = tuple(int(value) for value in path.get("target_mamba_slots", ()))
    return CheckpointTraceState(
        checkpoint_id=str(checkpoint_id),
        recurrent_present=bool(path["target_mamba_present"]),
        host_present=bool(path["target_mamba_host_present"]),
        in_mamba_lru=bool(path["target_mamba_in_lru"]),
        node_id=int(path["node_id"]),
        token_position=int(path["prefix_tokens"]),
        mamba_slots=slots,
    )


def allocator_state_from_accounting(
    accounting: Mapping[str, object],
) -> AllocatorTraceState:
    """把现有分配器快照转换为统一追踪状态。"""
    return AllocatorTraceState(
        available_slots=int(accounting["mamba_available"]),
        evictable_slots=int(accounting["mamba_evictable"]),
        protected_slots=int(accounting["mamba_protected"]),
    )


def handle_descriptor(handle: RuntimeCheckpointHandle) -> dict[str, object]:
    """构造诊断传输层使用的完整只读定位描述。"""
    return {
        "checkpoint_id": handle.checkpoint_id,
        "token_ids": list(handle.token_ids),
        "extra_key": handle.extra_key,
        "expected_node_id": handle.expected_node_id,
        "expected_prefix_sha256": handle.expected_prefix_digest,
    }


def validate_sequential_trace(
    rows: Sequence[Mapping[str, object]],
    eviction_order: Sequence[str],
    tracked_checkpoint_ids: Sequence[str],
) -> None:
    """验证每个目标均按 S0 至 S4 完整且连续出现。"""
    expected = [
        (target, boundary)
        for target in eviction_order
        for boundary in TRACE_BOUNDARIES
    ]
    actual = [
        (str(row["target_checkpoint_id"]), str(row["boundary"]))
        for row in rows
    ]
    if actual != expected:
        raise RuntimeError(
            "连续追踪顺序不完整："
            f"实际 {actual}，预期 {expected}"
        )
    expected_ids = set(str(value) for value in tracked_checkpoint_ids)
    for index, row in enumerate(rows):
        if int(row["sequence"]) != index:
            raise RuntimeError("连续追踪序号不连续")
        checkpoints = row.get("checkpoints")
        if not isinstance(checkpoints, Mapping):
            raise RuntimeError("连续追踪行缺少检查点快照")
        if set(checkpoints) != expected_ids:
            raise RuntimeError("连续追踪行的检查点集合不一致")
        if "mamba_allocator" not in row:
            raise RuntimeError("连续追踪行缺少 Mamba 分配器快照")


def find_first_rematerializations(
    rows: Sequence[Mapping[str, object]],
    checkpoint_ids: Sequence[str],
) -> dict[str, RematerializationEvent | None]:
    """查找每个检查点在首次缺失后的第一次重新设备驻留。"""
    result: dict[str, RematerializationEvent | None] = {}
    for checkpoint_id in checkpoint_ids:
        previous: CheckpointTraceState | None = None
        absent_seen = False
        event = None
        for row in rows:
            checkpoints = row.get("checkpoints")
            if not isinstance(checkpoints, Mapping):
                raise RuntimeError("追踪行缺少检查点状态")
            raw_state = checkpoints.get(checkpoint_id)
            if not isinstance(raw_state, Mapping):
                raise RuntimeError(f"追踪行缺少 {checkpoint_id} 状态")
            state = CheckpointTraceState(
                checkpoint_id=checkpoint_id,
                recurrent_present=bool(raw_state["recurrent_present"]),
                host_present=bool(raw_state["host_present"]),
                in_mamba_lru=bool(raw_state["in_mamba_lru"]),
                node_id=int(raw_state["node_id"]),
                token_position=int(raw_state["token_position"]),
                mamba_slots=tuple(
                    int(value) for value in raw_state.get("mamba_slots", ())
                ),
            )
            if previous is not None and absent_seen:
                if not previous.recurrent_present and state.recurrent_present:
                    event = RematerializationEvent(
                        checkpoint_id=checkpoint_id,
                        first_rematerialization_target=str(
                            row["target_checkpoint_id"]
                        ),
                        boundary=str(row["boundary"]),
                        triggering_operation=str(row["runtime_operation"]),
                        previous_state=previous,
                        new_state=state,
                    )
                    break
            if not state.recurrent_present:
                absent_seen = True
            previous = state
        result[checkpoint_id] = event
    return result


def instrument_evict_mamba_only(
    adapter: object,
    handle: RuntimeCheckpointHandle,
    recorder: SequentialTraceRecorder,
) -> None:
    """包裹现有适配器内部边界，不改变其真实调用顺序或原语。"""
    original_find = adapter._find_exact_node
    original_primitive = adapter._evict_mamba_component_only
    find_count = 0

    def traced_find(token_ids, extra_key):
        nonlocal find_count
        result = original_find(token_ids, extra_key)
        find_count += 1
        if find_count == 1:
            recorder.record(
                target_checkpoint_id=handle.checkpoint_id,
                boundary="S1",
            )
        return result

    def traced_primitive(node):
        original_primitive(node)
        recorder.record(
            target_checkpoint_id=handle.checkpoint_id,
            boundary="S2",
        )

    recorder.record(
        target_checkpoint_id=handle.checkpoint_id,
        boundary="S0",
    )
    adapter._find_exact_node = traced_find
    adapter._evict_mamba_component_only = traced_primitive
    try:
        adapter.evict_mamba_only(handle)
    finally:
        adapter._find_exact_node = original_find
        adapter._evict_mamba_component_only = original_primitive


class SequentialTraceRuntimeAdapter:
    """通过专用测试传输动作追踪连续驱逐，不改变控制器接口。"""

    def __init__(
        self,
        client: object,
        tracked_handles: Mapping[str, RuntimeCheckpointHandle],
        *,
        nonce_namespace: str = "flowstate_step12h9a",
    ) -> None:
        if not nonce_namespace:
            raise ValueError("追踪控制请求命名空间不能为空")
        self._client = client
        self._tracked_handles = dict(tracked_handles)
        self._nonce_namespace = nonce_namespace
        self._previous_target: str | None = None
        self.evicted_checkpoint_ids: list[str] = []
        self.eviction_responses: list[dict[str, object]] = []
        self.trace_rows: list[dict[str, object]] = []

    def _tracked_descriptors(self) -> list[dict[str, object]]:
        """按标识排序构造每次查询使用的相同追踪集合。"""
        return [
            handle_descriptor(self._tracked_handles[checkpoint_id])
            for checkpoint_id in sorted(self._tracked_handles)
        ]

    def _append_trace_rows(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> None:
        """把多次控制响应合并为单调连续的全局追踪序列。"""
        for row in rows:
            self.trace_rows.append(
                {
                    **dict(row),
                    "sequence": len(self.trace_rows),
                }
            )

    def _record_s4(self, target_checkpoint_id: str, next_target: str | None) -> None:
        """用独立 scheduler safe point 记录调用间边界。"""
        response = self._client._call(
            {
                "op": "checkpoint_control",
                "nonce": f"{self._nonce_namespace}:s4:{target_checkpoint_id}",
                "action": "flowstate_trace_snapshot",
                "target_checkpoint_id": target_checkpoint_id,
                "next_target_checkpoint_id": next_target,
                "tracked_handles": self._tracked_descriptors(),
            }
        )
        self._append_trace_rows(response["trace_rows"])

    def evict_mamba_only(self, handle: RuntimeCheckpointHandle) -> None:
        """保持控制器单目标调用语义并采集当前目标 S0 至 S3。"""
        if self._previous_target is not None:
            self._record_s4(self._previous_target, handle.checkpoint_id)
        response = self._client._call(
            {
                "op": "checkpoint_control",
                "nonce": f"{self._nonce_namespace}:evict:{handle.checkpoint_id}",
                "action": "flowstate_trace_evict_mamba_only",
                **handle_descriptor(handle),
                "target_checkpoint_id": handle.checkpoint_id,
                "tracked_handles": self._tracked_descriptors(),
            }
        )
        self.evicted_checkpoint_ids.append(handle.checkpoint_id)
        self.eviction_responses.append(response)
        self._append_trace_rows(response["trace_rows"])
        self._previous_target = handle.checkpoint_id

    def finish(self) -> None:
        """为最后一个目标补充序列结束时的 S4 边界。"""
        if self._previous_target is None:
            return
        self._record_s4(self._previous_target, None)
        self._previous_target = None
