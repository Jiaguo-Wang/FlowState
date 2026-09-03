from __future__ import annotations

import pytest

from evaluation.openhands_sequential_eviction_rematerialization_audit import (
    BOUNDARY_OPERATIONS,
    LRU_SECOND_EVICTION_ORDER,
    LRU_TRACKED_CHECKPOINTS,
    TRACE_BOUNDARIES,
    AllocatorTraceState,
    CheckpointTraceState,
    SequentialTraceRecorder,
    SequentialTraceRuntimeAdapter,
    TraceSnapshot,
    find_first_rematerializations,
    instrument_evict_mamba_only,
    validate_sequential_trace,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController
from flowstate.optimizer import AllocationResult
from flowstate.state_catalog import CheckpointCandidate


SHORT_IDS = ("A1", "B1", "C1", "C2", "D1", "D2")
EVICTION_ORDER = ("A1", "B1", "C1", "C2")


def trace_state(checkpoint_id: str, present: bool) -> CheckpointTraceState:
    """构造一个稳定节点上的循环状态快照。"""
    order = SHORT_IDS.index(checkpoint_id) + 1
    return CheckpointTraceState(
        checkpoint_id=checkpoint_id,
        recurrent_present=present,
        host_present=False,
        in_mamba_lru=present,
        node_id=order,
        token_position=order * 64,
        mamba_slots=(order,) if present else (),
    )


def snapshot(states: dict[str, bool]) -> TraceSnapshot:
    """构造包含六个检查点和分配器计数的全量快照。"""
    present_count = sum(states.values())
    return TraceSnapshot(
        checkpoints={
            checkpoint_id: trace_state(checkpoint_id, states[checkpoint_id])
            for checkpoint_id in SHORT_IDS
        },
        allocator=AllocatorTraceState(
            available_slots=28 - present_count,
            evictable_slots=present_count,
            protected_slots=0,
        ),
    )


def complete_rows(
    rematerialize_at: tuple[str, str] | None = None,
) -> list[dict[str, object]]:
    """构造按驱逐顺序变化的完整 S0 至 S4 追踪。"""
    states = {checkpoint_id: True for checkpoint_id in SHORT_IDS}
    rows = []
    sequence = 0
    for target in EVICTION_ORDER:
        for boundary in TRACE_BOUNDARIES:
            if boundary == "S2":
                states[target] = False
            if rematerialize_at == (target, boundary):
                states["A1"] = True
                states["B1"] = True
            current = snapshot(states)
            rows.append(
                {
                    "sequence": sequence,
                    "target_checkpoint_id": target,
                    "boundary": boundary,
                    "runtime_operation": BOUNDARY_OPERATIONS[boundary],
                    "checkpoints": {
                        checkpoint_id: state.row()
                        for checkpoint_id, state in current.checkpoints.items()
                    },
                    "mamba_allocator": current.allocator.row(),
                }
            )
            sequence += 1
    return rows


def runtime_handle(checkpoint_id: str) -> RuntimeCheckpointHandle:
    """构造传输测试使用的稳定运行时句柄。"""
    order = SHORT_IDS.index(checkpoint_id) + 1
    return RuntimeCheckpointHandle(
        checkpoint_id=checkpoint_id,
        token_ids=(order, order + 10),
        expected_node_id=order,
        expected_prefix_digest=f"摘要-{checkpoint_id}",
    )


def candidate(checkpoint_id: str) -> CheckpointCandidate:
    """构造控制器顺序测试使用的驻留候选。"""
    order = SHORT_IDS.index(checkpoint_id) + 1
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=f"工作流-{checkpoint_id[0]}",
        lineage_path=("openhands", checkpoint_id[0]),
        token_pos=order * 64,
        memory_bytes=1,
        recurrent_resident=True,
        fa_resident=True,
    )


def test_frozen_lru_trace_scope_matches_failed_gate() -> None:
    assert len(LRU_SECOND_EVICTION_ORDER) == 4
    assert len(LRU_TRACKED_CHECKPOINTS) == 6
    assert LRU_SECOND_EVICTION_ORDER == tuple(
        sorted(LRU_SECOND_EVICTION_ORDER)
    )


def test_recorder_rejects_incomplete_checkpoint_snapshot() -> None:
    recorder = SequentialTraceRecorder(
        SHORT_IDS,
        lambda: TraceSnapshot(
            checkpoints={"A1": trace_state("A1", True)},
            allocator=AllocatorTraceState(27, 1, 0),
        ),
    )
    with pytest.raises(RuntimeError, match="集合不完整"):
        recorder.record(target_checkpoint_id="A1", boundary="S0")


def test_complete_trace_requires_s0_through_s4_for_every_target() -> None:
    rows = complete_rows()
    validate_sequential_trace(rows, EVICTION_ORDER, SHORT_IDS)
    assert len(rows) == len(EVICTION_ORDER) * len(TRACE_BOUNDARIES)


def test_trace_validation_detects_missing_boundary() -> None:
    rows = complete_rows()
    del rows[7]
    with pytest.raises(RuntimeError, match="顺序不完整"):
        validate_sequential_trace(rows, EVICTION_ORDER, SHORT_IDS)


def test_first_rematerialization_reports_target_operation_and_states() -> None:
    rows = complete_rows(rematerialize_at=("C1", "S0"))
    events = find_first_rematerializations(rows, ("A1", "B1"))
    for checkpoint_id in ("A1", "B1"):
        event = events[checkpoint_id]
        assert event is not None
        assert event.first_rematerialization_target == "C1"
        assert event.boundary == "S0"
        assert event.triggering_operation == BOUNDARY_OPERATIONS["S0"]
        assert event.previous_state.recurrent_present is False
        assert event.new_state.recurrent_present is True


def test_no_false_rematerialization_before_first_absence() -> None:
    rows = complete_rows()
    events = find_first_rematerializations(rows, ("D1", "D2"))
    assert events == {"D1": None, "D2": None}


class FakeInstrumentedAdapter:
    """模拟适配器内部真实查找、原语和后置查找顺序。"""

    def __init__(self, states: dict[str, bool]) -> None:
        self.states = states
        self.operations: list[str] = []

    def _find_exact_node(self, token_ids, extra_key):
        del token_ids, extra_key
        self.operations.append("查找")
        return object(), ()

    def _evict_mamba_component_only(self, node):
        del node
        self.operations.append("原语")
        self.states["A1"] = False

    def evict_mamba_only(self, handle):
        node, _ = self._find_exact_node(handle.token_ids, handle.extra_key)
        self._evict_mamba_component_only(node)
        self._find_exact_node(handle.token_ids, handle.extra_key)


def test_internal_instrumentation_preserves_adapter_call_order() -> None:
    states = {checkpoint_id: True for checkpoint_id in SHORT_IDS}
    adapter = FakeInstrumentedAdapter(states)
    original_find = adapter._find_exact_node
    original_primitive = adapter._evict_mamba_component_only
    recorder = SequentialTraceRecorder(SHORT_IDS, lambda: snapshot(states))
    instrument_evict_mamba_only(adapter, runtime_handle("A1"), recorder)
    assert adapter.operations == ["查找", "原语", "查找"]
    assert [row["boundary"] for row in recorder.rows] == ["S0", "S1", "S2"]
    assert adapter._find_exact_node == original_find
    assert adapter._evict_mamba_component_only == original_primitive


class FakeControlClient:
    """模拟诊断传输响应并记录控制调用顺序。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def _call(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(request)
        action = request["action"]
        target = str(request["target_checkpoint_id"])
        boundaries = ("S4",) if action == "flowstate_trace_snapshot" else (
            "S0",
            "S1",
            "S2",
            "S3",
        )
        return {
            "trace_rows": [
                {
                    "sequence": 0,
                    "target_checkpoint_id": target,
                    "boundary": boundary,
                    "runtime_operation": BOUNDARY_OPERATIONS[boundary],
                    "checkpoints": {},
                    "mamba_allocator": {},
                }
                for boundary in boundaries
            ]
        }


class FrozenOptimizer:
    """返回固定选择集合，便于确认控制器语义未变化。"""

    def __init__(self, selected_ids: tuple[str, ...]) -> None:
        self._selected_ids = selected_ids

    def select(self, continuations, candidates, budget_bytes):
        del continuations
        by_id = {item.checkpoint_id: item for item in candidates}
        selected = tuple(by_id[value] for value in self._selected_ids)
        used_bytes = sum(item.memory_bytes for item in selected)
        assert used_bytes <= budget_bytes
        return AllocationResult(
            selected=selected,
            total_benefit_ms=0.0,
            recovery_cost_before_ms=0.0,
            recovery_cost_after_ms=0.0,
            used_bytes=used_bytes,
        )


def test_runtime_adapter_places_s4_between_target_calls() -> None:
    client = FakeControlClient()
    handles = {
        checkpoint_id: runtime_handle(checkpoint_id)
        for checkpoint_id in SHORT_IDS
    }
    adapter = SequentialTraceRuntimeAdapter(client, handles)
    adapter.evict_mamba_only(handles["A1"])
    adapter.evict_mamba_only(handles["B1"])
    adapter.finish()
    actions = [call["action"] for call in client.calls]
    assert actions == [
        "flowstate_trace_evict_mamba_only",
        "flowstate_trace_snapshot",
        "flowstate_trace_evict_mamba_only",
        "flowstate_trace_snapshot",
    ]
    assert adapter.evicted_checkpoint_ids == ["A1", "B1"]
    assert [row["boundary"] for row in adapter.trace_rows] == [
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
    ]
    assert [row["sequence"] for row in adapter.trace_rows] == list(range(10))


def test_runtime_adapter_uses_isolated_nonce_namespace() -> None:
    client = FakeControlClient()
    handles = {
        checkpoint_id: runtime_handle(checkpoint_id)
        for checkpoint_id in SHORT_IDS
    }
    adapter = SequentialTraceRuntimeAdapter(
        client,
        handles,
        nonce_namespace="最终门禁:barrier2",
    )
    adapter.evict_mamba_only(handles["A1"])
    adapter.finish()
    assert all(
        str(call["nonce"]).startswith("最终门禁:barrier2:")
        for call in client.calls
    )


def test_controller_selection_and_eviction_order_are_unchanged() -> None:
    client = FakeControlClient()
    candidates = tuple(candidate(checkpoint_id) for checkpoint_id in SHORT_IDS)
    handles = {
        checkpoint_id: runtime_handle(checkpoint_id)
        for checkpoint_id in SHORT_IDS
    }
    adapter = SequentialTraceRuntimeAdapter(client, handles)
    selected = ("D1", "D2")
    controller = StateController(
        FrozenOptimizer(selected),
        adapter,
    )
    allocation = controller.reconcile((), candidates, handles, 2)
    adapter.finish()
    assert {item.checkpoint_id for item in allocation.selected} == set(selected)
    assert adapter.evicted_checkpoint_ids == list(EVICTION_ORDER)
    assert all(
        call["action"]
        in {
            "flowstate_trace_evict_mamba_only",
            "flowstate_trace_snapshot",
        }
        for call in client.calls
    )
