from __future__ import annotations

from collections.abc import Sequence

import pytest

from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import ReconcileExecutionError, StateController
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


_CHECKPOINT_SIZE_VALUE = 49.125 * 1024 * 1024
assert _CHECKPOINT_SIZE_VALUE.is_integer()
CHECKPOINT_SIZE_BYTES = int(_CHECKPOINT_SIZE_VALUE)
assert CHECKPOINT_SIZE_BYTES == 51_511_296


class FakeAdapter:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.called_checkpoint_ids: list[str] = []
        self.evicted_checkpoint_ids: list[str] = []

    def evict_mamba_only(self, handle: RuntimeCheckpointHandle) -> None:
        self.called_checkpoint_ids.append(handle.checkpoint_id)
        if handle.checkpoint_id == self.fail_on:
            raise RuntimeError(f"注入驱逐失败：{handle.checkpoint_id}")
        self.evicted_checkpoint_ids.append(handle.checkpoint_id)


def make_continuations() -> tuple[PendingContinuation, ...]:
    return tuple(
        PendingContinuation(
            continuation_id=f"B{index}",
            workflow_id=f"W{index}",
            lineage_path=("P", "B"),
            anchor_pos=32_768,
            resident_fa_frontier=32_768,
        )
        for index in range(1, 5)
    )


def make_candidates() -> tuple[CheckpointCandidate, ...]:
    parents = tuple(
        CheckpointCandidate(
            checkpoint_id=f"P{index}",
            workflow_id=f"W{index}",
            lineage_path=("P",),
            token_pos=32_768,
            memory_bytes=CHECKPOINT_SIZE_BYTES,
        )
        for index in range(1, 5)
    )
    children = tuple(
        CheckpointCandidate(
            checkpoint_id=f"C{index}",
            workflow_id=f"W{index}",
            lineage_path=("P", "A"),
            token_pos=32_832,
            memory_bytes=CHECKPOINT_SIZE_BYTES,
        )
        for index in range(1, 5)
    )
    return parents + children


def make_handles(
    candidates: Sequence[CheckpointCandidate],
) -> dict[str, RuntimeCheckpointHandle]:
    return {
        candidate.checkpoint_id: RuntimeCheckpointHandle(
            checkpoint_id=candidate.checkpoint_id,
            token_ids=(index + 1,),
        )
        for index, candidate in enumerate(candidates)
    }


def make_controller(adapter: FakeAdapter) -> StateController:
    optimizer = GlobalOptimizer(RecoveryCostModel())
    return StateController(optimizer, adapter)


def test_wp3b_controller_gate() -> None:
    continuations = make_continuations()
    candidates = make_candidates()
    handles = make_handles(candidates)
    adapter = FakeAdapter()
    controller = make_controller(adapter)

    result = controller.reconcile(
        continuations,
        candidates,
        handles,
        4 * CHECKPOINT_SIZE_BYTES,
    )

    selected_ids = tuple(
        candidate.checkpoint_id for candidate in result.selected
    )
    assert selected_ids == ("P1", "P2", "P3", "P4")
    assert adapter.evicted_checkpoint_ids == ["C1", "C2", "C3", "C4"]
    assert not set(selected_ids).intersection(adapter.evicted_checkpoint_ids)


def test_missing_runtime_handle_is_rejected_before_any_eviction() -> None:
    candidates = make_candidates()
    handles = make_handles(candidates)
    del handles["C3"]
    adapter = FakeAdapter()
    controller = make_controller(adapter)

    with pytest.raises(ValueError, match="C3"):
        controller.reconcile(
            make_continuations(),
            candidates,
            handles,
            4 * CHECKPOINT_SIZE_BYTES,
        )

    assert adapter.evicted_checkpoint_ids == []


def test_handle_mapping_key_mismatch_is_rejected_before_selection() -> None:
    candidates = make_candidates()
    handles = make_handles(candidates)
    handles["C1"] = RuntimeCheckpointHandle(
        checkpoint_id="C2",
        token_ids=(999,),
    )
    adapter = FakeAdapter()
    controller = make_controller(adapter)

    with pytest.raises(ValueError, match="键 C1，句柄 C2"):
        controller.reconcile(
            make_continuations(),
            candidates,
            handles,
            4 * CHECKPOINT_SIZE_BYTES,
        )

    assert adapter.called_checkpoint_ids == []


def test_partial_eviction_failure_reports_progress_and_stops() -> None:
    continuations = make_continuations()[:3]
    candidates = tuple(
        candidate
        for candidate in make_candidates()
        if candidate.workflow_id in {"W1", "W2", "W3"}
    )
    handles = make_handles(candidates)
    adapter = FakeAdapter(fail_on="C2")
    controller = make_controller(adapter)

    with pytest.raises(ReconcileExecutionError) as error_info:
        controller.reconcile(
            continuations,
            candidates,
            handles,
            3 * CHECKPOINT_SIZE_BYTES,
        )

    error = error_info.value
    assert error.completed_evictions == ("C1",)
    assert error.failed_checkpoint_id == "C2"
    assert isinstance(error.cause, RuntimeError)
    assert adapter.evicted_checkpoint_ids == ["C1"]
    assert adapter.called_checkpoint_ids == ["C1", "C2"]
    assert "C3" not in adapter.called_checkpoint_ids


def test_non_resident_candidate_is_not_evicted_again() -> None:
    candidate = CheckpointCandidate(
        checkpoint_id="已驱逐检查点",
        workflow_id="W1",
        lineage_path=("P",),
        token_pos=32_768,
        memory_bytes=CHECKPOINT_SIZE_BYTES,
        recurrent_resident=False,
    )
    adapter = FakeAdapter()
    controller = make_controller(adapter)

    result = controller.reconcile(
        make_continuations(),
        [candidate],
        {},
        0,
    )

    assert result.selected == ()
    assert adapter.evicted_checkpoint_ids == []


def test_candidate_order_does_not_change_selected_or_evicted_results() -> None:
    continuations = make_continuations()
    candidates = make_candidates()
    by_id = {
        candidate.checkpoint_id: candidate
        for candidate in candidates
    }
    candidate_orders = (
        candidates,
        tuple(reversed(candidates)),
        (
            by_id["C3"],
            by_id["P2"],
            by_id["C1"],
            by_id["P4"],
            by_id["C4"],
            by_id["P1"],
            by_id["C2"],
            by_id["P3"],
        ),
    )
    outcomes = []

    for candidate_order in candidate_orders:
        adapter = FakeAdapter()
        controller = make_controller(adapter)
        result = controller.reconcile(
            continuations,
            candidate_order,
            make_handles(candidate_order),
            4 * CHECKPOINT_SIZE_BYTES,
        )
        outcomes.append(
            (
                tuple(
                    candidate.checkpoint_id
                    for candidate in result.selected
                ),
                tuple(adapter.evicted_checkpoint_ids),
            )
        )

    expected = (
        ("P1", "P2", "P3", "P4"),
        ("C1", "C2", "C3", "C4"),
    )
    assert outcomes == [expected, expected, expected]
