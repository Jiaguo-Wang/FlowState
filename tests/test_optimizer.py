from __future__ import annotations

import pytest

from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


CHECKPOINT_SIZE_BYTES = 51_511_296


@pytest.fixture(scope="module")
def recovery_cost_model() -> RecoveryCostModel:
    return RecoveryCostModel()


@pytest.fixture(scope="module")
def optimizer(recovery_cost_model: RecoveryCostModel) -> GlobalOptimizer:
    return GlobalOptimizer(recovery_cost_model)


def make_continuation(
    workflow_id: str = "W1",
    *,
    continuation_id: str = "B1",
    lineage_path: tuple[str, ...] = ("P", "B"),
    anchor_pos: int = 32_768,
    resident_fa_frontier: int = 32_768,
) -> PendingContinuation:
    return PendingContinuation(
        continuation_id=continuation_id,
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        anchor_pos=anchor_pos,
        resident_fa_frontier=resident_fa_frontier,
    )


def make_checkpoint(
    checkpoint_id: str,
    *,
    workflow_id: str = "W1",
    lineage_path: tuple[str, ...] = ("P",),
    token_pos: int = 32_768,
    memory_bytes: int = CHECKPOINT_SIZE_BYTES,
    recurrent_resident: bool = True,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        token_pos=token_pos,
        memory_bytes=memory_bytes,
        recurrent_resident=recurrent_resident,
    )


def test_empty_candidates_return_empty_result(
    optimizer: GlobalOptimizer,
    recovery_cost_model: RecoveryCostModel,
) -> None:
    continuation = make_continuation()

    result = optimizer.select([continuation], [], CHECKPOINT_SIZE_BYTES)

    expected_cost = recovery_cost_model.estimate(32_768, 32_768)
    assert result.selected == ()
    assert result.total_benefit_ms == 0.0
    assert result.recovery_cost_before_ms == pytest.approx(expected_cost)
    assert result.recovery_cost_after_ms == pytest.approx(expected_cost)
    assert result.used_bytes == 0


def test_zero_budget_returns_empty_result(optimizer: GlobalOptimizer) -> None:
    result = optimizer.select(
        [make_continuation()],
        [make_checkpoint("P1")],
        0,
    )

    assert result.selected == ()
    assert result.total_benefit_ms == 0.0
    assert result.used_bytes == 0


def test_negative_budget_is_rejected(optimizer: GlobalOptimizer) -> None:
    with pytest.raises(ValueError, match="内存预算必须大于等于零"):
        optimizer.select([], [], -1)


def test_duplicate_checkpoint_id_is_rejected(
    optimizer: GlobalOptimizer,
) -> None:
    candidates = (
        make_checkpoint("重复检查点", workflow_id="W1"),
        make_checkpoint("重复检查点", workflow_id="W2"),
    )

    with pytest.raises(ValueError, match="重复检查点"):
        optimizer.select(
            [make_continuation()],
            candidates,
            2 * CHECKPOINT_SIZE_BYTES,
        )


def test_one_slot_selects_largest_marginal_gain(
    optimizer: GlobalOptimizer,
) -> None:
    candidates = [
        make_checkpoint("浅检查点", token_pos=8_192),
        make_checkpoint("深检查点", token_pos=32_768),
    ]

    result = optimizer.select(
        [make_continuation()],
        candidates,
        CHECKPOINT_SIZE_BYTES,
    )

    assert tuple(candidate.checkpoint_id for candidate in result.selected) == (
        "深检查点",
    )


def test_zero_gain_candidate_does_not_consume_budget(
    optimizer: GlobalOptimizer,
) -> None:
    unrelated = make_checkpoint("无依赖检查点", workflow_id="W2")

    result = optimizer.select(
        [make_continuation(workflow_id="W1")],
        [unrelated],
        CHECKPOINT_SIZE_BYTES,
    )

    assert result.selected == ()
    assert result.used_bytes == 0


def test_non_resident_high_value_candidate_cannot_be_selected(
    optimizer: GlobalOptimizer,
) -> None:
    candidates = [
        make_checkpoint(
            "非驻留深检查点",
            token_pos=32_768,
            recurrent_resident=False,
        ),
        make_checkpoint("驻留浅检查点", token_pos=8_192),
    ]

    result = optimizer.select(
        [make_continuation()],
        candidates,
        CHECKPOINT_SIZE_BYTES,
    )

    assert tuple(candidate.checkpoint_id for candidate in result.selected) == (
        "驻留浅检查点",
    )


def test_different_checkpoint_sizes_are_rejected(
    optimizer: GlobalOptimizer,
) -> None:
    candidates = [
        make_checkpoint("P1"),
        make_checkpoint("P2", memory_bytes=CHECKPOINT_SIZE_BYTES + 1),
    ]

    with pytest.raises(ValueError, match="只支持等大小检查点"):
        optimizer.select(
            [make_continuation()],
            candidates,
            2 * CHECKPOINT_SIZE_BYTES,
        )


def test_equal_gain_selection_is_deterministic(
    optimizer: GlobalOptimizer,
) -> None:
    continuations = [
        make_continuation("W1", continuation_id="B1"),
        make_continuation("W2", continuation_id="B2"),
    ]
    candidates = [
        make_checkpoint("Z", workflow_id="W1"),
        make_checkpoint("A", workflow_id="W2"),
    ]

    selected_orders = [
        tuple(
            candidate.checkpoint_id
            for candidate in optimizer.select(
                continuations,
                candidates,
                CHECKPOINT_SIZE_BYTES,
            ).selected
        )
        for _ in range(10)
    ]

    assert selected_orders == [("A",)] * 10


def test_wp3b_offline_optimizer_gate(
    optimizer: GlobalOptimizer,
    recovery_cost_model: RecoveryCostModel,
) -> None:
    checkpoint_size_mib = 49.125 * 1024 * 1024
    assert checkpoint_size_mib.is_integer()
    checkpoint_size = int(checkpoint_size_mib)
    assert checkpoint_size == CHECKPOINT_SIZE_BYTES

    continuations = [
        make_continuation(f"W{index}", continuation_id=f"B{index}")
        for index in range(1, 5)
    ]
    parents = [
        make_checkpoint(f"P{index}", workflow_id=f"W{index}")
        for index in range(1, 5)
    ]
    children = [
        make_checkpoint(
            f"C{index}",
            workflow_id=f"W{index}",
            lineage_path=("P", "A"),
            token_pos=32_832,
        )
        for index in range(1, 5)
    ]
    candidates = [
        children[3],
        parents[3],
        children[2],
        parents[2],
        children[1],
        parents[1],
        children[0],
        parents[0],
    ]
    budget_bytes = 4 * checkpoint_size

    gaps_before = [recovery_gap(continuation, ()) for continuation in continuations]
    result = optimizer.select(continuations, candidates, budget_bytes)
    gaps_after = [
        recovery_gap(continuation, result.selected)
        for continuation in continuations
    ]

    selected_ids = {candidate.checkpoint_id for candidate in result.selected}
    expected_total_cost = 4 * recovery_cost_model.estimate(32_768, 32_768)
    assert selected_ids == {"P1", "P2", "P3", "P4"}
    assert tuple(candidate.checkpoint_id for candidate in result.selected) == (
        "P1",
        "P2",
        "P3",
        "P4",
    )
    assert gaps_before == [32_768, 32_768, 32_768, 32_768]
    assert gaps_after == [0, 0, 0, 0]
    assert result.recovery_cost_before_ms == pytest.approx(expected_total_cost)
    assert result.recovery_cost_after_ms == 0.0
    assert result.total_benefit_ms == pytest.approx(expected_total_cost)
    assert result.used_bytes == budget_bytes
