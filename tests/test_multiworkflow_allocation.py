from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


_CHECKPOINT_SIZE_VALUE = 49.125 * 1024 * 1024
assert _CHECKPOINT_SIZE_VALUE.is_integer()
CHECKPOINT_SIZE_BYTES = int(_CHECKPOINT_SIZE_VALUE)
assert CHECKPOINT_SIZE_BYTES == 51_511_296

EXPECTED_SELECTED_IDS = (
    "W1_PARENT",
    "W2_PARENT",
    "W4_PARENT",
)


@dataclass(frozen=True)
class OfflineScenario:
    continuations: tuple[PendingContinuation, ...]
    candidates: tuple[CheckpointCandidate, ...]


def make_continuation(
    continuation_id: str,
    workflow_id: str,
    lineage_path: tuple[str, ...],
    planning_target: int,
) -> PendingContinuation:
    return PendingContinuation(
        continuation_id=continuation_id,
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        anchor_pos=planning_target,
        resident_fa_frontier=planning_target,
    )


def make_checkpoint(
    checkpoint_id: str,
    workflow_id: str,
    lineage_path: tuple[str, ...],
    token_pos: int,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        token_pos=token_pos,
        memory_bytes=CHECKPOINT_SIZE_BYTES,
    )


@pytest.fixture(scope="module")
def recovery_cost_model() -> RecoveryCostModel:
    return RecoveryCostModel()


@pytest.fixture(scope="module")
def optimizer(recovery_cost_model: RecoveryCostModel) -> GlobalOptimizer:
    return GlobalOptimizer(recovery_cost_model)


@pytest.fixture(scope="module")
def scenario() -> OfflineScenario:
    continuations = (
        make_continuation("W1-B", "W1", ("ROOT1", "B"), 32_768),
        make_continuation("W1-C", "W1", ("ROOT1", "C"), 32_768),
        make_continuation("W2", "W2", ("ROOT2", "B"), 16_384),
        make_continuation("W3", "W3", ("ROOT3", "B"), 8_192),
        make_continuation("W4-A", "W4", ("ROOT4", "A"), 4_096),
        make_continuation("W4-B", "W4", ("ROOT4", "B"), 4_096),
        make_continuation("W4-C", "W4", ("ROOT4", "C"), 4_096),
    )
    candidates = (
        make_checkpoint("W1_PARENT", "W1", ("ROOT1",), 32_768),
        make_checkpoint("W1_SHALLOW", "W1", ("ROOT1",), 16_384),
        make_checkpoint("W2_PARENT", "W2", ("ROOT2",), 16_384),
        make_checkpoint("W3_PARENT", "W3", ("ROOT3",), 8_192),
        make_checkpoint("W4_PARENT", "W4", ("ROOT4",), 4_096),
    )
    return OfflineScenario(continuations, candidates)


def total_recovery_cost(
    model: RecoveryCostModel,
    continuations: Sequence[PendingContinuation],
    selected: Sequence[CheckpointCandidate],
) -> float:
    return sum(
        model.estimate(recovery_gap(continuation, selected))
        for continuation in continuations
    )


def candidate_benefit(
    model: RecoveryCostModel,
    continuations: Sequence[PendingContinuation],
    candidate: CheckpointCandidate,
) -> float:
    cost_before = total_recovery_cost(model, continuations, ())
    cost_after = total_recovery_cost(model, continuations, (candidate,))
    return cost_before - cost_after


def test_recovery_profile_and_initial_candidate_benefits(
    recovery_cost_model: RecoveryCostModel,
    scenario: OfflineScenario,
) -> None:
    phi = {
        replay_tokens: recovery_cost_model.estimate(replay_tokens)
        for replay_tokens in (4_096, 8_192, 16_384, 32_768)
    }
    benefits = {
        candidate.checkpoint_id: candidate_benefit(
            recovery_cost_model,
            scenario.continuations,
            candidate,
        )
        for candidate in scenario.candidates
    }

    assert phi[4_096] < phi[8_192] < phi[16_384] < phi[32_768]
    assert benefits["W1_PARENT"] == pytest.approx(2 * phi[32_768])
    assert benefits["W1_SHALLOW"] == pytest.approx(
        2 * (phi[32_768] - phi[16_384])
    )
    assert benefits["W2_PARENT"] == pytest.approx(phi[16_384])
    assert benefits["W3_PARENT"] == pytest.approx(phi[8_192])
    assert benefits["W4_PARENT"] == pytest.approx(3 * phi[4_096])
    assert benefits["W1_SHALLOW"] > 0.0
    assert benefits["W4_PARENT"] > benefits["W3_PARENT"]

    print("恢复成本剖面：", phi)
    print("候选检查点的初始恢复收益：", benefits)


def test_global_budget_selects_highest_total_recovery_benefit(
    optimizer: GlobalOptimizer,
    recovery_cost_model: RecoveryCostModel,
    scenario: OfflineScenario,
) -> None:
    budget_bytes = 3 * CHECKPOINT_SIZE_BYTES
    result = optimizer.select(
        scenario.continuations,
        scenario.candidates,
        budget_bytes,
    )
    selected_ids = tuple(
        candidate.checkpoint_id for candidate in result.selected
    )
    final_gaps = {
        continuation.continuation_id: recovery_gap(
            continuation,
            result.selected,
        )
        for continuation in scenario.continuations
    }

    assert set(selected_ids) == set(EXPECTED_SELECTED_IDS)
    assert selected_ids == EXPECTED_SELECTED_IDS
    assert "W1_SHALLOW" not in selected_ids
    assert result.used_bytes == budget_bytes
    assert len(result.selected) == 3
    assert {candidate.workflow_id for candidate in result.selected} == {
        "W1",
        "W2",
        "W4",
    }
    assert final_gaps == {
        "W1-B": 0,
        "W1-C": 0,
        "W2": 0,
        "W3": 8_192,
        "W4-A": 0,
        "W4-B": 0,
        "W4-C": 0,
    }
    assert result.recovery_cost_after_ms == pytest.approx(
        recovery_cost_model.estimate(8_192)
    )


def test_shallow_checkpoint_value_depends_on_selected_set(
    recovery_cost_model: RecoveryCostModel,
    scenario: OfflineScenario,
) -> None:
    candidates = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    parent = candidates["W1_PARENT"]
    shallow = candidates["W1_SHALLOW"]

    initial_gain = candidate_benefit(
        recovery_cost_model,
        scenario.continuations,
        shallow,
    )
    cost_with_parent = total_recovery_cost(
        recovery_cost_model,
        scenario.continuations,
        (parent,),
    )
    cost_with_both = total_recovery_cost(
        recovery_cost_model,
        scenario.continuations,
        (parent, shallow),
    )
    gain_after_parent = cost_with_parent - cost_with_both

    assert initial_gain > 0.0
    assert gain_after_parent == 0.0


def test_candidate_input_order_does_not_change_selection(
    optimizer: GlobalOptimizer,
    scenario: OfflineScenario,
) -> None:
    candidates = scenario.candidates
    by_id = {
        candidate.checkpoint_id: candidate
        for candidate in candidates
    }
    candidate_orders = (
        candidates,
        tuple(reversed(candidates)),
        (
            by_id["W3_PARENT"],
            by_id["W1_SHALLOW"],
            by_id["W4_PARENT"],
            by_id["W2_PARENT"],
            by_id["W1_PARENT"],
        ),
    )

    selected_orders = tuple(
        tuple(
            candidate.checkpoint_id
            for candidate in optimizer.select(
                scenario.continuations,
                candidate_order,
                3 * CHECKPOINT_SIZE_BYTES,
            ).selected
        )
        for candidate_order in candidate_orders
    )

    assert selected_orders == (EXPECTED_SELECTED_IDS,) * 3
