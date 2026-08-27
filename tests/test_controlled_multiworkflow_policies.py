from __future__ import annotations

from dataclasses import replace

from evaluation.controlled_multiworkflow_v1.policies import (
    select_equal_share,
    select_global_lru,
    select_recovery_only,
)
from evaluation.controlled_multiworkflow_v1.scenario import (
    CheckpointRecency,
    build_scenario,
)
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate


def _baseline_selections(
    candidates: tuple[CheckpointCandidate, ...],
) -> dict[str, tuple[str, ...]]:
    scenario = build_scenario()
    model = RecoveryCostModel()
    return {
        "Global-LRU": select_global_lru(
            candidates,
            scenario.metadata.checkpoint_recency,
            scenario.budget_bytes,
        ),
        "Equal-Share": select_equal_share(
            scenario.continuations,
            candidates,
            scenario.metadata.workflow_order,
            scenario.budget_bytes,
        ),
        "Recovery-Only": select_recovery_only(
            scenario.continuations,
            candidates,
            scenario.budget_bytes,
            model,
        ),
    }


def _flowstate_selection(
    candidates: tuple[CheckpointCandidate, ...],
) -> tuple[str, ...]:
    scenario = build_scenario()
    result = GlobalOptimizer(RecoveryCostModel()).select(
        scenario.continuations,
        candidates,
        scenario.budget_bytes,
    )
    return tuple(candidate.checkpoint_id for candidate in result.selected)


def test_policy_metadata_is_explicit_and_frozen() -> None:
    metadata = build_scenario().metadata

    assert metadata.workflow_order == ("W1", "W2", "W3", "W4")
    assert tuple(
        (item.checkpoint_id, item.creation_order, item.last_access_order)
        for item in metadata.checkpoint_recency
    ) == (
        ("W1_SHALLOW", 1, 1),
        ("W1_PARENT", 2, 2),
        ("W2_PARENT", 3, 3),
        ("W3_PARENT", 4, 4),
        ("W4_PARENT", 5, 5),
    )


def test_frozen_policy_selections() -> None:
    scenario = build_scenario()
    flowstate_ids = _flowstate_selection(scenario.candidates)
    selections = _baseline_selections(scenario.candidates)

    assert flowstate_ids == ("W1_PARENT", "W2_PARENT", "W4_PARENT")
    assert selections == {
        "Global-LRU": ("W4_PARENT", "W3_PARENT", "W2_PARENT"),
        "Equal-Share": ("W1_PARENT", "W2_PARENT", "W3_PARENT"),
        "Recovery-Only": ("W1_PARENT", "W1_SHALLOW", "W2_PARENT"),
    }


def test_all_policies_obey_capacity_uniqueness_and_residency() -> None:
    scenario = build_scenario()
    nonresident = replace(
        scenario.candidates[0],
        checkpoint_id="A_NONRESIDENT",
        recurrent_resident=False,
    )
    candidates = scenario.candidates + (nonresident,)
    recency = scenario.metadata.checkpoint_recency + (
        CheckpointRecency("A_NONRESIDENT", 999, 999),
    )
    selections = {
        "FlowState": _flowstate_selection(candidates),
        "Global-LRU": select_global_lru(
            candidates,
            recency,
            scenario.budget_bytes,
        ),
        "Equal-Share": select_equal_share(
            scenario.continuations,
            candidates,
            scenario.metadata.workflow_order,
            scenario.budget_bytes,
        ),
        "Recovery-Only": select_recovery_only(
            scenario.continuations,
            candidates,
            scenario.budget_bytes,
            RecoveryCostModel(),
        ),
    }

    for selected_ids in selections.values():
        assert len(selected_ids) == 3
        assert len(set(selected_ids)) == 3
        assert "A_NONRESIDENT" not in selected_ids


def test_policy_results_do_not_depend_on_candidate_input_order() -> None:
    scenario = build_scenario()
    candidates = scenario.candidates
    orders = (
        candidates,
        tuple(reversed(candidates)),
        (candidates[2], candidates[4], candidates[0], candidates[3], candidates[1]),
    )
    expected = _baseline_selections(orders[0])
    expected_flowstate = _flowstate_selection(orders[0])

    for order in orders[1:]:
        assert _baseline_selections(order) == expected
        assert _flowstate_selection(order) == expected_flowstate
