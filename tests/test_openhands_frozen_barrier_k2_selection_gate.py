from __future__ import annotations

import inspect

from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    CHECKPOINT_SIZE_BYTES,
    ENGINE_CONFIGURATION_COMMON_BARRIER,
    LOGICAL_K,
)
from evaluation.openhands_frozen_barrier_k2_selection_gate import (
    ENGINE_CONFIGURATION_K2_SELECTION,
    _normalize,
    marconi_score_rows,
    run_selectors,
    runtime_mutation_result,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


def candidate(checkpoint_id: str, workflow_id: str, token_pos: int):
    """构造 selector 单测所需的同大小候选。"""
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        lineage_path=("openhands", workflow_id),
        token_pos=token_pos,
        memory_bytes=CHECKPOINT_SIZE_BYTES,
        recurrent_resident=True,
        fa_resident=True,
    )


def continuation(workflow_id: str, target: int):
    """构造与候选同 lineage 的待续请求。"""
    return PendingContinuation(
        continuation_id=f"{workflow_id}2",
        workflow_id=workflow_id,
        lineage_path=("openhands", workflow_id),
        anchor_pos=target,
        resident_fa_frontier=target,
    )


def frozen_inputs():
    """构造四候选、四 pending 与冻结在线 metadata。"""
    positions = {"A": 3968, "B": 2432, "C": 3712, "D": 3520}
    targets = {"A": 3968, "B": 2437, "C": 3712, "D": 3520}
    candidates = tuple(
        candidate(f"{label}1", label, positions[label])
        for label in "ABCD"
    )
    continuations = tuple(
        continuation(label, targets[label]) for label in "ABCD"
    )
    metadata = tuple(
        {
            "checkpoint_id": f"{label}1",
            "creation_order": order,
            "last_access_order": order,
            "marconi_recency": float(order),
            "marconi_incremental_span": float(positions[label]),
        }
        for order, label in enumerate("ABCD", start=1)
    )
    return candidates, continuations, metadata


def test_all_three_selectors_share_one_eligible_universe() -> None:
    candidates, continuations, metadata = frozen_inputs()
    result = run_selectors(candidates, continuations, metadata)
    assert result["common_candidate_universe"] is True
    assert result["selection_valid"] is True
    assert result["flowstate_budget_valid"] is True
    inputs = result["candidate_ids_by_policy_input"]
    assert inputs["LRU"] == inputs["Marconi"] == inputs["FlowState"]
    assert all(
        len(result[name]["selected_checkpoint_ids"]) <= LOGICAL_K
        for name in ("lru", "marconi", "flowstate")
    )


def test_marconi_score_rows_match_frozen_formula() -> None:
    candidates, _, metadata = frozen_inputs()
    rows = marconi_score_rows(candidates, metadata)
    by_id = {row["checkpoint_id"]: row for row in rows}
    assert by_id["A1"]["normalized_recency"] == 0.0
    assert by_id["D1"]["normalized_recency"] == 1.0
    assert by_id["A1"]["normalized_flop_efficiency"] == 1.0
    assert by_id["B1"]["normalized_flop_efficiency"] == 0.0
    assert all(
        row["final_score"]
        == row["normalized_recency"]
        + row["normalized_flop_efficiency"]
        for row in rows
    )


def test_marconi_tied_normalization_follows_existing_zero_behavior() -> None:
    assert _normalize({"A": 7.0, "B": 7.0}) == {"A": 0.0, "B": 0.0}


def test_flowstate_reports_final_allocation_without_private_trace() -> None:
    candidates, continuations, metadata = frozen_inputs()
    result = run_selectors(candidates, continuations, metadata)["flowstate"]
    assert result["used_bytes"] <= BUDGET_BYTES
    assert result["recovery_cost_after_ms"] <= result["recovery_cost_before_ms"]
    assert result["total_benefit_ms"] >= 0.0
    assert result["per_step_marginal_trace"] == "NOT EXPOSED"


def test_empty_runtime_difference_is_fully_unchanged() -> None:
    snapshot = {
        "tree": {"structure_rows": [[1, None, [], 0]]},
        "accounting": {"mamba_available": 28},
        "recency_rows": [[1, 1.0, 1.0, 0]],
        "mamba_lru_order_mru_to_lru": [],
        "reference_rows": [[1, 0, 0, 0, 0, 0, 0]],
        "full_evictable_leaf_ids": [],
    }
    result = runtime_mutation_result(
        {"semantic_snapshot": snapshot},
        {"semantic_snapshot": snapshot},
    )
    assert result["state_equal"] is True
    assert result["changed_fields"] == []
    assert all(
        value is True
        for key, value in result.items()
        if key.endswith("_unchanged")
    )


def test_gate_reuses_snapshot_engine_configuration_without_override() -> None:
    assert ENGINE_CONFIGURATION_K2_SELECTION == ENGINE_CONFIGURATION_COMMON_BARRIER
    assert ENGINE_CONFIGURATION_K2_SELECTION is not ENGINE_CONFIGURATION_COMMON_BARRIER


def test_gate_source_contains_no_controller_or_eviction_call() -> None:
    module = __import__(
        "evaluation.openhands_frozen_barrier_k2_selection_gate",
        fromlist=["unused"],
    )
    source = inspect.getsource(module)
    forbidden = (
        "StateController",
        ".reconcile(",
        "evict_mamba_only(",
    )
    assert all(value not in source for value in forbidden)
