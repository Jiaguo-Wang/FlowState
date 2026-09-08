"""RQ5-A 冻结 allocation snapshot 设计消融测试。"""

from __future__ import annotations

from evaluation.rq3_frozen_snapshot_evaluator import (
    FrozenCheckpointRuntimeEvidence,
    FrozenOnlineInformationBoundary,
    build_allocation_snapshot,
    evaluate_objective,
)
from evaluation.rq5_frozen_snapshot_ablation import (
    VARIANTS,
    _aggregate_rows,
    _marginal_structure,
    _select_greedy,
    _select_standalone_topk,
    _selection_diagnostics,
    load_gap_only_model,
)
from evaluation.sota_metadata import CONTROLLED_MARCONI_ALPHA
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


def _digest(character: str) -> str:
    """构造测试使用的稳定摘要。"""
    return character * 64


def _snapshot():
    """构造包含同一 pending 冗余候选的最小冻结快照。"""
    pending = [
        PendingContinuation("P1", "W1", ("W1", "下一步"), 8192, 8192),
        PendingContinuation("P2", "W2", ("W2", "下一步"), 4096, 4096),
    ]
    candidates = [
        CheckpointCandidate("A", "W1", ("W1",), 8192, 1024),
        CheckpointCandidate("B", "W1", ("W1",), 7168, 1024),
        CheckpointCandidate("C", "W2", ("W2",), 4096, 1024),
    ]
    identifiers = [item.checkpoint_id for item in candidates]
    return build_allocation_snapshot(
        allocation_epoch=3,
        snapshot_id="rq5-test",
        pending_continuations=pending,
        eligible_candidates=candidates,
        creation_order_by_checkpoint={value: index for index, value in enumerate(identifiers)},
        last_access_order_by_checkpoint={value: index for index, value in enumerate(identifiers)},
        marconi_flop_saved_by_checkpoint={value: 1.0 for value in identifiers},
        access_frequency_by_checkpoint={value: 1 for value in identifiers},
        frequency_observed_through_epoch=3,
        marconi_alpha=CONTROLLED_MARCONI_ALPHA,
        logical_budget_k=2,
        budget_bytes=2048,
        runtime_evidence=[
            FrozenCheckpointRuntimeEvidence(value, index, _digest("a"), _digest("b"))
            for index, value in enumerate(identifiers)
        ],
        residency_snapshot_digest=_digest("c"),
        online_boundary=FrozenOnlineInformationBoundary(
            materialized_through_epoch=3,
            visible_continuation_ids=("P1", "P2"),
        ),
    )


def test_gap_only_model_is_frozen_and_position_independent() -> None:
    """冻结 M0 在相同 G 下不得随 T 变化。"""
    model, evidence = load_gap_only_model()
    assert evidence["heldout_used_for_fit"] is False
    assert evidence["frozen_model_status"] == "FAIL"
    assert model.estimate(4096, 4096) == model.estimate(4096, 131072)
    assert model.estimate(4096, 8192) == 238.13663222916668


def test_full_selector_and_standalone_share_budget_and_candidates() -> None:
    """两种 selector 必须使用同一 K 与候选全集。"""
    snapshot = _snapshot()
    full_order, _ = _select_greedy(snapshot, RecoveryCostModel())
    standalone_ids, standalone_order, _ = _select_standalone_topk(snapshot)
    assert len(full_order) <= snapshot.logical_budget_k
    assert len(standalone_ids) <= snapshot.logical_budget_k
    assert set(standalone_ids) == set(standalone_order)
    assert set(full_order) <= {"A", "B", "C"}
    assert set(standalone_ids) <= {"A", "B", "C"}


def test_zero_current_marginal_uses_common_objective() -> None:
    """依次选入已被深检查点覆盖的浅检查点时应识别零当前边际。"""
    diagnostics = _selection_diagnostics(_snapshot(), ("A", "B"))
    assert diagnostics["selected_count"] == 2
    assert diagnostics["zero_current_marginal_count"] == 1
    assert diagnostics["zero_current_marginal_rate"] == 0.5


def test_marginal_structure_classifies_same_pending_redundancy() -> None:
    """单 workflow 链上的候选覆盖重叠应归入同一 pending 冗余。"""
    structure = _marginal_structure(_snapshot(), ("A", "C"))
    assert structure["marginal_changed_events"] > 0
    assert structure["same_pending_redundancy_events"] > 0
    assert structure["cross_pending_overlap_events"] == 0


def test_common_objective_is_identical_for_same_selected_set() -> None:
    """公共评分必须直接复用冻结 RQ3 objective。"""
    snapshot = _snapshot()
    selected, _ = _select_greedy(snapshot, RecoveryCostModel())
    first = evaluate_objective(snapshot, selected)
    second = evaluate_objective(snapshot, tuple(reversed(selected)))
    assert first.total_recovery_cost_ms == second.total_recovery_cost_ms


def test_aggregate_contains_all_variants_and_paired_effects() -> None:
    """聚合结果必须同时包含三种 variant 与两项 paired 对比。"""
    variants = {
        name: {
            "common_cost_ms": 1.0,
            "zero_current_marginal_count": 0,
            "selected_count": 1,
            "full_coverage": False,
        }
        for name in VARIANTS
    }
    comparisons = {
        name: {
            "ablation_minus_full_cost_ms": 0.0,
            "full_relative_reduction": 0.0,
            "outcome": "tie",
            "selected_set_different": False,
            "selection_order_different": False,
        }
        for name in ("without_position_aware", "without_set_dependent")
    }
    row = {
        "population": "OpenHands",
        "snapshot_id": "S",
        "budget_ratio": 0.25,
        "variants": variants,
        "comparisons": comparisons,
        "position_ranking": {
            "static_positive_ranking_different": False,
            "static_top_candidate_different": False,
        },
        "marginal_structure": {
            "candidate_stage_comparisons": 1,
            "marginal_changed_events": 0,
            "same_pending_redundancy_events": 0,
            "cross_pending_overlap_events": 0,
            "other_events": 0,
        },
    }
    aggregate = _aggregate_rows([row])
    entry = aggregate["OpenHands"]["budgets"]["25%"]
    assert set(entry["variants"]) == set(VARIANTS)
    assert entry["full_vs_without_position_aware"]["win_tie_loss"]["tie"] == 1
    assert entry["full_vs_without_set_dependent"]["selected_set_difference_rate"] == 0.0
