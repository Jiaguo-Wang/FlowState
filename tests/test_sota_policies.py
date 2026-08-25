from __future__ import annotations

from dataclasses import replace
import math

import pytest

import evaluation.sota_policies as policies_module
from evaluation.sota_policies import (
    KVFlowStylePolicy,
    MarconiStylePolicy,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


def _continuation(
    continuation_id: str,
    *,
    workflow_id: str = "W",
    lineage_path: tuple[str, ...] = ("P", "B"),
    target: int = 100,
) -> PendingContinuation:
    return PendingContinuation(
        continuation_id=continuation_id,
        workflow_id=workflow_id,
        lineage_path=lineage_path,
        anchor_pos=target,
        resident_fa_frontier=target,
    )


def _candidate(
    checkpoint_id: str,
    *,
    workflow_id: str = "W",
    lineage_path: tuple[str, ...] = ("P",),
    token_pos: int = 50,
    memory_bytes: int = 100,
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


def test_kvflow_prefers_smaller_steps_and_reports_result_metadata() -> None:
    continuations = (
        _continuation("soon", workflow_id="WA"),
        _continuation("later", workflow_id="WB"),
    )
    candidates = (
        _candidate("A", workflow_id="WA"),
        _candidate("B", workflow_id="WB"),
    )

    result = KVFlowStylePolicy().select(
        continuations,
        candidates,
        budget_k=1,
        steps_to_execution_by_continuation={"soon": 1, "later": 5},
        last_access_by_checkpoint={"A": 1.0, "B": 100.0},
    )

    assert result.policy_name == "KVFlow-style"
    assert result.selected_checkpoint_ids == ("A",)
    assert result.budget_k == 1


def test_kvflow_shared_checkpoint_uses_minimum_steps() -> None:
    continuations = (
        _continuation("B1", lineage_path=("P", "B1")),
        _continuation("B2", lineage_path=("P", "B2")),
    )

    priority = KVFlowStylePolicy().priority(
        _candidate("PARENT"),
        continuations,
        {"B1": 4, "B2": 1},
    )

    assert priority == 1


def test_kvflow_checkpoint_without_future_dependency_has_infinite_priority() -> None:
    priority = KVFlowStylePolicy().priority(
        _candidate("OTHER", workflow_id="OTHER"),
        (_continuation("B1"),),
        {"B1": 1},
    )

    assert math.isinf(priority)


def test_kvflow_tie_break_residency_capacity_and_order_invariance() -> None:
    continuation = _continuation("B1")
    candidates = (
        _candidate("B"),
        _candidate("A"),
        _candidate("NONRESIDENT", recurrent_resident=False),
    )
    snapshot = tuple(replace(candidate) for candidate in candidates)
    policy = KVFlowStylePolicy()
    metadata = {"B1": 2}
    recency = {"A": 1.0, "B": 1.0}

    normal = policy.select(
        (continuation,),
        candidates,
        5,
        metadata,
        recency,
    )
    reversed_result = policy.select(
        (continuation,),
        tuple(reversed(candidates)),
        5,
        metadata,
        recency,
    )

    assert normal.selected_checkpoint_ids == ("A", "B")
    assert reversed_result.selected_checkpoint_ids == normal.selected_checkpoint_ids
    assert len(set(normal.selected_checkpoint_ids)) == 2
    assert candidates == snapshot


def test_kvflow_equal_priority_prefers_newer_checkpoint() -> None:
    continuation = _continuation("B1")
    candidates = (_candidate("A_OLD"), _candidate("Z_NEW"))

    result = KVFlowStylePolicy().select(
        (continuation,),
        candidates,
        1,
        {"B1": 1},
        {"A_OLD": 1.0, "Z_NEW": 2.0},
    )

    assert result.selected_checkpoint_ids == ("Z_NEW",)


def test_kvflow_includes_no_dependency_candidate_only_after_useful_one() -> None:
    continuation = _continuation("B1")
    candidates = (
        _candidate("A_UNUSED", workflow_id="OTHER"),
        _candidate("Z_USEFUL"),
    )

    result = KVFlowStylePolicy().select(
        (continuation,),
        candidates,
        2,
        {"B1": 3},
        {"A_UNUSED": 100.0, "Z_USEFUL": 1.0},
    )

    assert result.selected_checkpoint_ids == ("Z_USEFUL", "A_UNUSED")


def test_kvflow_rejects_missing_or_invalid_steps_metadata() -> None:
    continuation = _continuation("B1")
    policy = KVFlowStylePolicy()
    candidate = _candidate("A")

    with pytest.raises(ValueError, match="B1"):
        policy.select((continuation,), (candidate,), 1, {}, {"A": 1.0})
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="steps-to-execution"):
            policy.select(
                (continuation,),
                (candidate,),
                1,
                {"B1": invalid},
                {"A": 1.0},
            )


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_kvflow_rejects_missing_or_nonfinite_recency(invalid: float) -> None:
    continuation = _continuation("B1")
    candidate = _candidate("A")
    policy = KVFlowStylePolicy()

    with pytest.raises(ValueError, match="last_access"):
        policy.select((continuation,), (candidate,), 1, {"B1": 1}, {})
    with pytest.raises(ValueError, match="有限数值"):
        policy.select(
            (continuation,),
            (candidate,),
            1,
            {"B1": 1},
            {"A": invalid},
        )


def test_kvflow_does_not_use_flowstate_recovery_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        """在策略访问禁用目标时立即使测试失败。"""
        raise AssertionError("KVFlow-style 不得调用 FlowState recovery objective")

    for name in (
        "RecoveryCostModel",
        "recovery_gap",
        "executable_frontier",
        "GlobalOptimizer",
        "marginal_gain",
    ):
        monkeypatch.setattr(policies_module, name, forbidden, raising=False)

    result = KVFlowStylePolicy().select(
        (_continuation("B1"),),
        (_candidate("A"),),
        1,
        {"B1": 1},
        {"A": 1.0},
    )

    assert result.selected_checkpoint_ids == ("A",)


def test_kvflow_does_not_score_token_depth_or_coverage_count() -> None:
    continuations = (
        _continuation("A1", workflow_id="WA"),
        _continuation("A2", workflow_id="WA"),
        _continuation("B1", workflow_id="WB"),
    )
    candidates = (
        _candidate("Z_MANY", workflow_id="WA", token_pos=1),
        _candidate("A_ONE", workflow_id="WB", token_pos=99),
    )
    steps = {"A1": 3, "A2": 3, "B1": 3}

    result = KVFlowStylePolicy().select(
        continuations,
        candidates,
        1,
        steps,
        {"Z_MANY": 1.0, "A_ONE": 1.0},
    )

    assert result.selected_checkpoint_ids == ("A_ONE",)


def test_kvflow_reuses_core_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = policies_module.is_compatible
    calls = 0

    def tracked_compatibility(checkpoint, continuation):
        """记录策略对核心兼容性函数的直接调用。"""
        nonlocal calls
        calls += 1
        return original(checkpoint, continuation)

    monkeypatch.setattr(
        policies_module,
        "is_compatible",
        tracked_compatibility,
    )

    KVFlowStylePolicy().select(
        (_continuation("B1"),),
        (_candidate("A"),),
        1,
        {"B1": 1},
        {"A": 1.0},
    )

    assert calls == 1


def test_marconi_alpha_zero_strictly_reduces_to_recency() -> None:
    candidates = (_candidate("OLD"), _candidate("NEW"))

    result = MarconiStylePolicy().select(
        candidates,
        1,
        {"OLD": 1.0, "NEW": 2.0},
        {"OLD": 1_000.0, "NEW": 0.0},
        alpha=0.0,
    )

    assert result.policy_name == "Marconi-style"
    assert result.selected_checkpoint_ids == ("NEW",)
    assert result.budget_k == 1


def test_marconi_equal_recency_prefers_higher_flop_efficiency() -> None:
    candidates = (_candidate("A"), _candidate("B"))

    result = MarconiStylePolicy().select(
        candidates,
        1,
        {"A": 1.0, "B": 1.0},
        {"A": 20.0, "B": 10.0},
        alpha=1.0,
    )

    assert result.selected_checkpoint_ids == ("A",)


def test_marconi_equal_flop_saved_prefers_smaller_memory() -> None:
    candidates = (
        _candidate("A", memory_bytes=50),
        _candidate("B", memory_bytes=100),
    )

    result = MarconiStylePolicy().select(
        candidates,
        1,
        {"A": 1.0, "B": 1.0},
        {"A": 20.0, "B": 20.0},
        alpha=1.0,
    )

    assert result.selected_checkpoint_ids == ("A",)


def test_marconi_equal_dimensions_normalize_to_zero_and_tie_by_id() -> None:
    candidates = (_candidate("B"), _candidate("A"))

    result = MarconiStylePolicy().select(
        candidates,
        2,
        {"A": 7.0, "B": 7.0},
        {"A": 30.0, "B": 30.0},
        alpha=4.0,
    )

    assert result.selected_checkpoint_ids == ("A", "B")


def test_marconi_equal_flop_efficiency_does_not_create_difference() -> None:
    candidates = (
        _candidate("A", memory_bytes=50),
        _candidate("B", memory_bytes=100),
    )

    result = MarconiStylePolicy().select(
        candidates,
        1,
        {"A": 1.0, "B": 2.0},
        {"A": 10.0, "B": 20.0},
        alpha=10.0,
    )

    assert result.selected_checkpoint_ids == ("B",)


def test_marconi_nonresident_does_not_join_normalization_or_need_metadata() -> None:
    candidates = (
        _candidate("A"),
        _candidate("NONRESIDENT", recurrent_resident=False),
    )

    result = MarconiStylePolicy().select(
        candidates,
        2,
        {"A": 1.0},
        {"A": 1.0},
        alpha=1.0,
    )

    assert result.selected_checkpoint_ids == ("A",)


@pytest.mark.parametrize(
    ("last_access", "flop_saved", "message"),
    (
        ({}, {"A": 1.0}, "last_access"),
        ({"A": 1.0}, {}, "flop_saved"),
    ),
)
def test_marconi_rejects_missing_metadata(
    last_access: dict[str, float],
    flop_saved: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MarconiStylePolicy().select(
            (_candidate("A"),),
            1,
            last_access,
            flop_saved,
            alpha=1.0,
        )


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_marconi_rejects_nonfinite_recency(invalid: float) -> None:
    with pytest.raises(ValueError, match="有限数值"):
        MarconiStylePolicy().select(
            (_candidate("A"),),
            1,
            {"A": invalid},
            {"A": 1.0},
            alpha=1.0,
        )


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf, -1.0))
def test_marconi_rejects_invalid_flop_saved(invalid: float) -> None:
    with pytest.raises(ValueError):
        MarconiStylePolicy().select(
            (_candidate("A"),),
            1,
            {"A": 1.0},
            {"A": invalid},
            alpha=1.0,
        )


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf, -1.0))
def test_marconi_rejects_invalid_alpha(invalid: float) -> None:
    with pytest.raises(ValueError):
        MarconiStylePolicy().select(
            (_candidate("A"),),
            1,
            {"A": 1.0},
            {"A": 1.0},
            alpha=invalid,
        )


def test_marconi_rejects_nonpositive_memory() -> None:
    candidate = _candidate("A")
    object.__setattr__(candidate, "memory_bytes", 0)

    with pytest.raises(ValueError, match="memory_bytes"):
        MarconiStylePolicy().select(
            (candidate,),
            1,
            {"A": 1.0},
            {"A": 1.0},
            alpha=1.0,
        )


def test_marconi_does_not_use_flowstate_recovery_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        """在策略访问禁用目标时立即使测试失败。"""
        raise AssertionError("Marconi-style 不得调用 FlowState recovery objective")

    for name in (
        "RecoveryCostModel",
        "recovery_gap",
        "executable_frontier",
        "GlobalOptimizer",
        "marginal_gain",
    ):
        monkeypatch.setattr(policies_module, name, forbidden, raising=False)

    result = MarconiStylePolicy().select(
        (_candidate("A"),),
        1,
        {"A": 1.0},
        {"A": 1.0},
        alpha=1.0,
    )

    assert result.selected_checkpoint_ids == ("A",)


def test_marconi_tie_break_order_invariance_and_input_immutability() -> None:
    candidates = (_candidate("C"), _candidate("A"), _candidate("B"))
    snapshot = tuple(replace(candidate) for candidate in candidates)
    metadata = {candidate.checkpoint_id: 1.0 for candidate in candidates}
    policy = MarconiStylePolicy()

    normal = policy.select(candidates, 2, metadata, metadata, alpha=1.0)
    reversed_result = policy.select(
        tuple(reversed(candidates)),
        2,
        metadata,
        metadata,
        alpha=1.0,
    )

    assert normal.selected_checkpoint_ids == ("A", "B")
    assert reversed_result.selected_checkpoint_ids == normal.selected_checkpoint_ids
    assert candidates == snapshot


@pytest.mark.parametrize("policy_name", ("kvflow", "marconi"))
def test_both_policies_return_empty_for_zero_budget(policy_name: str) -> None:
    candidate = _candidate("A")
    if policy_name == "kvflow":
        result = KVFlowStylePolicy().select(
            (_continuation("B1"),),
            (candidate,),
            0,
            {"B1": 1},
            {"A": 1.0},
        )
    else:
        result = MarconiStylePolicy().select(
            (candidate,),
            0,
            {"A": 1.0},
            {"A": 1.0},
            alpha=1.0,
        )

    assert result.selected_checkpoint_ids == ()


@pytest.mark.parametrize("invalid", (-1, 1.5, True))
def test_both_policies_reject_invalid_budget(invalid: object) -> None:
    with pytest.raises(ValueError, match="budget_k"):
        KVFlowStylePolicy().select((), (), invalid, {}, {})
    with pytest.raises(ValueError, match="budget_k"):
        MarconiStylePolicy().select((), invalid, {}, {}, alpha=0.0)


def test_both_policies_reject_duplicate_checkpoint_ids() -> None:
    candidates = (_candidate("A"), _candidate("A"))

    with pytest.raises(ValueError, match="A"):
        KVFlowStylePolicy().select((), candidates, 1, {}, {})
    with pytest.raises(ValueError, match="A"):
        MarconiStylePolicy().select(
            candidates,
            1,
            {"A": 1.0},
            {"A": 1.0},
            alpha=1.0,
        )
