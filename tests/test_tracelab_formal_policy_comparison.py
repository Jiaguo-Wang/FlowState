from __future__ import annotations

from datetime import datetime, timedelta
import inspect
from pathlib import Path

import pytest

from evaluation.public_agent_trace.formal_policy_comparison import (
    BOOTSTRAP_SEED,
    BUDGET_RATIOS,
    EXPECTED_MAIN_SNAPSHOT_COUNT,
    EXPECTED_MAIN_X_HISTOGRAM,
    EXPECTED_SECONDARY_SNAPSHOT_COUNT,
    FrozenSnapshot,
    PolicyResult,
    bootstrap_flow_vs_marconi,
    evaluate_selection,
    improvement_distributions,
    load_frozen_protocol,
    paired_comparisons,
    select_checkpoint_ids,
)
from evaluation.public_agent_trace.tracelab_context_pressure import (
    analyze_snapshot,
)
from evaluation.public_agent_trace.tracelab_final_protocol import (
    build_final_snapshot_policy_metadata,
    demand_relative_budget,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    CompletedRoundFact,
    PendingRoundFact,
    build_trace_snapshot,
    validate_snapshot,
)
from flowstate.recovery_model import (
    FORMAL_RECOVERY_MODEL_METADATA,
    RecoveryCostModel,
)


def _frozen_snapshot() -> FrozenSnapshot:
    observed_at = datetime(2026, 8, 26, 12, 0, 0)
    workflows = ("W1", "W2")
    completed = []
    pending = []
    for workflow_index, workflow_id in enumerate(workflows, start=1):
        completed.extend(
            (
                CompletedRoundFact(
                    workflow_id=workflow_id,
                    round_pk=workflow_index * 100,
                    round_index=0,
                    run_position=0,
                    input_tokens_total=1_024 * workflow_index,
                    current_prefix_tokens=0,
                    known_at_time=observed_at - timedelta(seconds=2),
                ),
                CompletedRoundFact(
                    workflow_id=workflow_id,
                    round_pk=workflow_index * 100 + 1,
                    round_index=1,
                    run_position=1,
                    input_tokens_total=2_048 * workflow_index,
                    current_prefix_tokens=1_024 * workflow_index,
                    known_at_time=observed_at - timedelta(seconds=1),
                ),
            )
        )
        pending.append(
            PendingRoundFact(
                workflow_id=workflow_id,
                round_pk=workflow_index * 100 + 1,
                round_index=1,
                run_position=1,
                input_tokens_total=2_048 * workflow_index,
                current_prefix_tokens=1_024 * workflow_index,
                known_at_time=observed_at,
                observed_tool_call_ids=(f"tool-{workflow_index}",),
            )
        )
    snapshot = build_trace_snapshot(
        snapshot_id="synthetic-frozen",
        scale="Small",
        time_domain="claude",
        observed_at=observed_at,
        active_workflow_ids=workflows,
        completed_rounds=completed,
        pending_rounds=pending,
    )
    analysis = analyze_snapshot(snapshot)
    assert analysis["x"] == 2
    return FrozenSnapshot(
        cohort="main",
        source_row={"snapshot_id": snapshot.snapshot_id},
        snapshot=snapshot,
        exact_parent_count=2,
        policy_metadata=build_final_snapshot_policy_metadata(snapshot),
    )


def _result(
    snapshot_id: str,
    ratio: float,
    policy: str,
    cost: float,
) -> PolicyResult:
    return PolicyResult(
        cohort="main",
        snapshot_id=snapshot_id,
        provider="claude",
        concurrency_bucket="Small",
        x=2,
        candidate_count=4,
        pending_count=2,
        budget_ratio=ratio,
        budget_k=1,
        policy=policy,
        selected_checkpoint_ids=(f"{snapshot_id}-{policy}",),
        selected_count=1,
        total_recovery_gap_tokens=int(cost),
        mean_recovery_gap_tokens=cost / 2,
        total_formal_recovery_cost_ms=cost,
        mean_formal_recovery_cost_ms=cost / 2,
        executable_hit_count=0,
        executable_hit_ratio=0.0,
        selection_overhead_ms=0.0,
        continuation_results=(),
    )


def _comparison_rows(costs: dict[str, tuple[float, float]]) -> tuple[PolicyResult, ...]:
    rows = []
    for ratio in BUDGET_RATIOS:
        for snapshot_id, (marconi_cost, flow_cost) in costs.items():
            rows.extend(
                (
                    _result(snapshot_id, ratio, "Global-LRU", marconi_cost),
                    _result(snapshot_id, ratio, "KVFlow-style", marconi_cost),
                    _result(snapshot_id, ratio, "Marconi-style", marconi_cost),
                    _result(snapshot_id, ratio, "FlowState", flow_cost),
                )
            )
    return tuple(rows)


def test_frozen_protocol_has_exact_main_cohort() -> None:
    protocol = load_frozen_protocol()
    main = protocol["selected_main_snapshots"]
    observed: dict[int, int] = {}
    for row in main:
        observed[int(row["x"])] = observed.get(int(row["x"]), 0) + 1
    assert len(main) == EXPECTED_MAIN_SNAPSHOT_COUNT == 105
    assert observed == EXPECTED_MAIN_X_HISTOGRAM
    assert len(protocol["selected_secondary_snapshots"]) == (
        EXPECTED_SECONDARY_SNAPSHOT_COUNT
    )


def test_exact_budget_formula_and_x2_collapse() -> None:
    assert tuple(demand_relative_budget(2, ratio) for ratio in BUDGET_RATIOS) == (
        1,
        1,
        1,
        2,
    )
    assert demand_relative_budget(6, 0.25) == 1
    assert demand_relative_budget(6, 0.50) == 3
    assert demand_relative_budget(6, 0.75) == 4
    assert demand_relative_budget(6, 1.00) == 6


def test_frozen_policy_metadata_is_leakage_free() -> None:
    frozen = _frozen_snapshot()
    metadata = frozen.policy_metadata
    assert set(value for _, value in metadata.steps_to_execution_by_continuation) == {1}
    assert metadata.marconi_alpha == 1.0
    assert validate_snapshot(frozen.snapshot) == ()
    assert frozen.snapshot.future_prefix_used is False
    assert frozen.snapshot.runtime_residency_inferred is False
    assert frozen.snapshot.llm_level_branching_introduced is False


def test_baseline_selection_does_not_call_flowstate_optimizer() -> None:
    class RaisingOptimizer:
        def select(self, *args, **kwargs):
            raise AssertionError("baseline 不得调用正式 recovery optimizer")

    frozen = _frozen_snapshot()
    optimizer = RaisingOptimizer()
    for policy in ("Global-LRU", "KVFlow-style", "Marconi-style"):
        selected = select_checkpoint_ids(policy, frozen, 1, optimizer)  # type: ignore[arg-type]
        assert len(selected) == 1
    with pytest.raises(AssertionError):
        select_checkpoint_ids("FlowState", frozen, 1, optimizer)  # type: ignore[arg-type]


def test_all_policies_use_same_formal_evaluator() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def estimate(self, gap_tokens: int, target_tokens: int) -> float:
            self.calls.append((gap_tokens, target_tokens))
            return float(gap_tokens + target_tokens)

    frozen = _frozen_snapshot()
    selected = ()
    totals = []
    for policy in ("Global-LRU", "KVFlow-style", "Marconi-style", "FlowState"):
        model = RecordingModel()
        result = evaluate_selection(
            frozen,
            0.25,
            1,
            policy,
            selected,
            model,  # type: ignore[arg-type]
        )
        totals.append(result.total_formal_recovery_cost_ms)
        assert model.calls == [(2_048, 2_048), (4_096, 4_096)]
    assert len(set(totals)) == 1


def test_formal_model_is_position_aware_and_in_domain() -> None:
    model = RecoveryCostModel()
    assert FORMAL_RECOVERY_MODEL_METADATA.name == "position_aware_quadratic_v1"
    assert model.estimate(4_096, 32_768) != model.estimate(4_096, 65_536)
    assert model.estimate(0, 131_072) == 0.0


def test_paired_comparison_counts_win_tie_and_loss() -> None:
    rows = _comparison_rows(
        {
            "win": (10.0, 5.0),
            "tie": (7.0, 7.0),
            "loss": (4.0, 6.0),
        }
    )
    paired = paired_comparisons(rows)
    item = next(
        row
        for row in paired
        if row["budget_ratio"] == 0.25
        and row["baseline"] == "Marconi-style"
    )
    assert (item["win_count"], item["tie_count"], item["loss_count"]) == (
        1,
        1,
        1,
    )


def test_improvement_distribution_handles_zero_costs() -> None:
    rows = _comparison_rows(
        {
            "both-zero": (0.0, 0.0),
            "bad-zero": (0.0, 1.0),
            "flow-zero": (2.0, 0.0),
            "normal": (4.0, 2.0),
        }
    )
    item = improvement_distributions(rows)[0]
    assert item["both_zero"] == 1
    assert item["baseline_zero_flow_nonzero"] == 1
    assert item["baseline_nonzero_flow_zero"] == 1
    assert item["defined_snapshot_count"] == 2


def test_bootstrap_is_deterministic_and_uses_snapshot_unit() -> None:
    rows = _comparison_rows(
        {"a": (10.0, 5.0), "b": (20.0, 10.0), "c": (30.0, 15.0)}
    )
    first = bootstrap_flow_vs_marconi(rows, seed=BOOTSTRAP_SEED, iterations=100)
    second = bootstrap_flow_vs_marconi(rows, seed=BOOTSTRAP_SEED, iterations=100)
    assert first == second
    assert first["sampling_unit"] == "snapshot"
    assert first["iterations"] == 100


def test_secondary_slice_is_exactly_frozen_x4_set() -> None:
    protocol = load_frozen_protocol()
    secondary = protocol["selected_secondary_snapshots"]
    assert len(secondary) == 37
    assert all(int(row["x"]) >= 4 for row in secondary)


def test_evaluator_has_no_gpu_or_sglang_dependency() -> None:
    from evaluation.public_agent_trace import formal_policy_comparison

    source = inspect.getsource(formal_policy_comparison)
    assert "import torch" not in source
    assert "import sglang" not in source
    assert "SGLangAdapter" not in source
    assert "cuda(" not in source


def test_formal_protocol_file_is_external_to_result_directory() -> None:
    protocol = load_frozen_protocol()
    assert protocol["execution"]["gpu_executed"] is False
    assert protocol["validation"]["future_field_leakage_violations"] == 0
    assert Path(
        "evaluation/public_agent_trace/tracelab_nontrivial_protocol.json"
    ).is_file()
