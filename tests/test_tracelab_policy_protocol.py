"""验证 TraceLab 策略评估协议冻结结果。"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

from evaluation.public_agent_trace.tracelab_policy_protocol import (
    MARCONI_ALPHA,
    PROFILER_MAX_GAP_TOKENS,
    ProtocolSnapshotEvent,
    aggregate_protocol,
    analyze_snapshot_structure,
    build_snapshot_policy_metadata,
    choose_protocol_events,
    concurrency_scale,
    exact_parent_ids,
    immediate_ancestor_gap,
    is_profiler_supported,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    CompletedRoundFact,
    PendingRoundFact,
    build_trace_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_policy_protocol.json"
)
SOURCE_PATH = PROTOCOL_PATH.with_suffix(".py")


def _snapshot():
    """构造只含当前与历史事实的最小线性快照。"""
    observed_at = datetime(2026, 8, 26, 12, 0, 0)
    workflow_id = "claude:session-1:run:000001"
    completed = tuple(
        CompletedRoundFact(
            workflow_id=workflow_id,
            round_pk=index + 1,
            round_index=index,
            run_position=index,
            input_tokens_total=4096 * (index + 1),
            current_prefix_tokens=0,
            known_at_time=observed_at - timedelta(seconds=3 - index),
        )
        for index in range(3)
    )
    pending = (
        PendingRoundFact(
            workflow_id=workflow_id,
            round_pk=3,
            round_index=2,
            run_position=2,
            input_tokens_total=12_288,
            current_prefix_tokens=0,
            known_at_time=observed_at,
            observed_tool_call_ids=("tool-1", "tool-2"),
        ),
    )
    return build_trace_snapshot(
        snapshot_id="unit-small-claude",
        scale="Small",
        time_domain="claude",
        observed_at=observed_at,
        active_workflow_ids=(workflow_id,),
        completed_rounds=completed,
        pending_rounds=pending,
    )


def _event(
    *,
    provider: str,
    scale: str,
    round_pk: int,
) -> ProtocolSnapshotEvent:
    """构造确定性分层采样事件。"""
    return ProtocolSnapshotEvent(
        snapshot_id=f"{provider}-{scale}-{round_pk}",
        scale=scale,
        provider=provider,
        trigger_session_id=f"session-{round_pk}",
        trigger_run_ordinal=1,
        trigger_round_pk=round_pk,
        observed_at=datetime(2026, 8, 26, 12, 0, round_pk),
        active_run_count=2 if scale == "Small" else 6,
    )


def test_profiler_supported_boundary_is_frozen_at_32k() -> None:
    """有效域必须包含 32K 边界并排除更长 run。"""
    assert PROFILER_MAX_GAP_TOKENS == 32_768
    assert is_profiler_supported(0)
    assert is_profiler_supported(32_768)
    assert not is_profiler_supported(32_769)
    with pytest.raises(ValueError, match="必须非负"):
        is_profiler_supported(-1)


@pytest.mark.parametrize(
    ("active_runs", "expected"),
    (
        (0, None),
        (1, None),
        (2, "Small"),
        (4, "Small"),
        (5, "Medium"),
        (8, "Medium"),
        (9, "Large"),
    ),
)
def test_concurrency_scale_boundaries(
    active_runs: int,
    expected: str | None,
) -> None:
    """并发层边界必须与预注册协议一致。"""
    assert concurrency_scale(active_runs) == expected


def test_protocol_sampling_is_stratified_and_order_invariant() -> None:
    """每个非空 provider×scale 层只保留一个稳定事件。"""
    events = (
        _event(provider="claude", scale="Small", round_pk=1),
        _event(provider="claude", scale="Small", round_pk=2),
        _event(provider="codex", scale="Small", round_pk=3),
        _event(provider="claude", scale="Medium", round_pk=4),
    )
    normal = choose_protocol_events(events)
    reversed_result = choose_protocol_events(reversed(events))
    assert normal == reversed_result
    assert len(normal) == 3
    assert len({(item.provider, item.scale) for item in normal}) == 3


def test_policy_metadata_uses_only_frozen_current_history_fields() -> None:
    """KVFlow 与 Marconi 元数据必须只由当前和历史快照生成。"""
    snapshot = _snapshot()
    metadata = build_snapshot_policy_metadata(snapshot)

    assert metadata.marconi_alpha == MARCONI_ALPHA == 1.0
    assert metadata.steps_to_execution_by_continuation == (
        (snapshot.continuations[0].continuation_id, 1),
    )
    assert tuple(
        item.last_access_order for item in metadata.checkpoint_recency
    ) == (1, 2, 3)
    assert metadata.last_access_by_checkpoint == tuple(
        (item.checkpoint_id, float(item.last_access_order))
        for item in metadata.checkpoint_recency
    )
    assert tuple(
        value for _, value in metadata.marconi_flop_saved_by_checkpoint
    ) == (4096.0, 4096.0, 4096.0)


def test_exact_parent_and_immediate_ancestor_are_structural() -> None:
    """父状态可用性与 spacing 必须只由线性结构和 token 位置决定。"""
    snapshot = _snapshot()
    continuation = snapshot.continuations[0]
    assert exact_parent_ids(continuation, snapshot.candidates) == (
        snapshot.candidates[2].checkpoint_id,
    )
    assert immediate_ancestor_gap(continuation, snapshot.candidates) == 4096


def test_budget_structure_does_not_execute_selection() -> None:
    """结构分析只计算预算与 coverage，不产生 policy selection。"""
    snapshot = _snapshot()
    metadata = build_snapshot_policy_metadata(snapshot)
    analysis = analyze_snapshot_structure(snapshot, metadata)
    summary = aggregate_protocol((snapshot,), (analysis,))

    assert [row["k"] for row in analysis["budgets"]] == [1, 1, 2]
    assert analysis["exact_parent_available_count"] == 1
    assert analysis["distinct_exact_parent_count"] == 1
    assert summary["exact_parent_availability_fraction"] == 1.0
    assert summary["budget_contention"]["25%"]["k_ge_p_fraction"] == 1.0


def test_frozen_artifact_records_real_cohort_and_zero_policy_runs() -> None:
    """冻结 artifact 必须保存真实 cohort、空层与零策略执行证据。"""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    statistics = protocol["profiler_supported_cohort"]["statistics"]

    assert statistics["全部"]["run_count"] == 3196
    assert statistics["claude"]["run_count"] == 638
    assert statistics["codex"]["run_count"] == 2558
    assert protocol["summary"]["snapshot_count"] == 3
    assert protocol["summary"]["scale_counts"] == {
        "Small": 2,
        "Medium": 1,
        "Large": 0,
    }
    assert len(protocol["sampling_protocol"]["empty_strata"]) == 3
    assert protocol["validation"]["formal_policy_runs"] == 0
    assert protocol["validation"]["phi_calls"] == 0
    assert protocol["validation"]["oracle_runs"] == 0


def test_frozen_artifact_preserves_no_leakage_and_gate_results() -> None:
    """正式协议必须冻结无泄漏元数据与诚实的 WEAK gate。"""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validation = protocol["validation"]
    gates = protocol["gates"]

    assert validation["profiler_domain_violations"] == 0
    assert validation["flowstate_leakage_violations"] == 0
    assert validation["exact_parent_missing"] == 0
    assert gates == {
        "profiler_supported_cohort": "PASS",
        "non_trivial_state_contention": "WEAK",
        "kvflow_metadata_well_defined": "WEAK",
        "marconi_metadata_well_defined": "PASS",
        "flowstate_metadata_leakage_free": "PASS",
        "ready_for_step_10d": "WEAK",
    }


def test_protocol_implementation_contains_no_policy_or_phi_execution() -> None:
    """协议实现不得导入或调用正式优化器、成本模型和策略选择。"""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "RecoveryCostModel",
        "GlobalOptimizer",
        "KVFlowStylePolicy",
        "MarconiStylePolicy",
        ".select(",
    )
    assert all(fragment not in source for fragment in forbidden)

