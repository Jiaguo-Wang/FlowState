"""验证 TraceLab 最终离线评估协议与冻结 artifact。"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import pytest

from evaluation.public_agent_trace.tracelab_context_pressure import (
    ContextSnapshotEvent,
)
from evaluation.public_agent_trace.tracelab_final_protocol import (
    DEMAND_RETENTION_RATIOS,
    MAIN_COHORT,
    MAIN_COHORT_MAX_TOKENS,
    MARCONI_ALPHA,
    MAX_SNAPSHOTS_PER_STRATUM,
    RankedSnapshot,
    build_final_snapshot_policy_metadata,
    choose_unique_ranked_snapshots,
    demand_relative_budget,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    CompletedRoundFact,
    PendingRoundFact,
    SampledSnapshotEvent,
    TraceSnapshot,
    build_trace_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_final_protocol.json"
)
REPORT_PATH = ARTIFACT_PATH.with_name("TRACELAB_FINAL_PROTOCOL.md")
SOURCE_PATH = ARTIFACT_PATH.with_suffix(".py")
PROTECTED_HASHES = {
    "evaluation/controlled_multiworkflow_v1/policies.py": (
        "8df5a1391b651f3a55090e13b8abb9d2a520de0a94abeb6a7339fdcb49445a24"
    ),
    "evaluation/sota_policies.py": (
        "b276aff22d2dc1adcdb33b15a7a94dc608fa916789ba0f2e5d5fbe0b3189d212"
    ),
    "evaluation/sota_metadata.py": (
        "df6582dd9a5dd15e984e9cefdd899e1d0b8bc9292399e7862608d04162a283c2"
    ),
    "evaluation/public_agent_trace/tracelab_context_pressure.py": (
        "0375b757797a61bc427424b5cb24c8dee44cadb91b2ff9f36226d2680857d9c2"
    ),
    "evaluation/public_agent_trace/tracelab_context_pressure.json": (
        "a40a357a8179cbf5cf876c14afa15ebe5f46c8b70a6d681c41fef5f6384cff72"
    ),
    "evaluation/public_agent_trace/tracelab_policy_protocol.py": (
        "757e3740cd442787a8cfa1c178ce640d9b7ab99a82ee0cd73a0445d0222f8c5d"
    ),
    "evaluation/public_agent_trace/tracelab_policy_protocol.json": (
        "82cd75ff0d8d80fa9eba2dc80909bf5fd0f432402e1e03e0fb848ff3a197e553"
    ),
    "evaluation/public_agent_trace/tracelab_to_flowstate.py": (
        "445b8b1064449a76c9c2bfba9b3710af90f10853bb46c34aec9af7673d9de2e0"
    ),
    "evaluation/public_agent_trace/tracelab_workload.json": (
        "a2c77cedac796e97dde1120ed3a002d83dfd51ae7070337878d1f89d8634bba1"
    ),
    "flowstate/recovery_model.py": (
        "9a13bc4f7778b9e1835ddb04237d54815ff86c7e9c57b42d293e73c5bb404082"
    ),
    "motivation/README.md": (
        "a066a70f1fb13bba472147fc6847ec8b80f6d7dd8d02fa3d698677abced659a8"
    ),
}


def _artifact() -> dict:
    """读取冻结协议 artifact。"""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _empty_snapshot(
    snapshot_id: str,
    provider: str,
    scale: str,
    active_ids: tuple[str, ...],
) -> TraceSnapshot:
    """构造仅用于采样与 active-set 去重的最小快照。"""
    return TraceSnapshot(
        snapshot_id=snapshot_id,
        scale=scale,
        time_domain=provider,
        observed_at=datetime(2026, 8, 26),
        active_workflow_ids=active_ids,
        candidates=(),
        continuations=(),
        checkpoint_metadata=(),
        continuation_metadata=(),
        retention_budgets=(),
    )


def _ranked(
    *,
    snapshot_id: str,
    rank_key: str,
    provider: str = "claude",
    scale: str = "Small",
    bucket: str = "<=32K",
    active_ids: tuple[str, ...],
    round_pk: int,
) -> RankedSnapshot:
    """构造带固定 rank 的采样候选。"""
    event = ContextSnapshotEvent(
        cohort=MAIN_COHORT,
        cutoff_tokens=MAIN_COHORT_MAX_TOKENS,
        event=SampledSnapshotEvent(
            snapshot_id=snapshot_id,
            scale=scale,
            provider=provider,
            context_bucket=bucket,
            trigger_session_id=f"session-{round_pk}",
            trigger_run_ordinal=1,
            trigger_round_pk=round_pk,
            observed_at=datetime(2026, 8, 26),
            trace_observed_active_runs=len(active_ids),
        ),
    )
    return RankedSnapshot(
        event=event,
        snapshot=_empty_snapshot(
            snapshot_id,
            provider,
            scale,
            active_ids,
        ),
        rank_key=rank_key,
    )


@pytest.mark.parametrize(
    ("x_value", "ratio", "expected"),
    (
        (0, 0.25, 1),
        (1, 0.25, 1),
        (5, 0.25, 1),
        (5, 0.50, 2),
        (5, 0.75, 3),
        (5, 1.00, 5),
    ),
)
def test_demand_relative_budget_uses_exact_parent_count(
    x_value: int,
    ratio: float,
    expected: int,
) -> None:
    """预算必须只按 X 向下取整并保留一的下限。"""
    assert demand_relative_budget(x_value, ratio) == expected


def test_sampling_is_order_invariant_and_deduplicates_active_sets() -> None:
    """哈希排名较前的重复 active set 必须唯一保留。"""
    records = (
        _ranked(
            snapshot_id="late-duplicate",
            rank_key="b",
            active_ids=("W1", "W2"),
            round_pk=1,
        ),
        _ranked(
            snapshot_id="early-duplicate",
            rank_key="a",
            active_ids=("W2", "W1"),
            round_pk=2,
        ),
        _ranked(
            snapshot_id="unique",
            rank_key="c",
            active_ids=("W3", "W4"),
            round_pk=3,
        ),
    )
    normal = choose_unique_ranked_snapshots(records)
    reversed_result = choose_unique_ranked_snapshots(tuple(reversed(records)))

    assert normal == reversed_result
    assert {item.snapshot.snapshot_id for item in normal} == {
        "early-duplicate",
        "unique",
    }


def test_sampling_enforces_at_most_five_per_stratum() -> None:
    """一个非空 stratum 不得超过冻结的五个快照。"""
    records = tuple(
        _ranked(
            snapshot_id=f"snapshot-{index}",
            rank_key=f"{index:02d}",
            active_ids=(f"W{index}", f"X{index}"),
            round_pk=index,
        )
        for index in range(8)
    )
    selected = choose_unique_ranked_snapshots(records)
    assert len(selected) == MAX_SNAPSHOTS_PER_STRATUM == 5


def test_trace_zero_token_checkpoint_is_preserved_without_pruning() -> None:
    """原始零 token checkpoint 应保留，并获得零增量 FLOP proxy。"""
    observed_at = datetime(2026, 8, 26, 12, 0, 0)
    workflow_id = "claude:session:run:000001"
    snapshot = build_trace_snapshot(
        snapshot_id="zero-token",
        scale="Small",
        time_domain="claude",
        observed_at=observed_at,
        active_workflow_ids=(workflow_id,),
        completed_rounds=(
            CompletedRoundFact(
                workflow_id=workflow_id,
                round_pk=1,
                round_index=0,
                run_position=0,
                input_tokens_total=0,
                current_prefix_tokens=0,
                known_at_time=observed_at - timedelta(seconds=1),
            ),
        ),
        pending_rounds=(
            PendingRoundFact(
                workflow_id=workflow_id,
                round_pk=1,
                round_index=0,
                run_position=0,
                input_tokens_total=0,
                current_prefix_tokens=0,
                known_at_time=observed_at,
                observed_tool_call_ids=("tool",),
            ),
        ),
    )
    metadata = build_final_snapshot_policy_metadata(snapshot)

    assert len(snapshot.candidates) == 1
    assert metadata.marconi_flop_saved_by_checkpoint == (
        (snapshot.candidates[0].checkpoint_id, 0.0),
    )
    assert metadata.steps_to_execution_by_continuation == (
        (snapshot.continuations[0].continuation_id, 1),
    )


def test_frozen_artifact_records_c128_rationale_and_dense_sampling() -> None:
    """Artifact 必须固定 C128、密集采样与 X=0 的独立角色。"""
    artifact = _artifact()
    main = artifact["main_cohort"]
    summary = artifact["summary"]

    assert main["name"] == MAIN_COHORT == "C128"
    assert main["maximum_input_tokens_total"] == 131_072
    assert main["eligible_runs"] == 27_888
    assert main["frozen_before_policy_performance"] is True
    assert any("162.401%" in item for item in main["rationale"])
    assert any("120.690%" in item for item in main["rationale"])
    assert summary["snapshot_counts"] == {
        "total": 57,
        "by_scale": {"Small": 29, "Medium": 23, "Large": 5},
        "by_provider": {"claude": 34, "codex": 23},
    }
    assert artifact["demand_filter"]["characterization_snapshot_count"] == 59
    assert artifact["demand_filter"]["x_zero_snapshots_excluded"] == 2
    assert summary["unique_active_runs"] == 165
    assert summary["duplicate_active_run_set_count"] == 0


def test_frozen_snapshots_satisfy_demand_and_budget_invariants() -> None:
    """所有正式快照必须 X>0、X=P、无重复且预算严格来自 X。"""
    artifact = _artifact()
    seen_active_sets = set()
    stratum_counts: dict[tuple[str, str, str], int] = {}

    for row in artifact["snapshots"]:
        structure = row["structure"]
        sampling = row["sampling"]
        event = sampling["event"]["event"]
        signature = tuple(sampling["active_run_set"])
        stratum = (
            event["provider"],
            event["context_bucket"],
            event["scale"],
        )
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        assert signature not in seen_active_sets
        seen_active_sets.add(signature)
        assert structure["x"] > 0
        assert structure["x"] == structure["p"]
        assert structure["n"] == len(row["snapshot"]["candidates"])
        assert all(
            budget["k"]
            == max(1, math.floor(structure["x"] * budget["ratio"]))
            for budget in row["demand_relative_budgets"]
        )
        assert tuple(
            budget["ratio"] for budget in row["demand_relative_budgets"]
        ) == DEMAND_RETENTION_RATIOS
    assert max(stratum_counts.values()) <= MAX_SNAPSHOTS_PER_STRATUM


def test_policy_metadata_and_no_leakage_gates_are_frozen() -> None:
    """STE、alpha、无泄漏与 policy-not-run 证据必须完整。"""
    artifact = _artifact()
    validation = artifact["validation"]
    gates = artifact["gates"]

    for row in artifact["snapshots"]:
        metadata = row["policy_metadata"]
        snapshot = row["snapshot"]
        assert metadata["marconi_alpha"] == MARCONI_ALPHA == 1.0
        assert all(
            value == 1
            for _, value in metadata["steps_to_execution_by_continuation"]
        )
        assert snapshot["future_prefix_used"] is False
        assert snapshot["runtime_residency_inferred"] is False
        assert snapshot["llm_level_branching_introduced"] is False
    assert validation["future_field_leakage_violations"] == 0
    assert validation["llm_level_branching_violations"] == 0
    assert validation["duplicate_active_run_set_count"] == 0
    assert validation["x_zero_violations"] == 0
    assert artifact["execution"] == {
        "policy_comparison_executed": False,
        "phi_called": False,
        "gpu_executed": False,
    }
    assert gates == {
        "cohort_frozen": "PASS",
        "sampling_frozen": "PASS",
        "demand_relative_budget_frozen": "PASS",
        "policy_metadata_frozen": "PASS",
        "ready_for_profiler_extension": "PASS",
        "ready_for_policy_comparison": "NO",
    }


def test_recovery_domain_metrics_and_weighting_are_preregistered() -> None:
    """128K 门禁、指标顺序与 snapshot 等权规则不得缺失。"""
    artifact = _artifact()
    requirement = artifact["recovery_model_requirement"]
    evaluation = artifact["evaluation_protocol"]

    assert requirement["maximum_required_validated_gap_tokens"] == 131_072
    assert requirement["independent_profiler_validation_required"] is True
    assert requirement["validated_to_128k"] is False
    assert requirement["linear_extrapolation_allowed"] is False
    assert requirement["clamp_to_32k_allowed"] is False
    assert requirement["formal_phi_modified"] is False
    assert evaluation["policies"] == [
        "Global-LRU",
        "KVFlow-style",
        "Marconi-style",
        "FlowState",
    ]
    assert evaluation["snapshot_weight"] == "每个 selected snapshot 等权"
    assert evaluation["pending_weighted_results_separate"] is True


def test_protocol_source_does_not_execute_policy_or_phi() -> None:
    """协议脚本不得调用 optimizer、policy select、Phi 或 recovery gap。"""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "GlobalOptimizer(",
        "RecoveryCostModel(",
        ".select(",
        "recovery_gap(",
        "executable_frontier(",
    )
    assert all(value not in source for value in forbidden)


def test_report_states_final_boundaries_without_performance_claims() -> None:
    """报告必须明确冻结范围、128K blocker 与零次策略执行。"""
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "主 cohort 正式冻结为 **C128**" in report
    assert "不能进入正式 policy comparison" in report
    assert "独立 recovery profiler 必须验证至 128K" in report
    assert "没有运行 policy、Phi 或 GPU" in report


def test_protected_core_policy_and_prior_artifacts_are_unchanged() -> None:
    """核心、策略、motivation 与 10C.2 之前的冻结文件不得变化。"""
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        data = (ROOT / relative_path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected_hash
