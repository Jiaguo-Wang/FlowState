"""验证 TraceLab context coverage 与状态压力审计。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.public_agent_trace.tracelab_context_pressure import (
    BUDGET_RATIOS,
    CONTEXT_COHORTS,
    ContextSnapshotEvent,
    analyze_snapshot,
    budget_k,
    choose_context_events,
    context_bucket,
    context_cohort_contains,
    exact_parent_checkpoint_ids,
    lineage_recovery_envelope,
)
from evaluation.public_agent_trace.tracelab_to_flowstate import (
    SampledSnapshotEvent,
    TraceSnapshot,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_context_pressure.json"
)
SOURCE_PATH = ARTIFACT_PATH.with_suffix(".py")
PROTECTED_HASHES = {
    "evaluation/public_agent_trace/tracelab_to_flowstate.py": (
        "445b8b1064449a76c9c2bfba9b3710af90f10853bb46c34aec9af7673d9de2e0"
    ),
    "evaluation/public_agent_trace/tracelab_policy_protocol.py": (
        "757e3740cd442787a8cfa1c178ce640d9b7ab99a82ee0cd73a0445d0222f8c5d"
    ),
    "evaluation/controlled_multiworkflow_v1/scenario.py": (
        "608f729c2670f249201402063bc2d354d85bc7a43657d4be5f77c13ff6fe5909"
    ),
    "evaluation/scalable_multiworkflow_v2/scenario.py": (
        "a39ec5a1a9761ccefcefb4763eb10ce142895fc53197bd0f4d66746cc71e5bdd"
    ),
    "flowstate/recovery_model.py": (
        "f3fe216592ad62c26e5bf7936f907823745942f7f34b483b8dfbc2fbd8fda1f5"
    ),
}


def _candidate(
    checkpoint_id: str,
    path: tuple[str, ...],
    token_pos: int,
) -> CheckpointCandidate:
    """构造同一 workflow 的逻辑 checkpoint。"""
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id="W",
        lineage_path=path,
        token_pos=token_pos,
        memory_bytes=51_511_296,
    )


def _event(
    *,
    cohort: str,
    provider: str,
    scale: str,
    bucket: str,
    round_pk: int,
) -> ContextSnapshotEvent:
    """构造用于确定性采样测试的真实事件描述。"""
    cutoff = dict(CONTEXT_COHORTS)[cohort]
    return ContextSnapshotEvent(
        cohort=cohort,
        cutoff_tokens=cutoff,
        event=SampledSnapshotEvent(
            snapshot_id=f"{cohort}-{provider}-{scale}-{round_pk}",
            scale=scale,
            provider=provider,
            context_bucket=bucket,
            trigger_session_id=f"session-{round_pk}",
            trigger_run_ordinal=1,
            trigger_round_pk=round_pk,
            observed_at=datetime(2026, 8, 26, 12, 0, round_pk),
            trace_observed_active_runs=2,
        ),
    )


def _artifact() -> dict:
    """读取冻结结构审计 artifact。"""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("tokens", "expected_bucket"),
    (
        (32_768, "<=32K"),
        (32_769, "32K-64K"),
        (65_536, "32K-64K"),
        (65_537, "64K-128K"),
        (131_072, "64K-128K"),
        (131_073, "128K-256K"),
        (262_144, "128K-256K"),
        (262_145, ">256K"),
    ),
)
def test_context_cutoff_boundaries(
    tokens: int,
    expected_bucket: str,
) -> None:
    """固定 context bucket 边界必须逐 token 正确。"""
    assert context_bucket(tokens) == expected_bucket


def test_cumulative_cohort_membership_is_nested() -> None:
    """较短累计 cohort 的 run 必须属于所有更长 cohort。"""
    cutoffs = [cutoff for _, cutoff in CONTEXT_COHORTS]
    for token_count in (0, 32_768, 65_536, 131_072, 262_144):
        memberships = [
            context_cohort_contains(token_count, cutoff)
            for cutoff in cutoffs
        ]
        assert memberships == sorted(memberships)


def test_snapshot_sampling_is_deterministic_without_synthetic_fill() -> None:
    """输入顺序不得改变分层选择，空 stratum 不得被补造。"""
    events = (
        _event(
            cohort="C32",
            provider="claude",
            scale="Small",
            bucket="<=32K",
            round_pk=1,
        ),
        _event(
            cohort="C32",
            provider="claude",
            scale="Small",
            bucket="<=32K",
            round_pk=2,
        ),
        _event(
            cohort="C64",
            provider="codex",
            scale="Medium",
            bucket="32K-64K",
            round_pk=3,
        ),
    )
    normal = choose_context_events(events)
    reversed_result = choose_context_events(tuple(reversed(events)))
    assert normal == reversed_result
    assert len(normal) == 2
    assert {
        (
            item.cohort,
            item.event.scale,
            item.event.provider,
            item.event.context_bucket,
        )
        for item in normal
    } == {
        ("C32", "Small", "claude", "<=32K"),
        ("C64", "Medium", "codex", "32K-64K"),
    }


@pytest.mark.parametrize(
    ("base_count", "ratio", "expected"),
    (
        (0, 0.25, 1),
        (1, 0.25, 1),
        (7, 0.25, 1),
        (7, 0.50, 3),
        (7, 0.75, 5),
        (7, 1.00, 7),
    ),
)
def test_three_budget_formulas_share_the_frozen_rounding_rule(
    base_count: int,
    ratio: float,
    expected: int,
) -> None:
    """N、P、X 三种 base 都必须使用同一向下取整和下限。"""
    assert budget_k(base_count, ratio) == expected


def test_x_counts_distinct_shared_exact_parent() -> None:
    """两个 continuation 共享一个 exact parent 时 X 必须为一。"""
    candidate = _candidate("parent", ("P",), 100)
    continuations = (
        PendingContinuation("A", "W", ("P", "A"), 100, 100),
        PendingContinuation("B", "W", ("P", "B"), 100, 100),
    )
    snapshot = TraceSnapshot(
        snapshot_id="shared-parent",
        scale="Small",
        time_domain="claude",
        observed_at=datetime(2026, 8, 26),
        active_workflow_ids=("W",),
        candidates=(candidate,),
        continuations=continuations,
        checkpoint_metadata=(),
        continuation_metadata=(),
        retention_budgets=(),
    )

    assert exact_parent_checkpoint_ids(continuations[0], (candidate,)) == (
        "parent",
    )
    analysis = analyze_snapshot(snapshot)
    assert analysis["p"] == 2
    assert analysis["x"] == 1


def test_g1_g2_g4_g8_follow_linear_lineage_history() -> None:
    """多级 gap 必须沿 lineage 从最近历史 checkpoint 向前计算。"""
    paths = tuple(
        tuple(f"S{step}" for step in range(length))
        for length in range(1, 10)
    )
    candidates = tuple(
        _candidate(f"cp-{index}", paths[index - 1], index * 1000)
        for index in range(1, 10)
    )
    continuation = PendingContinuation(
        continuation_id="pending",
        workflow_id="W",
        lineage_path=paths[8],
        anchor_pos=9000,
        resident_fa_frontier=9000,
    )

    assert lineage_recovery_envelope(continuation, candidates) == {
        "G1": 1000,
        "G2": 2000,
        "G4": 4000,
        "G8": 8000,
    }


def test_insufficient_history_is_unavailable() -> None:
    """历史 checkpoint 数不足时不得以零补齐。"""
    candidates = (
        _candidate("cp-1", ("P",), 1000),
        _candidate("cp-2", ("P", "A"), 2000),
        _candidate("exact", ("P", "A", "B"), 3000),
    )
    continuation = PendingContinuation(
        "pending",
        "W",
        ("P", "A", "B"),
        3000,
        3000,
    )

    assert lineage_recovery_envelope(continuation, candidates) == {
        "G1": 1000,
        "G2": 2000,
        "G4": None,
        "G8": None,
    }


def test_frozen_artifact_has_nested_real_cohorts_and_no_leakage() -> None:
    """真实 artifact 必须记录嵌套 cohort、真实 overlap 与零泄漏。"""
    artifact = _artifact()
    summaries = [item["summary"] for item in artifact["cohorts"]]

    assert [item["cohort"] for item in summaries] == [
        "C32",
        "C64",
        "C128",
        "C256",
    ]
    assert [item["eligible_runs"] for item in summaries] == sorted(
        item["eligible_runs"] for item in summaries
    )
    assert all(item["overlapping_runs"] > 0 for item in summaries)
    assert all(
        sum(item["integrity"].values()) == 0 for item in summaries
    )
    assert artifact["validation"]["nested_cohort_violations"] == 0
    assert artifact["frozen_semantics"]["synthetic_concurrency"] is False


def test_exact_parent_and_budget_audit_are_structurally_consistent() -> None:
    """X、P 与三种预算结果必须保持冻结结构语义。"""
    artifact = _artifact()
    for cohort in artifact["cohorts"]:
        summary = cohort["summary"]
        assert summary["exact_parent_availability_ratio"] == 1.0
        for normalization in (
            "candidate_relative",
            "pending_relative",
            "exact_parent_relative",
        ):
            for ratio in BUDGET_RATIOS:
                row = summary["budget_normalization"][normalization][
                    f"{int(ratio * 100)}%"
                ]
                assert row["snapshot_count"] == summary["snapshot_count"]
                assert all(value >= 1 for value in row["k_values"])


def test_artifact_marks_zero_demand_and_unavailable_ratios_honestly() -> None:
    """零 pending snapshot 不得被删除，也不得产生伪造除法结果。"""
    artifact = _artifact()
    summaries = {
        item["summary"]["cohort"]: item["summary"]
        for item in artifact["cohorts"]
    }
    assert summaries["C128"]["zero_pending_snapshot_count"] == 1
    assert summaries["C256"]["zero_pending_snapshot_count"] == 2
    assert summaries["C256"]["state_pressure_ratios"]["N/P"][
        "count"
    ] == summaries["C256"]["snapshot_count"] - 2


def test_no_policy_phi_or_gpu_execution_is_present() -> None:
    """结构审计不得导入或调用策略、Phi、优化器或 GPU runtime。"""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "RecoveryCostModel",
        "GlobalOptimizer",
        "KVFlowStylePolicy",
        "MarconiStylePolicy",
        ".select(",
        "torch",
        "cuda",
    )
    assert all(fragment not in source for fragment in forbidden)
    validation = _artifact()["validation"]
    assert validation["policy_comparison_runs"] == 0
    assert validation["phi_calls"] == 0
    assert validation["gpu_runs"] == 0


def test_preexisting_semantics_core_and_workloads_are_unchanged() -> None:
    """本步骤不得覆盖既有 trace 语义、controlled workload 或 core。"""
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        actual_hash = hashlib.sha256(
            (ROOT / relative_path).read_bytes()
        ).hexdigest()
        assert actual_hash == expected_hash

