"""验证 TraceLab C128 非平凡 demand 全量结构审计。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.public_agent_trace.tracelab_nontrivial_demand import (
    MAIN_COHORT,
    MAIN_COHORT_MAX_TOKENS,
    DemandSnapshotEvent,
    budget_discreteness,
    candidate_event_rank,
    project_workload,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_nontrivial_demand.json"
)
REPORT_PATH = ARTIFACT_PATH.with_name("TRACELAB_NONTRIVIAL_DEMAND.md")
SOURCE_PATH = ARTIFACT_PATH.with_suffix(".py")
PROTECTED_HASHES = {
    "evaluation/public_agent_trace/TRACELAB_FINAL_PROTOCOL.md": (
        "e902187d4681ca971d61811da64137f3098ea747ad21676cbe88acb7feac1a03"
    ),
    "evaluation/public_agent_trace/tracelab_final_protocol.json": (
        "e8eeb0808d4db9761754fe5e4db6c7384f3621c992c0e955d103ed6faaeb0a46"
    ),
    "evaluation/public_agent_trace/tracelab_final_protocol.py": (
        "e6cdc288402ae8fd16234710342ab1076453f61f84610c4b0504bbcad50dce79"
    ),
    "tests/test_tracelab_final_protocol.py": (
        "2bec1b87cbc6835daafb8128cf62e24be383cf0232d610cd3d1ac300b6d880b4"
    ),
    "flowstate/recovery_model.py": (
        "9a13bc4f7778b9e1835ddb04237d54815ff86c7e9c57b42d293e73c5bb404082"
    ),
    "evaluation/controlled_multiworkflow_v1/policies.py": (
        "8df5a1391b651f3a55090e13b8abb9d2a520de0a94abeb6a7339fdcb49445a24"
    ),
    "evaluation/sota_policies.py": (
        "b276aff22d2dc1adcdb33b15a7a94dc608fa916789ba0f2e5d5fbe0b3189d212"
    ),
    "evaluation/sota_metadata.py": (
        "df6582dd9a5dd15e984e9cefdd899e1d0b8bc9292399e7862608d04162a283c2"
    ),
    "motivation/README.md": (
        "a066a70f1fb13bba472147fc6847ec8b80f6d7dd8d02fa3d698677abced659a8"
    ),
}


def _artifact() -> dict:
    """读取冻结的全量需求审计 artifact。"""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _event(
    *,
    round_pk: int,
    x_value: int,
    active_ids: tuple[str, ...],
    provider: str = "claude",
    scale: str = "Small",
    bucket: str = "<=32K",
) -> DemandSnapshotEvent:
    """构造用于 deterministic projection 的纯结构事件。"""
    return DemandSnapshotEvent(
        provider=provider,
        context_bucket=bucket,
        scale=scale,
        trigger_session_id=f"session-{round_pk}",
        trigger_run_ordinal=1,
        trigger_round_pk=round_pk,
        observed_at=datetime(2026, 8, 26),
        active_run_ids=active_ids,
        active_workflow_count=len(active_ids),
        candidate_count=10,
        pending_count=x_value,
        exact_parent_count=x_value,
        pending_anchors=tuple(4096 for _ in range(x_value)),
    )


def test_c128_cutoff_and_candidate_event_semantics_are_frozen() -> None:
    """审计 cohort 必须继续使用 C128 和现有 tool-call 时点。"""
    artifact = _artifact()
    cohort = artifact["fixed_cohort"]
    event = artifact["candidate_snapshot_event"]

    assert MAIN_COHORT == cohort["name"] == "C128"
    assert MAIN_COHORT_MAX_TOKENS == 131_072
    assert cohort["maximum_input_tokens_total"] == 131_072
    assert cohort["strictly_closed_only"] is True
    assert "最后一个已发出 tool call" in event["definition"]
    assert event["synthetic_snapshot"] is False
    assert event["future_fields_used"] is False


@pytest.mark.parametrize(
    ("x_value", "expected", "levels"),
    (
        (2, (1, 1, 1, 2), 2),
        (3, (1, 1, 2, 3), 3),
        (4, (1, 2, 3, 4), 4),
        (5, (1, 2, 3, 5), 4),
        (6, (1, 3, 4, 6), 4),
        (7, (1, 3, 5, 7), 4),
    ),
)
def test_budget_discreteness_uses_frozen_demand_formula(
    x_value: int,
    expected: tuple[int, ...],
    levels: int,
) -> None:
    """每个 X 的四档 K 与 distinct levels 必须精确。"""
    row = budget_discreteness(x_value)
    assert tuple(row["k_by_ratio"].values()) == expected
    assert row["distinct_k_levels"] == levels


def test_projection_is_deterministic_thresholded_and_deduplicated() -> None:
    """投影不得依赖输入顺序，并须应用 X gate 与 active-set 去重。"""
    duplicate_early = _event(
        round_pk=1,
        x_value=2,
        active_ids=("W1", "W2"),
    )
    duplicate_late = _event(
        round_pk=2,
        x_value=3,
        active_ids=("W2", "W1"),
    )
    trivial = _event(
        round_pk=3,
        x_value=1,
        active_ids=("W3", "W4"),
    )
    unique = _event(
        round_pk=4,
        x_value=4,
        active_ids=("W5", "W6"),
    )
    records = (duplicate_early, duplicate_late, trivial, unique)

    normal = project_workload(records, minimum_x=2, max_per_stratum=5)
    reversed_result = project_workload(
        tuple(reversed(records)),
        minimum_x=2,
        max_per_stratum=5,
    )

    assert normal == reversed_result
    assert all(event.exact_parent_count >= 2 for event in normal)
    assert len({tuple(sorted(item.active_run_ids)) for item in normal}) == 2
    expected_duplicate = min(
        (duplicate_early, duplicate_late),
        key=lambda item: (candidate_event_rank(item), item.trigger_round_pk),
    )
    assert expected_duplicate in normal
    assert unique in normal


def test_projection_respects_per_stratum_limit() -> None:
    """每个非空 stratum 的 projected snapshots 不得超过上限。"""
    events = tuple(
        _event(
            round_pk=index,
            x_value=2,
            active_ids=(f"W{index}", f"X{index}"),
        )
        for index in range(30)
    )
    assert len(project_workload(events, 2, 5)) == 5
    assert len(project_workload(events, 2, 10)) == 10
    assert len(project_workload(events, 2, 20)) == 20


def test_full_c128_x_histogram_is_exact_and_complete() -> None:
    """全量事件 X histogram 必须无遗漏且与累计 cohort 一致。"""
    artifact = _artifact()
    histogram = artifact["exact_x_histogram"]

    assert artifact["all_candidate_snapshot_count"] == 40_849
    assert histogram == {
        "X=0": 1_706,
        "X=1": 22_103,
        "X=2": 14_690,
        "X=3": 1_931,
        "X=4": 145,
        "X=5": 8,
        "X=6": 253,
        "X>6": 13,
    }
    assert sum(histogram.values()) == 40_849
    assert artifact["classifications"]["X>=2"]["snapshot_count"] == 17_040
    assert artifact["classifications"]["X>=4"]["snapshot_count"] == 419


def test_x_definition_matches_frozen_exact_parent_semantics() -> None:
    """X 必须继续表示 known pending 对应的 distinct exact parents。"""
    artifact = _artifact()
    definition = artifact["definitions"]
    validation = artifact["validation"]

    assert "distinct exact-parent recurrent states" in definition["x"]
    assert definition["known_anchor"] == "current_round.input_tokens_total"
    assert definition["multiple_tool_calls_create_fanout"] is False
    assert validation["current_sample_state_mismatches"] == 0


def test_nontrivial_cohorts_preserve_provider_and_concurrency_evidence() -> None:
    """X>=2 与 X>=4 必须分别保留真实 provider 与 scale 分布。"""
    classifications = _artifact()["classifications"]
    x2 = classifications["X>=2"]
    x4 = classifications["X>=4"]

    assert x2["provider_counts"] == {"claude": 13_504, "codex": 3_536}
    assert x2["scale_counts"] == {
        "Small": 15_868,
        "Medium": 1_132,
        "Large": 40,
    }
    assert x2["unique_active_runs"] == 3_372
    assert x4["provider_counts"] == {"claude": 416, "codex": 3}
    assert x4["scale_counts"] == {
        "Small": 103,
        "Medium": 316,
        "Large": 0,
    }
    assert x4["unique_active_runs"] == 139


def test_workload_size_projection_is_frozen_without_policy_results() -> None:
    """三个 X gate 与三档密度的投影数量必须可复核。"""
    projection = _artifact()["workload_size_projection"]
    expected = {
        "X>=2": {"max5": 60, "max10": 105, "max20": 185},
        "X>=3": {"max5": 41, "max10": 77, "max20": 134},
        "X>=4": {"max5": 27, "max10": 37, "max20": 38},
    }
    assert {
        threshold: {
            limit: projection[threshold][limit]["snapshot_count"]
            for limit in expected[threshold]
        }
        for threshold in expected
    } == expected
    assert all(
        row["duplicate_active_run_set_count"] == 0
        for threshold in projection.values()
        for row in threshold.values()
    )


def test_current_57_histogram_explains_identical_pressure_fraction() -> None:
    """当前三档 K<X 必须都且只对应 X>=2 的 22 个快照。"""
    current = _artifact()["current_57_snapshot_audit"]

    assert current["exact_histogram"] == {
        "X=1": 35,
        "X=2": 13,
        "X=3": 2,
        "X=4": 1,
        "X=5": 1,
        "X=6": 5,
        "X>6": 0,
    }
    assert current["cumulative_histogram"] == {
        "X>=2": 22,
        "X>=3": 9,
        "X>=4": 7,
    }
    assert current["k_lt_x_fraction_by_ratio"] == {
        "25%": pytest.approx(22 / 57),
        "50%": pytest.approx(22 / 57),
        "75%": pytest.approx(22 / 57),
        "100%": 0.0,
    }


def test_recovery_domain_uses_real_known_anchors_without_phi() -> None:
    """Recovery domain 必须来自真实 anchor，并保持 Phi 调用为零。"""
    requirement = _artifact()["recovery_domain_requirement"]

    assert requirement["X>=2"]["known_anchor_tokens"]["median"] == 69_802
    assert requirement["X>=2"]["known_anchor_tokens"]["p95"] == 117_358
    assert requirement["X>=2"]["maximum_required_gap_tokens"] == 130_969
    assert requirement["X>=4"]["known_anchor_tokens"]["median"] == 46_004
    assert requirement["X>=4"]["known_anchor_tokens"]["p95"] == 71_801
    assert requirement["X>=4"]["maximum_required_gap_tokens"] == 124_278
    assert requirement["phi_called"] is False


def test_diagnosis_is_structural_and_recommends_no_execution() -> None:
    """最终判断必须止于 sampling 建议，不运行建议。"""
    artifact = _artifact()
    diagnosis = artifact["diagnosis"]

    assert diagnosis["enough_x_ge_2"] == "YES"
    assert diagnosis["enough_x_ge_4"] == "WEAK"
    assert diagnosis["current_sample_underrepresents_nontrivial_demand"] == "YES"
    assert diagnosis["profiler_extension_before_sampling_revision"] == "NO"
    assert diagnosis["recommended_next_step"] == "在 C128 内重新冻结 X>=2 sampling"
    assert artifact["execution"] == {
        "policy_comparison_executed": False,
        "phi_called": False,
        "gpu_executed": False,
    }


def test_source_has_no_policy_phi_or_synthetic_execution() -> None:
    """审计脚本不得执行 policy、Phi 或制造 synthetic concurrency。"""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "GlobalOptimizer(",
        "RecoveryCostModel(",
        "MarconiStylePolicy(",
        "KVFlowStylePolicy(",
        ".select(",
        "recovery_gap(",
        "time_shift",
    )
    assert all(value not in source for value in forbidden)


def test_report_states_evidence_scope_and_recommendation() -> None:
    """报告必须回答四个 gate 并禁止将结构样本量冒充 policy 结论。"""
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "C128 共形成 40,849 个" in report
    assert "X>=2 是否充足：**YES**" in report
    assert "X>=4 是否充足：**WEAK**" in report
    assert "在 C128 内重新冻结 X>=2 sampling" in report
    assert "不构成具体 policy effect 的统计功效分析" in report


def test_core_phi_policies_motivation_and_10c3_are_unchanged() -> None:
    """核心、Phi、策略、motivation 与 Step 10C.3 不得变化。"""
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_hash
