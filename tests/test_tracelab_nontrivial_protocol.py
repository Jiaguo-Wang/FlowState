"""验证 TraceLab 非平凡需求工作负载的最终冻结协议。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from evaluation.public_agent_trace.tracelab_nontrivial_demand import (
    DemandSnapshotEvent,
)
from evaluation.public_agent_trace.tracelab_nontrivial_protocol import (
    DEMAND_RETENTION_RATIOS,
    MAIN_COHORT,
    MAIN_COHORT_MAX_TOKENS,
    MAIN_X_THRESHOLD,
    MAX_SNAPSHOTS_PER_STRATUM,
    REQUIRED_RECOVERY_GAP_TOKENS,
    SECONDARY_X_THRESHOLD,
    construct_nontrivial_protocol,
    summarize_demand_budgets,
    summarize_selected_events,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_nontrivial_protocol.json"
)
REPORT_PATH = ARTIFACT_PATH.with_name("TRACELAB_NONTRIVIAL_PROTOCOL.md")
SOURCE_PATH = ARTIFACT_PATH.with_suffix(".py")
PROTECTED_HASHES = {
    "evaluation/public_agent_trace/TRACELAB_NONTRIVIAL_DEMAND.md": (
        "ead83bb1b88a7b8a4867268a76b92d531da13a2a3ff21fb8317fa9cbce00a461"
    ),
    "evaluation/public_agent_trace/tracelab_nontrivial_demand.json": (
        "33f315d21fe53ac676149c65377614aa12da51b6fe4bd5d58f150ff4a80d993a"
    ),
    "evaluation/public_agent_trace/tracelab_nontrivial_demand.py": (
        "c612bcf28c0f585e2456acc5f681cc0fd468418db9f4ae7284389f6554bf16c9"
    ),
    "tests/test_tracelab_nontrivial_demand.py": (
        "f373cb4ef65c951054004fb511a7c05ae191cd8a318311c5d316ad02a49fd19c"
    ),
    "evaluation/public_agent_trace/TRACELAB_FINAL_PROTOCOL.md": (
        "e902187d4681ca971d61811da64137f3098ea747ad21676cbe88acb7feac1a03"
    ),
    "evaluation/public_agent_trace/tracelab_final_protocol.json": (
        "e8eeb0808d4db9761754fe5e4db6c7384f3621c992c0e955d103ed6faaeb0a46"
    ),
    "evaluation/public_agent_trace/tracelab_final_protocol.py": (
        "e6cdc288402ae8fd16234710342ab1076453f61f84610c4b0504bbcad50dce79"
    ),
    "flowstate/recovery_model.py": (
        "f3fe216592ad62c26e5bf7936f907823745942f7f34b483b8dfbc2fbd8fda1f5"
    ),
    "evaluation/controlled_multiworkflow_v1/scenario.py": (
        "608f729c2670f249201402063bc2d354d85bc7a43657d4be5f77c13ff6fe5909"
    ),
    "evaluation/controlled_multiworkflow_v1/policies.py": (
        "cbca81712c41bbcadd12a923fb6387c1c2d96976ce2ab38e21a9afcc62b9d375"
    ),
    "evaluation/scalable_multiworkflow_v2/scenario.py": (
        "a39ec5a1a9761ccefcefb4763eb10ce142895fc53197bd0f4d66746cc71e5bdd"
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
    """读取最终冻结协议制品。"""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _event(x_value: int, round_pk: int) -> DemandSnapshotEvent:
    """构造纯结构预算与汇总测试事件。"""
    return DemandSnapshotEvent(
        provider="claude",
        context_bucket="<=32K",
        scale="Small",
        trigger_session_id=f"session-{round_pk}",
        trigger_run_ordinal=1,
        trigger_round_pk=round_pk,
        observed_at=datetime(2026, 8, 26),
        active_run_ids=(f"W{round_pk}", f"X{round_pk}"),
        active_workflow_count=2,
        candidate_count=4 * x_value,
        pending_count=x_value,
        exact_parent_count=x_value,
        pending_anchors=tuple(4096 for _ in range(x_value)),
    )


def test_main_and_secondary_cohorts_are_frozen() -> None:
    """C128、X>=2 主集合与 X>=4 次级集合必须固定。"""
    protocol = _artifact()
    main = protocol["main_policy_cohort"]
    secondary = protocol["secondary_high_contention_slice"]

    assert MAIN_COHORT == "C128"
    assert MAIN_COHORT_MAX_TOKENS == 131_072
    assert MAIN_X_THRESHOLD == main["minimum_x"] == 2
    assert main["rule"] == "C128 AND X>=2"
    assert SECONDARY_X_THRESHOLD == secondary["minimum_x"] == 4
    assert secondary["rule"] == "C128 AND X>=4"
    assert secondary["not_main_policy_cohort"] is True
    assert secondary["not_used_for_parameter_selection"] is True


def test_full_c128_characterization_remains_visible() -> None:
    """协议必须保留自然 X 分布和冻结的 41.715%。"""
    full = _artifact()["full_c128_characterization"]

    assert full["candidate_snapshot_count"] == 40_849
    assert full["X=0"]["count"] == 1_706
    assert full["X=1"]["count"] == 22_103
    assert full["X>=2"]["count"] == 17_040
    assert full["X>=3"]["count"] == 2_350
    assert full["X>=4"]["count"] == 419
    assert full["x_greater_equal_2_fraction_frozen"] == 17_040 / 40_849
    assert full["hidden_from_paper"] is False


def test_main_selection_has_exact_frozen_counts_and_histogram() -> None:
    """主集合每层最多十个投影的快照、来源与 X 直方图必须精确。"""
    selection = _artifact()["main_policy_cohort"]["selection"]

    assert selection["snapshot_count"] == 105
    assert selection["unique_active_runs"] == 284
    assert selection["provider_counts"] == {"claude": 67, "codex": 38}
    assert selection["scale_counts"] == {
        "Small": 60,
        "Medium": 38,
        "Large": 7,
    }
    assert selection["x_histogram"] == {
        "X=2": 83,
        "X=3": 9,
        "X=4": 4,
        "X=5": 1,
        "X=6": 8,
        "X>6": 0,
    }
    assert selection["duplicate_active_run_set_count"] == 0


def test_selected_snapshot_structure_is_frozen() -> None:
    """W、N、P、X、N/X 的正式分布必须来自选中事件。"""
    structure = _artifact()["main_policy_cohort"]["selection"]["structure"]

    assert structure["W"]["median"] == 3
    assert structure["W"]["p90"] == 7
    assert structure["N"]["median"] == 19
    assert structure["N"]["p95"] == 79
    assert structure["P"]["median"] == 2
    assert structure["X"]["p95"] == 6
    assert structure["N/X"]["median"] == 8.5
    assert structure["N/X"]["p90"] == 29.5


def test_max10_strata_and_active_sets_are_valid() -> None:
    """正式主集合选择必须满足每层最多十个与全局活动集合去重。"""
    snapshots = _artifact()["selected_main_snapshots"]
    strata: dict[tuple[str, str, str], int] = {}
    active_sets = set()
    for row in snapshots:
        stratum = (row["provider"], row["context_bucket"], row["scale"])
        strata[stratum] = strata.get(stratum, 0) + 1
        active_set = tuple(sorted(row["active_run_ids"]))
        assert active_set not in active_sets
        active_sets.add(active_set)
        assert row["x"] >= MAIN_X_THRESHOLD
    assert max(strata.values()) <= MAX_SNAPSHOTS_PER_STRATUM == 10


def test_protocol_construction_is_deterministic() -> None:
    """重复读取相同的策略执行前审计必须生成完全相同的协议。"""
    first = construct_nontrivial_protocol()
    second = construct_nontrivial_protocol()
    assert first == second == _artifact()


def test_pure_summary_and_budget_functions_do_not_require_policy() -> None:
    """结构汇总与预算必须只依赖 X 和快照事实。"""
    events = tuple(_event(x_value, index) for index, x_value in enumerate((2, 3, 4), start=1))
    summary = summarize_selected_events(events)
    budget = summarize_demand_budgets(events)

    assert summary["x_histogram"] == {
        "X=2": 1,
        "X=3": 1,
        "X=4": 1,
        "X=5": 0,
        "X=6": 0,
        "X>6": 0,
    }
    for label, ratio in zip(("25%", "50%", "75%", "100%"), DEMAND_RETENTION_RATIOS):
        expected = [
            max(1, math.floor(event.exact_parent_count * ratio))
            for event in events
        ]
        assert budget[label]["k_distribution"]["mean"] == sum(expected) / 3


def test_budget_and_discreteness_are_exact_parent_relative() -> None:
    """四档 K、压力比例与三类离散模式必须冻结。"""
    protocol = _artifact()
    budget_protocol = protocol["budget_protocol"]
    budget = budget_protocol["selected_summary"]

    assert budget_protocol["operationalization"] == "exact-parent demand X"
    assert budget_protocol["ratios"] == [0.25, 0.5, 0.75, 1.0]
    assert budget_protocol["candidate_relative_budget"] is False
    assert budget["25%"]["k_distribution"]["mean"] == 1
    assert budget["50%"]["k_distribution"]["mean"] == 1.2
    assert budget["75%"]["k_distribution"]["mean"] == 148 / 105
    assert budget["100%"]["k_distribution"]["mean"] == 262 / 105
    assert [budget[label]["k_lt_x_fraction"] for label in ("25%", "50%", "75%", "100%")] == [1.0, 1.0, 1.0, 0.0]
    assert budget["constrained_ratio_patterns"] == {
        "25=50=75": 83,
        "25=50!=75": 9,
        "25_50_75_all_distinct": 13,
    }


def test_sampling_representativeness_is_weak_and_not_retuned() -> None:
    """分层样本构成变化必须如实记录，且不能触发继续调样。"""
    representative = _artifact()["sampling_representativeness"]

    assert representative["diagnosis"] == "WEAK"
    assert representative["all_material_categories_preserved"] is True
    assert representative["all_natural_categories_preserved"] is False
    assert representative["omitted_natural_categories"] == ["X 分布:X>6"]
    assert representative["maximum_absolute_category_difference"] > 0.35
    assert representative["sampling_changed_after_audit"] is False


def test_secondary_high_contention_slice_is_fixed_and_separate() -> None:
    """X>=4 切片必须使用相同采样规则且不能替代主结果。"""
    secondary = _artifact()["secondary_high_contention_slice"]
    selection = secondary["selection"]

    assert secondary["label"] == "high-contention secondary slice"
    assert selection["snapshot_count"] == 37
    assert selection["unique_active_runs"] == 135
    assert selection["provider_counts"] == {"claude": 35, "codex": 2}
    assert selection["scale_counts"] == {
        "Small": 15,
        "Medium": 22,
        "Large": 0,
    }
    assert selection["duplicate_active_run_set_count"] == 0


def test_policy_metadata_metrics_and_freeze_declaration_are_complete() -> None:
    """策略元数据、主要指标与最终冻结项不得缺失。"""
    protocol = _artifact()
    metadata = protocol["policy_metadata_protocol"]
    declaration = protocol["freeze_declaration"]

    assert metadata["kvflow"]["steps_to_execution"] == 1
    assert metadata["marconi"]["alpha"] == 1.0
    assert metadata["flowstate"]["future_prefix_used"] is False
    assert metadata["flowstate"]["future_round_used"] is False
    assert len(protocol["evaluation_metrics"]["primary"]) == 4
    assert len(protocol["evaluation_metrics"]["secondary"]) == 3
    assert all(
        value == "FROZEN"
        for key, value in declaration.items()
        if key != "future_policy_performance_can_change_protocol"
    )
    assert declaration["future_policy_performance_can_change_protocol"] is False


def test_recovery_profiler_is_next_gate_and_policy_is_not_ready() -> None:
    """128K 独立 profiler 必须继续阻止正式策略比较。"""
    protocol = _artifact()
    requirement = protocol["recovery_model_requirement"]
    readiness = protocol["readiness"]

    assert REQUIRED_RECOVERY_GAP_TOKENS == 131_072
    assert requirement["validated_gap_required_tokens"] == 131_072
    assert requirement["observed_main_anchor_max_tokens"] == 130_969
    assert requirement["validated_to_required_domain"] is False
    assert requirement["linear_extrapolation_from_32k_allowed"] is False
    assert requirement["clamp_allowed"] is False
    assert requirement["formal_phi_modified"] is False
    assert readiness == {
        "ready_for_recovery_profiler_extension": "PASS",
        "ready_for_policy_comparison": "NO",
        "reason": "正式恢复代价模型尚未独立验证至 128K",
    }


def test_no_future_synthetic_policy_phi_or_gpu_execution() -> None:
    """全部完整性门禁与零执行计数必须通过。"""
    protocol = _artifact()
    validation = protocol["validation"]
    sampling = protocol["sampling_protocol"]

    assert all(value == 0 for value in validation.values())
    assert sampling["synthetic_concurrency"] is False
    assert sampling["agent_run_time_shift"] is False
    assert sampling["future_round_used"] is False
    assert sampling["phi_used"] is False
    assert sampling["policy_used"] is False
    assert protocol["execution"] == {
        "policy_comparison_executed": False,
        "phi_called": False,
        "gpu_executed": False,
    }


def test_source_does_not_execute_policy_or_phi() -> None:
    """协议脚本不得调用正式策略、优化器、Phi 或恢复缺口。"""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "GlobalOptimizer(",
        "RecoveryCostModel(",
        "KVFlowStylePolicy(",
        "MarconiStylePolicy(",
        ".select(",
        "recovery_gap(",
    )
    assert all(value not in source for value in forbidden)


def test_report_records_amendment_limits_and_hard_freeze() -> None:
    """报告必须说明策略执行前修订、WEAK 限制与不可再调协议。"""
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "C128 AND X>=2" in report
    assert "采样代表性评为 **WEAK**" in report
    assert "旧 57-snapshot set 保留" in report
    assert "不是事后性能调优" in report
    assert "后续不得因为 FlowState 表现好坏修改" in report
    assert "Policy comparison 准备状态：**NO**" in report


def test_core_controlled_workloads_policies_and_prior_artifacts_unchanged() -> None:
    """核心、受控工作负载、策略、动机材料与旧制品不得变化。"""
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_hash
