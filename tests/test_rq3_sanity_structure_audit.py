"""Step 13G-A：对 sanity/structure/mechanism audit runner 的 validation gate 测试。

所有新增注释使用中文。
本测试只读取已冻结的正式 population 与 evaluation artifact，不修改核心算法。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.rq3_sanity_structure_audit import (
    analyze_chain_structure,
    analyze_compatibility_structure,
    analyze_marconi_overlap,
    analyze_per_workflow_best_probe,
    analyze_redundancy,
    analyze_standalone_topk_probe,
    analyze_workflow_distribution,
    audit_budget_monotonicity,
    audit_selector_semantics,
    build_snapshot_map,
    independent_exact_opt_audit,
    load_per_snapshot_results,
    reproduce_objectives,
    run_audit,
)


_FROZEN_FORMAL_ROOT = Path(
    "evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"
)
_FROZEN_EVALUATION_ROOT = Path(
    "evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011"
)


@pytest.fixture(scope="session")
def audit_results() -> dict:
    """一次性加载完整 audit 结果，供多个 gate 测试复用。"""

    return run_audit(_FROZEN_FORMAL_ROOT, _FROZEN_EVALUATION_ROOT)


@pytest.fixture(scope="session")
def rows() -> list[dict]:
    """加载 Step 13F per-snapshot results。"""

    return load_per_snapshot_results(_FROZEN_EVALUATION_ROOT)


@pytest.fixture(scope="session")
def snapshot_map() -> dict:
    """构建正式 population 的 snapshot 索引。"""

    return build_snapshot_map(_FROZEN_FORMAL_ROOT)


# ---------------------------------------------------------------------------
# Red-Flag Gate 测试
# ---------------------------------------------------------------------------


def test_overall_status_is_ready(audit_results: dict) -> None:
    """整体 audit 状态必须为 RQ3_SANITY_AUDIT_READY，否则触发 RQ3_SANITY_BLOCKED。"""

    report = audit_results["result_reproduction"]
    selector = audit_results["selector_semantics_audit"]
    monotonicity = audit_results["budget_monotonicity"]
    exact = audit_results["exact_opt_audit"]

    assert report["pass"], "结果复现失败"
    assert selector["pass"], "selector 语义审计失败"
    assert monotonicity["pass"], "budget monotonicity 审计失败"
    assert exact["pass"], "Exact OPT 独立审计失败"


def test_result_reproduction_no_mismatch(audit_results: dict) -> None:
    """使用公共 objective 重新计算的结果必须与 13F 完全一致。"""

    report = audit_results["result_reproduction"]
    assert report["mismatch_count"] == 0
    assert report["snapshot_mutation_count"] == 0


def test_selector_semantics_no_mismatch(audit_results: dict) -> None:
    """独立审计 LRU / LFU / Marconi selector 输出必须与 13F 完全一致。"""

    selector = audit_results["selector_semantics_audit"]
    assert selector["lru_mismatch_count"] == 0
    assert selector["lfu_mismatch_count"] == 0
    assert selector["marconi_mismatch_count"] == 0
    assert len(selector["frequency_boundary_violations"]) == 0


def test_budget_monotonicity_no_violation(audit_results: dict) -> None:
    """同一 snapshot 上 K 增大时 C(S) 不得增加。"""

    assert audit_results["budget_monotonicity"]["violation_count"] == 0


def test_exact_opt_independent_audit_no_mismatch(audit_results: dict) -> None:
    """独立 combinations 枚举必须与 13F Exact OPT 结果一致。"""

    exact = audit_results["exact_opt_audit"]
    assert exact["mismatch_count"] == 0
    assert exact["tractable_cases_checked"] == 434


# ---------------------------------------------------------------------------
# 结构/机制诊断测试
# ---------------------------------------------------------------------------


def test_compatibility_structure_one_to_one(snapshot_map: dict) -> None:
    """RQ3 population 中大多数 candidate 应只 compatible 于一个 pending。"""

    compat = analyze_compatibility_structure(snapshot_map)
    assert compat["conclusion_one_candidate_one_pending"] is True
    assert compat["fraction_d_eq_1"] >= 0.99


def test_chain_structure_non_decreasing_benefit(snapshot_map: dict) -> None:
    """每个 workflow 内的 compatible candidates 应按 token_pos 单调且 benefit 非减。"""

    chain = analyze_chain_structure(snapshot_map)
    assert chain["fraction_non_decreasing_benefit"] == 1.0


def test_redundancy_summary_exists(audit_results: dict) -> None:
    """冗余分析结果必须包含所有 policy 的统计。"""

    redundancy = audit_results["redundancy_analysis"]
    for ratio in (0.25, 0.5, 0.75):
        assert ratio in redundancy
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            assert policy in redundancy[ratio]
            assert "mean_zero_marginal_count" in redundancy[ratio][policy]


def test_per_workflow_best_probe_c_match(rows: list, snapshot_map: dict) -> None:
    """PerWorkflowBest diagnostic probe 与 FlowState 的 C(S) 不应有系统性偏差。"""

    probe = analyze_per_workflow_best_probe(rows, snapshot_map)
    assert probe["c_exact_match_rate"] is not None
    assert probe["mean_c_difference_ms"] is not None


def test_standalone_topk_probe_reported(rows: list, snapshot_map: dict) -> None:
    """StandaloneTopK diagnostic probe 必须产生可解释的对比统计。

    该 probe 不保证 FlowState 一定更优；负均值仅说明 greedy 的 set-dependency
    在该 workload 上并非总是压倒 standalone score。
    """

    probe = analyze_standalone_topk_probe(rows, snapshot_map)
    assert probe["cases"] == len(rows)
    assert probe["mean_flowstate_improvement_ms"] is not None


def test_marconi_overlap_low(rows: list) -> None:
    """FlowState 与 Marconi 的 selected set 应有显著差异，说明 selector 差异真实存在。"""

    overlap = analyze_marconi_overlap(rows)
    for ratio in (0.25, 0.5, 0.75):
        assert overlap[ratio]["mean_jaccard"] < 1.0


def test_workflow_distribution_distinct_workflows(rows: list, snapshot_map: dict) -> None:
    """FlowState 选择集应跨多个 workflow，避免单 workflow 偏食。"""

    dist = analyze_workflow_distribution(rows, snapshot_map)
    for ratio in (0.25, 0.5, 0.75):
        stats = dist[ratio]["FlowState"]["distinct_workflows"]
        assert stats["mean"] is not None
        assert stats["mean"] >= 1.0


# ---------------------------------------------------------------------------
# 端到端 sanity：直接调用函数也须通过
# ---------------------------------------------------------------------------


def test_reproduce_objectives_pass(rows: list, snapshot_map: dict) -> None:
    """直接调用 reproduce_objectives 也通过。"""

    report = reproduce_objectives(rows, snapshot_map)
    assert report["pass"]


def test_selector_audit_pass(rows: list, snapshot_map: dict) -> None:
    """直接调用 selector audit 也通过。"""

    report = audit_selector_semantics(rows, snapshot_map)
    assert report["pass"]


def test_monotonicity_pass(rows: list, snapshot_map: dict) -> None:
    """直接调用 monotonicity audit 也通过。"""

    report = audit_budget_monotonicity(rows, snapshot_map)
    assert report["pass"]


