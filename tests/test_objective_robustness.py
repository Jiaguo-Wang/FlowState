"""验证 objective robustness 分析的精确性、确定性与数据隔离。"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import pytest

from evaluation.controlled_multiworkflow_v1.scenario import build_scenario
from evaluation.cost_model_sensitivity import (
    LOCAL_KNOT_GAPS,
    LOCAL_SCALE_FACTORS,
    REPRESENTATIVE_POINTS,
    build_point_scenario,
    load_profiler_v2_model,
    scenario_fingerprint,
)
from evaluation.objective_robustness import (
    EXPECTED_UNCERTAINTY_MODEL_COUNT,
    analyze_point_objective_regret,
    build_feasible_subset_space,
    build_uncertainty_models,
    objective_from_histogram,
    run_objective_robustness_analysis,
    selected_ids_from_mask,
    selection_gap_histogram,
)
from flowstate.executable_state import recovery_gap
from flowstate.recovery_model import RecoveryCostModel


@pytest.fixture(scope="module")
def frozen_models():
    """构造一次固定不确定性集合供本模块复用。"""
    return build_uncertainty_models(
        RecoveryCostModel(),
        load_profiler_v2_model(),
    )


@pytest.fixture(scope="module")
def full_result():
    """执行一次完整四点分析，避免重复精确枚举 K12。"""
    return run_objective_robustness_analysis()


def test_uncertainty_grid_is_exactly_the_frozen_set(frozen_models) -> None:
    """确认没有新增扰动，也没有混入整体比例缩放。"""
    assert len(frozen_models) == EXPECTED_UNCERTAINTY_MODEL_COUNT == 18
    assert tuple(model.model_kind for model in frozen_models[:2]) == (
        "formal",
        "profiler_v2",
    )
    local = frozen_models[2:]
    assert {
        (model.perturbed_gap, model.perturbation_fraction)
        for model in local
    } == {
        (gap, scale - 1.0)
        for gap in LOCAL_KNOT_GAPS
        for scale in LOCAL_SCALE_FACTORS
    }
    assert all("global" not in model.model_name for model in frozen_models)


def test_feasible_subset_enumeration_is_complete_and_unique() -> None:
    """用五候选小场景验证全部预算内子集恰好枚举一次。"""
    scenario = build_scenario()
    space = build_feasible_subset_space(scenario)
    subsets = tuple(space.iter_subsets())
    expected_count = sum(
        math.comb(len(scenario.candidates), size)
        for size in range(space.capacity + 1)
    )
    assert expected_count == 26
    assert len(subsets) == expected_count
    assert len({subset.selected_mask for subset in subsets}) == expected_count
    assert all(subset.selected_count <= space.capacity for subset in subsets)
    assert all(
        sum(subset.gap_counts) == len(scenario.continuations)
        for subset in subsets
    )


def test_histogram_objective_matches_direct_core_objective() -> None:
    """逐个复核小场景全部子集的 histogram 与核心直接计算。"""
    scenario = build_scenario()
    model = RecoveryCostModel()
    space = build_feasible_subset_space(scenario)
    candidate_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    for subset in space.iter_subsets():
        selected_ids = selected_ids_from_mask(space, subset.selected_mask)
        histogram = {
            gap: count
            for gap, count in zip(space.gap_values, subset.gap_counts)
            if count
        }
        selected = tuple(candidate_by_id[item] for item in selected_ids)
        direct = sum(
            model.estimate(recovery_gap(continuation, selected))
            for continuation in scenario.continuations
        )
        assert objective_from_histogram(histogram, model) == pytest.approx(
            direct
        )
        assert histogram == selection_gap_histogram(
            scenario,
            selected_ids,
        )


def test_old_phi_regret_is_zero_and_exact_optimum_dominates_formal(
    full_result,
) -> None:
    """确认正式模型 regret 为零，所有模型 exact optimum 均不劣。"""
    for point in full_result.points:
        old_phi = point.model_results[0]
        assert old_phi.model_name == "Old Phi"
        assert old_phi.absolute_regret_ms == pytest.approx(0.0, abs=1e-9)
        assert old_phi.relative_regret == pytest.approx(0.0, abs=1e-12)
        for row in point.model_results:
            assert row.optimal_cost_ms <= row.formal_cost_ms + 1e-9
            assert row.absolute_regret_ms >= 0.0


def test_analysis_is_deterministic_on_exact_k4_point(frozen_models) -> None:
    """确认输入不变时精确最优集合、成本与 tie-break 完全稳定。"""
    point = REPRESENTATIVE_POINTS[0]
    first = analyze_point_objective_regret(point, frozen_models)
    second = analyze_point_objective_regret(point, frozen_models)
    assert asdict(first) == asdict(second)


def test_full_analysis_does_not_read_step9b(monkeypatch) -> None:
    """确认成本模型构造与 exact calculation 不读取 Step 9B artifact。"""
    opened_paths = []
    original_open = Path.open

    def traced_open(path: Path, *args, **kwargs):
        opened_paths.append(path.resolve())
        if "runtime_artifacts" in path.parts:
            raise AssertionError("禁止读取 Step 9B runtime artifacts")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced_open)
    models = build_uncertainty_models(
        RecoveryCostModel(),
        load_profiler_v2_model(),
    )
    result = analyze_point_objective_regret(
        REPRESENTATIVE_POINTS[2],
        models,
    )
    assert result.uncertainty_model_count == 18
    assert not any("runtime_artifacts" in path.parts for path in opened_paths)


def test_formal_phi_and_workload_remain_unchanged(frozen_models) -> None:
    """确认精确枚举不修改正式 Phi、预算、候选或待续请求。"""
    formal_model = frozen_models[0].estimator
    values_before = tuple(
        formal_model.estimate(gap) for gap in (0,) + LOCAL_KNOT_GAPS
    )
    point = REPRESENTATIVE_POINTS[1]
    scenario = build_point_scenario(point)
    fingerprint_before = scenario_fingerprint(scenario)
    analyze_point_objective_regret(point, frozen_models)
    assert tuple(
        formal_model.estimate(gap) for gap in (0,) + LOCAL_KNOT_GAPS
    ) == values_before
    assert scenario_fingerprint(scenario) == fingerprint_before


def test_sota_k8_profiler_exact_best_matches_frozen_marconi(
    full_result,
) -> None:
    """复核关键点的 v2 exact best、Marconi 与非硬编码 regret。"""
    comparison = full_result.sota_k8_profiler_v2
    assert comparison.profiler_v2_best_equals_marconi is True
    assert set(comparison.profiler_v2_exact_best_checkpoint_ids) == set(
        comparison.marconi_checkpoint_ids
    )
    assert comparison.optimal_cost_under_v2_ms <= (
        comparison.formal_cost_under_v2_ms
    )
    expected = (
        comparison.formal_cost_under_v2_ms
        - comparison.optimal_cost_under_v2_ms
    ) / comparison.optimal_cost_under_v2_ms
    assert comparison.relative_regret == pytest.approx(expected)
    assert 0.03 < comparison.relative_regret < 0.04


def test_gap_risk_is_recorded_for_formal_and_every_exact_best(
    full_result,
) -> None:
    """确认描述性 gap 风险没有缺项且不进入 optimizer。"""
    for point in full_result.points:
        for row in point.model_results:
            assert row.formal_gap_risk.total_gap_tokens >= 0
            assert row.optimal_gap_risk.total_gap_tokens >= 0
            assert row.formal_gap_risk.max_gap_tokens >= (
                row.formal_gap_risk.p95_gap_tokens
            )
            assert row.optimal_gap_risk.max_gap_tokens >= (
                row.optimal_gap_risk.p95_gap_tokens
            )
