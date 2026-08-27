"""验证恢复成本模型敏感性分析的隔离性与精确性。"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import pytest

from evaluation.controlled_multiworkflow_v1.scenario import build_scenario
from evaluation.cost_model_sensitivity import (
    EXPECTED_PERTURBATION_COUNT,
    GLOBAL_SCALE_FACTORS,
    LOCAL_KNOT_GAPS,
    LOCAL_SCALE_FACTORS,
    ExactGapPerturbedCostModel,
    REPRESENTATIVE_POINTS,
    ScaledCostModel,
    analyze_point,
    build_point_scenario,
    classify_stability,
    evaluate_selection,
    find_ranking_margin,
    load_profiler_v2_model,
    run_sensitivity_analysis,
    scenario_fingerprint,
    select_flowstate,
)
from evaluation.recovery_profiler_v2.analyze import CALIBRATION_GAPS
from flowstate.executable_state import recovery_gap
from flowstate.recovery_model import (
    HistoricalRecoveryCostModel as RecoveryCostModel,
)


def test_profiler_v2_model_only_reads_step9d_artifact(monkeypatch) -> None:
    """确认 v2 模型加载不访问 Step 9B runtime artifacts。"""
    opened_paths = []
    original_open = Path.open

    def traced_open(path: Path, *args, **kwargs):
        opened_paths.append(path.resolve())
        if "runtime_artifacts" in path.parts:
            raise AssertionError("禁止读取 Step 9B runtime artifacts")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced_open)
    model = load_profiler_v2_model()
    assert tuple(gap for gap, _ in model.knots) == CALIBRATION_GAPS
    assert len(opened_paths) == 1
    assert opened_paths[0].name == "model_comparison.json"
    assert opened_paths[0].parent.name == "recovery_profiler_v2"


def test_fixed_perturbation_grid_is_complete() -> None:
    """确认预定义网格没有根据结果增删参数。"""
    assert GLOBAL_SCALE_FACTORS == (0.90, 0.95, 1.05, 1.10)
    assert LOCAL_SCALE_FACTORS == GLOBAL_SCALE_FACTORS
    assert LOCAL_KNOT_GAPS == (4096, 8192, 16384, 32768)
    assert 1 + len(GLOBAL_SCALE_FACTORS) + (
        len(LOCAL_KNOT_GAPS) * len(LOCAL_SCALE_FACTORS)
    ) == EXPECTED_PERTURBATION_COUNT


def test_perturbation_models_are_deterministic_and_local() -> None:
    """确认整体扰动和单 knot 扰动严格遵守定义。"""
    formal = RecoveryCostModel()
    scaled = ScaledCostModel(formal, 0.90)
    local = ExactGapPerturbedCostModel(formal, 8192, 1.10)
    assert scaled.estimate(0) == 0.0
    assert scaled.estimate(4096) == pytest.approx(
        formal.estimate(4096) * 0.90
    )
    assert local.estimate(8192) == pytest.approx(
        formal.estimate(8192) * 1.10
    )
    assert local.estimate(4096) == formal.estimate(4096)
    assert local.estimate(16384) == formal.estimate(16384)
    assert local.estimate(8192) == local.estimate(8192)


def test_analysis_preserves_budget_candidates_and_formal_phi() -> None:
    """确认反事实选择不修改冻结场景或正式成本模型。"""
    point = REPRESENTATIVE_POINTS[2]
    scenario = build_point_scenario(point)
    fingerprint_before = scenario_fingerprint(scenario)
    budget_before = scenario.budget_bytes
    candidate_ids_before = tuple(
        candidate.checkpoint_id for candidate in scenario.candidates
    )
    formal = RecoveryCostModel()
    formal_values_before = tuple(
        formal.estimate(gap) for gap in (0,) + LOCAL_KNOT_GAPS
    )
    profiler = load_profiler_v2_model()
    result = analyze_point(point, formal, profiler)
    assert result.budget_bytes == budget_before
    assert scenario.budget_bytes == budget_before
    assert tuple(
        candidate.checkpoint_id for candidate in scenario.candidates
    ) == candidate_ids_before
    assert scenario_fingerprint(scenario) == fingerprint_before
    assert tuple(
        formal.estimate(gap) for gap in (0,) + LOCAL_KNOT_GAPS
    ) == formal_values_before
    assert len(result.perturbations) == EXPECTED_PERTURBATION_COUNT
    assert tuple(
        row.model_name for row in result.perturbations[1:5]
    ) == (
        "Old Phi global -10%",
        "Old Phi global -5%",
        "Old Phi global +5%",
        "Old Phi global +10%",
    )


def test_second_best_enumeration_matches_independent_brute_force() -> None:
    """用小场景全子集枚举复核第二名和 margin。"""
    scenario = build_scenario()
    model = RecoveryCostModel()
    formal_ids = select_flowstate(scenario, model)
    margin = find_ranking_margin(scenario, formal_ids, model)
    candidates = tuple(
        sorted(scenario.candidates, key=lambda item: item.checkpoint_id)
    )
    capacity = scenario.budget_bytes // candidates[0].memory_bytes
    rows = []
    formal_set = set(formal_ids)
    for subset_size in range(capacity + 1):
        for subset in combinations(candidates, subset_size):
            selected_ids = tuple(
                candidate.checkpoint_id for candidate in subset
            )
            row = evaluate_selection(
                scenario,
                selected_ids,
                model,
                model_name="独立暴力枚举",
                formal_selected_ids=formal_ids,
            )
            rows.append((row.objective_ms, tuple(sorted(selected_ids))))
    best_cost = min(cost for cost, _ in rows)
    alternate = min(
        (row for row in rows if set(row[1]) != formal_set),
        key=lambda row: (row[0], row[1]),
    )
    assert margin.best_objective_ms == pytest.approx(best_cost)
    assert margin.second_best_objective_ms == pytest.approx(alternate[0])
    assert margin.second_best_checkpoint_ids == alternate[1]
    assert margin.margin_ms == pytest.approx(
        alternate[0] - margin.best_objective_ms
    )


def test_sota_k8_margin_matches_full_independent_enumeration() -> None:
    """对最关键近似平局点独立枚举全部可行子集。"""
    scenario = build_point_scenario(REPRESENTATIVE_POINTS[-1])
    model = RecoveryCostModel()
    formal_ids = select_flowstate(scenario, model)
    formal_set = set(formal_ids)
    margin = find_ranking_margin(scenario, formal_ids, model)
    candidates = tuple(
        sorted(scenario.candidates, key=lambda item: item.checkpoint_id)
    )
    capacity = scenario.budget_bytes // candidates[0].memory_bytes
    best_alternate = None
    for subset_size in range(capacity + 1):
        for subset in combinations(candidates, subset_size):
            selected_ids = tuple(
                candidate.checkpoint_id for candidate in subset
            )
            if set(selected_ids) == formal_set:
                continue
            cost = sum(
                model.estimate(recovery_gap(continuation, subset))
                for continuation in scenario.continuations
            )
            row = (cost, selected_ids)
            if best_alternate is None or row < best_alternate:
                best_alternate = row
    assert best_alternate is not None
    assert margin.second_best_objective_ms == pytest.approx(
        best_alternate[0]
    )
    assert margin.second_best_checkpoint_ids == best_alternate[1]


def test_formal_results_match_frozen_offline_artifacts() -> None:
    """确认四个正式选择与 Step 8 冻结离线结果完全一致。"""
    repository_root = Path(__file__).resolve().parents[1]
    sources = {
        "scalable_n16": (
            repository_root
            / "evaluation/scalable_multiworkflow_v2/offline_summary.json"
        ),
        "sota_signal": (
            repository_root
            / "evaluation/sota_signal_stress_v1/offline_summary.json"
        ),
    }
    rows_by_point = {}
    for scenario_name, path in sources.items():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload["rows"]:
            if row["policy_name"] != "FlowState":
                continue
            workflow_count = row.get("workflow_count")
            if scenario_name == "scalable_n16" and workflow_count != 16:
                continue
            rows_by_point[(scenario_name, row["budget_checkpoints"])] = row

    model = RecoveryCostModel()
    for point in REPRESENTATIVE_POINTS:
        scenario = build_point_scenario(point)
        selected_ids = select_flowstate(scenario, model)
        frozen = rows_by_point[
            (point.scenario_name, point.budget_checkpoints)
        ]
        assert selected_ids == tuple(frozen["selected_checkpoint_ids"])
        evaluated = evaluate_selection(
            scenario,
            selected_ids,
            model,
            model_name="Old Phi",
            formal_selected_ids=selected_ids,
        )
        assert evaluated.objective_ms == pytest.approx(
            frozen["estimated_recovery_cost_ms"]
        )


def test_overall_scaling_cannot_change_flowstate_selection() -> None:
    """确认正比例缩放只改变目标数值，不改变边际排序。"""
    scenario = build_point_scenario(REPRESENTATIVE_POINTS[-1])
    formal = RecoveryCostModel()
    expected = set(select_flowstate(scenario, formal))
    for scale in GLOBAL_SCALE_FACTORS:
        actual = select_flowstate(scenario, ScaledCostModel(formal, scale))
        assert set(actual) == expected


@pytest.mark.parametrize(
    ("rate", "expected"),
    (
        (1.0, "Stable"),
        (0.90, "Stable"),
        (0.899, "Moderately sensitive"),
        (0.50, "Moderately sensitive"),
        (0.499, "Sensitive"),
    ),
)
def test_stability_classification(rate: float, expected: str) -> None:
    """确认稳定性分类边界与冻结定义一致。"""
    assert classify_stability(rate) == expected


def test_full_analysis_does_not_read_step9b(monkeypatch) -> None:
    """确认四点完整分析不把 Step 9B artifact 送入 optimizer。"""
    opened_paths = []
    original_open = Path.open

    def traced_open(path: Path, *args, **kwargs):
        opened_paths.append(path.resolve())
        if "runtime_artifacts" in path.parts:
            raise AssertionError("禁止读取 Step 9B runtime artifacts")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced_open)
    result = run_sensitivity_analysis()
    assert len(result.points) == 4
    assert all(
        len(point.perturbations) == EXPECTED_PERTURBATION_COUNT
        for point in result.points
    )
    assert result.data_isolation[
        "step9b_data_used_for_optimization"
    ] is False
    assert not any("runtime_artifacts" in path.parts for path in opened_paths)
