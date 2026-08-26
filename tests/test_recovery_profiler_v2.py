"""验证 Recovery Profiler v2 的数据隔离、顺序与模型语义。"""

from __future__ import annotations

import pytest

from evaluation.recovery_profiler_v2 import analyze
from evaluation.recovery_profiler_v2.analyze import (
    ALL_GAPS,
    CALIBRATION_GAPS,
    MEASURED_REPETITIONS,
    VALIDATION_GAPS,
    WARMUP_REPETITIONS,
    compare_models,
    fit_linear_model,
    fit_piecewise_model,
    summarize_measurements,
)
from evaluation.recovery_profiler_v2.profile_runner import (
    ANCHOR_POS,
    DEEP_CHECKPOINT_ID,
    ORDER_SEED,
    SHALLOW_CHECKPOINT_ID,
    build_profile_scenario,
    build_profile_schedule,
    validate_runtime_gap,
)
from flowstate.recovery_model import RecoveryCostModel


def _synthetic_measurement_rows(
    held_out_scale: float = 1.0,
) -> tuple[dict[str, object], ...]:
    """构造只供模型切分单测使用的汇总行。"""
    rows = []
    for gap in ALL_GAPS:
        scale = held_out_scale if gap in VALIDATION_GAPS else 1.0
        rows.append(
            {
                "gap_tokens": gap,
                "measured_phi_ms": 0.04 * gap * scale,
            }
        )
    return tuple(rows)


def _synthetic_raw_records() -> tuple[dict[str, object], ...]:
    """构造包含明显异常 warmup 的完整原始样本。"""
    records = []
    case_index = 0
    for gap in ALL_GAPS:
        for repetition in range(WARMUP_REPETITIONS):
            records.append(
                {
                    "case_id": f"warmup_{case_index}",
                    "target_gap": gap,
                    "is_warmup": True,
                    "status": "PASS",
                    "gap_match": True,
                    "ttft_ms": 100_000.0 + repetition,
                }
            )
            case_index += 1
        for repetition in range(MEASURED_REPETITIONS):
            records.append(
                {
                    "case_id": f"measured_{case_index}",
                    "target_gap": gap,
                    "is_warmup": False,
                    "status": "PASS",
                    "gap_match": True,
                    "ttft_ms": 100.0 + 0.04 * gap + repetition / 10.0,
                }
            )
            case_index += 1
    return tuple(records)


def test_gap_splits_are_disjoint_and_complete() -> None:
    """确认 calibration 与 held-out validation 严格隔离。"""
    assert set(CALIBRATION_GAPS).isdisjoint(VALIDATION_GAPS)
    assert set(ALL_GAPS) == set(CALIBRATION_GAPS) | set(VALIDATION_GAPS)
    assert CALIBRATION_GAPS == (0, 4096, 8192, 16384, 32768)
    assert VALIDATION_GAPS == (2048, 6144, 12288, 24576)


def test_schedule_is_deterministic_cyclic_and_balanced() -> None:
    """确认固定 seed 顺序可复现且 measured 位置分布平衡。"""
    schedule = build_profile_schedule()
    assert schedule == build_profile_schedule(ORDER_SEED)
    assert len(schedule) == len(ALL_GAPS) * (
        WARMUP_REPETITIONS + MEASURED_REPETITIONS
    )
    assert len({case.case_id for case in schedule}) == len(schedule)
    assert tuple(case.target_gap for case in schedule[: len(ALL_GAPS)]) != (
        ALL_GAPS
    )
    for gap in ALL_GAPS:
        assert sum(
            case.target_gap == gap and case.is_warmup for case in schedule
        ) == WARMUP_REPETITIONS
        assert sum(
            case.target_gap == gap and not case.is_warmup
            for case in schedule
        ) == MEASURED_REPETITIONS
        positions = [
            case.gap_order_position
            for case in schedule
            if case.target_gap == gap and not case.is_warmup
        ]
        counts = [positions.count(index) for index in range(len(ALL_GAPS))]
        assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("gap", ALL_GAPS)
def test_profile_scenario_changes_only_executable_frontier(gap: int) -> None:
    """确认场景固定 H，并只用 recurrent 驻留状态表达目标 E。"""
    scenario, selected_ids = build_profile_scenario(gap)
    continuation = scenario.continuations[0]
    candidates = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    assert continuation.planning_target == ANCHOR_POS
    assert candidates[DEEP_CHECKPOINT_ID].token_pos == ANCHOR_POS
    if gap == 0:
        assert selected_ids == (DEEP_CHECKPOINT_ID,)
        assert set(candidates) == {DEEP_CHECKPOINT_ID}
    elif gap == ANCHOR_POS:
        assert selected_ids == ()
        assert set(candidates) == {DEEP_CHECKPOINT_ID}
    else:
        assert selected_ids == (SHALLOW_CHECKPOINT_ID,)
        assert candidates[SHALLOW_CHECKPOINT_ID].token_pos == ANCHOR_POS - gap
        assert set(candidates) == {
            SHALLOW_CHECKPOINT_ID,
            DEEP_CHECKPOINT_ID,
        }


def test_runtime_gap_requires_exact_h_e_g_match() -> None:
    """确认 profiler 不接受任何一个 token 的 runtime 偏差。"""
    assert validate_runtime_gap(
        8192,
        {
            "physical_fa_hit": 32768,
            "executable_prefix": 24576,
            "replay_gap": 8192,
        },
    ) == {"runtime_H": 32768, "runtime_E": 24576, "runtime_G": 8192}
    with pytest.raises(RuntimeError, match="runtime H 不匹配"):
        validate_runtime_gap(
            8192,
            {
                "physical_fa_hit": 32769,
                "executable_prefix": 24576,
                "replay_gap": 8193,
            },
        )
    with pytest.raises(RuntimeError, match="runtime E 不匹配"):
        validate_runtime_gap(
            8192,
            {
                "physical_fa_hit": 32768,
                "executable_prefix": 24575,
                "replay_gap": 8193,
            },
        )


def test_warmup_does_not_enter_statistics() -> None:
    """确认极端 warmup 值不会改变 measured 统计。"""
    rows = summarize_measurements(_synthetic_raw_records())
    by_gap = {int(row["gap_tokens"]): row for row in rows}
    assert by_gap[0]["ttft_mean_ms"] == pytest.approx(100.55)
    assert by_gap[8192]["measured_phi_ms"] == pytest.approx(327.68)
    assert all(row["n"] == MEASURED_REPETITIONS for row in rows)


def test_validation_values_do_not_change_fitted_models() -> None:
    """确认 held-out 数值只参与评估，不参与候选模型拟合。"""
    original = compare_models(_synthetic_measurement_rows())
    changed = compare_models(_synthetic_measurement_rows(held_out_scale=8.0))
    assert original["models"] == changed["models"]
    assert original["fit_split"]["validation_used_for_fitting"] is False
    assert original["fit_split"]["step9b_data_used_for_fitting"] is False


def test_step9b_reference_is_not_used_for_fitting(monkeypatch) -> None:
    """确认 Step 9C 对照值变化不会改变任何拟合参数。"""
    rows = _synthetic_measurement_rows()
    before = compare_models(rows)["models"]
    monkeypatch.setattr(
        analyze,
        "STEP9C_REFERENCE_INCREMENTAL_MS",
        {4096: 1e9, 8192: 1e9, 16384: 1e9, 32768: 1e9},
    )
    after = compare_models(rows)["models"]
    assert before == after


def test_all_models_satisfy_zero_and_monotonicity() -> None:
    """确认旧模型与两个候选都满足零点和单调约束。"""
    calibration = {gap: 0.04 * gap for gap in CALIBRATION_GAPS}
    linear = fit_linear_model(calibration)
    piecewise = fit_piecewise_model(calibration)
    old = RecoveryCostModel()
    assert old.estimate(0) == 0.0
    assert linear.estimate(0) == 0.0
    assert piecewise.estimate(0) == 0.0
    for model in (old, linear, piecewise):
        values = [model.estimate(gap) for gap in ALL_GAPS]
        assert values == sorted(values)


def test_piecewise_rejects_nonmonotone_calibration() -> None:
    """确认单调候选不会偷偷修正违反约束的 calibration 数据。"""
    calibration = {
        0: 0.0,
        4096: 100.0,
        8192: 90.0,
        16384: 300.0,
        32768: 600.0,
    }
    with pytest.raises(ValueError, match="单调非减"):
        fit_piecewise_model(calibration)
