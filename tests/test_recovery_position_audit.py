"""验证恢复成本位置审计的冻结设计与离线诊断。"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from evaluation.recovery_position_audit import (
    EXPECTED_LEGACY_TRIALS,
    EXPECTED_MATRIX_TRIALS,
    MEASURED_REPETITIONS,
    POSITION_GAPS,
    TARGET_POSITIONS,
    WARMUP_REPETITIONS,
    analyze_position_dependence,
    build_legacy_schedule,
    build_matrix_schedule,
    build_protocol_diff,
    classify_discrepancy,
    diagnostic_models,
    grade_gap_only_assumption,
    grade_legacy_reproduction,
    leave_one_position_out_splits,
    summarize_legacy,
    summarize_position_matrix,
)
from evaluation.recovery_profiler_128k import build_position_scenario


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "evaluation" / "recovery_position_audit.py"
FORMAL_PHI_HASH = "f3fe216592ad62c26e5bf7936f907823745942f7f34b483b8dfbc2fbd8fda1f5"
TRACE_PROTOCOL_HASH = "bc39435266dc97fd9c55e9cf13e215f1ece2481e8f0cee85c36512c3a08688d4"


def test_fixed_position_and_gap_sets() -> None:
    """T、G、重复次数和总 trial 数必须固定。"""
    assert TARGET_POSITIONS == (32_768, 65_536, 98_304, 131_072)
    assert POSITION_GAPS == (0, 4_096, 8_192, 16_384, 32_768)
    assert WARMUP_REPETITIONS == 2
    assert MEASURED_REPETITIONS == 12
    assert EXPECTED_LEGACY_TRIALS == 70
    assert EXPECTED_MATRIX_TRIALS == 280


def test_matrix_schedule_is_deterministic_balanced_and_respects_identity() -> None:
    """每个 T/G 单元必须有相同重复数且严格满足 E=T-G。"""
    schedule = build_matrix_schedule()
    assert schedule == build_matrix_schedule()
    assert len(schedule) == EXPECTED_MATRIX_TRIALS
    for target in TARGET_POSITIONS:
        for gap in POSITION_GAPS:
            selected = tuple(
                case
                for case in schedule
                if case.target_position == target and case.target_gap == gap
            )
            assert sum(case.is_warmup for case in selected) == 2
            assert sum(not case.is_warmup for case in selected) == 12
            assert all(
                case.target_frontier == target - gap for case in selected
            )


def test_legacy_schedule_reuses_step9d_cases() -> None:
    """Legacy 阶段只过滤 Step 9D 原计划，不重写旧 case。"""
    schedule = build_legacy_schedule()
    assert len(schedule) == EXPECTED_LEGACY_TRIALS
    assert {case.target_gap for case in schedule} == set(POSITION_GAPS)
    assert all(case.target_frontier == 32_768 - case.target_gap for case in schedule)


@pytest.mark.parametrize("target", TARGET_POSITIONS)
@pytest.mark.parametrize("gap", POSITION_GAPS)
def test_generalized_scenario_has_exact_target_frontier(target: int, gap: int) -> None:
    """通用场景必须保持 T、E、G 和 candidate lineage 语义。"""
    scenario, selected_ids = build_position_scenario(target, gap)
    continuation = scenario.continuations[0]
    assert continuation.planning_target == target
    assert max(candidate.token_pos for candidate in scenario.candidates) == target
    selected = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
        if candidate.checkpoint_id in selected_ids
    }
    expected_frontier = target - gap
    if expected_frontier == 0:
        assert not selected
    else:
        assert max(candidate.token_pos for candidate in selected.values()) == expected_frontier


def test_position_specific_baselines_are_independent() -> None:
    """每个 T 必须减去自己的 G=0 baseline。"""
    records = _synthetic_records(position_slope=0.0)
    rows = summarize_position_matrix(records)
    by_key = {
        (row["target_position"], row["gap_tokens"]): row for row in rows
    }
    for target in TARGET_POSITIONS:
        assert by_key[(target, 0)]["measured_phi_ms"] == 0.0
        assert by_key[(target, 4_096)]["measured_phi_ms"] == pytest.approx(204.8)
        assert by_key[(target, 0)]["position_baseline_mean_ms"] == pytest.approx(
            20.0055 + target / 1024.0
        )


def test_position_range_ratio_and_gate_thresholds() -> None:
    """位置范围比例和 5%/10% gate 必须按预注册边界判定。"""
    pass_rows = _gap_rows((100.0, 102.0, 101.0, 103.0))
    weak_rows = _gap_rows((100.0, 106.0, 103.0, 104.0))
    fail_rows = _gap_rows((100.0, 104.0, 108.0, 112.0))
    assert grade_gap_only_assumption(pass_rows) == "PASS"
    assert grade_gap_only_assumption(weak_rows) == "WEAK"
    assert grade_gap_only_assumption(fail_rows) == "FAIL"


def test_position_dependence_reports_same_gap_trends() -> None:
    """相同 G 随 T 增长时必须触发 gap-only FAIL。"""
    records = _synthetic_records(position_slope=0.000002)
    rows = summarize_position_matrix(records)
    result = analyze_position_dependence(rows)
    assert result["gap_only_assumption"] == "FAIL"
    assert all(
        row["trend"] == "单调增长" for row in result["gap_rows"]
    )
    assert result["step10d1_sublinear_consistency"] == "YES"


def test_legacy_reproduction_uses_own_baseline_and_fixed_grade() -> None:
    """历史复现误差必须以旧协议自己的 G=0 为基线。"""
    historical = {gap: 0.05 * gap for gap in POSITION_GAPS}
    records = []
    for gap in POSITION_GAPS:
        for repetition in range(MEASURED_REPETITIONS):
            records.append(
                {
                    "target_gap": gap,
                    "is_warmup": False,
                    "status": "PASS",
                    "correctness_pass": True,
                    "gap_match": True,
                    "ttft_ms": 77.0 + historical[gap],
                }
            )
    result = summarize_legacy(records, historical)
    assert result["grade"] == "PASS"
    assert grade_legacy_reproduction((0.05, 0.01)) == "PASS"
    assert grade_legacy_reproduction((0.0501,)) == "WEAK"
    assert grade_legacy_reproduction((0.1001,)) == "FAIL"


def test_leave_one_position_out_has_four_disjoint_folds() -> None:
    """LOPO 必须依次只留出一个 T，训练集包含其余三个 T。"""
    splits = leave_one_position_out_splits()
    assert len(splits) == 4
    assert {row["held_out_target"] for row in splits} == set(TARGET_POSITIONS)
    for row in splits:
        assert row["held_out_target"] not in row["training_targets"]
        assert len(row["training_targets"]) == 3


def test_position_aware_diagnostic_improves_position_signal() -> None:
    """双线性模型应能识别预先注入的位置项。"""
    rows = summarize_position_matrix(_synthetic_records(position_slope=0.000002))
    result = diagnostic_models(rows)
    assert result["position_aware"]["mae_ms"] < result["gap_only"]["mae_ms"]
    assert (
        result["position_aware"]["leave_one_position_out"]["mape_percent"]
        < result["gap_only"]["leave_one_position_out"]["mape_percent"]
    )


def test_discrepancy_classification_is_preregistered() -> None:
    """分类只由 legacy 与 position 两个 gate 的组合决定。"""
    assert classify_discrepancy("PASS", "FAIL") == "POSITION_DEPENDENCE"
    assert classify_discrepancy("FAIL", "PASS") == "ENVIRONMENT_OR_PROTOCOL_DRIFT"
    assert classify_discrepancy("FAIL", "FAIL") == "MIXED"
    assert classify_discrepancy("PASS", "PASS") == "UNRESOLVED"


def test_protocol_diff_records_absolute_positions() -> None:
    """协议差异必须逐 gap 公开 Step 9D 与 Step 10D.1 的 T/E/G。"""
    diff = build_protocol_diff()
    assert diff["step9d"]["positions"]["4096"] == {
        "T": 32_768,
        "E": 28_672,
        "G": 4_096,
    }
    assert diff["step10d1"]["positions"]["4096"] == {
        "T": 131_072,
        "E": 126_976,
        "G": 4_096,
    }
    assert diff["step9d"]["engine_configuration"]["context_length"] == 45_056
    assert diff["step10d1"]["engine_configuration"]["context_length"] == 131_200


def test_source_has_no_policy_dependency() -> None:
    """位置审计不得导入或执行任何 allocation policy。"""
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("polic" in name.lower() for name in imported)


def test_formal_phi_and_tracelab_protocol_are_unchanged() -> None:
    """本步骤不得改写正式 Phi 或 TraceLab 冻结协议。"""
    assert hashlib.sha256(
        (ROOT / "flowstate" / "recovery_model.py").read_bytes()
    ).hexdigest() == FORMAL_PHI_HASH
    assert hashlib.sha256(
        (
            ROOT
            / "evaluation"
            / "public_agent_trace"
            / "tracelab_nontrivial_protocol.py"
        ).read_bytes()
    ).hexdigest() == TRACE_PROTOCOL_HASH


def _synthetic_records(position_slope: float) -> list[dict[str, object]]:
    """构造具有独立 position baseline 和可控位置项的样本。"""
    records = []
    for target in TARGET_POSITIONS:
        baseline = 20.0 + target / 1024.0
        for gap in POSITION_GAPS:
            phi = 0.05 * gap + position_slope * gap * target
            for repetition in range(MEASURED_REPETITIONS):
                records.append(
                    {
                        "target_H": target,
                        "target_gap": gap,
                        "is_warmup": False,
                        "status": "PASS",
                        "correctness_pass": True,
                        "gap_match": True,
                        "ttft_ms": baseline + phi + repetition / 1000.0,
                    }
                )
    return records


def _gap_rows(values: tuple[float, ...]) -> tuple[dict[str, object], ...]:
    """构造四个 gap 使用同一位置变化形状的 gate 输入。"""
    mean_value = sum(values) / len(values)
    ratio = (max(values) - min(values)) / mean_value
    monotonic = all(left <= right for left, right in zip(values, values[1:]))
    significant = monotonic and (values[-1] - values[0]) / mean_value > 0.05
    return tuple(
        {
            "gap_tokens": gap,
            "position_range_ratio": ratio,
            "significant_monotonic_trend": significant,
        }
        for gap in POSITION_GAPS[1:]
    )
