from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import pytest

from evaluation import recovery_model_freeze as freeze


def _formal_phi_hash() -> str:
    path = Path(__file__).resolve().parents[1] / "flowstate" / "recovery_model.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _models() -> dict[str, object]:
    return freeze.fit_candidate_models(freeze.load_calibration_points())


def test_calibration_point_freeze() -> None:
    points = freeze.load_calibration_points()
    assert len(points) == 20
    assert sum(point.subset == "position_matrix" for point in points) == 16
    assert sum(point.subset == "long_gap" for point in points) == 4
    assert len({(point.target_tokens, point.gap_tokens) for point in points}) == 20
    assert {
        point.gap_tokens for point in points if point.subset == "long_gap"
    } == {49_152, 65_536, 98_304, 131_072}


def test_calibration_does_not_use_historical_step9d() -> None:
    points = freeze.load_calibration_points()
    assert all("recovery_profiler_v2" not in point.source for point in points)
    assert all("Step 9D" not in point.source for point in points)


def test_128k_32k_point_is_not_double_weighted() -> None:
    points = freeze.load_calibration_points()
    matches = [
        point
        for point in points
        if point.target_tokens == 131_072 and point.gap_tokens == 32_768
    ]
    assert len(matches) == 1
    assert matches[0].subset == "position_matrix"


def test_heldout_points_are_exact_and_disjoint() -> None:
    heldout = set(freeze.build_heldout_pairs())
    calibration = {
        (point.target_tokens, point.gap_tokens)
        for point in freeze.load_calibration_points()
    }
    assert len(heldout) == 15
    assert {target for target, _ in heldout} == {49_152, 81_920, 114_688}
    assert len({pair for pair in heldout if pair[1] > 0}) == 12
    assert not heldout.intersection(calibration)
    assert all(0 <= gap <= target for target, gap in heldout)
    assert all(target - gap >= 0 for target, gap in heldout)


def test_heldout_schedule_is_deterministic_and_balanced() -> None:
    first = freeze.build_heldout_schedule()
    second = freeze.build_heldout_schedule()
    assert first == second
    assert len(first) == 210
    assert sum(case.is_warmup for case in first) == 30
    assert sum(not case.is_warmup for case in first) == 180
    for target, gap in freeze.build_heldout_pairs():
        matches = [
            case
            for case in first
            if case.target_position == target and case.target_gap == gap
        ]
        assert sum(case.is_warmup for case in matches) == 2
        assert sum(not case.is_warmup for case in matches) == 12
        assert all(case.target_frontier == target - gap for case in matches)


def test_candidate_formulas_are_exact() -> None:
    models = {
        "M0": {
            "parameters": {
                "knots": [
                    {"gap_ki": 0.0, "cost_ms": 0.0},
                    {"gap_ki": 8.0, "cost_ms": 80.0},
                ]
            }
        },
        "M1": {"parameters": {"a": 2.0, "b": 0.5}},
        "M2": {"parameters": {"a": 2.0, "b": 0.5, "c": -0.25}},
    }
    target = 32 * 1024
    gap = 4 * 1024
    assert freeze.predict_model("M0", models["M0"], target, gap) == 40.0
    assert freeze.predict_model("M1", models["M1"], target, gap) == 72.0
    assert freeze.predict_model("M2", models["M2"], target, gap) == 68.0
    assert freeze.predict_model("M2", models["M2"], target, 0) == 0.0


def test_m0_knots_use_gapwise_mean_without_position() -> None:
    points = freeze.load_calibration_points()
    models = freeze.fit_candidate_models(points)
    knot_rows = models["M0"]["parameters"]["knots"]
    knots = {
        int(round(float(row["gap_ki"]) * 1024)): float(row["cost_ms"])
        for row in knot_rows
    }
    expected = sum(
        point.measured_phi_ms for point in points if point.gap_tokens == 4_096
    ) / 4
    assert knots[4_096] == pytest.approx(expected)
    assert freeze.predict_model("M0", models["M0"], 32_768, 4_096) == pytest.approx(
        freeze.predict_model("M0", models["M0"], 131_072, 4_096)
    )


def test_fit_uses_ki_token_scale() -> None:
    models = _models()
    assert models["fit_units"] == "Ki token"
    assert models["heldout_used_for_fit"] is False
    assert models["M1"]["parameters"]["a"] == pytest.approx(41.6591898259)
    assert models["M1"]["parameters"]["b"] == pytest.approx(0.193514388984)
    assert models["M2"]["parameters"]["c"] == pytest.approx(-0.156201917412)


def test_calibration_diagnostics_have_all_subsets() -> None:
    points = freeze.load_calibration_points()
    rows = freeze.calibration_diagnostics(points, freeze.fit_candidate_models(points))
    assert len(rows) == 9
    assert {(row["model"], row["subset"]) for row in rows} == {
        (model, subset)
        for model in freeze.MODEL_ORDER
        for subset in ("position_matrix", "long_gap", "combined")
    }


def test_structural_grid_passes_for_frozen_candidates() -> None:
    result = freeze.structural_validation(_models())
    assert tuple(result["targets"]) == freeze.STRUCTURAL_TARGETS
    assert result["gap_step"] == 4_096
    assert all(
        result["models"][model]["status"] == "PASS"
        for model in freeze.MODEL_ORDER
    )


def test_structural_grid_detects_negative_or_nonmonotone_model() -> None:
    models = _models()
    broken = dict(models)
    broken["M1"] = {
        "formula": "a*g + b*g*t",
        "parameters": {"a": -1.0, "b": 0.0},
    }
    result = freeze.structural_validation(broken)
    assert result["models"]["M1"]["status"] == "FAIL"
    assert result["models"]["M1"]["negative_failures"]


def test_position_specific_baseline_adjustment() -> None:
    records = []
    for target in freeze.HELDOUT_TARGETS:
        for gap in freeze.HELDOUT_GAPS_BY_TARGET[target]:
            for repetition in range(2):
                records.append(
                    {
                        "target_H": target,
                        "target_gap": gap,
                        "is_warmup": True,
                        "status": "PASS",
                        "correctness_pass": True,
                        "ttft_ms": target / 1024 + gap / 1024,
                    }
                )
            for repetition in range(12):
                records.append(
                    {
                        "target_H": target,
                        "target_gap": gap,
                        "is_warmup": False,
                        "status": "PASS",
                        "correctness_pass": True,
                        "ttft_ms": target / 1024 + gap / 1024,
                    }
                )
    rows = freeze.summarize_heldout(records)
    assert len(rows) == 15
    assert all(
        row["measured_phi_ms"] == pytest.approx(row["gap_ki"])
        for row in rows
    )


@pytest.mark.parametrize(
    ("structural", "mape", "maximum", "expected"),
    (
        ("PASS", 5.0, 10.0, "PASS"),
        ("PASS", 7.0, 9.0, "WEAK"),
        ("PASS", 4.0, 15.0, "WEAK"),
        ("PASS", 10.1, 5.0, "FAIL"),
        ("PASS", 5.0, 20.1, "FAIL"),
        ("FAIL", 0.0, 0.0, "FAIL"),
    ),
)
def test_model_gate_thresholds(
    structural: str,
    mape: float,
    maximum: float,
    expected: str,
) -> None:
    assert freeze.grade_model(
        structural,
        {"mape_percent": mape, "max_relative_error_percent": maximum},
    ) == expected


def test_formal_selection_rule() -> None:
    assert freeze.select_model({"M0": "FAIL", "M1": "PASS", "M2": "PASS"})[0] == "M1"
    assert freeze.select_model({"M0": "PASS", "M1": "PASS", "M2": "PASS"})[0] == "M0"
    assert freeze.select_model({"M0": "WEAK", "M1": "WEAK", "M2": "FAIL"})[0] == "NONE"
    assert freeze.select_model({"M0": "FAIL", "M1": "FAIL", "M2": "PASS"})[0] == "M2"


def test_submodular_compatibility_check() -> None:
    models = _models()
    structural = freeze.structural_validation(models)
    assert freeze.submodular_structure_preserved("M1", models, structural)
    assert freeze.submodular_structure_preserved("M2", models, structural)
    assert not freeze.submodular_structure_preserved("NONE", models, structural)


def test_heldout_is_not_an_input_to_fitting() -> None:
    signature = inspect.signature(freeze.fit_candidate_models)
    assert tuple(signature.parameters) == ("points",)
    source = inspect.getsource(freeze.fit_candidate_models)
    assert "HELDOUT" not in source
    assert "heldout_predictions" not in source
    assert "summarize_heldout" not in source


def test_runtime_module_has_no_policy_or_trace_dependency() -> None:
    source = Path(freeze.__file__).read_text(encoding="utf-8")
    import_lines = tuple(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )
    assert all("policies" not in line for line in import_lines)
    assert all("tracelab" not in line.lower() for line in import_lines)


def test_formal_recovery_model_is_not_modified_by_offline_analysis() -> None:
    before = _formal_phi_hash()
    points = freeze.load_calibration_points()
    models = freeze.fit_candidate_models(points)
    freeze.calibration_diagnostics(points, models)
    freeze.structural_validation(models)
    after = _formal_phi_hash()
    assert before == after


def test_prediction_metrics_are_exact() -> None:
    metrics = freeze.prediction_metrics((10.0, 20.0), (11.0, 18.0))
    assert metrics["mae_ms"] == 1.5
    assert metrics["mape_percent"] == pytest.approx(10.0)
    assert metrics["max_absolute_error_ms"] == 2.0
    assert metrics["max_relative_error_percent"] == pytest.approx(10.0)
    assert math.isfinite(metrics["r_squared"])
