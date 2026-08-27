from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from flowstate.recovery_model import (
    FORMAL_RECOVERY_MODEL_METADATA,
    HistoricalRecoveryCostModel,
    RecoveryCostModel,
)


@pytest.fixture(scope="module")
def model() -> RecoveryCostModel:
    return RecoveryCostModel()


def test_formal_metadata_is_frozen() -> None:
    metadata = FORMAL_RECOVERY_MODEL_METADATA
    assert metadata.name == "position_aware_quadratic_v1"
    assert metadata.coefficient_a == 37.828150
    assert metadata.coefficient_b == 0.345974143
    assert metadata.coefficient_c == -0.156201917
    assert metadata.calibration_artifact == (
        "recovery_model_freeze_20260826_154235_266020"
    )
    assert metadata.maximum_target_tokens == 131_072
    assert metadata.output_unit == "ms"


def test_zero_gap_is_exact_zero(model: RecoveryCostModel) -> None:
    for target_tokens in (0, 32_768, 65_536, 131_072):
        assert model.estimate(0, target_tokens) == 0.0
        assert model.cost(0, target_tokens) == 0.0


def test_exact_m2_formula_and_ki_token_conversion(
    model: RecoveryCostModel,
) -> None:
    gap_tokens = 8_192
    target_tokens = 65_536
    gap_ki_tokens = 8.0
    target_ki_tokens = 64.0
    expected = (
        37.828150 * gap_ki_tokens
        + 0.345974143 * gap_ki_tokens * target_ki_tokens
        - 0.156201917 * gap_ki_tokens * gap_ki_tokens
    )
    assert model.estimate(gap_tokens, target_tokens) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("gap_tokens", "target_tokens", "message"),
    [
        (-1, 0, "gap_tokens"),
        (0, -1, "target_tokens"),
        (4_096, 2_048, "不能大于"),
        (4_096, 131_073, "超出"),
    ],
)
def test_invalid_inputs_are_rejected(
    model: RecoveryCostModel,
    gap_tokens: int,
    target_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        model.estimate(gap_tokens, target_tokens)


def test_formal_domain_is_nonnegative_and_fixed_target_monotonic(
    model: RecoveryCostModel,
) -> None:
    for target_tokens in range(0, 131_073, 4_096):
        gaps = tuple(range(0, target_tokens + 1, 4_096))
        costs = tuple(
            model.estimate(gap_tokens, target_tokens)
            for gap_tokens in gaps
        )
        assert all(cost >= 0.0 for cost in costs)
        assert costs == tuple(sorted(costs))


def test_derivative_is_positive_over_continuous_domain(
    model: RecoveryCostModel,
) -> None:
    for target_tokens in range(0, 131_073, 4_096):
        for gap_tokens in range(0, target_tokens + 1, 4_096):
            assert (
                model.derivative_ms_per_ki_token(
                    gap_tokens,
                    target_tokens,
                )
                > 0.0
            )


def test_historical_model_remains_explicitly_available() -> None:
    historical = HistoricalRecoveryCostModel()
    assert historical.estimate(0) == 0.0
    assert historical.estimate(32_768) == pytest.approx(1499.5078251231462)


def test_historical_non_monotonic_profile_is_rejected(tmp_path: Path) -> None:
    profile_path = tmp_path / "replay_cost.csv"
    with profile_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "expected_replay_tokens",
                "recovery_latency_median_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "expected_replay_tokens": 0,
                    "recovery_latency_median_ms": 10.0,
                },
                {
                    "expected_replay_tokens": 1024,
                    "recovery_latency_median_ms": 30.0,
                },
                {
                    "expected_replay_tokens": 4096,
                    "recovery_latency_median_ms": 29.0,
                },
            ]
        )
    fit_path = tmp_path / "fit_metrics.json"
    fit_path.write_text(
        json.dumps(
            {
                "raw_run_ols": {
                    "intercept_ms": 10.0,
                    "slope_ms_per_token": 0.01,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="1024 token.*4096 token"):
        HistoricalRecoveryCostModel(profile_path, fit_path)
