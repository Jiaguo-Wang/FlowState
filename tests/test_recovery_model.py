import csv
import json
from pathlib import Path

import pytest

from flowstate.recovery_model import RecoveryCostModel


_ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "motivation"
    / "artifacts"
    / "replay_cost_20260819"
)


@pytest.fixture(scope="module")
def model() -> RecoveryCostModel:
    return RecoveryCostModel()


def test_zero_replay_has_zero_cost(model: RecoveryCostModel) -> None:
    assert model.estimate(0) == 0.0


@pytest.mark.parametrize(
    ("replay_tokens", "expected_cost_ms"),
    [
        (1024, 31.80306009016931),
        (4096, 178.72895603068173),
        (8192, 378.5894282627851),
        (16384, 772.4913060665131),
        (32768, 1499.5078251231462),
    ],
)
def test_profile_points_have_positive_recovery_cost(
    model: RecoveryCostModel,
    replay_tokens: int,
    expected_cost_ms: float,
) -> None:
    cost_ms = model.estimate(replay_tokens)
    assert cost_ms > 0.0
    assert cost_ms == pytest.approx(expected_cost_ms)


def test_cost_is_monotonic_non_decreasing(model: RecoveryCostModel) -> None:
    replay_lengths = [0, 1024, 2048, 4096, 8192, 16384, 32768, 40000]
    costs = [model.estimate(replay_tokens) for replay_tokens in replay_lengths]
    assert costs == sorted(costs)


def test_interpolation_for_middle_value(model: RecoveryCostModel) -> None:
    lower_cost = model.estimate(1024)
    middle_cost = model.estimate(2048)
    upper_cost = model.estimate(4096)
    assert lower_cost < middle_cost < upper_cost


def test_extrapolation_uses_existing_fit(model: RecoveryCostModel) -> None:
    with (_ARTIFACT_DIRECTORY / "fit_metrics.json").open(
        "r", encoding="utf-8"
    ) as handle:
        fit_metrics = json.load(handle)
    slope = float(fit_metrics["raw_run_ols"]["slope_ms_per_token"])

    cost_ms = model.estimate(40000)
    assert cost_ms > 0.0
    assert cost_ms == pytest.approx(slope * 40000)


def test_negative_replay_tokens_raise_value_error(
    model: RecoveryCostModel,
) -> None:
    with pytest.raises(ValueError, match="必须大于等于零"):
        model.estimate(-1)


def test_non_monotonic_profile_is_rejected(tmp_path: Path) -> None:
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
        RecoveryCostModel(profile_path, fit_path)


def test_wp3b_recovery_sanity(model: RecoveryCostModel) -> None:
    penalty_ms = model.estimate(32768) - model.estimate(0)
    print(f"WP3B 恢复成本校验：Phi(32768) - Phi(0) = {penalty_ms:.3f} ms")
    assert 1000.0 < penalty_ms < 2000.0
