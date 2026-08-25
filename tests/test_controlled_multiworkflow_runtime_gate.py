from __future__ import annotations

import pytest

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    aggregate_observations,
    validate_runtime_observation,
)


def test_retained_checkpoint_accepts_only_anchor_boundary_gap() -> None:
    observation = validate_runtime_observation(
        anchor_pos=100,
        planning_gap=0,
        metrics={
            "physical_fa_hit": 101,
            "executable_prefix": 100,
            "replay_gap": 1,
        },
    )

    assert observation == {
        "physical_hit": 101,
        "executable_prefix": 100,
        "recovery_gap": 1,
    }


def test_evicted_checkpoint_requires_full_depth_recovery() -> None:
    observation = validate_runtime_observation(
        anchor_pos=100,
        planning_gap=100,
        metrics={
            "physical_fa_hit": 101,
            "executable_prefix": 0,
            "replay_gap": 101,
        },
    )

    assert observation["recovery_gap"] == 101


def test_retained_checkpoint_rejects_large_runtime_gap() -> None:
    with pytest.raises(RuntimeError, match="executable frontier"):
        validate_runtime_observation(
            anchor_pos=100,
            planning_gap=0,
            metrics={
                "physical_fa_hit": 101,
                "executable_prefix": 80,
                "replay_gap": 21,
            },
        )


def test_runtime_observation_aggregate() -> None:
    aggregate = aggregate_observations(
        {
            "A": {
                "physical_hit": 101,
                "executable_prefix": 100,
                "recovery_gap": 1,
            },
            "B": {
                "physical_hit": 51,
                "executable_prefix": 0,
                "recovery_gap": 51,
            },
        }
    )

    assert aggregate == {
        "physical_hit_tokens": 152,
        "executable_hit_tokens": 100,
        "recovery_gap_tokens": 52,
        "executable_prefix_ratio": pytest.approx(100 / 152),
    }
