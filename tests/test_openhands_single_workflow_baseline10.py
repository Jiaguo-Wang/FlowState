from __future__ import annotations

from pathlib import Path

from evaluation.openhands_single_workflow_baseline10 import (
    ALIGNMENT_TOKENS,
    build_summary,
    exact_lcp,
)


def test_exact_lcp_stops_at_first_difference() -> None:
    assert exact_lcp([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert exact_lcp([1, 2], [1, 2, 3]) == 2


def test_alignment_matches_frozen_hybrid_chunk() -> None:
    assert ALIGNMENT_TOKENS == 64
    assert 4011 // ALIGNMENT_TOKENS * ALIGNMENT_TOKENS == 3968
    assert 4490 // ALIGNMENT_TOKENS * ALIGNMENT_TOKENS == 4480


def test_summary_accepts_valid_runtime_records(tmp_path: Path) -> None:
    records = []
    for turn in range(1, 11):
        records.append(
            {
                "turn": turn,
                "offline_input_tokens": 4000 + turn,
                "request_completed": True,
                "token_count_exact": True,
                "prefix_reuse_exact": None if turn == 1 else True,
                "runtime_metrics_available": True,
                "runtime_bounds_valid": True,
                "runtime_gap_valid": True,
                "oom": False,
                "truncation_or_clipping": False,
            }
        )

    summary = build_summary(
        records=records,
        artifact=tmp_path,
        n_turns=84,
        fatal_error=None,
        environment={},
    )

    assert summary["status"] == "PASS"
    assert summary["prefix_reuse_consistency"] == 9
    assert summary["runtime_semantic_status"] == "PASS"
