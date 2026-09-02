from __future__ import annotations

import inspect

from evaluation.openhands_policy_paired_ttft_runtime_gate import (
    ENGINE_CONFIGURATION_PAIRED_TTFT,
    ENGINE_LIFECYCLE_COUNT,
    POLICY_ORDER,
    TRIAL_COUNT,
    _sample_statistics,
    audit_paired_runtime_semantics,
    build_paired_trial,
    build_summary,
    summarize_policy_run,
)
from evaluation.openhands_policy_runtime_heg_outcome_gate import (
    ENGINE_CONFIGURATION_HEG_OUTCOME,
)


POSITIONS = {"A": 400, "B": 240, "C": 370, "D": 350}


def outcome(
    label: str,
    *,
    h: int,
    e: int,
    ttft: float,
    digest: str | None = None,
):
    """构造一个正式测量请求的最小有效记录。"""
    return {
        "workflow": label,
        "status": "PASS",
        "h": h,
        "e": e,
        "g": h - e,
        "ttft_ms": ttft,
        "request_latency_ms": ttft + 1.0,
        "cached_tokens": e,
        "server_prompt_tokens": 1_000,
        "input_token_digest": digest or f"摘要-{label}",
    }


def policy_run(
    condition: str,
    ttft_offset: float = 0.0,
    *,
    digest_override: tuple[str, str] | None = None,
):
    """构造符合 12H.6 policy-specific 语义的运行结果。"""
    if condition == "LM":
        states = {
            "A": (500, 0),
            "B": (300, 0),
            "C": (370, 370),
            "D": (350, 350),
        }
    else:
        states = {
            "A": (400, 400),
            "B": (300, 0),
            "C": (370, 370),
            "D": (500, 240),
        }
    records = []
    for order, label in enumerate("ABCD", start=1):
        digest = None
        if digest_override is not None and label == digest_override[0]:
            digest = digest_override[1]
        h, e = states[label]
        records.append(
            outcome(
                label,
                h=h,
                e=e,
                ttft=10.0 * order + ttft_offset,
                digest=digest,
            )
        )
    return {
        "condition": condition,
        "status": "PASS",
        "engine_ordinal": 1 if condition == "LM" else 2,
        "engine_lifecycle": "independent_fresh",
        "candidate_rows": [
            {"workflow_label": label, "token_pos": position}
            for label, position in POSITIONS.items()
        ],
        "outcome_records": records,
        "mapping_invariants": {"selected_residency_exact": True},
        "native_mamba_capacity_eviction": False,
        "fa_kv_cascade": False,
        "future_information_used": False,
    }


def clean_boundary():
    """构造没有未来信息泄漏的审计行。"""
    return [
        {
            "r_plus_2_message_consumed": False,
            "r_plus_2_request_materialized": False,
            "pending_assistant_output_read": False,
        }
    ]


def test_five_pairs_prepare_ten_fresh_engine_lifecycles() -> None:
    assert TRIAL_COUNT == 5
    assert POLICY_ORDER == ("LM", "F")
    assert ENGINE_LIFECYCLE_COUNT == 10


def test_engine_configuration_is_the_frozen_h6_configuration_copy() -> None:
    assert ENGINE_CONFIGURATION_PAIRED_TTFT == (
        ENGINE_CONFIGURATION_HEG_OUTCOME
    )
    assert ENGINE_CONFIGURATION_PAIRED_TTFT is not (
        ENGINE_CONFIGURATION_HEG_OUTCOME
    )
    assert ENGINE_CONFIGURATION_PAIRED_TTFT["max_mamba_cache_size"] == 28


def test_policy_run_summary_uses_only_a2_through_d2() -> None:
    summary = summarize_policy_run(policy_run("LM"))
    assert summary["aggregate_ttft_ms"] == 100.0
    assert summary["aggregate_latency_ms"] == 104.0
    assert summary["aggregate_g"] == 800
    assert set(summary["outcomes"]) == set("ABCD")


def test_paired_delta_is_flowstate_minus_lm() -> None:
    trial = build_paired_trial(
        1,
        policy_run("LM"),
        policy_run("F", -1.0),
    )
    assert trial["status"] == "PASS"
    assert trial["lm"]["aggregate_ttft_ms"] == 100.0
    assert trial["flowstate"]["aggregate_ttft_ms"] == 96.0
    assert trial["paired_delta_ms"] == -4.0
    assert trial["flowstate_faster"] is True
    assert trial["workflow_paired_delta_ms"] == {
        "A": -1.0,
        "B": -1.0,
        "C": -1.0,
        "D": -1.0,
    }


def test_runtime_semantics_cover_all_four_workflows() -> None:
    audit = audit_paired_runtime_semantics(
        policy_run("LM"),
        policy_run("F"),
    )
    assert audit["status"] == "PASS"
    assert audit["all_g_equals_h_minus_e"] is True
    assert audit["policy_specific_heg_reproduced"] is True
    assert audit["workflow_semantics"] == {
        "A": True,
        "B": True,
        "C": True,
        "D": True,
    }


def test_runtime_semantics_reject_wrong_a_retention_outcome() -> None:
    flowstate = policy_run("F")
    flowstate["outcome_records"][0]["e"] = 0
    flowstate["outcome_records"][0]["g"] = 400
    audit = audit_paired_runtime_semantics(
        policy_run("LM"),
        flowstate,
    )
    assert audit["status"] == "FAIL"
    assert audit["workflow_semantics"]["A"] is False


def test_runtime_semantics_reject_input_mismatch() -> None:
    audit = audit_paired_runtime_semantics(
        policy_run("LM"),
        policy_run("F", digest_override=("D", "不同摘要")),
    )
    assert audit["status"] == "FAIL"
    assert audit["request_input_ids_equal"] is False


def test_descriptive_statistics_do_not_perform_significance_test() -> None:
    stats = _sample_statistics([-4.0, -2.0, 0.0, 2.0, 4.0])
    assert stats["mean"] == 0.0
    assert stats["median"] == 0.0
    assert stats["min"] == -4.0
    assert stats["max"] == 4.0
    assert stats["sample_std"] is not None


def test_summary_calculates_five_pair_statistics_without_leakage(
    tmp_path,
) -> None:
    trials = [
        build_paired_trial(
            index,
            policy_run("LM"),
            policy_run("F", -float(index)),
        )
        for index in range(1, 6)
    ]
    summary = build_summary(
        artifact=tmp_path,
        trials=trials,
        boundary_audit=clean_boundary(),
        environment=None,
    )
    assert summary["status"] == "PASS"
    assert summary["valid_pair_count"] == 5
    assert summary["paired_summary"]["flowstate_faster_count"] == 5
    assert summary["paired_summary"]["paired_delta_ms"]["values"] == [
        -4.0,
        -8.0,
        -12.0,
        -16.0,
        -20.0,
    ]
    assert summary["paired_summary"]["paired_delta_ms"]["mean"] == -12.0
    assert summary["paired_summary"]["paired_delta_ms"]["median"] == -12.0
    assert summary["future_leakage"] is False
    assert (
        summary["runtime_correctness"][
            "same_request_inputs_across_trials"
        ]
        is True
    )
    assert (
        summary["runtime_correctness"][
            "same_candidate_positions_across_trials"
        ]
        is True
    )
    assert summary["statistical_significance_test_performed"] is False
    assert summary["statistical_significance_claim"] is False


def test_summary_marks_incomplete_pairs_partial(tmp_path) -> None:
    trials = [
        build_paired_trial(1, policy_run("LM"), policy_run("F"))
    ]
    summary = build_summary(
        artifact=tmp_path,
        trials=trials,
        boundary_audit=clean_boundary(),
        environment=None,
    )
    assert summary["status"] == "FAIL"
    assert summary["verdict"] == "PARTIAL"
    assert summary["valid_pair_count"] == 1


def test_summary_rejects_cross_trial_input_changes(tmp_path) -> None:
    trials = [
        build_paired_trial(
            index,
            policy_run(
                "LM",
                digest_override=(
                    ("D", "变化摘要") if index == 5 else None
                ),
            ),
            policy_run(
                "F",
                digest_override=(
                    ("D", "变化摘要") if index == 5 else None
                ),
            ),
        )
        for index in range(1, 6)
    ]
    summary = build_summary(
        artifact=tmp_path,
        trials=trials,
        boundary_audit=clean_boundary(),
        environment=None,
    )
    assert summary["status"] == "FAIL"
    assert (
        summary["runtime_correctness"][
            "same_request_inputs_across_trials"
        ]
        is False
    )


def test_gate_source_has_alternating_fresh_runs_and_no_future_turns() -> None:
    module = __import__(
        "evaluation.openhands_policy_paired_ttft_runtime_gate",
        fromlist=["unused"],
    )
    source = inspect.getsource(module)
    assert "for trial_index in range(1, TRIAL_COUNT + 1)" in source
    assert "for policy_offset, condition in enumerate(POLICY_ORDER)" in source
    assert "run_condition(" in source
    assert "PENDING_TURN + 1" not in source
    assert "statistical_significance_test_performed" in source
