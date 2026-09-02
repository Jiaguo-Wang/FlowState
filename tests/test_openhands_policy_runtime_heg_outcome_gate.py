from __future__ import annotations

import inspect

from evaluation.openhands_common_barrier_snapshot_gate import SCHEDULE
from evaluation.openhands_policy_runtime_heg_outcome_gate import (
    ENGINE_CONFIGURATION_HEG_OUTCOME,
    PENDING_SCHEDULE,
    RUN_ORDER,
    attach_barrier_prediction,
    build_summary,
    compare_policy_outcomes,
    prepare_outcome_requests,
    validate_condition_outcomes,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    ENGINE_CONFIGURATION_ACTUATOR_MAPPING,
    EXPECTED_EVICTIONS,
    SELECTED_SETS,
)


def source_requests():
    """构造两个条件共享的四个 pending 输入。"""
    return {
        (label, 2): {
            "workflow_label": label,
            "workflow_id": f"workflow-{label}",
            "turn": 2,
            "rid": f"原始-{label}",
            "input_ids": [order, order + 10, order + 20],
        }
        for order, label in enumerate("ABCD", start=1)
    }


def prediction(label: str, h_pred: int = 90):
    """构造 barrier FA frontier 预测行。"""
    return {
        "workflow_label": label,
        "anchor_pos": 100,
        "resident_fa_frontier": h_pred,
        "planning_target": min(100, h_pred),
    }


def outcome(
    label: str,
    *,
    h: int,
    e: int,
    h_pred: int | None = None,
    digest: str | None = None,
):
    """构造已通过 token 与 runtime 关系检查的 outcome。"""
    if h_pred is None:
        h_pred = h
    return {
        "workflow": label,
        "workflow_id": f"workflow-{label}",
        "turn": 2,
        "status": "PASS",
        "request_completed": True,
        "token_count_exact": True,
        "runtime_metrics_valid": True,
        "offline_input_tokens": 1_000,
        "server_prompt_tokens": 1_000,
        "cached_tokens": h,
        "h": h,
        "e": e,
        "g": h - e,
        "h_pred_barrier": h_pred,
        "h_actual_minus_h_pred": h - h_pred,
        "barrier_prediction_equality_required": label == "A",
        "barrier_prediction_exact": None if label != "A" else h == h_pred,
        "input_token_digest": digest or f"摘要-{label}",
    }


def run(condition: str, records):
    """构造跨条件 outcome 比较需要的最小 run。"""
    positions = {"A": 400, "B": 240, "C": 370, "D": 350}
    return {
        "condition": condition,
        "status": "PASS",
        "engine_lifecycle": "independent_fresh",
        "candidate_rows": [
            {"workflow_label": label, "token_pos": position}
            for label, position in positions.items()
        ],
        "outcome_records": records,
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


def test_schedules_and_selected_sets_are_frozen() -> None:
    assert tuple(f"{label}{turn}" for label, turn in SCHEDULE) == (
        "A1",
        "B1",
        "C1",
        "D1",
    )
    assert tuple(f"{label}{turn}" for label, turn in PENDING_SCHEDULE) == (
        "A2",
        "B2",
        "C2",
        "D2",
    )
    assert RUN_ORDER == ("LM", "F")
    assert SELECTED_SETS["LM"][-2:] == (
        "OPENHANDS_BARRIER_C_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_001",
    )
    assert EXPECTED_EVICTIONS["F"] == (
        "OPENHANDS_BARRIER_B_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_001",
    )


def test_engine_configuration_is_an_unmodified_copy() -> None:
    assert ENGINE_CONFIGURATION_HEG_OUTCOME == (
        ENGINE_CONFIGURATION_ACTUATOR_MAPPING
    )
    assert ENGINE_CONFIGURATION_HEG_OUTCOME is not (
        ENGINE_CONFIGURATION_ACTUATOR_MAPPING
    )
    assert ENGINE_CONFIGURATION_HEG_OUTCOME["max_mamba_cache_size"] == 28


def test_two_conditions_use_identical_tokens_and_distinct_rids() -> None:
    requests = source_requests()
    lm = prepare_outcome_requests(requests, "LM")
    flowstate = prepare_outcome_requests(requests, "F")
    assert [item["workflow_label"] for item in lm] == list("ABCD")
    assert [item["input_ids"] for item in lm] == [
        item["input_ids"] for item in flowstate
    ]
    assert [item["input_token_digest"] for item in lm] == [
        item["input_token_digest"] for item in flowstate
    ]
    assert set(item["rid"] for item in lm).isdisjoint(
        item["rid"] for item in flowstate
    )


def test_barrier_prediction_equality_only_applies_to_a2() -> None:
    a = attach_barrier_prediction(
        {"workflow": "A", "h": 90},
        prediction("A", 90),
    )
    b = attach_barrier_prediction(
        {"workflow": "B", "h": 95},
        prediction("B", 90),
    )
    assert a["barrier_prediction_equality_required"] is True
    assert a["barrier_prediction_exact"] is True
    assert b["barrier_prediction_equality_required"] is False
    assert b["barrier_prediction_exact"] is None
    assert b["h_actual_minus_h_pred"] == 5
    assert b["sequential_runtime_evolution_allowed"] is True


def test_condition_validation_requires_valid_heg_and_exact_a2_prediction() -> None:
    records = [
        outcome(label, h=500 + order, e=400)
        for order, label in enumerate("ABCD")
    ]
    result = validate_condition_outcomes(records)
    assert result["status"] == "PASS"
    assert result["a2_h_pred_equals_h_actual"] is True
    records[0] = {**records[0], "barrier_prediction_exact": False}
    assert validate_condition_outcomes(records)["status"] == "FAIL"


def test_policy_comparison_checks_a_and_d_selected_set_signals() -> None:
    lm_records = [
        outcome("A", h=500, e=300),
        outcome("B", h=300, e=200),
        outcome("C", h=500, e=370),
        outcome("D", h=500, e=350),
    ]
    flow_records = [
        outcome("A", h=500, e=400),
        outcome("B", h=300, e=200),
        outcome("C", h=500, e=370),
        outcome("D", h=500, e=240),
    ]
    comparison = compare_policy_outcomes(
        run("LM", lm_records),
        run("F", flow_records),
    )
    assert comparison["status"] == "PASS"
    assert comparison["request_input_ids_equal"] is True
    assert comparison["a2_selected_set_signal"] is True
    assert comparison["d2_selected_set_signal"] is True
    assert comparison["c2_retained_reference_valid"] is True
    assert comparison["eg_differences"]["A"] is True
    assert comparison["eg_differences"]["D"] is True
    assert comparison["at_least_one_runtime_heg_outcome_differs"] is True
    assert comparison["outcome_difference_required_for_correctness"] is False


def test_policy_comparison_records_missing_d2_effect_without_failing() -> None:
    lm_records = [
        outcome("A", h=500, e=300),
        outcome("B", h=300, e=200),
        outcome("C", h=500, e=370),
        outcome("D", h=500, e=350),
    ]
    flow_records = [
        outcome("A", h=500, e=400),
        outcome("B", h=300, e=200),
        outcome("C", h=500, e=370),
        outcome("D", h=500, e=350),
    ]
    comparison = compare_policy_outcomes(
        run("LM", lm_records),
        run("F", flow_records),
    )
    assert comparison["status"] == "PASS"
    assert comparison["d2_selected_set_signal"] is False
    assert comparison["outcome_difference_required_for_correctness"] is False


def test_summary_preserves_online_boundary_and_no_ttft_claim(tmp_path) -> None:
    lm_records = [
        outcome("A", h=500, e=300),
        outcome("B", h=300, e=200),
        outcome("C", h=500, e=370),
        outcome("D", h=500, e=350),
    ]
    flow_records = [
        outcome("A", h=500, e=400),
        outcome("B", h=300, e=200),
        outcome("C", h=500, e=370),
        outcome("D", h=500, e=240),
    ]
    summary = build_summary(
        artifact=tmp_path,
        runs=(run("LM", lm_records), run("F", flow_records)),
        boundary_audit=clean_boundary(),
        environment=None,
    )
    assert summary["status"] == "PASS"
    assert summary["future_leakage"] is False
    assert summary["ttft_telemetry_saved_only"] is True
    assert summary["ttft_compared"] is False
    assert summary["statistical_test_performed"] is False
    assert summary["performance_claim_made"] is False


def test_gate_source_prepares_two_fresh_lifecycles_without_future_turns() -> None:
    module = __import__(
        "evaluation.openhands_policy_runtime_heg_outcome_gate",
        fromlist=["unused"],
    )
    source = inspect.getsource(module)
    assert "for ordinal, condition in enumerate(RUN_ORDER" in source
    assert "FormalEndToEndGateEngine(" in source
    assert "engine.shutdown()" in source
    assert "build_pending_set(" in source
    assert "execute_request(engine, client, request, ordinal)" in source
    assert "PENDING_TURN + 1" not in source
    assert "ttft_compared\": False" in source
