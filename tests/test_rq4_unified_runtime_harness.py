from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.rq3_formal_policy_evaluation import (
    _load_allocation_snapshot,
    create_budget_variant,
)
from evaluation.rq4_unified_runtime_harness import (
    DEFAULT_FORMAL_ROOT,
    FORMAL_RUN_COUNT,
    FORMAL_SNAPSHOT_IDS,
    POLICIES,
    RQ4CorrectnessError,
    SMOKE_SNAPSHOT_IDS,
    _snapshot_path_for_group,
    _run_correctness,
    _validate_pending_record,
    build_current_runtime_handles,
    build_run_plan,
    rq4_logical_k,
    select_frozen_policy,
    summarize_formal,
    summarize_smoke,
    trace_has_rematerialization,
    validate_rebuilt_snapshot,
)


@pytest.mark.parametrize(
    ("candidate_count", "expected"),
    ((8, 2), (12, 3), (16, 4), (20, 5)),
)
def test_rq4_logical_k_matches_frozen_budget(
    candidate_count: int,
    expected: int,
) -> None:
    """确认 RQ4 25% 预算不再依赖固定 K=2。"""
    assert rq4_logical_k(candidate_count) == expected


def test_smoke_run_plan_freezes_six_dynamic_k_runs() -> None:
    """确认两个 snapshot 的三策略计划和 selected IDs 完整冻结。"""
    plan = build_run_plan(DEFAULT_FORMAL_ROOT, SMOKE_SNAPSHOT_IDS, 1)
    assert len(plan) == 6
    by_snapshot = {}
    for row in plan:
        by_snapshot.setdefault(row["snapshot_id"], []).append(row)
        assert len(row["selected_candidate_ids"]) <= row["logical_k"]
    assert {row["policy"] for row in plan} == set(POLICIES)
    assert {row["logical_k"] for row in by_snapshot[SMOKE_SNAPSHOT_IDS[0]]} == {2}
    assert {row["logical_k"] for row in by_snapshot[SMOKE_SNAPSHOT_IDS[1]]} == {5}


def test_formal_run_plan_exactly_reuses_rq4_a_population() -> None:
    """确认正式计划只复用 RQ4-A 的 24 个冻结 snapshot。"""
    plan = build_run_plan(DEFAULT_FORMAL_ROOT, FORMAL_SNAPSHOT_IDS, 3)
    assert len(FORMAL_SNAPSHOT_IDS) == 24
    assert len(set(FORMAL_SNAPSHOT_IDS)) == 24
    assert len(plan) == FORMAL_RUN_COUNT
    assert {row["snapshot_id"] for row in plan} == set(FORMAL_SNAPSHOT_IDS)
    assert {row["repetition"] for row in plan} == {1, 2, 3}
    for snapshot_id in FORMAL_SNAPSHOT_IDS:
        rows = [row for row in plan if row["snapshot_id"] == snapshot_id]
        assert len(rows) == 9
        assert {row["policy"] for row in rows} == set(POLICIES)


def test_frozen_selectors_are_deterministic_at_dynamic_k() -> None:
    """确认三种冻结 selector 在 K=5 时均确定且不超过预算。"""
    path = _snapshot_path_for_group(DEFAULT_FORMAL_ROOT, 119)
    snapshot = _load_allocation_snapshot(path)
    variant = create_budget_variant(snapshot, 5)
    for policy in POLICIES:
        first = select_frozen_policy(variant, policy)
        second = select_frozen_policy(variant, policy)
        assert first == second
        assert len(first) <= 5


def test_rebuilt_snapshot_requires_full_exact_match() -> None:
    """确认 universe 或完整 digest 不一致时 fail closed。"""
    path = _snapshot_path_for_group(DEFAULT_FORMAL_ROOT, 16)
    snapshot = _load_allocation_snapshot(path)
    result = validate_rebuilt_snapshot(snapshot, snapshot)
    assert result["logical_universe_exact"] is True
    changed = replace(snapshot, residency_snapshot_digest="0" * 64)
    with pytest.raises(RQ4CorrectnessError, match="digest"):
        validate_rebuilt_snapshot(snapshot, changed)


def test_current_handle_mapping_is_complete_and_rejects_ambiguity() -> None:
    """确认句柄只由当前 replay token prefix 构造且 identity 唯一。"""
    from evaluation.openhands_common_barrier_snapshot_gate import token_digest

    first_ids = [11, 12, 13]
    second_ids = [21, 22, 23]
    observations = (
        SimpleNamespace(
            checkpoint_id="A",
            workflow_label="A",
            turn=1,
            token_pos=3,
            node_id=10,
            prefix_digest=token_digest(first_ids),
        ),
        SimpleNamespace(
            checkpoint_id="B",
            workflow_label="B",
            turn=1,
            token_pos=3,
            node_id=20,
            prefix_digest=token_digest(second_ids),
        ),
    )
    trace = SimpleNamespace(checkpoints=observations)
    requests = {
        ("A", 1): {"input_ids": first_ids},
        ("B", 1): {"input_ids": second_ids},
    }
    handles = build_current_runtime_handles(trace, requests)
    assert set(handles) == {"A", "B"}
    assert handles["A"].expected_node_id == 10

    ambiguous = SimpleNamespace(
        checkpoints=(
            observations[0],
            SimpleNamespace(
                checkpoint_id="C",
                workflow_label="A",
                turn=1,
                token_pos=3,
                node_id=10,
                prefix_digest=token_digest(first_ids),
            ),
        )
    )
    with pytest.raises(RQ4CorrectnessError, match="同一 runtime identity"):
        build_current_runtime_handles(ambiguous, requests)


def test_rematerialization_detection() -> None:
    """确认驱逐后重新设备驻留会被识别。"""
    clean = [
        {"checkpoints": {"A": {"recurrent_present": True}}},
        {"checkpoints": {"A": {"recurrent_present": False}}},
        {"checkpoints": {"A": {"recurrent_present": False}}},
    ]
    contaminated = [
        *clean,
        {"checkpoints": {"A": {"recurrent_present": True}}},
    ]
    assert trace_has_rematerialization(clean, ("A",)) is False
    assert trace_has_rematerialization(contaminated, ("A",)) is True


def test_pending_gate_requires_heg_ttft_and_residency() -> None:
    """确认 resumed request 缺少任一关键事实都会 fail closed。"""
    record = {
        "status": "PASS",
        "request_completed": True,
        "runtime_metrics_valid": True,
        "h": 10,
        "e": 7,
        "g": 3,
        "ttft_ms": 12.5,
        "oom": False,
        "truncation_or_clipping": False,
        "expected_actual_residency_exact": True,
    }
    _validate_pending_record(record)
    with pytest.raises(RQ4CorrectnessError, match="G=H-E"):
        _validate_pending_record({**record, "g": 2})
    with pytest.raises(RQ4CorrectnessError, match="TTFT"):
        _validate_pending_record({**record, "ttft_ms": None})


def _valid_formal_result(index: int) -> dict[str, object]:
    """构造只用于汇总与 correctness 单元测试的有效 run。"""
    pending = [
        {
            "h": 10,
            "e": 8,
            "g": 2,
            "runtime_metrics_valid": True,
            "ttft_ms": 10.0,
            "native_mamba_capacity_eviction": False,
            "fa_kv_cascade": False,
            "truncation_or_clipping": False,
            "oom": False,
            "expected_actual_residency_exact": True,
            "request_completed": True,
        }
        for _ in range(4)
    ]
    return {
        "run_id": str(index),
        "status": "PASS",
        "policy": POLICIES[index % len(POLICIES)],
        "logical_k": 2,
        "fresh_engine_empty": True,
        "worker_exit_code": 0,
        "worker_timed_out": False,
        "worker_process_wall_s": 12.0,
        "replay_validation": {
            "full_snapshot_digest_exact": True,
            "candidate_universe_exact": True,
            "pending_universe_exact": True,
        },
        "selection": {
            "logical_k": 2,
            "selected_candidate_ids_exact": True,
        },
        "handle_mapping": {
            "status": "PASS",
            "complete": True,
            "unique_runtime_identities": True,
        },
        "reconcile": {
            "unexpected_rematerialization": False,
            "invariants": {
                "status": "PASS",
                "selected_residency_exact": True,
                "fa_residency_preserved": True,
                "native_mamba_capacity_eviction": False,
                "fa_kv_cascade": False,
            },
        },
        "pending_requests": pending,
        "future_leakage": False,
        "gpu_wait_before": {"stable": True},
        "gpu_wait_after": {"stable": True},
        "timings": {
            "engine_init_s": 1.0,
            "replay_s": 2.0,
            "reconcile_s": 3.0,
            "pending_resume_s": 4.0,
            "total_run_s": 10.0,
        },
    }


def test_formal_summary_and_run_correctness_cover_all_gates() -> None:
    """确认 216 个有效 run 才能得到正式 READY。"""
    plan = [{"run_id": str(index)} for index in range(FORMAL_RUN_COUNT)]
    runs = [_valid_formal_result(index) for index in range(FORMAL_RUN_COUNT)]
    assert _run_correctness(runs[0])["status"] == "PASS"
    summary = summarize_formal(
        Path("artifact"),
        plan,
        runs,
        {"stable": True},
    )
    assert summary["status"] == "RQ4_RUNTIME_FORMAL_READY"
    assert summary["pending_completed"] == 864
    assert summary["g_equals_h_minus_e_count"] == 864
    assert summary["cross_run_contamination"] is False


def test_smoke_summary_keeps_snapshot_requests_as_telemetry() -> None:
    """确认六个有效 run 汇总为 24 个请求且使用 worker 墙钟估时。"""
    run_plan = [{"run_id": str(index)} for index in range(6)]
    runs = []
    for index in range(6):
        pending = [
            {
                "h": 10,
                "e": 8,
                "g": 2,
                "runtime_metrics_valid": True,
                "ttft_ms": 10.0,
                "native_mamba_capacity_eviction": False,
                "fa_kv_cascade": False,
                "truncation_or_clipping": False,
                "oom": False,
            }
            for _ in range(4)
        ]
        runs.append(
            {
                "run_id": str(index),
                "status": "PASS",
                "handle_mapping": {"status": "PASS"},
                "reconcile": {
                    "unexpected_rematerialization": False,
                    "invariants": {
                        "status": "PASS",
                        "fa_residency_preserved": True,
                        "native_mamba_capacity_eviction": False,
                        "fa_kv_cascade": False,
                    },
                },
                "pending_requests": pending,
                "future_leakage": False,
                "timings": {
                    "engine_init_s": 1.0,
                    "replay_s": 2.0,
                    "reconcile_s": 3.0,
                    "pending_resume_s": 4.0,
                    "total_run_s": 10.0,
                },
                "worker_process_wall_s": 12.0,
            }
        )
    summary = summarize_smoke(
        Path("artifact"),
        run_plan,
        runs,
        {"stable": True},
    )
    assert summary["status"] == "RQ4_RUNTIME_SMOKE_READY"
    assert summary["pending_completed"] == 24
    assert summary["g_equals_h_minus_e_count"] == 24
    assert summary["reestimated_216_run_gpu_hours"] == pytest.approx(0.72)


def test_smoke_summary_accepts_early_invalid_run() -> None:
    """确认 Engine 初始化前失败也能生成 BLOCKED 汇总。"""
    summary = summarize_smoke(
        Path("artifact"),
        [{"run_id": str(index)} for index in range(6)],
        [
            {
                "run_id": "0",
                "status": "INVALID",
                "reconcile": None,
                "handle_mapping": None,
                "pending_requests": [],
                "future_leakage": False,
                "worker_process_wall_s": 1.0,
            }
        ],
        {"stable": True},
    )
    assert summary["status"] == "RQ4_RUNTIME_SMOKE_BLOCKED"
    assert summary["passed_runs"] == 0
