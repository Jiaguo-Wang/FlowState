from __future__ import annotations

from pathlib import Path

from evaluation.openhands_final_multi_round_runtime_gate import (
    POLICY_RUN_COUNTS,
    _build_summary,
    _trace_has_rematerialization,
)


def trace_row(a_present: bool, b_present: bool) -> dict[str, object]:
    """构造同一 barrier 内的最小循环状态追踪行。"""
    return {
        "checkpoints": {
            "A1": {"recurrent_present": a_present},
            "B1": {"recurrent_present": b_present},
        }
    }


def barrier(status: str = "PASS") -> dict[str, object]:
    """构造最终汇总使用的 barrier 结果。"""
    return {
        "status": status,
        "unexpected_rematerialization": False,
        "invariants": {
            "fa_residency_preserved": True,
            "native_mamba_capacity_eviction": False,
            "fa_kv_cascade": False,
        },
    }


def run(policy: str, run_index: int, status: str = "PASS") -> dict[str, object]:
    """构造一个完整三轮 lifecycle 汇总。"""
    return {
        "policy": policy,
        "run": run_index,
        "status": status,
        "barriers": [barrier(), barrier()],
        "requests": [
            {"oom": False, "truncation_or_clipping": False}
            for _ in range(12)
        ],
    }


def all_runs() -> list[dict[str, object]]:
    """按冻结的 3/2/2 次数构造全部成功 lifecycle。"""
    return [
        run(policy, run_index)
        for policy, count in POLICY_RUN_COUNTS
        for run_index in range(1, count + 1)
    ]


def test_policy_repeat_counts_are_frozen() -> None:
    assert POLICY_RUN_COUNTS == (("LRU", 3), ("Marconi", 2), ("FlowState", 2))


def test_trace_detects_absent_to_present_in_same_barrier() -> None:
    rows = [
        trace_row(True, True),
        trace_row(False, True),
        trace_row(True, True),
    ]
    assert _trace_has_rematerialization(rows, ("A1",)) is True


def test_trace_accepts_stable_absence_after_eviction() -> None:
    rows = [
        trace_row(True, True),
        trace_row(False, True),
        trace_row(False, False),
    ]
    assert _trace_has_rematerialization(rows, ("A1", "B1")) is False


def test_complete_seven_runs_are_ready() -> None:
    summary = _build_summary(Path("/tmp/最终门禁"), all_runs(), {"gpu": "H100"})
    assert summary["result"] == "MULTI_ROUND_RUNTIME_READY"
    assert summary["completed_run_count"] == 7


def test_first_failed_run_makes_gate_not_ready() -> None:
    runs = all_runs()
    runs[2] = run("LRU", 3, status="FAIL")
    summary = _build_summary(Path("/tmp/最终门禁"), runs, {"gpu": "H100"})
    assert summary["result"] == "MULTI_ROUND_RUNTIME_NOT_READY"
