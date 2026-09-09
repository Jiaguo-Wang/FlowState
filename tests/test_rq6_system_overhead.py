"""验证 RQ6 分阶段计时、聚合与正确性门禁。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import evaluation.rq6_runtime_overhead as runtime_overhead

from evaluation.rq3_frozen_snapshot_evaluator import (
    FrozenCheckpointRuntimeEvidence,
    FrozenOnlineInformationBoundary,
    build_allocation_snapshot,
)
from evaluation.rq6_system_overhead import (
    StageTimer,
    construct_executable_state,
    instrumentation_equivalence,
    overhead_summary,
    percentile,
    reference_flowstate_selection,
    runtime_correctness,
    timed_flowstate_selection,
    validate_record_schema,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


def _snapshot():
    """构造包含两个工作流和三个等大小候选的最小冻结快照。"""
    pending = (
        PendingContinuation("p1", "w1", ("root", "w1"), 8192, 8192),
        PendingContinuation("p2", "w2", ("root", "w2"), 12288, 12288),
    )
    candidates = (
        CheckpointCandidate("c1", "w1", ("root", "w1"), 4096, 10),
        CheckpointCandidate("c2", "w1", ("root", "w1"), 8192, 10),
        CheckpointCandidate("c3", "w2", ("root", "w2"), 8192, 10),
    )
    order = {candidate.checkpoint_id: index for index, candidate in enumerate(candidates)}
    return build_allocation_snapshot(
        allocation_epoch=3,
        snapshot_id="rq6-test",
        pending_continuations=pending,
        eligible_candidates=candidates,
        creation_order_by_checkpoint=order,
        last_access_order_by_checkpoint=order,
        marconi_flop_saved_by_checkpoint={key: 1.0 for key in order},
        access_frequency_by_checkpoint={key: 1 for key in order},
        frequency_observed_through_epoch=3,
        marconi_alpha=1.0,
        logical_budget_k=1,
        budget_bytes=10,
        runtime_evidence=tuple(
            FrozenCheckpointRuntimeEvidence(
                checkpoint_id=candidate.checkpoint_id,
                node_id=index,
                runtime_identity_digest=f"{index + 1:064x}",
                checkpoint_handle_digest=f"{index + 10:064x}",
            )
            for index, candidate in enumerate(candidates)
        ),
        residency_snapshot_digest="f" * 64,
        online_boundary=FrozenOnlineInformationBoundary(
            materialized_through_epoch=3,
            visible_continuation_ids=("p1", "p2"),
        ),
    )


def _record(kind: str = "cpu_allocation") -> dict:
    """返回一个字段完整的原始记录。"""
    return {
        "schema_version": "flowstate.rq6_per_epoch_overhead.v1",
        "record_kind": kind,
        "population": "测试",
        "snapshot_id": "s",
        "k": 1,
        "pending_count": 2,
        "candidate_count": 3,
        "timings_ns": {
            "introspection": 1_000_000 if kind == "runtime_control" else None,
            "construction": 2_000_000,
            "allocation": 3_000_000,
            "reconciliation": 4_000_000 if kind == "runtime_control" else None,
            "total_control": 10_000_000,
        },
        "instrumentation_equivalent": True,
        "snapshot_immutable": True,
        "heg_equivalent": True,
        "future_leakage": False,
        "status": "PASS",
    }


def test_stage_timer_boundaries_and_total_timer() -> None:
    """各 stage 与 total 必须产生独立非负的纳秒计时。"""
    snapshot = _snapshot()
    result = timed_flowstate_selection(snapshot)
    assert result.construction_ns > 0
    assert result.allocation_ns > 0
    assert result.total_ns >= result.construction_ns + result.allocation_ns
    with StageTimer(enabled=False) as disabled:
        pass
    assert disabled.elapsed_ns == 0


def test_instrumentation_on_off_selected_set_equivalence() -> None:
    """计时开关与 RQ3 冻结 dispatcher 必须选择同一集合。"""
    snapshot = _snapshot()
    timed = timed_flowstate_selection(snapshot)
    disabled = timed_flowstate_selection(snapshot, instrumentation=False)
    assert timed.selected_checkpoint_ids == disabled.selected_checkpoint_ids
    assert timed.selected_checkpoint_ids == reference_flowstate_selection(snapshot)
    assert disabled.total_ns == 0


def test_frozen_snapshot_immutability() -> None:
    """计时前后冻结快照摘要必须完全不变。"""
    snapshot = _snapshot()
    before = snapshot.content_digest()
    result = timed_flowstate_selection(snapshot)
    assert result.snapshot_digest_before == before
    assert result.snapshot_digest_after == before


def test_aggregation_correctness() -> None:
    """聚合必须按 stage 从纳秒转换为毫秒。"""
    rows = [_record(), _record()]
    summary = overhead_summary(rows)
    stage = summary["by_kind"]["cpu_allocation"]["stages_ms"]["allocation"]
    assert stage == {"count": 2, "mean": 3.0, "median": 3.0, "p95": 3.0, "max": 3.0}


def test_percentile_linear_interpolation() -> None:
    """P95 使用冻结的线性插值规则。"""
    assert percentile([0.0, 10.0], 95.0) == pytest.approx(9.5)
    assert percentile([3.0], 95.0) == 3.0
    with pytest.raises(ValueError):
        percentile([], 95.0)


def test_runtime_overhead_record_schema() -> None:
    """完整 schema 通过，缺 stage 时必须失败。"""
    record = _record("runtime_control")
    validate_record_schema(record)
    del record["timings_ns"]["reconciliation"]
    with pytest.raises(ValueError, match="reconciliation"):
        validate_record_schema(record)


def test_no_future_information_access() -> None:
    """任一 future flag 都必须令执行态构造 fail closed。"""
    snapshot = _snapshot()
    boundary = replace(snapshot.online_boundary, future_request_included=True)
    with pytest.raises(ValueError, match="未来信息"):
        construct_executable_state(replace(snapshot, online_boundary=boundary))


def test_instrumentation_equivalence_gate() -> None:
    """selected set、H/E/G、不可变性与在线边界必须共同通过。"""
    passed = instrumentation_equivalence([_record()])
    assert passed["status"] == "PASS"
    failed_row = _record()
    failed_row["heg_equivalent"] = False
    assert instrumentation_equivalence([failed_row])["status"] == "FAIL"


def test_runtime_correctness_rejects_cross_run_contamination() -> None:
    """即使其它字段通过，跨 run 污染也必须阻断结果。"""
    row = _record("runtime_control")
    row["run_id"] = "r"
    row["cross_run_contamination"] = True
    row["correctness"] = {
        "handle_mapping_pass": True,
        "recurrent_residency_validation_pass": True,
        "fa_preserved": True,
        "native_recurrent_eviction_zero": True,
        "unexpected_rematerialization_zero": True,
        "fa_cascade_zero": True,
        "oom_zero": True,
        "truncation_zero": True,
        "future_leakage_zero": True,
        "fresh_engine_empty": True,
        "gpu_cleanup_pass": True,
    }
    assert runtime_correctness([row])["status"] == "FAIL"


def test_state_construction_has_no_state_changing_introspection() -> None:
    """CPU 构造只复制冻结对象，不改变 residency 或 snapshot 内容。"""
    snapshot = _snapshot()
    before = snapshot.canonical_serialization()
    prepared = construct_executable_state(snapshot)
    assert prepared.compatibility
    assert all(candidate.recurrent_resident for candidate in prepared.candidates)
    assert snapshot.canonical_serialization() == before


def test_future_information_is_rejected() -> None:
    """任一未来边界标记必须使执行态构造失败。"""
    snapshot = _snapshot()
    future_boundary = replace(
        snapshot.online_boundary,
        future_request_included=True,
    )
    future_snapshot = replace(snapshot, online_boundary=future_boundary)
    with pytest.raises(ValueError, match="未来信息"):
        construct_executable_state(future_snapshot)


def test_runtime_introspection_has_fixed_read_only_lookup_count(monkeypatch) -> None:
    """运行时 introspection 只能做一次 census 与一次只读状态检查。"""
    calls = {"census": 0, "inspect": 0}

    class Runtime:
        def census(self, *_args, **_kwargs):
            calls["census"] += 1
            return {"mamba_node_count": 3}

    def inspect(*_args, **_kwargs):
        calls["inspect"] += 1
        return {"c1": {"recurrent_resident": True, "fa_resident": True}}, []

    monkeypatch.setattr(runtime_overhead, "inspect_candidate_states", inspect)
    census, states, inspections, elapsed = (
        runtime_overhead.read_only_runtime_introspection(
            Runtime(),
            object(),
            (object(),),
            {"c1": object()},
            label="只读检查",
            ordinal=1,
            previous_census={},
        )
    )
    assert calls == {"census": 1, "inspect": 1}
    assert census == {"mamba_node_count": 3}
    assert states["c1"]["recurrent_resident"] is True
    assert inspections == []
    assert elapsed >= 0
