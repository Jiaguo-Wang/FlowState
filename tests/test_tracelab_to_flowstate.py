"""验证 TraceLab-to-FlowState workload 的无未来泄漏与确定性。"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timedelta
from inspect import signature
import json
from pathlib import Path

import pytest

from evaluation.public_agent_trace import tracelab_to_flowstate as converter


SNAPSHOT_TIME = datetime(2026, 8, 26, 12, 0, 0)


def _completed(
    *,
    workflow_id: str = "workflow-1",
    round_pk: int = 1,
    round_index: int = 10,
    run_position: int = 0,
    input_tokens_total: int = 4096,
    current_prefix_tokens: int = 1024,
    known_at_time: datetime | None = None,
) -> converter.CompletedRoundFact:
    """构造当前 snapshot 已经完成的 round fact。"""
    return converter.CompletedRoundFact(
        workflow_id=workflow_id,
        round_pk=round_pk,
        round_index=round_index,
        run_position=run_position,
        input_tokens_total=input_tokens_total,
        current_prefix_tokens=current_prefix_tokens,
        known_at_time=known_at_time or SNAPSHOT_TIME,
    )


def _pending(
    *,
    workflow_id: str = "workflow-1",
    round_pk: int = 1,
    round_index: int = 10,
    run_position: int = 0,
    input_tokens_total: int = 4096,
    current_prefix_tokens: int = 1024,
    known_at_time: datetime | None = None,
    tool_ids: tuple[str, ...] = ("tool-1",),
) -> converter.PendingRoundFact:
    """构造仅依赖已发出 tool calls 的 pending fact。"""
    return converter.PendingRoundFact(
        workflow_id=workflow_id,
        round_pk=round_pk,
        round_index=round_index,
        run_position=run_position,
        input_tokens_total=input_tokens_total,
        current_prefix_tokens=current_prefix_tokens,
        known_at_time=known_at_time or SNAPSHOT_TIME,
        observed_tool_call_ids=tool_ids,
    )


def _snapshot(
    *,
    completed_rounds: tuple[converter.CompletedRoundFact, ...] | None = None,
    pending_rounds: tuple[converter.PendingRoundFact, ...] | None = None,
) -> converter.TraceSnapshot:
    """构造一个最小的可消费离线 snapshot。"""
    return converter.build_trace_snapshot(
        snapshot_id="snapshot-1",
        scale="Small",
        time_domain="codex",
        observed_at=SNAPSHOT_TIME,
        active_workflow_ids=("workflow-1",),
        completed_rounds=completed_rounds or (_completed(),),
        pending_rounds=pending_rounds or (_pending(),),
    )


def test_linear_lineage_is_deterministic_prefix() -> None:
    """确认 lineage 只编码真实 round 顺序并保持 tuple prefix。"""
    shallow = converter.linear_lineage_path(1)
    deep = converter.linear_lineage_path(3)
    assert shallow == ("step:000000", "step:000001")
    assert deep[: len(shallow)] == shallow


def test_snapshot_uses_existing_flowstate_models() -> None:
    """确认输出直接使用核心 candidate 与 continuation 类型。"""
    snapshot = _snapshot()
    candidate = snapshot.candidates[0]
    continuation = snapshot.continuations[0]
    assert candidate.memory_bytes == converter.CHECKPOINT_MEMORY_BYTES
    assert candidate.token_pos == 4096
    assert continuation.anchor_pos == 4096
    assert continuation.resident_fa_frontier == 4096
    assert candidate.workflow_id == continuation.workflow_id


def test_current_prefix_is_metadata_and_does_not_change_anchor() -> None:
    """确认当前 prefix 只作描述，anchor 始终来自当前 input。"""
    first = _snapshot(
        completed_rounds=(_completed(current_prefix_tokens=0),),
        pending_rounds=(_pending(current_prefix_tokens=0),),
    )
    second = _snapshot(
        completed_rounds=(_completed(current_prefix_tokens=4000),),
        pending_rounds=(_pending(current_prefix_tokens=4000),),
    )
    assert first.continuations[0].anchor_pos == 4096
    assert second.continuations[0].anchor_pos == 4096
    assert first.candidates[0].token_pos == second.candidates[0].token_pos


def test_multiple_tool_calls_create_only_one_pending() -> None:
    """确认一个 round 的多个工具调用不会引入 LLM-level fanout。"""
    snapshot = _snapshot(
        pending_rounds=(_pending(tool_ids=("a", "b", "c")),)
    )
    assert len(snapshot.continuations) == 1
    assert snapshot.continuation_metadata[0].tool_call_count == 3
    assert not snapshot.llm_level_branching_introduced


def test_pending_without_observed_tool_is_rejected() -> None:
    """确认 pending 不能由没有已发出 tool call 的 round 产生。"""
    with pytest.raises(ValueError, match="至少一个已发出 tool call"):
        _snapshot(pending_rounds=(_pending(tool_ids=()),))


def test_future_completed_round_is_rejected() -> None:
    """确认 snapshot 之后完成的 round 不能生成 checkpoint。"""
    future = SNAPSHOT_TIME + timedelta(seconds=1)
    with pytest.raises(ValueError, match="snapshot 之后完成"):
        _snapshot(completed_rounds=(_completed(known_at_time=future),))


def test_future_tool_call_is_rejected() -> None:
    """确认 snapshot 之后才发出的工具调用不能生成 pending。"""
    future = SNAPSHOT_TIME + timedelta(seconds=1)
    with pytest.raises(ValueError, match="snapshot 之后发出"):
        _snapshot(pending_rounds=(_pending(known_at_time=future),))


def test_cross_workflow_fact_is_rejected() -> None:
    """确认 checkpoint 与 continuation 均不能跨 active workflow。"""
    with pytest.raises(ValueError, match="非 active workflow"):
        _snapshot(completed_rounds=(_completed(workflow_id="workflow-2"),))
    with pytest.raises(ValueError, match="非 active workflow"):
        _snapshot(pending_rounds=(_pending(workflow_id="workflow-2"),))


def test_retention_budget_uses_floor_and_minimum_one() -> None:
    """确认 retention ratio 的 K 严格向下取整且至少为一。"""
    assert tuple(
        budget.k for budget in converter.retention_budgets(3)
    ) == (1, 1, 2)
    assert tuple(
        budget.k for budget in converter.retention_budgets(8)
    ) == (2, 4, 6)


def test_online_mapping_interface_has_no_future_round_fields() -> None:
    """确认 snapshot builder 的输入类型与接口不存在 future round 字段。"""
    forbidden = {
        "next_prefix_tokens",
        "next_input_tokens_total",
        "next_output_tokens",
        "future_timing_events",
    }
    fact_fields = {
        field.name for field in fields(converter.CompletedRoundFact)
    } | {field.name for field in fields(converter.PendingRoundFact)}
    builder_parameters = set(
        signature(converter.build_trace_snapshot).parameters
    )
    assert forbidden.isdisjoint(fact_fields)
    assert forbidden.isdisjoint(builder_parameters)


def test_stratified_sampling_is_deterministic() -> None:
    """确认输入顺序变化不影响每个预注册 stratum 的样本。"""
    events = (
        converter.SampledSnapshotEvent(
            snapshot_id="a",
            scale="Small",
            provider="codex",
            context_bucket="<=32K",
            trigger_session_id="s1",
            trigger_run_ordinal=1,
            trigger_round_pk=1,
            observed_at=SNAPSHOT_TIME,
            trace_observed_active_runs=2,
        ),
        converter.SampledSnapshotEvent(
            snapshot_id="b",
            scale="Small",
            provider="codex",
            context_bucket="<=32K",
            trigger_session_id="s2",
            trigger_run_ordinal=1,
            trigger_round_pk=2,
            observed_at=SNAPSHOT_TIME,
            trace_observed_active_runs=3,
        ),
        converter.SampledSnapshotEvent(
            snapshot_id="c",
            scale="Medium",
            provider="claude",
            context_bucket=">256K",
            trigger_session_id="s3",
            trigger_run_ordinal=3,
            trigger_round_pk=3,
            observed_at=SNAPSHOT_TIME,
            trace_observed_active_runs=6,
        ),
    )
    first = converter.choose_stratified_events(events)
    second = converter.choose_stratified_events(tuple(reversed(events)))
    assert first == second
    assert len(first) == 2


def test_integrity_validator_rejects_anchor_mismatch() -> None:
    """确认独立 gate 能检测 anchor 与当前轮 input 不一致。"""
    snapshot = _snapshot()
    broken = replace(
        snapshot,
        continuations=(
            replace(snapshot.continuations[0], anchor_pos=4095),
        ),
    )
    assert any(
        "anchor 不等于当前输入" in violation
        for violation in converter.validate_snapshot(broken)
    )


def test_snapshot_explicitly_disclaims_runtime_residency() -> None:
    """确认逻辑 resident 字段不会被表述为 runtime truth。"""
    snapshot = _snapshot()
    assert snapshot.runtime_residency_inferred is False
    assert snapshot.future_prefix_used is False
    assert converter.validate_snapshot(snapshot) == ()


def test_external_duckdb_is_not_copied_into_repository() -> None:
    """确认 workload builder 始终引用仓库之外的 DuckDB。"""
    repository_root = Path(__file__).resolve().parents[1]
    database_path = converter.DEFAULT_DATABASE_PATH
    assert repository_root not in database_path.parents
    assert database_path.suffix == ".duckdb"


def test_frozen_workload_artifact_passes_integrity_gate() -> None:
    """确认保存的 workload 与构造摘要一致且没有泄漏标记。"""
    workload = json.loads(
        converter.DEFAULT_WORKLOAD_PATH.read_text(encoding="utf-8")
    )
    assert workload["summary"]["snapshot_count"] == 27
    assert len(workload["snapshots"]) == 27
    assert workload["summary"]["leakage_violation_count"] == 0
    assert workload["gates"]["leakage_free_construction"] is True
    for snapshot in workload["snapshots"]:
        assert snapshot["future_prefix_used"] is False
        assert snapshot["runtime_residency_inferred"] is False
        assert snapshot["llm_level_branching_introduced"] is False
        assert snapshot["candidate_count"] == len(snapshot["candidates"])
        assert snapshot["pending_continuation_count"] == len(
            snapshot["continuations"]
        )
