"""验证 TraceLab 语义审计的确定性分段与无未来信息边界。"""

from __future__ import annotations

from dataclasses import fields
from inspect import signature
from pathlib import Path

import pytest

from evaluation.public_agent_trace import tracelab_semantics


def _online_snapshot(
    *, tool_ids: tuple[str, ...] = ("tool-1",)
) -> tracelab_semantics.OnlineRoundSnapshot:
    """构造仅含当前已知信息的测试快照。"""
    return tracelab_semantics.build_online_snapshot(
        provider="codex",
        session_id="session-1",
        run_ordinal=2,
        round_index=7,
        input_tokens_total=4096,
        prefix_tokens=4000,
        newly_append_tokens=96,
        output_tokens=32,
        emitted_tool_call_ids=tool_ids,
        occurred_timing_event_types=("tool_call",),
    )


def test_user_message_starts_run_and_leading_round_is_excluded() -> None:
    """确认首个 user message 前的 round 不被强行归入 Agent Run。"""
    rounds = (
        tracelab_semantics.AgentRunRound("codex", "s", 0, 0),
        tracelab_semantics.AgentRunRound("codex", "s", 1, 1),
        tracelab_semantics.AgentRunRound("codex", "s", 2, 0),
        tracelab_semantics.AgentRunRound("codex", "s", 3, 2),
        tracelab_semantics.AgentRunRound("codex", "s", 4, 0),
    )
    segments = tracelab_semantics.segment_agent_runs(rounds)
    assert tuple(segment.round_indices for segment in segments) == (
        (1, 2),
        (3, 4),
    )
    assert tuple(segment.run_ordinal for segment in segments) == (1, 2)


def test_run_segmentation_is_input_order_independent() -> None:
    """确认输入列表顺序不会改变按 round_index 得到的分段。"""
    rounds = (
        tracelab_semantics.AgentRunRound("claude", "b", 1, 0),
        tracelab_semantics.AgentRunRound("codex", "a", 1, 0),
        tracelab_semantics.AgentRunRound("claude", "b", 0, 1),
        tracelab_semantics.AgentRunRound("codex", "a", 0, 1),
    )
    assert tracelab_semantics.segment_agent_runs(rounds) == (
        tracelab_semantics.segment_agent_runs(tuple(reversed(rounds)))
    )


def test_run_segmentation_rejects_duplicate_round_index() -> None:
    """确认 session 内重复顺序键会明确失败。"""
    rounds = (
        tracelab_semantics.AgentRunRound("codex", "s", 0, 1),
        tracelab_semantics.AgentRunRound("codex", "s", 0, 0),
    )
    with pytest.raises(ValueError, match="round_index 重复"):
        tracelab_semantics.segment_agent_runs(rounds)


def test_multiple_tools_create_one_llm_pending_signal() -> None:
    """确认 tool-level 数量不会被解释为 LLM-level fanout。"""
    snapshot = _online_snapshot(tool_ids=("a", "b", "c"))
    signal = tracelab_semantics.build_pending_continuation_signal(snapshot)
    assert signal is not None
    assert signal.tool_call_count == 3
    assert signal.source_round_index == snapshot.round_index


def test_no_tool_does_not_create_pending_signal() -> None:
    """确认没有已发出工具调用时不制造 pending continuation。"""
    assert (
        tracelab_semantics.build_pending_continuation_signal(
            _online_snapshot(tool_ids=())
        )
        is None
    )


def test_anchor_uses_only_current_input_boundary() -> None:
    """确认 anchor 严格等于当前 input，而不加入 output。"""
    snapshot = _online_snapshot()
    anchor = tracelab_semantics.build_known_historical_anchor(snapshot)
    assert anchor.token_pos == 4096
    assert anchor.token_pos != snapshot.input_tokens_total + snapshot.output_tokens
    assert anchor.source_round_index == 7


def test_online_snapshot_interface_has_no_future_fields() -> None:
    """确认 online 类型与构造接口均不接收任何下一轮字段。"""
    forbidden = {
        "next_prefix_tokens",
        "next_input_tokens_total",
        "next_output_tokens",
        "future_tool_result_size",
        "future_timing_events",
    }
    dataclass_fields = {
        field.name
        for field in fields(tracelab_semantics.OnlineRoundSnapshot)
    }
    parameter_names = set(
        signature(tracelab_semantics.build_online_snapshot).parameters
    )
    assert forbidden.isdisjoint(dataclass_fields)
    assert forbidden.isdisjoint(parameter_names)
    with pytest.raises(TypeError):
        tracelab_semantics.build_online_snapshot(
            provider="codex",
            session_id="s",
            run_ordinal=1,
            round_index=0,
            input_tokens_total=10,
            prefix_tokens=8,
            newly_append_tokens=2,
            output_tokens=1,
            next_prefix_tokens=11,
        )


def test_ex_post_validation_is_separate_and_signed() -> None:
    """确认下一轮 prefix 只进入独立事后验证并保留差值方向。"""
    anchor = tracelab_semantics.build_known_historical_anchor(
        _online_snapshot()
    )
    validation = tracelab_semantics.validate_anchor_ex_post(
        anchor,
        actual_next_prefix_tokens=4032,
    )
    assert validation.signed_delta_tokens == -64
    assert validation.absolute_delta_tokens == 64
    assert anchor.token_pos == 4096


def test_current_token_identity_is_validated() -> None:
    """确认当前 token accounting 不一致时拒绝构造 online snapshot。"""
    with pytest.raises(ValueError, match="input token 恒等式"):
        tracelab_semantics.build_online_snapshot(
            provider="codex",
            session_id="s",
            run_ordinal=1,
            round_index=0,
            input_tokens_total=10,
            prefix_tokens=8,
            newly_append_tokens=3,
            output_tokens=1,
        )


def test_external_database_is_not_copied_into_repository() -> None:
    """确认语义审计继续只读引用仓库之外的 TraceLab 数据。"""
    repository_root = Path(__file__).resolve().parents[1]
    database_path = tracelab_semantics.DEFAULT_DATABASE_PATH
    assert repository_root not in database_path.parents
    assert database_path.suffix == ".duckdb"
