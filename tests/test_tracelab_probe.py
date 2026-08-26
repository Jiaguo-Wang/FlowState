"""验证 TraceLab probe 的只读边界、schema 解析与确定性采样。"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.public_agent_trace import tracelab_probe


def _schema(table_name: str, columns: tuple[str, ...]) -> dict[str, object]:
    return {
        "table_name": table_name,
        "columns": tuple(
            {"column_name": column_name} for column_name in columns
        ),
    }


def test_main_round_table_is_identified_from_real_required_columns() -> None:
    """确认主表识别依赖完整真实字段集合，而不是名称猜测。"""
    schemas = (
        _schema("events", ("round_pk", "timestamp")),
        _schema(
            "actual_rounds",
            tuple(sorted(tracelab_probe.MAIN_ROUND_REQUIRED_COLUMNS)),
        ),
    )
    assert tracelab_probe.identify_main_round_table(schemas) == (
        "actual_rounds"
    )


def test_main_round_table_rejects_ambiguous_or_missing_schema() -> None:
    """确认 schema 不足或存在两个候选时明确失败。"""
    complete = tuple(sorted(tracelab_probe.MAIN_ROUND_REQUIRED_COLUMNS))
    with pytest.raises(ValueError, match="无法唯一识别"):
        tracelab_probe.identify_main_round_table((_schema("x", ("id",)),))
    with pytest.raises(ValueError, match="无法唯一识别"):
        tracelab_probe.identify_main_round_table(
            (_schema("x", complete), _schema("y", complete))
        )


def test_sample_session_selection_is_deterministic_and_complete() -> None:
    """确认输入顺序不影响跨 provider 的固定短 session 选择。"""
    rows = (
        {
            "provider": "claude",
            "session_id": "c3",
            "round_count": 5,
            "min_round_index": 0,
            "max_round_index": 4,
        },
        {
            "provider": "codex",
            "session_id": "x2",
            "round_count": 2,
            "min_round_index": 0,
            "max_round_index": 1,
        },
        {
            "provider": "claude",
            "session_id": "c1",
            "round_count": 2,
            "min_round_index": 0,
            "max_round_index": 1,
        },
        {
            "provider": "claude",
            "session_id": "c2",
            "round_count": 3,
            "min_round_index": 0,
            "max_round_index": 2,
        },
        {
            "provider": "codex",
            "session_id": "x1",
            "round_count": 4,
            "min_round_index": 0,
            "max_round_index": 3,
        },
        {
            "provider": "claude",
            "session_id": "incomplete",
            "round_count": 3,
            "min_round_index": 0,
            "max_round_index": 4,
        },
    )
    expected = ("c1", "c2", "c3", "x1", "x2")
    assert tracelab_probe.choose_sample_sessions(rows) == expected
    assert tracelab_probe.choose_sample_sessions(tuple(reversed(rows))) == (
        expected
    )


def test_dag_field_scan_does_not_promote_unrelated_ids() -> None:
    """确认只报告名称相关字段，不把 round_id 或 turn_id 当作 parent。"""
    schemas = (
        _schema(
            "rounds",
            ("round_id", "turn_id", "root_session_id"),
        ),
        _schema(
            "tool_calls",
            ("continuation_of_tool_call_id", "tool_call_id"),
        ),
    )
    assert tracelab_probe.find_dag_like_fields(schemas) == (
        "rounds.root_session_id",
        "tool_calls.continuation_of_tool_call_id",
    )


def test_database_connection_is_forced_read_only(monkeypatch) -> None:
    """确认外部 DuckDB 连接始终显式使用只读模式。"""
    calls = []
    sentinel = object()

    class FakeDuckDB:
        @staticmethod
        def connect(path: str, *, read_only: bool):
            calls.append((path, read_only))
            return sentinel

    monkeypatch.setattr(
        tracelab_probe.importlib,
        "import_module",
        lambda name: FakeDuckDB,
    )
    path = Path("/tmp/public-trace.duckdb")
    assert tracelab_probe.open_database_read_only(path) is sentinel
    assert calls == [(str(path), True)]


def test_default_database_is_external_to_repository() -> None:
    """确认 probe 引用外部文件，不把数据库复制进 FlowState。"""
    repository_root = Path(__file__).resolve().parents[1]
    assert repository_root not in tracelab_probe.DEFAULT_DATABASE_PATH.parents
    assert tracelab_probe.DEFAULT_DATABASE_PATH.suffix == ".duckdb"


def test_identifier_quoting_is_deterministic() -> None:
    """确认动态 schema 标识符不会破坏只读查询。"""
    assert tracelab_probe.quote_identifier('a"b') == '"a""b"'
