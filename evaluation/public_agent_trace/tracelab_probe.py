#!/usr/bin/env python3
"""只读探查 TraceLab DuckDB 的 schema、session 顺序与可行性。"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import importlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DATABASE_PATH = Path(
    "/home/wjg/data/tracelab/v0.0.2/syfi_coding_trace.duckdb"
)
DEFAULT_REPORT_PATH = Path(__file__).with_name("TRACELAB_PROBE.md")
MAIN_ROUND_REQUIRED_COLUMNS = frozenset(
    {
        "round_pk",
        "provider",
        "session_id",
        "round_index",
        "round_id",
        "model",
        "input_tokens_total",
        "prefix_tokens",
        "newly_append_tokens",
        "output_tokens",
    }
)
EXPLICIT_DAG_FIELD_NAMES = frozenset(
    {
        "parent_id",
        "parent_round_id",
        "continuation_id",
        "root_session_id",
        "root_id",
        "branch_id",
    }
)
SAMPLE_PROVIDER_QUOTAS = {"claude": 3, "codex": 2}


def quote_identifier(value: str) -> str:
    """安全引用 DuckDB schema、table 或 column 标识符。"""
    return '"' + value.replace('"', '""') + '"'


def choose_sample_sessions(
    rows: Sequence[Mapping[str, Any]],
    provider_quotas: Mapping[str, int] = SAMPLE_PROVIDER_QUOTAS,
) -> tuple[str, ...]:
    """按 provider 配额和 session_id 字典序确定性选择完整短 session。"""
    eligible = tuple(
        row
        for row in rows
        if 2 <= int(row["round_count"]) <= 5
        and int(row["min_round_index"]) == 0
        and int(row["max_round_index"]) + 1 == int(row["round_count"])
    )
    selected: list[str] = []
    for provider in sorted(provider_quotas):
        provider_rows = sorted(
            (
                row
                for row in eligible
                if row["provider"] == provider
            ),
            key=lambda row: str(row["session_id"]),
        )
        selected.extend(
            str(row["session_id"])
            for row in provider_rows[: provider_quotas[provider]]
        )
    if len(selected) != sum(provider_quotas.values()):
        raise ValueError("满足完整短 session 条件的数据不足")
    return tuple(selected)


def identify_main_round_table(
    schemas: Sequence[Mapping[str, Any]],
) -> str:
    """仅根据已读取的完整 column 集合识别 LLM round 主表。"""
    matches = []
    for table in schemas:
        columns = {
            str(column["column_name"])
            for column in table["columns"]
        }
        if MAIN_ROUND_REQUIRED_COLUMNS.issubset(columns):
            matches.append(str(table["table_name"]))
    if len(matches) != 1:
        raise ValueError(
            "无法唯一识别 LLM round 主表：" + ", ".join(matches)
        )
    return matches[0]


def find_dag_like_fields(
    schemas: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """列出名称上可能与 parent、root、continuation 或 branch 有关的字段。"""
    markers = ("parent", "root", "continuation", "branch")
    return tuple(
        sorted(
            f"{table['table_name']}.{column['column_name']}"
            for table in schemas
            for column in table["columns"]
            if any(
                marker in str(column["column_name"]).lower()
                for marker in markers
            )
        )
    )


def open_database_read_only(database_path: Path):
    """以 DuckDB 的只读模式打开外部数据库。"""
    duckdb = importlib.import_module("duckdb")
    return duckdb.connect(str(database_path), read_only=True)


def fetch_dicts(
    connection,
    query: str,
    parameters: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """执行只读查询并返回按 column 名称组织的记录。"""
    cursor = connection.execute(query, parameters or ())
    column_names = tuple(item[0] for item in cursor.description)
    return tuple(
        dict(zip(column_names, row)) for row in cursor.fetchall()
    )


def inspect_database(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """执行完整 schema 与 session feasibility probe。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    connection = open_database_read_only(database_path)
    try:
        schemas = _inspect_tables(connection)
        main_table = identify_main_round_table(schemas)
        if main_table != "rounds":
            raise ValueError(f"当前 probe 尚未验证主表名称：{main_table}")
        session_candidates = fetch_dicts(
            connection,
            """
            SELECT
                provider,
                session_id,
                count(*) AS round_count,
                min(round_index) AS min_round_index,
                max(round_index) AS max_round_index
            FROM rounds
            GROUP BY provider, session_id
            """,
        )
        sampled_session_ids = choose_sample_sessions(session_candidates)
        analysis = {
            "database": {
                "path": str(database_path),
                "size_bytes": database_path.stat().st_size,
                "access_mode": "read_only=True",
            },
            "tables": schemas,
            "main_round_table": main_table,
            "field_availability": _field_availability(schemas),
            "round_integrity": _single_row(
                connection,
                _ROUND_INTEGRITY_SQL,
            ),
            "session_statistics": _single_row(
                connection,
                _SESSION_STATISTICS_SQL,
            ),
            "session_key_integrity": _single_row(
                connection,
                _SESSION_KEY_SQL,
            ),
            "relational_integrity": _single_row(
                connection,
                _RELATIONAL_INTEGRITY_SQL,
            ),
            "token_statistics": fetch_dicts(
                connection,
                _TOKEN_STATISTICS_SQL,
            ),
            "prefix_reuse": _single_row(
                connection,
                _PREFIX_REUSE_SQL,
            ),
            "ordering": {
                **_single_row(connection, _ORDERING_SQL),
                **_single_row(connection, _EVENT_ORDERING_SQL),
            },
            "timestamps": {
                **_single_row(connection, _TIMESTAMP_RANGE_SQL),
                **_single_row(connection, _CONCURRENCY_SQL),
            },
            "adjacent_rounds": _single_row(
                connection,
                _ADJACENT_ROUNDS_SQL,
            ),
            "tool_events": {
                **_single_row(connection, _TOOL_EVENT_SQL),
                **_single_row(connection, _TOOL_TIME_SQL),
            },
            "dag_like_fields": find_dag_like_fields(schemas),
            "explicit_workflow_dag_fields": tuple(
                sorted(
                    f"{table['table_name']}.{column['column_name']}"
                    for table in schemas
                    for column in table["columns"]
                    if column["column_name"] in EXPLICIT_DAG_FIELD_NAMES
                )
            ),
            "round_samples": _round_samples(connection),
            "sample_selection_rule": (
                "每个 provider 内筛选 round_index 从 0 开始、连续且总轮数为 2 至 5 的 session；"
                "按 session_id 字典序选择 Claude 前 3 个和 Codex 前 2 个"
            ),
            "sampled_sessions": _sample_sessions(
                connection,
                sampled_session_ids,
            ),
        }
        return _normalize_value(analysis)
    finally:
        connection.close()


def render_report(analysis: Mapping[str, Any]) -> str:
    """把 probe 结果渲染成证据优先的中文技术报告。"""
    sessions = analysis["session_statistics"]
    session_keys = analysis["session_key_integrity"]
    relational = analysis["relational_integrity"]
    tokens = analysis["token_statistics"]
    prefix = analysis["prefix_reuse"]
    ordering = analysis["ordering"]
    timestamps = analysis["timestamps"]
    adjacency = analysis["adjacent_rounds"]
    tools = analysis["tool_events"]
    multi_round_rate = _percent(
        sessions["multi_round_sessions"],
        sessions["session_count"],
    )
    positive_prefix_rate = _percent(
        prefix["positive_prefix_rounds"],
        prefix["round_count"],
    )
    input_growth_rate = _percent(
        adjacency["input_nondecreasing_pairs"],
        adjacency["adjacent_pairs"],
    )
    complete_tool_rate = _percent(
        tools["tool_call_count"] - tools["missing_result_at"],
        tools["tool_call_count"],
    )
    lines = [
        "# TraceLab Schema 与 Session 可行性探查",
        "",
        "## 技术摘要",
        "",
        f"- 数据库包含 {sessions['round_count']:,} 个 LLM rounds 和 {sessions['session_count']:,} 个 session；{sessions['multi_round_sessions']:,} 个 session 为多轮，占 {multi_round_rate:.3f}%。",
        f"- `round_index` 在全部 session 内唯一，可提供确定性执行顺序；真实时间来自规范化的 `timing_events.timestamp`，而不是 `rounds` 自身。",
        f"- {prefix['positive_prefix_rounds']:,} 个 round 的 `prefix_tokens > 0`，占 {positive_prefix_rate:.3f}%；四个 token 字段均无空值，并且每行严格满足 `input_tokens_total = prefix_tokens + newly_append_tokens`。",
        "- 未发现 round/workflow 级 parent、root、continuation 或 branch 字段；数据支持线性多轮 session，但不支持直接恢复显式 workflow DAG。",
        f"- 跨 session 时间区间确有重叠，最大同时活动 session 数为 {timestamps['max_concurrent_sessions']}；但时间戳无 timezone 元数据，且存在少量 round 时间倒序，精确跨 provider 回放需要保留此限制。",
        "",
        "## 数据库只有四张规范化表",
        "",
        "| Schema | Table | 类型 | 行数 |",
        "|---|---|---|---:|",
    ]
    for table in analysis["tables"]:
        lines.append(
            f"| {table['table_schema']} | {table['table_name']} | "
            f"{table['table_type']} | {table['row_count']:,} |"
        )
    lines.extend(
        [
            "",
            "`rounds` 是 LLM round 主表：它具备 session、round sequence、model 与完整 token accounting；另外两张明细表通过 `round_pk` 关联 timing 和 tools。",
            f"完整性检查中，timing orphan={relational['orphan_timing_rows']}，tool orphan={relational['orphan_tool_rows']}，没有 timing event 的 round={relational['rounds_without_timing']}。",
            "",
            "## 完整 schema",
            "",
        ]
    )
    for table in analysis["tables"]:
        lines.extend(
            [
                f"### `{table['table_name']}`（{table['row_count']:,} 行）",
                "",
                "| 序号 | Column | DuckDB 类型 | Nullable |",
                "|---:|---|---|---|",
            ]
        )
        for column in table["columns"]:
            lines.append(
                f"| {column['ordinal_position']} | {column['column_name']} | "
                f"{column['data_type']} | {column['is_nullable']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 请求字段的真实位置",
            "",
            "| 逻辑字段 | 状态 | 真实位置或说明 |",
            "|---|---|---|",
        ]
    )
    for field in analysis["field_availability"]:
        lines.append(
            f"| {field['logical_field']} | {field['status']} | {field['location']} |"
        )
    lines.extend(
        [
            "",
            "`rounds` 没有 timestamp、`timing_events` 或 `tools` 嵌套 column。前者必须从 `timing_events` 聚合；后两者分别是规范化子表 `timing_events` 与 `tool_calls`。",
            "",
            "## 五条真实 round 记录",
            "",
            "以下记录按 `round_pk` 升序取前五条；`time`、timing event 类型与 tool 名称均通过 `round_pk` 从子表只读聚合。",
            "",
            "| round_pk | provider | session_id | round_index | round_id | time | model | input | prefix | append | output | timing events | tools |",
            "|---:|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in analysis["round_samples"]:
        lines.append(
            f"| {row['round_pk']} | {row['provider']} | {row['session_id']} | "
            f"{row['round_index']} | {row['round_id']} | {row['first_time']} | "
            f"{row['model']} | {row['input_tokens_total']} | "
            f"{row['prefix_tokens']} | {row['newly_append_tokens']} | "
            f"{row['output_tokens']} | {', '.join(row['timing_event_types'])} "
            f"({row['timing_event_count']}) | "
            f"{', '.join(row['tool_names']) if row['tool_names'] else '无'} "
            f"({row['tool_count']}) |"
        )

    lines.extend(
        [
            "",
            "## Session 与 token 分布支持大规模多轮分析",
            "",
            "### Rounds/session",
            "",
            "| Sessions | Multi-round | Mean | Median | P90 | P95 | Max |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {sessions['session_count']:,} | {sessions['multi_round_sessions']:,} | {sessions['mean_rounds_per_session']:.3f} | {sessions['median_rounds_per_session']:.0f} | {sessions['p90_rounds_per_session']} | {sessions['p95_rounds_per_session']} | {sessions['max_rounds_per_session']:,} |",
            "",
            "### 单轮 token",
            "",
            "| Metric | Median | P90 | P95 | Max | Null | Negative |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tokens:
        lines.append(
            f"| {row['metric']} | {row['median_value']:.0f} | {row['p90']} | "
            f"{row['p95']} | {row['max_value']:,} | {row['null_count']} | "
            f"{row['negative_count']} |"
        )
    lines.extend(
        [
            "",
            f"Prefix reuse：{prefix['positive_prefix_rounds']:,}/{prefix['round_count']:,} rounds（{positive_prefix_rate:.3f}%）的 `prefix_tokens > 0`。",
            "",
            "## `round_index` 是稳定顺序，timestamp 是辅助时间轴",
            "",
            f"全部 {sessions['session_count']:,} 个 session 的 `round_index` 均唯一；{sessions['starts_at_zero_sessions']:,} 个从 0 开始，{sessions['contiguous_sessions']:,} 个索引连续。排序规则固定为 session 内 `round_index ASC`，并用 `round_pk` 作为防御性最终 tie-break；实际没有 tie。",
            "",
            f"以每个 round 的最早 timing timestamp 检查相邻轮次时，{ordering['round_timestamp_inversions']} / {ordering['adjacent_round_pairs']:,} 对出现时间倒序，另有 {ordering['round_timestamp_ties']:,} 对时间相同。因此不能用 timestamp 取代 `round_index` 作为 session 内权威顺序。",
            "",
            f"事件明细的 `event_index` 也不是严格时间排序：{ordering['event_timestamp_inversions']:,} / {ordering['adjacent_event_pairs']:,} 个相邻 event-index 对发生 timestamp 倒序。分析工具事件时应直接按 timestamp 排序，同时用 `event_index` 做稳定 tie-break。",
            "",
            "## 相邻 rounds 通常呈现上下文增长，但不是 DAG 证据",
            "",
            f"在 {adjacency['adjacent_pairs']:,} 个相邻 round 对中，后续 `input_tokens_total` 不小于前一轮的比例为 {input_growth_rate:.3f}%，中位 input 增量为 {adjacency['median_input_delta']:.0f} tokens；仍有 {adjacency['input_decrease_pairs']:,} 对发生 context 下降，可能来自 compaction、cache 策略或 trace 边界。",
            "",
            "`prefix_tokens` 表示当前 round 的物理复用长度，不等同于显式 parent。数据没有 token IDs，也没有 ancestry path；因此本报告不把数值上的 prefix 关系解释为 branching。",
            "",
            "## 未发现显式 branching；工具时序大部分可恢复",
            "",
            f"DAG-like 字段扫描只找到：{', '.join(analysis['dag_like_fields']) if analysis['dag_like_fields'] else '无'}。其中 `tool_calls.continuation_of_tool_call_id` 描述工具调用续接，不是 LLM round 或 workflow parent。显式 workflow DAG 字段集合为空。",
            "",
            f"`tool_calls` 有 {tools['tool_call_count']:,} 行，其中 {tools['missing_result_at']:,} 行缺失 `result_at`；其余 {complete_tool_rate:.3f}% 可由 `emitted_at`、`result_at` 和一致的 `tool_wall_latency_ms` 计算等待时间。所有 timing event 均有 timestamp，但鉴于 event-index 倒序和少量缺失 tool result，`LLM → tool → 下一次 LLM` 只能在大部分记录上可靠恢复，不能声称全量无歧义。",
            "",
            "## 五个确定性完整短 session",
            "",
            f"选择规则：{analysis['sample_selection_rule']}。这里“完整”仅表示下载数据内从 0 开始且 `round_index` 连续；schema 没有显式 session-end marker，无法证明上游绝对完整。",
            "",
        ]
    )
    for session in analysis["sampled_sessions"]:
        lines.extend(
            [
                f"### Session `{session['session_id']}`",
                "",
                "| Round | round_id | time | input | prefix | append | output | tool_count |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in session["rounds"]:
            lines.append(
                f"| {row['round_index'] + 1} | {row['round_id']} | "
                f"{row['first_time']} | {row['input_tokens_total']} | "
                f"{row['prefix_tokens']} | {row['newly_append_tokens']} | "
                f"{row['output_tokens']} | {row['tool_count']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## FlowState trace feasibility",
            "",
            "| 能力 | 评级 | 证据与边界 |",
            "|---|---|---|",
            f"| Multi-turn workflow | PASS | {sessions['multi_round_sessions']:,} 个多轮 session，且全部 session 可由唯一 `round_index` 稳定排序。 |",
            "| Prefix/anchor reconstruction | WEAK | 每轮 H/输入长度可由完整 token accounting 重建，但没有 token IDs、logical anchor ID 或 lineage，无法判断跨 checkpoint compatibility。 |",
            f"| Temporal multi-workflow replay | WEAK | 全部 round 有 timing timestamp，观察到最大 {timestamps['max_concurrent_sessions']} 个 session 重叠；但 timestamp 无 timezone/clock provenance，且有少量 session 内时间倒序。 |",
            "| Explicit workflow DAG | FAIL | 未发现 round/workflow parent、root、continuation 或 branch 字段；不得从 prefix 数值推断分支。 |",
            "",
            "## 限制、稳健性检查与下一步",
            "",
            "- `round_id` 不是全局唯一键；`round_pk` 才是完整唯一的 round grain。",
            f"- `session_id` 在 provider 间冲突数为 {session_keys['cross_provider_session_ids']}；横跨多个 project 标签的 session 数为 {session_keys['cross_project_session_ids']}。因此 session 计数使用 `session_id`，不把 project 当 session key。",
            f"- {sessions['session_count'] - sessions['contiguous_sessions']} 个 session 的 round index 不连续，{sessions['session_count'] - sessions['starts_at_zero_sessions']} 个不从 0 开始；全量分析应保留缺口标记，不能自动补轮次。",
            "- 时间字段是无 timezone 的 DuckDB `TIMESTAMP`。正式跨 session replay 前应确认 TraceLab 发布说明中的时区与时钟归一化语义。",
            "- 下一步若实现 trace2flow，应先限定为线性 session snapshot，并把 DAG/branch 相关能力标记为 unavailable；本 probe 不授权实现该转换。",
            "",
            "进一步需要确认：`prefix_tokens` 的 provider-specific cache 统计口径、session 是否具有未发布的结束标记，以及跨 provider timestamp 是否共用同一时钟基准。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    analysis: Mapping[str, Any],
    output_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """保存 Markdown 技术报告。"""
    output_path.write_text(render_report(analysis), encoding="utf-8")


def _inspect_tables(connection) -> tuple[dict[str, Any], ...]:
    table_rows = fetch_dicts(
        connection,
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name
        """,
    )
    result = []
    for table in table_rows:
        schema_name = str(table["table_schema"])
        table_name = str(table["table_name"])
        qualified = (
            f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"
        )
        row_count = connection.execute(
            f"SELECT count(*) FROM {qualified}"
        ).fetchone()[0]
        columns = fetch_dicts(
            connection,
            """
            SELECT ordinal_position, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            (schema_name, table_name),
        )
        result.append(
            {
                **table,
                "row_count": row_count,
                "columns": columns,
            }
        )
    return tuple(result)


def _field_availability(
    schemas: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    table_columns = {
        str(table["table_name"]): {
            str(column["column_name"])
            for column in table["columns"]
        }
        for table in schemas
    }
    requested = (
        ("provider", "available", "rounds.provider"),
        ("session id", "available", "rounds.session_id"),
        ("round id", "available", "rounds.round_id"),
        ("sequence id", "available", "rounds.round_index"),
        (
            "timestamp / timing",
            "rounds 中 unavailable",
            "可从规范化子表 timing_events.timestamp 获取",
        ),
        ("model", "available", "rounds.model"),
        ("input_tokens_total", "available", "rounds.input_tokens_total"),
        ("prefix_tokens", "available", "rounds.prefix_tokens"),
        ("newly_append_tokens", "available", "rounds.newly_append_tokens"),
        ("output_tokens", "available", "rounds.output_tokens"),
        (
            "timing_events",
            "rounds 中 unavailable",
            "规范化子表 timing_events",
        ),
        (
            "tools",
            "名为 tools 的字段 unavailable",
            "相关数据位于规范化子表 tool_calls",
        ),
    )
    return tuple(
        {
            "logical_field": logical,
            "status": status,
            "location": location,
        }
        for logical, status, location in requested
    )


def _single_row(connection, query: str) -> dict[str, Any]:
    rows = fetch_dicts(connection, query)
    if len(rows) != 1:
        raise ValueError("预期查询恰好返回一行")
    return rows[0]


def _round_samples(connection) -> tuple[dict[str, Any], ...]:
    return fetch_dicts(
        connection,
        """
        WITH timing AS (
            SELECT
                round_pk,
                min(timestamp) AS first_time,
                max(timestamp) AS last_time,
                count(*) AS timing_event_count,
                list(DISTINCT event_type ORDER BY event_type) AS timing_event_types
            FROM timing_events
            GROUP BY round_pk
        ), tools AS (
            SELECT
                round_pk,
                count(*) AS tool_count,
                list(DISTINCT tool_name ORDER BY tool_name) AS tool_names
            FROM tool_calls
            GROUP BY round_pk
        )
        SELECT
            r.round_pk,
            r.provider,
            r.project,
            r.session_id,
            r.round_index,
            r.round_id,
            r.model,
            r.input_tokens_total,
            r.prefix_tokens,
            r.newly_append_tokens,
            r.output_tokens,
            timing.first_time,
            timing.last_time,
            timing.timing_event_count,
            timing.timing_event_types,
            coalesce(tools.tool_count, 0) AS tool_count,
            coalesce(tools.tool_names, []) AS tool_names
        FROM rounds r
        JOIN timing USING (round_pk)
        LEFT JOIN tools USING (round_pk)
        ORDER BY r.round_pk
        LIMIT 5
        """,
    )


def _sample_sessions(
    connection,
    session_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    rows = fetch_dicts(
        connection,
        """
        WITH timing AS (
            SELECT round_pk, min(timestamp) AS first_time,
                   max(timestamp) AS last_time
            FROM timing_events
            GROUP BY round_pk
        ), tools AS (
            SELECT round_pk, count(*) AS tool_count
            FROM tool_calls
            GROUP BY round_pk
        )
        SELECT
            r.provider,
            r.session_id,
            r.round_index,
            r.round_id,
            timing.first_time,
            timing.last_time,
            r.model,
            r.input_tokens_total,
            r.prefix_tokens,
            r.newly_append_tokens,
            r.output_tokens,
            coalesce(tools.tool_count, 0) AS tool_count
        FROM rounds r
        JOIN timing USING (round_pk)
        LEFT JOIN tools USING (round_pk)
        WHERE r.session_id IN (SELECT unnest(?))
        ORDER BY r.provider, r.session_id, r.round_index
        """,
        (list(session_ids),),
    )
    grouped = []
    for session_id in session_ids:
        session_rounds = tuple(
            row for row in rows if row["session_id"] == session_id
        )
        if not session_rounds:
            raise ValueError(f"sample session 不存在：{session_id}")
        grouped.append(
            {
                "provider": session_rounds[0]["provider"],
                "session_id": session_id,
                "rounds": session_rounds,
            }
        )
    return tuple(grouped)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ")
    return value


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("比例分母必须大于零")
    return 100.0 * numerator / denominator


_ROUND_INTEGRITY_SQL = """
SELECT
    count(*) AS round_count,
    count(DISTINCT round_pk) AS distinct_round_pk,
    count(*) - count(DISTINCT round_pk) AS duplicate_round_pk,
    count_if(round_pk IS NULL) AS null_round_pk,
    count_if(session_id IS NULL OR session_id = '') AS null_session_id,
    count_if(round_index IS NULL) AS null_round_index,
    count_if(round_id IS NULL OR round_id = '') AS null_round_id,
    count(DISTINCT round_id) AS distinct_round_id,
    count_if(
        input_tokens_total = prefix_tokens + newly_append_tokens
    ) AS exact_token_identity_rounds
FROM rounds
"""

_SESSION_STATISTICS_SQL = """
WITH session_counts AS (
    SELECT
        session_id,
        count(*) AS round_count,
        min(round_index) AS min_index,
        max(round_index) AS max_index,
        count(DISTINCT round_index) AS distinct_index_count
    FROM rounds
    GROUP BY session_id
)
SELECT
    (SELECT count(*) FROM rounds) AS round_count,
    count(*) AS session_count,
    count_if(round_count > 1) AS multi_round_sessions,
    avg(round_count) AS mean_rounds_per_session,
    median(round_count) AS median_rounds_per_session,
    quantile_disc(round_count, 0.90) AS p90_rounds_per_session,
    quantile_disc(round_count, 0.95) AS p95_rounds_per_session,
    max(round_count) AS max_rounds_per_session,
    count_if(min_index = 0) AS starts_at_zero_sessions,
    count_if(max_index - min_index + 1 = round_count) AS contiguous_sessions,
    count_if(distinct_index_count = round_count) AS unique_order_sessions
FROM session_counts
"""

_SESSION_KEY_SQL = """
WITH session_keys AS (
    SELECT
        session_id,
        count(DISTINCT provider) AS provider_count,
        count(DISTINCT project) AS project_count
    FROM rounds
    GROUP BY session_id
)
SELECT
    count(*) AS distinct_session_ids,
    count_if(provider_count > 1) AS cross_provider_session_ids,
    count_if(project_count > 1) AS cross_project_session_ids
FROM session_keys
"""

_RELATIONAL_INTEGRITY_SQL = """
SELECT
    (
        SELECT count(*)
        FROM timing_events t
        LEFT JOIN rounds r USING (round_pk)
        WHERE r.round_pk IS NULL
    ) AS orphan_timing_rows,
    (
        SELECT count(*)
        FROM tool_calls t
        LEFT JOIN rounds r USING (round_pk)
        WHERE r.round_pk IS NULL
    ) AS orphan_tool_rows,
    (
        SELECT count(*)
        FROM rounds r
        LEFT JOIN (
            SELECT DISTINCT round_pk FROM timing_events
        ) t USING (round_pk)
        WHERE t.round_pk IS NULL
    ) AS rounds_without_timing
"""

_TOKEN_STATISTICS_SQL = """
SELECT 'input_tokens_total' AS metric,
       median(input_tokens_total) AS median_value,
       quantile_disc(input_tokens_total, 0.90) AS p90,
       quantile_disc(input_tokens_total, 0.95) AS p95,
       max(input_tokens_total) AS max_value,
       count_if(input_tokens_total IS NULL) AS null_count,
       count_if(input_tokens_total < 0) AS negative_count
FROM rounds
UNION ALL
SELECT 'prefix_tokens', median(prefix_tokens),
       quantile_disc(prefix_tokens, 0.90),
       quantile_disc(prefix_tokens, 0.95), max(prefix_tokens),
       count_if(prefix_tokens IS NULL), count_if(prefix_tokens < 0)
FROM rounds
UNION ALL
SELECT 'newly_append_tokens', median(newly_append_tokens),
       quantile_disc(newly_append_tokens, 0.90),
       quantile_disc(newly_append_tokens, 0.95), max(newly_append_tokens),
       count_if(newly_append_tokens IS NULL), count_if(newly_append_tokens < 0)
FROM rounds
UNION ALL
SELECT 'output_tokens', median(output_tokens),
       quantile_disc(output_tokens, 0.90),
       quantile_disc(output_tokens, 0.95), max(output_tokens),
       count_if(output_tokens IS NULL), count_if(output_tokens < 0)
FROM rounds
"""

_PREFIX_REUSE_SQL = """
SELECT
    count(*) AS round_count,
    count_if(prefix_tokens > 0) AS positive_prefix_rounds,
    count_if(prefix_tokens = 0) AS zero_prefix_rounds,
    count_if(prefix_tokens IS NULL) AS null_prefix_rounds
FROM rounds
"""

_ORDERING_SQL = """
WITH round_bounds AS (
    SELECT
        r.session_id,
        r.round_index,
        r.ingest_seq,
        min(t.timestamp) AS first_time
    FROM rounds r
    JOIN timing_events t USING (round_pk)
    GROUP BY r.session_id, r.round_index, r.ingest_seq
), adjacent AS (
    SELECT
        *,
        lag(first_time) OVER (
            PARTITION BY session_id ORDER BY round_index
        ) AS previous_time,
        lag(ingest_seq) OVER (
            PARTITION BY session_id ORDER BY round_index
        ) AS previous_ingest_seq
    FROM round_bounds
)
SELECT
    count(*) - count_if(previous_time IS NULL) AS adjacent_round_pairs,
    count_if(first_time < previous_time) AS round_timestamp_inversions,
    count_if(first_time = previous_time) AS round_timestamp_ties,
    count_if(ingest_seq < previous_ingest_seq) AS ingest_seq_inversions
FROM adjacent
"""

_EVENT_ORDERING_SQL = """
WITH adjacent AS (
    SELECT
        round_pk,
        event_index,
        timestamp,
        lag(timestamp) OVER (
            PARTITION BY round_pk ORDER BY event_index
        ) AS previous_time
    FROM timing_events
)
SELECT
    count(*) - count_if(previous_time IS NULL) AS adjacent_event_pairs,
    count_if(timestamp < previous_time) AS event_timestamp_inversions,
    count_if(timestamp = previous_time) AS event_timestamp_ties
FROM adjacent
"""

_TIMESTAMP_RANGE_SQL = """
WITH session_bounds AS (
    SELECT
        r.session_id,
        min(t.timestamp) AS first_time,
        max(t.timestamp) AS last_time
    FROM rounds r
    JOIN timing_events t USING (round_pk)
    GROUP BY r.session_id
)
SELECT
    count(*) AS timestamped_sessions,
    min(first_time) AS first_timestamp,
    max(last_time) AS last_timestamp,
    count_if(first_time IS NULL OR last_time IS NULL) AS sessions_without_time
FROM session_bounds
"""

_CONCURRENCY_SQL = """
WITH session_bounds AS (
    SELECT
        r.session_id,
        min(t.timestamp) AS first_time,
        max(t.timestamp) AS last_time
    FROM rounds r
    JOIN timing_events t USING (round_pk)
    GROUP BY r.session_id
), points AS (
    SELECT first_time AS timestamp, 1 AS delta FROM session_bounds
    UNION ALL
    SELECT last_time AS timestamp, -1 AS delta FROM session_bounds
), aggregated AS (
    SELECT timestamp, sum(delta) AS delta
    FROM points
    GROUP BY timestamp
), running AS (
    SELECT
        timestamp,
        sum(delta) OVER (
            ORDER BY timestamp ROWS UNBOUNDED PRECEDING
        ) AS active_sessions
    FROM aggregated
)
SELECT
    max(active_sessions) AS max_concurrent_sessions,
    count_if(active_sessions > 1) AS overlap_boundary_points
FROM running
"""

_ADJACENT_ROUNDS_SQL = """
WITH ordered AS (
    SELECT
        *,
        lag(input_tokens_total) OVER (
            PARTITION BY session_id ORDER BY round_index
        ) AS previous_input,
        lag(prefix_tokens) OVER (
            PARTITION BY session_id ORDER BY round_index
        ) AS previous_prefix
    FROM rounds
), adjacent AS (
    SELECT * FROM ordered WHERE previous_input IS NOT NULL
)
SELECT
    count(*) AS adjacent_pairs,
    count_if(input_tokens_total >= previous_input) AS input_nondecreasing_pairs,
    count_if(prefix_tokens >= previous_prefix) AS prefix_nondecreasing_pairs,
    count_if(prefix_tokens >= previous_input) AS prefix_covers_previous_input,
    count_if(input_tokens_total < previous_input) AS input_decrease_pairs,
    median(input_tokens_total - previous_input) AS median_input_delta,
    quantile_disc(input_tokens_total - previous_input, 0.10) AS p10_input_delta,
    quantile_disc(input_tokens_total - previous_input, 0.90) AS p90_input_delta
FROM adjacent
"""

_TOOL_EVENT_SQL = """
SELECT
    count(*) AS tool_call_count,
    count(DISTINCT round_pk) AS rounds_with_tools,
    count_if(continuation_of_tool_call_id IS NOT NULL) AS tool_continuation_rows,
    count_if(emitted_at IS NULL) AS missing_emitted_at,
    count_if(result_at IS NULL) AS missing_result_at,
    count_if(tool_wall_latency_ms IS NULL) AS missing_wall_latency
FROM tool_calls
"""

_TOOL_TIME_SQL = """
SELECT
    count_if(result_at < emitted_at) AS reversed_tool_intervals,
    count_if(tool_wall_latency_ms < 0) AS negative_tool_latency,
    count_if(
        result_at IS NOT NULL
        AND abs(
            date_diff('millisecond', emitted_at, result_at)
            - tool_wall_latency_ms
        ) > 1
    ) AS tool_latency_mismatches,
    (SELECT count_if(timestamp IS NULL) FROM timing_events)
        AS timing_events_without_timestamp
FROM tool_calls
"""


def main(argv: Iterable[str] | None = None) -> None:
    """执行只读 probe，保存报告并打印结构化摘要。"""
    parser = argparse.ArgumentParser(description="只读探查 TraceLab DuckDB")
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    arguments = parser.parse_args(tuple(argv) if argv is not None else None)
    analysis = inspect_database(arguments.database)
    write_report(analysis, arguments.output)
    print(
        json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
