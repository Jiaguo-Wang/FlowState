#!/usr/bin/env python3
"""只读审计 TraceLab 的 Agent Run、在线信息边界与 anchor 可行性。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.public_agent_trace.tracelab_probe import (
    DEFAULT_DATABASE_PATH,
    fetch_dicts,
    open_database_read_only,
)


DEFAULT_REPORT_PATH = Path(__file__).with_name("TRACELAB_SEMANTICS.md")

ONLINE_KNOWN_INFORMATION = (
    "当前 session 与 Agent Run 标识",
    "已经发生的 rounds",
    "当前 round_index",
    "当前 input_tokens_total、prefix_tokens、newly_append_tokens、output_tokens",
    "当前轮已经发出的 tool calls",
    "已经发生的 timing events",
)

ONLINE_FORBIDDEN_FUTURE_INFORMATION = (
    "下一轮 prefix_tokens",
    "下一轮 input_tokens_total",
    "下一轮 output_tokens",
    "尚未返回的 future tool result 大小",
    "future timing events",
)


@dataclass(frozen=True)
class OnlineRoundSnapshot:
    """只包含当前决策时点已经可知的 round 信息。"""

    provider: str
    session_id: str
    run_ordinal: int
    round_index: int
    input_tokens_total: int
    prefix_tokens: int
    newly_append_tokens: int
    output_tokens: int
    emitted_tool_call_ids: tuple[str, ...]
    occurred_timing_event_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider 不能为空")
        if not self.session_id:
            raise ValueError("session_id 不能为空")
        if self.run_ordinal <= 0:
            raise ValueError("run_ordinal 必须大于零")
        if self.round_index < 0:
            raise ValueError("round_index 必须非负")
        token_values = (
            self.input_tokens_total,
            self.prefix_tokens,
            self.newly_append_tokens,
            self.output_tokens,
        )
        if any(value < 0 for value in token_values):
            raise ValueError("当前 round 的 token 计数必须非负")
        if self.input_tokens_total != (
            self.prefix_tokens + self.newly_append_tokens
        ):
            raise ValueError("当前 round 的 input token 恒等式不成立")


@dataclass(frozen=True)
class AgentRunRound:
    """用于确定性 Agent Run 分段的最小 round 记录。"""

    provider: str
    session_id: str
    round_index: int
    current_user_message_count: int

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index 必须非负")
        if self.current_user_message_count < 0:
            raise ValueError("current_user_message_count 必须非负")


@dataclass(frozen=True)
class AgentRunSegment:
    """由一个真实 user_message round 开始的连续 Agent Run。"""

    provider: str
    session_id: str
    run_ordinal: int
    round_indices: tuple[int, ...]


@dataclass(frozen=True)
class PendingContinuationSignal:
    """当前工具调用完成后可能需要下一次 LLM 调用的单一信号。"""

    provider: str
    session_id: str
    run_ordinal: int
    source_round_index: int
    tool_call_count: int


@dataclass(frozen=True)
class KnownHistoricalAnchor:
    """仅由当前轮已知输入边界构造的逻辑 token anchor。"""

    token_pos: int
    source_round_index: int
    source: str = "current_input_boundary"


@dataclass(frozen=True)
class AnchorValidation:
    """使用下一轮 prefix 对已冻结 anchor 做事后验证的结果。"""

    anchor_token_pos: int
    actual_next_prefix_tokens: int
    signed_delta_tokens: int
    absolute_delta_tokens: int


def segment_agent_runs(
    rounds: Iterable[AgentRunRound],
) -> tuple[AgentRunSegment, ...]:
    """按真实 user_message 边界分段，并忽略首个边界前的孤立 rounds。"""
    grouped: dict[tuple[str, str], list[AgentRunRound]] = {}
    for round_record in rounds:
        key = (round_record.provider, round_record.session_id)
        grouped.setdefault(key, []).append(round_record)

    segments: list[AgentRunSegment] = []
    for (provider, session_id), records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda item: item.round_index)
        indices = tuple(item.round_index for item in ordered)
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"session 内 round_index 重复：{provider}/{session_id}"
            )
        current: list[int] | None = None
        run_ordinal = 0
        for record in ordered:
            if record.current_user_message_count > 0:
                if current is not None:
                    segments.append(
                        AgentRunSegment(
                            provider=provider,
                            session_id=session_id,
                            run_ordinal=run_ordinal,
                            round_indices=tuple(current),
                        )
                    )
                run_ordinal += 1
                current = []
            if current is not None:
                current.append(record.round_index)
        if current is not None:
            segments.append(
                AgentRunSegment(
                    provider=provider,
                    session_id=session_id,
                    run_ordinal=run_ordinal,
                    round_indices=tuple(current),
                )
            )
    return tuple(segments)


def build_online_snapshot(
    *,
    provider: str,
    session_id: str,
    run_ordinal: int,
    round_index: int,
    input_tokens_total: int,
    prefix_tokens: int,
    newly_append_tokens: int,
    output_tokens: int,
    emitted_tool_call_ids: Sequence[str] = (),
    occurred_timing_event_types: Sequence[str] = (),
) -> OnlineRoundSnapshot:
    """通过显式当前轮参数构造快照，使 future 字段无法混入接口。"""
    return OnlineRoundSnapshot(
        provider=provider,
        session_id=session_id,
        run_ordinal=run_ordinal,
        round_index=round_index,
        input_tokens_total=input_tokens_total,
        prefix_tokens=prefix_tokens,
        newly_append_tokens=newly_append_tokens,
        output_tokens=output_tokens,
        emitted_tool_call_ids=tuple(emitted_tool_call_ids),
        occurred_timing_event_types=tuple(occurred_timing_event_types),
    )


def build_pending_continuation_signal(
    snapshot: OnlineRoundSnapshot,
) -> PendingContinuationSignal | None:
    """有至少一个已发出工具调用时，只构造一个 LLM-level pending 信号。"""
    if not snapshot.emitted_tool_call_ids:
        return None
    return PendingContinuationSignal(
        provider=snapshot.provider,
        session_id=snapshot.session_id,
        run_ordinal=snapshot.run_ordinal,
        source_round_index=snapshot.round_index,
        tool_call_count=len(snapshot.emitted_tool_call_ids),
    )


def build_known_historical_anchor(
    snapshot: OnlineRoundSnapshot,
) -> KnownHistoricalAnchor:
    """把当前轮已经执行的输入边界冻结为已知历史 anchor。"""
    return KnownHistoricalAnchor(
        token_pos=snapshot.input_tokens_total,
        source_round_index=snapshot.round_index,
    )


def validate_anchor_ex_post(
    anchor: KnownHistoricalAnchor,
    *,
    actual_next_prefix_tokens: int,
) -> AnchorValidation:
    """仅在事后用下一轮 prefix 评估 anchor，不改变构造规则。"""
    if actual_next_prefix_tokens < 0:
        raise ValueError("actual_next_prefix_tokens 必须非负")
    delta = actual_next_prefix_tokens - anchor.token_pos
    return AnchorValidation(
        anchor_token_pos=anchor.token_pos,
        actual_next_prefix_tokens=actual_next_prefix_tokens,
        signed_delta_tokens=delta,
        absolute_delta_tokens=abs(delta),
    )


def inspect_semantics(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    """对外部 TraceLab DuckDB 执行只读语义审计。"""
    if not database_path.is_file():
        raise FileNotFoundError(f"TraceLab 数据库不存在：{database_path}")
    connection = open_database_read_only(database_path)
    try:
        result = {
            "database": {
                "path": str(database_path),
                "size_bytes": database_path.stat().st_size,
                "access_mode": "read_only=True",
            },
            "boundary_validity": _single_row(connection, _BOUNDARY_SQL),
            "run_statistics": fetch_dicts(connection, _RUN_STATISTICS_SQL),
            "runs_per_session": fetch_dicts(
                connection, _RUNS_PER_SESSION_SQL
            ),
            "leading_rounds": _single_row(connection, _LEADING_ROUNDS_SQL),
            "tool_continuations": _single_row(
                connection, _TOOL_CONTINUATION_SQL
            ),
            "anchor_validation": fetch_dicts(
                connection, _ANCHOR_VALIDATION_SQL
            ),
            "context_feasibility": _single_row(
                connection, _CONTEXT_FEASIBILITY_SQL
            ),
            "temporal_overlap": fetch_dicts(
                connection, _TEMPORAL_OVERLAP_SQL
            ),
            "online_known_information": ONLINE_KNOWN_INFORMATION,
            "online_forbidden_future_information": (
                ONLINE_FORBIDDEN_FUTURE_INFORMATION
            ),
            "anchor_rule": (
                "在当前 round 的 tool call 已经发出时，使用当前轮 "
                "input_tokens_total 作为 Known Historical Anchor；"
                "不加入 output_tokens，也不读取下一轮字段"
            ),
            "gates": {
                "agent_run_segmentation": "WEAK",
                "pending_continuation": "WEAK",
                "anchor_reconstruction": "WEAK",
                "direct_policy_evaluation": "WEAK",
            },
        }
        return _normalize(result)
    finally:
        connection.close()


def render_report(analysis: Mapping[str, Any]) -> str:
    """把语义审计结果渲染为中文技术报告。"""
    boundary = analysis["boundary_validity"]
    runs = {row["provider"]: row for row in analysis["run_statistics"]}
    overall = runs["全部"]
    runs_per_session = {
        row["provider"]: row for row in analysis["runs_per_session"]
    }
    leading = analysis["leading_rounds"]
    tools = analysis["tool_continuations"]
    anchors = {
        row["provider"]: row for row in analysis["anchor_validation"]
    }
    anchor = anchors["全部"]
    context = analysis["context_feasibility"]
    overlap = {
        row["provider"]: row for row in analysis["temporal_overlap"]
    }
    overlap_all = overlap["全部"]

    lines = [
        "# TraceLab 无未来信息语义审计",
        "",
        "## 结论",
        "",
        "TraceLab 可以用真实 `user_message` 边界确定性划分 Agent Run，也可以在工具调用发出后形成一个不依赖未来结果的 LLM-level pending 信号。当前轮 `input_tokens_total` 是合法的已知历史 token 边界，但它不是下一轮物理 prefix 的精确预测器；数据又缺少 token IDs、checkpoint residency 和 lineage，因此四项 gate 均评为 WEAK。",
        "",
        "这意味着后续可以设计显式标注假设的离线 trace2flow，但不能把转换结果当成真实 runtime checkpoint 事实，更不能从 tool-call 数量推断 LLM fanout。",
        "",
        "## Agent Run 边界有真实事件支持",
        "",
        f"在 {boundary['round_count']:,} 个 rounds 中，`current_user_message_count > 0` 的 {boundary['boundary_rounds']:,} 个边界与 `timing_events.event_type = 'user_message'` 的存在性逐行一致：presence mismatch={boundary['presence_mismatch']}，count mismatch={boundary['count_mismatch']}。其中 {boundary['multi_message_boundary_rounds']:,} 个 round 同时包含多个 user messages，但仍只启动一个 Agent Run。",
        "",
        "冻结分段规则：session 内按 `round_index ASC` 排序；每个包含 user message 的 round 启动新 run，直到下一个此类 round 之前。首个 user-message 边界前的 round 不分配给任何 run；不使用 inactivity threshold。",
        "",
        f"共有 {overall['run_count']:,} 个 Agent Runs，覆盖 {overall['sessions_with_runs']:,} 个 session。{leading['leading_rounds']:,} 个 rounds 位于首个边界之前，{leading['sessions_with_leading_rounds']:,} 个 session 受影响，另有 {leading['sessions_without_runs']:,} 个 session 完全没有可识别边界。schema 没有 session-end marker，因此每个 session 的最后一个 run 是右删失的；严格闭合 run 数为 {overall['strictly_closed_runs']:,}。",
        "",
        "### Runs/session",
        "",
        "| Provider | Sessions | Mean | Median | P90 | P95 | Max | Zero-run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in ("全部", "claude", "codex"):
        row = runs_per_session[provider]
        lines.append(
            f"| {provider} | {row['session_count']:,} | {row['mean_runs_per_session']:.3f} | {row['median_runs_per_session']:.0f} | {row['p90_runs_per_session']:.0f} | {row['p95_runs_per_session']:.0f} | {row['max_runs_per_session']:,} | {row['zero_run_sessions']:,} |"
        )
    lines.extend(
        [
            "",
            "### Rounds/run",
            "",
            "| Provider | Runs | Mean | Median | P90 | P95 | Max | 单轮 | >=2 | >=5 | >=10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for provider in ("全部", "claude", "codex"):
        row = runs[provider]
        lines.append(
            f"| {provider} | {row['run_count']:,} | {row['mean_rounds_per_run']:.3f} | {row['median_rounds_per_run']:.0f} | {row['p90_rounds_per_run']:.0f} | {row['p95_rounds_per_run']:.0f} | {row['max_rounds_per_run']:,} | {_ratio(row['single_round_runs'], row['run_count'])} | {_ratio(row['at_least_2_rounds'], row['run_count'])} | {_ratio(row['at_least_5_rounds'], row['run_count'])} | {_ratio(row['at_least_10_rounds'], row['run_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Online 信息边界",
            "",
            "### 当前可用",
            "",
        ]
    )
    lines.extend(
        f"- {item}" for item in analysis["online_known_information"]
    )
    lines.extend(
        [
            "",
            "### 禁止进入 online 构造",
            "",
        ]
    )
    lines.extend(
        f"- {item}"
        for item in analysis["online_forbidden_future_information"]
    )
    lines.extend(
        [
            "",
            "实现使用显式参数构造 `OnlineRoundSnapshot`，类型中不存在 next-round 或 future 字段。事后验证通过另一个函数执行，不能回写 anchor。",
            "",
            "## 工具调用只能产生一个弱 pending 信号",
            "",
            f"有 tool call 的 {tools['tool_rounds']:,} 个 rounds 中，{tools['tool_with_any_next_round']:,} 个事后观察到下一 LLM round（{_ratio(tools['tool_with_any_next_round'], tools['tool_rounds'])}），{tools['tool_without_next_round']:,} 个没有下一轮（{_ratio(tools['tool_without_next_round'], tools['tool_rounds'])}）。有下一轮的工具 round 中，{tools['tool_with_same_run_next_round']:,} 个下一轮仍在同一 Agent Run，{tools['tool_with_new_run_next_round']:,} 个下一轮已由新 user message 启动新 run。",
            "",
            f"没有 tool call 的 {tools['no_tool_rounds']:,} 个 rounds 中，仍有 {tools['no_tool_with_next_round']:,} 个存在下一轮（{_ratio(tools['no_tool_with_next_round'], tools['no_tool_rounds'])}）。这些下一轮事实仅用于事后 characterization，不进入 online 决策。",
            "",
            "冻结 pending 规则：当前 round 已经发出至少一个 tool call 时，创建一个“等待工具完成后可能继续 LLM”的信号。多个 tool calls 只增加该信号的 tool count，不创建多个 FlowState pending continuations。工具失败、用户终止或 session 截断使该信号并非必然兑现，因此评级为 WEAK。",
            "",
            "## Leakage-free anchor",
            "",
            f"候选规则：{analysis['anchor_rule']}。`input_tokens_total` 是当前 prompt 已经确定的输入边界，能够直接映射为逻辑 `token_pos`；加入当前 output 会假设尚未观测到的序列化边界，读取下一轮 prefix 则构成未来信息泄漏。",
            "",
            "该 anchor 只表示已执行的历史边界。TraceLab 没有 token IDs、实际 recurrent checkpoint 生成/驻留状态或 lineage，因此它不能单独证明 checkpoint compatibility。",
            "",
            "### 下一轮 prefix 的事后验证",
            "",
            "主验证 cohort 只包含“当前轮有工具调用，且下一 LLM round 仍属于同一 Agent Run”的记录。signed delta 定义为 `actual_next_prefix_tokens - known_anchor`；验证结果不参与 anchor 规则。",
            "",
            "| Provider | N | Min | P05 | P10 | Median | P90 | P95 | Max | Exact | <=16 | <=64 | <=256 | <=1024 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for provider in ("全部", "claude", "codex"):
        row = anchors[provider]
        lines.append(
            f"| {provider} | {row['observation_count']:,} | {row['min_delta']:,} | {row['p05_delta']:.0f} | {row['p10_delta']:.0f} | {row['median_delta']:.0f} | {row['p90_delta']:.0f} | {row['p95_delta']:.0f} | {row['max_delta']:,} | {_ratio(row['exact_matches'], row['observation_count'])} | {_ratio(row['within_16'], row['observation_count'])} | {_ratio(row['within_64'], row['observation_count'])} | {_ratio(row['within_256'], row['observation_count'])} | {_ratio(row['within_1024'], row['observation_count'])} |"
        )

    lines.extend(
        [
            "",
            f"整体 exact match 仅 {_ratio(anchor['exact_matches'], anchor['observation_count'])}，但绝对误差在 1024 tokens 内的比例为 {_ratio(anchor['within_1024'], anchor['observation_count'])}。Claude 与 Codex 的分布差异明显，说明当前输入边界是 leakage-free 的历史 anchor，却不能被解释成下一轮物理 prefix 的无偏估计。",
            "",
            "## 完整 Agent Run 的 context feasibility",
            "",
            f"只统计存在后续 user-message 边界的 {context['strictly_closed_runs']:,} 个严格闭合 runs；右删失的 session 最后一个 run 不计入“完整 run”。不裁剪、不缩放 token。",
            "",
            "| Context 上限 | Runs | 比例 |",
            "|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("32K", "within_32k"),
        ("64K", "within_64k"),
        ("128K", "within_128k"),
        ("256K", "within_256k"),
    ):
        lines.append(
            f"| {label} | {context[key]:,} | {_ratio(context[key], context['strictly_closed_runs'])} |"
        )
    lines.extend(
        [
            "",
            f"完整 run 的最大 `input_tokens_total` 为 {context['max_input_tokens_total']:,}。本审计不据此选择 context cutoff。",
            "",
            "## Run-level temporal overlap",
            "",
            f"全部 {overlap_all['run_count']:,} 个 runs 均有 timing bounds；{overlap_all['overlap_runs']:,} 个与另一 session 的 run 区间重叠，占 {_ratio(overlap_all['overlap_runs'], overlap_all['run_count'])}。在每个 run 的开始时点计数，concurrency mean={overlap_all['mean_concurrency']:.3f}、median={overlap_all['median_concurrency']:.0f}、P90={overlap_all['p90_concurrency']:.0f}、P95={overlap_all['p95_concurrency']:.0f}、max={overlap_all['max_concurrency']:.0f}。",
            "",
            f"分桶：concurrency=1 有 {overlap_all['concurrency_1']:,} 个 runs，2–4 有 {overlap_all['concurrency_2_4']:,} 个，5–8 有 {overlap_all['concurrency_5_8']:,} 个，>=9 有 {overlap_all['concurrency_9_plus']:,} 个。",
            "",
            "时间戳没有 timezone 或跨 provider clock provenance；区间重叠只能证明数据中观察到的并发形态，不能保证可按同一绝对时钟精确回放。",
            "",
            "## Gate",
            "",
            "| Gate | 评级 | 原因 |",
            "|---|---|---|",
            "| Agent-run segmentation | WEAK | user-message 边界与 timing events 全量一致，且排序确定；但首段缺边界、最后 run 右删失，schema 无 session-end marker。 |",
            "| Leakage-free pending continuation | WEAK | 已发出的 tool call 是当前可知信号，且绝大多数事后存在下一轮；但 continuation 并非必然发生，多个 tool calls 也不能解释为 LLM fanout。 |",
            "| Leakage-free anchor reconstruction | WEAK | 当前输入边界是无泄漏且可映射到 token_pos 的历史 anchor；但缺少 token IDs、lineage 与实际 checkpoint residency，且与下一轮物理 prefix 通常不精确相等。 |",
            "| Direct FlowState policy evaluation | WEAK | pending 与 anchor 均达到 WEAK，可进入严格标注假设的离线 trace2flow；但 candidate/residency/compatibility 仍需外部建模，不能作为真实 runtime allocator 证据。 |",
            "",
            "## 推荐用途与限制",
            "",
            "TraceLab 适合提供真实世界多轮、长上下文、prefix reuse、tool-gap 和 temporal overlap 的 workload evidence。后续若进入 trace2flow，应把当前输入边界规则预先冻结，把工具 round 映射为至多一个 pending，并将 checkpoint candidate、lineage 与 residency 明确标为建模假设。",
            "",
            "不得用下一轮 token 或 timing 字段决定当前 anchor，不得把多个工具调用当作分支，不得从 prefix 数值反推显式 workflow DAG，也不得把该数据集上的离线结果表述为 SGLang runtime correctness。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    analysis: Mapping[str, Any],
    output_path: Path = DEFAULT_REPORT_PATH,
) -> None:
    """将语义审计保存为 Markdown。"""
    output_path.write_text(render_report(analysis), encoding="utf-8")


def _single_row(connection, query: str) -> dict[str, Any]:
    rows = fetch_dicts(connection, query)
    if len(rows) != 1:
        raise ValueError("预期查询恰好返回一行")
    return rows[0]


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_normalize(item) for item in value)
    return value


def _ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise ValueError("比例分母必须大于零")
    return f"{100.0 * numerator / denominator:.3f}%"


_RUN_ASSIGNMENT_CTE = """
WITH ordered AS (
    SELECT
        r.*,
        sum(CASE WHEN current_user_message_count > 0 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY session_id
                ORDER BY round_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS run_ordinal
    FROM rounds r
), assigned AS (
    SELECT * FROM ordered WHERE run_ordinal > 0
), run_sizes AS (
    SELECT
        provider,
        session_id,
        run_ordinal,
        count(*) AS round_count
    FROM assigned
    GROUP BY provider, session_id, run_ordinal
), closed_runs AS (
    SELECT
        *,
        run_ordinal < max(run_ordinal) OVER (PARTITION BY session_id)
            AS is_strictly_closed
    FROM run_sizes
)
"""


_BOUNDARY_SQL = """
WITH user_events AS (
    SELECT
        round_pk,
        count(*) FILTER (WHERE event_type = 'user_message') AS user_event_count
    FROM timing_events
    GROUP BY round_pk
)
SELECT
    count(*) AS round_count,
    count(*) FILTER (WHERE r.current_user_message_count > 0) AS boundary_rounds,
    count(*) FILTER (WHERE u.user_event_count > 0) AS timing_user_message_rounds,
    count(*) FILTER (
        WHERE (r.current_user_message_count > 0) <> (u.user_event_count > 0)
    ) AS presence_mismatch,
    count(*) FILTER (
        WHERE r.current_user_message_count <> u.user_event_count
    ) AS count_mismatch,
    count(*) FILTER (
        WHERE r.current_user_message_count > 1
    ) AS multi_message_boundary_rounds,
    count(*) FILTER (
        WHERE r.current_input_event_count <>
              r.current_user_message_count + r.current_tool_result_count
    ) AS input_event_identity_mismatch
FROM rounds r
JOIN user_events u USING (round_pk)
"""


_RUN_STATISTICS_SQL = _RUN_ASSIGNMENT_CTE + """
, provider_stats AS (
    SELECT
        provider,
        count(*) AS run_count,
        count(DISTINCT session_id) AS sessions_with_runs,
        count(*) FILTER (WHERE is_strictly_closed) AS strictly_closed_runs,
        avg(round_count) AS mean_rounds_per_run,
        median(round_count) AS median_rounds_per_run,
        quantile_disc(round_count, 0.90) AS p90_rounds_per_run,
        quantile_disc(round_count, 0.95) AS p95_rounds_per_run,
        max(round_count) AS max_rounds_per_run,
        count(*) FILTER (WHERE round_count = 1) AS single_round_runs,
        count(*) FILTER (WHERE round_count >= 2) AS at_least_2_rounds,
        count(*) FILTER (WHERE round_count >= 5) AS at_least_5_rounds,
        count(*) FILTER (WHERE round_count >= 10) AS at_least_10_rounds
    FROM closed_runs
    GROUP BY provider
), overall AS (
    SELECT
        '全部' AS provider,
        count(*) AS run_count,
        count(DISTINCT session_id) AS sessions_with_runs,
        count(*) FILTER (WHERE is_strictly_closed) AS strictly_closed_runs,
        avg(round_count) AS mean_rounds_per_run,
        median(round_count) AS median_rounds_per_run,
        quantile_disc(round_count, 0.90) AS p90_rounds_per_run,
        quantile_disc(round_count, 0.95) AS p95_rounds_per_run,
        max(round_count) AS max_rounds_per_run,
        count(*) FILTER (WHERE round_count = 1) AS single_round_runs,
        count(*) FILTER (WHERE round_count >= 2) AS at_least_2_rounds,
        count(*) FILTER (WHERE round_count >= 5) AS at_least_5_rounds,
        count(*) FILTER (WHERE round_count >= 10) AS at_least_10_rounds
    FROM closed_runs
)
SELECT * FROM overall
UNION ALL
SELECT * FROM provider_stats
ORDER BY provider
"""


_RUNS_PER_SESSION_SQL = """
WITH counts AS (
    SELECT
        provider,
        session_id,
        count(*) FILTER (WHERE current_user_message_count > 0) AS run_count
    FROM rounds
    GROUP BY provider, session_id
), provider_stats AS (
    SELECT
        provider,
        count(*) AS session_count,
        avg(run_count) AS mean_runs_per_session,
        median(run_count) AS median_runs_per_session,
        quantile_disc(run_count, 0.90) AS p90_runs_per_session,
        quantile_disc(run_count, 0.95) AS p95_runs_per_session,
        max(run_count) AS max_runs_per_session,
        count(*) FILTER (WHERE run_count = 0) AS zero_run_sessions
    FROM counts
    GROUP BY provider
), overall AS (
    SELECT
        '全部' AS provider,
        count(*) AS session_count,
        avg(run_count) AS mean_runs_per_session,
        median(run_count) AS median_runs_per_session,
        quantile_disc(run_count, 0.90) AS p90_runs_per_session,
        quantile_disc(run_count, 0.95) AS p95_runs_per_session,
        max(run_count) AS max_runs_per_session,
        count(*) FILTER (WHERE run_count = 0) AS zero_run_sessions
    FROM counts
)
SELECT * FROM overall
UNION ALL
SELECT * FROM provider_stats
ORDER BY provider
"""


_LEADING_ROUNDS_SQL = """
WITH marked AS (
    SELECT
        provider,
        session_id,
        round_index,
        sum(CASE WHEN current_user_message_count > 0 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY session_id ORDER BY round_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS run_ordinal
    FROM rounds
), session_flags AS (
    SELECT
        session_id,
        count(*) FILTER (WHERE run_ordinal = 0) AS leading_round_count,
        max(run_ordinal) AS run_count
    FROM marked
    GROUP BY session_id
)
SELECT
    (SELECT count(*) FROM marked WHERE run_ordinal = 0) AS leading_rounds,
    count(*) FILTER (WHERE leading_round_count > 0) AS sessions_with_leading_rounds,
    count(*) FILTER (WHERE run_count = 0) AS sessions_without_runs
FROM session_flags
"""


_TOOL_CONTINUATION_SQL = _RUN_ASSIGNMENT_CTE + """
, tools AS (
    SELECT round_pk, count(*) AS tool_count
    FROM tool_calls
    GROUP BY round_pk
), sequenced AS (
    SELECT
        a.*,
        coalesce(t.tool_count, 0) AS tool_count,
        lead(a.round_pk) OVER (
            PARTITION BY a.session_id ORDER BY a.round_index
        ) AS next_round_pk,
        lead(a.run_ordinal) OVER (
            PARTITION BY a.session_id ORDER BY a.round_index
        ) AS next_run_ordinal
    FROM assigned a
    LEFT JOIN tools t USING (round_pk)
)
SELECT
    count(*) FILTER (WHERE tool_count > 0) AS tool_rounds,
    count(*) FILTER (
        WHERE tool_count > 0 AND next_round_pk IS NOT NULL
    ) AS tool_with_any_next_round,
    count(*) FILTER (
        WHERE tool_count > 0 AND next_round_pk IS NULL
    ) AS tool_without_next_round,
    count(*) FILTER (
        WHERE tool_count > 0 AND next_round_pk IS NOT NULL
          AND next_run_ordinal = run_ordinal
    ) AS tool_with_same_run_next_round,
    count(*) FILTER (
        WHERE tool_count > 0 AND next_round_pk IS NOT NULL
          AND next_run_ordinal <> run_ordinal
    ) AS tool_with_new_run_next_round,
    count(*) FILTER (WHERE tool_count = 0) AS no_tool_rounds,
    count(*) FILTER (
        WHERE tool_count = 0 AND next_round_pk IS NOT NULL
    ) AS no_tool_with_next_round,
    count(*) FILTER (
        WHERE tool_count = 0 AND next_round_pk IS NULL
    ) AS no_tool_without_next_round
FROM sequenced
"""


_ANCHOR_VALIDATION_SQL = _RUN_ASSIGNMENT_CTE + """
, tools AS (
    SELECT round_pk, count(*) AS tool_count
    FROM tool_calls
    GROUP BY round_pk
), sequenced AS (
    SELECT
        a.*,
        coalesce(t.tool_count, 0) AS tool_count,
        lead(a.prefix_tokens) OVER (
            PARTITION BY a.session_id ORDER BY a.round_index
        ) AS next_prefix_tokens,
        lead(a.run_ordinal) OVER (
            PARTITION BY a.session_id ORDER BY a.round_index
        ) AS next_run_ordinal
    FROM assigned a
    LEFT JOIN tools t USING (round_pk)
), deltas AS (
    SELECT
        provider,
        next_prefix_tokens - input_tokens_total AS delta
    FROM sequenced
    WHERE tool_count > 0
      AND next_prefix_tokens IS NOT NULL
      AND next_run_ordinal = run_ordinal
), provider_stats AS (
    SELECT
        provider,
        count(*) AS observation_count,
        min(delta) AS min_delta,
        quantile_disc(delta, 0.05) AS p05_delta,
        quantile_disc(delta, 0.10) AS p10_delta,
        median(delta) AS median_delta,
        quantile_disc(delta, 0.90) AS p90_delta,
        quantile_disc(delta, 0.95) AS p95_delta,
        max(delta) AS max_delta,
        count(*) FILTER (WHERE delta = 0) AS exact_matches,
        count(*) FILTER (WHERE abs(delta) <= 16) AS within_16,
        count(*) FILTER (WHERE abs(delta) <= 64) AS within_64,
        count(*) FILTER (WHERE abs(delta) <= 256) AS within_256,
        count(*) FILTER (WHERE abs(delta) <= 1024) AS within_1024
    FROM deltas
    GROUP BY provider
), overall AS (
    SELECT
        '全部' AS provider,
        count(*) AS observation_count,
        min(delta) AS min_delta,
        quantile_disc(delta, 0.05) AS p05_delta,
        quantile_disc(delta, 0.10) AS p10_delta,
        median(delta) AS median_delta,
        quantile_disc(delta, 0.90) AS p90_delta,
        quantile_disc(delta, 0.95) AS p95_delta,
        max(delta) AS max_delta,
        count(*) FILTER (WHERE delta = 0) AS exact_matches,
        count(*) FILTER (WHERE abs(delta) <= 16) AS within_16,
        count(*) FILTER (WHERE abs(delta) <= 64) AS within_64,
        count(*) FILTER (WHERE abs(delta) <= 256) AS within_256,
        count(*) FILTER (WHERE abs(delta) <= 1024) AS within_1024
    FROM deltas
)
SELECT * FROM overall
UNION ALL
SELECT * FROM provider_stats
ORDER BY provider
"""


_CONTEXT_FEASIBILITY_SQL = _RUN_ASSIGNMENT_CTE + """
, run_context AS (
    SELECT
        c.provider,
        c.session_id,
        c.run_ordinal,
        c.is_strictly_closed,
        max(a.input_tokens_total) AS max_input_tokens_total
    FROM closed_runs c
    JOIN assigned a USING (provider, session_id, run_ordinal)
    GROUP BY c.provider, c.session_id, c.run_ordinal, c.is_strictly_closed
)
SELECT
    count(*) FILTER (WHERE is_strictly_closed) AS strictly_closed_runs,
    count(*) FILTER (
        WHERE is_strictly_closed AND max_input_tokens_total <= 32768
    ) AS within_32k,
    count(*) FILTER (
        WHERE is_strictly_closed AND max_input_tokens_total <= 65536
    ) AS within_64k,
    count(*) FILTER (
        WHERE is_strictly_closed AND max_input_tokens_total <= 131072
    ) AS within_128k,
    count(*) FILTER (
        WHERE is_strictly_closed AND max_input_tokens_total <= 262144
    ) AS within_256k,
    max(max_input_tokens_total) FILTER (
        WHERE is_strictly_closed
    ) AS max_input_tokens_total
FROM run_context
"""


_TEMPORAL_OVERLAP_SQL = _RUN_ASSIGNMENT_CTE + """
, run_bounds AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        min(t.timestamp) AS start_time,
        max(t.timestamp) AS end_time
    FROM assigned a
    JOIN timing_events t USING (round_pk)
    WHERE t.timestamp IS NOT NULL
    GROUP BY a.provider, a.session_id, a.run_ordinal
), concurrency_at_start AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        1 + count(b.session_id) AS concurrency
    FROM run_bounds a
    LEFT JOIN run_bounds b
      ON a.session_id <> b.session_id
     AND b.start_time <= a.start_time
     AND b.end_time > a.start_time
    GROUP BY a.provider, a.session_id, a.run_ordinal
), overlap_flags AS (
    SELECT
        a.provider,
        a.session_id,
        a.run_ordinal,
        count(b.session_id) > 0 AS has_overlap
    FROM run_bounds a
    LEFT JOIN run_bounds b
      ON a.session_id <> b.session_id
     AND a.start_time < b.end_time
     AND b.start_time < a.end_time
    GROUP BY a.provider, a.session_id, a.run_ordinal
), combined AS (
    SELECT c.*, o.has_overlap
    FROM concurrency_at_start c
    JOIN overlap_flags o USING (provider, session_id, run_ordinal)
), provider_stats AS (
    SELECT
        provider,
        count(*) AS run_count,
        count(*) FILTER (WHERE has_overlap) AS overlap_runs,
        avg(concurrency) AS mean_concurrency,
        median(concurrency) AS median_concurrency,
        quantile_disc(concurrency, 0.90) AS p90_concurrency,
        quantile_disc(concurrency, 0.95) AS p95_concurrency,
        max(concurrency) AS max_concurrency,
        count(*) FILTER (WHERE concurrency = 1) AS concurrency_1,
        count(*) FILTER (WHERE concurrency BETWEEN 2 AND 4) AS concurrency_2_4,
        count(*) FILTER (WHERE concurrency BETWEEN 5 AND 8) AS concurrency_5_8,
        count(*) FILTER (WHERE concurrency >= 9) AS concurrency_9_plus
    FROM combined
    GROUP BY provider
), overall AS (
    SELECT
        '全部' AS provider,
        count(*) AS run_count,
        count(*) FILTER (WHERE has_overlap) AS overlap_runs,
        avg(concurrency) AS mean_concurrency,
        median(concurrency) AS median_concurrency,
        quantile_disc(concurrency, 0.90) AS p90_concurrency,
        quantile_disc(concurrency, 0.95) AS p95_concurrency,
        max(concurrency) AS max_concurrency,
        count(*) FILTER (WHERE concurrency = 1) AS concurrency_1,
        count(*) FILTER (WHERE concurrency BETWEEN 2 AND 4) AS concurrency_2_4,
        count(*) FILTER (WHERE concurrency BETWEEN 5 AND 8) AS concurrency_5_8,
        count(*) FILTER (WHERE concurrency >= 9) AS concurrency_9_plus
    FROM combined
)
SELECT * FROM overall
UNION ALL
SELECT * FROM provider_stats
ORDER BY provider
"""


def main(argv: Sequence[str] | None = None) -> int:
    """执行只读审计并写入 Markdown；可选输出结构化结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="外部 TraceLab DuckDB 路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown 报告输出路径",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="同时将结构化结果输出到标准输出",
    )
    arguments = parser.parse_args(argv)
    analysis = inspect_semantics(arguments.database)
    write_report(analysis, arguments.report)
    if arguments.json:
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
