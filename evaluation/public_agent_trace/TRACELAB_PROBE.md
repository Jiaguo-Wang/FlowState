# TraceLab Schema 与 Session 可行性探查

## 技术摘要

- 数据库包含 665,453 个 LLM rounds 和 8,058 个 session；7,471 个 session 为多轮，占 92.715%。
- `round_index` 在全部 session 内唯一，可提供确定性执行顺序；真实时间来自规范化的 `timing_events.timestamp`，而不是 `rounds` 自身。
- 657,575 个 round 的 `prefix_tokens > 0`，占 98.816%；四个 token 字段均无空值，并且每行严格满足 `input_tokens_total = prefix_tokens + newly_append_tokens`。
- 未发现 round/workflow 级 parent、root、continuation 或 branch 字段；数据支持线性多轮 session，但不支持直接恢复显式 workflow DAG。
- 跨 session 时间区间确有重叠，最大同时活动 session 数为 51；但时间戳无 timezone 元数据，且存在少量 round 时间倒序，精确跨 provider 回放需要保留此限制。

## 数据库只有四张规范化表

| Schema | Table | 类型 | 行数 |
|---|---|---|---:|
| main | rounds | BASE TABLE | 665,453 |
| main | timing_events | BASE TABLE | 2,688,829 |
| main | tool_calls | BASE TABLE | 743,819 |
| main | trace_source | BASE TABLE | 1 |

`rounds` 是 LLM round 主表：它具备 session、round sequence、model 与完整 token accounting；另外两张明细表通过 `round_pk` 关联 timing 和 tools。
完整性检查中，timing orphan=0，tool orphan=0，没有 timing event 的 round=0。

## 完整 schema

### `rounds`（665,453 行）

| 序号 | Column | DuckDB 类型 | Nullable |
|---:|---|---|---|
| 1 | round_pk | BIGINT | YES |
| 2 | ingest_seq | BIGINT | YES |
| 3 | provider | VARCHAR | YES |
| 4 | project | VARCHAR | YES |
| 5 | session_id | VARCHAR | YES |
| 6 | round_index | BIGINT | YES |
| 7 | round_id | VARCHAR | YES |
| 8 | model | VARCHAR | YES |
| 9 | input_tokens_total | BIGINT | YES |
| 10 | prefix_tokens | BIGINT | YES |
| 11 | newly_append_tokens | BIGINT | YES |
| 12 | claude_uncached_input_tokens | BIGINT | YES |
| 13 | claude_cache_creation_input_tokens | BIGINT | YES |
| 14 | claude_cache_read_input_tokens | BIGINT | YES |
| 15 | output_tokens | BIGINT | YES |
| 16 | current_input_event_count | BIGINT | YES |
| 17 | current_user_message_count | BIGINT | YES |
| 18 | current_tool_result_count | BIGINT | YES |
| 19 | current_user_message_chars | BIGINT | YES |
| 20 | current_tool_result_chars | BIGINT | YES |
| 21 | current_input_chars | BIGINT | YES |
| 22 | first_input_event_type | VARCHAR | YES |
| 23 | user | VARCHAR | YES |
| 24 | store | VARCHAR | YES |
| 25 | trace_key | VARCHAR | YES |
| 26 | turn_id | UUID | YES |
| 27 | reasoning_output_tokens | BIGINT | YES |

### `timing_events`（2,688,829 行）

| 序号 | Column | DuckDB 类型 | Nullable |
|---:|---|---|---|
| 1 | round_pk | BIGINT | YES |
| 2 | event_index | BIGINT | YES |
| 3 | event_type | VARCHAR | YES |
| 4 | source | VARCHAR | YES |
| 5 | timestamp | TIMESTAMP | YES |
| 6 | tool_call_id | VARCHAR | YES |
| 7 | tool_index | BIGINT | YES |
| 8 | tool_name | VARCHAR | YES |
| 9 | is_error | BOOLEAN | YES |
| 10 | result_chars | BIGINT | YES |
| 11 | content_chars | BIGINT | YES |

### `tool_calls`（743,819 行）

| 序号 | Column | DuckDB 类型 | Nullable |
|---:|---|---|---|
| 1 | round_pk | BIGINT | YES |
| 2 | tool_index | BIGINT | YES |
| 3 | tool_name | VARCHAR | YES |
| 4 | tool_call_id | VARCHAR | YES |
| 5 | emitted_at | TIMESTAMP | YES |
| 6 | result_at | TIMESTAMP | YES |
| 7 | tool_wall_latency_ms | BIGINT | YES |
| 8 | tool_internal_latency_ms | BIGINT | YES |
| 9 | is_error | BOOLEAN | YES |
| 10 | input_chars | BIGINT | YES |
| 11 | result_chars | BIGINT | YES |
| 12 | continuation_of_tool_call_id | VARCHAR | YES |
| 13 | command_status | VARCHAR | YES |
| 14 | command_exit_code | HUGEINT | YES |
| 15 | executables | VARCHAR[] | YES |
| 16 | executable_parse_status | VARCHAR | YES |
| 17 | executable_parse_reason | VARCHAR | YES |
| 18 | command_skeleton | VARCHAR | YES |

### `trace_source`（1 行）

| 序号 | Column | DuckDB 类型 | Nullable |
|---:|---|---|---|
| 1 | path | VARCHAR | YES |

## 请求字段的真实位置

| 逻辑字段 | 状态 | 真实位置或说明 |
|---|---|---|
| provider | available | rounds.provider |
| session id | available | rounds.session_id |
| round id | available | rounds.round_id |
| sequence id | available | rounds.round_index |
| timestamp / timing | rounds 中 unavailable | 可从规范化子表 timing_events.timestamp 获取 |
| model | available | rounds.model |
| input_tokens_total | available | rounds.input_tokens_total |
| prefix_tokens | available | rounds.prefix_tokens |
| newly_append_tokens | available | rounds.newly_append_tokens |
| output_tokens | available | rounds.output_tokens |
| timing_events | rounds 中 unavailable | 规范化子表 timing_events |
| tools | 名为 tools 的字段 unavailable | 相关数据位于规范化子表 tool_calls |

`rounds` 没有 timestamp、`timing_events` 或 `tools` 嵌套 column。前者必须从 `timing_events` 聚合；后两者分别是规范化子表 `timing_events` 与 `tool_calls`。

## 五条真实 round 记录

以下记录按 `round_pk` 升序取前五条；`time`、timing event 类型与 tool 名称均通过 `round_pk` 从子表只读聚合。

| round_pk | provider | session_id | round_index | round_id | time | model | input | prefix | append | output | timing events | tools |
|---:|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|
| 1 | claude | claude:f4b60bc0-eff2-74b3-a2b9-18e00467ed32 | 0 | round_e14eb886e31cbb79 | 2026-05-31 14:18:50.524000 | claude-opus-4-7 | 35522 | 0 | 35522 | 30 | text, user_message (2) | 无 (0) |
| 2 | claude | claude:a2ec2d48-b5c7-0f1e-587f-3bc7b8dab1bb | 0 | round_06a121f55fc1e41a | 2026-06-02 23:55:34.568000 | claude-opus-4-8 | 31186 | 0 | 31186 | 318 | reasoning, text, tool_call, user_message (5) | Bash, Read (2) |
| 3 | claude | claude:a2ec2d48-b5c7-0f1e-587f-3bc7b8dab1bb | 1 | round_6644210440d01241 | 2026-06-02 23:55:38.957000 | claude-opus-4-8 | 34545 | 29695 | 4850 | 459 | reasoning, text, tool_call, tool_result (5) | Bash (1) |
| 4 | claude | claude:a2ec2d48-b5c7-0f1e-587f-3bc7b8dab1bb | 2 | round_5f58396b1cf64622 | 2026-06-02 23:55:49.728000 | claude-opus-4-8 | 35130 | 34543 | 587 | 52 | reasoning, tool_result (2) | 无 (0) |
| 5 | claude | claude:a2ec2d48-b5c7-0f1e-587f-3bc7b8dab1bb | 3 | round_82665e8df5ebde57 | 2026-06-02 23:55:51.894000 | claude-opus-4-8 | 35153 | 35128 | 25 | 228 | reasoning, tool_call, user_message (4) | Bash (1) |

## Session 与 token 分布支持大规模多轮分析

### Rounds/session

| Sessions | Multi-round | Mean | Median | P90 | P95 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 8,058 | 7,471 | 82.583 | 16 | 146 | 305 | 21,351 |

### 单轮 token

| Metric | Median | P90 | P95 | Max | Null | Negative |
|---|---:|---:|---:|---:|---:|---:|
| input_tokens_total | 132092 | 338664 | 519461 | 999,944 | 0 | 0 |
| prefix_tokens | 126336 | 326527 | 500798 | 999,469 | 0 | 0 |
| newly_append_tokens | 1045 | 6963 | 14365 | 999,001 | 0 | 0 |
| output_tokens | 249 | 1332 | 2281 | 64,000 | 0 | 0 |

Prefix reuse：657,575/665,453 rounds（98.816%）的 `prefix_tokens > 0`。

## `round_index` 是稳定顺序，timestamp 是辅助时间轴

全部 8,058 个 session 的 `round_index` 均唯一；7,991 个从 0 开始，7,987 个索引连续。排序规则固定为 session 内 `round_index ASC`，并用 `round_pk` 作为防御性最终 tie-break；实际没有 tie。

以每个 round 的最早 timing timestamp 检查相邻轮次时，47 / 657,395 对出现时间倒序，另有 3,175 对时间相同。因此不能用 timestamp 取代 `round_index` 作为 session 内权威顺序。

事件明细的 `event_index` 也不是严格时间排序：148,480 / 2,023,376 个相邻 event-index 对发生 timestamp 倒序。分析工具事件时应直接按 timestamp 排序，同时用 `event_index` 做稳定 tie-break。

## 相邻 rounds 通常呈现上下文增长，但不是 DAG 证据

在 657,395 个相邻 round 对中，后续 `input_tokens_total` 不小于前一轮的比例为 98.068%，中位 input 增量为 743 tokens；仍有 12,701 对发生 context 下降，可能来自 compaction、cache 策略或 trace 边界。

`prefix_tokens` 表示当前 round 的物理复用长度，不等同于显式 parent。数据没有 token IDs，也没有 ancestry path；因此本报告不把数值上的 prefix 关系解释为 branching。

## 未发现显式 branching；工具时序大部分可恢复

DAG-like 字段扫描只找到：tool_calls.continuation_of_tool_call_id。其中 `tool_calls.continuation_of_tool_call_id` 描述工具调用续接，不是 LLM round 或 workflow parent。显式 workflow DAG 字段集合为空。

`tool_calls` 有 743,819 行，其中 769 行缺失 `result_at`；其余 99.897% 可由 `emitted_at`、`result_at` 和一致的 `tool_wall_latency_ms` 计算等待时间。所有 timing event 均有 timestamp，但鉴于 event-index 倒序和少量缺失 tool result，`LLM → tool → 下一次 LLM` 只能在大部分记录上可靠恢复，不能声称全量无歧义。

## 五个确定性完整短 session

选择规则：每个 provider 内筛选 round_index 从 0 开始、连续且总轮数为 2 至 5 的 session；按 session_id 字典序选择 Claude 前 3 个和 Codex 前 2 个。这里“完整”仅表示下载数据内从 0 开始且 `round_index` 连续；schema 没有显式 session-end marker，无法证明上游绝对完整。

### Session `claude:0022b369-4fa6-1772-4d11-8681e3e6b1d3`

| Round | round_id | time | input | prefix | append | output | tool_count |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | round_e7c2431158c53b12 | 2026-07-01 04:31:57.800000 | 17813 | 14039 | 3774 | 256 | 1 |
| 2 | round_08a1a0098122e3b7 | 2026-07-01 04:32:18.522000 | 19219 | 16193 | 3026 | 147 | 1 |
| 3 | round_66a55fe035ac4606 | 2026-07-01 04:32:45.507000 | 21481 | 19217 | 2264 | 1 | 1 |
| 4 | round_79a7b403d6be8ece | 2026-07-01 04:33:13.834000 | 23779 | 21479 | 2300 | 1514 | 1 |

### Session `claude:003e62b1-78bf-737b-7a25-009de6df1306`

| Round | round_id | time | input | prefix | append | output | tool_count |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | round_d2d228846f6f98ba | 2026-06-04 06:31:32.222000 | 15583 | 13060 | 2523 | 229 | 3 |
| 2 | round_25c6f9de03fa98a1 | 2026-06-04 06:31:35.279000 | 20783 | 15580 | 5203 | 119 | 1 |
| 3 | round_174a6bc3cc860004 | 2026-06-04 06:31:41.331000 | 21273 | 20782 | 491 | 1025 | 0 |

### Session `claude:00500900-ea20-09d8-f153-0848b74b1312`

| Round | round_id | time | input | prefix | append | output | tool_count |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | round_8809c67f5b507716 | 2026-05-29 03:39:52.285000 | 15713 | 12978 | 2735 | 150 | 2 |
| 2 | round_e8ac8280880415d7 | 2026-05-29 03:40:17.290000 | 17614 | 15710 | 1904 | 256 | 2 |
| 3 | round_8b0fe823e7b2ab7d | 2026-05-29 03:43:10.059000 | 18613 | 17613 | 1000 | 147 | 2 |
| 4 | round_d3970f2e9b5867ff | 2026-05-29 03:51:29.810000 | 21886 | 12978 | 8908 | 1 | 2 |

### Session `codex:00031494-b05c-debc-285b-9d67fcbbc308`

| Round | round_id | time | input | prefix | append | output | tool_count |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | e1b03649-4fa4-6aea-bc96-4c53f27d860c:0 | 2026-06-30 03:51:00.459000 | 21764 | 7040 | 14724 | 95 | 0 |
| 2 | a535de09-b405-45c0-d0a4-ae876e7bf558:0 | 2026-06-30 03:51:08.810000 | 23442 | 21376 | 2066 | 29 | 0 |
| 3 | 856cfc4c-0590-d3db-86e7-981d1e2a1664:0 | 2026-06-30 03:51:17.456000 | 24261 | 22912 | 1349 | 118 | 0 |
| 4 | 29a934b2-8570-7835-49cb-ed94489f6a46:0 | 2026-06-30 03:51:33.186000 | 25036 | 23936 | 1100 | 29 | 0 |

### Session `codex:0267fd53-508f-a0ab-064c-af3b29710696`

| Round | round_id | time | input | prefix | append | output | tool_count |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | turn_63bbc761d2fa6f09:0 | 2026-02-01 08:23:20.972000 | 8259 | 6656 | 1603 | 89 | 1 |
| 2 | turn_63bbc761d2fa6f09:1 | 2026-02-01 08:23:27.705000 | 8595 | 8192 | 403 | 72 | 1 |
| 3 | turn_63bbc761d2fa6f09:2 | 2026-02-01 08:23:30.321000 | 11181 | 8576 | 2605 | 72 | 1 |
| 4 | turn_63bbc761d2fa6f09:3 | 2026-02-01 08:23:34.924000 | 13942 | 11136 | 2806 | 421 | 0 |
| 5 | turn_63bbc761d2fa6f09:4 | 2026-02-01 08:41:04.579000 | 14351 | 8192 | 6159 | 355 | 0 |

## FlowState trace feasibility

| 能力 | 评级 | 证据与边界 |
|---|---|---|
| Multi-turn workflow | PASS | 7,471 个多轮 session，且全部 session 可由唯一 `round_index` 稳定排序。 |
| Prefix/anchor reconstruction | WEAK | 每轮 H/输入长度可由完整 token accounting 重建，但没有 token IDs、logical anchor ID 或 lineage，无法判断跨 checkpoint compatibility。 |
| Temporal multi-workflow replay | WEAK | 全部 round 有 timing timestamp，观察到最大 51 个 session 重叠；但 timestamp 无 timezone/clock provenance，且有少量 session 内时间倒序。 |
| Explicit workflow DAG | FAIL | 未发现 round/workflow parent、root、continuation 或 branch 字段；不得从 prefix 数值推断分支。 |

## 限制、稳健性检查与下一步

- `round_id` 不是全局唯一键；`round_pk` 才是完整唯一的 round grain。
- `session_id` 在 provider 间冲突数为 0；横跨多个 project 标签的 session 数为 1。因此 session 计数使用 `session_id`，不把 project 当 session key。
- 71 个 session 的 round index 不连续，67 个不从 0 开始；全量分析应保留缺口标记，不能自动补轮次。
- 时间字段是无 timezone 的 DuckDB `TIMESTAMP`。正式跨 session replay 前应确认 TraceLab 发布说明中的时区与时钟归一化语义。
- 下一步若实现 trace2flow，应先限定为线性 session snapshot，并把 DAG/branch 相关能力标记为 unavailable；本 probe 不授权实现该转换。

进一步需要确认：`prefix_tokens` 的 provider-specific cache 统计口径、session 是否具有未发布的结束标记，以及跨 provider timestamp 是否共用同一时钟基准。
