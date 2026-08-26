# TraceLab 到 FlowState 的无未来泄漏离线 Workload

## 技术摘要

本构造只使用 76,561 个严格闭合 Agent Runs，并从真实 tool-call 观测时点生成 27 个 snapshot。所有 online anchor 均严格等于当前轮 `input_tokens_total`，future-prefix leakage violations=0。

workload 包含 3,736 个逻辑 checkpoint 实例和 71 个 LLM-level pending 实例。它可以供现有 FlowState 数据模型做离线方法比较，但评级为 WEAK：TraceLab 不提供 token IDs、runtime checkpoint residency 或 workflow DAG。

## 冻结语义没有引入未来字段

- Agent Run 仍由 `current_user_message_count > 0` 启动。
- tool call round 的 anchor 只取当前 `input_tokens_total`。
- 一个 round 无论发出多少 tool calls，最多生成一个 continuation。
- lineage 仅由真实 `round_index` 执行顺序形成线性 tuple prefix。
- 当前 `prefix_tokens` 只保存在 characterization metadata，不参与 anchor 或 planning target。
- round 完成可知性只依赖 snapshot 前已观察到的 `tool_call`/`usage_report`，或同 run 下一轮已经开始；查询不会把未来 timing event 注入 candidate。
- `recurrent_resident=True`、`fa_resident=True` 和 `resident_fa_frontier=anchor` 是逻辑 snapshot 可消费性假设，不是从 TraceLab 推断出的 GPU residency truth。

## 严格闭合 cohort 排除了右删失状态

| 项目 | 数量 |
|---|---:|
| 原始 rounds | 665,453 |
| 首个 user-message 前孤立 rounds | 4,367 |
| 可分配 Agent Runs | 84,538 |
| 严格闭合 Agent Runs | 76,561 |
| 右删失 Agent Runs | 7,977 |
| 严格闭合 rounds | 538,771 |
| 右删失 rounds | 122,315 |
| 缺少完成事件的严格闭合 rounds | 0 |

最后一个 Agent Run 因没有 session-end marker 被排除；首个真实 user-message 边界之前的记录不被强行归入 run。过滤只依赖边界完整性，不依赖任何 policy 或 recovery cost。

## 三种 scale 来自已观察 concurrency 分布

固定 seed 为 `20260826`。Small、Medium、Large 分别对应同 provider 内 2–4、5–8、>=9 个 trace-observed active runs；每个非空 `scale × provider × context bucket` stratum 选择一个固定 SHA-256 键最小的多轮 trigger run 时点。

最终 Small=10、Medium=9、Large=8，合计 27 个 snapshots。缺失 strata=3，均因源数据中不存在对应 concurrency/context/provider 组合，而不是人为补齐或按结果改样。

完整 run 最大 context 只用于客观采样分层，不进入 snapshot 的 anchor、pending、checkpoint value 或后续 policy 输入。跨 provider 不混合时间域；并发只称为 trace-observed concurrency，不声称精确生产 arrival replay。

## Snapshot 规模与 anchor 分布

| 指标 | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Active workflows | 5.926 | 5 | 10 | 11 | 13 |
| Candidates | 138.370 | 38 | 455 | 626 | 1034 |
| Pending | 2.630 | 2 | 6 | 6 | 7 |
| Anchor depth tokens | 186820.394 | 131197 | 416685 | 719154 | 809093 |

共有 123 个不同 Agent Runs 出现在 sampled snapshots 中。candidate 只能来自 `known_at_time <= snapshot.observed_at` 的完成 rounds；pending 只能来自最新已开始 round 中截至 snapshot 已经观察到的 tool calls。

## Context buckets 保留全部长上下文 cohort

| Bucket | Runs | Rounds | Pending signals | Logical candidates |
|---|---:|---:|---:|---:|
| <=32K | 3,196 | 8,113 | 5,106 | 8,113 |
| 32K-64K | 7,432 | 31,111 | 24,401 | 31,111 |
| 64K-128K | 17,260 | 105,274 | 89,618 | 105,274 |
| 128K-256K | 27,525 | 293,023 | 254,670 | 293,023 |
| >256K | 21,148 | 101,250 | 81,774 | 101,250 |

`>256K` bucket 没有被删除、裁剪或 downscale。表中的 candidate 是“每个完成 round 一个逻辑 recurrent checkpoint”的 source-cohort 计数，不是 GPU 上实际存在的状态。

## Budget 只按 retention ratio 生成元数据

| Ratio | K mean | K median | K P90 | K P95 | K max |
|---:|---:|---:|---:|---:|---:|
| 25% | 34.259 | 9 | 113 | 156 | 258 |
| 50% | 68.963 | 19 | 227 | 313 | 517 |
| 75% | 103.407 | 28 | 341 | 469 | 775 |

每个 snapshot 对 N 个 candidates 使用 `max(1, floor(N × ratio))`。本步骤只保存 K，不调用 FlowState、KVFlow、Marconi 或 Oracle。

## 完整性 Gate 全部通过

- future-field leakage violations：0
- snapshot 之后完成的 checkpoint：0
- 无已发出 tool call 的 pending：0
- 同一 round 生成多个 LLM-level pending：0
- 非线性或跨 workflow lineage：0
- anchor 与当前轮 input 不一致：0
- future prefix 影响 online mapping：0
- runtime residency inference：0

## 限制与下一步边界

该 workload 的 workflow identity、线性 lineage 和 logical checkpoint catalog 是 TraceLab 事实上的确定性离线映射，不是显式 DAG 或 SGLang runtime observation。当前 prefix metadata 可用于描述真实 provider cache reuse，但不能改写 FlowState planning target。

下一步若进行离线 policy comparison，必须读取已冻结 JSON，不得重采样；应把结果描述为 trace-derived logical snapshot comparison，而不是生产 arrival replay 或 GPU correctness。
