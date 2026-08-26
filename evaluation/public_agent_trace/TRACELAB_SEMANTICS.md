# TraceLab 无未来信息语义审计

## 结论

TraceLab 可以用真实 `user_message` 边界确定性划分 Agent Run，也可以在工具调用发出后形成一个不依赖未来结果的 LLM-level pending 信号。当前轮 `input_tokens_total` 是合法的已知历史 token 边界，但它不是下一轮物理 prefix 的精确预测器；数据又缺少 token IDs、checkpoint residency 和 lineage，因此四项 gate 均评为 WEAK。

这意味着后续可以设计显式标注假设的离线 trace2flow，但不能把转换结果当成真实 runtime checkpoint 事实，更不能从 tool-call 数量推断 LLM fanout。

## Agent Run 边界有真实事件支持

在 665,453 个 rounds 中，`current_user_message_count > 0` 的 84,538 个边界与 `timing_events.event_type = 'user_message'` 的存在性逐行一致：presence mismatch=0，count mismatch=0。其中 7,662 个 round 同时包含多个 user messages，但仍只启动一个 Agent Run。

冻结分段规则：session 内按 `round_index ASC` 排序；每个包含 user message 的 round 启动新 run，直到下一个此类 round 之前。首个 user-message 边界前的 round 不分配给任何 run；不使用 inactivity threshold。

共有 84,538 个 Agent Runs，覆盖 7,977 个 session。4,367 个 rounds 位于首个边界之前，164 个 session 受影响，另有 81 个 session 完全没有可识别边界。schema 没有 session-end marker，因此每个 session 的最后一个 run 是右删失的；严格闭合 run 数为 76,561。

### Runs/session

| Provider | Sessions | Mean | Median | P90 | P95 | Max | Zero-run |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 8,058 | 10.491 | 1 | 17 | 36 | 1,867 | 81 |
| claude | 5,319 | 8.963 | 1 | 11 | 24 | 1,867 | 10 |
| codex | 2,739 | 13.459 | 2 | 28 | 53 | 1,818 | 71 |

### Rounds/run

| Provider | Runs | Mean | Median | P90 | P95 | Max | 单轮 | >=2 | >=5 | >=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 84,538 | 7.820 | 2 | 17 | 30 | 10,203 | 38.397% | 61.603% | 30.639% | 17.959% |
| claude | 47,675 | 6.404 | 2 | 14 | 24 | 1,105 | 28.380% | 71.620% | 28.789% | 15.312% |
| codex | 36,863 | 9.651 | 1 | 22 | 37 | 10,203 | 51.352% | 48.648% | 33.033% | 21.382% |

## Online 信息边界

### 当前可用

- 当前 session 与 Agent Run 标识
- 已经发生的 rounds
- 当前 round_index
- 当前 input_tokens_total、prefix_tokens、newly_append_tokens、output_tokens
- 当前轮已经发出的 tool calls
- 已经发生的 timing events

### 禁止进入 online 构造

- 下一轮 prefix_tokens
- 下一轮 input_tokens_total
- 下一轮 output_tokens
- 尚未返回的 future tool result 大小
- future timing events

实现使用显式参数构造 `OnlineRoundSnapshot`，类型中不存在 next-round 或 future 字段。事后验证通过另一个函数执行，不能回写 anchor。

## 工具调用只能产生一个弱 pending 信号

有 tool call 的 570,722 个 rounds 中，569,672 个事后观察到下一 LLM round（99.816%），1,050 个没有下一轮（0.184%）。有下一轮的工具 round 中，561,571 个下一轮仍在同一 Agent Run，8,101 个下一轮已由新 user message 启动新 run。

没有 tool call 的 90,364 个 rounds 中，仍有 83,437 个存在下一轮（92.334%）。这些下一轮事实仅用于事后 characterization，不进入 online 决策。

冻结 pending 规则：当前 round 已经发出至少一个 tool call 时，创建一个“等待工具完成后可能继续 LLM”的信号。多个 tool calls 只增加该信号的 tool count，不创建多个 FlowState pending continuations。工具失败、用户终止或 session 截断使该信号并非必然兑现，因此评级为 WEAK。

## Leakage-free anchor

候选规则：在当前 round 的 tool call 已经发出时，使用当前轮 input_tokens_total 作为 Known Historical Anchor；不加入 output_tokens，也不读取下一轮字段。`input_tokens_total` 是当前 prompt 已经确定的输入边界，能够直接映射为逻辑 `token_pos`；加入当前 output 会假设尚未观测到的序列化边界，读取下一轮 prefix 则构成未来信息泄漏。

该 anchor 只表示已执行的历史边界。TraceLab 没有 token IDs、实际 recurrent checkpoint 生成/驻留状态或 lineage，因此它不能单独证明 checkpoint compatibility。

### 下一轮 prefix 的事后验证

主验证 cohort 只包含“当前轮有工具调用，且下一 LLM round 仍属于同一 Agent Run”的记录。signed delta 定义为 `actual_next_prefix_tokens - known_anchor`；验证结果不参与 anchor 规则。

| Provider | N | Min | P05 | P10 | Median | P90 | P95 | Max | Exact | <=16 | <=64 | <=256 | <=1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 561,571 | -982,690 | -2089 | -918 | -10 | -1 | 123 | 144,602 | 0.740% | 43.597% | 48.813% | 65.170% | 90.513% |
| claude | 257,597 | -982,690 | -311 | -37 | -2 | -1 | -1 | 79,354 | 1.548% | 89.902% | 90.175% | 93.899% | 96.746% |
| codex | 303,974 | -394,601 | -2966 | -1363 | -278 | 93 | 349 | 144,602 | 0.055% | 4.358% | 13.762% | 40.825% | 85.230% |

整体 exact match 仅 0.740%，但绝对误差在 1024 tokens 内的比例为 90.513%。Claude 与 Codex 的分布差异明显，说明当前输入边界是 leakage-free 的历史 anchor，却不能被解释成下一轮物理 prefix 的无偏估计。

## 完整 Agent Run 的 context feasibility

只统计存在后续 user-message 边界的 76,561 个严格闭合 runs；右删失的 session 最后一个 run 不计入“完整 run”。不裁剪、不缩放 token。

| Context 上限 | Runs | 比例 |
|---:|---:|---:|
| 32K | 3,196 | 4.174% |
| 64K | 10,628 | 13.882% |
| 128K | 27,888 | 36.426% |
| 256K | 55,413 | 72.378% |

完整 run 的最大 `input_tokens_total` 为 999,944。本审计不据此选择 context cutoff。

## Run-level temporal overlap

全部 84,538 个 runs 均有 timing bounds；77,885 个与另一 session 的 run 区间重叠，占 92.130%。在每个 run 的开始时点计数，concurrency mean=4.998、median=5、P90=8、P95=10、max=33。

分桶：concurrency=1 有 7,749 个 runs，2–4 有 32,061 个，5–8 有 36,702 个，>=9 有 8,026 个。

时间戳没有 timezone 或跨 provider clock provenance；区间重叠只能证明数据中观察到的并发形态，不能保证可按同一绝对时钟精确回放。

## Gate

| Gate | 评级 | 原因 |
|---|---|---|
| Agent-run segmentation | WEAK | user-message 边界与 timing events 全量一致，且排序确定；但首段缺边界、最后 run 右删失，schema 无 session-end marker。 |
| Leakage-free pending continuation | WEAK | 已发出的 tool call 是当前可知信号，且绝大多数事后存在下一轮；但 continuation 并非必然发生，多个 tool calls 也不能解释为 LLM fanout。 |
| Leakage-free anchor reconstruction | WEAK | 当前输入边界是无泄漏且可映射到 token_pos 的历史 anchor；但缺少 token IDs、lineage 与实际 checkpoint residency，且与下一轮物理 prefix 通常不精确相等。 |
| Direct FlowState policy evaluation | WEAK | pending 与 anchor 均达到 WEAK，可进入严格标注假设的离线 trace2flow；但 candidate/residency/compatibility 仍需外部建模，不能作为真实 runtime allocator 证据。 |

## 推荐用途与限制

TraceLab 适合提供真实世界多轮、长上下文、prefix reuse、tool-gap 和 temporal overlap 的 workload evidence。后续若进入 trace2flow，应把当前输入边界规则预先冻结，把工具 round 映射为至多一个 pending，并将 checkpoint candidate、lineage 与 residency 明确标为建模假设。

不得用下一轮 token 或 timing 字段决定当前 anchor，不得把多个工具调用当作分支，不得从 prefix 数值反推显式 workflow DAG，也不得把该数据集上的离线结果表述为 SGLang runtime correctness。
