# Agentic 数据集轻量级适用性审计：零下载阶段

审计日期：2026-10-13。此阶段只查看 Hugging Face 官方数据卡、Viewer 页面元数据与可见 Schema；未下载数据文件、未发起完整仓库下载，也未运行轨迹分析。由于本环境对外部 HTTP 请求受限，Hugging Face API、Dataset Server 的 Parquet 元数据与按行查询不可用；数据卡页面可读，但原始文件逐文件大小和样本内容无法取得。因此没有进入数据文件小样本下载阶段。

## FreeInference Agentic Trace

官方卡片称数据集覆盖 2026-05-17 至 2026-09-05，含 12,002 sessions、1,186,582 LLM requests、1,213,347 tool calls；主数据是按天 JSONL.zst，每行一个 session，requests 按会话顺序保存。Tool call 可嵌套完整 subagent session。request 记录 `start_ms`、`end_ms`、`step_trigger`、`link_reason`、`mutation`、工具调用和可选的 subagent。由此可识别同一 session 内的调用顺序和网关请求区间，并能找到显式子代理委派结构；但卡片不证明父子请求在服务端并发，也没有原生 Ready/queued/join 事件。

数据卡明确说明不发布 message text。输入只以 `o200k_base` 编码后的 16-token block IDs 表示，输入渲染顺序为 tool definitions、system prompt、messages，末尾不完整 block 被丢弃；此 tokenization 不是具体 serving provider 的 tokenizer。它适合 prefix/block locality 统计，不足以恢复目标模型完整 token 输入或证明 token-exact compatibility。卡片中的 session/request 时间戳可说明服务请求实际发生过，但不是 checkpoint 物化、recurrent slot identity 或 allocation barrier 证据。

| 检查项 | 卡片/Viewer 证据 | 审计判断 |
|---|---|---|
| 数据卡、字段 | [官方数据卡](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace)，Viewer 可见 `session_id`、`start_ms`、`end_ms`、requests 嵌套 Schema | 已核对 |
| 文件目录与逐文件大小 | 卡片给出 `release/<week>/traces/<day>.jsonl.zst` 和 `processed/` Parquet 组织；总文件大小 8.93 GB；逐文件清单/API 未能读取 | 部分可见；逐文件大小 UNKNOWN |
| 完整消息文本 | 明确不发布 | 无法重建完整请求 |
| Parent/Subagent | tool call 的 `subagent` 内嵌完整子 session；父子嵌套证据可读 | 有结构线索；并行/Ready 语义未证实 |
| 请求开始/结束 | request `start_ms`、`end_ms`；session `start_ms`、`end_ms` | 有网关时间；不等于调度生命周期 |
| Prefix/Session/Workflow | session_id、step/link/mutation、block_ids；block IDs 按周局部 | 有 block-prefix 与 session 关联；无原始 token IDs |
| 当时已物化输入 | request timestamp 与顺序存在 | 只证明请求曾被发送；不能恢复完整输入或状态快照 |
| 远程 Parquet Schema/统计 | 卡片和 Viewer 显示配置/列概览；Dataset Server API 请求不可达，Viewer split 也报服务端错误 | 未能取得嵌套 Parquet Schema/精确统计 |

## GATC v6

官方卡片说明 HF 仓库是 1% 样本，1,189 sessions；表格包括 `trajectories`、`messages`、`sft_steps` 等。原始记录保留消息内容、工具调用/结果、按序 steps、`subagent` 类型、step 时间 `ts` 和 `duration_ms`；`session_kind` 能将一部分 session 标为 `sub-agent`。完整文件中长消息仍可能有数据自身的截断标记，且文本经过去标识化。

Schema 使用 `trajectory_id`、`actor_id`、`repo_id`，但卡片没有列出可将某个父 session 的 subagent tool call 与另一个 trajectory_id 连接起来的稳定 parent/child 外键。`started_at` 是 session 级时间，step 级 `ts`/`duration_ms` 不等价于 LLM 请求起止时间，也没有 Ready/dispatch/join 事件。数据提供消息文本但未提供实际 token IDs；模型名被映射为 capability tier 或匿名占位，工具与文本也可能被截断/脱敏，不能据此声称已精确复原目标模型请求。

| 检查项 | 卡片/Viewer 证据 | 审计判断 |
|---|---|---|
| 数据卡、字段 | [官方数据卡](https://huggingface.co/datasets/GOAT-AI/gatcv6)，Viewer 列出 messages 配置；Schema 明示 trajectories/messages | 已核对 |
| 文件目录与逐文件大小 | 总文件大小 66.1 MB；官方说明每表 JSONL、压缩格式；Viewer/API 未给出逐文件大小 | 总量已核对；逐文件大小 UNKNOWN |
| 完整消息文本 | messages 中有 user/assistant/tool 内容，trajectories 有 step text/args/result；部分长内容或 withheld 内容被截断/移除 | 可读文本但非完整无损 |
| Parent/Subagent | 存在 `type=subagent`、`session_kind=sub-agent`；没有文档化 parent trajectory 外键 | 委派可见，跨记录父子关联不完整 |
| 请求开始/结束 | `started_at`/`duration_s` 为 session 级；step `ts`/`duration_ms` | 没有可靠的每个 LLM request 起止时间 |
| Prefix/Session/Workflow | trajectory_id、actor/repo 伪名，步骤有序 | 有 session/project 分组；无精确执行 fork/prefix 关系 |
| 当时已物化输入 | 历史步骤、工具调用与顺序记录 | 可区分已记录步骤与后续步骤；无法证明某个未来请求当时已 Ready |
| 远程 Schema/统计 | 官方卡片公开表 Schema 和总体计数；样本行 Viewer 可见但长值截断；API 行查询未成功 | 零下载 Schema 可读；案例级查询受限 |

## 样本与网络边界

没有下载 `.jsonl.zst`、Parquet 或 GATC 文件，也没有把 Viewer 截断的行当作完整样本。FreeInference Viewer 的 week split 报 `ClientConnectionError`；本地到 `huggingface.co/api` 的 HTTPS 连接被运行环境拒绝；GATC Viewer 虽能显示部分 messages 行，但 trajectories 的直接 Viewer 路由无法通过当前浏览工具访问。遵照 100 MB 限制，不尝试替代下载方式或扩大读取范围。样本统计与候选 Session 留空/标记未审计，避免把页面示例误报成系统抽样结果。

官方来源：

- https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace
- https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/viewer/traces/week_2026_05_17
- https://huggingface.co/datasets/GOAT-AI/gatcv6
