# Agentic 数据集轻量级适用性审计：零下载阶段（2026-10-13）与网络恢复后的补审计（2026-10-15）

## 第一阶段（2026-10-13，零下载）

此阶段只查看 Hugging Face 官方数据卡、Viewer 页面元数据与可见 Schema；未下载数据文件、未发起完整仓库下载，也未运行轨迹分析。当时环境对外部 HTTP 请求受限，Hugging Face API、Dataset Server 的 Parquet 元数据与按行查询不可用；数据卡页面可读，但原始文件逐文件大小和样本内容无法取得。

### FreeInference Agentic Trace

官方卡片称数据集覆盖 2026-05-17 至 2026-09-05，含 12,002 sessions、1,186,582 LLM requests、1,213,347 tool calls；主数据是按天 JSONL.zst，每行一个 session，requests 按会话顺序保存。Tool call 可嵌套完整 subagent session。request 记录 `start_ms`、`end_ms`、`step_trigger`、`link_reason`、`mutation`、工具调用和可选的 subagent。由此可识别同一 session 内的调用顺序和网关请求区间，并能找到显式子代理委派结构；但卡片不证明父子请求在服务端并发，也没有原生 Ready/queued/join 事件。

数据卡明确说明不发布 message text。输入只以 `o200k_base` 编码后的 16-token block IDs 表示，输入渲染顺序为 tool definitions、system prompt、messages，末尾不完整 block 被丢弃；此 tokenization 不是具体 serving provider 的 tokenizer。它适合 prefix/block locality 统计，不足以恢复目标模型完整 token 输入或证明 token-exact compatibility。卡片中的 session/request 时间戳可说明服务请求实际发生过，但不是 checkpoint 物化、recurrent slot identity 或 allocation barrier 证据。

| 检查项 | 卡片/Viewer 证据 | 审计判断 |
|---|---|---|
| 数据卡、字段 | [官方数据卡](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace)，Viewer 可见 `session_id`、`start_ms`、`end_ms`、requests 嵌套 Schema | 已核对 |
| 完整消息文本 | 明确不发布 | 无法重建完整请求 |
| Parent/Subagent | tool call 的 `subagent` 内嵌完整子 session；父子嵌套证据可读 | 有结构线索；并行/Ready 语义未证实（2026-10-15 已补案例核验，见下） |
| 请求开始/结束 | request `start_ms`、`end_ms`；session `start_ms`、`end_ms` | 有网关时间；不等于调度生命周期 |
| Prefix/Session/Workflow | session_id、step/link/mutation、block_ids；block IDs 按周局部 | 有 block-prefix 与 session 关联；无原始 token IDs |
| 当时已物化输入 | request timestamp 与顺序存在 | 只证明请求曾被发送；不能恢复完整输入或状态快照 |

### GATC v6

官方卡片说明 HF 仓库是 1% 样本，1,189 sessions；表格包括 `trajectories`、`messages`、`sft_steps` 等。原始记录保留消息内容、工具调用/结果、按序 steps、`subagent` 类型、step 时间 `ts` 和 `duration_ms`；`session_kind` 能将一部分 session 标为 `sub-agent`。完整文件中长消息仍可能有数据自身的截断标记，且文本经过去标识化。

Schema 使用 `trajectory_id`、`actor_id`、`repo_id`，但卡片没有列出可将某个父 session 的 subagent tool call 与另一个 trajectory_id 连接起来的稳定 parent/child 外键。`started_at` 是 session 级时间，step 级 `ts`/`duration_ms` 不等价于 LLM 请求起止时间，也没有 Ready/dispatch/join 事件。数据提供消息文本但未提供实际 token IDs；模型名被映射为 capability tier 或匿名占位，工具与文本也可能被截断/脱敏，不能据此声称已精确复原目标模型请求。

| 检查项 | 卡片/Viewer 证据 | 审计判断 |
|---|---|---|
| 数据卡、字段 | [官方数据卡](https://huggingface.co/datasets/GOAT-AI/gatcv6)，Viewer 列出 messages 配置；Schema 明示 trajectories/messages | 已核对 |
| 完整消息文本 | messages 中有 user/assistant/tool 内容，trajectories 有 step text/args/result；部分长内容或 withheld 内容被截断/移除 | 可读文本但非完整无损 |
| Parent/Subagent | 存在 `type=subagent`、`session_kind=sub-agent`；没有文档化 parent trajectory 外键 | 委派可见，跨记录父子关联不完整 |
| 请求开始/结束 | `started_at`/`duration_s` 为 session 级；step `ts`/`duration_ms` | 没有可靠的每个 LLM request 起止时间 |
| Prefix/Session/Workflow | trajectory_id、actor/repo 伪名，步骤有序 | 有 session/project 分组；无精确执行 fork/prefix 关系 |
| 当时已物化输入 | 历史步骤、工具调用与顺序记录 | 可区分已记录步骤与后续步骤；无法证明某个未来请求当时已 Ready |

## 第二阶段补审计（2026-10-15，网络已恢复）

运行环境对 huggingface.co 与 datasets-server.huggingface.co 的访问恢复（两者均 HTTP 200）。按任务第二阶段规则执行，未下载整个仓库。

### 逐文件清单（零数据下载，经 Hub tree API 取得）

- **FreeInference**：仓库共 268 个对象、8.93 GB；119 个按天 `.jsonl.zst`（中位 2.17 MB，最大 15.75 MB，最小 0.03 MB），另有 64 个 processed Parquet 与 32 个 JSON。Dataset Server 已收录全部 17 个周 split 的 Parquet（如 week_2026_05_17 为 1,125,642 B）。
- **GATC v6**：仓库共 66.96 MB——trajectories 25.73 MB、messages 25.15 MB、sft_steps 15.28 MB、labels 0.10 MB、error_recovery 0.62 MB 等。

### Dataset Server 行查询与 Viewer redact 发现

- FreeInference 的 `/info`、`/rows` 可用：确认 `requests[]` 完整嵌套字段含 `ttft_ms`、`provider_tokens{prompt,completion,cached}`、`role_tokens`、`mutation`、`block_ids`、`tool_calls[].subagent`（递归同构 session）。
- **Viewer 与 rows API 对 `block_ids` 做 redact**（返回字面量 "redacted in viewer; see release/..."），真实 block_ids 只存在于 release 日文件中——这是必须下载日分片才能做 block 级分析的确凿依据。
- GATC 的 Dataset Server 三端点均返回"数据集不存在或需要认证"；Hub resolve 下载经 huggingface.co 与 hf-mirror.com（HF_ENDPOINT）均 401：**GATC 为 gated dataset，样本级审计在无授权凭据下 BLOCKED**。本轮不尝试绕过。

### 小样本下载与流式处理（FreeInference）

- 下载 2 个日文件，合计 29.14 MB（上限 100 MB）：`release/2026-08-30_2026-09-05/traces/2026-08-30.jsonl.zst`（15.75 MB）与 `release/2026-08-09_2026-08-15/traces/2026-08-09.jsonl.zst`（13.39 MB）。每个文件置于独立空目录 `raw/fi_<day>/`。
- `zstd -dc` 子进程流式解压 + 逐行 `json.loads`，单行最大实测 30.5 MB（一个 175 请求的 session）；任何时刻仅保留一个 session 在内存。
- 初检 464 个 session（08-30 取文件顺序前 250/共 251 行；08-09 全量 214 行），选择方式已固化在 `sample_audit.py`。
- 单行实测：`block_ids` 为整数列表（首例 16,437 块），`provider_tokens.cached=288256/288997`（99.7% 提供商侧前缀缓存命中）。

### 补审计中发现的关键负面/边界事实

1. **block 渲染非 token 无损**：62,354 个请求上 `len(block_ids)*16 / provider_tokens.prompt` 中位 0.932、p5 0.873、p95 1.148、最大 2.604——block 序列既可少算也可多算，证实在 provider 真实 serving 模板下无法从 block_ids 重建 exact token 输入。
2. **subagent 上下文不是父历史的 token 前缀延伸**：1,448 对（subagent 首请求 vs 父请求）中 0 对完全延伸父前缀；1,060 对部分共享（中位 152 块，≈system+tooldefs 头部），388 对零共享。
3. **主对话严格串行**：候选 session 钻取中 top-top 重叠对全部为 0；时间重叠只出现在并行 subagent 之间（sub-sub）及 subagent 与父窗口交界（top-sub）。
4. **周扫描最终完成（部分周）**：16 个周 split 中 13 个完整扫描、2 个因响应体过大 IncompleteRead、1 个 HTTP 500、另有 2 周仅部分行（week_2026_07_19 100/786、week_2026_08_30 600/1806）。合计远程扫描 8,617 个 session（Viewer redact 行，仅结构计数），488 个含 subagent（≈5.7%），每周均有出现（每周 3–84 个）。本轮两日样本的 subagent 占比 30/464≈6.5% 与周级 5.7% 一致，样本在该维度具有代表性。

官方来源：

- https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace
- https://datasets-server.huggingface.co/info?dataset=harvardMadsys/freeinference_agentic_trace
- https://huggingface.co/datasets/GOAT-AI/gatcv6

复现：`PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python -I evaluation/new_dataset_suitability_audit/sample_audit.py evaluation/new_dataset_suitability_audit/raw/fi_2026-08-30/2026-08-30.jsonl.zst evaluation/new_dataset_suitability_audit/raw/fi_2026-08-09/2026-08-09.jsonl.zst`（输出 `session_metrics.jsonl`、`parent_child_lcp.jsonl`、`day_summaries.json`）。
