# 证据清单（Evidence Manifest）

调研日期：2026-10-12。本清单记录 `dataset_survey.md` 与 `dataset_comparison.csv` 中每一项关键判断的证据来源与证据等级。

**证据等级定义**

- **[实测]**：本调研在本机下载/解析了原始数据或调用 HuggingFace datasets-server API 直接读取了真实记录（最高等级）。
- **[官方]**：来自 dataset card、官方 README、官方文档页的原文陈述。
- **[论文]**：来自论文摘要/正文（arXiv 页面）的陈述。
- **[二手]**：搜索结果、第三方页面，未经官方渠道确认。

---

## 1. semianalysisai/cc-traces-weka-062126

| 判断 | 等级 | 证据 |
|---|---|---|
| 顶层 schema 为 `id, models, block_size, hash_id_scope, requests` | [实测] | `curl -r 0-4000000 https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl` 下载头部，解析首条 trace：`block_size=64`、`hash_id_scope="local"`、`models=['claude-haiku-4-5-20251001','claude-opus-4-8']`、135 个 requests（130 个 `type:"s"`、1 个 `type:"n"`、4 个 `type:"subagent"`） |
| 普通请求字段 `t, model, in, out, hash_ids, api_time, ttft, think_time, type`；示例 `in=380608` tokens 对应 `hash_ids` 长度 5947（×64 tokens/块，精确吻合） | [实测] | 同上文件首条普通请求完整解析 |
| subagent 请求含 `agent_id, subagent_type, duration_ms, total_tokens, tool_use_count, status, models` 与**嵌套 `requests[]`（子会话完整请求序列）** | [实测] | 同上文件，`{"t":86541.149,"type":"subagent","agent_id":"subagent_001_f552fe6f",...,"requests":[7项]}` |
| 规模：393 traces / 56798 主 turns / 1697 subagent 组 / 42029 子请求 / 98827 总请求 / ~21.6B input tokens | [官方] | dataset card Stats 块（https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126） |
| 无原始 prompt 文本；`in` 由 64-token 块计数导出；图片内容在源头即排除 | [官方] | dataset card："requests carry `hash_ids` ... instead of content"；"The `in` field is derived from hash-block counts" |
| 来源为真实 Claude Code 会话（SemiAnalysis Claude Code proxy traces，主 agent 通常 Claude Opus、子 agent Claude Haiku）；v7、≥20 请求/会话、CC CLI ≥2.1.139、ISL cap 990,016、并发子 agent cap 10 | [官方] | dataset card 与 NVIDIA AIPerf 文档 https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/replay-weka-agentic-coding-traces.md |
| 采集工具为 claude-code-proxy / kv-cache-tester；"All traces are fully anonymized — no conversation IDs, timestamps, or real agent IDs"（注意：该句指旧版 traces/ 目录格式，062126 版含相对时间 `t`） | [官方] | https://github.com/callanjfox/kv-cache-tester README |
| AIPerf 回放语义：SPAWN/SPAWN_JOIN 分支、hash_id 最长公共前缀检测隐藏并行 agent、`--fixed-schedule` 按记录时间戳发出、`--cache-bust` 防 KV 复用 | [官方] | AIPerf 文档页（同上） |
| License Apache-2.0；公开下载无需认证 | [官方] | dataset card + AIPerf 文档 "Both datasets are public — no HuggingFace authentication or token is required" |

**尚未核实**：全量 393 traces 的 hash 前缀共享深度分布、subagent 并发窗口宽度分布（属 Phase 2 统计）；SemiAnalysis 采集者人数/背景（card 未披露，推断为 SemiAnalysis 团队成员）。

---

## 2. sammshen/lmcache-agentic-traces

| 判断 | 等级 | 证据 |
|---|---|---|
| schema：`session_id, model, input(完整累积 messages), output_length, pre_gap` | [实测] | datasets-server first-rows：features 列表 + 首行 `session_id="swebench__astropy__astropy-12907__minimax"`，input 首条为完整 OpenHands system prompt |
| 实测 24880 迭代 / **767 sessions** / 629 任务；card 自称 787 sessions，存在 20 个 session 差异（如实记录） | [实测] | 全量下载 5 个 parquet 分片（共 2.37GB）后 duckdb 聚合：`SELECT session_id, model, count(*) ... GROUP BY 1,2` |
| **120 个任务存在 ≥2 个不同模型的 session**（102 个 2 模型 + 18 个 3 模型）；模型分布 minimax-m2.5(490)/claude-sonnet-4-6(204)/deepseek-v3.1(65)/claude-opus-4-6(8)；来源 swebench 665/gaia 94/wildclaw 8 | [实测] | 同上聚合 + `task = session_id.rsplit("__",1)[0]` 分组 |
| **同任务多模型 session 的 input 前缀逐字符一致**：`astropy-13033` 的 minimax session（68 条消息）是 deepseek session（147 条）的严格前缀，共同前缀 ≈143K 字符；8 组抽查中 5 组分歧点为 None（前缀包含）、3 组在第 3-4 条消息处分歧 | [实测] | `/tmp/lmcache_check/analyze_lmcache.py`：逐消息 `json.dumps` 比较第一处分歧 |
| 关键定性：两"不同模型"session 连 assistant 输出都逐字一致 → 该结构是**回放/续跑/重标注构造**，不是两模型独立采样，也不是运行时原生 fork | [实测]（推断基于实测数据） | 抽查两 session 第 #4 条 assistant 消息（"Let me start by exploring the repository structure..."）逐字一致 |
| `pre_gap` 为客户端工具/思考耗时（秒），可重建会话内到达过程；card 称 ">90% of tokens in a typical request have already been processed in the previous request" | [官方] | dataset card https://huggingface.co/datasets/sammshen/lmcache-agentic-traces |
| 来源为 SWE-bench Verified (OpenHands CodeAct) / GAIA (Inspect AI) / WildClaw 上多模型 benchmark 运行（非生产流量） | [官方] | dataset card |
| License CC-BY-4.0 | [官方] | dataset card |

**尚未核实**：787 vs 767 的 session 口径差异原因；wildclaw 8 个 session 的内部结构。

---

## 3. TraceLab (uw-syfi/TraceLab)

| 判断 | 等级 | 证据 |
|---|---|---|
| v0.0.2 数据实测 **665453 行**、**8058 sessions**、357 projects；store 分布 `.claude` 293032 / `.claude.back` 12413 / `.codex` 360008 | [实测] | 下载 `https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.jsonl.gz`（96MB）解压 1.2GB 全量解析 |
| 顶层字段 26 个：`provider, project, session_id, round_index, round_id, model, input_tokens_total, prefix_tokens, newly_append_tokens, claude_uncached_input_tokens, claude_cache_creation_input_tokens, claude_cache_read_input_tokens, output_tokens, reasoning_output_tokens, timing_events, tools, current_input_*, first_input_event_type, user, store, trace_key`（codex 记录另有 `turn_id`） | [实测] | 前 100000 行字段计数（100% 出现） |
| `timing_events[]` 含毫秒级 ISO8601 时间戳；event_type 分布：tool_call 743819 / tool_result 743222 / reasoning 392632 / usage_report 360008 / text 353702 / user_message 95446 | [实测] | 全量解析统计 |
| 无 prompt 原文：脱敏丢弃全部输入内容，仅保留字符/ token 计量（`current_input_chars` 等） | [实测]+[官方] | 实测无文本字段；README："sanitization drops `tools[].input` entirely while preserving `input_chars`" |
| 来源为 43 名开发者真实 Claude Code / Codex 日常会话；rounds/session 均值 54.6 中位 13 最大 2503（100K 行抽样内 1832 sessions） | [实测]+[官方] | README v0.0.1 口径 357K rounds/43 users（v0.0.2 实测 665K rounds，版本差异如实记录）；抽样统计为本机计算 |
| 论文结论：长自主循环、长上下文短输出、高但不完美的 prefix 命中率、人类节奏间隔 | [论文] | arXiv:2606.30560 摘要 |
| License：代码 Apache-2.0，数据 CC BY 4.0 | [官方] | GitHub README |

**尚未核实**：session 之间是否存在 resume/compact 导致的跨会话前缀复用（数据含 `project` 字段可支撑该分析，本轮未做）。

---

## 4. Exgentic/agent-llm-traces

| 判断 | 等级 | 证据 |
|---|---|---|
| 1781 行（=traces）；字段 `harness, benchmark, models, max_tokens, total_tokens, session_id, spans[], collected_at`；span 含 OTel `gen_ai.input.messages/output.messages/tool.definitions`、`start_time/end_time`、`span_id/trace_id/session_id`；**无 parent_span_id** | [实测]（subagent 经 datasets-server） | datasets-server first-rows + size（983.6MB） |
| 5 harness（claude_code, openai_solo, tool_calling, tool_calling_with_shortlisting, smolagents_code）× 6 benchmark（SWE-bench, AppWorld, BrowseCompPlus, TAU-bench×3）× 5 模型 | [官方] | dataset card https://huggingface.co/datasets/Exgentic/agent-llm-traces |
| License CDLA-Permissive-2.0；无论文 | [官方] | dataset card |

---

## 5. 用户指定候选（Task 1 五项）

### 5.1 nebius/SWE-rebench-openhands-trajectories
- [实测] datasets-server first-rows：schema 为 `trajectory_id, instance_id, repo, trajectory(role/content/tool_calls/tool_call_id 消息列表), tools(工具 schema), model_patch, exit_status, resolved, ...`；无时间戳字段。
- [实测] 5 个偏移量（0/1000/5000/20000/60000）各抽 100 行：500 行中仅 2 个 instance 出现重复 → **无显著同任务多 attempt 结构**。
- [官方] card：67074 行、2.08GB、平均 64.3 turns、OpenHands v0.54.0 + Qwen3-Coder-480B、CC-BY-4.0。

### 5.2 ulab-uiuc/MARBLE (MultiAgentBench)
- [官方] GitHub README：ACL 2025 Main、MIT、框架+benchmark（werewolf 示例脚本）。
- [实测] `data/` 目录仅 `map.json, map_description.json, score.json, url_prefix.json` 四个文件（GitHub 页面浏览）；**未发布既有轨迹数据集**。
- [论文] arXiv:2503.01935。

### 5.3 OpenDCAI/AgentFlow
- [官方] GitHub README：自称 "Unified Agent Data Synthesis Framework"（2026，@misc 引用，无正式 venue）；三阶段管线（Trajectory Sampling → Selection → QA Synthesis）；描述含 "building a branching trajectory tree with concurrent expansion and action de-duplication"；HF 有数据集 collection；Apache-2.0。
- **注意**：该 AgentFlow（北大 OpenDCAI 数据合成框架）与 UCLA 的 AgentFlow: Incremental Stream Processing（serving 论文）同名不同项目，引用时必须区分（[二手] 确认存在同名论文）。
- 未实测其 HF 数据格式（trace 格式未文档化）。

### 5.4 hkust-nlp/AgentBoard
- [官方] README："We also provide baseline run loggings in `data/baseline_results`"；每行 JSON 含 statistics/trajectory/prompt；WebArena 含 rendered HTML、Playwright traces、网络流量记录；数据 GPL-2.0（copyleft）、代码 Apache-2.0；NeurIPS 2024 Oral（arXiv:2401.13178）。

### 5.5 DeepNLP/Multi-Agents-Parallel-Orchestration-Dataset
- [官方] card：50 条（1K 全量 WIP）；`session_id → trace_id → function_calls[].messages/tools/tool_calls` 嵌套结构；真实平台用户请求驱动（qwen3 系列 + tavily/google/Perplexity 并行）；隐私过滤声明；**license 未标注**（YAML 元数据缺失）；viewer Parquet 转换失败需手动下载。

---

## 6. 次要候选（subagent 核查，方法：datasets-server API / GitHub raw / arXiv API 实测）

### 6.1 openbmb/UltraInteract_sft / _pair
- [实测] `_sft` 288579 行（字段 `task, dataset, instruction, response, id, parent_id`）；`_pair` 219522 行（`trajectory(chosen/rejected), id, parent_id`）。**裸名 `openbmb/UltraInteract` 返回 401**（疑 gated/下架），公开的是两个子仓库。
- [官方] README："data with the same parent_id have the same instruction but different actions"（偏好树）；MIT；arXiv:2404.02078（Eurus）；GitHub https://github.com/OpenBMB/Eurus。

### 6.2 joonspk-research/generative_agents
- [实测] storage：21 个 3-agent demo checkpoint（最长 8655 step，movement/ 每 step 一个 JSON）；`reverie/meta.json` 含 `fork_sim_code`（checkpoint 级分叉）；`personas/*/associative_memory/nodes.json` 记忆节点含 `created/depth/poornancy` 等；**无 LLM prompt 原文**（需跑代码才能产生）。
- [官方] UIST 2023；Apache-2.0；25-agent 两周实验完整日志未发布。

### 6.3 Azure/AzureLLMInferenceDataset
- [实测+官方] **不在 HuggingFace**；GitHub https://github.com/Azure/AzurePublicDataset；仅 3 列 `TIMESTAMP, ContextTokens, GeneratedTokens`；官方 md："Due to customer privacy requirements (e.g., GDPR), we do not have visibility into the content of the prompts"；无 session id；2023 版 1 天（19366+8819 行），2024 版 1 周（code 16.8M + conv 27.3M 行）；CC-BY-4.0；配套论文 Splitwise (ISCA'24)/DynamoLLM (HPCA'25)。网传"45 天/20 亿请求"未获公开来源证实。

### 6.4 microsoft/ParrotServe (Parrot)
- [实测] 原 `mobileparrot/parrot` 404；官方 artifact 为 `microsoft/ParrotServe`（MIT）；**无任何 workload trace 数据文件**（benchmark/ 仅脚本）；OSDI 2024（https://www.usenix.org/presentation/lin-chaofan）。

### 6.5 zjunlp/WorFBench
- [实测] WorFBench_test 2146 行 / train 18679 行；字段 `source, conversations, target, id, category`；target 是 "(START,1) (1,2)…" 形式的 workflow DAG 文本标签 → **是生成任务的标签，不是执行轨迹**。
- [官方] ICLR 2025（arXiv:2410.07869）；HF apache-2.0。

### 6.6 OpenBMB/ToolBench
- [实测] 官方 HF `ToolBench/ToolBench` 404；镜像 `Adorg/ToolBench` datasets-server 报格式混杂错误（无法直接解析）。
- [官方] ICLR'24 Spotlight；16,464 API / 126K+ 轨迹；轨迹由 ChatGPT **DFS 树搜索**生成（论文口径，树结构为搜索构造）。

---

## 7. 相关系统研究（数据未公开，负面证据）

| 条目 | 等级 | 证据 |
|---|---|---|
| AgentSysBench / "From LLM Inference to Agentic Workloads"（arXiv:2608.15127，2026-08）：10 应用受控部署 + 3 应用生产 trace（~178799 sessions 论文口径），发现 search query 级 cross-request redundancy；**摘要页无任何 artifact/dataset 链接** | [论文+实测核查] | https://arxiv.org/abs/2608.15127 摘要页实测无发布声明 |
| MAST "Why Do Multi-Agent LLM Systems Fail?"（arXiv:2503.13657，~897 引用）：14 失败模式标注数据集，规模百级 | [论文+二手] | arXiv + Berkeley 项目页 |
| NVIDIA AIPerf "Agentic Code dataset generator"：合成多轮 coding agent trace 供 KV cache 基准 | [官方] | AIPerf 文档 |
| MLPerf Inference agentic benchmark 使用 DeepSWE 数据集 agentic coding traces | [二手] | 搜索结果（未深查） |

---

## 8. 本调研产生的可复现核查工件

- `/tmp/tracelab_check/syfi_coding_trace.jsonl`（1.2GB，TraceLab v0.0.2 全量）+ `inspect_tracelab.py`（字段/分布统计）
- `/tmp/lmcache_check/{0..4}.parquet`（lmcache-agentic-traces 全量 2.37GB）+ `analyze_lmcache.py`（任务×模型矩阵 + 分歧点分析）
- `/tmp/weka_head.jsonl`（cc-traces-weka traces.jsonl 头部 4MB 样本）
- `/tmp/swe_stats.json`（HF statistics API 对 SWE-rebench 报错的原始返回，证明该端点对该库不可用）

**完整性声明**：以上清单中的 [实测] 项均可在本机复现；[官方]/[论文] 项均附 URL。本调研未发现、也未捏造任何无法溯源的字段或规模声明。
