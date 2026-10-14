# Agentic Workflow 公开数据集调研与结构适用性分析（Phase 1）

调研日期：2026-10-12 ｜ 调研人：Claude（独立数据调研，未参考任何 FlowState 既有实验）
证据等级标注贯穿全文：**[实测]**（本机解析过原始数据）、**[官方]**（dataset card / README / 官方文档）、**[论文]**、**[二手]**。所有关键判断的原始出处见 `evidence_manifest.md`。

---

## A. 研究背景与筛选标准

### A.1 为什么普通 Tool-Calling 数据集不适合本研究

本研究的核心对象是 Hybrid LLM（Full-Attention KV Cache + Recurrent State）上**长生命周期 Agentic Workflow 的执行状态管理与缓存资源分配**。这要求轨迹数据能回答"多个 Continuation 何时共享哪些可复用状态、何时并发地竞争容量"。绝大多数现有 Agent/Tool-Calling 数据集为**训练目的**构造（SFT/偏好对），它们满足于"一条任务一条线性轨迹、有 thought-action-observation 三元组即可"，缺三类对 serving 研究致命的信息：

1. **缺"结构"**：没有 parent-child/fork/branch 字段，无法识别哪些 Continuation 从同一历史分出。训练数据把树拍平成独立序列后，共享关系永久丢失。
2. **缺"时间"**：没有请求级时间戳/到达过程，无法还原并发窗口、决策时刻的 pending tasks、tool pause-resume 间隔。离线轨迹"事后全知"不等于"决策时已知"（任务原则 3）。
3. **缺"可 tokenize 的原始输入"**：只有蒸馏后的 instruction-response，没有逐轮累积的真实 model input，就无法用真实 Tokenizer 检验 Exact Prefix Sharing 或 Recurrent Checkpoint 兼容性。

反过来，**serving 社区的传统 trace（如 Azure LLM Inference trace）有时间与规模，却因隐私把内容脱敏成 token 计数**，同样无法做内容级验证。本调研正是在这两类数据之间寻找同时满足结构、时间、内容三个维度的公开数据集。

### A.2 四层适用性判据（Task 3 的 Level 1–4）

- **Level 1 Workflow Dependency**：是否存在显式依赖字段/事件（不是"顺序相邻"的臆测）？
- **Level 2 Shared Execution History**：不同 Continuations 是否有**相同的 Conversation History / Token Prefix**（严格区分"相同任务描述"≠"相同历史"≠"相同 token 前缀"）？
- **Level 3 Concurrent / Pending Continuations**：能否用时间戳还原同时活动的 Continuations 与决策时刻已知的 Pending Tasks？
- **Level 4 Serving State Reuse Potential**：数据是否支持后续用真实 Tokenizer/Runtime 验证 Exact Prefix、Recurrent Checkpoint 兼容、恢复收益？本轮只评估**可验证性**，不声称轨迹已证明 GPU 上的真实驻留/复用。

### A.3 评分体系（每维 0–3 分）

| 维度 | 权重 |
|---|---:|
| D1 原始 LLM 输入与完整执行历史 | 20% |
| D2 明确的 Fork/Branch/Parent-Child 证据 | 25% |
| D3 多 Continuation 依赖与并发信息 | 20% |
| D4 可验证精确 Token Prefix 的信息完整性 | 20% |
| D5 开源可访问、复现成本及数据质量 | 15% |

---

## B. 候选数据集横向比较（16 个）

第一轮检索获得 16 个可核实候选（检索方向：agentic workflow traces / multi-agent trajectories / coding agent traces / LLM serving workload characterization / workflow DAG）。总表见 `dataset_comparison.csv`，此处为按总分排序的摘要（分数 = Σ 维度分×权重，满分 3）：

| # | 数据集 | 来源类型 | 规模 | 完整历史 | Fork/分支 | 时间/并发 | Prefix 可验证 | 总分 |
|---|---|---|---|---|---|---|---|---:|
| 1 | **cc-traces-weka-062126** | 真实使用（Claude Code proxy） | 393 traces / 98.8K 请求 / 21.6B tok | 无原文（hash 结构） | **YES 原生 subagent** | **YES t/think/api/ttft** | 块级(hash) | **2.40** |
| 2 | **lmcache-agentic-traces** | 实验轨迹（多模型 benchmark） | 767 sessions / 24.9K 迭代 | **YES 全文本** | PARTIAL 回放构造 | pre_gap | **YES 可 tokenize** | **2.35** |
| 3 | Exgentic/agent-llm-traces | 实验轨迹（5×6×5 矩阵） | 1781 traces | YES（OTel） | PARTIAL 无 parent_span | YES span 级 | YES | 2.10 |
| 4 | UltraInteract_sft/_pair | 实验轨迹（偏好树） | 288K+219K 行 | YES | YES parent_id（构造树） | NO | YES | 2.00 |
| 5 | ToolBench（镜像） | 实验轨迹（DFS 搜索树） | 126K+ 轨迹 | YES | PARTIAL 搜索树 | NO | YES | 1.80 |
| 6 | **TraceLab (uw-syfi)** | **真实使用（43 开发者）** | 665K rounds / 8058 sessions | 无原文（全计量） | PARTIAL | **YES 毫秒级** | 计量级(prefix_tokens) | **1.70** |
| 7 | AgentFlow (OpenDCAI) | 纯合成（数据合成框架） | HF collection | PARTIAL | PARTIAL 构造树 | UNKNOWN | PARTIAL | 1.70 |
| 8 | DeepNLP Multi-Agent Parallel | 真实使用（平台流量） | 50 条（1K WIP） | YES | PARTIAL 并行 trace | NO | PARTIAL | 1.60 |
| 9 | MARBLE (MultiAgentBench) | 框架（无既有轨迹） | — | 需自跑 | 框架级 | UNKNOWN | 需自跑 | 1.50 |
| 10 | SWE-rebench-openhands | 实验轨迹 | 67K 轨迹 / 2.08GB | YES | **NO（实测无重复）** | NO | PARTIAL | 1.45 |
| 11 | MAST 失败标注集 | 标注轨迹 | 百级 | PARTIAL | PARTIAL | NO | PARTIAL | 1.35 |
| 12 | AgentBoard baseline logs | 实验轨迹 | — | YES | NO | NO | PARTIAL | 1.10 |
| 13 | Generative Agents | 合成模拟 | 3-agent demo 8655 step | NO（无 prompt） | PARTIAL checkpoint | 模拟时间 | NO | 1.10 |
| 14 | AzureLLMInferenceDataset | 真实生产（脱敏） | 1 周 44M 行 | NO（3 列） | NO | YES | NO | 1.05 |
| 15 | WorFBench | 构造（DAG=标签） | 21K 行 | 输入侧 | NO | NO | PARTIAL | 0.85 |
| 16 | ParrotServe / AgentSysBench | 系统研究（无数据） | — | — | — | — | — | 0.30 / 0.15 |

> **覆盖性说明**：用户指定的 5 个候选全部核查（SWE-rebench、MARBLE、AgentFlow、AgentBoard、DeepNLP，见 §B.1–B.5 与负面发现 §E）。候选不止于此 16 个——TRAIL（148 条错误标注 trace，规模过小）、Kimi-Dev / SWE-agent Zenodo 轨迹（单 agent 线性、与 SWE-rebench 同性质）、SWE-Gym/R2E-Gym/SWE-smith（训练环境为主）、tau-bench/CRMArena（评估基准、无公开大规模轨迹）、rStar/ToT/LATS（MCTS 推理树、非 agentic tool use）等在初筛后因结构或可访问性弱于上表同类而未列入总表；如需可扩展。

### B.1–B.5 用户指定候选的核查结论速览

1. **SWE-rebench-openhands-trajectories**[官方+实测]：67,074 条 OpenHands 轨迹、完整 messages、CC-BY-4.0，是**最好的长会话语料**；但实测 5×100 行抽样中同 instance 重复仅 2 例 → 无同任务多 attempt 分支，无时间戳。D2=0、D3=0。
2. **MARBLE**[官方+实测]：ACL 2025 多智能体协作框架；`data/` 目录实测仅 4 个地图 JSON，**未发布轨迹数据集**；要轨迹需自行运行其仿真。作为"可自生成的受控多智能体负载"有价值，但复现成本高且非真实工作负载。
3. **AgentFlow (OpenDCAI)**[官方]：注意这是**数据合成框架**（branching trajectory tree 为其合成管线概念），与 UCLA 同名 serving 论文是两个项目。其 HF collection 为合成训练数据，trace 格式未文档化（未实证）。对"真实工作负载"研究是合成补充，不构成主负载。
4. **AgentBoard**[官方]：NeurIPS 2024 Oral，`data/baseline_results` 发布 baseline 运行日志（每例含 statistics/trajectory/prompt）。单 agent 为主、无 fork/时间戳；且数据 GPL-2.0 有 copyleft 传染。
5. **DeepNLP/Multi-Agents-Parallel-Orchestration**[官方]：概念上**正中本题**（session_id 下多 trace_id 并行 agent + 完整 messages + 真实平台流量），但仅 50 条（1K 全量 WIP）、license 未标注、viewer 损坏、无时间戳——按任务原则只能记为"值得跟踪，暂不可依赖"。

---

## C. 重点数据集详细分析（4 个）

### C.1 semianalysisai/cc-traces-weka-062126 —— 结构最完备的真实 Claude Code 会话 trace

**是什么**：SemiAnalysis 通过 claude-code-proxy（Weka KV-Cache-Tester 研究工具）捕获的真实 Claude Code 会话语料（corpus v7，2026-06 构建），经结构化脱敏后发布。Apache-2.0，HF 公开无需认证，1.85GB JSONL。

**实测 schema**[实测，/tmp/weka_head.jsonl]：
```
trace := { id, models[1..4], block_size: 64, hash_id_scope: "local", requests[] }
request(type="n"|"s") := { t, model, in, out, hash_ids[int...], api_time, ttft, think_time, type }
request(type="subagent") := { t, agent_id, subagent_type, duration_ms,
                             total_tokens, tool_use_count, status, models, requests[嵌套子会话] }
```
首条样本：135 请求 = 130 流式 + 1 非流式 + 4 个 subagent 组；普通请求 `in=380608` tokens ↔ `hash_ids` 长 5947（×64 token/块精确吻合）。

**四层评估**：
- **L1 依赖**：✅ subagent 组嵌在父会话请求流中，父在 SPAWN 处挂起、SPAWN_JOIN 处恢复（AIPerf 回放器语义[官方]）——显式 parent-child。
- **L2 共享历史**：✅（结构级）每个请求的 `hash_ids` 序列即其 KV-cache 块指纹；父与子、兄弟之间共享哪些 64-token 块可直接由 hash 前缀重合计算，且 AIPerf 正是用 hash 最长公共前缀检测隐藏并行 agent[官方]。
- **L3 并发/Pending**：✅ 每请求相对时间戳 `t` + `think_time`（客户端间隙）+ `api_time`/`ttft`（服务侧）+ subagent `duration_ms` → 可重建"父挂起期间哪些子会话在飞"的真实并发窗口与 pending 集合。1,697 个 subagent 组 / 42,029 子请求给了统计功效。
- **L4 Reuse 验证**：⚠️ 块级（64-token 粒度）前缀结构可做共享深度/复用率分析并可被 AIPerf 直接回放；但**无 prompt 原文**，无法用目标 Tokenizer 做内容级 exact-prefix 或 Recurrent Checkpoint 兼容性检验。`in` 由块计数导出，非真实 tokenization[官方]。

**限制**：393 条会话规模中等；采集者为一支团队（同质性偏 coding）；匿名化去掉了真实身份；无文本。

**推荐用途**：多 Workflow 缓存共享结构、子 agent 并发窗口与容量竞争建模、KV 感知回放实验的**首选结构基准**。

### C.2 sammshen/lmcache-agentic-traces —— 唯一可做内容级 Prefix 验证的多 Continuation 集

**是什么**：LMCache 团队发布、面向 stateful LLM serving（KV/prefix cache）基准的轨迹集：SWE-bench Verified（OpenHands CodeAct）、GAIA（Inspect AI）、WildClaw 任务在 4 个模型上的运行记录。CC-BY-4.0，2.37GB parquet。

**实测结构**[实测，全量下载]：767 sessions / 24,880 LLM 迭代；每行 = 一次 LLM 调用：`session_id, model, input(该轮完整累积 messages 含 tool_calls), output_length, pre_gap(客户端工具耗时秒)`。

**关键实证**：
- **629 个任务中 120 个有 ≥2 个模型的 session**（102×2 模型 + 18×3 模型）[实测]。
- 抽查 8 组同任务多模型对：5 组的短 session input 是长 session 的**严格逐字符前缀**（如 astropy-13033：68 轮 vs 147 轮，共同前缀 ≈143K 字符，连 assistant 输出都一致）；3 组在第 3–4 条消息（约 37K 字符模板前缀：system prompt + issue + 初始观察）后分歧[实测]。
- 由此定性：这些同源对是**回放/续跑/重标注构造**（两"模型"的前 68 轮完全一致不可能是独立采样），不是运行时原生 fork，但其 prompt 文本是真实 agent 运行产物，**可直接 tokenize 验证 exact prefix 与共享深度**[实测推断]。
- card 自述 ">90% of tokens in a typical request have already been processed in the previous request"[官方]——单会话内的累积前缀复用是其实测卖点。

**四层评估**：L1 ⚠️（无显式依赖字段，需从 session_id 聚合）；L2 ✅（内容级，可 tokenize）；L3 ⚠️（`pre_gap` 可重建会话内到达，但无绝对墙钟、无跨会话并发）；L4 ✅（文本在手，兼容任意 Tokenizer/Runtime 检验）。

**限制**：回放构造的"分支"不是原生 fork；任务域窄（3 个 benchmark）；无跨会话并发时间。

**推荐用途**：exact token prefix 共享、Recurrent Checkpoint 兼容性、同源多 Continuation 恢复收益的**内容级验证载体**。

### C.3 TraceLab (uw-syfi) —— 最大规模的真实使用计量 trace

**是什么**：UW SyFI Lab 43 名开发者日常 Claude Code + Codex 使用 trace（采集→脱敏→归一化），配套论文 arXiv:2606.30560（2026-06）。代码 Apache-2.0、数据 CC BY 4.0。

**实测结构**[实测，v0.0.2 全量 1.2GB 解析]：665,453 LLM rounds / 8,058 sessions / 357 projects（README v0.0.1 口径为 357K rounds，版本差异如实记录）。每行一轮 LLM 调用，26 个字段，含：
- **prefix/cache 计量**：`input_tokens_total, prefix_tokens, newly_append_tokens, claude_cache_creation_input_tokens, claude_cache_read_input_tokens`——直接给出每轮的前缀长度与 API 层缓存命中/写入；
- **时间**：`timing_events[]` 毫秒级 ISO8601 事件流（tool_call 743K / tool_result 743K / reasoning 393K / usage_report 360K / text 354K / user_message 95K 个事件）；
- **结构**：project → session → round 层级、`.claude/.codex` 双 store、模型名（opus-4-7/4-8、haiku-4-5、sonnet-4-6 等）；
- **无任何 prompt 原文**（脱敏仅保长度）。

**四层评估**：L1 ⚠️（层级有、显式 fork 无）；L2 ⚠️（prefix_tokens 可验证前缀**长度关系**与复用率，不能验证内容）；L3 ✅（毫秒时间戳 × 8058 真实会话 = 真实到达/并发过程；rounds/session 中位 13、长尾 2503）；L4 ⚠️（计量级校准用）。

**限制**：无文本；单一研究团队采集（coding 同质）；43 用户规模有限。

**推荐用途**：真实工作负载的**分布校准器**——到达过程、会话长度、前缀复用率、增量增长（append-length）分布，用于给 cc-traces/lmcache 上做的结构实验设定真实参数。

### C.4 Exgentic/agent-llm-traces —— 标准化格式（OpenTelemetry GenAI）的多 harness trace

[实测+官方] 1,781 traces：5 harness（claude_code、openai_solo、tool_calling、tool_calling_with_shortlisting、smolagents_code）× 6 benchmark（SWE-bench、AppWorld、BrowseComp+、TAU-bench×3）× 5 模型（Claude Opus 4.5、GPT-5.2、Gemini 3 Pro、DeepSeek-V3.2、Kimi-K2.5）。span 级 `start_time/end_time`、OTel `gen_ai.input.messages` 完整文本与工具 schema、CDLA-Permissive-2.0。**缺 parent_span_id**（层级靠时间重叠推断）、benchmark 顺序执行（并发为会话内相对并发）。它是"如果未来平台统一 trace 标准，数据会长什么样"的现实样板，也是跨 harness 前缀比较（同一 benchmark 任务在不同 harness/模型下的输入差异）的现成素材。

---

## D. Top 2–3 推荐名单

### D.1 推荐组合（互补而非互替）

三个数据集恰好覆盖三个正交维度，**没有单一数据集能同时满足全部四层**——这是本次调研最重要的结构性结论：

| 维度 | cc-traces-weka | lmcache-agentic | TraceLab |
|---|---|---|---|
| 真实性 | 真实使用 | benchmark 实验 | 真实使用 |
| 原生分支/依赖 | ✅ subagent 嵌套 | ⚠️ 回放构造 | ⚠️ 层级 |
| 并发/时间 | ✅ t/think/api | ⚠️ pre_gap | ✅ 毫秒级 |
| 内容级 prefix | ❌ hash 块级 | ✅ 全文本 | ❌ 计量级 |
| 角色 | 结构与并发基准 | 内容验证载体 | 分布校准器 |

### D.2 Top-1：cc-traces-weka-062126（下一阶段最值得深入检查的一个）

**为什么**：
1. 它是当前**唯一**同时具备"真实使用 + 原生 parent-child（subagent spawn）+ 请求级时间戳 + KV-cache 块级前缀指纹"的公开数据。Level 1/3 直接满分证据，Level 2 有块级代理。
2. 它的 `hash_ids` 就是为 KV-cache 研究设计的（block_size=64，与 vLLM/SGLang 页语义同量级），1,697 个 subagent 组给了"父挂起 + 多子并发"的统计功效——这正是容量竞争（多个 in-flight continuations 争 KV 显存）建模需要的原始形态。
3. 工具链现成：NVIDIA AIPerf 可直接按记录时间戳回放（`--fixed-schedule`）、支持 SPAWN_JOIN 并行子会话、`--cache-bust` 防污染，从数据到 serving 实验的路径最短。

**比常见 Agent Trace 更适合研究什么**：常见 trace（SWE-bench 类、ToolBench 类）是"每任务一条线性轨迹"；cc-traces-weka 记录的是**运行时的分支与并发本身**——谁在何时从哪个前缀分出、分出的子会话跑了多少 token、父等了多久。这是研究"多 Workflow 状态共享与资源竞争"的直接对象而非代理指标。

**下一阶段必须确认的事实**（本轮未完成）：
- 全量 393 traces 的 hash 前缀共享深度分布（父-子、兄弟-兄弟），以及 64-token 块粒度与真实 Tokenizer 块边界的偏差；
- subagent 并发窗口宽度的分布（同时 in-flight 的子会话数 × 各自上下文长度 → KV 容量需求的联合分布）；
- 采集者规模与会话多样性（card 未披露人数；若全部来自个位数开发者，外部效度需打折）；
- `in` 由块计数导出与真实 tokenization 的误差量级。

**转换为 Serving Workload 的难点**：无原文 → 回放时 prompt 由 hash 确定性重建（合成填充文本保持 cache 结构），因此适合**结构/容量**实验，不适合内容敏感的语义缓存研究；模型全部映射为 1–2 个目标模型时需注意原 trace 里主/子 agent 模型异构（Opus 主 + Haiku 子）对调度的影响。

### D.3 Top-2：sammshen/lmcache-agentic-traces

**为什么**：唯一"文本在手 + 天然多 Continuation"的集合。120 组同任务多模型对提供了可控的"同源分叉"样本（已实证 143K 字符级逐字前缀），可直接用目标 Tokenizer 量化 exact-prefix 共享深度、验证 Recurrent Checkpoint 在分叉点的兼容性假设、计算按需重 prefill 的比例。`pre_gap` 允许在会话内按真实客户端节奏回放。
**关键限制**：其"分支"是回放/续跑构造（必须如实标注为 Experimental，不能当作生产 fork 证据）；无跨会话并发；任务域窄。
**难点**：OpenHands 累积 messages 需按目标 runtime 的模板拼装规则重新 token化（system/tool 格式化差异会改变前缀对齐）；120 组对的样本量对统计检验偏小，适合做机制验证而非负载规模实验。

### D.4 Top-3：TraceLab（计量级）
**为什么**：8058 个真实会话/665K 轮的毫秒级时间线 + 每轮 prefix/cache 计量，是给任何结构实验提供真实参数（到达率、会话长度、前缀复用率、append 分布）的最佳校准源；论文本身已证明"高但不完美的 prefix 命中率"——为容量竞争研究提供了真实动机证据。
**难点**：无文本，只能与上述两者联合使用（结构实验用 cc-traces、内容实验用 lmcache、参数分布用 TraceLab）。

---

## E. 负面发现（看似相关、实际不适合的数据集）

| 数据集 | 表面吸引力 | 实测否定原因 |
|---|---|---|
| **WorFBench** (ICLR'25) | "Agentic Workflow 生成基准" | 其 workflow DAG 是**待生成的文本标签**（"(START,1) (1,2)…"），不是执行轨迹；无历史/时间/分支字段[实测] |
| **Parrot/ParrotServe** (OSDI'24) | 语义变量 = 请求 DAG，概念完美 | **从未发布任何 workload trace**；原 mobileparrot/parrot 已 404；DAG 只存在于运行时 API[实测] |
| **AgentSysBench**（From LLM Inference to Agentic Workloads, arXiv:2608.15127） | 10 应用 + ~179K 生产 sessions 的 agentic 负载画像 | 论文结论（cross-request redundancy、工具成本>推理成本）重要，但**生产 trace 未公开**，摘要页无 artifact 链接[实测核查] |
| **AzureLLMInferenceDataset** | 真实生产 trace、千万行级 | 仅 `TIMESTAMP/ContextTokens/GeneratedTokens` 三列，无 session id、无内容（GDPR 脱敏）——只能做到达过程，无法支撑状态共享研究[官方+实测]；网传"45 天/20 亿请求"未获公开来源证实 |
| **SWE-rebench-openhands-trajectories** | 67K 条真实 issue 驱动的完整轨迹 | 实测**无同任务多 attempt**（500 行抽样仅 2 重复）、无时间戳 → 无分支、无并发；只适合长会话语料[实测] |
| **MARBLE/MultiAgentBench** | ACL'25 多智能体协作基准 | `data/` 仅地图文件，**无轨迹数据发布**；自跑可生成但成本高、且为仿真负载[实测] |
| **AgentFlow (OpenDCAI)** | 名字含 Flow/branching tree | 是**数据合成框架**（与 UCLA 同名 serving 论文是两回事），树为构造；无文档化 trace 格式[官方] |
| **UltraInteract / ToolBench** | 显式树字段（parent_id / DFS 树） | 树是**搜索/偏好构造**、无时间戳、无真实环境并发；且分别存在裸仓 401（UltraInteract）与官方库 404（ToolBench）的访问问题[实测] |
| **Generative Agents** | 25-agent 并发小镇、checkpoint 分叉 | 日志是模拟器状态/记忆，**无 LLM prompt 原文**；25-agent 实验全量日志也未发布[实测] |
| **AgentBoard** | NeurIPS'24 Oral、有 baseline 日志 | 单 agent、无时间戳；数据 GPL-2.0 copyleft[官方] |
| **MAST** | 多智能体失败轨迹标注 | 规模百级，标注目标是失败模式而非负载结构[论文] |
| **DeepNLP Multi-Agent Parallel** | 真实平台流量的并行多 agent | 仅 50 条、license 缺失、viewer 损坏——跟踪但不可依赖[官方] |

---

## F. 结论：回答核心问题

> **目前公开的 Agentic Workflow 数据中，是否存在足够真实、足够完整的执行轨迹，可以用来进一步研究多 Workflow 的可执行状态共享和资源竞争？**

**存在，但不存在单一完备数据集；存在一个三件套组合，且各有必须承认的缺口。**

1. **真实性**：cc-traces-weka（真实 Claude Code 会话 + 原生 subagent 并发）与 TraceLab（43 开发者 8058 会话的真实计量）证明"真实使用级"公开数据在 2026 年已经出现；但生产平台级全量 trace（AgentSysBench、Azure 内部）依然不公开。
2. **完整性**：没有任何公开数据集同时具备"原生分支 + 时间戳 + 可 tokenize 原文"。结构最强者（cc-traces-weka）无文本；文本最全者（lmcache）的分支是回放构造；规模最大者（TraceLab）无内容。**内容级 exact-prefix 验证（Level 4）目前只能在实验构造的分支（lmcache 120 组对）上进行；"生产级 fork 的内容级验证"在公开数据中尚不存在**——这是数据缺口，应在论文中如实声明。
3. **对 Hybrid LLM 状态研究的直接意义**：cc-traces-weka 的 `hash_ids`（64-token 块）+ subagent 拓扑 + 时间戳，已足以支撑"共享祖先状态的分支结构、并发 KV 容量需求、恢复收益的结构上界"研究；lmcache 提供内容级 tokenize 下界验证；TraceLab 提供真实参数分布。三者拼合后，除"真实 GPU 上的 Recurrent State 驻留证据"（任何离线数据都不可能提供，需 Phase 2 运行时实验）之外的全部前置数据条件已满足。

> **最推荐先研究哪个数据集？**

**semianalysisai/cc-traces-weka-062126**。理由：它是唯一以"运行时分支与并发"为第一公民的真实使用数据（原生 subagent SPAWN/JOIN、请求级 t/think_time/api_time、KV-cache 块哈希前缀），直接命中 Level 1/2/3，且 AIPerf 回放工具链使其到 serving 实验的转化路径最短。第一个研究动作应是：全量统计 393 traces 的 hash 前缀共享深度与 subagent 并发窗口分布，用 TraceLab 校准到达参数，用 lmcache 的 120 组同源对做 Tokenizer 级 exact-prefix 交叉验证。

**严格边界声明（按研究原则）**：以上判断只说明"数据支持后续验证"。公开轨迹不证明真实 GPU 中存在共享 Recurrent State、物理缓存驻留或显存竞争——那些只能由下一阶段的受控 Runtime 实验建立。

---

## G. 完成标准对照（自评）

| 要求 | 状态 |
|---|---|
| 1. 公开候选清单与完整对比 | ✅ 16 个候选，`dataset_comparison.csv` + §B |
| 2. ≥3 个重点数据集原始数据核查 | ✅ 4 个（cc-traces-weka 头部样本实测 / lmcache 全量下载分析 / TraceLab 全量解析 / Exgentic datasets-server 实测） |
| 3. 哪些数据集能证明真实分支关系 | ✅ cc-traces-weka（原生 subagent，最强）；lmcache（回放构造）；UltraInteract/ToolBench（构造树）——严格分级陈述 |
| 4. 哪些具备 Prefix/Pending 验证条件 | ✅ lmcache（内容级）+ cc-traces-weka（结构级 pending 重建）+ TraceLab（计量级） |
| 5. Top 2–3 推荐 + 理由 | ✅ §D |
| 6. 尚缺哪些证据 | ✅ §F.2（生产级 fork 的内容级验证缺失；GPU 运行时证据不可能来自离线数据） |
| 7. 下一阶段最值得深入的一个 | ✅ cc-traces-weka-062126（含待确认事实清单） |

**判定：PASS**（检索、实测核查、负面结论完整；关键结论均有可追溯证据，见 `evidence_manifest.md`）。

**本轮停止边界**：未下载数十 GB 完整数据（TraceLab 96MB 与 lmcache 2.37GB 为核查必需的最小全量）、未做批量轨迹转换、未触碰任何推理系统/GPU、未构建任何人为有利的 Workflow。遵守任务原则 1–6。
