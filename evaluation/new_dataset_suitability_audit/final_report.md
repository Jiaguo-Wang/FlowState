# Agentic 数据集轻量级适用性审计：最终报告

审计日期：2026-10-13（第一阶段，零下载）＋ 2026-10-15（第二阶段，网络恢复后小样本）。不执行新的分配比较，不预设 FlowState 会获胜。

## 结论

| 数据集 | 状态 | 可支持的研究环节 | 尚不能支持的环节 |
|---|---|---|---|
| FreeInference Agentic Trace | **PARTIALLY_SUITABLE** | 真实会话/请求时序、显式嵌套 subagent 结构（含并行 subagent 的请求时间重叠实测）、block 级前缀分叉/包含/跨 session 前导共享统计、提供商侧缓存命中分布、13 种 harness 的结构多样性 | Exact Token Compatibility（block 渲染非无损，实测中位比值 0.932）、完整目标模型输入重放、同一边界的 Pending 状态、物理 recurrent checkpoint 候选与真实分配资格 |
| GATC v6 | **BLOCKED** | 数据卡/Schema 级结构分析已完成（第一阶段）：有序消息、subagent 委派线索、session/project 分组 | 样本级审计全部阻断：数据集访问受限，Hub 与镜像下载均 401，Viewer/Dataset Server 无认证不可用；父子关联、请求级时间、token 复原均无法核验 |

FreeInference 判 PARTIALLY_SUITABLE 而非 SUITABLE 的决定性理由与第一阶段一致并已被样本量化：`block_ids` 是 `o200k_base` 16-token 块的周局部整数编号，且渲染规则（tool definitions、system prompt、messages 顺序、丢弃末块）与 provider 真实 serving 模板不同构——62,354 个请求实测 `blocks×16/prompt` 中位 0.932、最大 2.604。它证明结构，不证明 token 身份。

GATC 判 BLOCKED 依据任务定义"无法可靠完成检查"：两个官方下载通道（huggingface.co、HF_ENDPOINT=hf-mirror.com）均对 `sample/` 文件返回 401"数据集访问受限，须有权限并登录"。本轮不尝试绕过认证；第一阶段的数据卡级分析保留作为其结构性潜力与缺口的记录（`dataset_preflight_report.md` GATC 节）。

## 核心审计问题逐项判断（FreeInference，基于 464 session 实测）

| 问题 | 判断 | 样本证据 |
|---|---|---|
| 1. 真实 Parent/Subagent 或 Fork/Join | **有 Parent/Subagent，无 Fork/Join 事件** | 30/464 session 含内嵌 subagent（递归同构 session 记录）；最大嵌套深度实测 1。无 join/dispatch 事件流。远程周扫描（13/16 周完整、8,617 session）测得全期 subagent 占比 ≈5.7%，与本样本 6.5% 一致 |
| 2. 时间重叠请求及 workflow 归属 | **有，且全部来自并行 subagent** | 31 个 session 存在 `[start_ms,end_ms]` 重叠（最大并发 8；单 session 最多 2,449 个 sub-sub 重叠对）；钻取证实 top-top=0——主对话串行，重叠只出现在 subagent 之间或其与父请求窗口交界；3/4 的 subagent 生命期与父请求窗口相交于 rank-1 案例 |
| 3. 区分已执行历史与未来输入 | **部分** | 请求时间、step/turn_index、mutation 因果顺序可识别已发生事件；无 Ready 队列或未来请求预告 |
| 4. 多 Continuation 共享执行历史 | **block 级成立，token 级不可证** | 275/464 session 有 block 前缀分叉（同前缀后 ≥2 不同后继）；单 session 最大共享前缀 68,924 块；597 万个"请求 A 是请求 B 前缀"的包含对。但 subagent 首请求 1,448 对中 0 对完全延伸父前缀（中位共享仅 152 块头部）——委派不继承父对话正文 |
| 5. 完整 Token 输入或可靠重建 | **不可** | 无消息文本；block 渲染与 provider prompt 比值中位 0.932、p95 1.148、max 2.604，非一对一映射 |
| 6. SGLang 受控因果合法 Pending Snapshot | **不直接支持** | 可用时间/结构设计受控回放的输入脚本（哪些请求、何时到达、历史多长），但无当时服务端状态、无 slot/generation、无 barrier 证据；构造的 Pending 只能是研究设计，不是数据原生事实 |
| 7. 真实物化 Recurrent Checkpoint 候选 | **不可证** | 前缀包含对（含 31 万对的 rank-3 案例）给出 checkpoint 可复用的结构形状；`provider_tokens.cached`（两日均值 0.898/0.962）证明提供商侧确有前缀缓存命中，但二者都不等于本地 recurrent 状态物化 |

### 五概念区分（不得混同）

| 概念 | FreeInference 证据 |
|---|---|
| 真实 Workflow 结构 | 成立：session + 递归 subagent + 13 种 harness |
| Block-level Prefix Sharing | 成立：分叉/包含/跨 session 前导共享（最多 44 session 共享前导 32 块） |
| Exact Token Compatibility | 不成立：block 渲染非无损（中位 0.932，max 2.604） |
| 真实 Recurrent Checkpoint 物化 | 不成立：无 runtime 状态记录 |
| 同一决策边界的状态分配资格 | 不成立：无 Ready/barrier/quota 事件 |

## 推荐继续分析的 Session（≤3，全部 FreeInference；GATC 因授权阻断无候选）

1. **`9b39a908bf7bc46b95f50967d877e3f8`**（2026-08-30，opencode）：并行 subagent 重叠最密集——612 展平请求、并发 8、2,463 重叠对（top-top 0 / top-sub 14 / sub-sub 2,449）、4 个 subagent 中 3 个生命期与父请求窗口相交、双模型双 provider。
2. **`e571369a6a1360d201f8ac75e1c5ab30`**（2026-08-09，claude-code）：6 subagent + 20 个 block 前缀分叉点 + 39 个 sub-sub 重叠对，单 provider 单模型，结构最干净。
3. **`709937a13aed1ad78c0c96247ef9b0c4`**（2026-08-30，opencode）：1,506 展平请求、最大共享前缀 57,498 块、313,685 个前缀包含对——append-only 深 prefix 链与 checkpoint 复用形状的规模上限案例。

完整指标与证据链见 `candidate_cases.json`；逐 session 指标见 `session_metrics.jsonl`；父子前缀明细见 `parent_child_lcp.jsonl`。

## 与 FlowState 研究需求的对接判断

- **可用于**：真实负载的结构参数化（请求到达序列、会话长度分布、subagent 并发窗口、前缀增长/分叉形状、cache 命中率参照）、分配问题的动机统计（"同 session 内多 continuation 共享历史后分叉"的真实频率）。
- **不可用于**：exact-token 合法性认证、runtime 物化证据、Allocation Boundary 上的分配资格——这些必须沿用仓库内已有受控路径（如 B4-B0/neutral collector 的已知输入回放）。
- **GATC 解锁条件**：获得数据集访问授权后，按 `sample/labels`（0.10 MB）定向筛 subagent 案例，再按需流式读取 trajectories 分片；其消息文本可读性是相对 FreeInference 的唯一潜在优势，但脱敏/截断与匿名模型身份仍是 exact 复原的硬边界。

## 完整性与复现

- 下载清单：2 个日文件共 29.14 MB（低于 100 MB 上限），位于 `raw/fi_2026-08-30/`、`raw/fi_2026-08-09/` 各自独立目录；GATC 下载尝试仅得到 124 字节 401 消息体，已删除，未保留任何 GATC 数据。
- 复现命令见 `dataset_preflight_report.md` 末节；脚本 `sample_audit.py` 只读已下载数据，不联网、不导入 engine 或选择器。
- 本轮未运行 FlowState 分配算法、未启动 GPU/Docker、未修改旧实验产物。

本轮完成并停止，等待下一轮判断。
