# Agentic 数据集轻量级适用性审计

## 结论

| 数据集 | 状态 | 可支持的研究环节 | 尚不能支持的环节 |
|---|---|---|---|
| FreeInference Agentic Trace | **PARTIALLY_SUITABLE** | 真实会话/请求时序、显式嵌套 subagent 结构、工具与请求变化、粗粒度 block-prefix locality 分析 | Exact Token Compatibility、完整目标模型输入重放、同一边界的 pending 状态、物理 recurrent checkpoint 候选与真实分配资格 |
| GATC v6 | **PARTIALLY_SUITABLE** | 有序的 agent/tool 消息、子代理委派线索、session/project 级结构分析 | 可靠父子 trajectory 关联、逐请求起止时间与并发证据、目标 tokenizer 输入复原、runtime resident recurrent 状态 |

两者都具有 Agentic 工作流分析价值，但仅凭现有公开字段均不能直接支持受控 SGLang Hybrid-State 分配回放。其公开数据不能证明真实 Recurrent Checkpoint 已在一个可核验的 Allocation Boundary 上物化。FreeInference 的 block IDs 是 `o200k_base` 的 16-token 块摘要，不是完整输入 token 序列；GATC 的消息文本虽可读，也没有目标模型 tokenizer、完整模板配置和无损输入身份保证。

## 核心问题逐项判断

| 问题 | FreeInference | GATC |
|---|---|---|
| 真实 Parent/Subagent 或 Fork/Join | 有嵌套 subagent session 与工具委派字段；不等同于真实并行 fork/join | 有 subagent step/session-kind；缺少明确跨 trajectory 父子键 |
| 同一 Workflow 的重叠 LLM 请求 | 有 request start/end，但未获得案例行，不能确认重叠与 workflow 对应 | 无可靠 LLM request 起止时间，无法判断 |
| 区分已执行与未来输入 | 请求时间、step 与 tool-call 因果顺序可识别已记录事件；没有未来 Ready 队列 | 有序历史可区分已记录步骤，但没有 Ready/dispatch 边界 |
| 多 Continuations 共享已执行历史 | block-prefix mutation 可作粗粒度线索，受 16-token 分块及无消息文本限制 | 可在 transcript 内容上做文本层面比较，但不能证明 token-exact 或真实 fork |
| 完整 token 输入/可靠重建 | 不可；仅 block IDs，且 tokenizer 与服务模型可能不同 | 不可认证；无目标 tokenizer/template 身份，且存在脱敏和截断 |
| 因果合法 Pending Snapshot | 不足；没有状态边界与 Ready 队列证据 | 不足；没有调度事件及同步状态边界 |
| 真实物化 Recurrent Checkpoint 候选 | 无物理运行时状态、slot/generation 记录 | 无物理运行时状态、slot/generation 记录 |

因此不能把真实工作流结构、block-level prefix、exact-token compatibility、recurrent materialization、同一决策边界资格合并称为“可执行共享”。

## 样本读取状态与推荐案例

本轮没有读取任何完整样本，`candidate_cases.json` 不推荐具体 Session。FreeInference Viewer 的样本 split 报连接错误；GATC 的完整 trajectories 行接口无法从当前环境访问。Hugging Face 数据卡与 Schema 可用，足以判定二者的结构性潜力和关键证据缺口，但不足以遴选满足任务门槛的真实会话。`sample_statistics.csv` 的计数明确是数据卡级总量，不是本轮小样本统计。

没有下载任何数据文件。FreeInference 卡片报告总仓库大小 8.93 GB；GATC 卡片报告 66.1 MB，但本轮没有下载 GATC 全仓库或其任何分片。数据卡可见逐文件组织方式，逐文件大小因 Hub API 不可达而 UNKNOWN。

## 最小下一步

若进入下一轮，应先恢复只读 Dataset Server/API 查询，再对 FreeInference 选定 1–2 个日分片、累计不超过 100 MB，流式处理最多 500 sessions；对 GATC 通过远程 rows 查询筛选 subagent 轨迹，只有接口仍不可用时才考虑按明确大小上限取极少目标文件。定向筛出的案例还需独立验证父子关联、目标模型输入 token identity、同一 Allocation Boundary 的 Pending/Residency、TP ranks 与 recurrent ownership。未完成这些验证前，两数据集只能用于结构探索，不能作为 FlowState runtime allocation workload 认证。

## 证据来源

- [FreeInference 官方数据卡](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace)
- [FreeInference 官方 Viewer split](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/viewer/traces/week_2026_05_17)
- [GATC v6 官方数据卡与 Schema](https://huggingface.co/datasets/GOAT-AI/gatcv6)
- 逐项审计见 [dataset_preflight_report.md](dataset_preflight_report.md)；统计口径见 [sample_statistics.csv](sample_statistics.csv)。
