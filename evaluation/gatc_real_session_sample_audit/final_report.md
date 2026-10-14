# GATC 真实会话样本审计

## 结论：BLOCKED

本轮没有取得 GATC 的实际记录行，因此不能执行 Session 筛选或对父子关系、共享历史、Tokenizer 输入进行案例级验证。D1 阶段已用 Python HTTPS 请求 Hugging Face API，收到操作系统级拒绝：`URLError: [Errno 1] Operation not permitted`；Dataset Viewer 的 trajectories 路由也不可访问。根据本任务要求，不再重复网络请求，不尝试其他下载路径，也不转而读取 FreeInference。

| 指标 | 本轮已验证数 | 解释 |
|---|---:|---|
| 实际检查的 Session | 0 | 无实际样本文件可读 |
| 可验证 Parent → Subagent 执行关系 | 0 | 不是“数据中不存在”；尚未读取样本 |
| 超越 System Prompt 的共享历史案例 | 0 | 尚未读取完整消息并逐案核验 |
| 可重建完整目标输入的案例 | 0 | 尚未取得消息；也需模板与目标 tokenizer 身份 |
| 值得进入 Runtime Replay 的案例 | 0 | 当前没有经证据链认证的具体案例 |

这些零值表示“本轮可验证数”，不代表数据集真实发生频率为零。

## 需要从可联网环境提供的输入

需先从 Hugging Face 仓库文件树确认文件的真实远端路径；D1 只能看到逻辑表名和 Schema，不能核实仓库内实际文件名，因此这里不编造路径。请提供以下最小输入，保留原始格式及来源信息：

1. 仓库 `GOAT-AI/gatcv6` 的文件清单或 Hub tree 元数据，包含每个文件的远端相对路径、大小和 revision/commit SHA。
2. `trajectories` 表对应的原始数据文件，至少覆盖仓库中的全部 1,189 个样本记录；用于按固定规则筛选最多 20 个 subagent/复杂工具会话，并保留完整 `steps`、`ts`、`duration_ms`、`args` 和 `result` 字段。
3. `messages` 表对应的原始数据文件；用于对选中 `trajectory_id` 读取完整 `messages` 内容。仅提供 Viewer 截断行不够。
4. 若 `session_kind` 或子代理筛选字段只在其他表中，提供 `labels` 表文件及其真实远端路径；否则可由完整 trajectories 自身字段筛选。

可将文件放入本目录的 `input/` 子目录，并附一个来源说明，记录下载 URL、revision、文件大小与 SHA-256。无需提供仓库其他表，也不要先下载全仓库。若在联网端不能按 `trajectory_id` 抽取 messages，可提供 messages 原始表；本地后续只按筛选 ID 流式读取。

## 当前证据边界

D1 的官方卡片说明 GATC v6 有 `trajectories` 与 `messages` 表；trajectory steps 包含 `subagent` 类型、顺序和 step 时间，消息表含 user/assistant/tool 内容。卡片还指出仓库为 1% 样本，部分文本会脱敏或截断。这些是 Schema 级能力，不是本轮针对任何具体 Session 的审计结果。D1 报告记录了当时 Hub API/Viewer 的访问错误：[`evaluation/new_dataset_suitability_audit/dataset_preflight_report.md`](../new_dataset_suitability_audit/dataset_preflight_report.md)。

在取得上述文件后，才能按全体样本的固定筛选规则选出不超过 20 个会话，核对父子记录关联、实际消息历史、step 时间顺序、Qwen3.5 tokenizer 重建条件及构造 Pending Snapshot 的因果边界。即使这些检查通过，GATC 本身仍不含 SGLang 物理 Recurrent Checkpoint 驻留证明；Runtime Replay 价值还需在后续独立执行中认证。
