# 独立多 Workflow 工作负载采样协议（预注册，先于一切 Token 核验固定）

本文件在运行任何选择脚本之前写定。选择顺序只依赖 B5-C1 已冻结的清单与 B5-C2 已冻结的来源锁，不依赖任何 Token 内容、恢复收益、Exact 胜负或预期 Checkpoint 位置。

## 1. 数据与顺序基准

- 原始数据：`/home/wjg/data/agentic_coding_trajectories/sessions.parquet`，SHA256 必须与 B5-C1 `source_manifest.json` 记录一致后才可继续。
- 组清单：B5-C1 `audit_summary.json` 的 `openhands.multi_session_task_groups`（1216 个同任务多 Session 组；其文件 SHA 必须与 B5-C1 `artifact_manifest.json` 一致）。
- 任务（instance_id）规范顺序：`sorted(multi_session_task_groups.keys())` 字典序。
- 任务内 Session 规范顺序：`sorted(session_ids)` 字典序（与 B5-C2 的任务内规则完全一致）。

## 2. 排除项

排除 B5-C2 已占用的 2 个 instance_id（以其 `source_selection_lock.json` 实际记录为准，脚本内硬编码值仅作交叉校验）：
`0b01001001__spectree-64`、`12rambau__sepal_ui-411`。

本轮 8 个任务与 B5-C2 的 4 条 Session 及 2 个任务不得有任何重叠；组间（本轮 4 组之间）任务与 Session 亦不得重叠。

## 3. 选择规则（固定）

1. 从规范顺序中剔除排除项后，依序取前 8 个任务，记为 T1..T8。
2. 连续两任务成组：W1=(T1,T2)，W2=(T3,T4)，W3=(T5,T6)，W4=(T7,T8)。
3. 每个任务取其规范顺序的前 2 个 Session：组内任务一为标签 A、B；任务二为标签 C、D。
4. 每条 Session 的合法性条件（在来源锁阶段检查）：
   - 在 parquet 中按 `session_id` 唯一命中一行，且该行 `ground_truth_meta_json.instance_id` 与任务一致；
   - `messages_json` 可解析为消息列表；
   - 存在第 3 条 `role=='assistant'` 的消息（第三次请求标记，截断索引 `cut` 取该消息下标）。
5. 任何合法性失败：**记录失败原样保留，不替换、不跳到下一任务**；该组标记为不完整，CPU 重建阶段如实记录。
6. 截断语义与 B5-C2 完全一致：只保存 `raw[:cut]`（即第三次请求之前的全部消息）；第三次 Assistant 输出内容与之后的消息一律不读。

## 4. 输出与顺序约束

- `source_selection_lock.json` 必须在分词之前写出（含每条记录的 parquet 行号、instance_id、session_id、source_id、source_dataset、assistant 索引、cut、截断历史文件名及其 SHA256、recorded_model）。
- 每组目录 `w1/`..`w4/` 保存该组 4 条截断历史；Token 重建只消费这些已锁定文件。
- 原始 parquet、B5-C1/B5-C2 既有产物一律只读。

## 5. 明确不做

- 不依据 Token LCP、消息结构、预期共享深浅、恢复收益或任何策略结果调整选择；
- 不为凑满 4 组而替换失败来源；
- 不修改输入规范化语义、Tokenizer、Chat Template 或历史截断方法（全部复用 B5-C2 已验证的冻结路径）；
- 不把本协议产出的任何文本/Token 结构表述为已物化的可执行 Recurrent State 共享。

## 6. 预期产物顺序

1. 本协议（已固定）；
2. `prepare_sources.py` 运行 → `source_selection_lock.json` + 16 份截断历史；
3. `tokenize_cpu.py`（冻结镜像 CPU 容器）→ 身份锁、Token 数组、渲染文本、offset、因果与边界核验、逐组前缀矩阵；
4. `finalize.py` 独立复核 → `cpu_input_audit.json`、`token_sharing_summary.csv`、`verification.json`；
5. 报告类交付（workload_inventory.md、runtime_collection_plan.md、final_report.md）。
