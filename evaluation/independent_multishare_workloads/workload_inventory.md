# 独立多 Workflow 工作负载清单（CPU 阶段统计与分类）

身份：每组均为 **Trace-derived Controlled Composition**（两真实 instance_id × 各 2 条真实 Session 的受控组合），不是自然并发，不是原生 Agent Fork。本清单只登记 CPU 重建与内容结构；一切 Recurrent 物化状态为 UNKNOWN。

## 1. 总览

| 组 | 任务一 / 任务二（instance_id） | 状态 | 结构分类 | 四方LCP | 任务一LCP(AB) | 任务二LCP(CD) | 跨任务LCP | 组内工具结果 |
|---|---|---|---|---:|---:|---:|---:|---:|
| w1 | 12rambau__sepal_ui-747 / 3YOURMIND__django-migration-linter-258 | REBUILT | nested_two_task | 2438 | 3451 | 3771 | 2438 | 8 |
| w2 | 4degrees__clique-26 / ARMmbed__mbed-tools-293 | REBUILT | nested_two_task | 2438 | 3511 | 3570 | 2438 | 8 |
| w3 | ASFHyP3__hyp3-sdk-53 / ASPP__pelita-696 | REBUILT | nested_two_task | 2438 | 3535 | 3471 | 2438 | 8 |
| w4 | ASPP__pelita-798 / Aarhus-Psychiatry-Research__timeseriesflattener-186 | REBUILT | nested_two_task | 2437 | 3549 | 3488 | 2437 | 8 |

- 4 组全部重建成功，**失败数 0，无删除、无替换样本**。
- 8 个 instance_id 互不相同，与 B5-C2 的 2 个任务、4 条 Session 零重叠；16 条 Session 互不重复。
- 四组结构同质（均为嵌套两任务型）是**确定性字典序规则的自然结果**——选择先于任何 Token 查看固定，未做结构偏好筛选；此同质性不外推到其余 1200+ 组。

## 2. 逐组登记（§四要求的七项）

### w1（12rambau__sepal_ui-747 × 3YOURMIND__django-migration-linter-258）

- **数据来源**：4 条 Session 的 parquet 零基行号、session_id、source_id、截断历史 SHA 见 `source_selection_lock.json`；均出自 nebius-swe-rebench-openhands 来源。
- **CPU 重建**：成功；第三次 Pending Token 数 A/B/C/D = 4250/4308/4497/4582。
- **共享结构**：四方共同前缀 2438（System 2428 ＋ 用户消息头部 10 Token），支持 {A,B,C,D}；AB=3451、CD=3771，支持分别恰为 {A,B}、{C,D}；跨任务 4 对全为 2438。
- **共同祖先与两任务共享子集**：存在共同内容前缀（2438）与两个互不重叠的任务内更深子集；**这是内容级 Token 结构，不是可执行 Recurrent State 共享**。
- **历史因果**：每 Session 消费索引 0–5（System、User、Assistant1、Tool1、Assistant2、Tool2），cut=6；组内 8 条工具结果全部可回溯先前 tool_call_id；墙钟 UNKNOWN。
- **后续 Runtime 需验证**：见 `runtime_collection_plan.md` §逐组采集（物化、FA/T、严格矩阵、支持集与预算竞争）。
- **失败原因**：无。

### w2（4degrees__clique-26 × ARMmbed__mbed-tools-293）

- 同上框架；Pending Token 数 4209/4122/4191/4667；四方 2438、AB=3511、CD=3570；跨任务全 2438；8 条工具结果全部有序可溯；无失败。
- 注意 B（4122）是全轮最短 Pending；D（4667）任务二内差异最大（546）。

### w3（ASFHyP3__hyp3-sdk-53 × ASPP__pelita-696）

- Pending Token 数 4215/4184/4548/4543；四方 2438、AB=3535、CD=3471；跨任务全 2438；8 条工具结果全部有序可溯；无失败。
- C/D（4548/4543）是全轮任务内最浅差异对（差 5 Token，LCP 3471）。

### w4（ASPP__pelita-798 × Aarhus-Psychiatry-Research__timeseriesflattener-186）

- Pending Token 数 4866/4847/4513/6971；**四方 2437**（全轮唯一比 2438 少 1 的组，System 仍 2428，用户头部共享 9 Token）；AB=3549、CD=3488；跨任务全 2437；8 条工具结果全部有序可溯；无失败。
- D（6971）为全轮最长 Pending，仍远小于引擎上下文 131200。

## 3. 结构解读边界（必须随行）

1. 共同前缀主体是数据集生产方的统一 System 提示（2428）加用户消息头部；跨任务分叉发生在 User 的工作区路径等身份字段，任务内更深 LCP 位于首条历史 Assistant 文字或工具参数——**不是两个任务共享同一原生 Fork**，与 B5-C2 的定性一致。
2. 所有 LCP/消息边界都只是 Token 位置，不构成 Checkpoint 创建列表，不要求 Runtime 在这些精确位置保留状态。
3. 本轮未观测 FA frontier 与 Planning Target，L≤T 逐边确认只能留给 Runtime 阶段。

## 4. 复核入口

- 逐组前缀矩阵与分叉 Token/字符范围：`token_prefix_matrix.json`；汇总结：`token_sharing_summary.csv`。
- 逐请求 48 份数组与摘要：`requests.json` ＋ `w{n}/IMW{n}_{label}_TURN_{t}_tokens.json`。
- 截断历史：`w{n}/{组}_{标签}_source_history.json`；渲染全文与逐 Token offset 同目录。
- 因果与边界：`causal_checks.json`；独立复核结论：`verification.json`、`cpu_input_audit.json`。
