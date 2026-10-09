# 代表性案例：冻结分配结果机制审计

所有成本单位均为模型预测 ms，不代表 runtime latency。案例在全量 573 cases 审计完成后按以下规则确定。C 编号只用于本案例内映射；P 编号顺序与冻结 per-continuation 记录一致。

## 案例 A：rq3-openhands-main-g186-round4

选择规则：OpenHands 对 Marconi 最大绝对改善；同时也是对 StandaloneTopK 最大改善。预算 25%，K=4；候选 16，待续请求 4。

冻结来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:463`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_A.json](case_A.json)。

| 请求 | 原始 continuation ID | T | workflow（冻结编码） |
|---|---|---:|---|
| P1 | `RQ3_G186_A_PENDING_TURN_005` | 8832 | `nebius-swe-rebench-openhands::chatcmpl-e4535f7b0d208d13e2892a0dbb388496` |
| P2 | `RQ3_G186_B_PENDING_TURN_005` | 6144 | `nebius-swe-rebench-openhands::chatcmpl-9b28ea19360ff9a29e10ccd0805a847e` |
| P3 | `RQ3_G186_C_PENDING_TURN_005` | 15296 | `nebius-swe-rebench-openhands::chatcmpl-fe2e01aaeb9955f5968685c153778505` |
| P4 | `RQ3_G186_D_PENDING_TURN_005` | 4736 | `nebius-swe-rebench-openhands::chatcmpl-37aa034e029c1b2555b8288ead49d267` |

OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。

| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |
|---|---|---:|---|---|
| C1 | `RQ3_G186_A_TURN_001` | 4096 | P1 | 154.970, 0.000, 0.000, 0.000 |
| C2 | `RQ3_G186_A_TURN_002` | 4416 | P1 | 167.288, 0.000, 0.000, 0.000 |
| C3 | `RQ3_G186_A_TURN_003` | 6464 | P1 | 246.842, 0.000, 0.000, 0.000 |
| C4 | `RQ3_G186_A_TURN_004` | 8832 | P1 | 340.385, 0.000, 0.000, 0.000 |
| C5 | `RQ3_G186_B_TURN_001` | 2432 | P2 | 0.000, 91.201, 0.000, 0.000 |
| C6 | `RQ3_G186_B_TURN_002` | 5248 | P2 | 0.000, 199.004, 0.000, 0.000 |
| C7 | `RQ3_G186_B_TURN_003` | 6016 | P2 | 0.000, 228.815, 0.000, 0.000 |
| C8 | `RQ3_G186_B_TURN_004` | 6144 | P2 | 0.000, 233.801, 0.000, 0.000 |
| C9 | `RQ3_G186_C_TURN_001` | 14016 | P3 | 0.000, 0.000, 553.901, 0.000 |
| C10 | `RQ3_G186_C_TURN_002` | 14400 | P3 | 0.000, 0.000, 569.900, 0.000 |
| C11 | `RQ3_G186_C_TURN_003` | 15104 | P3 | 0.000, 0.000, 599.345, 0.000 |
| C12 | `RQ3_G186_C_TURN_004` | 15296 | P3 | 0.000, 0.000, 607.402, 0.000 |
| C13 | `RQ3_G186_D_TURN_001` | 3456 | P4 | 0.000, 0.000, 0.000, 129.973 |
| C14 | `RQ3_G186_D_TURN_002` | 3712 | P4 | 0.000, 0.000, 0.000, 139.742 |
| C15 | `RQ3_G186_D_TURN_003` | 4160 | P4 | 0.000, 0.000, 0.000, 156.886 |
| C16 | `RQ3_G186_D_TURN_004` | 4736 | P4 | 0.000, 0.000, 0.000, 179.015 |

| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |
|---|---|---:|---:|
| FlowState | C12, C4, C8, C16 | 0.000 | 0.000 |
| LFU | C13, C9, C5, C1 | 430.557 | 430.557 |
| LRU | C16, C15, C12, C11 | 574.186 | 574.186 |
| Marconi | C9, C16, C15, C11 | 582.242 | 582.242 |
| StandaloneTopK | C9, C10, C11, C12 | 753.200 | 753.200 |

StandaloneTopK 完整正收益静态排序：C12 → C11 → C10 → C9 → C4 → C3 → C8 → C7 → C6 → C16 → C2 → C15 → C1 → C14 → C13 → C5。

| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |
|---|---|---:|---|---|
| 1 | C12 | 607.402 | 0.000, 0.000, 0.000, 0.000 | 0.000, 0.000, 607.402, 0.000 |
| 2 | C4 | 340.385 | 0.000, 0.000, 607.402, 0.000 | 340.385, 0.000, 0.000, 0.000 |
| 3 | C8 | 233.801 | 340.385, 0.000, 607.402, 0.000 | 0.000, 233.801, 0.000, 0.000 |
| 4 | C16 | 179.015 | 340.385, 233.801, 607.402, 0.000 | 0.000, 0.000, 0.000, 179.015 |

首个分歧：第 2 步；共同 retained set = ['C12']。
- flowstate：C4；独立收益 340.385 ms；相对上述 retained set 的 marginal 340.385 ms。
- standalone：C11；独立收益 599.345 ms；相对上述 retained set 的 marginal 0.000 ms。

| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |
|---|---|---:|---:|---:|---:|
| P1 | FlowState | 8832 | 0 | 0.000 | 0.000 |
| P2 | FlowState | 6144 | 0 | 0.000 | 0.000 |
| P3 | FlowState | 15296 | 0 | 0.000 | 0.000 |
| P4 | FlowState | 4736 | 0 | 0.000 | 0.000 |
| P1 | LFU | 4096 | 4736 | 185.415 | 185.415 |
| P2 | LFU | 2432 | 3712 | 142.599 | 142.599 |
| P3 | LFU | 14016 | 1280 | 53.501 | 53.501 |
| P4 | LFU | 3456 | 1280 | 49.041 | 49.041 |
| P1 | LRU | 0 | 8832 | 340.385 | 340.385 |
| P2 | LRU | 0 | 6144 | 233.801 | 233.801 |
| P3 | LRU | 15296 | 0 | 0.000 | 0.000 |
| P4 | LRU | 4736 | 0 | 0.000 | 0.000 |
| P1 | Marconi | 0 | 8832 | 340.385 | 340.385 |
| P2 | Marconi | 0 | 6144 | 233.801 | 233.801 |
| P3 | Marconi | 15104 | 192 | 8.056 | 8.056 |
| P4 | Marconi | 4736 | 0 | 0.000 | 0.000 |
| P1 | StandaloneTopK | 0 | 8832 | 340.385 | 340.385 |
| P2 | StandaloneTopK | 0 | 6144 | 233.801 | 233.801 |
| P3 | StandaloneTopK | 15296 | 0 | 0.000 | 0.000 |
| P4 | StandaloneTopK | 0 | 4736 | 179.015 | 179.015 |

Marconi 价值信号：原始 last_access_order、marconi_flop_saved、alpha 和归一化 utility 均见本案例 JSON；以原始 utility 重新排序，选中集合核对通过。

Marconi 最终集合中可删除而不增加成本的 checkpoint：C9, C15。对每个 c，检查的是 Δ(c | S_Marconi \ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。

英文论文解释：In this OpenHands snapshot, StandaloneTopK spends all four slots on one workflow. After retaining its deepest checkpoint, FlowState assigns the remaining slots to three other workflows, reducing modeled recovery cost by 753.200 ms relative to StandaloneTopK and 582.242 ms relative to Marconi Adaptation. These are modeled allocation costs, not measured runtime latencies.

## 案例 B：rq3-openhands-main-g073-round3

选择规则：OpenHands 对 Marconi 严格改善排序的下中位案例。预算 25%，K=3；候选 12，待续请求 4。

冻结来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:184`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_B.json](case_B.json)。

| 请求 | 原始 continuation ID | T | workflow（冻结编码） |
|---|---|---:|---|
| P1 | `RQ3_G073_A_PENDING_TURN_004` | 4352 | `nebius-swe-rebench-openhands::chatcmpl-47252e2b3626e348bf795ecdba300eb6` |
| P2 | `RQ3_G073_B_PENDING_TURN_004` | 5056 | `nebius-swe-rebench-openhands::chatcmpl-14c2bbcf85bf22575ca5200f900c0167` |
| P3 | `RQ3_G073_C_PENDING_TURN_004` | 4800 | `nebius-swe-rebench-openhands::chatcmpl-e38652061c3298ebad56ad8202c5ea67` |
| P4 | `RQ3_G073_D_PENDING_TURN_004` | 4928 | `nebius-swe-rebench-openhands::chatcmpl-793609b8ba2e43854b96894fce4c46fe` |

OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。

| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |
|---|---|---:|---|---|
| C1 | `RQ3_G073_A_TURN_001` | 3392 | P1 | 127.492, 0.000, 0.000, 0.000 |
| C2 | `RQ3_G073_A_TURN_002` | 3712 | P1 | 139.697, 0.000, 0.000, 0.000 |
| C3 | `RQ3_G073_A_TURN_003` | 4352 | P1 | 164.197, 0.000, 0.000, 0.000 |
| C4 | `RQ3_G073_B_TURN_001` | 2432 | P2 | 0.000, 91.117, 0.000, 0.000 |
| C5 | `RQ3_G073_B_TURN_002` | 4480 | P2 | 0.000, 169.213, 0.000, 0.000 |
| C6 | `RQ3_G073_B_TURN_003` | 5056 | P2 | 0.000, 191.403, 0.000, 0.000 |
| C7 | `RQ3_G073_C_TURN_001` | 3776 | P3 | 0.000, 0.000, 142.196, 0.000 |
| C8 | `RQ3_G073_C_TURN_002` | 4160 | P3 | 0.000, 0.000, 156.894, 0.000 |
| C9 | `RQ3_G073_C_TURN_003` | 4800 | P3 | 0.000, 0.000, 181.489, 0.000 |
| C10 | `RQ3_G073_D_TURN_001` | 3520 | P4 | 0.000, 0.000, 0.000, 132.435 |
| C11 | `RQ3_G073_D_TURN_002` | 3904 | P4 | 0.000, 0.000, 0.000, 147.106 |
| C12 | `RQ3_G073_D_TURN_003` | 4928 | P4 | 0.000, 0.000, 0.000, 186.443 |

| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |
|---|---|---:|---:|
| FlowState | C6, C12, C9 | 164.197 | 0.000 |
| LFU | C10, C7, C4 | 357.785 | 193.588 |
| LRU | C12, C11, C9 | 355.600 | 191.403 |
| Marconi | C10, C7, C5 | 279.689 | 115.491 |
| StandaloneTopK | C6, C9, C12 | 164.197 | 0.000 |

StandaloneTopK 完整正收益静态排序：C6 → C12 → C9 → C5 → C3 → C8 → C11 → C7 → C2 → C10 → C1 → C4。

| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |
|---|---|---:|---|---|
| 1 | C6 | 191.403 | 0.000, 0.000, 0.000, 0.000 | 0.000, 191.403, 0.000, 0.000 |
| 2 | C12 | 186.443 | 0.000, 191.403, 0.000, 0.000 | 0.000, 0.000, 0.000, 186.443 |
| 3 | C9 | 181.489 | 0.000, 191.403, 0.000, 186.443 | 0.000, 0.000, 181.489, 0.000 |

FlowState 与 StandaloneTopK 选中集合相同，无集合分歧。

| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |
|---|---|---:|---:|---:|---:|
| P1 | FlowState | 0 | 4352 | 164.197 | 0.000 |
| P2 | FlowState | 5056 | 0 | 0.000 | 0.000 |
| P3 | FlowState | 4800 | 0 | 0.000 | 0.000 |
| P4 | FlowState | 4928 | 0 | 0.000 | 0.000 |
| P1 | LFU | 0 | 4352 | 164.197 | 0.000 |
| P2 | LFU | 2432 | 2624 | 100.286 | 100.286 |
| P3 | LFU | 3776 | 1024 | 39.294 | 39.294 |
| P4 | LFU | 3520 | 1408 | 54.008 | 54.008 |
| P1 | LRU | 0 | 4352 | 164.197 | 0.000 |
| P2 | LRU | 0 | 5056 | 191.403 | 191.403 |
| P3 | LRU | 4800 | 0 | 0.000 | 0.000 |
| P4 | LRU | 4928 | 0 | 0.000 | 0.000 |
| P1 | Marconi | 0 | 4352 | 164.197 | 0.000 |
| P2 | Marconi | 4480 | 576 | 22.190 | 22.190 |
| P3 | Marconi | 3776 | 1024 | 39.294 | 39.294 |
| P4 | Marconi | 3520 | 1408 | 54.008 | 54.008 |
| P1 | StandaloneTopK | 0 | 4352 | 164.197 | 0.000 |
| P2 | StandaloneTopK | 5056 | 0 | 0.000 | 0.000 |
| P3 | StandaloneTopK | 4800 | 0 | 0.000 | 0.000 |
| P4 | StandaloneTopK | 4928 | 0 | 0.000 | 0.000 |

Marconi 价值信号：原始 last_access_order、marconi_flop_saved、alpha 和归一化 utility 均见本案例 JSON；以原始 utility 重新排序，选中集合核对通过。

Marconi 最终集合中可删除而不增加成本的 checkpoint：无。对每个 c，检查的是 Δ(c | S_Marconi \ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。

英文论文解释：This median-positive Marconi comparison yields a 115.491 ms modeled-cost reduction, while FlowState and StandaloneTopK have identical objectives. The example therefore supports a difference in checkpoint valuation relative to Marconi, but does not demonstrate an additional benefit from set-dependent updates.

## 案例 C：rq3-openhands-main-g138-round4

选择规则：OpenHands 对 StandaloneTopK 严格改善排序的下中位案例。预算 25%，K=4；候选 16，待续请求 4。

冻结来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:346`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_C.json](case_C.json)。

| 请求 | 原始 continuation ID | T | workflow（冻结编码） |
|---|---|---:|---|
| P1 | `RQ3_G138_A_PENDING_TURN_005` | 4608 | `nebius-swe-rebench-openhands::chatcmpl-e1a1319c833eddf8d28c2a417eacbc9a` |
| P2 | `RQ3_G138_B_PENDING_TURN_005` | 4608 | `nebius-swe-rebench-openhands::chatcmpl-d5314ba7cfc4079f7d5c8091ed7591d3` |
| P3 | `RQ3_G138_C_PENDING_TURN_005` | 5312 | `nebius-swe-rebench-openhands::chatcmpl-0bf8c15690d42e1d11e95ae9fee056ae` |
| P4 | `RQ3_G138_D_PENDING_TURN_005` | 4672 | `nebius-swe-rebench-openhands::chatcmpl-a35a3bd225fdf0c10040229516f39807` |

OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。

| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |
|---|---|---:|---|---|
| C1 | `RQ3_G138_A_TURN_001` | 3520 | P1 | 132.399, 0.000, 0.000, 0.000 |
| C2 | `RQ3_G138_A_TURN_002` | 3904 | P1 | 147.066, 0.000, 0.000, 0.000 |
| C3 | `RQ3_G138_A_TURN_003` | 4480 | P1 | 169.149, 0.000, 0.000, 0.000 |
| C4 | `RQ3_G138_A_TURN_004` | 4608 | P1 | 174.070, 0.000, 0.000, 0.000 |
| C5 | `RQ3_G138_B_TURN_001` | 2432 | P2 | 0.000, 91.082, 0.000, 0.000 |
| C6 | `RQ3_G138_B_TURN_002` | 3904 | P2 | 0.000, 147.066, 0.000, 0.000 |
| C7 | `RQ3_G138_B_TURN_003` | 4032 | P2 | 0.000, 151.965, 0.000, 0.000 |
| C8 | `RQ3_G138_B_TURN_004` | 4608 | P2 | 0.000, 174.070, 0.000, 0.000 |
| C9 | `RQ3_G138_C_TURN_001` | 3904 | P3 | 0.000, 0.000, 147.154, 0.000 |
| C10 | `RQ3_G138_C_TURN_002` | 4288 | P3 | 0.000, 0.000, 161.874, 0.000 |
| C11 | `RQ3_G138_C_TURN_003` | 4800 | P3 | 0.000, 0.000, 181.568, 0.000 |
| C12 | `RQ3_G138_C_TURN_004` | 5312 | P3 | 0.000, 0.000, 201.340, 0.000 |
| C13 | `RQ3_G138_D_TURN_001` | 3392 | P4 | 0.000, 0.000, 0.000, 127.527 |
| C14 | `RQ3_G138_D_TURN_002` | 3776 | P4 | 0.000, 0.000, 0.000, 142.180 |
| C15 | `RQ3_G138_D_TURN_003` | 4416 | P4 | 0.000, 0.000, 0.000, 166.699 |
| C16 | `RQ3_G138_D_TURN_004` | 4672 | P4 | 0.000, 0.000, 0.000, 176.541 |

| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |
|---|---|---:|---:|
| FlowState | C12, C16, C4, C8 | 0.000 | 0.000 |
| LFU | C13, C9, C5, C1 | 227.859 | 227.859 |
| LRU | C16, C15, C12, C11 | 348.139 | 348.139 |
| Marconi | C9, C13, C15, C16 | 402.325 | 402.325 |
| StandaloneTopK | C4, C11, C12, C16 | 174.070 | 174.070 |

StandaloneTopK 完整正收益静态排序：C12 → C11 → C16 → C4 → C8 → C3 → C15 → C10 → C7 → C9 → C2 → C6 → C14 → C1 → C13 → C5。

| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |
|---|---|---:|---|---|
| 1 | C12 | 201.340 | 0.000, 0.000, 0.000, 0.000 | 0.000, 0.000, 201.340, 0.000 |
| 2 | C16 | 176.541 | 0.000, 0.000, 201.340, 0.000 | 0.000, 0.000, 0.000, 176.541 |
| 3 | C4 | 174.070 | 0.000, 0.000, 201.340, 176.541 | 174.070, 0.000, 0.000, 0.000 |
| 4 | C8 | 174.070 | 174.070, 0.000, 201.340, 176.541 | 0.000, 174.070, 0.000, 0.000 |

首个分歧：第 2 步；共同 retained set = ['C12']。
- flowstate：C16；独立收益 176.541 ms；相对上述 retained set 的 marginal 176.541 ms。
- standalone：C11；独立收益 181.568 ms；相对上述 retained set 的 marginal 0.000 ms。

| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |
|---|---|---:|---:|---:|---:|
| P1 | FlowState | 4608 | 0 | 0.000 | 0.000 |
| P2 | FlowState | 4608 | 0 | 0.000 | 0.000 |
| P3 | FlowState | 5312 | 0 | 0.000 | 0.000 |
| P4 | FlowState | 4672 | 0 | 0.000 | 0.000 |
| P1 | LFU | 3520 | 1088 | 41.670 | 41.670 |
| P2 | LFU | 2432 | 2176 | 82.988 | 82.988 |
| P3 | LFU | 3904 | 1408 | 54.186 | 54.186 |
| P4 | LFU | 3392 | 1280 | 49.014 | 49.014 |
| P1 | LRU | 0 | 4608 | 174.070 | 174.070 |
| P2 | LRU | 0 | 4608 | 174.070 | 174.070 |
| P3 | LRU | 5312 | 0 | 0.000 | 0.000 |
| P4 | LRU | 4672 | 0 | 0.000 | 0.000 |
| P1 | Marconi | 0 | 4608 | 174.070 | 174.070 |
| P2 | Marconi | 0 | 4608 | 174.070 | 174.070 |
| P3 | Marconi | 3904 | 1408 | 54.186 | 54.186 |
| P4 | Marconi | 4672 | 0 | 0.000 | 0.000 |
| P1 | StandaloneTopK | 4608 | 0 | 0.000 | 0.000 |
| P2 | StandaloneTopK | 0 | 4608 | 174.070 | 174.070 |
| P3 | StandaloneTopK | 5312 | 0 | 0.000 | 0.000 |
| P4 | StandaloneTopK | 4672 | 0 | 0.000 | 0.000 |

Marconi 价值信号：原始 last_access_order、marconi_flop_saved、alpha 和归一化 utility 均见本案例 JSON；以原始 utility 重新排序，选中集合核对通过。

Marconi 最终集合中可删除而不增加成本的 checkpoint：C13, C15。对每个 c，检查的是 Δ(c | S_Marconi \ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。

英文论文解释：At the first divergent selection, the statically preferred checkpoint has zero additional benefit relative to the retained deeper checkpoint. FlowState instead covers another pending workflow, reducing the final modeled recovery cost by 174.070 ms relative to StandaloneTopK.

## 案例 D：rq3-agentx-g006-epoch35

选择规则：AgentX 第一个严格优于 StandaloneTopK 的案例。预算 25%，K=8；候选 35，待续请求 2。

冻结来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:19`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_D.json](case_D.json)。

| 请求 | 原始 continuation ID | T | workflow（冻结编码） |
|---|---|---:|---|
| P1 | `63d3960fcc384388d5e7e4df8287cd1be081::aux:003::pending0000` | 125248 | `63d3960fcc384388d5e7e4df8287cd1be081` |
| P2 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::pending0008` | 125759 | `63d3960fcc384388d5e7e4df8287cd1be081` |

OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。

| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |
|---|---|---:|---|---|
| C1 | `63d3960fcc384388d5e7e4df8287cd1be081::aux:000::req0000` | 384 |  | 0.000, 0.000 |
| C2 | `63d3960fcc384388d5e7e4df8287cd1be081::aux:003::req0000` | 125248 |  | 0.000, 0.000 |
| C3 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:001::req0000` | 113984 |  | 0.000, 0.000 |
| C4 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:001::req0001` | 115968 |  | 0.000, 0.000 |
| C5 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:001::req0002` | 115520 |  | 0.000, 0.000 |
| C6 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:001::req0003` | 117504 |  | 0.000, 0.000 |
| C7 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0000` | 117440 |  | 0.000, 0.000 |
| C8 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0001` | 117888 |  | 0.000, 0.000 |
| C9 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0002` | 117440 |  | 0.000, 0.000 |
| C10 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0003` | 121408 |  | 0.000, 0.000 |
| C11 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0004` | 123648 |  | 0.000, 0.000 |
| C12 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0005` | 123264 |  | 0.000, 0.000 |
| C13 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0006` | 124160 |  | 0.000, 0.000 |
| C14 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0007` | 123648 |  | 0.000, 0.000 |
| C15 | `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0008` | 125760 |  | 0.000, 0.000 |
| C16 | `63d3960fcc384388d5e7e4df8287cd1be081::req0000` | 47168 | P1, P2 | 2263.017, 2263.789 |
| C17 | `63d3960fcc384388d5e7e4df8287cd1be081::req0001` | 48448 | P1, P2 | 2333.666, 2334.459 |
| C18 | `63d3960fcc384388d5e7e4df8287cd1be081::req0002` | 49728 | P2 | 0.000, 2405.618 |
| C19 | `63d3960fcc384388d5e7e4df8287cd1be081::req0003` | 85120 | P2 | 0.000, 4566.493 |
| C20 | `63d3960fcc384388d5e7e4df8287cd1be081::req0004` | 87104 | P2 | 0.000, 4698.674 |
| C21 | `63d3960fcc384388d5e7e4df8287cd1be081::req0005` | 89600 | P2 | 0.000, 4866.631 |
| C22 | `63d3960fcc384388d5e7e4df8287cd1be081::req0006` | 95488 | P2 | 0.000, 5270.192 |
| C23 | `63d3960fcc384388d5e7e4df8287cd1be081::req0007` | 97408 | P2 | 0.000, 5404.021 |
| C24 | `63d3960fcc384388d5e7e4df8287cd1be081::req0008` | 100288 | P2 | 0.000, 5606.824 |
| C25 | `63d3960fcc384388d5e7e4df8287cd1be081::req0009` | 101760 | P2 | 0.000, 5711.433 |
| C26 | `63d3960fcc384388d5e7e4df8287cd1be081::req0010` | 102720 | P2 | 0.000, 5780.004 |
| C27 | `63d3960fcc384388d5e7e4df8287cd1be081::req0011` | 103104 | P2 | 0.000, 5807.509 |
| C28 | `63d3960fcc384388d5e7e4df8287cd1be081::req0012` | 104064 |  | 0.000, 0.000 |
| C29 | `63d3960fcc384388d5e7e4df8287cd1be081::req0013` | 104448 |  | 0.000, 0.000 |
| C30 | `63d3960fcc384388d5e7e4df8287cd1be081::req0014` | 108224 |  | 0.000, 0.000 |
| C31 | `63d3960fcc384388d5e7e4df8287cd1be081::req0015` | 110144 |  | 0.000, 0.000 |
| C32 | `63d3960fcc384388d5e7e4df8287cd1be081::req0016` | 111360 |  | 0.000, 0.000 |
| C33 | `63d3960fcc384388d5e7e4df8287cd1be081::req0017` | 110784 |  | 0.000, 0.000 |
| C34 | `63d3960fcc384388d5e7e4df8287cd1be081::req0018` | 112704 |  | 0.000, 0.000 |
| C35 | `63d3960fcc384388d5e7e4df8287cd1be081::req0019` | 114368 |  | 0.000, 0.000 |

| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |
|---|---|---:|---:|
| FlowState | C27, C17 | 6832.743 | 0.000 |
| LFU | C16, C17, C18, C19, C20, C21, C22, C23 | 7236.232 | 403.489 |
| LRU | C2, C17, C16, C15, C14, C13, C12, C11 | 10305.793 | 3473.050 |
| Marconi | C16, C19, C2, C22, C10, C7, C9, C24 | 7104.078 | 271.335 |
| StandaloneTopK | C20, C21, C22, C23, C24, C25, C26, C27 | 9166.410 | 2333.666 |

StandaloneTopK 完整正收益静态排序：C27 → C26 → C25 → C24 → C23 → C22 → C21 → C20 → C17 → C19 → C16 → C18。

| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |
|---|---|---:|---|---|
| 1 | C27 | 5807.509 | 0.000, 0.000 | 0.000, 5807.509 |
| 2 | C17 | 2333.666 | 0.000, 5807.509 | 2333.666, 0.000 |

首个分歧：第 2 步；共同 retained set = ['C27']。
- flowstate：C17；独立收益 4668.125 ms；相对上述 retained set 的 marginal 2333.666 ms。
- standalone：C26；独立收益 5780.004 ms；相对上述 retained set 的 marginal 0.000 ms。

| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |
|---|---|---:|---:|---:|---:|
| P1 | FlowState | 48448 | 76800 | 5132.248 | 0.000 |
| P2 | FlowState | 103104 | 22655 | 1700.496 | 0.000 |
| P1 | LFU | 48448 | 76800 | 5132.248 | 0.000 |
| P2 | LFU | 97408 | 28351 | 2103.984 | 403.489 |
| P1 | LRU | 48448 | 76800 | 5132.248 | 0.000 |
| P2 | LRU | 48448 | 77311 | 5173.546 | 3473.050 |
| P1 | Marconi | 47168 | 78080 | 5202.897 | 70.649 |
| P2 | Marconi | 100288 | 25471 | 1901.181 | 200.686 |
| P1 | StandaloneTopK | 0 | 125248 | 7465.914 | 2333.666 |
| P2 | StandaloneTopK | 103104 | 22655 | 1700.496 | 0.000 |

Marconi 价值信号：UNKNOWN：正式逐案例文件未保留原始排序 metadata；仅核对冻结输出与共同成本，不臆测内部排序原因。

Marconi 最终集合中可删除而不增加成本的 checkpoint：C19, C2, C22, C10, C7, C9。对每个 c，检查的是 Δ(c | S_Marconi \ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。

英文论文解释：After retaining req0011, req0010 adds no recovery benefit. FlowState instead selects the shared ancestor req0001, whose incremental benefit at this step is confined to the other pending continuation. The resulting 2333.666 ms improvement supports redundancy-aware reallocation, not a claim that simultaneous gains on multiple branches caused this win.

## 案例 E：rq3-agentx-g018-epoch30

选择规则：AgentX 第二个严格优于 StandaloneTopK 的案例。预算 25%，K=7；候选 30，待续请求 2。

冻结来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:55`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_E.json](case_E.json)。

| 请求 | 原始 continuation ID | T | workflow（冻结编码） |
|---|---|---:|---|
| P1 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::aux:002::pending0000` | 126847 | `d77fa1e0115f7fca67ca88b95a053d3a41bc` |
| P2 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::pending0009` | 124287 | `d77fa1e0115f7fca67ca88b95a053d3a41bc` |

OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。

| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |
|---|---|---:|---|---|
| C1 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::aux:000::req0000` | 384 |  | 0.000, 0.000 |
| C2 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::aux:002::req0000` | 126848 |  | 0.000, 0.000 |
| C3 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0000` | 114368 |  | 0.000, 0.000 |
| C4 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0001` | 116160 |  | 0.000, 0.000 |
| C5 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0002` | 115648 |  | 0.000, 0.000 |
| C6 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0003` | 115648 |  | 0.000, 0.000 |
| C7 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0004` | 120640 |  | 0.000, 0.000 |
| C8 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0005` | 120064 |  | 0.000, 0.000 |
| C9 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0006` | 120064 |  | 0.000, 0.000 |
| C10 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0007` | 123008 |  | 0.000, 0.000 |
| C11 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0008` | 123840 |  | 0.000, 0.000 |
| C12 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0009` | 123328 |  | 0.000, 0.000 |
| C13 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0000` | 43840 | P1, P2 | 2083.857, 2080.264 |
| C14 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` | 47360 | P1, P2 | 2276.008, 2272.126 |
| C15 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0002` | 82624 | P2 | 0.000, 4397.974 |
| C16 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0003` | 89408 | P2 | 0.000, 4849.432 |
| C17 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0004` | 91584 | P2 | 0.000, 4997.143 |
| C18 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0005` | 93824 | P2 | 0.000, 5150.673 |
| C19 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0006` | 96384 | P2 | 0.000, 5327.966 |
| C20 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0007` | 97920 | P2 | 0.000, 5435.279 |
| C21 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0008` | 99904 | P2 | 0.000, 5574.932 |
| C22 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0009` | 100480 | P2 | 0.000, 5615.696 |
| C23 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0010` | 102144 | P2 | 0.000, 5734.015 |
| C24 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0011` | 101568 | P2 | 0.000, 5692.965 |
| C25 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0012` | 105344 | P2 | 0.000, 5963.868 |
| C26 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0013` | 104768 |  | 0.000, 0.000 |
| C27 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0014` | 107328 |  | 0.000, 0.000 |
| C28 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0015` | 107968 |  | 0.000, 0.000 |
| C29 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0016` | 106560 |  | 0.000, 0.000 |
| C30 | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0017` | 114944 |  | 0.000, 0.000 |

| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |
|---|---|---:|---:|
| FlowState | C25, C14 | 6745.075 | 0.000 |
| LFU | C13, C14, C15, C16, C17, C18, C19 | 7380.976 | 635.902 |
| LRU | C2, C14, C13, C12, C11, C10, C9 | 10436.817 | 3691.742 |
| Marconi | C13, C15, C2, C16, C3, C14, C8 | 7859.511 | 1114.436 |
| StandaloneTopK | C19, C20, C21, C22, C23, C24, C25 | 9021.082 | 2276.008 |

StandaloneTopK 完整正收益静态排序：C25 → C23 → C24 → C22 → C21 → C20 → C19 → C18 → C17 → C16 → C14 → C15 → C13。

| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |
|---|---|---:|---|---|
| 1 | C25 | 5963.868 | 0.000, 0.000 | 0.000, 5963.868 |
| 2 | C14 | 2276.008 | 0.000, 5963.868 | 2276.008, 0.000 |

首个分歧：第 2 步；共同 retained set = ['C25']。
- flowstate：C14；独立收益 4548.134 ms；相对上述 retained set 的 marginal 2276.008 ms。
- standalone：C23；独立收益 5734.015 ms；相对上述 retained set 的 marginal 0.000 ms。

| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |
|---|---|---:|---:|---:|---:|
| P1 | FlowState | 47360 | 79487 | 5321.929 | 0.000 |
| P2 | FlowState | 105344 | 18943 | 1423.145 | 0.000 |
| P1 | LFU | 47360 | 79487 | 5321.929 | 0.000 |
| P2 | LFU | 96384 | 27903 | 2059.047 | 635.902 |
| P1 | LRU | 47360 | 79487 | 5321.929 | 0.000 |
| P2 | LRU | 47360 | 76927 | 5114.887 | 3691.742 |
| P1 | Marconi | 47360 | 79487 | 5321.929 | 0.000 |
| P2 | Marconi | 89408 | 34879 | 2537.582 | 1114.436 |
| P1 | StandaloneTopK | 0 | 126847 | 7597.937 | 2276.008 |
| P2 | StandaloneTopK | 105344 | 18943 | 1423.145 | 0.000 |

Marconi 价值信号：UNKNOWN：正式逐案例文件未保留原始排序 metadata；仅核对冻结输出与共同成本，不臆测内部排序原因。

Marconi 最终集合中可删除而不增加成本的 checkpoint：C13, C15, C2, C3, C8。对每个 c，检查的是 Δ(c | S_Marconi \ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。

英文论文解释：The second strict AgentX case exhibits the same mechanism: after req0012, FlowState selects req0001 for a 2276.008 ms gain on the uncovered continuation, whereas the next static choice adds zero benefit. Although req0001 has positive standalone benefit for both continuations, only one contributes to its current marginal gain.

## 案例 F：rq3-agentx-g000-epoch109

选择规则：AgentX 不同集合但 objective 相同：按预算、ID 升序首例。预算 25%，K=27；候选 109，待续请求 3。

冻结来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:1`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_F.json](case_F.json)。

| 请求 | 原始 continuation ID | T | workflow（冻结编码） |
|---|---|---:|---|
| P1 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::pending0002` | 110527 | `02bc0afb13f7a2d9efa86c28511261d85c0e` |
| P2 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:001::pending0000` | 20223 | `02bc0afb13f7a2d9efa86c28511261d85c0e` |
| P3 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:002::pending0000` | 110527 | `02bc0afb13f7a2d9efa86c28511261d85c0e` |

OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。

| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |
|---|---|---:|---|---|
| C1 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0000` | 96064 |  | 0.000, 0.000, 0.000 |
| C2 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0001` | 106560 |  | 0.000, 0.000, 0.000 |
| C3 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0002` | 110528 |  | 0.000, 0.000, 0.000 |
| C4 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:001::req0000` | 20224 |  | 0.000, 0.000, 0.000 |
| C5 | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:002::req0000` | 110528 |  | 0.000, 0.000, 0.000 |
| C6 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` | 37760 | P1, P3 | 1740.926, 0.000, 1740.926 |
| C7 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0001` | 33024 | P1, P3 | 1499.274, 0.000, 1499.274 |
| C8 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0002` | 43520 | P1 | 2043.833, 0.000, 0.000 |
| C9 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0003` | 44800 | P1 | 2112.488, 0.000, 0.000 |
| C10 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0004` | 49664 | P1 | 2377.829, 0.000, 0.000 |
| C11 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` | 51328 | P1 | 2470.222, 0.000, 0.000 |
| C12 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0006` | 70080 |  | 0.000, 0.000, 0.000 |
| C13 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0007` | 76160 |  | 0.000, 0.000, 0.000 |
| C14 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0008` | 79680 |  | 0.000, 0.000, 0.000 |
| C15 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0009` | 83584 |  | 0.000, 0.000, 0.000 |
| C16 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0010` | 91520 |  | 0.000, 0.000, 0.000 |
| C17 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0011` | 94208 |  | 0.000, 0.000, 0.000 |
| C18 | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0012` | 96064 |  | 0.000, 0.000, 0.000 |
| C19 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0000` | 17536 |  | 0.000, 0.000, 0.000 |
| C20 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0001` | 17728 |  | 0.000, 0.000, 0.000 |
| C21 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0002` | 17856 |  | 0.000, 0.000, 0.000 |
| C22 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0003` | 18048 |  | 0.000, 0.000, 0.000 |
| C23 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0004` | 18816 |  | 0.000, 0.000, 0.000 |
| C24 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0005` | 19136 |  | 0.000, 0.000, 0.000 |
| C25 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0006` | 19264 |  | 0.000, 0.000, 0.000 |
| C26 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0007` | 24064 |  | 0.000, 0.000, 0.000 |
| C27 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0008` | 37056 |  | 0.000, 0.000, 0.000 |
| C28 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0009` | 44032 |  | 0.000, 0.000, 0.000 |
| C29 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0010` | 44224 |  | 0.000, 0.000, 0.000 |
| C30 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0011` | 44800 |  | 0.000, 0.000, 0.000 |
| C31 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0012` | 44928 |  | 0.000, 0.000, 0.000 |
| C32 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0013` | 52800 |  | 0.000, 0.000, 0.000 |
| C33 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0014` | 52992 |  | 0.000, 0.000, 0.000 |
| C34 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0015` | 54592 |  | 0.000, 0.000, 0.000 |
| C35 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0016` | 56640 |  | 0.000, 0.000, 0.000 |
| C36 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0017` | 56768 |  | 0.000, 0.000, 0.000 |
| C37 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0018` | 63424 |  | 0.000, 0.000, 0.000 |
| C38 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0019` | 65920 |  | 0.000, 0.000, 0.000 |
| C39 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0020` | 66048 |  | 0.000, 0.000, 0.000 |
| C40 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0021` | 69760 |  | 0.000, 0.000, 0.000 |
| C41 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0022` | 71104 |  | 0.000, 0.000, 0.000 |
| C42 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0023` | 71424 |  | 0.000, 0.000, 0.000 |
| C43 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0024` | 71872 |  | 0.000, 0.000, 0.000 |
| C44 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0025` | 72192 |  | 0.000, 0.000, 0.000 |
| C45 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0026` | 73536 |  | 0.000, 0.000, 0.000 |
| C46 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0027` | 73664 |  | 0.000, 0.000, 0.000 |
| C47 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0028` | 75136 |  | 0.000, 0.000, 0.000 |
| C48 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0029` | 75072 |  | 0.000, 0.000, 0.000 |
| C49 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0030` | 77440 |  | 0.000, 0.000, 0.000 |
| C50 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0031` | 77696 |  | 0.000, 0.000, 0.000 |
| C51 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0032` | 78400 |  | 0.000, 0.000, 0.000 |
| C52 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0000` | 16576 |  | 0.000, 0.000, 0.000 |
| C53 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0001` | 18560 |  | 0.000, 0.000, 0.000 |
| C54 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0002` | 18688 |  | 0.000, 0.000, 0.000 |
| C55 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0003` | 19200 |  | 0.000, 0.000, 0.000 |
| C56 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0004` | 19776 |  | 0.000, 0.000, 0.000 |
| C57 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0005` | 29760 |  | 0.000, 0.000, 0.000 |
| C58 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0006` | 29888 |  | 0.000, 0.000, 0.000 |
| C59 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0007` | 30528 |  | 0.000, 0.000, 0.000 |
| C60 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0008` | 39488 |  | 0.000, 0.000, 0.000 |
| C61 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0009` | 42432 |  | 0.000, 0.000, 0.000 |
| C62 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0010` | 43904 |  | 0.000, 0.000, 0.000 |
| C63 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0011` | 47488 |  | 0.000, 0.000, 0.000 |
| C64 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0012` | 53696 |  | 0.000, 0.000, 0.000 |
| C65 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0013` | 56064 |  | 0.000, 0.000, 0.000 |
| C66 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0014` | 57216 |  | 0.000, 0.000, 0.000 |
| C67 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0015` | 57792 |  | 0.000, 0.000, 0.000 |
| C68 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0016` | 58816 |  | 0.000, 0.000, 0.000 |
| C69 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0017` | 58944 |  | 0.000, 0.000, 0.000 |
| C70 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0018` | 59136 |  | 0.000, 0.000, 0.000 |
| C71 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0019` | 60032 |  | 0.000, 0.000, 0.000 |
| C72 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0020` | 61312 |  | 0.000, 0.000, 0.000 |
| C73 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0021` | 62272 |  | 0.000, 0.000, 0.000 |
| C74 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0022` | 63168 |  | 0.000, 0.000, 0.000 |
| C75 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0023` | 63680 |  | 0.000, 0.000, 0.000 |
| C76 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0000` | 17600 |  | 0.000, 0.000, 0.000 |
| C77 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0001` | 17920 |  | 0.000, 0.000, 0.000 |
| C78 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0002` | 18112 |  | 0.000, 0.000, 0.000 |
| C79 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0003` | 18496 |  | 0.000, 0.000, 0.000 |
| C80 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0004` | 18688 |  | 0.000, 0.000, 0.000 |
| C81 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0005` | 20224 |  | 0.000, 0.000, 0.000 |
| C82 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0006` | 20352 |  | 0.000, 0.000, 0.000 |
| C83 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0007` | 20800 |  | 0.000, 0.000, 0.000 |
| C84 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0008` | 21120 |  | 0.000, 0.000, 0.000 |
| C85 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0009` | 21248 |  | 0.000, 0.000, 0.000 |
| C86 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0010` | 21952 |  | 0.000, 0.000, 0.000 |
| C87 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0011` | 22400 |  | 0.000, 0.000, 0.000 |
| C88 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0012` | 22528 |  | 0.000, 0.000, 0.000 |
| C89 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0013` | 23104 |  | 0.000, 0.000, 0.000 |
| C90 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0014` | 23232 |  | 0.000, 0.000, 0.000 |
| C91 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0015` | 23424 |  | 0.000, 0.000, 0.000 |
| C92 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0016` | 23552 |  | 0.000, 0.000, 0.000 |
| C93 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0017` | 24384 |  | 0.000, 0.000, 0.000 |
| C94 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0018` | 25088 |  | 0.000, 0.000, 0.000 |
| C95 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0019` | 25536 |  | 0.000, 0.000, 0.000 |
| C96 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0020` | 27776 |  | 0.000, 0.000, 0.000 |
| C97 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0021` | 30080 |  | 0.000, 0.000, 0.000 |
| C98 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0022` | 49344 |  | 0.000, 0.000, 0.000 |
| C99 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0023` | 50048 |  | 0.000, 0.000, 0.000 |
| C100 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0024` | 50432 |  | 0.000, 0.000, 0.000 |
| C101 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0025` | 60352 |  | 0.000, 0.000, 0.000 |
| C102 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0026` | 60608 |  | 0.000, 0.000, 0.000 |
| C103 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0027` | 64256 |  | 0.000, 0.000, 0.000 |
| C104 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0028` | 64704 |  | 0.000, 0.000, 0.000 |
| C105 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0029` | 64960 |  | 0.000, 0.000, 0.000 |
| C106 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0030` | 65344 |  | 0.000, 0.000, 0.000 |
| C107 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0031` | 65920 |  | 0.000, 0.000, 0.000 |
| C108 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0032` | 67136 |  | 0.000, 0.000, 0.000 |
| C109 | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0033` | 67712 |  | 0.000, 0.000, 0.000 |

| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |
|---|---|---:|---:|
| FlowState | C6, C11 | 9197.819 | 0.000 |
| LFU | C76, C77, C19, C78, C20, C79, C21, C80, C22, C81, C23, C82, C24, C83, C25, C84, C26, C85, C27, C86, C28, C52, C87, C29, C53, C88, C30 | 13408.967 | 4211.148 |
| LRU | C5, C7, C6, C3, C2, C1, C11, C10, C9, C8, C18, C17, C16, C15, C14, C13, C12, C4, C109, C108, C107, C106, C105, C104, C103, C102, C101 | 9197.819 | 0.000 |
| Marconi | C7, C4, C12, C98, C5, C76, C2, C101, C16, C8, C13, C6, C10, C3, C15, C14, C17, C1, C11, C9, C18, C103, C97, C96, C81, C108, C93 | 9197.819 | 0.000 |
| StandaloneTopK | C6, C7, C8, C9, C10, C11 | 9197.819 | 0.000 |

StandaloneTopK 完整正收益静态排序：C6 → C7 → C11 → C10 → C9 → C8。

| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |
|---|---|---:|---|---|
| 1 | C6 | 3481.852 | 0.000, 0.000, 0.000 | 1740.926, 0.000, 1740.926 |
| 2 | C11 | 729.296 | 1740.926, 0.000, 1740.926 | 729.296, 0.000, 0.000 |

首个分歧：第 2 步；共同 retained set = ['C6']。
- flowstate：C11；独立收益 2470.222 ms；相对上述 retained set 的 marginal 729.296 ms。
- standalone：C7；独立收益 2998.549 ms；相对上述 retained set 的 marginal 0.000 ms。

| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |
|---|---|---:|---:|---:|---:|
| P1 | FlowState | 51328 | 59199 | 3823.719 | 0.000 |
| P2 | FlowState | 0 | 20223 | 821.085 | 0.000 |
| P3 | FlowState | 37760 | 72767 | 4553.015 | 0.000 |
| P1 | LFU | 0 | 110527 | 6293.941 | 2470.222 |
| P2 | LFU | 0 | 20223 | 821.085 | 0.000 |
| P3 | LFU | 0 | 110527 | 6293.941 | 1740.926 |
| P1 | LRU | 51328 | 59199 | 3823.719 | 0.000 |
| P2 | LRU | 0 | 20223 | 821.085 | 0.000 |
| P3 | LRU | 37760 | 72767 | 4553.015 | 0.000 |
| P1 | Marconi | 51328 | 59199 | 3823.719 | 0.000 |
| P2 | Marconi | 0 | 20223 | 821.085 | 0.000 |
| P3 | Marconi | 37760 | 72767 | 4553.015 | 0.000 |
| P1 | StandaloneTopK | 51328 | 59199 | 3823.719 | 0.000 |
| P2 | StandaloneTopK | 0 | 20223 | 821.085 | 0.000 |
| P3 | StandaloneTopK | 37760 | 72767 | 4553.015 | 0.000 |

Marconi 价值信号：UNKNOWN：正式逐案例文件未保留原始排序 metadata；仅核对冻结输出与共同成本，不臆测内部排序原因。

Marconi 最终集合中可删除而不增加成本的 checkpoint：C7, C4, C12, C98, C5, C76, C2, C101, C16, C8, C13, C10, C3, C15, C14, C17, C1, C9, C18, C103, C97, C96, C81, C108, C93。对每个 c，检查的是 Δ(c | S_Marconi \ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。

英文论文解释：A shared checkpoint contributes 1740.926 ms to each of two continuations and is preferred over an alternative with a larger individual but smaller aggregate benefit. Nevertheless, FlowState and StandaloneTopK attain identical executable frontiers and the same final objective; the additional static selections are redundant.

## 案例 G：rq3-openhands-main-g009-round3

选择规则：OpenHands 对 Marconi 持平：按预算、ID 升序首例。预算 25%，K=3；候选 12，待续请求 4。

冻结来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:25`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_G.json](case_G.json)。

| 请求 | 原始 continuation ID | T | workflow（冻结编码） |
|---|---|---:|---|
| P1 | `RQ3_G009_A_PENDING_TURN_004` | 4736 | `nebius-swe-rebench-openhands::chatcmpl-e209c7e9d6440355698862640d50e9e1` |
| P2 | `RQ3_G009_B_PENDING_TURN_004` | 10944 | `nebius-swe-rebench-openhands::chatcmpl-235e79f6893e33a60c715360ac72ed54` |
| P3 | `RQ3_G009_C_PENDING_TURN_004` | 5568 | `nebius-swe-rebench-openhands::chatcmpl-12590702ff8556a3f2ab216c18d634f6` |
| P4 | `RQ3_G009_D_PENDING_TURN_004` | 5824 | `nebius-swe-rebench-openhands::chatcmpl-9ae47790dcbbb6b84840855699f9c9d2` |

OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。

| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |
|---|---|---:|---|---|
| C1 | `RQ3_G009_A_TURN_001` | 3712 | P1 | 139.742, 0.000, 0.000, 0.000 |
| C2 | `RQ3_G009_A_TURN_002` | 4096 | P1 | 154.433, 0.000, 0.000, 0.000 |
| C3 | `RQ3_G009_A_TURN_003` | 4736 | P1 | 179.015, 0.000, 0.000, 0.000 |
| C4 | `RQ3_G009_B_TURN_001` | 2432 | P2 | 0.000, 91.575, 0.000, 0.000 |
| C5 | `RQ3_G009_B_TURN_002` | 4352 | P2 | 0.000, 165.116, 0.000, 0.000 |
| C6 | `RQ3_G009_B_TURN_003` | 10944 | P2 | 0.000, 425.965, 0.000, 0.000 |
| C7 | `RQ3_G009_C_TURN_001` | 3520 | P3 | 0.000, 0.000, 132.507, 0.000 |
| C8 | `RQ3_G009_C_TURN_002` | 3840 | P3 | 0.000, 0.000, 144.737, 0.000 |
| C9 | `RQ3_G009_C_TURN_003` | 5568 | P3 | 0.000, 0.000, 211.301, 0.000 |
| C10 | `RQ3_G009_D_TURN_001` | 3520 | P4 | 0.000, 0.000, 0.000, 132.536 |
| C11 | `RQ3_G009_D_TURN_002` | 3968 | P4 | 0.000, 0.000, 0.000, 149.669 |
| C12 | `RQ3_G009_D_TURN_003` | 5824 | P4 | 0.000, 0.000, 0.000, 221.286 |

| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |
|---|---|---:|---:|
| FlowState | C6, C12, C9 | 179.015 | 0.000 |
| LFU | C10, C7, C4 | 680.948 | 501.934 |
| LRU | C12, C11, C9 | 604.979 | 425.965 |
| Marconi | C6, C12, C9 | 179.015 | 0.000 |
| StandaloneTopK | C6, C9, C12 | 179.015 | 0.000 |

StandaloneTopK 完整正收益静态排序：C6 → C12 → C9 → C3 → C5 → C2 → C11 → C8 → C1 → C10 → C7 → C4。

| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |
|---|---|---:|---|---|
| 1 | C6 | 425.965 | 0.000, 0.000, 0.000, 0.000 | 0.000, 425.965, 0.000, 0.000 |
| 2 | C12 | 221.286 | 0.000, 425.965, 0.000, 0.000 | 0.000, 0.000, 0.000, 221.286 |
| 3 | C9 | 211.301 | 0.000, 425.965, 0.000, 221.286 | 0.000, 0.000, 211.301, 0.000 |

FlowState 与 StandaloneTopK 选中集合相同，无集合分歧。

| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |
|---|---|---:|---:|---:|---:|
| P1 | FlowState | 0 | 4736 | 179.015 | 0.000 |
| P2 | FlowState | 10944 | 0 | 0.000 | 0.000 |
| P3 | FlowState | 5568 | 0 | 0.000 | 0.000 |
| P4 | FlowState | 5824 | 0 | 0.000 | 0.000 |
| P1 | LFU | 0 | 4736 | 179.015 | 0.000 |
| P2 | LFU | 2432 | 8512 | 334.390 | 334.390 |
| P3 | LFU | 3520 | 2048 | 78.794 | 78.794 |
| P4 | LFU | 3520 | 2304 | 88.750 | 88.750 |
| P1 | LRU | 0 | 4736 | 179.015 | 0.000 |
| P2 | LRU | 0 | 10944 | 425.965 | 425.965 |
| P3 | LRU | 5568 | 0 | 0.000 | 0.000 |
| P4 | LRU | 5824 | 0 | 0.000 | 0.000 |
| P1 | Marconi | 0 | 4736 | 179.015 | 0.000 |
| P2 | Marconi | 10944 | 0 | 0.000 | 0.000 |
| P3 | Marconi | 5568 | 0 | 0.000 | 0.000 |
| P4 | Marconi | 5824 | 0 | 0.000 | 0.000 |
| P1 | StandaloneTopK | 0 | 4736 | 179.015 | 0.000 |
| P2 | StandaloneTopK | 10944 | 0 | 0.000 | 0.000 |
| P3 | StandaloneTopK | 5568 | 0 | 0.000 | 0.000 |
| P4 | StandaloneTopK | 5824 | 0 | 0.000 | 0.000 |

Marconi 价值信号：原始 last_access_order、marconi_flop_saved、alpha 和归一化 utility 均见本案例 JSON；以原始 utility 重新排序，选中集合核对通过。

Marconi 最终集合中可删除而不增加成本的 checkpoint：无。对每个 c，检查的是 Δ(c | S_Marconi \ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。

英文论文解释：FlowState and Marconi Adaptation attain the same modeled recovery objective in this snapshot. Different checkpoint choices should not be interpreted as a performance improvement when their per-continuation executable frontiers coincide.

