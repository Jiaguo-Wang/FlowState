# 真实案例与顺序分歧

七例按机制覆盖选取，全部属于 objective ties；没有挑选不存在的正例。以下所有值均为 modeled ms。完整候选、兼容矩阵、收益及全部逐步 q/r 见 per_case_results.jsonl。

## 案例 1：rq3-openhands-main-g000-round2，K=2

无共享但预算真正受限的典型 tie。

来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:1`；新结果 `per_case_results.jsonl:1`。

预算结构：{'effective_pending': 4, 'minimum_checkpoints_for_all_frontiers': 4, 'K_binds_best_frontiers': True, 'all_candidate_frontiers': [4352, 3968, 4160, 4032], 'all_candidate_cost_ms': 0.0, 'positive_checkpoint_count': 8}

| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |
|---|---|---:|
| FlowState | `RQ3_G000_A_TURN_002` → `RQ3_G000_C_TURN_002` | 301.324181 |
| PCB | `RQ3_G000_A_TURN_002` → `RQ3_G000_C_TURN_002` | 301.324181 |
| LDR | `RQ3_G000_A_TURN_002` → `RQ3_G000_C_TURN_002` | 301.324181 |
| WFC | `RQ3_G000_A_TURN_002` → `RQ3_G000_C_TURN_002` | 301.324181 |
| StandaloneTopK | `RQ3_G000_A_TURN_002` → `RQ3_G000_C_TURN_002` | 301.324181 |
| LRU | `RQ3_G000_D_TURN_002` → `RQ3_G000_D_TURN_001` | 470.439871 |
| LFU | `RQ3_G000_D_TURN_001` → `RQ3_G000_C_TURN_001` | 347.881773 |
| Marconi | `RQ3_G000_D_TURN_001` → `RQ3_G000_C_TURN_001` | 347.881773 |

FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：

| pending 原始 ID | T | E | G | C_p, modeled ms |
|---|---:|---:|---:|---:|
| `RQ3_G000_A_PENDING_TURN_003` | 4352 | 4352 | 0 | 0.000000 |
| `RQ3_G000_B_PENDING_TURN_003` | 3968 | 0 | 3968 | 149.433630 |
| `RQ3_G000_C_PENDING_TURN_003` | 4160 | 4160 | 0 | 0.000000 |
| `RQ3_G000_D_PENDING_TURN_003` | 4032 | 0 | 4032 | 151.890551 |

PCB 与 FlowState 顺序也相同。

LDR 与 FlowState 顺序也相同。

WFC 与 FlowState 顺序也相同。

所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。

## 案例 2：rq3-openhands-main-g051-round5，K=5

位置感知收益改变顺序但未改变最终集合。

来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:133`；新结果 `per_case_results.jsonl:133`。

预算结构：{'effective_pending': 4, 'minimum_checkpoints_for_all_frontiers': 4, 'K_binds_best_frontiers': False, 'all_candidate_frontiers': [11200, 8320, 7808, 7808], 'all_candidate_cost_ms': 7.554497620495425, 'positive_checkpoint_count': 19}

| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |
|---|---|---:|
| FlowState | `RQ3_G051_A_TURN_005` → `RQ3_G051_B_TURN_005` → `RQ3_G051_D_TURN_004` → `RQ3_G051_C_TURN_005` | 7.554498 |
| PCB | `RQ3_G051_A_TURN_005` → `RQ3_G051_B_TURN_005` → `RQ3_G051_D_TURN_004` → `RQ3_G051_C_TURN_005` | 7.554498 |
| LDR | `RQ3_G051_A_TURN_005` → `RQ3_G051_B_TURN_005` → `RQ3_G051_C_TURN_005` → `RQ3_G051_D_TURN_004` | 7.554498 |
| WFC | `RQ3_G051_A_TURN_005` → `RQ3_G051_B_TURN_005` → `RQ3_G051_C_TURN_005` → `RQ3_G051_D_TURN_004` | 7.554498 |
| StandaloneTopK | `RQ3_G051_A_TURN_005` → `RQ3_G051_B_TURN_005` → `RQ3_G051_C_TURN_004` → `RQ3_G051_C_TURN_005` → `RQ3_G051_D_TURN_004` | 7.554498 |
| LRU | `RQ3_G051_D_TURN_005` → `RQ3_G051_D_TURN_004` → `RQ3_G051_C_TURN_005` → `RQ3_G051_C_TURN_004` → `RQ3_G051_B_TURN_005` | 444.002132 |
| LFU | `RQ3_G051_D_TURN_001` → `RQ3_G051_C_TURN_001` → `RQ3_G051_B_TURN_001` → `RQ3_G051_A_TURN_001` → `RQ3_G051_D_TURN_002` | 843.393641 |
| Marconi | `RQ3_G051_A_TURN_005` → `RQ3_G051_D_TURN_004` → `RQ3_G051_C_TURN_004` → `RQ3_G051_B_TURN_004` → `RQ3_G051_B_TURN_005` | 42.842833 |

FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：

| pending 原始 ID | T | E | G | C_p, modeled ms |
|---|---:|---:|---:|---:|
| `RQ3_G051_A_PENDING_TURN_006` | 11200 | 11200 | 0 | 0.000000 |
| `RQ3_G051_B_PENDING_TURN_006` | 8320 | 8320 | 0 | 0.000000 |
| `RQ3_G051_C_PENDING_TURN_006` | 7808 | 7808 | 0 | 0.000000 |
| `RQ3_G051_D_PENDING_TURN_006` | 7999 | 7808 | 191 | 7.554498 |

PCB 与 FlowState 顺序也相同。

LDR 与 FlowState 首次顺序分歧：第 3 步；共同前缀 `['RQ3_G051_A_TURN_005', 'RQ3_G051_B_TURN_005']`。

| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |
|---|---|---:|---:|---:|
| FlowState | `RQ3_G051_D_TURN_004` | 1 | 7808 | 299.520865 |
| LDR | `RQ3_G051_C_TURN_005` | 1 | 7808 | 299.473120 |

WFC 与 FlowState 首次顺序分歧：第 3 步；共同前缀 `['RQ3_G051_A_TURN_005', 'RQ3_G051_B_TURN_005']`。

| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |
|---|---|---:|---:|---:|
| FlowState | `RQ3_G051_D_TURN_004` | 1 | 7808 | 299.520865 |
| WFC | `RQ3_G051_C_TURN_005` | 1 | 7808 | 299.473120 |

所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。

## 案例 3：rq3-agentx-g000-epoch109，K=27

共享总收益与逐请求推荐改变顺序。

来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:1`；新结果 `per_case_results.jsonl:505`。

预算结构：{'effective_pending': 2, 'minimum_checkpoints_for_all_frontiers': 2, 'K_binds_best_frontiers': False, 'all_candidate_frontiers': [51328, 0, 37760], 'all_candidate_cost_ms': 9197.818797961245, 'positive_checkpoint_count': 6}

| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |
|---|---|---:|
| FlowState | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` | 9197.818798 |
| PCB | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` | 9197.818798 |
| LDR | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` | 9197.818798 |
| WFC | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` | 9197.818798 |
| StandaloneTopK | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0002` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0003` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0004` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` | 9197.818798 |
| LRU | `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:002::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0002` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0004` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0003` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0002` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0012` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0011` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0010` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0009` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0008` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0007` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0006` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:001::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0033` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0032` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0031` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0030` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0029` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0028` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0027` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0026` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0025` | 9197.818798 |
| LFU | `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0002` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0003` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0002` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0004` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0003` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0005` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0004` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0006` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0005` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0007` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0006` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0008` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0007` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0009` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0008` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0010` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0009` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0011` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0010` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_002_5cfd1090::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0012` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_001_5b12db49::req0011` | 13408.966835 |
| Marconi | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:001::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0006` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0022` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:002::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0001` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0025` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0010` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0002` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0007` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0004` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0002` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0009` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0008` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0011` → `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::req0000` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0003` → `02bc0afb13f7a2d9efa86c28511261d85c0e::req0012` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0027` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0021` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0020` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0005` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0032` → `02bc0afb13f7a2d9efa86c28511261d85c0e::sa:subagent_003_e7ee775f::req0017` | 9197.818798 |

FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：

| pending 原始 ID | T | E | G | C_p, modeled ms |
|---|---:|---:|---:|---:|
| `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:000::pending0002` | 110527 | 51328 | 59199 | 3823.719222 |
| `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:001::pending0000` | 20223 | 0 | 20223 | 821.084730 |
| `02bc0afb13f7a2d9efa86c28511261d85c0e::fa:002::pending0000` | 110527 | 37760 | 72767 | 4553.014846 |

PCB 与 FlowState 首次顺序分歧：第 1 步；共同前缀 `[]`。

| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |
|---|---|---:|---:|---:|
| FlowState | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` | 2 | 75520 | 3481.852412 |
| PCB | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` | 1 | 51328 | 2470.221831 |

LDR 与 FlowState 首次顺序分歧：第 1 步；共同前缀 `[]`。

| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |
|---|---|---:|---:|---:|
| FlowState | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0000` | 2 | 75520 | 3481.852412 |
| LDR | `02bc0afb13f7a2d9efa86c28511261d85c0e::req0005` | 1 | 51328 | 2470.221831 |

WFC 与 FlowState 顺序也相同。

所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。

## 案例 4：rq3-agentx-g005-epoch8，K=2

WFC 覆盖数量优先选取很浅共享项。

来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:16`；新结果 `per_case_results.jsonl:520`。

预算结构：{'effective_pending': 2, 'minimum_checkpoints_for_all_frontiers': 2, 'K_binds_best_frontiers': False, 'all_candidate_frontiers': [43840, 384], 'all_candidate_cost_ms': 605.8028459126812, 'positive_checkpoint_count': 2}

| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |
|---|---|---:|
| FlowState | `61962299108912449a53f411ed3cf2d1c013::req0001` → `61962299108912449a53f411ed3cf2d1c013::req0000` | 605.802846 |
| PCB | `61962299108912449a53f411ed3cf2d1c013::req0001` → `61962299108912449a53f411ed3cf2d1c013::req0000` | 605.802846 |
| LDR | `61962299108912449a53f411ed3cf2d1c013::req0001` → `61962299108912449a53f411ed3cf2d1c013::req0000` | 605.802846 |
| WFC | `61962299108912449a53f411ed3cf2d1c013::req0000` → `61962299108912449a53f411ed3cf2d1c013::req0001` | 605.802846 |
| StandaloneTopK | `61962299108912449a53f411ed3cf2d1c013::req0000` → `61962299108912449a53f411ed3cf2d1c013::req0001` | 605.802846 |
| LRU | `61962299108912449a53f411ed3cf2d1c013::sa:subagent_001_97adff5d::req0000` → `61962299108912449a53f411ed3cf2d1c013::req0000` | 2573.277764 |
| LFU | `61962299108912449a53f411ed3cf2d1c013::req0000` → `61962299108912449a53f411ed3cf2d1c013::fa:000::req0000` | 2573.277764 |
| Marconi | `61962299108912449a53f411ed3cf2d1c013::sa:subagent_001_97adff5d::req0000` → `61962299108912449a53f411ed3cf2d1c013::req0000` | 2573.277764 |

FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：

| pending 原始 ID | T | E | G | C_p, modeled ms |
|---|---:|---:|---:|---:|
| `61962299108912449a53f411ed3cf2d1c013::fa:000::pending0004` | 54527 | 43840 | 10687 | 570.050532 |
| `61962299108912449a53f411ed3cf2d1c013::sa:subagent_001_97adff5d::pending0000` | 1344 | 384 | 960 | 35.752314 |

PCB 与 FlowState 顺序也相同。

LDR 与 FlowState 顺序也相同。

WFC 与 FlowState 首次顺序分歧：第 1 步；共同前缀 `[]`。

| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |
|---|---|---:|---:|---:|
| FlowState | `61962299108912449a53f411ed3cf2d1c013::req0001` | 1 | 43840 | 1982.352785 |
| WFC | `61962299108912449a53f411ed3cf2d1c013::req0000` | 2 | 768 | 29.101912 |

所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。

## 案例 5：rq3-agentx-g006-epoch35，K=8

原 StandaloneTopK 严格胜例之一。

来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:19`；新结果 `per_case_results.jsonl:523`。

预算结构：{'effective_pending': 2, 'minimum_checkpoints_for_all_frontiers': 2, 'K_binds_best_frontiers': False, 'all_candidate_frontiers': [48448, 103104], 'all_candidate_cost_ms': 6832.743154524837, 'positive_checkpoint_count': 12}

| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |
|---|---|---:|
| FlowState | `63d3960fcc384388d5e7e4df8287cd1be081::req0011` → `63d3960fcc384388d5e7e4df8287cd1be081::req0001` | 6832.743155 |
| PCB | `63d3960fcc384388d5e7e4df8287cd1be081::req0011` → `63d3960fcc384388d5e7e4df8287cd1be081::req0001` | 6832.743155 |
| LDR | `63d3960fcc384388d5e7e4df8287cd1be081::req0011` → `63d3960fcc384388d5e7e4df8287cd1be081::req0001` | 6832.743155 |
| WFC | `63d3960fcc384388d5e7e4df8287cd1be081::req0001` → `63d3960fcc384388d5e7e4df8287cd1be081::req0011` | 6832.743155 |
| StandaloneTopK | `63d3960fcc384388d5e7e4df8287cd1be081::req0004` → `63d3960fcc384388d5e7e4df8287cd1be081::req0005` → `63d3960fcc384388d5e7e4df8287cd1be081::req0006` → `63d3960fcc384388d5e7e4df8287cd1be081::req0007` → `63d3960fcc384388d5e7e4df8287cd1be081::req0008` → `63d3960fcc384388d5e7e4df8287cd1be081::req0009` → `63d3960fcc384388d5e7e4df8287cd1be081::req0010` → `63d3960fcc384388d5e7e4df8287cd1be081::req0011` | 9166.409584 |
| LRU | `63d3960fcc384388d5e7e4df8287cd1be081::aux:003::req0000` → `63d3960fcc384388d5e7e4df8287cd1be081::req0001` → `63d3960fcc384388d5e7e4df8287cd1be081::req0000` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0008` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0007` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0006` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0005` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0004` | 10305.793264 |
| LFU | `63d3960fcc384388d5e7e4df8287cd1be081::req0000` → `63d3960fcc384388d5e7e4df8287cd1be081::req0001` → `63d3960fcc384388d5e7e4df8287cd1be081::req0002` → `63d3960fcc384388d5e7e4df8287cd1be081::req0003` → `63d3960fcc384388d5e7e4df8287cd1be081::req0004` → `63d3960fcc384388d5e7e4df8287cd1be081::req0005` → `63d3960fcc384388d5e7e4df8287cd1be081::req0006` → `63d3960fcc384388d5e7e4df8287cd1be081::req0007` | 7236.231656 |
| Marconi | `63d3960fcc384388d5e7e4df8287cd1be081::req0000` → `63d3960fcc384388d5e7e4df8287cd1be081::req0003` → `63d3960fcc384388d5e7e4df8287cd1be081::aux:003::req0000` → `63d3960fcc384388d5e7e4df8287cd1be081::req0006` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0003` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0000` → `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::req0002` → `63d3960fcc384388d5e7e4df8287cd1be081::req0008` | 7104.078206 |

FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：

| pending 原始 ID | T | E | G | C_p, modeled ms |
|---|---:|---:|---:|---:|
| `63d3960fcc384388d5e7e4df8287cd1be081::aux:003::pending0000` | 125248 | 48448 | 76800 | 5132.247644 |
| `63d3960fcc384388d5e7e4df8287cd1be081::fa:002::pending0008` | 125759 | 103104 | 22655 | 1700.495510 |

PCB 与 FlowState 顺序也相同。

LDR 与 FlowState 顺序也相同。

WFC 与 FlowState 首次顺序分歧：第 1 步；共同前缀 `[]`。

| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |
|---|---|---:|---:|---:|
| FlowState | `63d3960fcc384388d5e7e4df8287cd1be081::req0011` | 1 | 103104 | 5807.509135 |
| WFC | `63d3960fcc384388d5e7e4df8287cd1be081::req0001` | 2 | 96896 | 4668.125455 |

所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。

## 案例 6：rq3-agentx-g018-epoch30，K=7

原 StandaloneTopK 严格胜例之二。

来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:55`；新结果 `per_case_results.jsonl:559`。

预算结构：{'effective_pending': 2, 'minimum_checkpoints_for_all_frontiers': 2, 'K_binds_best_frontiers': False, 'all_candidate_frontiers': [47360, 105344], 'all_candidate_cost_ms': 6745.074698150545, 'positive_checkpoint_count': 13}

| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |
|---|---|---:|
| FlowState | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0012` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` | 6745.074698 |
| PCB | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0012` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` | 6745.074698 |
| LDR | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0012` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` | 6745.074698 |
| WFC | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0012` | 6745.074698 |
| StandaloneTopK | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0006` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0007` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0008` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0009` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0010` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0011` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0012` | 9021.082426 |
| LRU | `d77fa1e0115f7fca67ca88b95a053d3a41bc::aux:002::req0000` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0000` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0009` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0008` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0007` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0006` | 10436.816564 |
| LFU | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0000` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0002` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0003` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0004` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0005` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0006` | 7380.976495 |
| Marconi | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0000` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0002` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::aux:002::req0000` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0003` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0000` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` → `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::req0005` | 7859.511012 |

FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：

| pending 原始 ID | T | E | G | C_p, modeled ms |
|---|---:|---:|---:|---:|
| `d77fa1e0115f7fca67ca88b95a053d3a41bc::aux:002::pending0000` | 126847 | 47360 | 79487 | 5321.929271 |
| `d77fa1e0115f7fca67ca88b95a053d3a41bc::fa:001::pending0009` | 124287 | 105344 | 18943 | 1423.145427 |

PCB 与 FlowState 顺序也相同。

LDR 与 FlowState 顺序也相同。

WFC 与 FlowState 首次顺序分歧：第 1 步；共同前缀 `[]`。

| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |
|---|---|---:|---:|---:|
| FlowState | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0012` | 1 | 105344 | 5963.868027 |
| WFC | `d77fa1e0115f7fca67ca88b95a053d3a41bc::req0001` | 2 | 94720 | 4548.133889 |

所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。

## 案例 7：rq3-agentx-g012-epoch11，K=2

无任何正收益候选，四种方法都提前停止。

来源：`/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:37`；新结果 `per_case_results.jsonl:541`。

预算结构：{'effective_pending': 0, 'minimum_checkpoints_for_all_frontiers': 0, 'K_binds_best_frontiers': False, 'all_candidate_frontiers': [0, 0, 0, 0], 'all_candidate_cost_ms': 7396.436137209052, 'positive_checkpoint_count': 0}

| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |
|---|---|---:|
| FlowState |  | 7396.436137 |
| PCB |  | 7396.436137 |
| LDR |  | 7396.436137 |
| WFC |  | 7396.436137 |
| StandaloneTopK |  | 7396.436137 |
| LRU | `8ab0a0892b0035b681a85c71d603caeb9e0f::sa:subagent_002_784ba65b::req0001` → `8ab0a0892b0035b681a85c71d603caeb9e0f::sa:subagent_002_784ba65b::req0000` | 7396.436137 |
| LFU | `8ab0a0892b0035b681a85c71d603caeb9e0f::req0000` → `8ab0a0892b0035b681a85c71d603caeb9e0f::req0001` | 7396.436137 |
| Marconi | `8ab0a0892b0035b681a85c71d603caeb9e0f::req0000` → `8ab0a0892b0035b681a85c71d603caeb9e0f::sa:subagent_002_784ba65b::req0000` | 7396.436137 |

FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：

| pending 原始 ID | T | E | G | C_p, modeled ms |
|---|---:|---:|---:|---:|
| `8ab0a0892b0035b681a85c71d603caeb9e0f::pending0003` | 51840 | 0 | 51840 | 2401.415553 |
| `8ab0a0892b0035b681a85c71d603caeb9e0f::sa:subagent_001_a5d8b82b::pending0002` | 41727 | 0 | 41727 | 1856.573709 |
| `8ab0a0892b0035b681a85c71d603caeb9e0f::sa:subagent_002_784ba65b::pending0001` | 39039 | 0 | 39039 | 1717.984042 |
| `8ab0a0892b0035b681a85c71d603caeb9e0f::sa:subagent_003_0d0d665c::pending0000` | 33088 | 0 | 33088 | 1420.462833 |

PCB 与 FlowState 顺序也相同。

LDR 与 FlowState 顺序也相同。

WFC 与 FlowState 顺序也相同。

所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。

