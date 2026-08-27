# TraceLab 正式离线策略比较

本目录只使用 Step 10C.5 冻结的 105 个 C128 AND X>=2 快照及其 37 个 X>=4 次级切片。数据库以只读方式按冻结 trigger event 重建；没有重新采样、没有未来字段、没有 GPU 或 SGLang runtime 调用。

所有策略选择后的恢复成本统一由 `position_aware_quadratic_v1` 计算。Global-LRU、KVFlow-style 与 Marconi-style 的选择过程不读取该模型；FlowState 使用既有 `GlobalOptimizer` 的 set-dependent marginal recovery reduction。

## 主结果

| Budget | Policy | Mean total cost/snapshot (ms) | Gap tokens | EHR |
|---:|---|---:|---:|---:|
| 25% | Global-LRU | 4099.607753 | 8737530 | 0.396947 |
| 25% | KVFlow-style | 4091.103473 | 8715694 | 0.400763 |
| 25% | Marconi-style | 4580.144787 | 9719919 | 0.110687 |
| 25% | FlowState | 2940.443448 | 6601817 | 0.400763 |
| 50% | Global-LRU | 3689.894452 | 7825605 | 0.473282 |
| 50% | KVFlow-style | 3681.390172 | 7803769 | 0.477099 |
| 50% | Marconi-style | 4266.521400 | 8985796 | 0.122137 |
| 50% | FlowState | 2540.251031 | 5692028 | 0.480916 |
| 75% | Global-LRU | 3383.440345 | 7140158 | 0.530534 |
| 75% | KVFlow-style | 3334.538991 | 7028482 | 0.541985 |
| 75% | Marconi-style | 3883.573951 | 8127929 | 0.148855 |
| 75% | FlowState | 2143.801725 | 4789443 | 0.564885 |
| 100% | Global-LRU | 1972.574795 | 4107745 | 0.744275 |
| 100% | KVFlow-style | 1723.862558 | 3590524 | 0.770992 |
| 100% | Marconi-style | 2701.850765 | 5584853 | 0.351145 |
| 100% | FlowState | 0.000000 | 0 | 1.000000 |

## FlowState 与 Marconi 的成对结果

| Budget | Reduction | Win | Tie | Loss | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| 25% | 35.800% | 95 | 10 | 0 | [29.229%, 42.780%] |
| 50% | 40.461% | 96 | 9 | 0 | [34.518%, 46.277%] |
| 75% | 44.798% | 99 | 6 | 0 | [38.953%, 50.437%] |
| 100% | 100.000% | 99 | 6 | 0 | [100.000%, 100.000%] |

## 解释边界

主集合是预注册的结构覆盖型分层样本，对每个 snapshot 等权；它不是完整 C128 自然事件频率的概率估计。X>=4 次级切片 provider/concurrency 分布偏斜，只用于高竞争描述。TraceLab 不提供显式 LLM-level DAG、token IDs 或真实 runtime residency，本结果是 leakage-free 逻辑 checkpoint snapshot 上的正式离线比较。

Correctness gate：**PASS**；future-information violations：0。
