# Motivation Artifacts Index

本目录保存 FlowState Motivation 阶段的实验、runtime validation、分析结果和冻结报告。下表只索引已有证据；各目录中的 raw artifacts、findings 和 formal summary 保持原有口径。

特别说明：

- `runtime_validation_gap_replay_20260819` 是对 physical/executable frontier 错位及 replay 路径的 **runtime/instrumentation validation**，用于支撑 WP2 的系统语义前提，**不是单独的新 research contribution**。
- `wp3b_gate_20260820/formal_k4_20260821_102355` 是截至 2026-08-24 的 **current frozen controlled formal result**；它是受控实验结果，不代表完整 FlowState 系统已经实现。

| Experiment | Research question | Directory | Main result | Formal / validation | Status |
|---|---|---|---|---|---|
| WP1 — Recurrent checkpoint size | 一个 active/retained recurrent state 占用多少 HBM，checkpoint 容量是否构成有限资源？ | [`checkpoint_size_20260819/`](checkpoint_size_20260819/) | 普通 checkpoint payload 为 **49.125 MiB**；int8 checkpoint 静态计算为 **13.5 MiB**；64/128/256 个普通 checkpoints 为 3.0703125/6.140625/12.28125 GiB。 | Runtime tensor probe、源码 layout 推导与逐 byte 交叉验证；int8 未运行 serving benchmark。 | **Frozen — WP1 capacity evidence** |
| Runtime gap/replay validation | 当 FA-KV 物理命中 frontier `H` 深于 executable recurrent frontier `E` 时，gap 是否真实走完整 EXTEND/re-prefill？ | [`runtime_validation_gap_replay_20260819/`](runtime_validation_gap_replay_20260819/) | R2 实测 `H=32767`、`E=16384`、`G=16383`，scheduled EXTEND=16384，duplicate KV freed=16320；后续 forced-track 自愈至 checkpoint@32704。 | **Runtime/instrumentation validation，PASS 4/4；不是独立 research contribution。** | **Frozen — supporting runtime validation** |
| WP2 — Replay cost characterization | 在固定 physical FA-KV hit 下，Recovery Gap/replay token 数如何影响 recovery latency？ | [`replay_cost_20260819/`](replay_cost_20260819/) | 0K→32K replay 的 TTFT 中位数为 32.374→1531.882 ms；OLS slope **0.0459129 ms/token = 47.015 ms/Ki token**，`R²=0.999282`。 | 30/30 正式 runs 通过 runtime 硬门；另有正式 sweep 前的 8K gate。 | **Frozen — WP2 formal evidence** |
| WP3 — Workflow boundary value characterization | 相同大小的 checkpoints 是否因 workflow 位置不同而具有不同 future reuse/recovery value？ | [`workflow_boundary_value_20260819/`](workflow_boundary_value_20260819/) | Fork Parent exclusive reuse=5、single-miss avoided replay=32768；4 个 NORMAL checkpoints reuse=0；Join avoided replay=256。对应 saving 使用 WP2 slope 估算。 | 受控 `Parent → 4 Children → Join → Resume` trace；7/7 input dependency reconstruction，未做直接 eviction latency ablation。 | **Frozen — WP3 controlled characterization** |
| WP3A — Direct eviction causal validation | 只删除目标 recurrent checkpoint、保持 FA-KV resident，是否会因 boundary 类型不同产生不同 replay/latency 后果？ | [`wp3a_direct_eviction_20260819/`](wp3a_direct_eviction_20260819/) | 删除 Fork Parent 导致 32768-token replay，paired mean penalty **1472.748 ms**；删除 off-path NORMAL 没有增加 Join replay。 | 20/20 valid episodes；每个 target 5 个 paired repetitions，结构、allocator occupancy 与 Full component 均受控。 | **Frozen — WP3 controlled causal evidence** |
| WP3B — K=4 controlled policy comparison | 在相同 K=4 recurrent-state budget 下，保留 pending Fork Parents 与保留最近 child endpoints 的恢复结果有何差异？ | [`wp3b_gate_20260820/formal_k4_20260821_102355/`](wp3b_gate_20260820/formal_k4_20260821_102355/) | Prompt-LRU / Workflow-K mean TTFT = **1415.020 / 38.887 ms**；paired reduction **1376.133 ms（97.25%）**；mean gap reduction=32768 tokens。 | 10/10 complete/state-valid arms，10/10 expected-path arms，n=5 paired repetitions。 | **Current frozen controlled formal result** |

补充的 SGLang recurrent-state 源码路径调查位于 [`../docs/sglang_recurrent_state_path.md`](../docs/sglang_recurrent_state_path.md)。Motivation 的统一论文叙事和 `H/E/G` 定义位于 [`../README.md`](../README.md)，冻结规则位于 [`../FROZEN_MOTIVATION.md`](../FROZEN_MOTIVATION.md)。

