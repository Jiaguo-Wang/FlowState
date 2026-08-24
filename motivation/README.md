# FlowState Motivation

本目录冻结 FlowState 论文 Motivation 阶段的证据链。它研究的问题是：在 hybrid attention/recurrent 模型中，FA-KV 的物理命中并不总能直接转化为可执行前缀；当 recurrent checkpoint 容量有限时，checkpoint 的内存成本、缺失后的恢复成本，以及由 workflow 决定的未来价值必须同时考虑。

这里记录的是 Motivation、runtime validation 和 controlled experiments，**不表示完整的 FlowState 系统、在线策略或生产实现已经完成**。

## Core frontiers

为统一描述各实验，定义三个前缀量：

- **Physical Prefix Frontier `H`**：当前请求在物理缓存中能够命中的最深 FA-KV 前缀位置。
- **Executable State Frontier `E`**：具有兼容 recurrent state、能够作为实际恢复与继续执行起点的最深前缀位置。在本文研究的 SGLang 路径中，最终可复用的请求前缀会被 recurrent checkpoint 边界限制到 `E`。
- **Recovery Gap `G`**：物理命中但不能直接执行的区间长度：

```text
G = H - E
```

当 `H > E` 时，区间 `[E, H)` 虽已有 FA-KV，仍必须从 recurrent state frontier 开始按普通 EXTEND/re-prefill 路径重新执行。具体请求的 scheduled EXTEND 还可能包含新输入 token 或边界对齐余量；`G` 描述的是其中由 physical/executable frontier 错位产生的恢复区间。

这三个概念把 Motivation 串成一条完整链条：

```text
有限 recurrent-state 容量
        ↓
无法保留所有 checkpoint（WP1）
        ↓
checkpoint 缺失使 E 回退，而 H 可保持不变
        ↓
产生 Recovery Gap G，并形成真实恢复开销（runtime validation + WP2）
        ↓
不同 workflow boundary 对未来 E 的保护价值不同（WP3）
```

## WP1 — Recurrent-state capacity

WP1 回答“保留 recurrent checkpoint 需要付出多少容量成本”。在冻结的 Qwen3.5-9B、SGLang v0.5.17 配置下：

- 一个普通 recurrent state/checkpoint slot 的逻辑 payload 为 **51,511,296 bytes = 49.125 MiB**。
- native layout 为 FP32 temporal state `48 MiB` 加 BF16 conv state `1.125 MiB`。
- 64、128、256 个普通 checkpoints 分别对应 `3.0703125`、`6.140625`、`12.28125 GiB` 的 checkpoint payload。
- int8 checkpoint 的源码与 runtime-verified shape/dtype 静态计算结果为 **13.5 MiB**；本阶段没有运行 int8 serving benchmark。

因此，recurrent checkpoint 是一个有限、可量化、由 active state、temporary state 和 retained checkpoints 共同竞争的容量资源。WP1 支撑的是 **checkpoint memory cost**，不单独证明任何具体 retention、offload 或 quantization policy 的端到端收益。

主要证据：[WP1 findings](artifacts/checkpoint_size_20260819/findings.md)。

## WP2 — Replay/recovery cost

WP2 回答“当 `E` 落后于 `H` 时，Recovery Gap 的代价有多大”。runtime/instrumentation validation 先确认：当深 recurrent checkpoint 被逐出而 FA-KV 仍驻留时，请求确实从较浅的 `E` 开始完整 EXTEND/re-prefill，gap 中已有的 FA-KV 不能避免重算。

在固定 `H=32768` tokens 的正式 sweep 中，将实际 replay 从 0 增加到 32K，使服务端 recovery latency/TTFT 中位数从 **32.374 ms** 增至 **1531.882 ms**。30 个正式 runs 的描述性拟合为：

```text
RecoveryLatency_ms = 29.727 + 0.0459129 × ActualReplayTokens
R² = 0.999282
```

即在该模型、硬件、并发度与 warmed steady-state 范围内，actual replay token count 是很强的 recovery-cost proxy。WP2 支撑的是 **recovery cost**；该关系不是跨模型、硬件、并发度和 workload 的无条件定律。

主要证据：[runtime validation](artifacts/runtime_validation_gap_replay_20260819/findings.md) 与 [WP2 findings](artifacts/replay_cost_20260819/findings.md)。前者是对 `H > E` 执行语义的 runtime/instrumentation validation，不是独立的新 research contribution。

## WP3 — Workflow-dependent checkpoint value

WP3 回答“在容量相同的 checkpoints 中，哪些状态更值得为未来保留”。受控的 `32K Parent → 4 Children → Join → Resume` workflow 表明，相同大小的 boundary checkpoints 具有显著不同的 future value：

- `FORK_PARENT@32768` 被 4 个 children 和 Join 实际选择，缺失时会暴露长 recovery gap。
- 四个 `NORMAL@32832` checkpoints 虽被创建，但在该 workflow horizon 内没有成为后续 consumer 的 executable frontier。
- `JOIN@33024` 被 Resume 选择，但其相对最近同路径 fallback 的保护区间仅为 256 tokens。

直接逐出实验进一步给出 controlled causal evidence：保留 Fork Parent 时 replay 为 0、TTFT 中位数为 **38.332 ms**；只删除对应 recurrent checkpoint、保持 FA-KV resident 后，真实 replay 为 32768 tokens、TTFT 中位数为 **1507.858 ms**，5 个 paired effects 的平均 penalty 为 **1472.748 ms**。删除 off-path NORMAL checkpoint 没有增加 Join replay。

当前冻结的 K=4 controlled formal result 进一步比较了两种实验性保留选择：Prompt-LRU 平均 TTFT 为 **1415.020 ms**，Workflow-K 平均 TTFT 为 **38.887 ms**，平均 paired reduction 为 **1376.133 ms（97.25%）**，平均 gap reduction 为 32768 tokens。

这些结果支撑 **workflow-dependent future value**：checkpoint 的价值不仅由大小或最近访问时间决定，还取决于它是否处在未来 workflow 路径的可执行祖先位置。它们仍然是 controlled evidence，不等同于已经实现或验证了完整、通用、最优的 FlowState policy。

主要证据：[workflow boundary characterization](artifacts/workflow_boundary_value_20260819/findings.md)、[direct eviction validation](artifacts/wp3a_direct_eviction_20260819/findings.md) 与 [frozen K=4 formal result](artifacts/wp3b_gate_20260820/formal_k4_20260821_102355/formal_k4_summary.md)。

## Evidence map and scope

- SGLang checkpoint、eviction、restore 与 replay 路径见 [recurrent-state code-path study](docs/sglang_recurrent_state_path.md)。
- 实验目录、研究问题和结果状态见 [artifacts index](artifacts/README.md)。
- Motivation 冻结边界和后续实验规则见 [FROZEN_MOTIVATION.md](FROZEN_MOTIVATION.md)。
- 所有数值均以对应冻结 findings、formal summary 和 raw artifacts 为准；本索引只组织已有证据，不替代原始结果。

