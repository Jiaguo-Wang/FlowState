# 受控多工作流 v1 公平比较协议

## 冻结策略

本阶段比较四个策略臂。所有策略共享同一组等大小、当前仍驻留的循环状态候选与 `K=3` 全局预算。策略只输出保留的检查点标识，运行时动作统一交给 `StateController` 和 `SGLangAdapter.evict_mamba_only()`；任何策略都不得调用 SGLang 原生级联逐出。

- FlowState：使用现有 `GlobalOptimizer`，按集合相关的边际恢复收益选择。
- Global-LRU：按显式 `last_access_order` 从新到旧全局保留三个检查点；相同时依次用 `creation_order` 和检查点标识确定顺序。
- Equal-Share：按公开且与检查点价值无关的 `W1 → W2 → W3 → W4` 顺序轮转分配。每轮每个工作流至多获得一个位置，工作流内部优先选择当前最深的兼容检查点。
- Recovery-Only：为每个候选计算它对任一单独待续分支的最大恢复成本下降，以该固定分数排序。它不累计 fanout，也不在选入其他候选后重算覆盖价值。

场景中的检查点时序与运行时构建顺序一致：依次为 `W1_SHALLOW`、`W1_PARENT`、`W2_PARENT`、`W3_PARENT`、`W4_PARENT`。本版本没有额外访问，因此创建顺序与最近访问顺序相同。该元数据只属于 evaluation 层，不进入核心 `CheckpointCandidate`。

当前冻结场景在 `K=3` 下的离线选择如下：

| 策略 | 所选检查点标识 | 规划总恢复间隔 | 估计恢复成本 |
|---|---|---:|---:|
| FlowState | `W1_PARENT`、`W2_PARENT`、`W4_PARENT` | 8192 | 378.589 ms |
| Global-LRU | `W4_PARENT`、`W3_PARENT`、`W2_PARENT` | 65536 | 2999.016 ms |
| Equal-Share | `W1_PARENT`、`W2_PARENT`、`W3_PARENT` | 12288 | 536.187 ms |
| Recovery-Only | `W1_PARENT`、`W2_PARENT`、`W1_SHALLOW` | 20480 | 914.776 ms |

这些结果由冻结规则和场景元数据直接计算，策略实现中不包含面向预期结果的标识特判。当前三个基线都没有与 FlowState 得到相同选择；若后续规则或元数据改变，必须通过新版本场景重新冻结，不能静默改写本协议。

## 固定快照隔离评估

第一阶段正式比较采用 Snapshot-Isolated Evaluation，用于验证 FlowState 的 executable-state allocation objective 在固定 decision epoch 下是否优于 baseline。固定时点 `t` 的 pending set、candidate set 与预算保持不变，唯一变化是策略给出的 selected set。

每个策略与每个待续请求组成一个独立 case。每个 case 必须严格执行：

1. 使用全新进程或明确清空缓存。
2. 重建完全相同的五个检查点。
3. 验证五个候选均满足 `FA=True` 且 `Mamba=True`。
4. 执行当前策略的 `K=3` 分配，并记录选择与决策耗时。
5. 通过统一 controller 和 adapter 执行逐出。
6. 验证 selected、evicted、FA allocator、radix tree、path 与 `sanity_check()`。
7. 只发送当前 case 对应的一个待续请求，并记录 H、E 与 G。
8. 丢弃当前 runtime state，下一 case 重新从初始状态构建。

因此，每个 `policy × continuation` 都从等价的 post-build、pre-allocation 状态独立开始。四个策略和七个待续请求组成 `4 × 7 = 28` 个 isolated cases。case 列表由 `build_snapshot_cases()` 统一生成，runtime harness 不得散落手写组合。

禁止在一个策略臂中连续发送七个请求后，把所得七个 gap 直接解释为 snapshot objective。前面的请求会自然创建或更新循环状态，使后续请求不再面对原始 selected set。Step 7B 的连续请求只属于 runtime feasibility 证据，不构成本阶段的固定快照比较。

## 顺序、预热与重复

正式性能实验必须包含多个 repetition，并在 repetition 之间交替或随机化策略顺序。所有策略使用相同预热方法、请求内容、生成参数和计时边界。

第一轮 runtime correctness comparison 是上述 28 个独立 case，每个 case 只运行一次。该轮只能验证状态语义和采集链路，不能据此报告统计性能结论。Step 7B 中 W1-A 的 533 ms 只是 feasibility gate 的单次观测值，不得进入 baseline 性能结论。正式性能结论必须控制 warmup 与 policy order，并保留每次 repetition 的原始记录。

## 指标

主要指标：

- 总 recovery gap token 数。
- 每请求平均 recovery gap。
- Executable Prefix Ratio，即 `sum(E) / sum(H)`。
- 估计恢复成本，即 `sum Phi(G)`。

次要指标：

- 请求端到端时延。
- 能够可靠取得时的 TTFT P50 与 P95。
- 工作流完成时间。

每次运行还必须记录：

- 所选检查点标识。
- 每个工作流的 recovery gap。
- FA allocator invariant。
- 策略决策耗时。

任何缺失的 TTFT 必须标为不可用，不能用请求端到端时延替代。

## 在线顺序评估边界

Snapshot-Isolated Evaluation 不是在线 workflow completion benchmark。未来的 Online Sequential Evaluation 会允许请求执行后产生新检查点；每个 arrival 或 completion 都形成新的 decision epoch，系统必须重新获取 pending set、candidate set 与 residency snapshot，并重新执行 reconcile。

在线顺序评估需要单独定义状态演化、到达顺序、重新决策时点和完成时间指标，本步骤不实现该流程，也不把固定快照结果外推为在线完成时间结论。
