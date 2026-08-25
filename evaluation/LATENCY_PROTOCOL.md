# SOTA latency benchmark 冻结协议

## 目标与边界

Step 8E 已验证固定快照下的 planning recovery gap 与真实 runtime recovery gap 严格一致。Step 9 用相同 workload、相同 recurrent-state budget 和相同 mutation 路径，测量 Global-LRU、KVFlow-style、Marconi-style 与 FlowState 的 TTFT、完整请求 latency 和 recovery-related latency。

Step 9A 只冻结协议、case 计划、正确性门禁和统计方法，不启动图形处理器，也不产生性能结论。正式结果不得使用 Step 8E 的诊断计时代替重复测量。

## 冻结代表点

正式比较只包含以下四个点：

- Scalable N16，K=4
- Scalable N16，K=12
- SOTA-signal，K=4
- SOTA-signal，K=8

每个点只比较 Global-LRU、KVFlow-style、Marconi-style 与 FlowState。Oracle 不执行 GPU 请求，只在最终表格旁报告 FlowState 是否达到冻结的 offline Oracle objective。

## 等价类与 workload 权重

测量对象沿用 Step 8E 的 69 个 recovery-behavior equivalence class。每个类由以下键定义：

```text
(planning_target, planning_executable_frontier, planning_gap_tokens)
```

分组作用域是 `scenario × K × policy`，代表 continuation 使用字典序最小的 `continuation_id`。`class_multiplicity` 是该类覆盖的原始 logical continuation 数量。69 个 multiplicity 的总和必须为 800。

策略级 workload latency 必须用 `class_multiplicity` 加权。禁止把 69 个代表请求视为 69 个等权 workload 请求。

## Fresh-snapshot 测量生命周期

每个 `代表点 × policy-specific equivalence class × repetition` 都独立执行：

1. fresh 或 flush runtime。
2. 重建完全相同的全部 candidate checkpoint。
3. 调用冻结 policy 得到 selected IDs。
4. 经 `StateController` 和正式 `SGLangAdapter.evict_mamba_only()` reconcile。
5. 验证 FA、Mamba、allocator、tree/path 与 `sanity_check()`。
6. 同步 GPU，并确认 scheduler 位于安全时点。
7. 在 checkpoint build 和 reconcile 结束后记录 measurement start。
8. 只发送当前 representative continuation。
9. 通过流式首 token 到达时间记录 TTFT，通过请求开始至完成的客户端单调时钟记录 request latency。
10. 读取真实 runtime frontier 和 recovery gap，并执行严格正确性门禁。
11. 记录 measurement end，随后丢弃 snapshot。

`snapshot_build_ms` 与 `reconcile_ms` 单独记录。二者不得进入 TTFT 或 request latency。若 runtime metadata 未提供可信 TTFT，正式 harness 必须使用流式首 token 的客户端时间戳；禁止用完整 request latency 冒充 TTFT。

## 重复次数与执行顺序

- `warmup_repetitions = 2`
- `measured_repetitions = 10`
- `policy_order_seed = 20260825`

由于 Step 8E 的 equivalence class 在 policy 作用域内定义，不同 policy 的 class 数量可以不同。正式顺序以每个 `scenario × K × repetition` 的 policy block 为平衡单位；block 内的 equivalence class 按冻结键和 representative ID 排序。

四个 policy block 使用固定 seed 的循环 Latin rotation。代表点序号同时参与 rotation，使四个代表点合并后，每个 policy 在 measured 阶段的四个执行位置各出现 10 次，在 warmup 阶段各出现 2 次。禁止每次随机 shuffle。

## 正确性门禁

每个 warmup 和 measured 请求均执行相同正确性检查：

- `planning_target == runtime_fa_frontier`
- `planning_executable_frontier == runtime_executable_frontier`
- `planning_gap_tokens == runtime_gap_tokens`
- `runtime_gap == runtime_fa_frontier - runtime_executable_frontier`
- FA state 全部保留
- selected Mamba state 驻留，unselected Mamba state 已逐出
- FA allocator 不变
- node、prefix、tree/path 不变
- `sanity_check()` 通过
- 未调用 whole-node cascade eviction

任一条件失败时，该 case 标记失败并立即停止正式运行。失败 case 不得进入 latency 统计。

## 原始记录

每个 measured case 至少保存：

- `scenario`、`K`、`policy`
- `continuation_id`、`equivalence_class`、`class_multiplicity`
- `repetition`、`execution_order_position`
- `planning_target`、`planning_frontier`、`planning_gap`
- `runtime_fa_frontier`、`runtime_frontier`、`runtime_gap`
- `ttft_ms`、`request_latency_ms`
- `snapshot_build_ms`、`reconcile_ms`
- `correctness_pass`

同时保留每个 class 的全部原始 repetition 样本，不得只保存聚合均值。

## 统计方法

warmup 样本不进入统计。对每个 `scenario × K × policy`，以 `class_multiplicity` 为每条 repetition 样本的权重，分别对 TTFT 和 request latency 计算：

- weighted mean
- weighted median
- weighted P95

weighted median 和 weighted P95 使用加权经验分布的逆函数定义。随后对三个统计量分别计算相对同一代表点 Global-LRU 的 reduction：

```text
(Global-LRU - policy) / Global-LRU
```

单次 case 的时间只作为原始样本。正式论文结论必须来自全部 10 次 measured repetition，并保留 execution order 信息。

## Dry-run 规模与时长门禁

69 个 class 对应：

- warmup cases：`69 × 2 = 138`
- measured cases：`69 × 10 = 690`
- fresh snapshots 总数：828

时长估计使用 Step 8E 单进程完成 85 个 snapshot 生命周期的总时间作为吞吐参考，并乘以 1.25 的保守系数。该估计只用于运行排期，不是 latency 结果。若估计超过 6 小时，必须停止并报告，不得自行减少 repetition。
