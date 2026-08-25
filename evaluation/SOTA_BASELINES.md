# SOTA-style 基线策略

本文档冻结两个面向相同 recurrent checkpoint 候选集的策略级适配。两者与 FlowState 后续共享相同的待续请求、候选状态、内存预算、`StateController`、`SGLangAdapter` 和 Mamba-only mutation。这里比较的是已有状态的保留选择，不是原系统的完整复现。

## KVFlow-style

实现名称为 `KVFlow-style adaptation`。

原机制通过 Agent Step Graph 为未来 Agent invocation 提供 steps-to-execution。数值越小表示越接近执行，对应 KV 节点越值得保留；一个共享节点关联多个未来 invocation 时，使用其中最小的 steps-to-execution。

本适配要求调用方为每个活动 `PendingContinuation` 显式提供 `continuation_id -> steps_to_execution`，并为每个 eligible checkpoint 提供与 Global-LRU 完全相同的 `last_access_by_checkpoint`。策略使用核心 `is_compatible()` 找到候选覆盖的待续请求，并以最小 steps-to-execution 作为候选优先级；没有 future dependency 的候选优先级为正无穷。排序首先按 priority 从小到大，再按 last-access 从新到旧，最终才按 `checkpoint_id` 决定完全平局。workflow priority 始终是第一关键字，因此一个很新的无依赖候选不能压过 priority 有限的候选。

该适配不从 token depth、fanout、workflow 标识或 recovery cost 推导未来执行距离，也不将兼容请求数量作为 score。

未复现的 KVFlow 部分包括：

- Agent Step Graph runtime engine 与 Agent framework integration
- CPU cache 与 CPU 到 GPU proactive prefetch
- status-aware scheduling
- KVFlow 自身的 KV-node eviction runtime
- varying suffix 特殊处理
- SGLang v0.4.4 修改与前后端 transport

## Marconi-style

实现名称为 `Marconi-style FLOP-aware eviction adaptation`。

Marconi 面向 Hybrid LLM prefix caching，同时包含 judicious admission、FLOP-aware eviction 和 KV 与 SSM holistic management。其 eviction utility 为：

```text
S(n) = recency(n) + alpha * flop_efficiency(n)
flop_efficiency(n) = FLOPs_saved(n) / memory(n)
```

本适配固定所有策略的 candidate set，只使用 FLOP-aware eviction utility 完成相同 recurrent checkpoint 数量预算下的 retention selection。调用方必须显式提供每个 eligible checkpoint 的 `last_access`、`flop_saved` 和非负 `alpha`。

`flop_saved` 表示 checkpoint 或 radix node 自身命中带来的 incremental FLOPs saved。若候选存在父 checkpoint，子候选的数值只能表示在父节点 reuse 之上额外避免的计算，不能将父路径和子路径的累计 savings 无条件重复计入。具体 metadata 构造将在后续步骤单独冻结；策略本身不会根据 token position、recovery gap、Phi 或 fanout 推导该值。

Marconi 原论文要求对 recency 和 FLOP efficiency 进行 normalization。本实现采用 eligible candidates 内的 min-max normalization 到 `[0, 1]`，这是 deterministic、可复现的 policy-level adaptation 约定，不声称逐行复现 Marconi 原代码内部的 normalization 实现。若某一维的所有原始值相同，该维统一归一化为 `0.0`，不人为制造差异。

未复现的 Marconi 部分包括：

- judicious admission 与 speculative insertion
- branch-point 和 last-decoded-token admission
- alpha bootstrap 与 grid search
- radix-tree simulator
- KV 与 SSM holistic runtime
- whole cache-entry management 与 Marconi runtime

## 与 FlowState 的机制差异

KVFlow-style 主要利用 future execution proximity。Marconi-style 主要利用 recency 与 compute-saving/memory efficiency。FlowState 主要利用 executable-state recovery、known workflow dependency 和 set-dependent marginal coverage。

这些是机制边界，不预设任何策略在后续实验中的优劣。

## Controlled workload 的冻结 metadata

metadata 在运行 FlowState 与 SOTA-style 离线比较前固定，构造过程不读取任何 allocation result、Oracle result、Phi 或 recovery cost，也不会根据策略表现反向调整。

### KVFlow steps-to-execution

`controlled workload only models immediate next-step branches, therefore all pending continuations have steps-to-execution = 1.`

当前 v1、N=8 和 N=16 workload 的每个 pending continuation 都表示从已知 Parent/anchor 直接进入的下一步分支，因此统一赋值 `1`。该值不随 anchor depth、fanout、workflow 标识或 checkpoint depth 变化。真实多级 future execution distance 留待后续 Public Agent Trace 从 workflow graph 生成。

### 共享 recency

KVFlow-style 和 Marconi-style 与已有 Global-LRU 使用同一份 `CheckpointRecency`。builder 直接调用现有 Global-LRU 得到完整的新到旧全序，再将其转换为 `oldest = 1`、`newest = N` 的 `last_access_by_checkpoint` rank。这个转换保留 `last_access_order`、`creation_order` 和 `checkpoint_id` 的全部既有排序语义，不创建另一套访问历史。

### Marconi incremental FLOP proxy

在固定模型下，使用 incremental replay-token span 作为 incremental FLOPs saved 的等比例代理。对每个 checkpoint，先在同 workflow、同 lineage ancestry 上寻找 token position 更小且最近的 candidate；当前候选与该 parent candidate 的 token span 为增量值，不存在 parent candidate 时从零计算。

例如同一 lineage 上存在 `SHALLOW@16K` 与 `PARENT@32K` 时，两者的 incremental span 都是 `16K`，不能把后者记作 `32K`，否则会重复计算 shallow checkpoint 已代表的计算收益。这里的数值是 proportional FLOP units，不声称一个 token 等于一个 FLOP。Marconi-style 随后会归一化 FLOP efficiency，因此固定比例常数不改变 policy ordering。

### Marconi alpha

`alpha=1.0 is a preregistered deterministic adaptation choice for controlled snapshot experiments.`

recency 与 FLOP efficiency 都已归一化到 `[0, 1]`，固定 `alpha=1.0` 表示等权。当前 controlled workload 不进行调参。后续 sensitivity ablation 可独立考察 `alpha ∈ {0, 0.25, 0.5, 1, 2, 4}`；Public Agent Trace 若具有真实历史窗口，再考虑 bootstrap 或 tuning。
