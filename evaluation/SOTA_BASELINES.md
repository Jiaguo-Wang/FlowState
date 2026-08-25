# SOTA-style 基线策略

本文档冻结两个面向相同 recurrent checkpoint 候选集的策略级适配。两者与 FlowState 后续共享相同的待续请求、候选状态、内存预算、`StateController`、`SGLangAdapter` 和 Mamba-only mutation。这里比较的是已有状态的保留选择，不是原系统的完整复现。

## KVFlow-style

实现名称为 `KVFlow-style adaptation`。

原机制通过 Agent Step Graph 为未来 Agent invocation 提供 steps-to-execution。数值越小表示越接近执行，对应 KV 节点越值得保留；一个共享节点关联多个未来 invocation 时，使用其中最小的 steps-to-execution。

本适配要求调用方为每个活动 `PendingContinuation` 显式提供 `continuation_id -> steps_to_execution`。策略使用核心 `is_compatible()` 找到候选覆盖的待续请求，并以最小 steps-to-execution 作为候选优先级；没有 future dependency 的候选优先级为正无穷。排序从小到大，平局只按 `checkpoint_id` 决定。

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

这些是机制边界，不预设任何策略在后续实验中的优劣。N=8/N=16 workload 的 steps-to-execution、recency、incremental FLOPs saved 和 alpha 必须在后续步骤独立冻结，本步骤不生成这些 metadata，也不进行策略胜负比较。
