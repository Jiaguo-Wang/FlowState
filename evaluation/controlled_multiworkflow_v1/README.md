# 受控多工作流负载 v1

这个 workload 用于验证：多个 workflow 在共享 recurrent-state budget 下，checkpoint 的价值同时受到 recovery depth 和 pending workflow coverage 影响。

## 固定场景

| Workflow | Anchor | Pending fanout | Parent checkpoint 的覆盖范围 |
|---|---:|---:|---|
| W1 | 32768 | 2 | `W1_PARENT` 覆盖 2 个 32K continuation |
| W2 | 16384 | 1 | `W2_PARENT` 覆盖 1 个 16K continuation |
| W3 | 8192 | 1 | `W3_PARENT` 覆盖 1 个 8K continuation |
| W4 | 4096 | 3 | `W4_PARENT` 覆盖 3 个 4K continuation |

W1 另有 `W1_SHALLOW @16384`。它能够同时覆盖 W1 的两个 continuation，但只能把 executable frontier 推进到 16384，用于验证 checkpoint value 会随已选择集合发生变化。

每个 checkpoint 大小固定为 49.125 MiB，即 51,511,296 bytes。共享预算为 `K=3`，也就是 154,533,888 bytes。候选总数为 5，因此不同 workflow 必须在同一个全局预算下竞争，不能按 workflow 平均分配 quota。

## 边界

`scenario.py` 只定义逻辑 workload，并直接复用 FlowState 的 `PendingContinuation` 与 `CheckpointCandidate`。它不包含预期 optimizer 选择、不实现 baseline，也不启动任何 runtime 或 GPU 实验。

第一阶段采用 Snapshot-Isolated Evaluation：每个策略与每个待续请求都从等价的 post-build、pre-allocation 状态独立开始，只发送一个待续请求后便丢弃该 runtime state。它用于验证固定 decision epoch 下的 executable-state allocation objective，不是在线 workflow completion benchmark。

未来的 Online Sequential Evaluation 会在每次 arrival 或 completion 后重新取得 snapshot 并重新 reconcile，以显式处理新检查点和 residency 的变化；该流程不属于当前版本。
