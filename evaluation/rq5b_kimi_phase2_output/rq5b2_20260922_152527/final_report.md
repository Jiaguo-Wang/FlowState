# RQ5-B Kimi Phase 2

RQ5B_KIMI_PHASE2_READY

模型：`/data1/models/Kimi-Linear-48B-A3B-Instruct`。SGLang 0.5.17，提交 `29481685462732237d80d86076d6563e1f658102`。2 × NVIDIA H100 PCIe（各 81,559 MiB），TP=2。

Step 0：NEED_MINIMAL_ADAPTER。直接调用冻结 GlobalOptimizer、RecoveryCostModel、兼容性和 H/E/G 函数，以及 SGLangAdapter.evict_mamba_only。新增部分仅负责 TP=2 物理句柄发现、批量分发、只读元数据证据与聚合验证。未修改 SGLang core。

两个 workflow 各有 2K/4K 检查点，候选 A1、A2、B1、B2，逻辑预算 K=2；当前待续分支分别锚定这四个已物化位置。全祖先路径核对排除了遗漏的兼容 KDA 检查点。预算仅约束这四个候选，已物化的非候选分叉末端状态保持原样。

每个 epoch 一次逻辑批量观测、一次逻辑批量协调：每个批次分别发送一次完整请求到 TP0 和 TP1，共 4 个 rank RPC。两个调度器均空闲且期间无生成请求，协调先验证观测摘要和替换元数据未漂移；每个 rank 在协调内返回一次统一后验检查，宿主机据返回值完成一次 aggregate validation，无额外观测 RPC。本实验证明空闲安全时点的双 rank 实现，不主张跨 rank 故障事务回滚。

| 生命周期 | 状态 | GPU 清理 | S* |
| --- | --- | --- | --- |
| smoke | PASS | PASS | A2, B2 |
| rep01 | PASS | PASS | A2, B2 |
| rep02 | PASS | PASS | A2, B2 |
| rep03 | PASS | PASS | A2, B2 |

S* = {A2, B2}；A1 与 B1 的 KDA 在两个 rank 上均被删除，A2 与 B2 保持驻留。

| 当前待续分支 | H | 驱逐前 E/G | 驱逐后 E/G |
| --- | ---: | --- | --- |
| A1 / B1 | 2048 | 2048 / 0 | 0 / 2048 |
| A2 / B2 | 4096 | 4096 / 0 | 4096 / 0 |

冻结目标函数空选择成本为 461.528689040，S* 成本为 152.830777808；全部候选当前驻留时成本为 0.000000000。前者是 greedy 从空集开始的目标基线，不应与全部驻留状态混同。边际选择次序和各步收益见 allocator_output.json。

成本数值沿用冻结 Qwen 恢复模型接口及其 ms 标记，仅验证分配接口可执行；未经 Kimi 校准，不可解释为 Kimi 恢复时延预测。未拟合恢复模型、未采集 TTFT 或吞吐比较。

冻结文件核验：5161 个已有文件；变化数 0。实验执行源码变化数 0。相关测试：62 passed，见 logs/related_tests.log。

逐 rank 驻留和物理槽位见 per_rank_state/；只读证据同时覆盖 MLA/KDA 驻留、分配器和 LRU 指针及访问字段；全树差分、槽位释放计数与授权组件驱逐事件共同验证 recurrent-only mutation、无原生驱逐、无重物化和无 attention 级联。already absent 保持 absent 的分支由 CPU 测试覆盖；本次真实运行四个初始候选均已驻留。

每次生命周期从空缓存启动，分别完成真实物化，关闭后独立检查无计算进程且两卡回到基线显存。后续运行仅复用编译缓存，不复用运行时缓存或物理槽位。

FlowState 的同一 executable-state allocation 和 batched recurrent-state realization 可以在不改变核心语义的情况下从 Qwen3.5 的 FA+GDN 迁移到 Kimi Linear 的 MLA+KDA。

本轮止于 Phase 2，未启动 Phase 3 或其他 RQ 实验。
