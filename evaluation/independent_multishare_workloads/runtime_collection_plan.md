# 后续 Runtime 采集计划（仅计划，本轮不执行）

目标：为通过 CPU 核验的 w1–w4 提供统一的 GPU 采集方案，把内容级 Token 结构转换为可核验的严格 Runtime 事实。本计划不启动 GPU、不构造 Strict Snapshot、不运行任何分配算法；执行需另行授权。

## 1. 统一采集合同（每组一个独立 Engine 生命周期）

沿用 B5-C3 已验证的采集合同（`multishare_runtime_gate_b5c3/launch.py`、`runner.py`、`bridge.py`），不新造观察语义：

1. **配置身份**：B4 冻结引擎参数（Qwen3.5-9B、SGLang 0.5.17、`mamba_track_interval=256`、`max_mamba_cache_size=28`、上下文 131200）；镜像、模型、Tokenizer、模板与 CPU 阶段身份锁一致；启动前逐项核对配置实际生效值。
2. **冻结历史重放**：每组 8 条历史请求按 B5-C2 既定顺序提交——turn 1 的 A/B/C/D，再 turn 2 的 A/B/C/D；每条请求的实际接收 Token 数组、输入摘要与 CPU 数组逐一比对（`history_XX.json` 证据），输出长度 1、无回撤。
3. **Pending 准备**：4 条第三次输入在 Barrier 前准备并登记 `controlled_input_ready=True`；**不派发**，不读任何未来内容；`agent_native_ready` 保持"未认证"。
4. **Barrier 观察**：Engine 静止后单次同步观察——完整 census（树结构、Mamba 行、槽位记账、引用/别名）、逐 Pending `inspect_resident_fa_frontier`、只读无副作用校验（before==after）。
5. **候选事实**：只登记自然物化的 Recurrent Checkpoint（Node/Slot/组件完整性/引用清零）；不人工插入切点；候选外可恢复状态照记。
6. **严格矩阵**：用冻结 `build_edges`（B4-B0 contracts）对全部 候选×Pending 配对产生三态矩阵；任何 UNKNOWN 都使该组采集失败，不得默认准入。
7. **支持集与预算竞争验证**：按冻结 Phi 逐边计算正收益；登记每候选的严格正支持集合、跨 Pending fanout、同任务嵌套覆盖与 K_min；保留全部有效与无效配对。
8. **案例保留**：无论结构贫富（单例、无共享、仅系统级共享、多共享）一律原样报告；不更换 Session、不调预算、不重放取巧。

## 2. 逐组采集要点

| 组 | 采集关注 | 相对 B5-C3 的差异点 |
|---|---|---|
| w1 | 任务二 LCP 3771 为全轮最深任务内共享 | 验证深层共享是否实际物化且不被原生驱逐合并 |
| w2 | Pending 最短 4122、任务二内差距最大（546） | 验证浅/深候选并存时的配对完整性 |
| w3 | C/D 内容几乎等长（差 5）且 LCP 3471 | 验证近重复输入的严格边不因内容相似而误判 |
| w4 | 四方 2437（唯一少 1）；D=6971 全轮最长 | 验证长输入下 FA frontier/T 与 CPU 位置的偏差 |

每组独立生命周期、独立清理与 GPU 连续性检查（沿用 B5-C3 的 gpu_cleanup 纪律）；组间不共享 Engine 状态。

## 3. 必须在 Runtime 阶段回答的问题（对 CPU 结论的逐项升级）

1. 历史请求实际接收 Token == CPU 数组？（身份）
2. 自然物化的 Recurrent Checkpoint 位置/数量/所有权？（候选域）
3. 逐 Pending 实际 FA frontier 与 Planning Target；2428/2437–2438/3471–3771 这些位置是否落在 L≤T 内？（坐标）
4. 内容共同前缀是否对应多个**同时驻留**、支持不同 Pending 子集的真实状态？（共享真实性；内容支持 ≠ EPOCH_ADMISSIBLE ≠ RESUME_VALIDATED）
5. 28 槽配置下的实际预算竞争形态（K_min、fanout、覆盖重叠）？（结构）
6. 无效与贫结构案例是否同样完整保留？（无偏性）

## 4. 明确不做与前置门禁

- 不回填：不得用 Pending 执行后生成的节点补此前 Barrier；失败组原样报告。
- 不越权：采集阶段只观察；任何驱逐/保留/策略比较属于更后续阶段，另立门禁。
- 进入条件：本目录 `verification.json` 为 READY 且授权到位；容器/镜像/GPU 空闲核验沿用 B5-D 系列纪律。
