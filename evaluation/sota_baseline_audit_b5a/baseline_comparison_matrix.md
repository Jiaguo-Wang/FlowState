# Baseline 对比矩阵（Phase B5-A）

对比对象为**适配方案 B**（详见 adaptation_contract.md）：`Marconi-Adapt (utility-based eviction)` 与 `KVFlow-Adapt (steps-aware eviction)`，附两条参考线（Marconi V3 oracle、KVFlow-released LRU 路径）。所有方案共享：固定 CheckpointCandidate 集、逻辑预算 K、B1-C 统一计分。

## 1. 决策机制对比

| 维度 | Marconi-Adapt | KVFlow-Adapt | 参考：V3 oracle | 参考：released-LRU |
|---|---|---|---|---|
| 价值信号 | recency + α·FLOP 效率（原生公式） | steps-to-execution（原生语义） | 离线静态-α 全知回放 | last_access_time |
| 驱逐选择 | argmin(utility) | 阈值分档保持（hold_step 内保护，自高档向低档放行） | argmin(utility) | LRU |
| 共享候选聚合 | 无（逐条目独立效用） | 多续接引用取 min steps | 无 | 无 |
| 无信息缺省 | 归一化退化全 1 | 无标签=最远（1000） | — | — |
| 原生超参 | α∈{0,…,2.0} 网格 + bootstrap 窗口 | hold_step（默认 4，扫描 1-5） | α 事后最优 | 无 |
| 时间源 | 恢复命中事件的纪元序号 | pending 图上的纪元距离 | 全轨迹 | 命中事件序号 |

## 2. 信息与公平性

| 维度 | Marconi-Adapt | KVFlow-Adapt |
|---|---|---|
| 使用的 Strict Epoch 信息 | 候选集、恢复命中事件、静态模型配置 | 候选集、PendingContinuation/EPOCH_ADMISSIBLE 图 |
| 是否触碰未来真实请求 | 否（bootstrap 回放仅用 ≤t 纪元） | 否（pending 图为当前声明状态）——**等价性条款待评审** |
| 是否修改严格兼容矩阵 / 逻辑预算 | 否 / 否 | 否 / 否 |
| 价值信号是否被 FlowState 收益替代 | 否（仅 Marconi 原生公式） | 否（仅 steps 语义） |
| 与 FlowState 优化器的信息集关系 | 子集（无图结构信息） | 同集（都用 pending 图）——同信息不同算法，对比成立 |

## 3. 忠实性证据与风险

| 维度 | Marconi-Adapt | KVFlow-Adapt |
|---|---|---|
| 机制-源码对应 | 效用函数 radix_cache_hybrid.py:543-546；归一化 utils.py:11-19；FLOPs utils.py:46/65/104-107；调参 config_tuner.py:58-118 | 阈值保持 lora_hiradix_cache.py:269-295；min 聚合 :939-943 / agent_manager.py:45-53 |
| 已知论文-代码差异（备案） | D1 MLP 公式笔误（实现时修复）；D2 调参口径（建议论文口径）；D4 oracle 乐观性 | 采纳 HEAD 可证实形态（阈值式而非排序式）；论文排序形态 UNVERIFIED |
| 主要审稿风险 | "无准入层的 Marconi 还是 Marconi 吗" → 以"固定候选世界의 驱逐层忠实适配"定位并显式声明 | "steps 来自被测系统自己的图" → 以信息等价性条款+同信息集公平性论证应对 |
| 版本归属风险 | 低（repo HEAD=2025-03，与论文时序一致） | 高（squash+双项目混载；已全部标注 UNVERIFIED 边界） |

## 4. 工程与实验属性

| 维度 | Marconi-Adapt | KVFlow-Adapt |
|---|---|---|
| 实现量级 | 中（效用函数、归一化、bootstrap 回放调参） | 小（图距离 + 阈值驱逐） |
| CPU-only 可运行 | 是 | 是 |
| GPU 占用 | 无 | 无 |
| 主表（g001-g004 算法层）适用 | 是 | 是 |
| 敏感性/消融建议 | α 扫描曲线；V3 oracle 上界（注明 D4） | hold_step 扫描；released-LRU 参考线（源码事实：非分层默认路径即 LRU） |
| 产物目录（实现轮） | evaluation/baseline_marconi_adapt/ | evaluation/baseline_kvflow_adapt/ |

## 5. 主表列位建议（供论文写作参考，非本轮执行）

| 列 | 来源 |
|---|---|
| LRU（上界参考之一） | 现有实现 |
| KVFlow-Adapt (steps) | 方案 B |
| Marconi-Adapt (utility) | 方案 B |
| FlowState (ours) | 现有实现 |
| （可选）Marconi-V3 oracle | 敏感性分析，不进主表 |
