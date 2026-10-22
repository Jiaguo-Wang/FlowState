# Baseline 适配契约（Phase B5-A）

本文件定义 Marconi / KVFlow 两个 Baseline 在 FlowState 评测框架下的三种适配方案（A 全系统复现 / B 固定候选+逻辑预算下的忠实适配 / C 部分原理启发式代理），并给出推荐结论。本轮**只立约，不实现**。

## 0. 共同硬约束（适用于一切方案）

1. **固定候选集**：驱逐/保留的对象只能是严格纪元兼容矩阵产出的 `CheckpointCandidate` 集合；不得增删候选（不得引入 Marconi 的运行时准入或 KVFlow 的 workflow 注册改变候选空间）；
2. **逻辑预算 K 不变**：所有方案在同一逻辑预算（K 个槽位）下决策；
3. **禁止未来真实请求信息**：决策在纪元 t 只能使用 ≤t 的信息（含当前已声明的 PendingContinuation 结构——它是"已声明未执行"的当前状态，不是未来请求内容；此等价性论证见 §2.4，需评审确认）；
4. **价值信号不得偷换**：Baseline 的驱逐决策只能使用其**原生价值信号**（Marconi：recency+FLOP 效率；KVFlow：steps-to-execution），不得把 FlowState 的收益指标（恢复收益、纪元增益等）作为替代喂入；
5. **统一输出接口**：每个 Baseline 的产出 = 纪元序列上的驱逐/保留决策，流入与 FlowState 相同的下游（B1-C 恢复验证），命中率/收益口径与主方法完全一致。

---

## 1. Marconi 适配方案

### 方案 A：论文系统全量复现 — **判定：BLOCKED**
- 依赖：真实混合架构 GPU 运行时上的 radix tree + HybridStates 增量写入、match 时投机插入准入、驱逐时 KV 向孩子合并；官方 artifact 本身是 CPU trace 模拟器（`artifact_evaluation.md`），无服务路径；
- 与 FlowState 冲突：运行时准入直接违反固定候选集约束（约束 1）；树合并在槽位语义下无对应物；
- GPU：即便解除约束也需要端到端 SGLang v0.5.17 hybrid 改造，工程量与公平性风险均不可控。

### 方案 B：固定候选集上的效用驱逐忠实适配 — **推荐**
**正式 Baseline 名：`Marconi-Adapt (utility-based eviction)`**

- **保留的核心决策机制**（逐条对源码）：
  1. 效用函数 `U(c) = recency_score(c) + α · flop_efficiency(c)`（`radix_cache_hybrid.py:543-546`），驱逐 argmin；
  2. recency = `1/(t_now − t_lasthit(c))` + min-max 归一化、退化全 1（`utils.py:11-19`）；
  3. FLOPs 节省公式（相对"父节点"语义改写为相对候选前缀基线）：attn `8ld²+4l²d`、MLP `16ld²`、mamba `10ld`（camera-ready 修正版，`utils.py:104-107`），字节数用 `get_kvs_size`/`get_mamba_state_size` 同式计算，全部按 **Qwen3.5-9B hybrid 真实层配置**取参（24/4/28 为 Marconi 论文配置，不得照抄）；
  4. `flop_efficiency = 节省 FLOPs / 占用字节`（`utils.py:163-168`）；
  5. α 调参协议：前 `multiplier×N` 个纪元事件为 bootstrap 窗口，窗口内网格 {0,0.1,…,2.0}×CPU 并行回放（`config_tuner.py:61/84`），选择依据**二选一并注明**：论文口径（命中率）或代码口径（FLOPs 节省，`config_tuner.py:110`）——建议采用论文口径并在实现备注中记录该差异 D2；
  6. 修复备案缺陷 D1（`radix_cache_hybrid.py:526` MLP 项误用 attn 公式）后按论文语义实现。
- **Strict Epoch 已有输入**：候选集（=Marconi 语境的"已准入条目"，准入层由矩阵代管）；每次恢复/命中事件（recency 时间源，用纪元事件序号替代墙钟）；候选 token 长度与模型静态配置（FLOPs/字节计算输入）。
- **缺失/不可用信息**：无（该方案不需要任何运行时树信息）。
- **不可保留机制**：judicious admission（分支点/末 token 状态准入、≤2 状态限制——候选固定的世界无准入自由度）；树分裂/合并；V1/V3 仅作敏感性参考（V3 可作静态-α oracle 上界，注明其两指标独立取最大的乐观性 D4）。
- **算法层主表适用性：是**（同候选集、同预算、同下游口径的纯决策对比）。
- **GPU 前提：无**（逻辑层）；端到端 GPU 对比不在本轮范围，若未来需要须单独评审公平性（Marconi 无服务实现，只能对比"适配版"）。

### 方案 C：部分原理启发式代理 — 备选（不推荐进主表）
- 例如仅 recency（≈LRU on candidates）或仅 FLOPs/字节比排序；
- 学术风险：审稿人可指控"稻草人 Marconi"；仅当 B 的 α 调参协议被否决时作为降级，命名 `Marconi-Lite (recency-only)` 并明示非忠实。

---

## 2. KVFlow 适配方案

### 方案 A：论文系统全量复现 — **判定：BLOCKED（双重原因）**
- 外部依赖：分层 KV 缓存（host backup + write_back）、LoRA-per-agent 权重预取、外部 SScheduler 持续 POST `/v1/update` 提供 steps 元数据、请求携带 agent_id——FlowState 评测环境均不存在；
- 版本证据风险：当前 HEAD 与论文版本无法建立对应（历史压缩、双项目混载、`__lt__` 早退/`# NO USE` 等现状，见 kvflow_audit.md），"复现论文系统"缺乏可信代码基线；
- 与约束冲突：workflow 引擎声明的 steps-to-execution 若取自真实未来执行计划，将违反约束 3。

### 方案 B：固定候选集上的 steps-aware 驱逐忠实适配 — **有条件推荐（含一个待评审条款）**
**正式 Baseline 名：`KVFlow-Adapt (steps-aware eviction)`**

- **保留的核心决策机制**（逐条对源码）：
  1. 驱逐值 = steps-to-execution（越小越保）；
  2. 阈值式保持：`hold_step`（对应 `evict_pri_level`，默认 4）内的候选受保护，驱逐自 `hold_step-1` 档向下逐档放行，最后才动受保护档（`lora_hiradix_cache.py:269-277, 292-295`）——这是当前 HEAD **可证实**的优先级驱逐形态，采纳它而非论文表述的排序形态（排序形态在 HEAD 无实现，UNVERIFIED）；
  3. 共享候选聚合取 min：一个候选被多个 pending 续接引用时，取最小 steps（`lora_hiradix_cache.py:939-943` / `agent_manager.py:45-53`）；
  4. 无标签候选视作最远复用（缺省 1000，`:939`）。
- **Strict Epoch 已有输入**：候选集；PendingContinuation / EPOCH_ADMISSIBLE 图（当前声明状态）。
- **steps 的信息等价性论证（关键条款，需评审确认）**：KVFlow 的 steps 来自 workflow 引擎对执行计划的声明；FlowState 中 `候选 c 的 steps` := 当前 pending 图上该候选距下一次预期激活的纪元距离。该信息是**当前已声明结构**（≤t），非未来请求内容，满足约束 3 的字面与精神；且它与 FlowState 优化器使用同一信息源，构成同信息集下的算法对比（公平），不构成价值信号偷换（约束 4：喂入的是 KVFlow 原生的 steps 语义，不是 FlowState 收益）。**若评审认为不可接受，降级为 C。**
- **不可保留机制**：KV/LoRA 双预取（无分层缓存与 LoRA 设施，且预取属于运行时机制非决策机制）；节点三态 defer 调度；agent_id 传输链；workflow 解析。
- **算法层主表适用性：是**（同上）；`hold_step` 作为其原生超参参与调参（1-5 扫描，与其 `evict_pri_level` 默认域一致）。
- **GPU 前提：无**。
- **附注（事实基线）**：当前 HEAD 非分层路径在默认参数下是纯 LRU（`lora_radix_cache.py:317-340` + `__lt__:120` 早退）。主表已有的 LRU 对照可直接引用此源码事实作为 `KVFlow-released (LRU path)` 的依据，无需新实现。

### 方案 C：部分原理启发式代理 — 降级备选
- steps 不可用时退化为"引用计数 + LRU"；命名 `KVFlow-Lite` 并明示降级原因；只配进敏感性分析，不进主表。

---

## 3. 方案对照总表

| 维度 | Marconi B（推荐） | KVFlow B（有条件推荐） |
|---|---|---|
| 决策信号 | recency + α·FLOP效率（原生） | steps-to-execution（原生，源自 pending 图） |
| 原生超参 | α（bootstrap 网格回放） | hold_step（1-5） |
| 需评审条款 | 调参选择口径 D2（命中率 vs FLOPs） | steps 信息等价性（§2.4） |
| 实现量级 | 效用函数+归一化+回放调参（中） | 图距离+阈值驱逐（小） |
| 主表 | 适用 | 适用 |
| 已知备案缺陷 | D1（MLP 公式笔误，修复后实现） | 无（采纳 HEAD 可证实形态） |

## 4. 共同输出接口约定

- 输入：纪元事件流（候选产生、恢复命中、纪元边界）+ 静态模型配置；
- 输出：每纪元驱逐决策序列（与 B4-C MatrixOptimizer 同接口），经 B1-C 恢复验证统一计分；
- 产物目录（实现轮启用）：`evaluation/baseline_marconi_adapt/`、`evaluation/baseline_kvflow_adapt/`。

## 5. 本轮明确不做

不实现任何 Baseline；不执行 g001-g004 对比；不占用 GPU；不修改 `flowstate/` 核心与严格兼容矩阵。
