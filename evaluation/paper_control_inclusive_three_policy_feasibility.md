# P0-4：三策略完整控制周期性能对照可行性审查（2026-10-28）

## 0. 总裁决：**SMALL_HARNESS_CHANGE_REQUIRED**

核心 Runtime（SGLang v0.5.17 容器 + FlowState 严格分配入口）已具备三策略完整控制周期的全部能力，且关键环节均有**已冻结的 GPU 实验证据**；缺口全部位于 Runner 编排层（Baseline 在线 Select 接入、4-Pending 串行循环、T0–T4 统一计时、取证 RPC 出窗），不需要修改 Scheduler、Optimizer、Controller、Solver、冻结策略规则或核心状态生命周期。本轮只审查、不实现。

核心问题（spec 第二节）的回答：**能**——在同一 Strict Allocation Epoch、相同候选集合、预算和物理状态条件下，三种策略可以分别完成真实 Select、统一 Recurrent-only Reconciliation 和四 Pending Native Resume，并准确测量完整周期耗时；前提是完成下文第七节列出的 5 项 harness 级修改。

---

## 1. 可复用 Runner 与调用链（Task A 问题 1/2/6）

### 1.1 现有 Runner 调用链（代码级核实，非目录名推测）

**T2-G2 正式实验 Runner = `c_formal_perf_gpu/runner.py` 原样复用**。证据：[device_entry.py:34](t2g1_three_policy_gpu/formal_gpu1b/device_entry.py#L34) 在设备 UUID 校验后 `runpy.run_module('evaluation.c_formal_perf_gpu.runner', run_name='__main__')`，T2-G2 的 60/60 正式臂（GPU1b 批次）全部经此入口执行（见 memory: t2g1-three-policy-gpu）。

C/K3 Runner 单臂生命周期（[runner.py](c_formal_perf_gpu/runner.py)）：

| 阶段 | 位置 | 内容 |
|---|---|---|
| 重置 | L157 前 | fresh engine → 16 历史 replay → flush_cache → 16 二次 replay 重建 16 候选 |
| 对账 | L157–171 | census 16=16 → **单次 `b5d4_apply` RPC**（加载冻结选集，[control.py:62,75](c_formal_perf_gpu/control.py#L62)）→ PILOT_POLICY 校验 |
| 预热 | L173–203 | 形状预热 B v1.1（pt[:E]+确定性随机尾，恰在 E 处分叉） |
| 请求 | L205–298 | 正式请求（单 Pending，max_new_tokens=8）+ 测量模式断言（窗口零 A 类审计）+ E/G/Prefill/Cold 合同 |

**完整 Observe–Select–Reconcile–Resume 单臂闭环已存在**：G1 Hybrid Solver 验收（[final_report.md](hybrid_solver_gpu_gate_g1/final_report.md)，8/8 臂 PASS），其控制入口分段链为：

```
control.py:69  final_heavy_observation      真实物理状态观察
control.py:82  cpu_physical_mapping         CPU 物理映射
control.py:131 strict_epoch_authentication  Strict Epoch 构造与认证
control.py:141 plan_strict_hybrid_solve     Scheduler 进程内在线求解（timed）
control.py:236 StateController.reconcile_strict  物理协调（用本轮 Plan）
随后 Native Resume → Cold Reference 逐 Token 一致（4/4 臂）
```

**四 Pending 串行模式已存在**（另一场景）：`resume_epoch_e2e.py` 每 epoch 公共观察（t0）→ 协调+compact_census（control_end）→ 4 个串行单 Token 请求，epoch 结束=第 4 请求流结束，[epoch_metrics L99–112](resume_epoch_e2e.py#L99-L112) 提供恒等式 `Epoch = Control + Σ(4×请求) + Residual`。这是旧 Control-inclusive harness（P0-3 审计对象），证明四请求串行编排与闭合恒等式工程上可行，但其控制路径是重取证版本，不可直接搬用（见 §5）。

### 1.2 轻量控制路径

`control_path_optimized_transport`（tests/runtime/）已把控制面收敛为 3 个批量入口、请求校验路径逐字节不变；P0-3（[paper_runtime_epoch_bottleneck_audit.md](paper_runtime_epoch_bottleneck_audit.md)）证明旧 Epoch 请求间隙 97.8% 是取证 RPC 墙钟、其中 95% 是进程交接段而非状态管理计算（metrics 服务端 0.075ms/次 vs 2–1555ms 墙钟）。因此新对照必须采用批量/移窗方案（§5.4），不能复用旧 per-request inspect×14 模式。

**结论**：可复用件齐全——重置协议与正确性合同来自 C/K3 Runner，在线求解与协调分段来自 G1 pkg_greedy，四请求编排与恒等式模式来自 resume_epoch；但三者分属三个 harness，**没有任何一个现有 Runner 同时具备"在线 Select + 统一协调 + 4-Pending + T0–T4 计时"**，此即 SMALL_HARNESS_CHANGE_REQUIRED 的直接依据。

---

## 2. 三策略在线 Select 能力核查（Task A 问题 3/4；Task B.1）

### 2.1 FlowState-Greedy：真·Scheduler 内在线 Select，已冻结 GPU 证据

- [pkg_greedy/control.py:141](hybrid_solver_gpu_gate_g1/pkg_greedy/control.py#L141) `plan, decision = timed('plan_strict_hybrid_solve', solve_online)`：在真实 Scheduler 进程内调用生产入口 [GlobalOptimizer.plan_strict](../flowstate/optimizer.py#L86)（混合门：≤12 候选→Exact 枚举，超→Marginal Greedy；C/K3 16 候选自动走 Greedy）。
- [control.py:147](hybrid_solver_gpu_gate_g1/pkg_greedy/control.py#L147) `selection_source='GlobalOptimizer.plan_strict(在线)'`；冻结选集在求解完成后才读入（[L154](hybrid_solver_gpu_gate_g1/pkg_greedy/control.py#L154)），仅作事后 oracle（L171 `oracle_policy=...仅事后对照`）。
- 冻结证据：G1 实验 B 4/4 臂在线选集==冻结 FS 三元集 {C_B_P05120, C_C_P06144, C_D_P07168}，J=154.34895561599998 逐位一致，`oracle_match=True`（final_report.md §二）。

### 2.2 Marconi-Evict Adaptation：真实在线选择内核存在，未接 GPU 窗口

- 内核：[MarconiEvictPolicy.select](marconi_evict_t2m/evict_policy.py#L42-L92)——逐轮驱逐循环（可驱逐=叶/单孩子，每轮 `_build_marconi_metrics` 重评分，victim=min (utility, id) 确定性，最近保留祖先重接），输入仅为候选+分配时刻可见的 recency/parent，输出 (selected, evicted, rounds, call_ms) 完整轨迹。T2-Mini 已 8/8 测试且旧版一致性 12/12（memory: t2mini-marconi-evict）。
- 统一复现证据：[adapter.py:68](t2g1_three_policy_gpu/adapter.py#L68) 已在同一冻结声明输入上运行该内核并复现冻结选集（`marconi_matches_t2mini`），[L114](t2g1_three_policy_gpu/adapter.py#L114) 记录 `selection_call_ms`。
- **缺口**：T2-G2 的 GPU 窗口内没有该内核。`b5d4_apply` 服务端 [control.py:62](c_formal_perf_gpu/control.py#L62) `plans = json.loads((OUT/'policy_plans.json').read_text())`、[L75](c_formal_perf_gpu/control.py#L75) `selected_cpu = tuple(plans[policy]['selected_checkpoint_ids'])`——三策略（含 FS）在 T2-G2 中全部是**冻结选集加载+认证**，没有在线 Select。spec 明令：不得把冻结选集加载时间当作在线 Select 时间。

### 2.3 KVFlow-inspired：同上——真实内核、已复现、未接 GPU 窗口

- 内核：[kvflow.select](kvflow_inspired_retention_cpu/policy.py#L85)——声明图上的 STE 计算（known_at_epoch 门控，无未来信息）+ 前缀优先 + 确定性平局打破，不读恢复收益（与其原语义一致）；[adapter.py:74](t2g1_three_policy_gpu/adapter.py#L74) 复现冻结选集（`kvflow_matches_frozen`）。

### 2.4 公平接入方式（不改变任何冻结规则）

三个内核都是**纯函数**（声明/观察输入 → 选集），接入方式与 G1 对 FS 的做法完全同构：在控制入口（Scheduler 进程内）分别调用 `GlobalOptimizer.plan_strict` / `MarconiEvictPolicy().select` / `kvflow.select`，规则零修改，冻结选集仅作事后 oracle 对照（G1 已示范该模式且 8/8 通过）。**禁止**为端到端性能修改任何一方的评分/驱逐/STE 规则（spec Task B.1）。

---

## 3. 统一 Observation 和 Reconciliation 可行性（Task A 问题 5；Task B.2/B.3）

### 3.1 同一 Recurrent-only Reconciliation：已存在且策略无关

[StrictReconciler.reconcile_strict](../flowstate/strict_control.py#L193-L229)：bind → preflight → 逐驱逐 `evict_mamba_only`+每步 postcheck → final postcheck，fail-stop。协调行为完全由 `plan.evicted_ids` 驱动，与策略无关；postcheck（L166–188）验证 FA/非 Recurrent 摘要不变、槽位释放记账、驻留==Plan。

- FS 走此路径：G1 [pkg_greedy/control.py:236](hybrid_solver_gpu_gate_g1/pkg_greedy/control.py#L236) `StateController.reconcile_strict(epoch, plan, runtime)`（[controller.py:115](../flowstate/controller.py#L115) 静态入口）。
- 三策略共用此合同已有冻结证据：T2-G2 60/60 臂中三策略冻结选集全部经同一 b5d4_apply 认证+同一物理协调语义执行（memory: g2-b5c4-formal-gpu、t2g1-three-policy-gpu）。
- MarconiEvictPolicy 天然产出驱逐序（`evicted_checkpoint_ids` 按驱逐先后，[evict_policy.py:88](marconi_evict_t2m/evict_policy.py#L88)）；FS/KV 的 evicted 集合由 16 候选补集得出。三者的 Plan 装配可以统一到同一 StrictAllocationPlan 校验器。

**物理公平（Task B.3）满足**：所有策略实际驱逐未选中的 Recurrent Checkpoints（每步 postcheck），非仅改逻辑选集；FA-KV 保护合同三策略同一份。

### 3.2 元数据公平（Task B.2）

| 策略 | 在线信号 | 分配时刻可见性 | 依据 |
|---|---|---|---|
| FlowState | 活动前沿观察（真实物理状态） | 观察本身即在线 | G1 final_heavy_observation |
| Marconi-Evict | 声明 recency + parent | 是（T2-R 验证声明输入无未来信息） | [evict_policy.py:52-58](marconi_evict_t2m/evict_policy.py#L52-L58) |
| KVFlow-inspired | 声明 ASG/STE（known_at_epoch 门控） | 是（门控即为此设计） | [policy.py:85](kvflow_inspired_retention_cpu/policy.py#L85) |

无未来轨迹/后续执行结果泄漏。公共观察与认证（census、身份认证、物理映射）三策略共享同一实现；**各策略差异化的元数据获取成本必须分段计时**（FS 的活动前沿观察 vs 两 Baseline 的声明读取成本不同），列入 §5 分解表。

---

## 4. 初始状态重置与四 Pending 执行协议（Task B.4；Task D 重置）

### 4.1 每轮从相同已物化初始状态重启：已冻结证明

C/K3 重置协议（runner.py，T2-G2 五轮×三策略 60/60 全部走此协议）：fresh engine → 16 预热 replay → flush_cache → 16 二次 replay（重建 16 候选）→ census 16=16 对账 → apply → 形状预热 → 正式请求。每轮独立 docker 生命周期，物理初始状态由 census 认证。**零改动复用**。

### 4.2 四 Pending 串行执行与状态污染

- 现状：C/K3 正式臂为**单 Pending 逐臂**（C_A..C_D 各一臂，共享同轮重置）；resume_epoch 有 4×串行模式但是旧重取证控制路径。
- 需接入：apply 后的 4-Pending 串行请求循环 + **每 Pending 形状预热**（形状预热 B v1.1 模式适配任意 E：pt[:E]+确定性随机尾恰在 E 分叉——T2-G2/C 配对计时已验证该模式消除首形状编译分叉，memory: c-pair-timing-validation）。
- 隐式状态污染（Task B.4 重点）：串行请求间前一请求产生新状态。协议要求：①请求间必要在线检查（身份/物理复核）保留且三策略一致；②请求间取证类 RPC 按统一批量/移窗方案处理（三策略同构）；③每轮结束的 post_request 审计（A 类）在 T4 之后，天然出窗。resume_epoch 的 P0-3 审计已给出请求间隙构成的完整定量画像（inspect×14+metrics×3+census×3），新协议按同口径报告间隙与 Residual。

---

## 5. 完整计时边界（Task C）

### 5.1 T0–T4 定义与恒等式

```
T0 开始本轮状态观察与认证（公共观察起点）
T1 完成候选、兼容关系与物理状态确认（Strict Epoch 认证完成）
T2 完成本策略在线 Checkpoint Select（三内核各自 timed）
T3 完成 Recurrent-only Reconciliation 及必要安全后验
T4 四个 Pending 的 Native Resume 与请求执行全部完成
主指标：Control-inclusive Cycle Time = T4 − T0
恒等式：Cycle = (T1−T0) + (T2−T1) + (T3−T2) + (T4−T3)（逐段断言闭合）
```

### 5.2 分段记录

| 区间 | 内容 | 现有基础 |
|---|---|---|
| T0–T1 | Observation / Authentication / Physical Mapping | G1 timed 三段（control.py:69/82/131）直接复用 |
| T1–T2 | Select / Planner（三策略各自内核+元数据获取成本） | G1 `plan_strict_hybrid_solve`（:141）；MC/KV `call_ms`（evict_policy.py:87 / adapter.py:114） |
| T2–T3 | Plan 校验 + Reconciliation + 必要后验 | G1 协调段（:236）+ strict_control postcheck |
| T3–T4 | Native Matching/COW + 4×TTFT + 四请求完整执行 + 请求间隙/Residual | C/K3 TTFT 六段分解 + resume_epoch 恒等式模式 |
| T4 后 | 独立正确性验证（重型审计、Cold 对照、冻结 oracle 对照） | C/K3 post_request 审计（出窗） |

### 5.3 必要检查保留（不得删检查制造收益）

在线身份认证、物理映射复核、所有权检查、Reconciliation Postconditions 保留在周期内（T0–T3 已含）；研究专用重型内容审计（全量 census 摘要、逐候选双快照、Cold 重算）移到独立正确性验证阶段（T4 后）。若四个 Pending 为单 Token Continuations，需分别报告执行终点与 TTFT 的区别（resume_epoch 口径：TTFT=首 Token 到达，执行终点=流结束）。

### 5.4 取证 RPC 处理（P0-3 结论强制）

P0-3 证明旧模式请求间隙 97.8% 为取证 RPC 墙钟（95% 是进程交接而非计算），若不处理，三策略的周期差会被交接噪声掩盖（旧数据 FS−MC 残差差的 ~70% 来自 3–5 个停滞 epoch，且跨会话变号）。新 harness 必须采用已验证的批量化（flowstate_batch_introspection 模式）或把非必要取证移出 T0–T4 窗口；三策略同构处理，不得对某策略单独豁免。

### 5.5 跨进程时钟可信度

G1/C-K3 已用同一 CLOCK_MONOTONIC 体系在 runner（容器内）与控制入口（Scheduler 进程内）间衔接；E4/E5 统一计时审计（memory: e45-unified-timing）证明 54 生命周期五指标离线全可还原、零不可解释区间。T0–T4 各点打点位置全部在已有 timed() 基础设施上扩展，无新时钟源。

---

## 6. 必要正确性门禁（Task D；8 条全部可复用现有合同）

| # | 门禁 | 现有依据 |
|---|---|---|
| 1 | Strict Epoch 输入与兼容关系正确 | G1 `strict_epoch_authentication`（control.py:131）+ e4/e45 合同 |
| 2 | 三策略预算合法（K=3） | `_validate_budget_k`（evict_policy.py:44）+ adapter K 校验 + plan_strict 预算断言 |
| 3 | Select 选集符合各自策略 | 在线选集 vs 冻结选集事后 oracle（G1 `oracle_match` 模式 + adapter `fs_matches_pilot/kvflow_matches_frozen/marconi_matches_t2mini`） |
| 4 | 物理驻留集合与选集一致 | census 16=16 对账 + strict_control postcheck 驻留==Plan（L166–188） |
| 5 | FA-KV 不被 Recurrent-only Reconciliation 破坏 | postcheck FA/非 Recurrent 摘要不变断言（G1 8/8、T2-G2 60/60 已验） |
| 6 | Native Resume 的 E/G 与 Prefill 正确 | Prefill=G+32 冻结合同（G1 逐臂 E/G==冻结值） |
| 7 | 输出通过独立 Cold Reference 检查 | C/K3 冷参照 `cold_reference_C_{A..D}.json` 已存在（4×Pending 全覆盖），逐 Token 一致门禁 |
| 8 | 测量区间正确闭合 | T0–T4 恒等式断言（§5.1）+ epoch_metrics 模式（resume_epoch L99–112） |

---

## 7. 是否需要修改代码及修改范围（Task E）

**核心 Runtime：零修改。** Scheduler、Optimizer（plan_strict 及混合门）、Controller、Solver、Strict 语义、物理驱逐路径、冻结策略规则、状态生命周期、所有权模型全部不动。以下 5 项全部是 harness/编排层（新增文件或新控制入口，参照 G1 pkg_* 复制模式）：

| # | 缺口 | 性质 | 具体内容 |
|---|---|---|---|
| ① | Baseline 在线 Select 接入测量窗口 | 新增 Baseline Adapter + Runner 编排 | 扩展 b5d4_apply 或新建控制入口，嵌入 `MarconiEvictPolicy().select` / `kvflow.select`（内核零修改，调用方式照抄 [adapter.py:52-76](t2g1_three_policy_gpu/adapter.py#L52-L76)），T2 计时点+`call_ms` 记录；冻结选集仅事后 oracle（G1 模式） |
| ② | apply 后 4-Pending 串行循环 | Runner 编排 | 请求循环+每 Pending 形状预热（v1.1 模式适配任意 E）+请求间统一检查协议 |
| ③ | T0–T4 统一计时与恒等式 | Runner 计时点 | 在 G1 timed() 分段基础上补 T4 段与闭合断言 |
| ④ | 取证 RPC 批量化/移窗 | Runner 编排 | P0-3 已验证方案（批量入口/出窗），三策略同构 |
| ⑤ | 元数据获取成本分段记录 | Runner 计时点 | FS 活动前沿观察 vs MC/KV 声明读取，各自 timed |

**不涉及**新的 Runtime 状态所有权或并发问题：三内核均为纯函数、在 Scheduler 进程内同步执行（与 G1 `plan_strict_hybrid_solve` 完全相同的执行模式，该模式已 8/8 GPU 验证）。**不存在"理论上可以实现"被当作"已支持"的情况**——每一项能力均给出已冻结 GPU 证据或明确标记为缺口。

---

## 8. 最小 GPU 实验矩阵（Task D）

| 维度 | 配置 |
|---|---|
| Workload | C/K3 |
| Model | Qwen3.5-9B |
| GPU | H100 PCIe ×1（冻结镜像 sha256:1dfce…，同 G1/T2-G2） |
| Pending | 4（C_A..C_D 串行） |
| Candidates | 16 |
| Budget | K=3 |
| Policies | FlowState-Greedy / Marconi-Evict / KVFlow-inspired（全部在线 Select） |
| Rounds | 5（复用现有五轮协议；每轮从重置协议重启） |
| 规模 | 3 策略 × 4 Pending × 5 轮 = 60 生命周期（对齐 T2-G2 规模） |
| Execution | 同一轻量控制路径、真实 Select + 统一 Reconcile + Native Resume，T0–T4 计时 |

轮数仅支持基本配对验证（逐 Pending/逐轮配对差与符号计数，同 T2-G2 口径），**不提前承诺统计显著性**。冷参照与冻结 oracle 全部复用现有产物，无新参照生成成本。

---

## 9. 最终裁决：SMALL_HARNESS_CHANGE_REQUIRED

- **不是 READY_FOR_MINIMAL_GPU**：现有没有任何单个 Runner 同时支持在线三策略 Select + 4-Pending + T0–T4；T2-G2 的 b5d4_apply 只加载冻结选集（control.py:62,75），直接复用会把冻结加载冒充在线 Select，违反 spec 明令。
- **不是 BLOCKED**：FS 在线 Select 已被 G1 8/8 GPU 证明；两 Baseline 有真实、确定性、分配时刻可见输入的选择内核且已统一复现冻结选集（T2-G1 adapter）；统一策略无关 Reconciliation、重置协议、冷参照、8 条门禁全部存在并冻结验证过；不需要任何核心 Runtime/语义修改。

**下一步（本轮不执行）**：按 §7 清单实施 5 项 harness 修改（建议新包复制模式照 G1 pkg_greedy），实施后先跑 1 轮×3 策略的正确性闭环（8 门禁），再进入 60 生命周期正式矩阵。

---

## 附：证据索引

| 结论 | 证据 |
|---|---|
| T2-G2 复用 C/K3 Runner | device_entry.py:34 |
| b5d4_apply 冻结选集加载 | c_formal_perf_gpu/control.py:62,75 |
| FS Scheduler 内在线 Select | hybrid_solver_gpu_gate_g1/pkg_greedy/control.py:141,147,154；G1 final_report（8/8，J=154.34895561599998==冻结） |
| 生产求解入口与混合门 | flowstate/optimizer.py:86 |
| Marconi-Evict 真实内核 | marconi_evict_t2m/evict_policy.py:42-92（T2-Mini 8/8 测试、12/12 一致） |
| KVFlow-inspired 真实内核 | kvflow_inspired_retention_cpu/policy.py:85 |
| 三内核统一复现冻结选集 | t2g1_three_policy_gpu/adapter.py:52-76,114 |
| 统一策略无关协调 | flowstate/strict_control.py:193-229；controller.py:115；G1 control.py:236 |
| 重置协议与 census 对账 | c_formal_perf_gpu/runner.py（T2-G2 60/60） |
| 4-Pending 串行与恒等式模式 | resume_epoch_e2e.py:99-112,203-217 |
| 取证 RPC 定量画像与出窗方案 | paper_runtime_epoch_bottleneck_audit.md（P0-3，NO_CHANGE_NEEDED） |
| 形状预热 v1.1 | c_pair_timing 验证（memory: c-pair-timing-validation）；runner.py L173-203 |
| 冷参照 4× | cold_reference_C_{A..D}.json |
| 跨进程计时可信 | E4/E5 统一计时审计（memory: e45-unified-timing） |
