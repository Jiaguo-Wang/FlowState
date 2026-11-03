# FlowState ICDE — Solver Quality–Latency Tradeoff 审计

日期：2026-10-23 ｜ 判定：**GREEDY_PARTIAL** ｜ 脚本：`paper_solver_tradeoff_bench.py`（同目录）
原始数据：`paper_solver_tradeoff_raw.csv`（12,480 行计时）｜ 汇总：`paper_solver_tradeoff_summary.csv`（104 行）｜ 门禁记录：`paper_solver_tradeoff_gate.json`

> 口径声明：所有 ms 均为冻结 `position_aware_quadratic_v1` 下的 modeled ms 或本地 CPU Microbenchmark
> 计时；P95 仅为微基准经验分位数，不是生产在线 P95；CPU 求解时间不是生产控制路径耗时。同一快照的
> 四个 K 不视为独立工作负载（20 条 Strict 条件 = 5 快照 × K1–4）。共享机器微基准可能受后台负载干扰。

---

## 1. 输入、实现来源与正确性门禁（先于计时，全部通过）

**输入（冻结，只读，哈希核对通过后才计时）：**
- Strict 20 条件：`greedy_swap_structure_e3_step1/audit.py load()` 重建（w1–w4 ←
  `independent_runtime_allocation_e2/*/strict_epoch_snapshot.json`，B5-C4 ← `multishare_allocation_gate_b5c4/results.json`；
  e2/b5c4 manifest 与 b4c `input_integrity.json` 由该加载器自证）。
- C/K3：`kvflow_multiworkflow_cpu_main/workloads.json` 结构 C 情景（16 物理候选、4 当前 Pending、K=3），
  Epoch 构造逐字复刻 `run_cpu.py epoch(case,k,'current')`；`input_lock.json` 子集哈希运行前后两次核对。

**求解器实现（零改动复用，仅包装计时）：**

| 方法 | 实现位置 | 关键规则（已核实） |
|---|---|---|
| Marginal Greedy | `flowstate/optimizer.py:78-174`（经 e3 `audit.py:60` 闭包，等大小 size=1，预算=K 项） | 边际并列按 checkpoint_id 字典序；tol 1e-9 |
| Exact | e3 `audit.py:61-63`，同分规则与 `exact_planner_contract_gate_b5d1/planner.py:123-126` 同语义 | 容差 1e-9 内取最少项、再字典序 |
| Greedy+1-Swap | e3 `audit.py:42-56 swap()` | 初始集=**Greedy 解**；1-1 邻域；固定 (removed,added) 字典序**首改善**；仅严格下降接受（不跨等值平台）；无改善终止 |
| WFC | `p1_strong_baseline_validation/wfc.py`（哈希锁） | 排序键 (-q,-r,id)，不读 Phi |
| Swap-only（诊断用） | 同 swap()，从 Greedy 解出发 | 用于报告 1-Swap 增量成本 |

**门禁结果：**
1. Strict 20：四方法选集与 J 对 `condition_results.csv` **20/20 完全一致**（EPS=1e-9）；选集均满足
   预算（|S|≤K、无重复）与候选域约束；兼容关系即冻结 EPOCH_ADMISSIBLE 矩阵（评分函数对越界/未知
   候选直接抛错）。
2. Exact 准入：正式 `planner.allowed`（planner.py:121-122）对 20 条件全部放行（n=7、|P|=4、K≤4、
   枚举子集 99≤794）；**C/K3 n=16>12 被拒，Exact 列一律留空，不绕过门禁、不宣称任何最优性**。
3. C/K3：Greedy/WFC 选集与 `cpu_results.json` C/K3 行的 selected 及 current_J_ms 一致（154.349 /
   471.874）；1-Swap 无冻结参照（e3 未覆盖该输入），仅报告相对 Greedy 的变化。
4. "选集不同但 J 相同"：四方法两两共 **0 条**——所有 J 相等均伴随选集相等（含 Greedy≡WFC 20/20 同集同 J）。

## 2. Solver Time 对比（120 次独立计时/条件/方法，轮转交错，perf_counter_ns，热身 5 轮不计）

Strict 20 全局中位（ms）：**Greedy 0.0972**（P95 0.1371）｜Greedy+1-Swap 0.2209｜Swap-only 0.1079｜
**WFC 0.0270**｜**Exact 0.2238**（P95 0.5067，max 3.14）。

逐 K（各条件中位的均值，ms）：

| K | Greedy | Greedy+1-Swap | (Swap-only) | WFC | Exact | Exact−Greedy |
|---|---|---|---|---|---|---|
| 1 | 0.0452 | 0.0788 | 0.0349 | 0.0123 | 0.0381 | **−0.0071** |
| 2 | 0.0778 | 0.1696 | 0.0935 | 0.0210 | 0.1364 | +0.0586 |
| 3 | 0.1059 | 0.2326 | 0.1282 | 0.0281 | 0.3048 | +0.1989 |
| 4 | 0.1311 | 0.2491 | 0.1189 | 0.0345 | 0.4852 | +0.3541 |

与 e3 `timing_summary.csv` 的方法间排序完全一致（WFC < Greedy < 1-Swap ≈ Exact），绝对值因会话
负载略低，两套均为本地微基准口径。

**Setup 与求解分离**（各 5 次取中位）：Strict 快照 JSON 反序列化 5.26ms、Epoch 构造 0.049ms、收益矩阵
0.028ms；C/K3 workloads.json 反序列化 72.8ms、Epoch 0.069ms、矩阵 0.063ms。**纯求解时间比磁盘输入
解析低 2 个数量级以上**；在线路径使用内存态 Epoch，此列仅说明求解不是分配路径的成本瓶颈所在。

## 3. 严格配对质量对比（同快照、同矩阵、同 J）

Greedy **5/20 最优**（全部 K1）；**15/20 非最优**，质量损失（vs Exact）：

| 分组 | 条件数 | gap (modeled ms) | 收益比率 F/F_Exact |
|---|---|---|---|
| K1 | 0/5 | — | 1.0 |
| K2 | 5/5 | 72.7–82.6 | **0.848–0.864** |
| K3 | 5/5 | 14.7–34.2 | 0.939–0.973 |
| K4 | 5/5 | 9.8–36.6 | 0.938–0.982 |

- **Greedy→1-Swap**：13/15 条件修复至最优（残余 gap=0）；仅 w1-K3、w2-K3 残余 9.80/9.86 ms——
  局部最优需**三同步替换**才能跳出（e3 差异案例已证），1-1 邻域结构性不可达。
- **WFC≡Greedy**：20/20 同集同 J（即 5/20 最优）；但 WFC 快 2.6–3.6×。此等同只是这些输入上的
  经验事实（WFC 不读 Phi），非一般性质——C/K3 即反例（见 §5）。

## 4. Greedy / 1-Swap / Exact 的质量–延迟权衡（问题 A、B）

**A：1-Swap 修复 13 条非最优条件的 CPU 成本？** 在 15 条非最优条件上，1-Swap 额外时间（Swap-only
中位 − Greedy 中位）为 **0.066–0.154ms，中位 0.114ms**；含 Greedy 在内的总求解中位 ≤0.25ms。
即：以约 2 倍于 Greedy 的微秒级开销换取 13/15 条件最优 + 2 条件接近最优（残余 <10ms）。

**B：n=7/P=4/K≤4 时 Exact 是否便宜到 Greedy 无明显延迟优势？是。** Exact 全局中位 0.2238ms、
逐 K 开销差 −0.007～+0.354ms；K1 时 Exact 甚至更快（枚举 8 个子集 vs Greedy 多轮边际评分）。
**在该规模域内不存在支持 Greedy 的延迟论证**——0.5ms 级差异远小于输入解析（5ms）与任何在线协调
步骤。且 20/20 最优。生产规划器 `b5d1 planner.plan()` 在准入门内**本就默认 Exact**、仅在超域且显式
授权时回退 Greedy（planner.py:151-159）——实现现状与计时证据一致。

## 5. C/K3 适用性与缺失证据（问题 C：n=16，16 物理候选）

| 方法 | J (ms) | F=J0−J（J0=856.13） | 收益占比 | 求解中位 | 最优性 |
|---|---|---|---|---|---|
| Greedy | 154.349 | 701.78 | 81.9% | 0.2587ms | 未知（无参照） |
| Greedy+1-Swap | 154.349 | 701.78 | 81.9% | 0.4864ms | **1-Swap 局部最优（0 次交换）** |
| WFC | 471.874 | 384.26 | 44.9% | 0.0586ms | 未知 |
| Exact | — | — | — | — | **n=16>12 被 allowed 拒绝，留空** |

可得证据：Greedy 在该条件是 1-Swap 局部最优，且比 WFC 少付 317.5 modeled ms（收益占比 81.9% vs
44.9%；GPU 侧对应 C_B TTFT 37.2 vs 123.5ms 配对差）；求解 0.26ms 仍可忽略。缺失证据：**无任何
最优性证书或 gap 上界**（Exact 被门禁拒绝，分支定界等未实现且不属本审计）；仅 1 个条件、合成负载，
不能外推到任意 n>12 场景。`policy_plans.json` 的 SA（StandaloneTopK）选集亦为冻结参照（E=0/0/6144/7168，
被同主遮蔽），未列入本轮计时方法。

## 6. FlowState 当前求解器是否应当更换（问题 D）

**不更换，但理由必须改写。** 结果表明 Greedy 的工程合理性**不是**小规模延迟（§4：域内 Exact 更优且
≤0.5ms），而是： 超出 Exact 准入域（n>12）时是**已实现的唯一 Phi 感知选择器**（生产库唯一选择器
GlobalOptimizer；planner.plan 超域回退即落到它）； 在已审计的超域条件 C/K3 上质量最好
（1-Swap 局部最优、F 81.9% vs WFC 44.9%）、成本 0.26ms； 与 1-Swap 组合在域内可达 18/20 最优。
"原型便利"不是证据支持的理由；规模上限（794 子集门禁）才是可辩护的边界。论文不应写"Greedy 因快
被选中"，应写"准入门内用 Exact（已实现且为生产默认），超域回退 Greedy（局部最优、微秒级）"。

## 7. ICDE 可写结论与禁写

**可写：**
1. 在 n≤12、|P|≤4、K≤4 的已审计严格域内，Exact（≤794 子集枚举）中位 0.22ms、开销 ≤0.36ms 即达
   20/20 最优；Greedy 5/20、最差损失 15.2% 可达收益（F 比 0.848）；Greedy+1-Swap 以 +0.11ms 达 18/20。
2. 超域（n=16）唯一已审计条件上，Greedy 为 1-Swap 局部最优、收益占比 81.9%（WFC 44.9%），求解
   0.26ms——支持"超域回退 Greedy"的现有设计。
3. 求解时间（≤0.5ms 级）比输入反序列化低 2 个数量级，分配决策成本不在求解器选择上。

**禁写：**
- "Greedy 因延迟优势被选为分配器"（域内不成立，见 §4）；
- "Greedy/WFC 等价"（仅 20 条件同集同 J 的经验事实，C/K3 反例 F 差 317.5ms）；
- 把微基准 P95 说成在线 P95，或把 CPU 求解时间说成控制路径耗时；
- 对 C/K3 或任何 n>12 条件宣称"最优"（Exact 被门禁拒绝，无证书）；
- 用 O(·) 复杂度代替实测，或由 n=7/16 微基准外推大规模在线性能。

**判定：GREEDY_PARTIAL** —— 域内（n≤12）：现有证据**不支持**以 Greedy 取代 Exact/1-Swap（质量
损失真实、延迟理由不成立；好在生产 planner 域内本就默认 Exact）；超域（n>12）：Greedy 作为已实现
的 Phi 感知回退在唯一已审计条件上局部最优且成本可忽略，**支持保留**，但最优性无证书、样本仅 1 条件。
