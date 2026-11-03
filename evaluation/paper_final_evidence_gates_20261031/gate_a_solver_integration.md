# Gate A — Section IV-C Hybrid Solver（Exact–Greedy 混合求解机制）只读审计

日期：2026-10-31 ｜ 性质：只读审计（零 GPU、零代码修改、零历史数据修改）
审计对象：`flowstate/optimizer.py GlobalOptimizer.plan_strict` 及其严格分配调用链
git HEAD：`8cba094d2c5466759b70e4a5ef8f4824b61aade5`（2026-10-22 10:03:21 +0800）
已知状态：工作区 `flowstate/optimizer.py`、`tests/test_strict_epoch.py` 相对 HEAD 有未提交修改（即 hybrid 集成本身，+99 行；G1 `audit_readonly.md` §1 已记录此状态）。**被测代码身份以逐文件 SHA-256 锁定，不依赖 git 清洁**；下表哈希即实际运行的代码。

## A1. 调用链与 8 项机制问题（源码级证据）

### A1.1 源码身份（SHA-256，工作树实测）

| 文件 | SHA-256 | 角色 |
|---|---|---|
| flowstate/optimizer.py | `7dfa9aec307d1178d79d79ced0b5d42b0f6574754194c58d84db300c74e48176` | 混合求解入口 `plan_strict` |
| flowstate/strict_epoch.py | `5513f2703dd88e8f292c348baa07e198f30649c23c2df1a5699fe0066a1eb4a7` | Epoch 定义/验证/前沿 |
| flowstate/strict_control.py | `5ffeeaeef14606f3f1d280f41fa26a575b2f10c66ea5db28832939481a3ebab3` | `StrictReconciler` 物理协调 |
| flowstate/controller.py | `6a1ccdd9dbbcd3d1948a8d0cb7692886ae4bc1f1c7c87b51ba3ca8dfa4bb42a1` | `reconcile_strict` 委派入口 |
| flowstate/recovery_model.py | `f3fe216592ad62c26e5bf7936f907823745942f7f34b483b8dfbc2fbd8fda1f5` | 冻结 Φ（J 的成本内核） |

交叉锁定：该 `optimizer.py` 哈希逐字出现于 G1 全部 8 臂 `execution_lock_*.json` 与 `exact/source_identity.json`、五轮正式臂 `evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs/source_identity.json`（及 m5）——**GPU 门禁与五轮正式实验运行的就是当前工作树这份代码**。

### A1.2 调用链（问题 1）

```
Strict Allocation Epoch
  → GlobalOptimizer.plan_strict(epoch)                     [flowstate/optimizer.py:86]
      → epoch.validate()                                    [strict_epoch.py:101  入口全量校验]
      → score_ids 闭包 = Σ_p Φ(G_p, T_p)，G_p/T_p 由 epoch.frontiers(selected) 推得
                                                           [strict_epoch.py:175 只用显式准入边]
      → _exact_admission(n, |P|, K)                         [optimizer.py:28 规模门禁]
      │   通过 → _enumerate_exact_subsets(穷举 |S|≤K)       [optimizer.py:145]  → solver="Exact"
      │   拒绝 → _select_candidates(原 Marginal Greedy)     [optimizer.py:167]  → solver="Marginal Greedy"
      → StrictAllocationPlan + last_strict_decision 决策记录 {solver, gate_allowed, gate, selected_ids, objective_ms, pure_solve_ms}
  → StateController.reconcile_strict(epoch, plan, runtime)  [controller.py:114 → strict_control.py:193]
      → validate_plan → bind → preflight → 逐驱逐 before/eviction/postcheck → final_postcheck
      → 一次性总账；任何失败 → status=FAIL + failure_view + invalidate + 抛 StrictReconciliationError（无重试/回退/部分计划）
```

生产调用点（evaluation 侧唯一入口，主库零反向依赖）：`evaluation/p0_control_cycle_gpu/control.py:219`（三策略统一在线选择，FS=生产入口 `plan_strict`）；G1/Spec8 门禁在 Scheduler 进程内 `from flowstate.optimizer import GlobalOptimizer` 直接调用（`hybrid_solver_gpu_gate_g1/pkg_exact`、`paper_hybrid_gpu_gate_*`，见 A3）。

### A1.3 逐问作答（问题 2–8）

| # | 问题 | 答案与代码证据 |
|---|---|---|
| 2 | 规模门禁条件 | 同时检查四项：候选 n≤12、Pending ≤4、K≤4、可枚举子集数≤794（`_EXACT_MAX_*` 常量 [optimizer.py:22-25](../../flowstate/optimizer.py#L22-L25)；`_exact_admission` [:28](../../flowstate/optimizer.py#L28) 以 `sum(comb(n,i), i=0..min(n,K))` 计算子集数）。注释明确"严格沿用已验证边界(b5d1 合同)，只做规模判定，不扩大支持范围"。 |
| 3 | 超限行为 | 门禁拒绝 → 自动进 Marginal Greedy（`_select_candidates`，legacy 循环逐行未动），`last_strict_decision.solver="Marginal Greedy"`、`gate_allowed=False`。运行中一致性已验：Strict20 全部 20 次 `gate_allowed=True↔Exact`；C/K3 `gate_allowed=False↔Greedy`（n=16，子集 697，n>12 先拒）。 |
| 4 | Exact 选择规则 | 穷举全部 \|S\|≤K 子集，共享同一 `score_ids` 评分；**容差 `_FLOAT_TOLERANCE_MS=1e-9` 内取项数最少者，再按 ID 字典序**（与 b5d1 `choose_exact` 同规则迁移）。 |
| 5 | 同一 J、同一 Epoch | 两分支共用同一 `score_ids` 闭包（frontiers + 冻结 Φ），J 定义在同一 Strict Epoch 上完全一致；负总收益直接 `ValueError`（不允许"负改进"计划）。 |
| 6 | 计划落实 | 两分支产出同型 `StrictAllocationPlan`，统一经 `reconcile_strict` 一次性物化（bind/preflight/驱逐/postcheck 全阶段总账）；`validate_plan` 在协调前强制通过。 |
| 7 | 错误处理 | **无静默回退**：`validate` 拒绝（缺身份/摘要非法/Legacy 混用/Token 不一致/候选状态不完整/非等大小等，[strict_epoch.py:101](../../flowstate/strict_epoch.py#L101)）、求解器内部异常、负收益 ValueError 均原样上抛；不返回部分计划，`last_strict_decision` 不写入成功记录。 |
| 8 | 决策可审计性 | `last_strict_decision` 每次成功决策记录求解器身份/门禁判定/选集/目标值/纯求解时间——五轮正式臂 `reconciliation.json` 的 `strict_decision` 字段即此记录的落地产物。 |

## A2. CPU 门禁证据（既有报告，本审计只读引用）

来源：`evaluation/paper_hybrid_solver_integration_gate.md` + 机内原始结果 `.json` + 门禁脚本 `.py`（2026-10-23，判定 PASS）。**被测入口为正式 `GlobalOptimizer.plan_strict`，非独立 Planner**（其 Strict20 结果另行与两条独立 Oracle 交叉：e3 冻结 Exact 行、b5d1 `solve('Exact')` 另一实现路径，20/20 全等）。

| 门禁 | 结果 |
|---|---|
| Strict20（20 例，K=1–4，n=7、子集 8/29/64/99） | **20/20 走 Exact 分支**；选集与 J 对 e3 冻结行 20/20 集合相等、\|ΔJ\|≤1e-9；对 b5d1 独立 Oracle 20/20 全等；同 Epoch 直调原 `_select_candidates` 20/20 复现 e3 Greedy 行（Greedy 行为不变性） |
| C/K3 回退（n=16>12） | 自动 Marginal Greedy（纯求解 0.439ms）；选集 {NODE_57,56,55} == 冻结 GPU plan == policy_plans FlowState-Greedy（census 映射）== CPU 冻结选集；J=154.34895561599998 与冻结 GPU/CPU 逐位一致；逐 Pending E=(0,5120,6144,7168) 相等 |
| 错误路径（3 类注入） | 删边断裂/边状态 UNKNOWN → `validate` 抛错拒绝；mock Exact 内部异常 → 原样上抛；三者均**无 Greedy 回退、无计划返回、无成功决策记录** |
| 回归 | docker（CPU-only）10 个测试文件 **122 passed + 16 subtests**；`test_strict_epoch` 冻结 16 条件按混合合同通过（g002/g003 超域仍对冻结 Greedy 参照逐序断言） |
| 边界纯函数 | (12,4,4)=794 恰好压线准许；(13,4,4)/(12,5,4)/(12,4,5)/(16,4,3) 拒绝。附注：n≤12∧K≤4 时子集数恒≤794，794 为对齐冻结合同的冗余守护 |

该报告自身明确边界（§7.4）：CPU 集成门禁**不构成**在线 GPU 端到端验证——该缺口由 A3 的 GPU 证据补齐。

## A3. GPU 证据（区分"在线 Exact 求解路径"与"执行冻结 Exact 选集"）

按 Spec 要求分三类，**不混同**：

### 第一类：GPU 实际执行在线 Exact 求解路径（存在，两次独立验证）

1. **Spec 8 场景A**（`evaluation/paper_hybrid_solver_gpu_gate.md` + `paper_hybrid_gpu_gate_a/gate_a_result.json`，2026-10-24，**21/21 PASS**）：B5-C4、K=2。Scheduler 进程内经 `b5d4_apply` RPC 直接调用 `plan_strict` 在线求解；`strict_decision` 由运行写入 `reconciliation.json`：solver=**Exact**、gate_allowed=true、gate={candidates:7, pendings:4, budget_k:2, enumerable_subsets:29}、selected={B5C3_NODE_18, B5C3_NODE_27}、J=48.228728759089606、pure_solve 0.223ms。选集 **≠ Greedy 选集 {N20,N18}**（证明 Exact 分支真实改变选择，非复刻 Greedy），== b5d4b 冻结 Exact 臂 == 运行内独立 Decimal Oracle（29 子集穷举）== e3 冻结行。物理落实：补集 5/5 驱逐 returned、逐 Pending E/T/G==冻结 Exact 臂、正式请求 8 token 输出==b5d4b Cold 参照、请求后 FA 内容 sha256 不变。
2. **G1 实验A**（`evaluation/hybrid_solver_gpu_gate_g1/final_report.md` + `checks.json` + `exact/verify.json`，2026-10-25，**4/4 臂 PASS**）：w1/K2（NEW 场景，E4-1A 冻结 w1 历史）。完整证明链在单受控 Scheduler 生命周期内：真实物理状态观察 → Epoch 构造与验证 → **`plan_strict()` 在线求解（hybrid 自动分支）** → Plan 校验 → **`reconcile_strict()` 用本轮 Plan 物理协调** → Native Resume → Cold 逐 Token 一致（4/4 臂 8/8 token）。在线选集 {N18,N21}、J=53.77075886917578 ==e3 冻结行逐位。`final_report.md` §六.4 明确："两路径控制流均为「观察→在线 plan_strict→validate→用该 Plan 协调」；冻结选集只在求解完成后作对照（greedy）或作为容器内 Exact 参考 oracle（exact）……**未通过外部参数指定 Exact/Greedy**"。

### 第二类：GPU 实际执行在线 Greedy 求解路径（超门禁自动降级）

Spec 8 场景B（C/K3，**22/22 PASS**）+ G1 实验B（**4/4 PASS**，n=16>12 → gate_allowed=False → Marginal Greedy，在线选集==冻结 FS 三元集，J 逐位，13 驱逐全 returned，逐臂 8/8 token==Cold）。

### 第三类：执行冻结/预先算好的选集（**不构成在线 Exact 证据**）

- 五轮矩阵 15 臂（`evaluation/p0_control_cycle_gpu/arms/`，含 FS 臂）：FS 臂在线调 `plan_strict` 但 n=16 → **Marginal Greedy 分支**（`kernel_entry=flowstate.optimizer.GlobalOptimizer.plan_strict`，selection_call_ms 中位 16.38）；MC/KV 走各自真实内核。三策略选集均与冻结计划 oracle_match=True。**五轮性能数字全部出自 Greedy 分支或非 FS 内核，不得表述为 Exact 求解结果**。
- b5d4/b5d4b（exact_greedy_paired_gpu_*）Exact 臂：执行预先冻结的 Exact 选集（正确性参照）——属 Gate B 审计范围。
- C/K3 三个正式实验（pilot/pair-timing/formal）：冻结选集协议。

### A3 边界（必须写入论文的限制）

1. 在线 Exact 的 GPU 执行验证只覆盖**门禁规模**（n=7、\|P\|=4、K∈{1,2}；Spec8 A 与 G1 A 各一个工作负载），未在更大门内规模（n 至 12、K 至 4）重复。
2. Epoch 构建为"冻结历史重放 + 实时严格观察准入"（两报告均自我标注"非全在线 Epoch 构建"）；求解与协调本身完全在线。
3. 门禁为正确性/执行路径验证，单臂单次运行；计时字段仅为执行记录，**不构成性能口径**（两报告原文均作此声明）。
4. 五轮矩阵 FS 臂与 C/K3 正式实验因 n=16 结构性超门禁，按设计走 Greedy 分支——"生产混合入口"在正式性能实验中的实测形态是 Greedy 分支。

## A4. 裁决

**HYBRID_INTEGRATED_GPU_VERIFIED**（附范围限定）

混合求解机制已集成进生产主入口并在 GPU 上实际执行过在线 Exact 求解路径（含物化与逐 Token 正确性）：CPU 侧 Strict20 20/20（对双独立 Oracle）、C/K3 回退逐位一致、错误路径 fail-stop、122 测试；GPU 侧 Spec8 A 21/21 + G1 双路径 8/8。代码身份 SHA-256 与全部 GPU 执行锁交叉锁定。

各章节允许宣称：

| 位置 | 允许 | 不允许 |
|---|---|---|
| Section IV-C（机制描述） | 全部机制事实：四项门禁阈值(12/4/4/794)、容差 1e-9/最少项/字典序 tie-breaking、超域自动 Marginal Greedy、fail-stop 无静默回退、`last_strict_decision` 决策记录、单一 J 闭包 | 不得暗示 Exact 无规模上限或门禁可在线扩大 |
| Section V（方法/实现验证） | "生产入口 `plan_strict` 已混合集成并经 CPU 门禁（Strict20 20/20 对双 Oracle、C/K3 回退逐位、122 测试）与 GPU 门禁（Scheduler 进程内在线求解→`reconcile_strict` 物化→逐 Token 正确，Spec8 21/21+22/22、G1 8/8）验证" | 不得宣称在线 Exact 已在**生产规模性能实验**中运行；须注明 GPU 在线 Exact 验证限于门禁规模(n=7,K≤2)且非全在线 Epoch 构建 |
| Evaluation（性能数字） | 五轮矩阵/C/K3 数字如实归于 **Greedy 分支**（"FS 生产入口在线 Greedy 选点，与冻结计划逐位一致"）；B5-C4 Exact vs Greedy GPU 对照（Gate B 范围）按其自身证据链表述 | 不得把五轮 −470/−430ms 或任何 C/K3 数字归因于 Exact 求解；不得把门禁计时当性能口径 |

**证据缺口（如实记录，不构成 BLOCKED）**：门内更大规模（n≤12、K=3/4）的在线 Exact GPU 执行未单独验证（Strict20 已在 CPU 覆盖 K=1–4 全部 20 例，GPU 仅 K=2 两工作负载）。此为可选补强项，见 claim_evidence_matrix.md。
