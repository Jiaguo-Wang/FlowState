# FlowState — 实际求解器调用路径核查（Section IV-C 定位）

日期：2026-10-23 ｜ 只读审计，零代码/数据/论文修改 ｜ 判定：**PARTIAL**

证据等级声明：本文每一条结论均来自 (a) 当前代码的调用链（`import`/调用点逐条 grep 核实），或 (b) 冻结
执行证据（各目录 report/finalize/policy_plans 等）。不依据函数命名或设计意图作任何推断。

---

## 0. 一句话结论

`flowstate/` 主库的真实求解入口**只有 Marginal Greedy**（无枚举代码、不依赖 evaluation/）；Exact 及
"门内 Exact、超域 Greedy"切换只存在于 `evaluation/exact_planner_contract_gate_b5d1/planner.py`，其
`plan()` 从未被任何在线控制路径调用（仅 b5d1 自身离线合同测试执行并验证过）；Exact 曾在多条**实验性
在线控制入口**（b5d3/b5d4/b5d4a/E1/E5）中被外部指定并经生产 `reconcile_strict` 真实物化；而论文主性能
证据 C/K3 正式 GPU 配对**不在线求解**，使用冻结 CPU Greedy 选集在线认证后物化。多入口并存、无统一
混合求解路径。

## 1. 调用链全景

| 入口 | 求解器 | 在线/离线 | 关键代码位置 | 执行证据 |
|---|---|---|---|---|
| `GlobalOptimizer.plan_strict`（严格生产入口） | **仅 Marginal Greedy** | 生产库 | [optimizer.py:52-76](../flowstate/optimizer.py#L52-L76) → `_select_candidates`（[optimizer.py:78](../flowstate/optimizer.py#L78)） | b5d1 report.py:72"生产plan_strict的20条件Greedy回归也通过" |
| `GlobalOptimizer.select`（Legacy 生产入口） | 仅 Greedy | 生产库（legacy） | [optimizer.py:40-49](../flowstate/optimizer.py#L40-L49) → `_select_candidates` | RQ3/RQ5/RQ6 等全部历史评估调用 |
| `StateController.reconcile` | 调 `optimizer.select`（Greedy） | 在线（legacy 重选+驱逐） | [controller.py:55-112](../flowstate/controller.py#L55-L112) | r2b/r2c 等运行记录 |
| `StateController.reconcile_strict` | **不选求解器**，只执行现成 plan | 在线（严格协调） | [controller.py:114-118](../flowstate/controller.py#L114-L118) → [strict_control.py:193](../flowstate/strict_control.py#L193) | b5d3/b5d4/C-K3 等 |
| b5d1 `planner.plan()`（**唯一混合切换点**） | 门内 Exact；超域 LimitError；`allow_fallback=True` 才 Greedy | **仅离线** | [planner.py:151-159](exact_planner_contract_gate_b5d1/planner.py#L151-L159)、[133](exact_planner_contract_gate_b5d1/planner.py#L133) | 调用方全库仅 [run.py:39,53](exact_planner_contract_gate_b5d1/run.py#L39)、[verify.py:45](exact_planner_contract_gate_b5d1/verify.py#L45)；report.py:72（12 条件 Exact+Decimal Oracle、8 条件显式 Fallback） |
| b5d3 在线控制函数 | **硬编码 `'Exact'`**，`require(allowed)` 否则 BLOCKED | 在线（单 Scheduler 调用，真实驱逐） | [control.py:68-71](exact_reconciliation_correctness_gate_b5d3/control.py#L68-L71)、[115](exact_reconciliation_correctness_gate_b5d3/control.py#L115) | finalize.py:85（实际驱逐、Postcheck 通过） |
| b5d4 / b5d4a 配对 GPU | `solver=os.environ['B5D4_SOLVER']`（外部逐臂指定 Greedy/Exact），`require(allowed)` | 在线 | [b5d4/control.py:68-76](exact_greedy_paired_gpu_gate_b5d4/control.py#L68-L76)、[b5d4a/control.py:75-83](exact_greedy_paired_gpu_fix_b5d4a/control.py#L75-L83) | 各臂 reconciliation.json（finalize.py:93） |
| E1 轻量观察 / E5 Exact+Swap | 同上 env 指定；E5 的 Swap 臂**读冻结计划**不在线解 | 在线 | [e1/control.py:41-42,79-81](lightweight_runtime_e1/control.py#L41-L42)、[e5/control.py:79-81](exact_swap_runtime_e5_step1/control.py#L79-L81)、[e5/planner.py:11-21](exact_swap_runtime_e5_step1/planner.py#L11-L21) | 各自 finalize |
| **C/K3 正式 GPU 配对**（c_formal_perf_gpu / c_shared_gpu_pilot_k3 / c_pair_timing_gpu） | **不在线求解**：policy_plans.json 冻结选集 → `from_frozen/prepare/finish` 在线认证 | 在线物化、离线选集 | [c_formal_perf_gpu/control.py:1-8](c_formal_perf_gpu/control.py#L1-L8)（docstring 明示）、[112-118](c_formal_perf_gpu/control.py#L112-L118)、[192](c_formal_perf_gpu/control.py#L192) | policy_plans.json `selection_source`：FlowState-Greedy/WFC←cpu_results.json、StandaloneTopK←review_results.json |
| C/K3 冻结选集产生处 | `GlobalOptimizer._select_candidates`（Greedy，CPU 离线） | 离线 | [run_cpu.py:32](kvflow_multiworkflow_cpu_main/run_cpu.py#L32) | cpu_results.json（30 行冻结） |
| e2（w1–w4 离线评估） | Greedy 走 `d.solve('Greedy')`；Exact 是**自建穷举 Oracle**（域 n≤24/≤20000，宽于 allowed） | 离线 | [evaluate.py:33-41](independent_runtime_allocation_e2/evaluate.py#L33-L41) | allocation_results.json |
| b5d2 shadow | Greedy/Exact 并算计时 + Oracle，"不访问Runtime" | 离线 | [shadow.py:20-24](exact_runtime_shadow_gate_b5d2/shadow.py#L20-L24) | current_epoch_oracle.json |

依赖方向核实：`grep -rn "evaluation" flowstate/ --include=*.py` = **0 条**（主库不 import 任何评估
代码）；`grep -rn "Exact|itertools|combinations" flowstate/*.py` = **0 条**（主库无子集枚举求解器）。

## 2. 逐题回答

**Q1：`planner.plan()` 是否被主库或实际控制入口调用？** 否。全库 `.plan(` 调用仅存在于 b5d1 自身的
[run.py:39,53,79-87](exact_planner_contract_gate_b5d1/run.py#L39) 与 [verify.py:42-45](exact_planner_contract_gate_b5d1/verify.py#L42-L45)
（离线合同门禁测试，含拒绝路径测试）。`flowstate/` 对 planner 零引用；所有在线控制入口（b5d3/b5d4/
b5d4a/E1/E5/C-K3）都绕过 `plan()`，直接组合 `from_frozen/prepare/benefits/solve/finish`。

**Q2：`allowed()` 门禁是否进入主在线选择路径？** 没有。它出现于三类位置：(a) 在线控制函数的**前置
require 断言**（[b5d3:69](exact_reconciliation_correctness_gate_b5d3/control.py#L69)、[b5d4:69](exact_greedy_paired_gpu_gate_b5d4/control.py#L69)、
[b5d4a:76](exact_greedy_paired_gpu_fix_b5d4a/control.py#L76)、[E1:42](lightweight_runtime_e1/control.py#L42)、[E5:42](exact_swap_runtime_e5_step1/control.py#L42)）——
不满足即整体 BLOCKED，**不是"超域换 Greedy"的切换**；(b) `plan()` 内部切换（仅离线执行过）；
(c) e2 离线一致性断言（[evaluate.py:39](independent_runtime_allocation_e2/evaluate.py#L39)）。
C/K3 正式 GPU 路径**明确不调用** allowed（[c_formal_perf_gpu/control.py:7,118](c_formal_perf_gpu/control.py#L7)）。
另注意命名陷阱：e2 的状态标签 `ONLINE_DOMAIN_EXACT` 指"落在 planner.allowed 域内可复算"，与在线执行无关。

**Q3：`plan_strict()` 实际调用哪个求解器？** 仅 `_select_candidates`（Marginal Greedy，
[optimizer.py:70-71](../flowstate/optimizer.py#L70-L71) 的 `allocation = self._select_candidates(...)`）。
函数体内无求解器分支、无规模判断、无 Exact 调用。

**Q4：C/K3 正式 GPU 实验是否在线 Select？** 否。入口是 `policy_plans.json` 冻结 CPU 选集
（FlowState-Greedy/StandaloneTopK/WFC），在线仅做 `from_frozen→prepare→finish` 认证与生产
`reconcile_strict` 物化（[control.py:109-121](c_formal_perf_gpu/control.py#L109-L121)）；冻结选集由
[kvflow_multiworkflow_cpu_main/run_cpu.py:32](kvflow_multiworkflow_cpu_main/run_cpu.py#L32) 的
`GlobalOptimizer._select_candidates`（Greedy）在 CPU 离线产生。

**Q5：是否存在"已实现、实际调用且经过验证"的规模切换？** **不存在接入真实路径的切换。** 切换逻辑
已实现（`plan()`，planner.py:151-159）且被离线执行验证（b5d1：12 条件走 Exact、8 条件显式
`allow_fallback=True` 走 Greedy，report.py:72）；但在线实验中的求解器全部由外部指定（b5d3 硬编码
'Exact'；b5d4/b5d4a/E1/E5 环境变量白名单），且都带 `require(allowed)`——**在线 Exact 只在门内执行
过，在线"超域回退 Greedy"从未发生过**（超域即 BLOCKED）。

**Q6：Section III / IV-C 能准确宣称什么？**

可宣称（有直接证据）：
1. 生产库唯一选择器为 Marginal Greedy（`GlobalOptimizer`，严格入口 `plan_strict`、Legacy 入口 `select`）。
2. Exact 是独立的纯规划适配器（b5d1 planner），带 n≤12、|P|≤4、K≤4、≤794 子集的准入门；门内默认
   Exact，超域默认拒绝、显式授权才回退 Greedy；该合同经 20 条件离线验证（含 Decimal 独立 Oracle）。
3. 门内在线 Exact 已在实验控制路径真实执行并物化（b5d3 单臂、b5d4/b5d4a Exact/Greedy 配对臂、
   E1/E5），全部经生产 `StateController.reconcile_strict` 与后置检查。
4. C/K3 正式 GPU 配对使用冻结 CPU 选集在线认证物化，Select 不在在线路径（诚实描述实验协议）。

禁宣称（无实现或无验证）：
- "系统运行时按规模自动切换 Exact/Greedy"——切换从未被任何真实控制路径调用；
- "Exact 是生产/主控制求解器"——`flowstate/` 无枚举代码、不依赖 planner；
- "C/K3 GPU 实验在线执行了求解/Select"；
- 把 e2 的 `ONLINE_DOMAIN_EXACT` 标签当作在线执行证据；
- 把 b5d3/b5d4 的实验性 Exact 臂说成生产路径（求解器是外部指定/硬编码的实验臂）。

## 3. 判定：**PARTIAL**

- HYBRID_INTEGRATED 不成立：混合切换 `plan()` 无任何在线/生产调用点（§2 Q1/Q5）。
- GREEDY_ONLY 不完全成立：其表述"Exact 独立存在"低估了证据——Exact 在 5 条实验性在线控制入口经
  生产协调真实执行过（含真实驱逐与 Oracle 复核），并非仅离线存在。
- PARTIAL 成立：主控制库 Greedy-only、实验性在线 Exact 臂、离线混合切换、正式 GPU 冻结选集——
  **多个入口并存，不能证明统一的混合求解路径**。若论文需要单一表述：生产控制路径 = Greedy-only；
  Exact = 已验证的离线/实验性门内求解器；"自动规模切换"= 已实现未接入。

对 Section IV-C 的直接建议：求解器描述应落在"Greedy 为唯一生产选择器 + Exact 作为带准入门的离线
认证求解器（并在配对实验臂中在线执行过）"这一可证据化表述上；与 Spec 5（GREEDY_PARTIAL）结论合并
阅读：域内质量-延迟证据支持 Exact 优先，但接入生产切换属于**未来工作**，不是现状。
