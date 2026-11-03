# FlowState — 严格分配路径 Hybrid Solver 最小集成门禁报告

日期：2026-10-23 ｜ 判定：**PASS**（CPU 正确性门禁；不等于在线 GPU 端到端验证）
被测入口：`flowstate/optimizer.py GlobalOptimizer.plan_strict` ｜ 门禁脚本：`paper_hybrid_solver_integration_gate.py`（同目录）
机内原始结果：`paper_hybrid_solver_integration_gate.json`

---

## 1. 修改的文件及入口

| 文件 | 修改 | 未改动 |
|---|---|---|
| [flowstate/optimizer.py](../flowstate/optimizer.py) | ① `plan_strict` 重构为混合求解（规模门禁内 Exact 枚举、超域原 Greedy）；② 新增模块级 `_exact_admission`（准入门禁常量 12/4/4/794）；③ 新增 `_enumerate_exact_subsets`（迁移自已验证 Exact 实现：穷举 \|S\|≤K、容差 1e-9 内最少项再字典序——与 b5d1 `choose_exact` 同规则）；④ `__init__` 新增 `last_strict_decision` 决策记录（求解器/门禁判定/选集/目标值/纯求解时间）；⑤ 模块 docstring 与导入更新 | **Legacy `select`、`_select_candidates`（Greedy 循环逐行未动）**、`AllocationResult`、其余全部生产模块（`strict_epoch`/`strict_control`/`controller`/`state_catalog`/`recovery_model` 零改动） |
| [tests/test_strict_epoch.py](../tests/test_strict_epoch.py) | `test_frozen_16_conditions` 更新为混合合同：门内条件对照冻结 `Exact OPT` 参照（字典序终集 + J≤Greedy 参照），超域条件保持原 FlowState/Greedy 轨迹断言；新增求解器身份断言 | 其余 15 个测试与全部负例不变 |

依据规约：本轮允许修改必要求解器代码；**未**改 Runtime 驱逐/状态映射/物理协调、冻结选集与实验结果（20 个被读文件首尾哈希核对一致）、论文正文；未启动 GPU；未重建实验矩阵。

## 2. 新旧调用链对照

```
旧：plan_strict → epoch.validate → _select_candidates(Greedy) → StrictAllocationPlan
新：plan_strict → epoch.validate → score_ids(frontiers+冻结Phi 闭包，两分支共用)
     ├─ _exact_admission 通过 → _enumerate_exact_subsets(穷举+同分规则) → 终值评分 → Plan   [solver=Exact]
     └─ _exact_admission 拒绝 → _select_candidates(原 Greedy，未改)            → Plan   [solver=Marginal Greedy]
     每次成功决策写入 opt.last_strict_decision；任何异常原样上抛（无回退）
```
Exact 实现迁移自 b5d1 `planner.solve('Exact')`/`choose_exact`（[planner.py:123-139](exact_planner_contract_gate_b5d1/planner.py#L123-L139)），输出语义/容差/Tie-breaking 逐条保持；**运行时不依赖 evaluation/**（依赖方向仍为 evaluation→flowstate，主库零反向引用）。

## 3. Exact / Greedy 门禁判定

- 门禁条件：n≤12 且 \|P\|≤4 且 K≤4 且枚举子集数≤794（[optimizer.py](../flowstate/optimizer.py) `_exact_admission`）。
- 边界纯函数验证：`(12,4,4)`=794 子集→**准许**（恰好压线）；`(11,4,4)`=562→准许；`(13,4,4)`、`(12,5,4)`、`(12,4,5)`、`(16,4,3)`→**拒绝**。附注：n≤12∧K≤4 时子集数恒≤794（794=C(12,≤4)），该上限是对齐冻结合同的冗余守护，未删除。
- 运行中一致性：Strict 20 全部 20 次决策 `gate_allowed=True↔solver=Exact`；C/K3 `gate_allowed=False↔solver=Marginal Greedy`（n=16、子集 697，但 n>12 先拒）。
- 仅规模可切换：合同错误/UNKNOWN/求解器异常路径均不产生 Greedy 回退（见 §6）。

## 4. Strict 20 正确性结果（A）

输入：w1–w4（e2 冻结快照链）+ **B5-C4 的严格来源=b5c3 快照**（已核实与 b5c4/results.json 抽象输入的位置/兼容/目标逐项相等；抽象输入本身无严格形态）。Epoch 经 b5d1 `from_frozen` 构造（仅评估侧输入构造工具），逐条通过生产 `validate_plan`。

- **20/20 走 Exact**；门禁值 n=7、\|P\|=4、子集数 8/29/64/99（K=1–4）。
- 选集与 J 对 **e3 冻结 Exact 行**：20/20 集合相等、\|ΔJ\|≤1e-9。
- 对 **b5d1 `solve('Exact')` 独立 Oracle**（另一实现路径）：20/20 全等。
- **Greedy 行为不变性**：同 Epoch 直接调原 `_select_candidates`，20/20 复现 e3 Greedy 行选集。
- Exact 纯求解时间：min 0.092 / 中位 0.449 / max 0.718 ms（单进程连续运行，与 Spec 5 微基准同量级；非生产在线耗时）。

## 5. C/K3 回退正确性结果（B）

输入：`c_formal_perf_gpu/g1_CB_FS/run/reconciliation.json` 冻结的 `raw_epoch`（GPU 正式实验实际认证的同一 Epoch；重建字节 sha256==记录的 `snapshot_sha256`；physical 视图按冻结运行推导——`unchanged=True` 由当时 `prepare` 已通过反证、`window` 取 `apply_validation` 记录值；origins 用冻结记录）。

- 门禁拒绝（n=16>12）→ **自动 Marginal Greedy**，纯求解 0.439ms。
- 选集 `{NODE_57, NODE_56, NODE_55}` == 冻结 GPU plan == `policy_plans.json` FlowState-Greedy（经 census cpu↔physical 映射）== cpu_results 冻结选集。
- J=**154.34895561599998** modeled ms，与冻结 GPU plan J、CPU 冻结 J 逐位一致；`validate_plan` 通过；逐 Pending E=(0,5120,6144,7168) 与冻结记录相等。

## 6. 错误路径和 Fail-stop 结果（C）

| 注入 | 结果 |
|---|---|
| 删除一条边（完整 n×m 矩阵断裂） | `validate` 抛错拒绝；**无 Greedy 回退、无计划返回、无成功决策记录** |
| 边状态改为 UNKNOWN | 入口即拒；同上三条 |
| 注入求解器内部异常（mock `_enumerate_exact_subsets` 抛 RuntimeError） | 异常原样上抛；无部分有效计划；`last_strict_decision` 保持 None |

回归测试：docker（CPU-only）内 10 个测试文件 **122 passed + 16 subtests**（含 `test_optimizer`/`test_controller`/`test_strict_control` 等）；`test_strict_epoch` 冻结 16 条件按混合合同通过——其中 g002/g003 超域条件仍对冻结 Greedy 参照逐序断言，构成**超域回退的又一组冻结回归**；g001/g004 门内条件对冻结 `Exact OPT` 参照断言且 J≤Greedy 参照。

## 7. 对正式 GPU 实验的追加验证建议

1. **现有正式 GPU 结论不受影响**：C/K3 三轮（pilot/pair-timing/formal）均为冻结选集协议，本轮未改任何冻结选集或 Runtime 物化代码。
2. 若论文要宣称"生产控制路径已混合求解"，需一轮**在线验证**：在门内小规模臂上用新 `plan_strict` 在线选点并经生产 `reconcile_strict` 物化，对照冻结选集臂的 E/T/G 与计时合同（b5d4/b5d4a 已有的在线 Exact 臂证据可复用为门内在线执行参照）。
3. C/K3（n=16）在线行为应与冻结 FS 选集一致——本轮已 CPU 级证明等价（§5），GPU 端只需一轮抽样确认即可闭环。
4. 明确边界：本轮 PASS 是 CPU 集成门禁，**不**构成在线 GPU 端到端验证，论文表述应区分。

**判定：PASS** —— 主 Strict 入口（`plan_strict`）已集成 Hybrid Solver（门内 Exact、超域 Greedy、Fail-stop 无静默回退），A/B/C 三组 CPU 正确性门禁与既有回归全部通过。
