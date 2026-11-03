# Task 1：三策略完整控制周期 Runner 的 CPU 集成与门禁（2026-10-29）

## 0. 最终判定：**READY_FOR_GPU**

CPU 门禁 32/32 全部通过（`python3 -m evaluation.p0_control_inclusive_harness.cpu_gate_test` 可复算）。三种策略均经真实内核在线 Select 并复现冻结 oracle；T0–T4 编排与整数纳秒恒等式精确成立；取证与测量窗口结构分离；异常路径全部 fail-loud。剩余 7 项为**只能在 GPU 验证**的事项（已登记，不以 CPU 通过替代）。第一轮 GPU 实验的启动步骤见 §8（机械布线，全部复用已验证组件，无新算法逻辑）。

## 1. 真实调用链核查结论（spec §二）

| 环节 | 真实路径 | 证据 |
|---|---|---|
| FS 在线选择 | **生产入口 `GlobalOptimizer.plan_strict()`**（非内部内核、非冻结加载） | spy 直证调用 1 次；`last_strict_decision`：solver=Marginal Greedy、gate={16 候选,4 Pending,K=3,enumerable_subsets=697}（n=16 超门禁自动 Greedy） |
| MC 在线选择 | 真实内核 `MarconiEvictPolicy.select`（逐轮动态驱逐，规则零修改） | spy 直证；选集 {C_JOIN_P04032, C_JOIN_P04096, C_ROOT_P01024} == 冻结 oracle |
| KV 在线选择 | 真实内核 `kvflow.select`（ASG/STE，known_at_epoch 门控） | spy 直证；选集 {C_AB_P01984, C_AB_P02048, C_A_P04032} == 冻结 oracle |
| Epoch 构造 | `planner.from_frozen/prepare` 生产认证链（逐字复刻 integration gate 已证链路，复现冻结 J=154.34895561599998） | g1_CB_FS 冻结 reconciliation + snapshot_sha256 逐位校验 |
| Plan 构造 | FS=plan_strict 产物；MC/KV=**`planner.finish(prepared, selected)`**——与正式 b5d4_apply 为冻结选集建计划同一生产路径；三者均过 `validate_plan` | 60/60 冻结臂同路径先例 |
| census 对账 | 复用 G1 `physical_cpu_census` 纯函数，CPU 复算复现冻结 16=16 映射 | 检查 5b |
| 协调连接 | plan.logical_eviction_ids 驱动 StrictReconciler（策略无关，GPU 阶段真实物化） | 记录 evicted 补集 + MC 逐轮驱逐序（物理域） |

各策略决策可见输入如实记录（`inputs_visibility` 字段）：FS=Strict Epoch 快照域+Φ；MC=声明候选+访问序+父链（等大小字节）；KV=声明候选+图+ASG/STE。无未来信息；oracle（t2g1_plans.json）仅在**全部选择完成后**读取对照，其耗时单独记为 `oracle_check_ms`，绝不计入 Select。

## 2. 最小 Harness 集成（spec §三，5 项）

新增包 `evaluation/p0_control_inclusive_harness/`（G1 pkg 复制模式，零核心/Runtime/正式 Runner 改动）：

| # | 要求 | 实现 | 复用来源 |
|---|---|---|---|
| ① | 三策略真实 Select 接入统一入口+分别计时 | `select_entry.py`：build_ck3_epoch → select_flowstate/marconi/kvflow → run_three_policy_select（每策略 selection_call_ms+内核入口+决策记录） | adapter 调用方式+integration gate 构造链+pkg_greedy census |
| ② | 四 Pending 串行执行流程 | `cycle_orchestrator.py` RequestBackend 抽象 + 串行循环（每请求 preflight→TTFT→执行终点；请求间隙显式计量） | resume_epoch 模式（重写为轻量编排，非搬运旧重取证路径） |
| ③ | 统一 T0–T4 计时 | CycleClock 整数纳秒顺序边界分段；恒等式 `(T4−T0)==Σ分段+Σ间隙` **整数算术精确**断言 | resume_epoch 恒等式思想+G1 timed 分段 |
| ④ | 取证与测量窗口分离 | OutOfWindowEvidence：窗口内只入队，T4 后统一 flush（flush_ms 单列）；必要在线检查保留窗内 | P0-3 批量/出窗结论 |
| ⑤ | 完整记录 | selection_records.json：输入可见性/候选域/预算/选集（CPU+物理双域）/驱逐集（含 MC 驱逐序）/J/call_ms | — |

禁止重复实现的核查：三内核、epoch 构造、census 纯函数、Plan 构造/校验全部 import 既有实现，零复制。

## 3. CPU 正确性门禁结果（32/32 PASS）

1. **真实调用**：spy 计数 {plan_strict:1, marconi:1, kvflow:1}；FS solver/gate 合同；三选集复现冻结 oracle（集合语义）。
2. **记录完整**：8 必需字段×3 策略；预算 K=3；域⊆16；选集∪驱逐=全体且不相交。
3. **计时无重复/遗漏**：三策略干跑周期恒等式精确（cycle=12.81/12.85/12.97ms == Σ分段+Σ间隙，整数 ns）；分段=4(T0–T3)+2×4(请求)，顺序边界无重叠。
4. **四 Pending 关联**：执行顺序==声明顺序 [C_A,C_B,C_C,C_D]；每请求 0<TTFT<终点；请求间 exec_end 严格单调；T4==第 4 请求终点（offset≥0 已采样延迟如实记录）。
5. **Strict 合同保留**：validate_plan×3；census 复算==冻结 16=16；分段注册表含认证/对账/选择/协调全合同；协调后验（FA/非R摘要不变、槽位释放记账、驻留==Plan）在窗内；GPU 物化部分显式标注待验证。
6. **fail-loud**：内核注入异常直接传播（ValueError，无兜底）；注入错误选集被 oracle 拒绝（AssertionError，证明无静默采纳通道）；三个选择函数源码结构上不含任何冻结选集读取。
7. **冻结数据未破坏**：6 个冻结输入文件 sha 前后逐位一致；harness 仅写本包 `out/`。
8. **GPU 待验证登记**：7 项（见 §5）。

在线选择实测耗时（宿主 CPU 单次调用，诊断值非微基准）：FS 60.2ms（含 RecoveryCostModel 初始化+首次调用；GPU Scheduler 进程内将不同）、MC 0.37ms、KV 4.40ms。三策略自身选集的 strict J：FS 154.349 / MC 703.455 / KV 627.615（FS 最优，与 T2-G1 已知结论一致）。

## 4. spec 六问逐答

1. **三策略真正在线 Select？** 是——生产入口+两真实内核，spy 直证调用，冻结选集仅事后 oracle（6c 结构性证明选择函数无冻结读取）。
2. **四 Pending 完整周期具备可执行 Runner 路径？** 结构上是——编排/计时/关联/取证分离 CPU 全过；真实执行后端（Native Resume）按抽象接口在 GPU 阶段接入，编排代码零改动。
3. **T0–T4 统一无重复？** 是——整数纳秒顺序边界，恒等式精确成立（非浮点近似），重叠/遗漏均会触发断言。
4. **取证与测量窗口分离？** 是——窗口内仅入队，T4 后 flush 单独计时；必要在线检查（请求前检查、协调后验合同）保留窗内，未删除任何生产控制步骤。
5. **只能 GPU 验证的问题？** 见 §5 七项。
6. **是否具备启动第一轮 GPU 条件？** 具备——所有可在 CPU 验证的逻辑已验证；GPU 第一轮的启动步骤（§8）为机械布线，全部复用已验证组件（G1 8/8 的 FS 链路+60/60 冻结臂的 finish 构造链+本包 select_entry/orchestrator），无新算法逻辑。

## 5. GPU-only 待验证项（7 项，不以 CPU 通过替代）

1. 实时观察构造 Epoch（live observe_control + 实时 census 16=16，替代冻结构造链）
2. Recurrent-only Reconciliation 真实物化（reconcile_strict 逐驱逐 evict_mamba_only + 每步 postcheck）
3. FA 内容审计在真实树上的逐摘要比对
4. Native Resume 真实执行（E/G/Prefill 实测 + Prefill=G+32 合同）
5. Cold Reference 逐 Token 一致（4 Pending）
6. 四 Pending 串行真实计时值与请求间隙 Residual（CPU 仅验证恒等式结构）
7. 三策略完整周期 T0–T4 真实测量与配对统计

## 6. 修改范围合规

- 新增：`evaluation/p0_control_inclusive_harness/{select_entry.py, cycle_orchestrator.py, cpu_gate_test.py}` + `out/` 三个 JSON 产物。
- 未修改：核心 Runtime/Optimizer/Controller/Solver 语义、任何正式 Runner、冻结数据、论文正文；未启动 GPU；未执行正式性能实验。
- 全部新增注释/说明性文字为中文。

## 7. 复算方式

```bash
python3 -m evaluation.p0_control_inclusive_harness.select_entry   # 三策略在线选择+oracle（约 1 分钟）
python3 -m evaluation.p0_control_inclusive_harness.cpu_gate_test  # 32 项 CPU 门禁（约 2 分钟）
```

## 8. 第一轮 GPU 实验启动步骤（本轮不执行）

1. 按 G1 pkg 模式复制 C/K3 正式 Runner 为 GPU 实验包（不动 c_formal_perf_gpu 原件）；
2. 控制入口以本包 `select_entry` 替换冻结加载：Scheduler 进程内三策略各自真实 Select（PILOT_POLICY 选择策略，与 G1 的 plan_strict 在线求解同模式）；
3. `cycle_orchestrator.run_cycle` 接真实 RequestBackend（Native Resume）与实时观察/census/协调；
4. 先跑 1 轮×3 策略正确性闭环（8 门禁：P0-4 §6 表），全过后才进入 5 轮×3 策略正式矩阵；
5. 每轮启动前与轮间 `df -B1G /` 复查（Task 0 纪律）。

完成后停止，等待审查；不自行启动 GPU、不扩展为五轮。
