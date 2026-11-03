# Exact–Greedy 在线 Strict 主路径最终核查

**裁决：PASS（当前工作树范围）。** 当前主库 `GlobalOptimizer.plan_strict()` 已实现 Exact–Greedy 自动分派。此次通过该实际入口复算 Strict20，20/20 条走 Exact 并与冻结选集及目标一致；C/K3 自动走 Marginal Greedy。已有独立 GPU 生命周期包含两分支从在线求解、正式协调到 Native Resume 的完整证据，且核心源码摘要与当前工作树一致。没有启动新 GPU 实验。

**版本限定：混合入口尚未提交到 Git HEAD。** 当前 HEAD 为 `8cba094d2c5466759b70e4a5ef8f4824b61aade5`；`git status` 显示 `flowstate/optimizer.py` 已修改。HEAD 中该文件 SHA256 为 `b0060e1cc9fc5d14352a998e8ef95b9b9a8ed3a28a562303595bea749e6f360c`，不含 `_exact_admission`；当前工作树文件为 `7dfa9aec307d1178d79d79ced0b5d42b0f6574754194c58d84db300c74e48176`，与两组 GPU 执行锁一致。因此不能仅引用这个 commit 就声称提交版已包含混合能力；本次 PASS 对应工作树字节身份。[版本记录](evaluation/exact_greedy_mainpath_audit_20261103/git_version_detail.json)

## 1. 主库分派和错误处理

| 检查项 | 实际路径与行号 | 结论 |
|---|---|---|
| Strict 入口 | [optimizer.py:86](flowstate/optimizer.py:86) | 严格类型检查、`epoch.validate()` 在求解之前执行。 |
| 四项准入 | [optimizer.py:21](flowstate/optimizer.py:21)、[28](flowstate/optimizer.py:28) | n≤12、Pending≤4、K≤4、Σ(i=0..min(n,K)) C(n,i)≤794，四项同时满足。 |
| 真正 Exact | [optimizer.py:106](flowstate/optimizer.py:106)、[145](flowstate/optimizer.py:145) | 调用主库 `_enumerate_exact_subsets`，遍历包括空集的全部 \|S\|≤K 组合；未导入 Evaluation 求解器。 |
| 同分合同 | [optimizer.py:160](flowstate/optimizer.py:160) | J 最小值的 1e−9 ms 容差内，先选最少候选，再按 ID 字典序。 |
| 超限分支 | [optimizer.py:116](flowstate/optimizer.py:116)、[167](flowstate/optimizer.py:167) | 调用原 `_select_candidates` Marginal Greedy；两分支共用 `epoch.frontiers` 和正式 Φ 评分闭包。 |
| 决策与计划 | [optimizer.py:126](flowstate/optimizer.py:126) | 记录 `last_strict_decision`，返回正式 `StrictAllocationPlan`，含逐 Pending E/G/成本和逻辑驱逐补集。 |
| 非法 Epoch | [strict_epoch.py:101](flowstate/strict_epoch.py:101) | 验证候选唯一、Token/配置/内容摘要、等大小、全矩阵覆盖和证据条件；UNKNOWN 不可准入。 |
| 异常处理 | [optimizer.py:89](flowstate/optimizer.py:89) | 没有吞异常或异常转 Greedy 的分支。只有合法规模超限触发 Greedy。 |
| Legacy | [optimizer.py:70](flowstate/optimizer.py:70) | `select()` 仍为 Greedy-only，不能与 Strict 混合入口混称。 |

C/K3 的枚举数实际为 **697**，虽小于 794，但 n=16 超限，仍必须进入 Greedy。这说明不能把“子集数≤794”单独用作 Exact 准入条件。K 是至多保留数量，Exact 在无额外收益时可返回少于 K 项。

`plan_strict()` 不是 GPU 状态认证器，也不自行调用 Reconciliation；物理新鲜度和映射必须由调用者在控制窗口中补齐。其返回前不自动执行 `validate_plan()`，真实 GPU 调用者在返回后显式验证，且 `reconcile_strict` 的 Bind 再次验证。不能将只构造一个逻辑 Epoch 就理解为物理安全已获认证。

## 2. CPU 主库入口复核

复用 [既有 CPU Gate](evaluation/paper_hybrid_solver_integration_gate.py) 的冻结输入构造与独立参照，加载该模块后仅将报告输出位置改为本次独立目录，再运行其 `main()`。实际被测对象仍为当前主库 `GlobalOptimizer.plan_strict()`；独立 B5-D1 求解器只作 Oracle，不生成被测计划。未运行旧 Gate 的原始输出入口，旧 `paper_hybrid_solver_integration_gate.json` 摘要保持不变。

[本次脚本](evaluation/exact_greedy_mainpath_audit_20261103/recheck.py)、[成功日志](evaluation/exact_greedy_mainpath_audit_20261103/recheck_retry.log)、[机器结果](evaluation/exact_greedy_mainpath_audit_20261103/paper_hybrid_solver_integration_gate.json)。

| 快照 | n / Pending | K=1 J | K=2 J | K=3 J | K=4 J | 实际 Solver |
|---|---|---:|---:|---:|---:|---|
| w1 | 7 / 4 | 223.842223 | 53.770759 | 21.953375 | 0.000000 | 全 Exact |
| w2 | 7 / 4 | 214.019643 | 53.733180 | 21.965538 | 0.000000 | 全 Exact |
| w3 | 7 / 4 | 185.820309 | 25.559190 | 10.909073 | 1.142854 | 全 Exact |
| w4 | 7 / 4 | 205.344724 | 45.066824 | 10.871358 | 1.105138 | 全 Exact |
| B5-C4 | 7 / 4 | 228.054813 | 48.228729 | 16.411345 | 1.753119 | 全 Exact |

J 单位为模型化 ms，不是 GPU 延迟。全部 20 条均通过正式 `validate_plan`；选集与 E3 冻结 Exact 行一致，并与 B5-D1 独立枚举实现逐项一致。浮点最大 |ΔJ| 约 **2.84×10⁻¹⁴ ms**，小于既有 1e−9 容差；不应将“一致”写成全部浮点字节相同。

**C/K3：** n=16、Pending=4、K=3，自动选择 **Marginal Greedy**，J=**154.34895561599998 ms**。冻结生命周期对应集合 `{B5C3_NODE_55,B5C3_NODE_56,B5C3_NODE_57}`；由该生命周期映射回逻辑 CPU 身份为 `{C_B_P05120,C_C_P06144,C_D_P07168}`。与冻结 Greedy 计划、`policy_plans.json` 和 CPU 结果一致；这不是 C/K3 的 Exact 最优证书。旧 Node ID 只用于此次离线核对，不能绑定到新 Engine。

逐条件实际 Solver、全部选集、参考选集、J 和差值见 [solver_dispatch_cpu_recheck.csv](solver_dispatch_cpu_recheck.csv)，共 **21 行**，其中 Strict20 为 20 行。

额外复用的 3 项拒绝检查均通过：缺失配对、UNKNOWN、Exact 内部注入异常，无计划返回、不切换 Greedy。6 项规模边界均符合 12/4/4/794 合同；原 Greedy 对 Strict20 的 20 条行为也保持不变。此处异常测试使用新优化器实例；不要将 `last_strict_decision` 审计字段当作任意复用实例的最新成功凭证，实际应用必须使用本次调用成功返回的 Plan。

## 3. 正式协调与 GPU 主路径证据

完整分析见 [gpu_execution_evidence.md](gpu_execution_evidence.md)。这次并非仅复用 GPU 报告中的 PASS 字样，而是读取原始 `reconciliation.json`、`post_request.json`、输出、Cold、执行锁及清理记录：

| 分支代表臂 | 规模 / K | 在线选集 | J（模型 ms） | 实际协调 | Native / Cold |
|---|---|---|---:|---|---|
| G1 Exact `e2_w1k2_exact_B` | 7 / 4 / 2 | `{NODE_18,NODE_21}` | 53.770759 | 5 次驱逐，保留 2 | H=3712、E=3392、G=320；COW 完成；8 Token==Cold |
| G1 Greedy `g2_ck3_greedy_B_r3` | 16 / 4 / 3 | `{NODE_57,NODE_56,NODE_55}` | 154.348956 | 13 次驱逐，保留 3 | H=5120、E=5120、G=0；COW 完成；8 Token==Cold |

两个代表臂的 11 份主库源码和 8 份实验包源码均与各自执行锁匹配。两条调用链均在 Scheduler 控制 RPC 内实际执行 `plan_strict(epoch)`，使用其返回的同一个 Plan 做 `validate_plan→Bind/Preflight→StateController.reconcile_strict`。后者 [controller.py:115](flowstate/controller.py:115) 进入 [StrictReconciler:193](flowstate/strict_control.py:193)，重新 Bind、Preflight，逐项调用 Native `evict_mamba_only` 并 Postcheck，最后独立检查实际驻留集合；失败时使生命周期失效并抛出 `StrictReconciliationError`，不继续恢复。

同场景另有各 3 个独立 Pending 臂，合计 8 个成功 GPU 生命周期。此次核对其分派、选集/J、Cold 逐 Token、实际驱逐数、进程返回码和清理记录；两个代表臂另下钻全部关键原始状态。无需新增 Exact 在线验证任务。

## 4. 在线求解与冻结执行的边界

- G1 Exact 的 Oracle 枚举发生在主库 Plan 已返回之后，不能反向决定在线选集。[pkg_exact/control.py:96–120](evaluation/hybrid_solver_gpu_gate_g1/pkg_exact/control.py:96)
- G1 Greedy 在主库求解后才读取 `policy_plans.json` 作对照。[pkg_greedy/control.py:136–160](evaluation/hybrid_solver_gpu_gate_g1/pkg_greedy/control.py:136)
- 原 `c_formal_perf_gpu` / 经其复用的冻结性能路径在 [control.py:62–79](evaluation/c_formal_perf_gpu/control.py:62) 读取既定策略选集并映射。这类实验能够验证给定保留策略下的执行与性能，**不能单独证明在线 Select 或完整在线控制性能**。
- G1 使用真实观察、来源认证和在线求解，但 Epoch 构造复用了 Evaluation 的输入适配与重型实验 Hook；这证明主库求解与协调接口可在真实 Scheduler 中闭环，不等于已部署通用、低开销 Serving 管线。

## 5. 最终边界与论文表述

当前不是 Greedy-only 工作树，因此无需按“尚未集成 Exact”重写方法。建议 IV-C 明确“Strict 入口按四项规模门限自动分派，Legacy 入口保持 Greedy”；VI-A 标明测试工作树 SHA 与待提交状态；VI-E 分开报告 Exact 最优性与 Greedy 参考一致性；VI-F 分开报告 G1 在线正确性与冻结选集性能实验。没有修改论文正文。

本次 PASS 不证明稳定端到端净收益、硬 deadline、跨 Rank TP=2 或跨 Workflow 张量血缘。血缘继续为 `PROVENANCE_UNVERIFIED`。源码集成尚未进入 HEAD 是发布可追溯性事项，不是本次工作树执行正确性失败；正式发布前应锁定与 GPU 证据一致的版本，不应把旧 commit 当作混合求解实现身份。

初次新增审计脚本错误地将 GPU capture 当作旧版 B4 Snapshot 解码，触发 `KeyError: barrier_id`；只修正了本次审计器的反序列化，保留 [初次日志](evaluation/exact_greedy_mainpath_audit_20261103/recheck.log)。按原始规范 Epoch 字段重建并验证后通过，未改任何 Runtime、历史记录或准入条件。源文件摘要检查见 [source_integrity.json](evaluation/exact_greedy_mainpath_audit_20261103/source_integrity.json)。

**最终：PASS；不需要补跑 GPU。本轮停止。**
