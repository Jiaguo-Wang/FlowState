# Exact–Greedy GPU 执行证据复核

日期：2026-11-03。**结果：两条在线主库路径均有现成 GPU 证据，PASS。** 本次只读，未启动 GPU。精确核查值见 [gpu_recheck.json](evaluation/exact_greedy_mainpath_audit_20261103/gpu_recheck.json) 与 [additional_gpu_checks.json](evaluation/exact_greedy_mainpath_audit_20261103/additional_gpu_checks.json)。

## 1. 版本与执行环境

当前 HEAD=`8cba094d2c5466759b70e4a5ef8f4824b61aade5`，但 optimizer 有未提交修改；本次认证的是工作树 `optimizer.py` SHA256=`7dfa9aec307d1178d79d79ced0b5d42b0f6574754194c58d84db300c74e48176`。

[Exact source_identity](evaluation/hybrid_solver_gpu_gate_g1/exact/source_identity.json)、[Greedy source_identity](evaluation/hybrid_solver_gpu_gate_g1/greedy/source_identity.json) 各自记录 11 份主库文件。此次逐文件核对全部匹配当前工作树，也匹配代表臂的 execution_lock。运行器 [Exact:21–24](evaluation/hybrid_solver_gpu_gate_g1/pkg_exact/runner.py:21)、[Greedy:31–34](evaluation/hybrid_solver_gpu_gate_g1/pkg_greedy/runner.py:31) 在 Engine 执行前核验这些文件，未仅依赖聊天中的代码版本声明。

既有 GPU 环境：Qwen3.5-9B、SGLang 0.5.17、单 H100 PCIe、TP=1；冻结镜像 ID=`sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`。两代表臂当前实验包的 8 份文件也匹配各自锁定版本。该证据不认证 TP=2。

## 2. 小规模 Exact：w1/K2，Pending B

目录：[e2_w1k2_exact_B](evaluation/hybrid_solver_gpu_gate_g1/exact/e2_w1k2_exact_B/)。

| 证据链环节 | 原始记录与值 |
|---|---|
| 当前生命周期 Epoch | [reconciliation.json](evaluation/hybrid_solver_gpu_gate_g1/exact/e2_w1k2_exact_B/reconciliation.json) 的 `raw_epoch/epoch/physical_observation`；7 候选、4 Pending、K=2。重新核对 raw SHA，并重建 `StrictAllocationEpoch` 验证内容摘要。 |
| 在线分派 | `strict_decision`: `solver=Exact, gate_allowed=true, enumerable_subsets=29`；Plan 来自 [control.py:96](evaluation/hybrid_solver_gpu_gate_g1/pkg_exact/control.py:96) 的实际主库调用。 |
| Plan | `{B5C3_NODE_18,B5C3_NODE_21}`，J=53.77075886917578 ms；Oracle 在之后计算，29 子集、同选集同 J。 |
| 进入正式协调 | [control.py:142](evaluation/hybrid_solver_gpu_gate_g1/pkg_exact/control.py:142) 将该 Plan 传给 `StateController.reconcile_strict`；`apply_validation.same_scheduler_call=true`。 |
| 物理结果 | `reconciliation.status=PASS, phase=complete`，完成驱逐 20/22/24/25/27 五个 Node 对应逻辑候选；Binding 和 Plan 的 Keep/Evict 一致。 |
| Native 来源 | [post_request.json](evaluation/hybrid_solver_gpu_gate_g1/exact/e2_w1k2_exact_B/post_request.json) 的 `trace.matches` 与 Summary 一致：恢复 Node18，源槽3，工作槽19；H=3712、E=3392、G=320。该 Node/Slot 只属于此生命周期。 |
| 真实执行与 COW | `trace.copies` 实际返回，源前后摘要相同且等于复制后目标，源槽/工作槽不相交；Prefill `[3392,4308)` 共916 Tokens，之后 Decode 均正常返回。G=320 是历史缺口，额外输入596，不能将916写成320。 |
| 状态保护 | 协调前后 Selected 内容一致、Attention 摘要和完整 FA 内容一致；请求后 Selected 组件/内容、非重算 FA 内容与旧 FA 映射检查通过。 |
| 输出与清理 | [output.json](evaluation/hybrid_solver_gpu_gate_g1/exact/e2_w1k2_exact_B/output.json) 8 Token 与 [冻结 Cold](evaluation/hybrid_solver_gpu_gate_g1/exact/cold_reference_IMW1_B_TURN_003.json) 逐项相同；[process](evaluation/hybrid_solver_gpu_gate_g1/exact/e2_w1k2_exact_B_process.json) 返回0，[cleanup](evaluation/hybrid_solver_gpu_gate_g1/exact/e2_w1k2_exact_B_cleanup.json) PASS。 |

该路径不是外部指定 Exact，也不是加载独立规划器的现成输出。Evaluation 侧仍负责真实观察认证、来源映射和 Oracle；真正的分配搜索由主库执行。

## 3. 较大规模 Greedy：C/K3，Pending B

目录：[g2_ck3_greedy_B_r3](evaluation/hybrid_solver_gpu_gate_g1/greedy/g2_ck3_greedy_B_r3/)。

| 证据链环节 | 原始记录与值 |
|---|---|
| 当前生命周期 Epoch | [reconciliation.json](evaluation/hybrid_solver_gpu_gate_g1/greedy/g2_ck3_greedy_B_r3/reconciliation.json) 的 `raw_epoch/epoch/census`；16 候选、4 Pending、K=3；实际域和映射16=16，无额外未映射候选。 |
| 在线分派 | [control.py:136–147](evaluation/hybrid_solver_gpu_gate_g1/pkg_greedy/control.py:136)：调用主库后记录 `solver=Marginal Greedy, gate_allowed=false`；697子集虽≤794，n=16仍超限。 |
| Plan | 本生命周期 ID 为 `{B5C3_NODE_57,B5C3_NODE_56,B5C3_NODE_55}`；映射 CPU 集合 `{C_B_P05120,C_C_P06144,C_D_P07168}`；J=154.34895561599998 ms。 |
| 冻结参照用途 | [control.py:153–160](evaluation/hybrid_solver_gpu_gate_g1/pkg_greedy/control.py:153) 在求解结束后读入 `policy_plans.json`；它是事后行为对照，不是在线求解输入，也不是 Exact 最优性认证。 |
| 进入正式协调 | [control.py:236](evaluation/hybrid_solver_gpu_gate_g1/pkg_greedy/control.py:236) 传入同一 Plan；Apply-time 在同一次 Scheduler 调用内完成。 |
| 物理结果 | Ledger PASS/complete，13 次 Native Recurrent-only 驱逐完成，最终保留3；Selected 内容、Attention 摘要及请求后保护字段复核通过。 |
| Native 来源 | [post_request.json](evaluation/hybrid_solver_gpu_gate_g1/greedy/g2_ck3_greedy_B_r3/post_request.json)：Node55、源槽11、工作槽12；H=5120、E=5120、G=0。Node55 对应本次保留的 B 逻辑状态。 |
| 真实执行与 COW | 复制实际返回，源前后与目标复制后内容相同、槽位隔离；Prefill `[5120,5152)` 为32 Tokens，Prefill/Decode 均返回。 |
| 输出与清理 | [output.json](evaluation/hybrid_solver_gpu_gate_g1/greedy/g2_ck3_greedy_B_r3/output.json) 的8 Token 等于 [冻结 Cold](evaluation/hybrid_solver_gpu_gate_g1/greedy/cold_reference_C_B.json)；[process](evaluation/hybrid_solver_gpu_gate_g1/greedy/g2_ck3_greedy_B_r3_process.json) 返回0，[cleanup](evaluation/hybrid_solver_gpu_gate_g1/greedy/g2_ck3_greedy_B_r3_cleanup.json) PASS。 |

## 4. 已有 8 臂覆盖及失败记录

下表为此次重新读取的实际分派、Native H/E/G、输出与清理结果。各臂均为独立生命周期，不能把某一臂的物理地址复用到另一臂。

| 分支 | 原运行 ID | H | E | G | 驱逐 | 8 Token==Cold / 清理 |
|---|---|---:|---:|---:|---:|---|
| Exact | e1_w1k2_exact_A |3648|3392|256|5|PASS / PASS|
| Exact | e2_w1k2_exact_B |3712|3392|320|5|PASS / PASS|
| Exact | e3_w1k2_exact_C |4096|3712|384|5|PASS / PASS|
| Exact | e4_w1k2_exact_D |4160|3712|448|5|PASS / PASS|
| Greedy | g1_ck3_greedy_A_r3 |4096|0|4096|13|PASS / PASS|
| Greedy | g2_ck3_greedy_B_r3 |5120|5120|0|13|PASS / PASS|
| Greedy | g3_ck3_greedy_C_r3 |6144|6144|0|13|PASS / PASS|
| Greedy | g4_ck3_greedy_D_r3 |7168|7168|0|13|PASS / PASS|

G1 曾有7次程序编排失败尝试，原 [greedy/verify.json](evaluation/hybrid_solver_gpu_gate_g1/greedy/verify.json) 和 [最终报告](evaluation/hybrid_solver_gpu_gate_g1/final_report.md) 保留：E=0 无源分支的 Runner 假设错误，以及根目录独占留证文件造成的跨臂冲突。没有删除这些失败，也没有把它们计入8条成功链。本次不运行这些历史验证器，以免覆盖旧 `verify.json`。代表臂选用明确匹配现存包版本的 e2/g2_r3；不将旧启动器或旧包的豁免泛化为主库代码漂移豁免。

## 5. 可以与不可以认定的结论

**已验证：** `EPOCH_ADMISSIBLE → 当前计划 → RECONCILIATION_VALIDATED → 本案例 RESUME_VALIDATED` 的两分支闭环，真实 COW 与 Cold 一致。

**未由此证明：** 混合求解版已提交/发布；任意 Epoch 的物理合法性可由求解器独自认证；冻结性能矩阵等于在线求解性能；稳定净加速、低审计 Serving 开销或跨模型泛化；跨 Workflow 物理张量来源血缘。后者继续 `PROVENANCE_UNVERIFIED`。

因此没有必要再补一个 Exact 在线 GPU Smoke。本任务结束，不启动后续实验。
