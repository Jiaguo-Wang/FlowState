# Hybrid Solver GPU 最小闭环验证门禁（Spec 8）

**判定：PASS**（双分支经新入口 `GlobalOptimizer.plan_strict()`：实际求解、物理协调、请求正确性全部通过）

- 场景A（Exact 分支，B5-C4，K=2）：宿主机核验 **21/21 PASS**（`paper_hybrid_gpu_gate_a/gate_a_result.json`）
- 场景B（Greedy 分支，C/K3）：宿主机核验 **22/22 PASS**（`paper_hybrid_gpu_gate_b/gate_b_result.json`）

## 1. 范围与定位

本轮仅验证：Spec 7 集成进 `flowstate/optimizer.py` 的生产混合入口 `GlobalOptimizer.plan_strict()` 能否在真实 SGLang Hybrid Runtime 的 **Scheduler 进程内**完成求解器选择（Exact 门禁 / Marginal Greedy 自动降级），并通过现有 `StateController.reconcile_strict()` 安全落实分配结果。**这是 GPU 正确性/执行路径验证，不是性能实验**；未重跑完整实验矩阵，每场景单次运行，不为数据好看重复执行。全部计时字段仅为执行记录，不构成任何性能口径。

- 被测生产入口：`flowstate/optimizer.py` 的 `GlobalOptimizer.plan_strict()`（含 `_exact_admission` 门禁、`_enumerate_exact_subsets` Exact 分支、`_select_candidates` Greedy 分支、`last_strict_decision` 决策记录），Spec 7 集成版，哈希在 `source_identity.json` 的 `hybrid_entry_sources`（11 个 flowstate/*.py）中锁定并在运行内逐项核验。
- 求解器算法与门禁阈值零改动；冻结输入、策略选集、历史 GPU 结果零改动（execution_lock 完整性检查通过）。
- 环境：镜像 `sha256:1dfce903…f926`（lmsysorg/sglang:v0.5.17-cu129-runtime，与 b5d4b/g1_CB_FS 一致），单臂 GPU0（H100 PCIe，空闲检查+清理验证 PASS），repo 只读挂载、门禁目录 rw 挂载为 `/b5out`，端口 A=49997 / B=49998。控制路径经 `b5d4_apply` RPC 在 Scheduler 进程内执行 `control.apply`，其中直接 `from flowstate.optimizer import GlobalOptimizer` 在线求解。
- 复用路径：场景A = b5d4b 门禁骨架（bridge/observation/protocol/persistence/source_contract 逐字节复用，`cmp` 验证一致），control 的求解块替换为 `plan_strict` 在线调用；场景B = c_formal C/K3 census+双预热正确性路径，选集来源替换为 `plan_strict` 在线调用，`policy_plans.json` 降级为求解完成后的事后对照 Oracle。

## 2. 场景A：Exact 分支（B5-C4，K=2，run `a1_hybrid_exact`）

Epoch：复用 B5C3 冻结 8 条历史重放（1 token 生成、digest 一致）+ 实时严格观察准入，形成 7 候选 / 4 Pending / K=2 的 strict Epoch。**非全在线 Epoch 构建**（复用冻结准入证据）。

在线决策（`strict_decision`，由 Scheduler 进程内的 `plan_strict` 写入 `reconciliation.json`）：

| 字段 | 值 |
|---|---|
| solver | **Exact**（门禁内自动选择） |
| gate_allowed | true |
| gate | candidates=7, pendings=4, budget_k=2, enumerable_subsets=29 |
| selected_ids | {B5C3_NODE_18, B5C3_NODE_27} |
| objective_ms (J) | 48.228728759089606 |
| pure_solve_ms | 0.222861 |

`plan_strict_hybrid_solve` 阶段墙钟 8.32 ms（含 `validate_plan`），控制 RPC 总墙钟 4870.5 ms。

选集与 J 的对照（全部 ±1e-9 或集合相等）：
- == b5d4b `p01_Exact` 冻结计划（同环境同K Exact 臂）
- == 本轮运行内独立 Decimal Oracle（k≤2 共 29 子集穷举，`independent_oracle` 0.43 ms）
- == `greedy_swap_structure_e3_step1/condition_results.csv` 的 B5-C4/K=2/Exact 冻结行
- **与旧 Greedy 选集 {N20, N18} 存在明确差异**（本轮多出 N27，验证 Exact 分支真实改变了选择，而非复刻 Greedy）

物理与请求正确性：
- 逐 Pending E/T/G == 冻结 Exact 臂（目标 B5C2_C_TURN_003：E=3520 / T=3904 / G=384）
- Recurrent-only 协调：未选中补集 5/5 全部驱逐且 `returned=true`
- Native 匹配：best_match_node=27（== 计划绑定 NODE_27/slots[18]），E=3520，FA 命中 H=3904，G=384，`mamba_branching_seqlen=3904`；COW payload 前后哈希一致
- 正式请求（8 token，temperature=0）：输出 `[248058, 198, 27, 1628, 15462, 10318, 32634, 29]` == b5d4b 冻结 Cold 参照（该参照已经 10 臂逐Token验证）
- 请求后复核：Selected 内容不变、FA 映射全保留、受保护 FA 内容（sha256）不变

## 3. 场景B：Greedy 分支（C/K3，run `b1_hybrid_greedy`）

Epoch：复用 C/K3 冻结 16 条预热 ×2 轮重放（`history_pre` → `flush_cache` → `history`；端点物化于 round0，祖先 L−64 物化于 round1）+ 实时严格观察准入，形成 16 候选 / 4 Pending / K=3 的 strict Epoch。**非全在线 Epoch 构建**。

在线决策（`strict_decision`）：

| 字段 | 值 |
|---|---|
| solver | **Marginal Greedy**（门禁拒绝后自动降级） |
| gate_allowed | false（16 候选 > 门禁上限 12，697 子集在规模上限内但 n 超界即拒绝） |
| gate | candidates=16, pendings=4, budget_k=3, enumerable_subsets=697 |
| selected_ids | {B5C3_NODE_57, B5C3_NODE_56, B5C3_NODE_55} |
| objective_ms (J) | 154.34895561599998 |
| pure_solve_ms | 0.402763 |

`plan_strict_hybrid_solve` 阶段墙钟 17.0 ms（含 16×4 边际贪心 + `validate_plan`），控制 RPC 总墙钟 13748.5 ms。`selection_source = "GlobalOptimizer.plan_strict(在线)"`——本轮选择由新入口实际求解产生，`policy_plans.json` 仅在求解完成后读入作事后对照。

选集与 J 的对照（经 census 16=16 映射，CPU↔物理一致）：
- 物理选集 {NODE_57, NODE_56, NODE_55} == g1_CB_FS 冻结 GPU FlowState-Greedy 计划
- == `policy_plans.json` 冻结 FS 选集 {C_D_P07168, C_C_P06144, C_B_P05120}（经 census 映射）
- == `kvflow_multiworkflow_cpu_main/cpu_results.json` 冻结 FS 选集（经 census 映射）
- J == 冻结 J 154.34895561599998 == g1 冻结计划 J == cpu_results J（±1e-9）

物理与请求正确性：
- census 16=16，无未映射/缺失；物化时间线 == 冻结合同（round0/round1）
- 逐 Pending E/T/G == 冻结预期（C_A: 0/4096/4096；C_B: 5120/5120/0；C_C: 6144/6144/0；C_D: 7168/7168/0）
- Recurrent-only 协调：未选中补集 13/13 全部驱逐且 `returned=true`
- 目标 C_B（PILOT_PENDING）Native 恢复：source NODE_55/slots[11] == 计划绑定，E=5120，H=5120，G=0；`batch_prefix_length=5120`、prefill 仅 32 新 token（== 计划 E 准入 + 32）
- 正式请求（8 token）：输出 `[220, 197, 1, 198, 197, 1, 198, 197]` == g1_CB_FS 冻结 `cold_reference_C_B.json`
- 请求后复核：Selected 内容不变、FA 映射全保留、受保护 FA 内容不变

## 4. 规格八项必答逐条

| # | 问题 | 场景A | 场景B |
|---|---|---|---|
| 1 | 实际执行的求解器 | Exact（门禁内自动） | Marginal Greedy（门禁拒绝自动降级，非静默回退） |
| 2 | plan_strict 是否在 GPU 控制进程内调用 | 是：`strict_decision`（含 gate/pure_solve_ms）由 Scheduler 进程内 `control.apply` 写入 reconciliation.json；control timing 含 `plan_strict_hybrid_solve` 阶段；control_evidence/002_plan.json 在案 | 同左 |
| 3 | 选集 vs 冻结 Oracle | == b5d4b 冻结 Exact == 运行内 Decimal Oracle == e3 冻结行；≠ 旧 Greedy {N20,N18}（明确差异 N27） | == g1 冻结 GPU 计划 == policy_plans（census 映射）== cpu_results（census 映射） |
| 4 | 物理协调成功、未选中真实驱逐 | 5/5 驱逐全部 returned，residency == Exact 选集 | 13/13 驱逐全部 returned，residency == 冻结 FS 选集 |
| 5 | FA-KV 未受损 | 是（映射全保留 + 受保护内容 sha256 不变 + selected payload 不变） | 同左 |
| 6 | Native E/G 与计划一致 | E=3520/H=3904/G=384 == 计划 | E=5120/H=5120/G=0 == 计划 C_B，prefill=32 |
| 7 | Native Resume == Cold Reference | 8 token 逐Token一致（b5d4b 冻结 Cold，10 臂验证过） | 8 token 逐Token一致（g1 冻结 C_B Cold） |
| 8 | 失败/额外恢复来源/未验证步骤 | 无失败；无额外恢复来源（COW source==计划绑定，payload 哈希前后一致）；见 §5 局限 | 同左 |

## 5. 局限与如实记录

1. **Epoch 形成方式**：两场景均为"冻结历史/预热重放 + 实时观察准入"，**不是全在线 Epoch 构建**（复用 B5C3/C-K3 冻结准入证据）。
2. **Cold 参照复用**：场景A 复用 b5d4b 冻结 Cold（10 臂逐Token验证），场景B 复用 g1_CB_FS 冻结 `cold_reference_C_B.json`，未另起新的 Cold 生命周期——符合最小正确性验证原则。
3. **选源血缘**：`provenance_status = PROVENANCE_UNVERIFIED`（沿用 b5d4b 结论：首次 Census 不证明张量创建/继承/COW 血缘）。选源经唯一最深共享前缀策略 + `native_source_validation`（MATCHED）+ COW payload 哈希（source_before == source_after == destination_after）三重验证，但观察级 provenance 本身不升级为证明级；`cross_workflow_certified=false`。
4. **单次运行**：每场景单臂单次执行，无重复；计时字段（control_rpc_wall、request_wall、ttft 等）仅为执行记录，**不构成性能口径**，不可与 C 正式性能配对实验混用。
5. **verify.py 运行后自修复记录**（场景A）：docker 运行完成后，宿主机核验脚本修复自身核验键路径 bug（epoch.candidates 嵌套键 `candidate.checkpoint_id`），不影响任何运行输入或被测代码；已写入 `gate_a_result.json` 的 `post_run_notes`，完整性检查范围排除宿主机编排脚本。场景B 的同类修复在启动前完成，无运行后编辑。
6. **未修改项**：Hybrid Solver 算法与门禁阈值、冻结输入、策略选集、历史 GPU 结果、论文正文均零改动（execution_lock 全程未变核验通过）。

## 6. 证据索引

- 场景A 门禁目录：`evaluation/paper_hybrid_gpu_gate_a/`（prepare/launch/verify + source_identity + input_provenance + execution_lock/started + `a1_hybrid_exact_command.json`、`a1_hybrid_exact.runtime.log`、`a1_hybrid_exact_process.json`（returncode 0）、`a1_hybrid_exact_cleanup.json`（PASS）、`gate_a_result.json` 21/21）
- 场景A 运行产物：`paper_hybrid_gpu_gate_a/a1_hybrid_exact/`（summary / reconciliation / post_request / cold_reference_check / native_source_validation / native_trace / stream_chunks / output / request / initial / history_01–08 / `control_evidence/000–005` 六阶段 / 12 个 `rpc_*` 调用存证目录）
- 场景B 门禁目录：`evaluation/paper_hybrid_gpu_gate_b/`（同结构 + 5 个冻结输入 prepared_inputs / cpu_candidate_prefix_mapping / cpu_gate / policy_plans / cold_reference_C_B + `gate_b_result.json` 22/22）
- 场景B 运行产物：`paper_hybrid_gpu_gate_b/b1_hybrid_greedy/`（summary / reconciliation / post_request / native_trace / flush_state / history_pre_01–16 + history_01–16 / `control_evidence/000–005` / 38 个 `rpc_*` 调用存证目录）
- 被测代码：`evaluation/paper_hybrid_gpu_fix_a/`、`evaluation/paper_hybrid_gpu_fix_b/`（bridge/observation/protocol/persistence/source_contract 与 fix_b5d4a 逐字节一致；control/runner 为本轮新入口接线，中文注释）
- 冻结对照来源（只读）：`evaluation/exact_greedy_paired_gpu_gate_b5d4b/`、`evaluation/c_formal_perf_gpu/g1_CB_FS/`、`evaluation/greedy_swap_structure_e3_step1/condition_results.csv`、`evaluation/kvflow_multiworkflow_cpu_main/cpu_results.json`

## 7. 结论

生产混合入口 `GlobalOptimizer.plan_strict()` 在真实 SGLang Hybrid Runtime 的 Scheduler 进程内完成了门禁评估与求解器自动选择（A: Exact；B: Marginal Greedy），两分支选集/J 与全部冻结 Oracle 一致（A 另证与旧 Greedy 存在明确选集差异），经生产 `reconcile_strict()` 物理落实（A 5/5、B 13/13 Recurrent-only 驱逐全部返回），FA-KV 与 Selected 内容受保护未受损，Native E/G 与计划逐项一致，正式请求输出与冻结 Cold 参照逐Token一致。**判定 PASS**。本门禁为最小正确性验证，不构成在线 E2E 性能实验。
