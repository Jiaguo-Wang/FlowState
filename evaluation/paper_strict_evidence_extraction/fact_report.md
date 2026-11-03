# Strict 分配实验事实提取与独立复核报告（Task 1）

**验收状态：PASS**（指本项"数据提取与复核"工作完成：规定范围数据完整提取、关键数值与来源核验通过、19/19 项一致性检查全过。PASS 不意味着 FlowState 的研究假设得到支持。）

本轮只读提取 + 必要 CPU 复算，未修改任何生产代码、算法、模型系数、实验输入与冻结结果，未启动 GPU/SGLang，未新增工作负载或调参。所有新增脚本只写入本目录（`extract.py`、`extract_links.py`）。

## 1. 数据来源与版本

| 数据 | 路径 | 身份/版本 |
|---|---|---|
| Strict20 冻结结果（140 行） | `greedy_swap_structure_e3_step1/condition_results.csv` | evidence_manifest+source_integrity PASS；与 `strong_baseline_position_audit_e5_step0/all_policy_results.csv` 字节一致（sha256 相同） |
| Strict20 快照原始输入 | `independent_runtime_allocation_e2/w1–w4/strict_epoch_snapshot.json`；`multishare_allocation_gate_b5c4/results.json` | E2 冻结快照（w1–w4，独立多 Workflow 受控组合）+ B5-C4 历史发现案例；e3 生成时已锁哈希 |
| 逐步/替换轨迹 | `greedy_swap_structure_e3_step1/{selection_traces,swap_traces,local_optimum_neighbors}.json/csv` | e3 冻结产物 |
| C/K3 CPU 五策略 | `kvflow_multiworkflow_cpu_main/{cpu_results.json,workloads.json}`（MW_CPU_C，current=C_A..C_D） | input_lock 校验通过 |
| C/K3 StandaloneTopK | `kvflow_multiworkflow_cpu_review/review_results.json`（task2 K=3） | input_lock_verified=true；c_formal `policy_plans.json` SA 条目 selection_source 指向该复核 |
| C/K3 1-Swap | `paper_solver_tradeoff_summary.csv`（source=ck3） | `paper_solver_tradeoff_gate.json`：Exact 门禁拒绝 n=16>12，未绕过 |
| B5-C4 K2 GPU（Exact/Greedy） | `exact_greedy_paired_gpu_gate_b5d4b/`（p01–p05 ×2 + Cold） | 镜像 `sha256:1dfce903…f926`（与下述全部 GPU 实验同一冻结镜像）；cold_reference_correctness 10/10 |
| w1/K2 GPU（Exact vs Greedy） | `native_resume_performance_e4_phase1b/` + `e45_unified_timing/`（E4-1B） | e45 统一计时 18 生命周期 |
| w1/K3 GPU（Exact vs 1-Swap） | `exact_swap_runtime_e5_step1/` + `e45_unified_timing/`（E5-1） | e45 统一计时 36 生命周期 |
| C/K3 GPU 主矩阵 | `c_formal_perf_gpu/`（m1–m5×{A,B,C,D}×{FS,SA,WFC}=60 臂 + g1/g2 门禁 2 臂） | 协议 v1.2；finalize 62/62 PASS |
| Spec 8 Hybrid GPU 闭环 | `paper_hybrid_gpu_gate_a/`（a1）、`paper_hybrid_gpu_gate_b/`（b1） | 同一冻结镜像；gate 21/21、22/22 |
| 旧 573 条 | `greedy_swap_structure_e3_step1/legacy_573_structure.csv`（573 行）+ `legacy_summary.json` | 源=gate1_step1_allocation_equivalence（哈希在案）；只组织、未重跑策略 |

任务文所列 `p1_strict_baseline_validation` 在仓库中实际目录名为 **`p1_strong_baseline_validation`**（WFC/PCB/LDR 实现与 common 评分所在）；按实际目录整理。

## 2. 提取范围与完整性

- **Strict20：5 快照（w1–w4、B5-C4）× 4 预算（K=1..4）= 20 条件 × 7 策略 = 140 行全部保留**，不以胜负筛选；每条件候选 7 个、Pending 4 个。独立复算（p1 冻结实现 + 生产 Greedy 核心 + 全子集枚举，在同一重建严格 Epoch 上）**140/140 选集与 J（±1e-9）与 e3 冻结一致**。
- **C/K3：16 候选 / 4 Pending（C_A T=4096、C_B 5120、C_C 6144、C_D 7168）/ K=3 输入身份完整**（candidates_roster 含每个候选 L 与兼容 Pending）；Greedy/WFC/1-Swap/StandaloneTopK 复算与冻结一致；**Exact 证书 NOT_AVAILABLE**（n=16>12 门禁拒绝，未绕过、不启动新枚举）。
- **旧 573：OpenHands 504 条/168 快照 + AgentX 69 条/23 快照**，仅复用既有结构审计组织，未与 Strict20 合并计算任何总胜率。
- 产物：`strict_allocation_matrix.csv` 148 行、`selection_disagreements.csv` 148 行、`baseline_input_contract.csv` 11 行、`gpu_evidence_links.csv` 171 行（31 行有 GPU 实测、140 行 NOT_MEASURED/NOT_RUN）、`checks.json` 19 项全 PASS、`extraction_provenance.json`（源文件哈希台账）。

## 3. 核心客观数字

### 3.1 Strict20 决策关系计数（选集×目标联合判定，全部 20 条件输出，无筛选）

| 比较 | SAME_SET_SAME_OBJECTIVE | DIFFERENT_SET_DIFFERENT_OBJECTIVE | 其他 |
|---|---:|---:|---|
| A: StandaloneTopK vs Greedy | 16 | 4（w3/K3、w3/K4、w4/K4、B5-C4/K4，静态冗余） | 0 |
| B: WFC vs Greedy | **20** | 0 | 0（E2 16 条件逐步选择也一致；B5-C4 4 条件同集同前沿） |
| C1/C2: PCB/LDR vs Greedy | 0 | 20 / 20 | 0（K4 时 PCB/LDR==Exact 但仍≠Greedy） |
| D: Greedy vs Exact | 5（全部为 K=1） | 15 | 0 |
| E: 1-Swap vs Exact | 18 | 2（w1/K3、w2/K3：12 个 1 换 1 邻居均无严格改善，≥3 项同步替换才可离开） | 0 |
| F: PCB vs LDR（补充客观事实） | 20 | 0 | 0 |

- Exact：20/20 全子集枚举最优证书（n=7，|S|≤K 子集数≤99≤门禁 794；e3、e5_step0、tradeoff 三处独立一致）。
- 达到共同 Exact 目标的条件数：Exact 20、Greedy 5、1-Swap 18、WFC 5、StandaloneTopK 5、PCB 5、LDR 5。
- 分歧首步示例（`selection_disagreements.csv` first_divergence 列含全部条件的该字段）：B5-C4/K2 第 1 步 Greedy 选 N20（边际 364.116ms）而 Exact 集内当步最优替代 N18（279.340ms）；差异候选的独立收益、位置 L、兼容关系在 only_in_a/only_in_b 列逐条在案。

### 3.2 C/K3（模型化成本，modeled ms）

| 策略 | 选集（CPU 侧 ID） | J | 备注 |
|---|---|---:|---|
| FlowState-Greedy | C_D_P07168, C_C_P06144, C_B_P05120 | **154.3490** | |
| Greedy+1-Swap | 同上 | 154.3490 | 与 Greedy 同集同 J |
| StandaloneTopK | C_D_P07168, C_D_P07104, C_C_P06144 | 348.2340 | C_B 无恢复源（E=0,G=5120）；**与 Marconi Adaptation 同集同 J** |
| Marconi Adaptation | 同上 | 348.2340 | CPU 合成代理输入（alpha=1.0，flop 代理=位置 L，bytes=1） |
| WFC | C_ROOT_P01024, C_CD_P03072, C_AB_P02048 | 471.8743 | 固定浅共享 |
| LRU | C_D_P07168, C_JOIN_P04096, C_JOIN_P04032 | 582.0347 | 保留 2 个未来 JOIN 状态 |
| KVFlow-inspired | C_AB_P01984, C_AB_P02048, C_A_P04032 | 627.6150 | |
| Exact | — | NOT_AVAILABLE | 门禁拒绝，无证书 |

首步分歧：Greedy 选 C_D_P07168（q=1, r=7168）vs WFC 选 C_ROOT_P01024（q=4, r=4096）。**警示：C/K3 各 T=4096–7168 低于恢复模型校准目标区间（32768–131072），非零成本项均标记 OUTSIDE_CALIBRATION_TARGET_RANGE；J 为模型数值，非 GPU 延迟。**

### 3.3 旧 573 条（仅旧逻辑矩阵证据）

| 数据集 | 条件/快照 | 存在共享 | 存在重叠 | K<K_min | Greedy==PCB/LDR/WFC 同集 | 证书 | Strict Runtime 认证 |
|---|---|---:|---:|---:|---:|---|---|
| OpenHands | 504/168 | 0 | 504 | 90 | 504 | SEPARABLE_SUPPORT_TOPK×504 | 0 |
| AgentX | 69/23 | 66 | 33 | 0 | 69 | ALL_CANDIDATE_FRONTIERS_ATTAINED×66 + SEPARABLE_SUPPORT_TOPK×3 | 0 |

Greedy 在旧模型内 573/573 最优（可分支持/全候选前沿的代数证书）；这些是旧 Formal 矩阵证据，**不能升级为 Strict Runtime 认证**（AgentX 既有 Formal-only 兼容边问题未修复）。

### 3.4 GPU 证据链（只登记已有实测，两种协议不混算）

| 实验 | 条件/策略 | 生命周期 | 关键实测（中位） | 选集执行方式 |
|---|---|---|---|---|
| b5d4b | B5-C4/K2 Exact vs Greedy | 10 臂+Cold，10/10 | 见 links 行（含研究仪表口径） | 运行内求解（冻结历史重放 Epoch + Decimal Oracle 复核） |
| E4-1B | w1/K2 Exact vs Greedy（Pending A/C） | 18 | e45 stats（n=3） | 冻结选集执行 |
| E5-1 | w1/K3 Exact vs 1-Swap（A/B/C/D） | 36 | 三组 Exact−Swap 总和 −13.4/+1.9/−3.1 ms（方向不一致） | 冻结选集执行 |
| c_formal m* | C/K3 FS/SA/WFC × 4 Pending × 5 | 60（主矩阵）+2 门禁 | 例：FS C_B TTFT 37.63ms、CUDA prefill 30.81ms、E=5120、prefill=32；FS−SA 配对 TTFT 中位差 −163.2ms（C_B），C_A +1.0、C_C/C_D 同前沿 | 冻结选集执行（PILOT_POLICY 读 policy_plans.json） |
| Spec 8 a1/b1 | B5-C4/K2、C/K3 经生产 plan_strict | 2（最小正确性验证） | a1: Exact 在线 0.222861ms 纯求解；b1: Greedy 在线 0.402763ms | **生产 plan_strict 在线（Scheduler 进程内）** |

60 臂逐臂核验：census 16=16 映射选集==policy_plans、Native E/G==expected_E_G、cold_output_match 全 True。**不存在 GPU 实测的其余选集全部标 NOT_MEASURED（140 行），未根据模型化 J 推测实际 TTFT。**

## 4. 数据不一致与缺失证据（客观清单，未修改任何源文件）

1. **目录名差异**：任务文 `p1_strict_baseline_validation` → 实际 `p1_strong_baseline_validation`。
2. **C/K3 无独立 Exact 证书**：NOT_AVAILABLE（不启动新枚举）。
3. **Marconi 适配公平性 UNVERIFIED**：Strict20/B5-C4 缺 last_access/flop_saved 评分元数据（b5b0 门禁+b4c 审计在案）；C/K3 的 CPU 行用合成代理输入，非公平复现。
4. **PCB/LDR 无 C/K3 运行记录**；LRU/KVFlow-inspired 仅 CPU 合成输入、无 GPU。
5. **`paper_allocation_structure_gpu_ck3.csv` 的 n 列**：FS/SA 的 C_B 行 n=6（含 g1/g2 门禁臂），WFC C_B n=5；该文件为归因审计产物，正式统计须用 m* 主矩阵重算（本表已按 60 臂单独重算）。
6. **两种计时协议不混算**：b5d4b/E4/E5 含研究仪表（audit-inclusive），c_formal 为生产路径 TTFT（A 类仪表已移出）；`audit_attribution.csv`（p01_CB_*）属 c_pair 去污染协议，亦不混入。
7. **同名 Node 跨快照含义不同**：各快照内 B5C3_NODE_* 的位置 L 不同（roster 逐条件记录），不跨 Engine 复用物理 ID。
8. **20 条件≠20 独立工作负载**（5 快照×4 预算重复条件）；EPOCH_ADMISSIBLE≠RESUME_VALIDATED，GPU 侧来源血缘 PROVENANCE_UNVERIFIED。
9. **C/K3 J 的校准区间外警示**（见 §3.2）。
10. 数据级不一致：**未发现**（19/19 检查 PASS，含 140 行独立复算、跨产物 J/选集交叉、60 臂 census/E-G/Cold、573 行数与证书分布）。

## 5. CSV 字段说明与原始产物路径

- **strict_allocation_matrix.csv**（148 行）：condition_set(strict20/ck3)、case、K、n_candidates、n_pending、snapshot_source（快照原始文件路径）、candidates_roster（JSON：每候选 id/L/兼容 Pending）、policy、selected_ids、J_ms、gap_exact_ms、per_pending_EG（JSON：pid/T/E/G）、exact_optimality_certificate、swaps_applied（1-Swap）、gpu_evidence、notes。← 原始：e3 condition_results.csv + E2/b5c4 快照 + kvflow cpu_results/review/tradeoff。
- **selection_disagreements.csv**（148 行）：comparison（A/B/C1/C2/D/E/F + ck3 的 G/H/I/J）、relation（SAME_SET_SAME_OBJECTIVE / DIFFERENT_SET_SAME_OBJECTIVE / DIFFERENT_SET_DIFFERENT_OBJECTIVE / UNVERIFIED）、双方选集/J/ΔJ、pendings_with_EG_diff、only_in_a/only_in_b（差异候选 L/兼容/独立收益和）、first_divergence（逐步分叉步+双方当步边际；静态规则标无轨迹）、trace_evidence、gpu_evidence、notes。← 原始：e3 selection_traces/swap_traces + kvflow selection_traces。
- **baseline_input_contract.csv**（11 行）：method、implementation_source（文件:行）、selection_rule、decision_inputs、tie_break、frozen_params、workloads_compared、strict20_status、ck3_status、selection_protocol（离线复算/冻结选集/运行内求解/生产 plan_strict）、adaptation_vs_original（仅引既有源码与审计）、missing_metadata。
- **gpu_evidence_links.csv**（171 行）：experiment、condition、policy_cpu、pending、cpu_selection、cpu_J_ms、mapping_evidence（同 ID 命名空间或 census 16=16）、execution_mode、gpu_arms、n_lifecycles、pending_coverage、native_E/H/G、prefill_tokens、cuda_prefill_ms_median、ttft_ms_median、resume_vs_cold、stat_method、artifacts、notes。
- **checks.json**：19 项一致性检查（全部 PASS，含检查对象与细节）。**extraction_provenance.json**：全部读取源文件的 sha256 台账。

## 6. 停止声明

本任务为数据提取与复核，不含创新性判断、论文故事设计或后续实验规划；按规格在此停止，不自动启动新实验、不修改求解器、不撰写论文。
