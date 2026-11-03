# Evaluation 实验清单(experiment_inventory)

更新:2026-10-31 | 用途:论文数据包 `evaluation/evaluation_data_package/` 的来源索引;每项标注状态与数据位置。CPU/GPU 分开标注。

## 1. Recovery Cost Model(恢复成本模型)

| 项 | 位置 | 状态 |
|---|---|---|
| 位置矩阵校准(20 格=T{32K,64K,96K,128K}×G{0,4K,8K,16K,32K}) | `runtime_artifacts/recovery_position_audit_20260826_144654_852303/` | GPU,已冻结 |
| 长 gap 剖测(T=128K×G 至 128K,9 格) | `runtime_artifacts/recovery_profiler_128k_20260826_133712_596813/` | GPU,已冻结 |
| held-out 验证(12 点)+ 模型选择 M0/M1/M2 | `runtime_artifacts/recovery_model_freeze_20260826_154235_266020/`(heldout_predictions/summary/raw.csv、calibration_points.csv 20 点、candidate_models.json、model_selection.json) | GPU,已冻结;**MAPE 2.68%/15.29% 出处** |
| 早期模型对比(Step 9D,非论文数字) | `recovery_profiler_v2/`(model_comparison.json,Old Phi 2.247% 等) | GPU,已被 freeze 实验取代——**勿混用** |
| Step 9B 冻结 Φ vs 实测审计 | `RECOVERY_MODEL_AUDIT.md` + `recovery_model_audit.py`(数据 `runtime_artifacts/sota_latency_20260825_113526_839592/`) | GPU,4K–16K 系统性高估 23–36% |
| 短上下文使用域审计(外推/保守性/决策零差异) | `paper_recovery_model_short_context_audit/`(partB_pairs 24、partB_absolute 10、audit_data.json、audit.py) | CPU 审计,READY |
| 敏感性矩阵(4 代表点×21 扰动) | `cost_model_sensitivity.json` + `COST_MODEL_SENSITIVITY.md` | CPU |

实验模型:Qwen3.5-9B(单卡 H100 PCIe,SGLang 0.5.17,TTFT 客户端流式计时;环境见各制品 environment.txt)。冻结模型 M2 `position_aware_quadratic_v1`:Φ(G,T)=37.828150·g+0.345974143·g·t−0.156201917·g²(g/t 单位 Ki-token,输出 ms)。

## 2. Checkpoint Allocation(分配有效性;全部 CPU 模型化)

| 项 | 位置 | 状态 |
|---|---|---|
| legacy573 复核(OpenHands 504 条件/168 快照,无共享;AgentX 69/23,66 共享) | `paper_allocation_structure_legacy573.csv`(提取脚本 paper_allocation_structure_extract.py) | 已复核 |
| Strict20(20 例域内 Exact 全枚举+各法 J) | `paper_allocation_structure_strict20.csv`;结构再证 `paper_workload_structure_final/`(strict20_optima_nesting、sensitivity 80 行、summary.json) | 已复核;Exact 最优集唯一 20/20;非嵌套为结构性 |
| K1–K4 预算敏感性(191 快照×4 预算×4 策略,764 条件) | `rq3_workload_structure/budget_sensitivity_k1_k4_20261030/`(strategy_results 3820、per_snapshot 764、oracle 765、pending 13648、reproduction 2292、timing 51780、divergence 14) | 2026-10-30 PASS(另一会话产物,本包只读引用) |
| GPU C/K3 分配↔运行桥(逐 Pending E/G/prefill/TTFT 中位) | `paper_allocation_structure_gpu_ck3.csv`(n=5) | 与本包 gpu_per_pending.csv 同源可互证 |
| 独立多分享工作负载(4组×2任务×2Session 嵌套结构) | `independent_multishare_workloads/`(final_report.md、causal_checks.json 等) | READY(结构证据,供共享/替代关系论述) |
| P1 提交件(分配结构) | `submission_p1_allocation_structure/`(budget_structure_analysis.csv、paired_summary.json 等) | 已封存提交 |

关键数字:AgentX K1 分歧 9/23(共享快照);FS 对 PCB/LDR 3胜20平0负、对 WFC 6胜17平0负;K≥2 全部同集同 J;OpenHands 仅 K3 一例 0.0477ms 差;Oracle 门内 FS 396/396 最优。

## 3. Solver Quality & Overhead(求解器)

| 项 | 位置 | 状态 |
|---|---|---|
| Strict20 质量-耗时微基准(5 方法×20 例) | `paper_solver_tradeoff_summary.csv`(104 行)+ `paper_solver_tradeoff_raw.csv`(12480 行)+ `paper_solver_tradeoff_bench.py`;审计 `paper_solver_tradeoff_audit.md` | CPU;本包已独立复算 |
| 预算敏感性求解计时(191 快照×4 策略+门内 Exact) | `rq3_workload_structure/budget_sensitivity_k1_k4_20261030/solver_timing_raw.csv`(51780 批) | CPU;本包独立聚合与登记一致 |
| C/K3 简单策略核查(697 子集全枚举) | `paper_ck3_simple_policy_check/`(exact_enumeration.json:argmin 唯一==Greedy,gap 0;policy_comparison.csv) | CPU,已复核 |
| Strict 证据提取(分配矩阵 148 行+GPU 证据链) | `paper_strict_evidence_extraction/`(strict_allocation_matrix.csv、gpu_evidence_links.csv、fact_report.md) | 已复核 |
| Hybrid Solver GPU 闭环(在线 Exact/Greedy 双分支) | `paper_hybrid_gpu_gate_a/b`、`paper_hybrid_gpu_fix_a/b`(报告 paper_hybrid_solver_gpu_gate.md);G1 `hybrid_solver_gpu_gate_g1/` | GPU,Prior PASS(A 21/21、B 22/22、G1 8/8)——本包未重跑,引用其报告 |
| 求解器接入路径审计 | `paper_solver_integration_audit.md` | 只读审计 |

## 4. End-to-End GPU(三策略五轮矩阵)

| 项 | 位置 | 状态 |
|---|---|---|
| r1 Pilot(3 臂首轮) | `p0_control_cycle_gpu/arms/r1_ck3_{fs,mc,kv}/` + taskA/taskB 报告 | PASS,**不进五轮统计** |
| 五轮矩阵(15 臂) | `p0_control_cycle_gpu/arms/m{1..5}_ck3_{fs,mc,kv}/` + `five_round_ledger.jsonl`(15 行)+ 原报告 `five_round_matrix_audit_report.md` + 逐轮审计 R2/R3 | 15/15 PASS;本包独立复算见 `five_round_audit.md` |
| 三策略 T2-G1/G2(60+40 臂,TTFT 模型) | `t2g1_three_policy_gpu/formal_gpu1b/`、`b5c4_formal_gpu_g2/` | Prior PASS(TTFT≈39+0.033·(T−E) R²=0.985) |
| C 正式配对/配对计时 | `c_formal_perf_gpu/`、`c_pair_timing_gpu/` | Prior PASS |
| 控制成本归因/Epoch 瓶颈/短上下文成本 | `paper_control_cost_final_audit*`、`paper_runtime_epoch_bottleneck_audit*`、`paper_recovery_model_short_context_audit*` | 只读审计 |

## 5. 数据包文件(本包新增)

`recovery_model_data.csv`(301 行)、`allocation_data.csv`(618 行)、`solver_data.csv`(111 行)、`gpu_per_round.csv`(15)、`gpu_per_pending.csv`(60)、`gpu_overhead.csv`(3)、`five_round_audit.md`、`figure_data/`×5、`source_manifest.json`(来源 32 个含 sha256)、审计脚本 `audit_scripts/`×2。

## 6. 已知边界(整理时发现,写作时注意)

1. `recovery_profiler_v2` 的 MAPE(Old Phi 2.247% 等)是早期实验,**不是**论文的 2.68/15.29;论文数字唯一出处是 freeze 目录,口径=逐点相对误差**算术均值**(若用中位口径 M2=1.39%)。
2. AgentX 侧为 Formal 模型关系,不等于真实 token 前缀可恢复性(budget 报告 §8);OpenHands 为四独立 Session 受控组合,非原生多路 fork。
3. legacy573 的 AgentX `strict_runtime_certified=0`:strict 运行时证书未覆盖 AgentX 条件。
4. Hybrid GPU 闭环为既有报告结论,本包未重验。
