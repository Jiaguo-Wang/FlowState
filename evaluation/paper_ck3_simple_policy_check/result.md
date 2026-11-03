# C/K3 简单策略等价性与 Exact 最优性复核报告（Task 2）

**总判定：全部检查 PASS（15/15 项一致性核验通过）。** PASS 仅表示下述事实得到验证，不构成创新性、SOTA 优势或 GPU 上的性能结论。本轮 CPU-only、只读冻结输入，未修改生产代码与门禁、未启动 GPU。

## 1. 固定输入与来源

- 冻结 Epoch：`kvflow_multiworkflow_cpu_main/workloads.json` MW_CPU_C——16 候选（原始 ID/位置 L/Strict Compatibility 原样）、4 个 Ready Pending（C_A/C_B/C_C/C_D，T=4096/5120/6144/7168）、K=3；Epoch 构建逐行复刻 `run_cpu.py epoch(case,k)`（scope='current'，JOIN 候选当前兼容为空）。
- 冻结评分：`flowstate/recovery_model.py`（经 p1 `common.score`，E=max 兼容已选位置、G=T−E、J=ΣPhi）。
- 冻结参照：`cpu_results.json` C/K3 五行 + `kvflow_multiworkflow_cpu_review/review_results.json` task2 + `paper_solver_tradeoff_summary.csv`（source=ck3 四行）。三处参照重算全部逐位一致（含逐 Pending E/G）。
- 输入与来源 sha256 台账见 `checks.json → input_ledger`。

## 2. 检查A：PCB 与 LDR —— 均复现 FlowState-Greedy（PASS）

| 策略 | 选集 | J (ms) | 与 FS 关系 |
|---|---|---:|---|
| PCB | C_D_P07168, C_C_P06144, C_B_P05120 | 154.34895561599998 | SAME_SET_SAME_OBJECTIVE |
| LDR | C_D_P07168, C_C_P06144, C_B_P05120 | 154.34895561599998 | SAME_SET_SAME_OBJECTIVE |
| FlowState-Greedy（冻结） | 同上 | 154.34895561599998 | 参照 |

两策略使用 `p1_strong_baseline_validation/pcb.py、ldr.py` 原实现，无语义改动。机制：C/K3 中每个 Ready Pending 的最深合法候选恰好也是边际收益最大者（PCB 按 Phi 独立收益、LDR 按深度 L 排序去重，前 K=3 个不同 ID 一致落到 D/C/B 三个满覆盖候选），因此三种简单规则在本条件收敛到同一选集。排序与 tie-break 细节（(-score,cid,pid)、前 K 去重）已在 `checks.json` 记录，可复算。

## 3. 检查B：Gap-only Marginal Greedy —— 复现 FlowState-Greedy（PASS）

J_gap(S)=Σ_p(T_p−E_p(S))，每轮选边际缺口下降最大者，平局取 checkpoint_id 最小，预算 K=3（规则同生产 Greedy）：

| 轮 | 选择 | 边际缺口下降 | J_gap 变化 |
|---|---|---:|---|
| 1 | C_D_P07168 | 7168 tokens | 22528→15360 |
| 2 | C_C_P06144 | 6144 tokens | 15360→9216 |
| 3 | C_B_P05120 | 5120 tokens | 9216→4096 |

最终选集 == FlowState-Greedy，逐 Pending E/G 相同，用冻结 Position-aware Phi 重算 J=154.34895561599998（SAME_SET_SAME_OBJECTIVE）。每轮全部 16 候选边际值见 `gap_greedy_trace.json`。本事实只说明"该条件下两种估值导向同一选集"，不据此断言 Gap-only 或 Position-aware 在真实 GPU 上更优。

## 4. 检查C：独立 Exact 枚举 —— Greedy 全局最优（PASS）

- 枚举规模：C(16,0)+C(16,1)+C(16,2)+C(16,3)=**697**，全部以冻结 Strict Compatibility + Position-aware Phi 评分（生产 `plan_strict()` Exact 门禁未触碰，仍为 n≤12 拒绝 n=16）。
- 全局最小 J=**154.34895561599998**，唯一 argmin={C_B_P05120, C_C_P06144, C_D_P07168}（容差 1e-9，无并列）。
- **FlowState-Greedy 达到全局最优**：绝对差距 0.0 ms、相对差距 0.0。次优集 {C_B_P05056,…} J=156.8207（+2.47 ms）。
- 一致性：Greedy 重算 J == 冻结 `cpu_results.json` 值；`paper_solver_tradeoff_summary.csv` 的 ck3 四行（Greedy/1-Swap/Swap-only/WFC）选集与 J 与枚举评分全部一致。完整证书（含 top5 子集、输入哈希）见 `exact_enumeration.json`。

## 5. 检查D：零边际收益复核（PASS）

| 策略 | 候选 | J(S\{c})−J(S) | 判定 |
|---|---|---:|---|
| StandaloneTopK | C_D_P07168 | +2.5150 | 非零（D 前沿 7168→7104） |
| StandaloneTopK | **C_D_P07104** | **0.000000** | **最终边际恰为零** |
| StandaloneTopK | C_C_P06144 | +233.8007 | 非零 |
| WFC | **C_ROOT_P01024** | **0.000000** | **最终边际恰为零** |
| WFC | C_CD_P03072 | +154.6847 | 非零 |
| WFC | C_AB_P02048 | +76.8956 | 非零 |

- StandaloneTopK 确实保留了零边际收益的冗余候选：C_D_P07104 被同集合中更深的 C_D_P07168 完全覆盖（E_D=max(7168,7104)=7168），删除后所有 Pending E/G 与总 J 均不变。
- WFC 确实保留了不再产生恢复收益的浅共享候选：C_ROOT_P01024（L=1024，兼容全部 4 个 Pending）在集合内被更深的 C_AB_P02048（A/B 2048）与 C_CD_P03072（C/D 3072）逐 Pending 支配，删除后 E/G 与 J 均不变。
- 以上为策略行为事实（静态独立排序/固定浅共享规则的直接结果），不解释为策略实现错误。

## 6. 数据一致性、执行限制与异常

- 15/15 项检查全部 PASS（清单见 `checks.json`）：Epoch 身份、三组冻结参照重算、review 交叉、四项事实检查、枚举规模与一致性、两项零边际验证。
- 限制：J 为冻结模型数值（非 GPU 延迟）；C/K3 各 T=4096–7168 低于校准目标区间（cpu_results 标记 OUTSIDE_CALIBRATION_TARGET_RANGE）；StandaloneTopK 行取自与 Marconi 同集同 J 的冻结行并经 review task2 独立复核；Exact 仅审计证书，生产门禁未改。
- 无数据不一致；未发生异常。

## 7. 产物清单

`policy_comparison.csv`（7 策略对照）、`exact_enumeration.json`（697 子集证书）、`redundancy_check.csv`（逐候选最终边际，含删除前后逐 Pending E/G）、`gap_greedy_trace.json`（Gap-only 逐步轨迹）、`checks.json`（15 项检查+输入哈希台账）、`check.py`（复算脚本）。
