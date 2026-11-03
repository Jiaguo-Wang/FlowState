# Evaluation 实验数据独立审计总报告(data_audit_report)

日期:2026-10-31 | 范围:五轮 GPU 矩阵、Recovery Cost Model、Checkpoint Allocation、Solver Quality、数据包组装 | 纪律:零 GPU、零模型加载、零历史数据修改;仅新增审计脚本/数据/报告(注释全中文)。

**总裁决:四族实验数据全部完成独立审计;五轮 GPU 核心统计与原报告逐位一致;MAPE 2.68%/15.29% 由原始数据独立复算证实;发现原五轮报告 3 处表述缺陷与 1 处口径歧义(均不影响统计结论);证据缺口 6 项已标注(§6)。**

## 1. 五轮 GPU 实验(第一阶段,详见 five_round_audit.md)

- **15/15 臂独立复核全过**:正确性九项/身份(镜像+Runtime 16 文件+Runner/Observation 完整 SHA)/9 装配输入逐字节(15 臂+r1 互证)/泄漏检查(请求全部晚于 T3,选择不可能用未来信息;oracle_match×15)/总账 15 行交叉核对。
- **独立复算与原报告一致**:ΣTTFT 配对中位 FS−MC **−470.208 ms**、FS−KV **−429.711 ms**;逐轮值、逐 Pending 中位、selection/control/full/request-exec 中位全部同值。40 格配对符号:正格 10/40(全部 C_A,E=0 物理必然),负格 30/40。
- **R3 敏感性**:排除 R3 后中位 −469.575/−432.282(含 R3 −470.208/−429.711),结论不变。
- **发现的原报告缺陷**(不影响数字,需在论文中避免沿用):①§2.9 Observation SHA 笔误 `7c1dcef5`→正 `7c1dcefc`;②§2.8 MC/KV 选集集合写错(实测 MC={C_JOIN_P04032,C_JOIN_P04096,C_ROOT_P01024}、KV={C_AB_P01984,C_AB_P02048,C_A_P04032},均 3 枚);③§3.1 改善% 未注明分母——聚合口径(−64.22%/−62.78%)与逐轮口径(−64.23%/−61.74%)在 FS−KV 差 1.0pp;④"12–16%" J 保守度实测 11.66–15.71%(分母=|J 差|)。
- **新增数据**:TPOT(decode 段)三策略中位 71.9–74.0 ms、配对差逐轮换向无稳定方向——选集影响集中于恢复段;控制周期六级阶段精确分解(残差 0.0 ms),策略特有开销=selection_call 16.4 vs 2.0/4.4 ms,共同 Runtime(观察/认证/preflight/postcheck)与间隙占控制周期 ~99%。

## 2. Recovery Cost Model

- **MAPE 独立复算(由 heldout_predictions.csv 12 点从零重算)**:M0(gap-only)**15.291275%**、M1(bilinear)4.635750%、M2(位置感知 quadratic)**2.682480%**,与 model_selection.json 登记值六位小数一致。口径=**逐点相对误差算术均值**;若改用中位口径 M2=1.389%——论文引用必须注明均值口径。样本=12 held-out (T,G):T∈{49152,81920,114688}×fraction{0.125,0.25,0.5,0.75};分母=实测增量恢复成本(measured Φ=同 T 均值−G=0 基线均值)。
- **2.68% vs 15.29% = 位置感知 vs 不考虑位置**,同一样本、同一协议、预注册选择(M2 冻结为 `position_aware_quadratic_v1`);heldout 未参与拟合(`heldout_used_for_fit=false`)。
- 校准域:20 点(位置矩阵 16+长 gap 4),Qwen3.5-9B/H100;**短上下文使用域(T≈3.4–7.2K)在校准域(T≥32K)之外**——属外推,方向一致 19/19,Φ 保守高估中位 1.313×(12 格,区间 1.126–1.485);21 冻结条件下 M0/M2 决策零差异(短上下文审计)。
- 混淆警示:`recovery_profiler_v2/model_comparison.json` 的 Old Phi 2.247% 是早期 Step 9D 数字,**不得**与 2.68/15.29 混用。

## 3. Checkpoint Allocation(全部 CPU 模型化)

- **规模**(legacy573):OpenHands 504 条件/168 快照(候选 8–20,无共享,504/504 可分 TopK 结构);AgentX 69 条件/23 快照(候选 6–149,66/69 含共享,33 含重叠)。
- **K1–K4 预算敏感性**(191 快照×4 预算×4 策略,2026-10-30):754/764 条件四策略同集同 J;FS 全程 0 负(对 PCB/LDR/WFC 合计 3+3+6 胜全部在 AgentX K1、1+1+1 胜在 OpenHands K3 g051 例 0.0477ms);门内 Oracle FS **396/396 最优**(LDR/PCB 各 395、WFC 394);嵌套保持 2292+297 比较零违背(预算内);14 个分歧案例明细已入包(共享 vs 专属深度两种相反竞争,论文实例素材)。
- **Strict20**:Exact 全枚举最优集唯一 20/20;Greedy 域内最优 **5/20**,1-Swap 修复 13/15 → **18/20**;非嵌套为结构性(11/15 必然,paper_workload_structure_final)。
- **GPU 桥**:C/K3 冻结选集在五轮 GPU 复现(E/J 逐位),gpu_ck3 中位与五轮实测同域互证。

## 4. Solver Quality & Overhead(CPU 微基准与 GPU 闭环分开)

- **Strict20 微基准**(本包复算):Exact 中位 **0.220 ms**(20/20 最优);Greedy **0.092 ms**(5/20 最优,差距中位 31.8 ms);Greedy+1-Swap **0.221 ms**(18/20)——1-Swap 以 ~2.4× Greedy 时长换回 13/15 差距,WFC 0.025 ms 但仅 5/20。
- **191 快照求解计时**(51780 批,本包独立聚合,与登记逐位一致):FlowState 0.10451 ms、PCB 0.00981、LDR 0.00408、WFC 0.02680、门内 Exact 0.29187(中位口径:条件内批均→跨条件中位;Exact 仅 396 门内条件,不可跨域直接相除)。
- **C/K3 全枚举**:697/697 子集,argmin 唯一且 ==Greedy(gap=0,tol=1e-9)。
- **GPU 闭环**(引用既有报告,本包未重验):Hybrid A 21/21 Exact、B 22/22 Greedy、G1 8/8 在线双分支;生产 `plan_strict` 混合路由=Greedy 基线+域内 Exact 门控。

## 5. 可直接用于论文图表的数据

| 图表 | 数据文件 | 备注 |
|---|---|---|
| 恢复模型预测 vs 实测散点(M2/M0) | `figure_data/fig_recovery_pred_vs_measured.csv` | 12 held-out 点;MAPE 标注均值口径 |
| 五轮 ΣTTFT 配对(逐轮+中位) | `figure_data/fig_gpu_sum_ttft_paired.csv` | 五轮全负 |
| 逐 Pending 配对差箱线/条形 | `figure_data/fig_gpu_ttft_paired.csv` | C_A 正、C_B/C/D 负的结构图 |
| 预算敏感性胜负堆叠(K1–K4) | `figure_data/fig_budget_sensitivity.csv` | AgentX K1 分歧窗口 |
| 求解器质量-耗时权衡 | `figure_data/fig_solver_tradeoff.csv` | 5 方法 20 例 |
| 逐轮明细/开销分解 | `gpu_per_round.csv`、`gpu_per_pending.csv`、`gpu_overhead.csv` | 全部带 source_file 溯源列 |

## 6. 证据缺口与不可复核项(如实标注)

1. **AgentX Formal≠Raw**:模型关系不含真实 token 前缀/物理别名认证(budget 报告 §8;legacy573 `strict_runtime_certified=0`)——AgentX 数字只能进模型化结论,不得写成 GPU 实测。
2. **Oracle 覆盖**:368/764 条件超 Exact 门禁无 Gap;不以后补枚举充证。
3. **Hybrid GPU 闭环数字**(21/21、22/22、8/8)引用既有报告,本审计未重跑。
4. **控制周期净收益**:Control Cycle 与 TTFT 无定义口径的合成,本包不做(维持禁令)。
5. **MAPE 中位口径差异**(2.68 均值 vs 1.39 中位):论文须写明口径,防审稿质疑。
6. **五轮矩阵单案例**(C/K3,4 pending,n=5):外推到其他工作负载/K/并发未验证;27B 排除在授权外。

## 7. 结论

- 全部四族实验**完成数据审计**,数字可追溯(每行 source_file+sha256 清单 source_manifest.json)。
- 五轮 GPU:独立复算与现有报告一致(除 3 处表述缺陷)。
- 数据包 13 文件+5 图表数据+2 审计脚本就位,`evaluation/evaluation_data_package/`。

**审计到此完成;未撰写任何论文正文,未修改任何历史实验数据与既有报告。**
