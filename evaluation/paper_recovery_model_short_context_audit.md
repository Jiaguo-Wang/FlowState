# FlowState P1-1：短上下文恢复成本模型适用性与决策敏感性审计报告

日期：2026-10-28。判定：**READY**（五项任务全部完成；绝对精度部分如实标注外推限制与 NOT_IDENTIFIABLE 边界）。全程 CPU-only 只读：未启动 GPU、未修改 Runtime/Optimizer/Controller/Recovery Model、未重新拟合冻结 Φ 参数、未修改任何正式选集与历史实验结果、未重跑 T2-G2、未新增实验矩阵或基线。

产物目录 [paper_recovery_model_short_context_audit/](paper_recovery_model_short_context_audit/)：
[audit.py](paper_recovery_model_short_context_audit/audit.py)（一键复算，<10s）、[audit_data.json](paper_recovery_model_short_context_audit/audit_data.json)、[partB_pairs.csv](paper_recovery_model_short_context_audit/partB_pairs.csv)（24 配对明细）、[partB_absolute.csv](paper_recovery_model_short_context_audit/partB_absolute.csv)（10 绝对口径格）、[partC_conditions.csv](paper_recovery_model_short_context_audit/partC_conditions.csv)（21 决策条件）。10 项检查全 PASS。

## 0. 口径

冻结 Φ（`position_aware_quadratic_v1`，[flowstate/recovery_model.py](../flowstate/recovery_model.py)）：Φ(G,T)=37.828150·g+0.345974143·g·t−0.156201917·g²（g/t 为 Ki-token，输出 ms；Φ(0,T)=0 精确）。**Φ 预测的是相对同 T、零缺口条件的额外恢复成本，不是完整 TTFT**——绝对比较仅用"同 Pending、同测量协议的 G=0 格实测 + Φ(G,T)"，禁止跨 Pending 借用基线。Gap-only 对照采用同一预注册候选池中的 **M0**（分段线性 knots，仅依赖 G；长上下文 held-out MAPE 15.29% vs M2 的 2.68%）。三组 GPU 对照与校准数据完全独立（校准=2026-08-26 专用 replay profiler；对照=T2-G2/C-K3/B5-C4 正式 Runner，不同日期、负载、协议；held-out 未参与拟合，`heldout_used_for_fit=false`）。

## 1. Task A：冻结模型与训练域核查

| 域 | 点数 | T 范围 | G 范围 |
|---|---|---|---|
| 校准（16 position-matrix + 4 long-gap） | 20 | 32768–131072 | 4096–131072 |
| Held-out（未参与拟合） | 12 | 49152–114688 | 6144–86016 |
| **实际使用域**（T2-G2 + C/K3 + B5-C4 GPU + Strict20 全 20 条件） | 24 GPU 格+20 条件 | **3471–7168** | **0–7168** |

**结论：短上下文使用域（T≈3.4K–7.2K）完全落在校准与 held-out 的 T 支撑域之外**（最高使用 T=7168 仅为校准最小 T=32768 的 21.9%），且 **G∈(0,4096) 无任何校准/held-out 覆盖**（校准最小 gap=4096）。该使用在**形式验证域内**（0≤G≤T≤131072，实现拒绝域外输入），性质是"域内向短侧外推"（declared-domain interpolation outside calibration support），不是域外截断。长上下文 held-out 精度：M2 MAPE 2.68%、R²=0.9994（M1 4.64%、M0 15.29%）——该精度声明**不能**移植到短上下文（§2 给实测外推表现）。

## 2. Task B：短上下文预测与 GPU 实测对照

### 2.1 同 Pending 跨策略配对差（24 对，三数据源）

预测差=Φ(G_a,T)−Φ(G_b,T)，实测差=同轮同 Pending 请求级配对中位（TTFT 与 Prefill CUDA 双口径）：

- **预测差≠0 的 19 对：方向一致 19/19**（双口径全部同号）——覆盖 T2-G2 的 FS−MC/FS−KV（8 对）、C/K3 的 FS−SA 1 + FS−WFC 4 + SA−WFC 4（9 对）、B5-C4 的 Greedy−Exact 结构差异对（2 对）；
- **同前沿控制对 5 个**（pred=0）：C/K3 C_A FS−SA（E/G 相同，|Δ|=1.0ms）、C_C/C_D FS−SA（同 E/G，全量中位 +6.6/+21.1ms 为共享机器双峰模式残差，快模式口径 +2.8/+2.3ms 在噪声带内，final_report §3.2 已归因）、B5-C4 A/B（E 相同 3712，|Δ|≤0.7ms）——**同前沿 ⇒ 请求成本等价**得到再次验证，且标定出"非恢复段混杂 ≤3ms"，远小于 84–163ms 级恢复差（即下文跨策略借 G=0 基线的系统偏差 <4% 量级，可忽略）；
- **幅度：Φ 系统性高估**。预测/实测比值（12 个深恢复负差格）中位 **1.313**，范围 [1.126, 1.485]；缺口单价斜率对照（OLS，ms/Ki-token）：实测 T2G2 34.4 / C-K3 37.3 / B5-C4 30.4 vs Φ 39.0/39.0/39.1（Φ/实测=1.134/1.046/1.287）——高估来自长上下文校准单价在短上下文的保守外推，方向稳定（10/10 绝对格误差全正）。

### 2.2 绝对预测误差（仅同 Pending G=0 基线格）

10 格（T2-G2 C_B/C_C/C_D×MC/KV 与 C/K3 C_B×SA/WFC、C_C/C_D×WFC，基线=同 Pending FS 的 G=0 深恢复格）：|误差| 中位 **33.5ms**、max 37.8ms，相对误差 10.7%–34.9%（全部高估）。**NOT_IDENTIFIABLE 边界**：T2-G2 与 C/K3 的 C_A（三策略均 G>0、无 G=0 格）与 B5-C4 全部 Pending（最小 G=384）无同协议零缺口基线，绝对误差按 spec 不计算，只用相对配对差——未人为补造任何零缺口基线。

### 2.3 偏差来源分解

TTFT 配对差与 Prefill CUDA 配对差 **R²=0.9999**、|TTFT−PrefillCUDA| 中位 1.1ms（max 2.6ms）→ 配对差由恢复计算（Prefill）本身主导；请求固定成本（queue 3.3/pregap 1.3/delivery 2.8ms）在配对中抵消，decode CUDA 在首 Token 之后与 TTFT 无关，控制 RPC（~11s）在请求窗口外。**偏差不是固定成本或协议混杂，而是 Φ 短侧单价外推偏高（保守方向：模型预测的恢复成本高于实际）。**

## 3. Task C：分配决策敏感性（Position-aware vs Gap-only）

冻结兼容矩阵/候选/预算下，M2 与 M0 各自跑 Marginal Greedy 与 Exact 全枚举（Strict20 五快照×K1-4 共 20 条件 + C/K3 共 21 条件；不同模型 J 不直接相减，统一在 M2 Φ 域评价）：

| 复核项 | 结果 |
|---|---|
| M2 Greedy 复现冻结选集（校准锚） | **21/21**（Strict20==e3 冻结、C/K3==policy_plans FS 选集） |
| Greedy 选集改变（M0 vs M2） | **0/21** |
| Exact 最优集改变（M0 vs M2） | **0/21** |
| Greedy 逐步选择序列完全相同 | **21/21** |
| 统一 M2 尺度下选集成本差 | 全部 0.0ms |

**零改变的原因（结构而非巧合）**：两模型下候选对每 Pending 的收益都是恢复深度 L 的严格增函数（M0 收益=f(T)−f(T−L)，M2 收益=Φ(T,T)−Φ(T−L,T)，各自单调），故逐 Pending 的候选深度排序在两模型间**保序**；本数据候选深度分离大且 K≤4，贪心每步的边际最优唯一且稳定，Exact 枚举（Strict20 每条件 128 子集、C/K3 697 子集）在两模型下取到同一唯一最优集。即：**模型误差（含 15.29%→2.68% 的精度差）不改变任何 Checkpoint 收益排序、选择顺序、最终选集与模型化成本**——决策由兼容/深度结构决定，对成本模型形状不敏感。

## 4. Task D：T2-G2 三策略预测与实测排序

- **J 复算逐位一致**：由 per-Pending E/G 重算 J：FS 154.349 / MC 703.455 / KV 627.615，与冻结值 154.35/703.46/627.61 相等（J_modeled 即 ΣΦ(G,T)）；
- **排序一致**：模型化 J 排序 FS<KV<MC == 实测四 Pending 等权 TTFT 中位排序 FS(72.3–78.4)<KV(175.4–182.5)<MC(186.6–188.4)，5 轮全部同序；逐 Pending 12 格方向全对（§2.1 的 T2G2 8 对 + 4 个 MC−KV 隐含对全同号）；
- **逐 Pending 收益方向一致**：预测 MC 优于 KV 的格（C_C/C_D：MC gap 更小）实测 MC 亦优；KV 优于 MC 的格（C_A/C_B）实测亦然（MC−KV 差由 FS−MC 与 FS−KV 两配对差相减隐含推导，4/4 同号）；
- **"排序正确但绝对预测偏差明显"成立**：J 差与实测差同号但幅度高估 13%–48%（中位 ~31%）。**因此不得仅凭三策略排序一致宣称模型在短上下文具有高预测精度**——正确表述是"方向/排序/机制层面一致，绝对值保守高估、已量化"。

## 5. Task E：论文结论划分与补实验裁决

**可直接写入论文的结论**：

1. **长上下文 Held-out 精度**（主文一句 + Appendix 明细）：M2 在 48K–112K held-out MAPE 2.68%、R²=0.9994，显著优于 gap-only M0（15.29%）与 bilinear M1（4.64%）——位置感知项必要；
2. **短上下文外推表现**（主文）：使用域 T=3.4K–7.2K 在校准支撑域（T≥32K）之外，但三组独立 GPU 对照（160 正式臂）上 19/19 非零配对差方向全对、三策略 J 排序==实测排序、TTFT−PrefillCUDA 残差中位 1.1ms；绝对成本保守高估（相对 11%–35%，单价 1.05–1.29×），**模型化的恢复收益是实测收益的保守上界估计**；
3. **模型对 Checkpoint 选择的影响**（主文）：21 冻结条件下 Position-aware 与 Gap-only 的 Greedy/Exact 选集、选择序列、统一尺度成本**零差异**——分配决策由兼容与深度结构决定，成本模型形状（含其短侧外推误差）不改变决策；这说明**简单规则何时够用、位置感知何时必要应从结构（共享/嵌套）论证而非模型精度论证**（与工作负载结构实验结论衔接）；
4. **模型化目标与 GPU TTFT 关联**（Appendix）：配对差层面 R²=0.9999 由恢复计算主导；绝对 TTFT 不可由 Φ 直接预测（Φ 不含请求固定成本与 decode），仅差分语义有效。

**裁决：NO_NEW_GPU_NEEDED**。理由：①论文当前的建模结论（选策略、选 Checkpoint、排序与机制解释）全部由现有 160 正式臂证据支撑，且模型误差不改变任何决策；②绝对精度的短板已量化并可在论文中以"保守上界"口径如实表述，不构成结论风险；③补短上下文校准剖面（4–8K 的 G/T 网格）只会改进绝对精度数字，不改变任何已发表方向性结论，属可选增强而非必需——若审稿要求绝对精度声明，可原样复用 replay profiler Runner 补采（不改 Runtime），此处仅记录路径、不启动。

## 6. 复算索引

| 结论 | 数据来源 | 口径 |
|---|---|---|
| 训练/使用域对照 | freeze_20260826 校准 20 点 + held-out 12 点 + 24 GPU 格 + Strict20 | T/G 支撑域比较 |
| 24 配对差方向与幅度 | partB_pairs.csv | 同 Pending 跨策略，TTFT+PrefillCUDA 双口径中位 |
| 绝对误差 10 格 | partB_absolute.csv | 同 Pending G=0 基线 + Φ(G,T) |
| 斜率/比值 | audit_data.json B_slope/B_compare | OLS on (G, prefill_cuda)，G>0 格 |
| 决策敏感性 21 条件 | partC_conditions.csv | M0/M2 各自 Greedy+Exact，统一 M2 尺度 |
| T2-G2 排序 | formal_gpu1b round_equal_weight + pending_policy_matrix | 等权 TTFT 中位 vs ΣΦ(G,T) |

**最终判定：READY（NO_NEW_GPU_NEEDED）。**
