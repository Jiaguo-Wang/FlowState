# Gate B — Section II-C 共享分支 Exact vs Greedy GPU 实验（B5-C4）只读审计

日期：2026-10-31 ｜ 性质：只读审计（零 GPU、零代码修改、零历史数据修改；未重跑五轮、未启动 GPU）
裁决前置：本报告全部结论基于原始实验文件（逐文件 SHA-256 见 `b1_file_manifest.json` / `source_manifest.json`），**不使用 v6.5 正文或历史汇总报告作为实验证据**。

## B1. 原始文件定位（全部找到，含哈希）

v6.5 所述特征（四 Pending A/B/C/D、七逻辑 Checkpoint、公共位置 2432、A/B 分支 3712、C/D 分支 3520、K=1/K=2、J=228.05/48.23/130.80 ms、TTFT 配对改善 34.71/36.71/17.74 ms）与下列原始文件逐一对应：

| 层 | 原始位置（SHA-256 摘要见清单） | 内容 |
|---|---|---|
| 工作负载（GPU 实测构造） | `evaluation/multishare_runtime_gate_b5c3/`（final_report、strict_epoch_snapshot.json、sharing_support_matrix.json、candidate_provenance.json、physical_state_census.json 3.8MB） | 8 条历史重放后单同步窗口 Census 出 7 候选 + 4 Pending；位置/准入边/来源全部原始留档 |
| 模型化分配（离线） | `evaluation/multishare_allocation_gate_b5c4/`（results.json、exact_verification.json）；`evaluation/greedy_swap_structure_e3_step1/condition_results.csv`；`evaluation/paper_allocation_structure_strict20.csv`（B5-C4 四行 K=1–4） | K=1–4 Exact/Greedy/1-Swap/WFC 各法 J 与选集冻结行 |
| **GPU 性能对照（II-C 核心实验）** | `evaluation/b5c4_formal_gpu_g2/`：`formal/raw_results.csv`（40 臂 ×27 列）、`formal/{arm_status,checks}.json`、40 个 `r{1-5}_{exact\|greedy}_{a-d}/` 逐臂证据（reconciliation/summary/post_request/stream_chunks 等）、`preflight/{g2_reference,prepared_inputs,cold_reference_*}.json`、`pkg_g2/{control,greedy_adapter}.py`、`implementation_lock.json` | 2 策略 ×4 Pending ×5 轮；每臂独立 Docker/GPU 生命周期 |
| 前置配对实验链（如实披露，不混入 G2 统计） | `evaluation/exact_greedy_paired_gpu_gate_b5d4/`、`exact_greedy_paired_gpu_fix_b5d4a/`（root_cause_and_patch.md）、`exact_greedy_paired_gpu_gate_b5d4b/`（paired_differences.json 等） | 早期门禁→修复→b5d4b 单 Pending(C)×5 对：Prefill 改善但 TTFT 反向（D4b 发现，G2 协议据此重设计） |

说明：`evaluation/c_formal_perf_gpu/` 属 **C/K3 工作负载族**（n=16 超门禁、冻结选集协议），与本 Gate 的 B5-C4 共享分支案例是两个不同实验——II-C 若引用 C/K3 数字须按该族自身证据链表述，不得与 B5-C4 混同。

## B2. 工作负载合法性（结论：合法，且已自带"受控构造"标注义务）

1. **轨迹来源**：4 个 Session（A–D）源自真实 OpenHands 轨迹数据集 `nebius-swe-rebench-openhands`（prepared_inputs 逐 Pending 记录 workflow_id `chatcmpl-71420ffc…`、parquet 行号、token_prefix_sha256、完整 token_ids 4435–4862）；8 条历史请求 = 4 Session × 2 轮真实 Assistant/工具记录重放（completion=1 token，原生生成仅作缓存副作用，不加入 Pending）。
2. **分支构造（受控）**：公共前缀 Checkpoint N20@2432（A/B/C/D 全支持）；分支状态 N18@3712（A/B）、N21@3456 与 N27@3520（C/D）；专属状态 N22@4160（仅 A）、N24@4096（仅 B）、N25@3904（仅 C）。**位置全部来自真实 Cache 节点 Census**——b5c3 明示"没有插入候选、移动分叉边界、恢复/驱逐状态或更换Session"，且不预设 CPU LCP 值。
3. **七 Checkpoint 可追溯**：candidate_provenance.json 逐节点记录 first_observed_after / observed_after_requests / 兼容前缀；physical_state_census.json 保留完整 Conv/Temporal 摘要、组件切片 sha、Slot/Node、FA 路径、引用与同窗口不变性。
4. **完整 Recurrent State**：strict_epoch_snapshot 逐候选 state_identity（mamba present/slots、target_full_present、path_full_all_present）+ payload 组件 sha + slot 映射。
5. **物理身份**：逐生命周期由实时 Census 重建（G2 §5 明示 N18/N20/N27 为逻辑标签，不跨容器复用 Node/Slot ID）。
6. **Runtime 兼容**：全部候选/Pending 同 configuration_digest `936ca739…`（同模型 Qwen3.5-9B/同模板/同引擎配置）。
7. **无未来信息**：历史重放期间 Pending 未派发（snapshot `pending_dispatched=false`、prepared_inputs `pending_dispatch_allowed=false`）；T=min(len, FA) 由只读树遍历取得；准入矩阵（28 边：EPOCH_ADMISSIBLE=13/REJECTED=15/UNKNOWN=0）单同步窗口冻结。
8. **数据性质（必须标注）**：workload_type=**Trace-derived Controlled Composition**——受控共享分支构造，非自然同时 Ready、非原生 Fork。论文表述不得扩大为自然并发 Workflow 统计证据。b5c3 同时披露两点边界（须随文）：D 无专属候选（八条历史≠八个独占候选；D2 原生记录 3456 vs Census 3520 只报原始 Match 记录）；b5c3 阶段 EPOCH_ADMISSIBLE≠逐边 RESUME_VALIDATED（后续 b5d4/G2 以 Cold 逐 Token 参照补齐恢复输出验证）。

## B3. 模型化分配独立复算（`scripts/recompute_b5c4_allocation.py` → `b3_allocation_recompute.json`）

从 b5c3 冻结快照原始输入（7 候选位置、13 准入边、4 个 T）+ 冻结 Φ 公式（硬编码独立录入，与生产 `flowstate.recovery_model` 交叉核对**最大绝对差 0.0 ms**）从零重建 J(S)=Σ_p Φ(T_p−E_p(S), T_p)：

| K | 子集数(含空集) | Exact argmin（J, ms） | Marginal Greedy 重放（J, ms） | 与 strict20 冻结行一致 |
|---|---|---|---|---|
| 1 | 8 | {N20}，228.05481335524232 | {N20}，228.05481335524232 | ✅ 选集+J 双一致 |
| 2 | **29** | **{N18,N27}，48.228728759089606** | **{N20,N18}，130.8030838298517** | ✅（全精度逐位） |
| 3 | 64 | {N22,N24,N27}，16.411345 | {N20,N18,N27}，48.228729 | ✅ |
| 4 | 99 | {N22,N24,N25,N27}，1.753119 | {N20,N18,N22,N27}，31.093896 | ✅ |

- K=2 argmin **唯一**（次优 {N18,N21} 53.1056，差距 4.877 ms）——v6.5 的 228.05 / 48.23 / 130.80 全部复现。
- 两 K=2 计划逐 Pending E/G（与 G2 40 臂实测 E 逐臂一致）：Exact A/B E=3712(G=448/384)、C/D E=3520(G=384/46)；Greedy A/B E=3712(同)、C/D E=2432(G=1472/1134)。A/B 构成**前沿相同近零差对照**，结构差异全部集中于 C/D（E 差 1088 Token）。
- **离线最优计划 ≠ 正式生产 Exact 路径**：上表为离线重建；G2 的 Exact 选集是**运行内在 Scheduler 进程里由生产 `plan_strict()` 在线求得**（40 臂逐臂 `strict_decision.solver="Exact"`、objective 全精度=48.228728759089606），e3 冻结行仅作事后 Oracle（`summary.oracle_e3.match=True`）。两者数值一致但证据身份不同，论文引用时应表述为"在线求解，与离线参照逐位一致"。

## B4. 五轮 GPU 配对数据独立复算（`scripts/recompute_g2_paired.py` → `b4_g2_paired_recompute.json`）

从 `formal/raw_results.csv` 40 行独立重建（**先按轮次及 Pending 配对、再取中位**；不读取既有 paired_analysis.csv / paper 复算产物）：

### 逐 Pending（TTFT，ms；Δ=Greedy−Exact，正值=Exact 更快）

| Pending | Δ 逐轮 | **A 配对差中位（主口径）** | B 两组中位之差（仅对照） | Prefill Token Exact/Greedy |
|---|---|---|---|---|
| A | −1.483, −0.094, +0.802, +0.868, −1.517 | **−0.09** | −0.09 | 1150 / 1150 |
| B | +0.129, **−15.197, −12.800**, +1.074, +6.163（3/5 正；两个负值为 Exact-B 慢轮，见负向观测） | **+0.13** | +0.70 | 1148 / 1148 |
| C | +35.587, +31.877, +34.707, +40.342, +34.104（5/5 正） | **+34.71** | +34.70 | **915 / 2003** |
| D | +36.712, +29.535, +22.896, +41.262, +37.779（5/5 正） | **+36.71** | +36.70 | **966 / 2054** |

### 轮级等权（每轮 4 Pending 单对差等权均值 → 五轮中位）

ΔTTFT 逐轮 +17.74 / +11.53 / +11.40 / +20.89 / +19.13（**五轮全正**），中位 **+17.74**；ΔCUDA Prefill 中位 **+17.66**（五轮全正）。

**v6.5 数字核对（全部通过）**：C 34.71 ✅、D 36.71 ✅、等权 17.74 ✅、C Prefill 2003/915 ✅、D 2054/966 ✅、J 48.23/130.80（全精度逐臂==冻结参照 ✅）。注意 34.71/36.71 属**配对差中位口径**（两组中位相减为 34.70/36.70，差 0.01）——论文必须沿用配对口径并注明。

**配对合法性与中断检查**：40/40 臂有效（audit_entries==2、请求窗口零新增 Triton 编译、cold_output_match 全真、Prefill=len(pt)−E 合同逐臂成立、E∈{3712,3520,2432} 结构合法）；arm_status 零失败、零重试臂 → **无中断/续跑混入，20 对配对完整**。历史链如实披露：b5d4 门禁 → b5d4a 修复（root_cause_and_patch.md）→ b5d4b 单 Pending D4b（Prefill 改善 −30~−34ms 但 TTFT 反向 +30.6~+44.6ms，证据未做窗口分解，**不作归因、不混入 G2 统计**）；G2 协议（请求窗口零审计+形状预热 B+编译快照差分+奇偶轮交替）即为针对该发现的重新设计，两者关系已在 G2 paper 证据 §5 分列表述。

## B5. 执行路径语义（在线求解 vs 冻结选集等 5 问）

1. **选集来源=在线求解，非冻结**：Exact 臂 control 在 Scheduler 进程内 `optimizer.plan_strict(epoch)`，不读冻结选集（control.py 注释明示"在线,自动Exact"）；逐臂 reconciliation 内 `strict_decision`={solver:"Exact", gate_allowed:true, objective:48.228728759089606}。Greedy 臂经 `pkg_g2/greedy_adapter.py` **逐行复用生产 `_select_candidates` 与同一 frontiers 评分闭包**（仅跳过 Exact 枚举；其 `gate_allowed=true` 记录反而证明该臂域满足门禁、未篡改候选/门禁/预算）。冻结 e3 行只在求解完成后作事后 Oracle。
2. **真实物化为不同 Recurrent 保留**：Exact 臂保留 {N18,N27}、Greedy 臂保留 {N20,N18}（逐臂 plan.selected 实测不同）；各驱逐补集 5 个候选且全部 `returned=true`（mamba_evictable 7→2）；COW 命中源 Slot 正确。
3. **真执行模型恢复**：恢复不是模型化差值——逐臂实际计算 Prefill Token=len(pt)−E（915/966 vs 2003/2054），CUDA Prefill 事件配对差中位 C/D +33.77/+36.44 ms，TTFT 配对差 +34.71/+36.71 ms。
4. **Attention-side State 不变**：两臂 `H_physical_fa_hit` 同值（如 A 臂均 4160）；post_request FA map 保留；FA 命中与 E 分属不同状态侧（H=FA 树命中、E=Recurrent 前沿），改善全部来自 Recurrent 侧。
5. **无状态别名/额外恢复**：逐臂输出与独立 Cold 参照逐 Token 一致（40/40）；请求窗口零新增内核编译；audit_entries==2（基线+请求后）。**结论：GPU 支持的是"最优分配减少实际恢复计算"（实测 Prefill 计算量与 TTFT 双降），而非仅模型化成本差异**——但该结论限于 B5-C4/K2/4 Pending/5 轮配对口径。

## B6. 裁决

**FULL_GPU_EVIDENCE_VERIFIED**（限 B5-C4/K2/4 Pending/5 轮配对口径内的全部数字；附强制标注义务）

II-C 可保留的数字（全部经本审计独立复算逐位/口径核实）：
- 模型层：K=1 J=228.05（Exact==Greedy）；K=2 Exact 48.23 {N18,N27} vs Greedy 130.80 {N20,N18}（全精度 48.228728759089606 / 130.8030838298517）；（如引用）K=3 16.41、K=4 1.75 同样复现。
- 结构层：C/D Prefill Token 915/966 vs 2003/2054（E 差 1088）；A/B 前沿相同对照。
- 实测层：C/D ΔTTFT 配对差中位 **+34.71/+36.71 ms**（CUDA Prefill +33.77/+36.44）；四 Pending 等权五轮全正、中位 **+17.74 ms**（ΔPrefill +17.66）；A/B **中位**在 ±0.2 ms 内（B 含两个 Exact 慢轮逐轮观测 −15.2/−12.8 ms，见负向观测义务）。
- 执行层："两策略同一真实 Scheduler 进程内在线求解（生产 plan_strict / 生产 Greedy 路径复用）、相同协调路径落地、独立 GPU 生命周期 40 臂"。

必须随文的限定（缺一即须改写）：
1. 数据性质标注 **Trace-derived Controlled Composition**（受控共享分支构造），不得写成自然并发 Workflow 证据；
2. 中位数为**配对差口径**并注明 5 轮样本、不宣称显著性；
3. Exact 在线求解证据限于门禁规模（7 候选/K=2），不得外推为生产规模普遍 Exact；
4. 负向观测如实保留（Exact 5/40 更慢单臂，最大 15.2 ms；A/B 差值噪声级）；
5. D4b 历史反向发现如提及，必须按"协议差异对照"表述且不归因。

须修改/删除的表述类型（若 v6.5 存在）：把 34.71/36.71 写成"两组中位数之差"；把改善归因于"Exact 求解器更快"（实为选集结构差异——前沿前移）；将 B5-C4 结论与 C/K3/五轮矩阵数字混同引用；任何"显著/统计显著"字样。
