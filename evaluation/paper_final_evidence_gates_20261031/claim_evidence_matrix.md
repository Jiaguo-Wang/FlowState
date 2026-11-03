# 投稿前双门禁 · Claim–Evidence 矩阵（claim_evidence_matrix）

日期：2026-10-31 ｜ 依据：gate_a_solver_integration.md、gate_b_shared_branch_gpu.md 及其引用的原始文件（SHA-256 见 source_manifest.json）
裁决：Gate A = **HYBRID_INTEGRATED_GPU_VERIFIED**（附范围限定）；Gate B = **FULL_GPU_EVIDENCE_VERIFIED**（限 B5-C4/K2/4 Pending/5 轮配对口径，附强制标注义务）
"是否完整"=现有证据是否完整支持该 Claim 原样成立；"须收窄"=证据支持但需加限定/改口径的表述。

## A 组 — Section IV-C / V / Evaluation（Hybrid Solver 机制与验证）

| # | v6.5 原始 Claim（要旨） | 支持证据 | 是否完整 | 可保留表述 | 须收窄表述 | 需补测 |
|---|---|---|---|---|---|---|
| A1 | plan_strict 混合求解：门禁内 Exact 枚举、超域 Marginal Greedy | 源码 optimizer.py:22-165（SHA 7dfa9aec…）；集成门禁 §2-3 | ✅ | 机制描述全部成立（阈值 12/4/4/794、容差 1e-9/最少项/字典序、决策记录） | — | 否 |
| A2 | 门禁=规模四条件，错误拒绝非静默回退 | `_exact_admission` + 错误路径注入 3/3 + 122 测试 | ✅ | "fail-stop，无静默回退；validate 拒绝/内部异常均上抛" | — | 否 |
| A3 | Strict20 上 Exact 20/20 最优（若引用） | paper_hybrid_solver_integration_gate（正式入口，双独立 Oracle）；本包复算（evaluation_data_package） | ✅ | "20/20 选集+J 与冻结 Exact 行及独立 Oracle 全等；Greedy 域内 5/20" | — | 否 |
| A4 | C/K3 超门禁自动回退且行为不变 | 集成门禁 §5（J 154.34895561599998 逐位、选集==冻结）；五轮臂 kernel_entry 实测 plan_strict | ✅ | "n=16>12 → Marginal Greedy，在线选集==冻结 FS 选集" | — | 否 |
| A5 | "GPU 上执行了在线 Exact 求解" | Spec8 A 21/21（B5-C4 K2，strict_decision.solver=Exact）+ G1 A 4/4（w1/K2，用本轮 Plan 协调） | ✅（存在性） | "在线 Exact 路径已在 GPU 真实执行并物化（门禁规模 n=7、K≤2，两个工作负载）" | 不得写成"生产规模/正式性能实验中运行 Exact"；须注 Epoch 构建为冻结历史重放+实时观察准入 | 可选（见 A8） |
| A6 | 五轮矩阵性能数字（FS−MC −470.2 / FS−KV −429.7 ms）出自 FlowState | 五轮 15 臂 + 独立审计（evaluation_data_package/five_round_audit.md） | ✅ | "FS 臂经生产 plan_strict **在线 Greedy 分支**选点（==冻结计划），MC/KV 经各自真实内核" | **不得**归因于 Exact 求解；不得与 B5-C4 Exact 数字合并成同一结论 | 否 |
| A7 | 求解器时耗（Exact 0.22ms 中位等，若引用） | paper_solver_tradeoff（本包独立复算==登记） | ✅ | 注明 CPU 微基准、单进程、非生产在线耗时 | — | 否 |
| A8 | （若存在）"混合入口已在所有正式实验生效" | 五轮臂身份锁==当前 optimizer.py | 部分 | "五轮/C/K3 正式实验运行的就是混合入口（Greedy 分支）" | 正式性能实验中 Exact 分支未在线运行（n=16 结构性超门禁）——只能写"入口已部署且门禁规模经 GPU 验证" | **可选补强**：门内更大规模（n≤12,K=3/4）在线 Exact GPU 单轮正确性验证（非必须，Strict20 已 CPU 覆盖） |

## B 组 — Section II-C（共享分支 Exact vs Greedy GPU 实验）

| # | v6.5 原始 Claim（要旨） | 支持证据 | 是否完整 | 可保留表述 | 须收窄表述 | 需补测 |
|---|---|---|---|---|---|---|
| B1 | 工作负载：4 Pending、7 Checkpoint、公共 2432、A/B 分支 3712、C/D 分支 3520 | b5c3 census/snapshot/provenance（原始文件，本审计哈希锁定） | ✅ | 位置全部来自真实 Cache 节点；逐节点来源/组件/Slot 可追溯 | 必须标注 **Trace-derived Controlled Composition**（受控构造，非自然并发/原生 Fork）；随文披露 D 无专属候选 | 否 |
| B2 | K=1 最优 J=228.05；K=2 Exact 48.23 vs Greedy 130.80 | 本审计从原始输入+Φ 独立重算（`b3_allocation_recompute.json`）：228.05481335524232 / 48.228728759089606 / 130.8030838298517 | ✅ | 全精度数字；K=2 argmin 唯一（次优差 4.88ms）；29 子集含空集核对 | — | 否 |
| B3 | TTFT 配对改善 C 34.71 / D 36.71 / 等权 17.74 ms | 本审计从 raw_results.csv 独立重算（`b4_g2_paired_recompute.json`）：34.707/36.712/17.736 | ✅ | 配对差中位口径（同轮同 Pending 先配对再取中位） | **不得**写成两组中位数之差（该口径为 34.70/36.70）；不得用"显著"；5 轮样本须注明 | 否 |
| B4 | Prefill Token C 2003→915、D 2054→966 | 40 臂逐臂 prefill=len(pt)−E 合同（本审计逐臂复验） | ✅ | "实际恢复计算量减少 1088 Token" | — | 否 |
| B5 | 两策略"在线求解" | 逐臂 strict_decision（Exact=生产 plan_strict 在线；Greedy=greedy_adapter 逐行复用生产 `_select_candidates`，gate_allowed=true 证未篡改门禁） | ✅ | "同一真实 Scheduler 进程内在线求解，e3 冻结行仅事后 Oracle" | 不得暗示 Greedy 臂是生产入口自动选择（是实验设计的对照臂，复用生产内部路径） | 否 |
| B6 | "最优分配减少实际恢复计算"（而非仅模型化差异） | CUDA Prefill 配对差 +33.77/+36.44；TTFT +34.71/+36.71；物化保留集不同、驱逐 returned、FA 侧 H 同值 | ✅ | "实测恢复计算与 TTFT 双降；A/B 前沿相同对照近零差" | 限于本工作负载/K=2/5 轮；不得外推为一般生产负载收益 | 否 |
| B7 | （若引用）K=3 16.41 / K=4 1.75 | 本审计重算一致（strict20 行） | ✅ | 模型化数字 | 标注 CPU 模型化（无 GPU 执行） | 否 |
| B8 | （若提及历史）D4b 反向发现 | b5d4b paired_differences.json + G2 paper §5 | ✅ | "协议差异对照：D4b 请求窗口含审计/无形状预热，TTFT 反向未归因" | 不得混入 G2 统计；不得删除不提（如实披露义务） | 否 |

## 交叉禁则（两 Gate 共同）

1. B5-C4（II-C）与 C/K3/五轮矩阵是**不同工作负载与协议**，数字不得混同引用；`evaluation/c_formal_perf_gpu/` 属 C/K3 族。
2. 所有 GPU 门禁/闭环实验的计时字段仅为执行记录，正式性能口径只有：G2（B5-C4 Exact vs Greedy）、五轮矩阵（三策略）、t2g1/G2-B5C4 等正式实验各自协议。
3. "Exact 更优"的机理表述=**选集结构差异（恢复前沿前移）**，非"求解器运行更快"。
4. Control Cycle 与 TTFT 净额相减禁令维持（无定义口径）。

## 最小补测建议（仅 1 项可选，均不阻塞投稿）

- **可选（Gate A 范围补强）**：门内更大规模（n≤12、K=3/4）在线 Exact GPU 单轮正确性验证（复用 G1 骨架）。当前证据已由 Strict20 CPU 20/20 + GPU K=2 两负载覆盖，属"锦上添花"。
- 无必须补测项。两 Gate 全部关键数字已由本审计独立复算证实。
