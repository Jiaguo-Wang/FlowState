# FlowState P0-2：Runtime 控制成本与实际收益归因最终审计报告

日期：2026-10-28。判定：**READY**（补实验裁决：**NO_NEW_GPU_NEEDED**）。面向 Evaluation VI-F。全程 CPU-only 只读：未启动 GPU、未修改任何冻结实验数据/Runtime/Optimizer/Controller/Solver、未重跑 T2-G2、未增加 Baseline、未关闭或绕过正确性检查；所有数字均有可追溯的原始日志/冻结产物依据，不同 Harness 的测量互不拼接。

产物目录 [paper_control_cost_final_audit/](paper_control_cost_final_audit/)：
[recompute.py](paper_control_cost_final_audit/recompute.py)（一键复算，<5s）、[audit_data.json](paper_control_cost_final_audit/audit_data.json)。10 项检查全 PASS。

## 1. 控制路径与计时边界表（Task A）

控制路径 Observation → Strict Epoch/Authentication → Planning → Physical Binding → Reconciliation → Native Resume 的实测计时边界，按证据来源分三列（T2-G2=60 正式臂均值，[analysis_summary.json](t2g1_three_policy_gpu/formal_gpu1b/analysis_summary.json) control_decomposition；E4=12 测量生命周期七阶段拆分，[e4_control_timing_audit](e4_control_timing_audit/final_report.md)；分类依据 [audit_classification.md](c_formal_perf_gpu/audit_classification.md) A 研究仪表 / B 原生恢复 / C 生产状态管理）：

| 阶段 | 函数 / RPC（计时起止） | T2-G2 实测(ms) | E4 实测(ms) | 类 | 嵌套与可识别性 |
|---|---|---:|---:|---|---|
| 状态观察 | 轻量版 `metadata_observation`（FA 前沿/候选/物理视图） | （在各段内嵌） | 43.7 | C | 独立分段，可识别 |
| 状态观察（重型） | `final_heavy_observation` | 2403.9 | — | **A** | 独立分段；T2-G2/C-K3 特有研究仪表 |
| Strict Epoch 认证 | `strict_epoch_authentication`（历史读取+来源线索+编码摘要+合同认证） | 合并于下行 | 103.4 | C | 独立分段；非单纯数学校验 |
| 收益矩阵+求解+Plan | `benefit_matrix` / `solver`(Greedy/Exact) / `plan_build_validate` | 合并于下行 | 0.02+0.13+8.6 | C | 独立分段；求解亚毫秒级 |
| 认证+计划合计 | `strict_epoch_authentication_and_plan` | **345.0**（FS） | 112.2 | C | T2-G2 分段更粗（含计划），与 E4 拆分口径不同，不互减 |
| Apply-time 新鲜度预检 | `apply_time_freshness_preflight` | 1226.9 | 37.3 | **A**（T2-G2 重型预检含研究验证；E4 轻量版为保留的安全检查） | 两 Harness 语义不同，如实并列 |
| FA 内容审计 | `fa_content_audit_before` / `fa_content_postcheck_and_resume_baseline` | 934.9 / 1750.8 | — | **A** | 独立分段 |
| 逻辑→物理映射 | `cpu_physical_mapping` | 0.59 | — | C | 映射本身亚毫秒；census/溯源在未分段余额 |
| Reconciliation（含 Native 驱逐） | `strict_reconciliation_with_postchecks`（13 项真实驱逐+合同后检，**Native 驱逐嵌套在内不另计**） | **868.0**（FS） | 138.8 | C | 独立分段；operations[*] 可测 Adapter 范围但已含于本段，不重复累加 |
| RPC 未分段余额 | RPC 墙钟 − 上述分段和（端点差额） | ≈3969 | 482.4 | 混合（A 为主） | **构成部分 UNKNOWN**：census/apply 基线快照/溯源、序列化、响应编码传输、客户端解析与证据持久化（E4 源码级确认存在、独立计时缺失） |
| Apply RPC 墙钟合计 | 客户端调用前→`_call` 返回（含证据持久化） | 11499.2（FS 中位） | 814.5 | 混合 | 父区间；**位于请求 TTFT 窗口之外** |
| Native Matching / COW | 请求窗口内 `match_prefix` / `pool.copy_from` | 窗口内 | — | B | 实测 host 0.4–0.5 + CUDA 0.5–0.6ms；g1 实测模式复核 pregap 全部 ~1.1ms（A 类移除后） |
| Request TTFT | 请求提交→首 Token（T3−T2） | 36–290（12 格） | 74–102 | — | 独立窗口；A 类仪表移出后为生产路径 TTFT |
| Control-inclusive Epoch | 控制发起→流结束（T4−T0） | —（每臂单 Pending） | 7469.3 | 混合 | 含 Request Wall 6411.8——**约 7.5s 的主体是请求执行而非状态控制**；control_path Harness 同口径 2078.8（FS，旧） |

要点：①E4 七阶段由同一 `timed()` 顺序包围、互不嵌套；②Native Eviction 已含在 Reconciliation 内，全表无重复累加；③`between_request_validation_and_dispatch`（control_path Harness 实测 1051.9→1929.1ms）为完整 Epoch 中最大的非请求区间之一，**构成部分 UNKNOWN**；④T2-G2 与 E4 的同名分段口径不同（重型 vs 轻量 harness），本表并列展示、不互减不拼接。

## 2. T2-G2 三策略控制成本与请求收益复核（Task B）

| 指标 | FS | MC | KV |
|---|---:|---:|---:|
| Apply RPC 墙钟中位 | 11499.2 | 11060.7 | 11123.5 |
| **生产核心**（认证+计划+协调） | **1213.0** | 1185.3 | 1292.9 |
| 生产核心占墙钟比例 | 10.5% | 10.7% | 11.6% |
| A 类研究仪表四段合计 | 6316.3 | 6081.8 | 6148.7 |

- **同轮配对控制差**（含仪表，n=20）：FS−MC 中位 **+575.7** [−1091.7, +1348.1]、FS−KV **+370.9** [−1101.8, +1529.6]——范围跨零、轮间方差大，主要由 A 类仪表差异（如 fa_postcheck FS 1750.8 vs MC 1508.0，源于 FS 深恢复状态下受保护 FA 内容更多）驱动；
- **生产核心差**：FS−MC **+27.7ms**、FS−KV **−79.9ms**（FS 对 KV 核心更低）；
- **同轮四 Pending 等权 TTFT 配对差复核**：FS−MC **−114.127ms**、FS−KV **−105.728ms**（5/5 轮改善），与独立审计冻结值逐位一致；
- **核心问题回答：FlowState 没有为请求 TTFT 付出明显更高的必要控制成本**——生产核心与两个基线差异 <80ms/apply（±7% 内），且对 KV 为净负；RPC 墙钟上 FS 略高 ~0.4s 属 A 类研究仪表（深恢复状态哈希内容更多），按函数级分类不属生产成本。若按含仪表墙钟口径，则 **NOT_IDENTIFIABLE** 墙钟余额（~3.9s/臂）中生产/仪表的精确切分不可得，本报告不假定其可无成本移除。

## 3. 既有两轮控制优化的独立核验（Task C-1/2）

| 实验 | Harness | 前→后 | 降幅 | 语义等价证据 | 独立核验 |
|---|---|---:|---:|---|---|
| RQ6-E 批量观察/协调 | RQ6 冻结 24 快照×3=72 正式 run（前后同基线、同 `perf_counter_ns` 四阶段口径，before==RQ6 基线逐位） | 8398.847→**629.827** | **−92.50%**（13.3×） | 同步 RPC 51.5→2.0/epoch；13/13 门禁：selected-set 等价 72/72、H/E/G 288/288、FA preservation、无驱逐污染/意外重驻留/级联/未来泄漏 | 复算一致；reduction 逐位 −92.501% |
| Transport 优化 | control_path resume-epoch Harness（前后各自完整采集，**时段不同、不拼接**） | 656.771→**131.904** | **−79.9%** | `identical_plan=true`、同选集 216/216、同请求 H/E/G 864/864 | 复算一致 |

两轮优化为**不同 Harness，不得串联为一次连续加速**（RQ6 口径含每 epoch 重型验证后的统一验证另计；control_path 口径为轻量 resume-epoch 控制路径）。优化前后状态控制与正确性验证语义均保持（上表等价证据），未发现遗漏的同步调用（RPC 计数 51.5→2.0 为单调下降，重型验证移至批量协调后一次执行）。

## 4. Control-inclusive Resume-Epoch 结果核对（Task C-4/5）

- **FlowState 平均 Request Execution 最低**：旧 92.5ms / 新 91.2ms（LRU 163.4/163.2、Marconi 128.2/126.9）——前后两轮采集均成立；
- **完整 Epoch 配对差（基线−FS，bootstrap 95% CI）**：LRU 均值 +526.6，CI **[−1057.4, +2214.9]**（15胜9负）；Marconi 均值 −549.6，CI **[−1784.9, +699.4]**（8胜16负）——**两组 CI 均含零**，`statistically_stable_net_benefit_vs_both=false`；
- spec 给出的三条结论（Request Execution 最低、两组 CI 含零、未证明统计稳定完整控制净收益）**全部核对成立**。原因定位：完整 Epoch 中 `between_request_validation_and_dispatch`（1.0–1.9s，构成部分 UNKNOWN）与轮间方差（epoch CV 中位 0.26–0.36）远大于策略间控制差（优化后 FS 131.9 vs LRU 149.5）；**不得因请求性能较好而将此结果解释为端到端加速**——本报告遵守该边界。

## 5. 可识别与不可识别的控制成本（交付 5）

**可识别**：生产核心三段（认证+计划、协调及后验、映射）在 T2-G2/C-K3/E4 均有独立分段计时；A 类四段（重型观察、FA 审计×2、重型预检）有独立分段；Native Matching/COW 有请求窗口内实测（B 类，毫秒级）；求解器亚毫秒（E4 0.13ms；RQ6 CPU 口径 allocation 均值 0.75ms、P95 1.61ms）。

**NOT_IDENTIFIABLE（不估算、不扣除）**：①T2-G2/C-K3 RPC 墙钟未分段余额 ~3.9–4.4s/臂（census/apply 基线快照/溯源/序列化/传输/客户端持久化混合，无独立起止计时）；②E4 RPC 未分类余额 482.4ms（源码确认构成项存在但各自耗时 UNKNOWN，不能全部标为可删除成本）；③`between_request_validation_and_dispatch` 的内部构成；④服务端控制入口/出口、证据写盘 fsync、首 Decode 内部细分（[e45_unified_timing](e45_unified_timing/final_report.md) 判定非五指标必需，保留原样）。

## 6. 收益摊销分析（Task D，条件性）

同一 T2-G2 协议内（每生命周期恰一次 Apply，四 Pending 每轮各一请求，请求构成恒定）：

| 口径 | ΔC_control（FS 相对基线） | ΔL_saved（每请求等权） | N_break-even=⌈ΔC/ΔL⌉ |
|---|---:|---:|---:|
| T2-G2 生产核心 vs MC | +27.7ms/apply | 114.1ms | **1** |
| T2-G2 生产核心 vs KV | −79.9ms/apply | 105.7ms | **0（无额外成本，净）** |
| T2-G2 含研究仪表 RPC（配对中位） vs MC | +575.7ms | 114.1ms | 6 |
| T2-G2 含研究仪表 RPC（配对中位） vs KV | +370.9ms | 105.7ms | 4 |
| C/K3 生产核心 vs SA / vs WFC | +1.0 / −7.0ms/apply | 34.1 / 46.9ms | **1 / 0（净）** |

FS 自身生产核心摊销（C/K3）：1029ms/apply ÷ 163.2ms（C_B 型深恢复请求单次回收）≈ **7 次深恢复请求**/apply。**假设与边界**：①均为同协议配对测量，无跨 Harness 拼接；②含仪表口径的 ΔC 含 A 类研究仪表（非生产），仅作上限参考；③N_break-even 是条件性回本次数，**不是实测在线吞吐或端到端净收益**；④生产核心口径中 T2-G2 与 C-K3 的分段粗细不同（认证+计划 345 vs 160ms），各自协议内自洽、互不比较。

## 7. VI-F 可直接使用的论文结论（交付 7）

1. **控制开销的结构**：生产核心（认证+计划+协调+映射）实测 1.0–1.2s/apply，其中求解 <1ms、纯状态构造+分配亚毫秒（RQ6：CPU 全路径均值 0.31ms）；Apply RPC 墙钟的 ~89% 为实验专用研究仪表（A 类：重型观察/FA 内容审计/重型预检），按函数级分类不属生产成本；
2. **请求收益与控制成本解耦**：FS 获得 −114.1/−105.7ms 等权 TTFT（5/5 轮）所付出的**生产**控制差 <80ms/apply（对 KV 为净负；C/K3 上 +1/−7ms）——生产核心口径下 1 次请求即回本；B 类原生恢复路径（Matching/COW/绑定）实测毫秒级，不侵蚀请求收益；
3. **控制可优化性**：两轮独立优化（批量观察/协调 −92.5%、transport −79.9%）均以逐位语义等价（选集/H-E-G/FA/无污染门禁）为前提，同步 RPC 由 51.5 降至 2.0/epoch；
4. **完整控制周期**：Request Execution FS 最低（91–93ms），但完整 Epoch 配对差 95% CI 含零（LRU [−1057,+2215]、Marconi [−1785,+699]）——**作为明确局限如实报告**，归因于验证/分发间隔方差（CV 0.26–0.36）大于策略间控制差。

## 8. 不得宣称的系统性能结论（交付 8）

- 不得把 Apply RPC 墙钟（~11s）表述为生产控制开销（生产核心仅 ~10.5%）；
- 不得把 RQ6-E 的 −92.5% 与 transport 的 −79.9% 串联为一次连续加速（不同 Harness）；
- 不得把 Control-inclusive Epoch 的 CI 含零结果解释为端到端加速，也不得反向宣称"控制开销抵消了请求收益"（生产核心口径为净正）；
- 不得把 N_break-even（1/4/6/7 次）写成实际在线吞吐或端到端净收益；
- 不得宣称已识别 RPC 未分段余额（~3.9s）的构成或其可移除性；
- 不得用 E4 轻量口径（认证 103ms）否定 T2-G2 重型口径（345ms）——两者分段语义不同，各自协议内自洽。

## 9. 最小 GPU 补实验裁决（Task E）：**NO_NEW_GPU_NEEDED**

现有证据已足够支撑论文当前的 Runtime 正确性（13/13+语义等价门禁、Cold 逐 Token 一致）、控制优化（两轮 −92.5%/−79.9% 且语义保持）与请求级收益（−114/−106ms，5/5 轮）结论；完整控制净收益按 spec 的 NO_NEW_GPU_NEEDED 分支**作为明确局限报告**。不推荐投稿前补 GPU 的定量理由：完整 Epoch 差的策略间信号量级（优化后控制差 ~20–150ms）远小于 epoch 噪声（CV 中位 0.26–0.36 × epoch ~2–3s ⇒ 轮间 σ 数百 ms），要以 95% CI 下界>0 分离需将轮数扩至 ~10× 以上且仍受 `between_request_validation_and_dispatch`（构成 UNKNOWN）支配——属"改变正式验证协议/重构 Runtime"才能根治的问题（BLOCKED 分支描述的情形），不适合投稿前开展；且该结论对论文主张（请求级收益+控制可优化性）非必要。

**最终判定：READY（NO_NEW_GPU_NEEDED）。**
