# C Shared/Branched K=3 GPU 配对计时有效性验证 — 最终报告

- **结论：计时污染确认并逐阶段定位；去除编译项后，FlowState 的 GPU 恢复计算反而显著更快（31ms vs 196ms，6.3×）；Pilot 中"FS 无 Prefill/TTFT 改善"的表象由三类非恢复开销掩盖，全部已定量分解**
- **READY：24/24 生命周期 PASS（phase1 20 + phase2 4），正确性合同零失败**
- 固定条件与 pilot 完全一致（冻结 C K=3、16 预热、Qwen3.5-9B、SGLang v0.5.17、H100 PCIe TP=1、K=3）；control.py/bridge.py 与 pilot 逐字节相同；仅运行器增加计时仪表与两项加严断言（census 16/16/0、Cold 逐 Token 一致）

## 1. 实验设计

| 实验 | 对照 | 目的 |
|---|---|---|
| C_B | FS(E=5120,G=0) vs SA(E=0,G=5120)，H=5120 相同 | 恢复 Token 减少（32 vs 5152）是否产生可分辨延迟收益 |
| C_C | FS 与 SA 均 E=6144,G=0 | 相同恢复前沿 → 环境与测量波动的零假设带宽 |

- 每条件 5 次独立 Engine 生命周期（冻结交错顺序：奇数对 FS 先、偶数对 SA 先，对内两条件时间相邻），共 20 次
- phase2（污染确认后追加）：4 次 strict_warm 形状预热生命周期（throwaway → flush → 二次预热 → 唯一 apply → 正式请求），Triton 编译缓存跨 flush 保留
- 不把 5 次重复解释为充分统计显著性；CC 对内差作为经验零假设带宽

## 2. phase1：20 次生命周期逐段分解（TTFT ≈ queue + execwin + delivery，闭合断言 5ms 内）

| 条件（5 次中位） | E | queue | pregap | execwin | compute | delivery | TTFT | 请求窗新增编译 |
|---|---|---|---|---|---|---|---|---|
| CB_FS | 5120 | 2575 | 477 | 952 | 390 | 96 | **3583** | 9 |
| CB_SA | 0 | 1448 | 88 | 361 | 190 | 93 | **1909** | 8 |
| CC_FS | 6144 | 2544 | 479 | 957 | 393 | 96 | 3605 | 9 |
| CC_SA | 6144 | 2563 | 476 | 963 | 396 | 99 | 3623 | 9 |

配对差（FS−SA）：**CB 对内 ΔTTFT = +1543 / +1634 / +1727 / +1544 / +1785 / +1668ms（中位 1668）**；CC 对内 ΔTTFT = −49 ~ +77ms（零假设带宽 ≈ ±80ms）。同 Pending 同输入长度下，FS 比 SA 慢 1.67s，为 CC 带宽的 ~20×——单一发生命周期的 Pilot 倒置复现且稳定。

## 3. 污染定位：1.67s 全部归属三类非恢复开销

**（1）queue 段 +1127ms：E>0 准入的研究审计宿主开销（pre-batch）**
C_B 两策略 FA 命中与 radix 匹配输入完全相同，唯一差别是 FS 的 mamba 可执行前缀 E=5120>0：准入须做 mamba 状态绑定 + COW 暂存 + 源/目标 payload 逐字节哈希审计。CC 两策略均 E=6144（都付此项）→ 对内 queue 差 ≈0，交叉验证归属。

**（2）pregap 段 +389ms：E>0 准入在 batch 开始后、首 kernel 前的宿主工作**
所有 E>0 臂 pregap 470–505ms，所有 E=0 臂 85–102ms；phase2 形状预热潮（编译已消除）pregap 仍 475ms——证明该段与编译无关，是准入绑定/COW 暂存的批内宿主时间。

**（3）compute 事件窗 +200ms：首形状 Triton 编译+autotune**
FS C_B 的 32-token extend 形状在 16 条预热（全长 round0 + 64-token round1）中从未出现。逐臂 Triton 缓存快照证明：请求窗口内新增 9 个编译，其中 FS 独有 1 个 = **`chunk_fwd_kernel_o`（32-token extend 形状）**；两臂共有的 8 个为 decode/采样路径内核（`fused_recurrent_gated_delta_rule_packed_decode_kernel`、`causal_conv1d_update_kernel`、`track_mamba_states_all_layers_kernel` 等），在 prefill 之后编译，不影响 TTFT。autotune 基准在 CUDA 事件窗内运行 → 32-token "GPU 时间" 390ms。
另：首 decode 步事件窗 1.29–1.63s（首 decode 形状编译；引擎 `disable_cuda_graph=true`，无 Graph Capture）；引擎启动 26.8s。

## 4. phase2 形状预热验证（4 次 strict_warm，编译项消除）

| 臂（n=2） | E | queue | pregap | execwin | compute | TTFT | 请求窗编译 |
|---|---|---|---|---|---|---|---|
| w_CB_FS | 5120 | 2556 | 478 | 597 | **30.8** | 3250 | 7（仅decode类） |
| w_CB_SA | 0 | 1431 | 98 | 382 | **196.5** | 1908 | 7（仅decode类） |

- throwaway 恰好消化 2 个 extend 形状内核（`chunk_fwd_kernel_o` + `_fused_sigmoid_mul_kernel`，与 phase1 FS 独有集完全一致）；warm 请求窗不再有 extend 编译
- compute 390→**31ms**（−359ms，autotune 窗膨胀消除）；ΔTTFT 1668→**1340ms**，残余 = queue Δ(1125) + pregap Δ(380)，即成分（1）（2）——两项均为研究仪表宿主开销，与恢复计算无关
- 附带证据：二次预热墙钟 5.2s vs 首次 31.2s（同 16 请求）——首轮预热 83% 时间是编译
- **去污染后的真实恢复计算差：FS 31ms vs SA 196ms（32-token vs 5152-token mamba 重放），FS 快 6.3×、约 165ms/请求**——方向与"减少恢复 Token"一致，此前被 ~1.7s 仪表+首调用开销完全掩盖

## 5. 对任务问题的回答

**为什么 Pilot 中 FS 明显减少恢复 Token 却没有 Prefill/TTFT 改善？**
三重掩盖：E>0 准入审计（queue+pregap 共 ~1.5s，研究仪表）、首形状编译+autotune（~0.59s，一次性）、decode 内核编译（不伤 TTFT 但污染"请求窗口"口径）。去除编译项后 FS 的 GPU 恢复计算恢复其真实优势（6.3×）；TTFT 口径下 FS 仍慢 1.34s，其中 ~1.5s 属仪表宿主开销（COW 暂存与 payload 哈希在生产控制路径中不存在）。

**C_C 稳定、C_B 是否 FS 真实更慢？**
仪器化口径下是（+1543~+1785ms，5/5 复现），且已逐 GPU 阶段列明来源；计算口径下否（31 vs 196ms，FS 快）。负面信号不是恢复机制本身，而是仪表与首调用效应。

**是否可提出正式配对实验矩阵？** 可以（本轮不执行）：全 4 Pending × 3 策略 × ≥5 配对生命周期；每生命周期含认证形状预热（throwaway+flush+二次预热，本轮已验证可行且合同零放宽）；报告分层为 GPU 计算时间（去编译）、仪器化 TTFT、生产控制开销（认证+协调 <1.1s）三条线；E>0 准入审计移出计时路径或单独归因列。

## 6. 执行记录与修正

- phase2 attempt1：w1_CB_FS 在溯源检查 BLOCKED——node_id 为进程生命周期计数器、跨 flush 不稳定（同内容节点 18→51）；修正仅 runner 接线（二次预热写 history_XX.json 供溯源，首轮改 history_pre_XX.json 留证），control/bridge 零改动；证据保留 `*.attempt1`，详见 PHASE2_FIX.md
- phase1 首版门文件配对过滤 bug（恒真条件）已当场发现重建，逐臂原始数据未受影响

## 7. 产物索引

`lifecycle_results.csv`（24 行逐次全指标）、`pair_analysis.csv`（CB/CC 配对差）、`timing_scopes.md`（口径表）、`finalize_results.json`、`contamination_confirmed.json`（门+证据）、各臂 `<name>/run/`（summary/reconciliation/history*/stream_chunks/output/rpc_*/triton 快照）、宿主侧 runtime log/cleanup/gpu_before/process、`PHASE2_FIX.md`；协议 `protocol.json` v1.0
