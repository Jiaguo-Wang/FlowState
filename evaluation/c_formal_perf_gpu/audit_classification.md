# queue / pregap 阶段函数级审计分类（A 研究仪表 / B 原生恢复 / C 状态管理）

数据来源：`analysis_audit_attribution.py` 对 c_pair_timing_gpu 24 个生命周期 `native_trace.json`
的逐事件窗口归因（client/engine 同 CLOCK_MONOTONIC）；本表为测量模式（FS_MEASURE_MODE=1）的设计依据。

## 1. 请求计时窗口（发送 → 首 Token）内实际执行的函数

| 阶段 | 函数（代码位置） | 类 | 实测耗时（E>0 / E=0 臂） | 测量模式处置 |
|---|---|---|---|---|
| queue | HTTP/ZMQ 接收、调度排队、tokenizer 管道 | — | ~100–200ms（两臂共有） | 保留 |
| queue | `inputs()` Admission 钩子：`cache_state()` 全量状态快照（树遍历+semantic+selected payload 哈希+**FA 受保护内容全量哈希**） | **A** | FS≈1137ms / SA≈626ms（单次） | **移到请求后**（b5d4_state） |
| queue | `inputs()` `admission_gate()` 全量比对（nodes/semantic/selected vs 基线） | **A** | 含于上行的快照后比对（O(状态)） | **移到请求后**（等效更强检查：Cold 逐Token+全量状态） |
| queue | `match()` 钩子匹配前 `cache_state()` 第二次全量哈希 | **A** | FS≈1141ms / SA≈612ms（单次） | **移除**（窗口内不再调用） |
| queue | 原生 `match_prefix`（radix/FA 匹配、E 判定） | **B** | 0.6–1.3ms | 保留 |
| queue | `save()`：TRACE 全量 JSON 落盘（admission/match/逐批次各一次） | **A** | 每次 ~70–90ms（窗口内 2–3 次） | **移除**（TRACE 留内存，请求后持久化） |
| pregap | `copy()` 钩子 COW 前 `slot_payload()` 源槽位逐张量 D2H+sha256 | **A** | ~75ms | **移除** |
| pregap | COW 后 `slot_payload()` ×2（目标+源再哈希，发生在 `ns_after` 之后、首 kernel 之前） | **A** | ~150ms | **移除** |
| pregap | 原生 `pool.copy_from`（B 类 COW 设备拷贝）+ CUDA 事件 | **B** | host 0.4–0.5ms + CUDA 0.5–0.6ms | 保留（事件计时为口径A所需） |
| pregap | 引擎侧批次准备中的 mamba 绑定/槽位分配/状态准备（镜像 `mamba_component.prepare_for_caching_req` 等） | **B** | FS≈170ms / SA≈0（E>0 独有；=474−A类305） | 保留 |
| execwin | 前向计算（含首形状编译/autotune——已由形状预热消除） | 计算 | 见口径A | 保留 |

## 2. 窗口外（不动）

| 阶段 | 函数 | 类 | 实测 |
|---|---|---|---|
| apply（控制口径） | `final_heavy_observation`、`fa_content_audit_before/postcheck`、`cpu_physical_mapping`、census/溯源 | **A** | ~5.6–6.3s/臂（RPC 墙钟 10.6–12.0s 的主要部分） |
| apply | `strict_epoch_authentication_and_plan`（认证+计划）、`strict_reconciliation`（真实驱逐 13 项+合同后检） | **C** | 认证+计划 0.15–0.17s；协调 0.86–0.89s |
| apply 尾 | `admission_baseline = cache_state()` 全量基线 | **A** | FS≈1139 / SA≈776ms（计入控制墙钟） |
| 请求后 | `b5d4_state` 全量状态/FA 内容观察 | **A** | ~0.6–1.2s（正确性独立参考） |

## 3. 归因校验（phase1/2 实测 → 分解闭合；测量模式实测复核）

- CB_FS：queue 2490 ≈ 1137(admission 哈希) + 1141(match 哈希) + save×2 ~160 + 原生/HTTP ~50；
  pregap 474 ≈ 75(COW 前哈希) + 150(COW 后双哈希) + save ~80 + 残差 ~170
- CB_SA：queue 1448 ≈ 626 + 612 + save ~160 + 原生/HTTP ~50；pregap 88–96 全为原生批次准备
- FS 与 SA 哈希单次耗时差异（1139 vs 620）来源：`fa_protected_payload` 只哈希位置 < E 的受保护 FA 内容——
  SA 的 E=0 ⇒ 受保护集为空；FS 的 E=5120 ⇒ 几乎全量 FA 内容哈希。这解释了 phase1 中 FS 独有的 +1.5s
- 测量模式预期：两策略 queue 均降至 ~100–200ms；FS pregap 保留 ~170ms 原生绑定（B 类，不可移除）

### 3.1 测量模式实测复核（g1_CB_FS，v1.0 门禁臂，修正上节残差归因）

| 指标 | strict（c_pair） | 测量模式（g1_CB_FS 实测） |
|---|---|---|
| queue | 2490ms | **4.5ms** |
| pregap | 474ms | **1.1ms**（含 COW：host 0.22ms + CUDA 0.35ms；native match 1.1ms） |
| TTFT | ~3.3s | **59.8ms**（compute 50.6 + execwin 52.0） |
| 请求窗口 Triton 新增编译 | — | 0（形状预热窗口新增 9 项=目标 extend/decode 形状） |

**残差归因修正**：v1.0 估计中 "原生绑定 ~170ms（B 类）" 是间接残差推断；测量模式在**完整保留**
Native 绑定/COW/同步（B 类）的条件下实测 pregap 仅 ~1.1ms——该 170ms 残差实际属于 A 类仪表的
间接开销（TRACE 记录构造/内存增长/哈希同步副作用），原生恢复路径本身开销为毫秒级以下。
此修正对净收益核算是关键输入：B 类必要开销实测极小，FS 请求级优势不被必要控制操作侵蚀。

## 4. 安全分离论证（命名"生产路径 TTFT"的依据）

1. 被移出窗口的函数全部为**纯读**（哈希/快照/比对/落盘），不写任何引擎状态——代码级可证；
2. apply 段合同逐字节不变（control.py 与 pilot 相同），每生命周期仍做全量 census/认证/协调/FA 审计；
3. 请求后 `b5d4_state` 全量观察 + Cold 逐 Token 一致（语义等价最强检查）+ E/G/绑定/驱逐合同逐条断言；
4. 门禁与主矩阵断言：窗口内审计项 = 0（TRACE['audit'] 恰为 apply 基线+请求后各一次）、请求窗口
   Triton 新增编译 = 0、形状预热 B 原生准入 = 计划 E / FA 前沿 = T。
