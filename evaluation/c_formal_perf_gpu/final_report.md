# FlowState ICDE — 正式 GPU 性能配对实验 报告（c_formal_perf_gpu）

协议 v1.2（v1.0 → v1.1 形状预热修正 → v1.2 截止时间顺延，两项变更的完整记录见 protocol.json）。
工作负载：C Shared/Branched K=3，Qwen3.5-9B，SGLang v0.5.17（16 处 runtime_sources 修改镜像），H100 PCIe TP=1。
矩阵：3 策略（FlowState-Greedy=FS / StandaloneTopK=SA / WFC）× 4 Pending（C_A/C_B/C_C/C_D）× 5 独立生命周期 = 60 正式测量，
冻结交错顺序（rep-major，策略轮转），另有门禁 2 臂（C_B 最小对：FS/SA）。

## 0. 结论

62/62 生命周期 PASS（门禁 2 + 主矩阵 60），全部硬断言成立（每臂 audit==2、请求窗口 Triton 新增编译==0、
Cold 逐 Token 一致、E/G/Prefill 区间==冻结合同、形状预热准入==计划E）。

1. **FlowState 在深恢复 Pending 上有真实、稳健的请求级加速**：C_B 上 FS−SA 配对 TTFT 中位
   **−163.2ms [−167.3, −154.9]**（37.2 vs 202.5ms，5.4×），5 轮全部为负且对机器双峰模式稳健
   （两臂同处慢模式的 rep1/rep5 中差值仍为 −158/−155ms）；
2. **同前沿（C_C/C_D，E/G 相同）请求级平局**：全量中位 +6.6/+21.1ms 的差由共享机器双峰模式驱动
   （见 §5.1），快模式口径 FS−SA 仅 +2.8/+2.3ms（噪声带内）——恢复语义等价 ⇒ 请求成本等价，得到验证；
3. **FS 驱逐格（C_A，E=0）无优势**：FS≈SA（+1.0ms [−0.6,+4.3]），WFC 因保留 2048 反而最优（−71.7ms）；
4. **必要控制开销不侵蚀收益**：请求窗口内 B 类（COW/绑定/匹配）实测 ~2–6ms（COW CUDA 0.14–0.39ms +
   host 0.13–0.25ms + native match 0.5–1.1ms + pregap 1–3ms）；每 Epoch 一次的控制中，生产核心近似
   （认证+计划 0.16s、协调+合同后检 0.87s）≈ **1.03s/apply**，研究审计增强仪表另计（§3.3）；
5. **控制含入的净收益为正**：单次 apply 生产核心 1.03s 摊到 Epoch 内全部请求；C_B 型请求每次回收 163ms，
   ≈7 次深恢复请求即覆盖一次 apply 的生产核心成本（研究仪表按分类不属于生产成本，不参与此算式）；
6. **WFC vs FS**：WFC 是中间路径——在 FS/SA 都驱逐的 C_A 上最优（+71.7ms），在 B/C/D 上因固定浅前沿
   （2048/3072）落后 FS 84–105ms。FS 的自适应前沿在"值 Deep 恢复"的到达流上占优，其代价是值模型判错
   （或预算让位）时该到达完全冷启动（C_A 即实例）。

## 1. 研究审计与真实服务开销的分离（§1 交付）

函数级分类与实测依据：`audit_classification.md`（A 研究仪表 / B 原生恢复 / C 状态管理，含测量模式处置与安全分离论证）。
关键实测（c_pair 24 生命周期归因 + 本轮 g1/g2 门禁复核）：

- 请求计时窗口内的 A 类开销被完整移出/关闭：admission 全量快照哈希（FS≈1137ms/次、SA≈626ms/次，窗口内 2 次）、
  match 前第二次全量哈希、COW 前后 3×~75ms payload 哈希、TRACE 落盘（每次 70–90ms）；
- B 类（原生绑定/COW/必要同步）与 C 类（Observe/Select/Reconcile/真实驱逐）全部保留；
- **门禁实测复核（g2 对）**：移除 A 类后 FS queue 2490→2.9ms、pregap 474→0.9ms；SA queue 1448→2.8ms；
- **残差归因修正**：v1.0 估计的 "原生绑定 ~170ms" 实为 A 类仪表间接开销——B 类完整保留时 FS pregap 仅 0.9ms
  （COW host 0.14ms + CUDA 0.28ms + native match 0.67ms）；
- 安全分离论证成立（纯读代码、apply 合同逐字节不变、请求后全量观察 + Cold 逐 Token 一致 + 硬断言把守），
  因此下文请求口径可命名为**生产路径 TTFT**；
- apply 段（口径C）审计增强仪表完整保留（重型观察/FA 内容审计/apply 基线 ≈ 每 臂 5.6–8.5s，计入控制墙钟并单列分段），
  从不与请求口径混算。

## 2. 形状预热与编译零污染（§2 交付）

每生命周期：16 预热(history_pre) → flush_cache（状态同一性复验）→ 16 预热二次重放(history，census 溯源)
→ 唯一 b5d4_apply（完整冻结合同，census 16/16/0）→ 形状预热 B → 正式流式请求。

形状预热 B v1.1（attempt1 失败教训的修正，证据 g1_CB_SA）：输入 = pt[:计划E] + 确定性随机 T+32−E Token，
在 E 处首 Token 分叉——与正式请求同形（位置 E 准入、extend L=T+32−E、decode 起 kv=T+32），
但只在随机分支物化状态，正式请求原生匹配深度不超计划 E。

门禁与主矩阵断言（全 62 臂硬断言）：
- 预热行：executable_prefix == 计划E 且 physical_fa_hit == E；
- 正式行：executable_prefix == 计划E、physical_fa_hit == T、Prefill 区间 == E/G 合同；
- 请求窗口（post_shape_warm→post_request）Triton 新增编译 == 0；
- 窗口内审计项 == 0（TRACE['audit'] 全程恰 2 次：apply 基线 + 请求后）。

## 3. 三口径结果（§4 交付；主矩阵数字见 CSV）

数据文件：`lifecycle_results.csv`（62 行逐次全指标）、`pending_policy_matrix.csv`（12 格聚合）、
`pair_analysis.csv`（同轮同 Pending 两两差）、`finalize_results.json`（配对统计）、`timing_scopes.md`（口径表）。

### 3.1 口径A：GPU 计算事件（CUDA 事件窗）

主矩阵 5 轮配对（首批 extend CUDA，ms；中位 [min,max]；`pair_analysis.csv` d_cuda_prefill_ms）：

| Pending | FS | SA | WFC | FS−SA 中位 [range] |
|---|---|---|---|---|
| C_A (E=0/0/2048) | 151.4 | 150.1 | 79.7 | 0.5 [−1.2,+3.8] |
| C_B (E=5120/0/2048) | **30.7** | 192.8 | 115.7 | **−161.4 [−165.2,−149.7]** |
| C_C (E=6144/6144/3072) | 34.8* | 30.5* | 116.2 | +5.2* [−9.8,+21.1] |
| C_D (E=7168/7168/3072) | 50.1* | 33.9* | 154.6 | +17.7* [−13.6,+21.8] |

\* 同前沿格的计算差由机器双峰模式驱动（§5.1）：快模式下 FS/SA 的 32-Token extend 同为 29–35ms。
C_B：FS 30.2–32.5（快模式）vs SA 192–198，**6.2×，5 轮全部复现，与前两轮认证值（31/196ms）一致——Q1 成立**。
decode CUDA 与 TTFT 无关（首 Token 后），双峰见 §5.1；COW 设备拷贝 0.14–0.39ms/次（每臂≤1 次）。

### 3.2 口径B：请求级（生产路径 TTFT 及六段分解）

配对 TTFT 中位 [min,max]（ms；`pair_analysis.csv` d_ttft_ms；单值见 `pending_policy_matrix.csv`）：

| Pending | FS | SA | WFC | FS−SA | FS−WFC | SA−WFC |
|---|---|---|---|---|---|---|
| C_A | 159.3 | 157.8 | **87.3** | +1.0 [−0.6,+4.3] | +71.7 [+64.0,+79.2] | +70.2 [+63.0,+74.9] |
| C_B | **37.2** | 202.5 | 123.5 | **−163.2 [−167.3,−154.9]** | −84.4 [−92.0,−69.8] | +78.8 [+75.3,+87.8] |
| C_C | 42.7* | 37.6* | 126.1 | +6.6 [−10.6,+23.1] | −84.3 [−92.8,−61.0] | −86.7 [−93.5,−75.8] |
| C_D | 59.2* | 41.2* | 163.2 | +21.1 [−14.6,+24.8] | −105.4 [−126.9,−96.7] | −117.8 [−129.7,−111.6] |

\* 含双峰模式混入（C_C：FS 中招 2/5 轮 vs SA 1/5；C_D：FS 3/5 vs SA 2/5）；快模式口径 C_C 39.7 vs 36.9、
C_D 38.4 vs 36.1（差 +2.8/+2.3ms，噪声带内）——**同前沿平局（Q3）成立**。

六段分解（全 60 臂中位）：queue 3.3、pregap 1.3、execwin≈compute、postgap 0.2、delivery 2.8——
TTFT 由计算主导，A 类审计移除后 FS queue 2490→3–5ms、SA 1448→3ms（**Q2 成立**）；
闭合校验全部 <5ms。request_wall（含 8 Token decode）FS C_B ~170ms vs SA ~316ms（门禁臂参照）。

### 3.3 口径C：状态管理（控制墙钟与分段）

每生命周期一次 b5d4_apply 的 RPC 墙钟与分段（中位，ms；`lifecycle_results.csv` ctrl_* 列）：

| 分段 | FS | SA | 类别 |
|---|---|---|---|
| 认证+计划 strict_epoch_authentication_and_plan | ~165 | ~162 | **C 生产核心** |
| 协调+合同后检 strict_reconciliation_with_postchecks（含 13 项真实驱逐） | ~870 | ~875 | **C 生产核心** |
| 重型观察 final_heavy_observation | 2408–3116 | 同量级 | A 研究仪表 |
| FA 内容审计 before + postcheck/resume_baseline | 2727–2998 | 同量级 | A 研究仪表 |
| apply 时新鲜度预检 | 1231–1579 | 同量级 | A 研究仪表 |
| census/物理映射/apply 基线快照等（未分段剩余） | ~3000–4000 | 同量级 | A 研究仪表 |
| **RPC 墙钟合计** | **~11500** | **~12800** | 混合 |

生产核心合计 ≈ **1.03s/apply**（一次性，摊到 Epoch 内全部请求）；审计增强仪表 ≈ 10.5–11.8s/apply
（按 §1 分类不属生产成本；如强行计入，C_B 型请求 ~74 次回本——仅作上限参考）。
每臂实际驱逐/绑定/COW 次数与来源槽位在 run/summary.json·reconciliation.json 留档（census 16/16/0 全臂成立）。

## 4. 分析问题回答（§5）

**Q1 C_B 计算优势是否复现？** 复现。FS 首批 extend CUDA 30.7ms（30.2–32.5，快模式）vs SA 192.8ms
（192–198），6.2×；与前两轮认证值 31/196ms 一致；配对差 5 轮全部 −149.7~−165.2ms。

**Q2 移除研究审计后 TTFT 是否真实改善？** 是。A 类审计（窗口内 2 次全量哈希 FS≈2.28s/SA≈1.24s +
COW 哈希 + 落盘）移出窗口后：FS queue 2490→3–5ms、SA 1448→~3ms；C_B FS TTFT ~3.3s→37–52ms。
改善来自仪表分离而非运行语义变化（apply 合同逐字节不变，B/C 类全保留，§1 论证）。

**Q3 C_C 同 E/G 平局？** 平局成立（含噪声带）。全量配对中位 +6.6ms [−10.6,+23.1]，但其离散由共享
机器双峰模式（§5.1）驱动：快模式口径 FS 39.7 vs SA 36.9（+2.8ms，C_C）、38.4 vs 36.1（+2.3ms，C_D）。
差值符号随各轮模式分配翻转（C_C 逐轮 Δ：−0.7, +23.1, +21.1, +6.6, −10.6），不构成策略可归因差异。

**Q4 四 Pending 总体请求级收益？** 依 Pending 结构三态：深恢复（C_B）FS 大胜（−163ms）；
同深保留（C_C/C_D）平局（±噪声）；被驱逐（C_A）FS=SA（+1.0ms）且二者皆逊于 WFC（−71.7/−70.2）。
FS 无任何格逊于 SA（最差 +21.1 为模式混入，快模式 +2.3）；WFC 无格最差但仅 C_A 最优。

**Q5 计入必要控制后净收益？** 为正。窗口内必要控制（B 类 COW/绑定/匹配 + 排队投递）实测 ~2–6ms，
相对 C_B 的 163ms 收益可忽略；一次性成本按 §1 分类取生产核心（认证+计划 0.16s + 协调+后检 0.87s ≈
1.03s/apply），≈7 次 C_B 型请求即覆盖；研究仪表（10.5–11.8s/apply）不属生产成本、单列报告。

**Q6 WFC vs FS？** 固定前沿 vs 自适应前沿：C_A（FS E=0）WFC 快 71.7ms；C_B/C_C/C_D（FS E=5120/6144/7168
vs WFC 2048/3072）FS 快 84–105ms。FS 的收益来自"值深的到达保留得深"；其风险是值模型把预算让给其他
分支时，本到达完全冷启动（C_A 实例：三策略中唯一 E=0×2 的格）。到达流中深恢复占比决定两者优劣。

**统计口径声明**：所有数值为 5 次独立生命周期的中位数与极差（配对差为同轮同 Pending 配对）；
按规约不将 5 次重复解释为充分统计显著性证明。全部负面结果（C_A 无优势、同前沿非零中位、
快/慢模式混入）均如实报告。

## 5. 有效性与停止条件记录

### 5.1 机器双峰模式（对结果的已知混入，已量化）

60 臂中 17 臂处于"慢模式"（decode CUDA >150ms：中位 203 vs 快模式 121ms；32-Token extend prefill
29–35→42–52ms，+15–20ms 传入 TTFT）。模式横跨全部 Pending 与全部三策略（FS 9、SA 6、WFC 2 臂），
且时间相关（m1B 的 FS+SA 相邻两臂同中、m5A 两臂同中、m5B 两臂同中）——判定为宿主级机器状态
（推测为时钟/功耗状态），非策略、非代码路径差异。每臂启动前 GPU 空闲断言（显存<1000MiB 且无其他
计算进程）全部成立，故非显式争用。处置：主结果按协议报全量配对中位数（不剔除），快模式口径作为
明确标注的次要分析；关键结论（C_B）对两种模式均稳健。

- 门禁 attempt1（g1_CB_SA）BLOCKED：v1.0 形状预热在 E<T 格执行 pt[E:] 物化被驱逐状态，正式准入被抬高，
  runner 硬断言在计时前拦截；v1.1 修复后 attempt2（g2 对）全 PASS。证据：g1_CB_FS/、g1_CB_SA/、
  launch_gate_failure.json、protocol.gate_attempt1；
- 截止时间 v1.2 顺延：gate2 PASS 与主矩阵启动之间会话暂停约 10.5h，原 5h 窗口过期，零臂执行；
  顺延非任何实验结果所致，冻结输入/策略/预算/门禁证据未改动（protocol.deadline_note）；
- 每臂：GPU 空闲断言（<1000MiB 且无其他计算进程）、专用 --rm 容器、3× 清理校验（<100MiB）、
  实现锁逐臂复核、镜像与 runtime 源码哈希逐生命周期复核、产物 write-once；
- 全部负面结果与未测项如实报告；5 次重复不解释为充分统计显著性（报告均值/中位/离散）。

## 6. 产物清单

- 正式数据：lifecycle_results.csv / pending_policy_matrix.csv / pair_analysis.csv / finalize_results.json / timing_scopes.md
- 合同与协议：protocol.json（v1.2）、gate_passed.json、implementation_lock_gate2/main.json
- 审计分类：audit_classification.md、analysis_audit_attribution.py + audit_attribution.csv（c_pair 24 生命周期依据）
- 证据链：g1_CB_FS/（attempt1 PASS）、g1_CB_SA/（attempt1 BLOCKED 证据）、g2_CB_FS/、g2_CB_SA/、
  m{1-5}{A-D}_{FS,SA,WFC}/（各含 run/summary.json、native trace、Triton 快照、清理校验）
- 运行日志：runtime_gate.log、runtime_gate2.log、runtime_main.log、*_cleanup.json、*_gpu_before.json
