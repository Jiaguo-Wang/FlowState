# G2 论文证据整理：B5-C4/K2 严格多共享结构 Exact vs Marginal Greedy GPU 对照

**判定：PASS**（复算 10/10 检查通过，见 `paper/recompute_report.json`；本文件只读整理，未修改任何原始结果）

**口径声明（全篇适用）**：
- 「配对差中位数」＝同轮同 Pending 的单对差 Δ_r,p=(Greedy−Exact) 在五轮上的中位数（请求级主口径）；
- 「两组中位数之差」＝各自五轮中位数相减（辅助口径；`formal/paired_analysis.csv` 的 `delta_*_median` 列属此口径，已与复算逐一核对一致）；
- 「轮级等权」＝每轮四 Pending 等权均值（等于该轮 4 个单对差的均值）→ 取五轮中位数（集合级口径，实验协议规定）；
- 20 样本池化中位数之差（本例 +17.48 ms）为**非协议口径**（Pending 结构权重失真），仅作警示对照，任何表格与正文不得引用；
- 五轮样本，全文不使用「显著」「显著性」等表述。

---

## 1. 实验问题与对照设计

同一 B5-C4 严格兼容关系与 K=2 预算下（7 候选 / 4 Pending / 8 条冻结历史），比较两种选集的在线求解与真实恢复执行：

- **FlowState-Exact**：生产 `GlobalOptimizer.plan_strict` 在规模门禁内自动选择 Exact 枚举，选集 {N18, N27}（两个深层分支状态）；
- **Marginal Greedy**：经 `pkg_g2/greedy_adapter` 逐行复用 `plan_strict` 超域分支的生产 Greedy 内部路径（同一 frontiers 评分闭包与 `_select_candidates`），选集 {N20, N18}（公共祖先 N20 + N18）。控制路径断言 B5-C4 域满足 Exact 门禁（7×4×K2），证明 Greedy 臂不是靠篡改候选数/门禁/预算获得。

两臂走完全相同的 `validate_plan` → Bind/Preflight → `reconcile_strict` 协调路径与同一测量协议。研究问题：把公共祖先状态替换为两个深层分支状态，能否稳定减少实际 Prefill 开销与请求 TTFT。

执行安排（`formal/schedule.json` 预固定，launch 强制顺序）：2 策略 × 4 Pending × 5 轮 = 40 臂，**每臂独立 Docker/GPU 生命周期**；奇数轮 Exact 先、偶数轮 Greedy 先，策略块内 Pending 顺序 A→B→C→D。同轮同 Pending 构成配对单位（20 对）。

## 2. 原始数据位置与复算方法

| 数据 | 位置 |
|---|---|
| 逐臂原始证据 | `formal/r{1-5}_{exact\|greedy}_{a-d}/`（summary/reconciliation/stream_chunks/post_request/shape_warm_b 等） |
| 40 臂汇总行 | `formal/raw_results.csv`（30 列：选集/J/E/H/G_model/prefill_tokens/三口径计时/分解段/有效性字段） |
| 既有统计产物 | `formal/paired_analysis.csv`、`formal/checks.json`、`formal/arm_status.json` |
| 冻结参照 | `preflight/g2_reference.json`（e3 全精度 Exact/Greedy 行）、`preflight/cold_reference_*.json` |
| 独立复算 | `paper/recompute.py` → `paper/recompute_report.json`（20 对配对差、两种中位口径、轮级等权、有效性自 summary 重验、结构层合同、负向观测清单） |

复算方法：从 `raw_results.csv` 独立重建全部统计量，并逐臂回读 `summary.json` 复核有效性（status=PASS、audit==2、请求窗口零新增 Triton 编译、Cold 逐 Token 一致、计时字段与 CSV 一致）。与既有产物的交叉核对全部一致（含 `paired_analysis.csv` 的 delta 列确属「两组中位数之差」口径这一事实本身）。

## 3. 正式结果表

### 表 1：逻辑分配层（冻结兼容关系下的在线求解结果；40 臂全部与 e3 参照全精度一致）

| 指标 | Exact | Marginal Greedy |
|---|---|---|
| 选集（checkpoint） | {N18, N27} | {N20, N18} |
| 模型化恢复成本 J (ms) | 48.228729 | 130.803084 |

| Pending | T（目标） | Exact：E / G_model / Prefill Token | Greedy：E / G_model / Prefill Token |
|---|---|---|---|
| A（len=4862, H=4160） | 4160 | 3712 / 448 / 1150 | 3712 / 448 / 1150 |
| B（len=4860, H=4096） | 4096 | 3712 / 384 / 1148 | 3712 / 384 / 1148 |
| C（len=4435, H=3904） | 3904 | **3520 / 384 / 915** | **2432 / 1472 / 2003** |
| D（len=4486, H=3566） | 3566 | **3520 / 46 / 966** | **2432 / 1134 / 2054** |

E=原生可执行前缀（恢复前沿），G_model=T−E（模型化重算缺口），Prefill Token=len(pt)−E（实测合同，40/40 臂成立）。A/B 两策略选集对这两个 Pending 的前沿相同（3712），构成**近零差对照**；结构差异全部集中在 C/D（E 相差 1088 Token）。

### 表 2：请求级配对结果（每 Pending 五轮；单位 ms；Δ=Greedy−Exact，正值=Exact 更快）

| Pending | Exact TTFT 中位 | Greedy TTFT 中位 | **ΔTTFT 配对差中位数** | ΔTTFT 两组中位数之差 | Exact Prefill 中位 | Greedy Prefill 中位 | **ΔPrefill 配对差中位数** |
|---|---|---|---|---|---|---|---|
| A | 53.30 | 53.21 | **−0.09** | −0.09 | 46.28 | 46.16 | **+0.12** |
| B | 53.15 | 53.84 | **+0.13** | +0.70 | 46.49 | 46.51 | **+0.08** |
| C | 47.48 | 82.18 | **+34.71** | +34.70 | 40.57 | 74.34 | **+33.77** |
| D | 48.98 | 85.67 | **+36.71** | +36.70 | 42.23 | 78.24 | **+36.44** |

（TTFT 为流式首 Token 墙钟，CUDA Prefill 为同流 CUDA 事件区间；两种中位口径数值并列以杜绝混用。）

### 表 3：轮级等权配对汇总（每轮四 Pending 等权均值；单位 ms；Δ=Greedy−Exact）

| 轮次 | ΔTTFT（等权均值） | ΔCUDA Prefill（等权均值） |
|---|---|---|
| 1 | +17.74 | +17.66 |
| 2 | +11.53 | +12.82 |
| 3 | +11.40 | +11.50 |
| 4 | +20.89 | +18.63 |
| 5 | +19.13 | +18.35 |
| **五轮中位数** | **+17.74** | **+17.66** |

五轮 ΔTTFT 与 ΔPrefill **全部为正**。r2/r3 偏低主要由 Exact-B 单臂慢轮次（见下）拖低该轮 Exact 均值所致，属保留的负向观测而非剔除项。

### 表 4：保留的 Exact 更慢单臂轮次（负向观测全清单，未删除未替换）

| 轮次/Pending | Exact TTFT (ms) | Greedy 快出 (ms) |
|---|---|---|
| r1 / A | 54.4 | 1.48 |
| r2 / A | 53.3 | 0.09 |
| r2 / B | 68.1 | 15.20 |
| r3 / B | 66.9 | 12.80 |
| r5 / A | 54.1 | 1.52 |

### 表 5：执行与统计口径

| 项 | 值 |
|---|---|
| 独立生命周期 | 40/40（每臂独立 Docker 容器 + Engine + Triton 缓存；镜像 sha256:1dfce903…；H100 PCIe 单卡） |
| 调度顺序 | schedule.json 预固定；奇数轮 Exact 先、偶数轮 Greedy 先；块内 A→D；launch 拒绝跳序/重跑 |
| Runtime 计时口径 | A 计算：CUDA prefill/decode 与 COW 事件；B 请求：TTFT 六段分解（闭合残差<5ms）+request_wall；C 控制：apply RPC 墙钟与分段（online_solve 含求解+物化） |
| 测量有效性（40/40） | 请求窗口零 A 类审计（audit==2：apply 基线+请求后各一次）、请求窗口零新增 Triton 编译（快照差分）、形状预热 B 于正式计时前（E 处分叉、L=len(pt)−E）、输出与独立 Cold 参照逐 Token 一致 |
| 逻辑身份 | N18/N20/N27 为逻辑标签，物理身份逐生命周期由实时 Census 重建 |

## 4. 主要性能发现

1. **实测层（主结论）**：C/D 两 Pending 上 Exact 的 TTFT 配对差中位数为 +34.71 / +36.71 ms（Prefill +33.77 / +36.44 ms），即 Exact 更快约 35 ms；A/B（前沿相同对照）差值在 ±0.2 ms 内。轮级等权五轮 ΔTTFT 全正（+11.4 ~ +20.9 ms），五轮中位 +17.74 ms（ΔPrefill +17.66 ms）。
2. **结构层对应**：C/D 的收益与恢复前沿差完全对应——E 3520 vs 2432（差 1088 Token），G_model 384/46 vs 1472/1134，实际 Prefill Token 915/966 vs 2003/2054（差恰为 1088）。
3. **模型层对应**：在线 J 48.229 vs 130.803 ms（Exact 低 82.57 ms，约 63%），40 臂与 e3 冻结参照全精度一致。
4. **三层关系**：模型化 J 降低与 Native Gap 缩小是结构事实；实测 TTFT/Prefill 改善与之方向一致且数量级可由 Token 差解释，但按协议表述为「实测配对差」，不由模型层推导、不宣称显著性。
5. **波动与负向观测**：Exact 存在 5 个更慢单臂轮次（A×3 幅度 0.09–1.52 ms 属噪声级；B×2 幅度 12.8/15.2 ms 属系统内波动），全部保留；A/B 的正负差值均未超出噪声量级，与「前沿相同则无收益」的结构预期一致。

## 5. D4b 与 G2 的证据对应

**D4b 记录在案的反向**（`exact_greedy_paired_gpu_gate_b5d4b/paired_differences.json`，单 Pending C、5 对、全部 valid）：`prefill_cuda_event_ms`（Exact−Greedy）= −33.83 / −31.73 / −30.48 / −32.88 / −32.32 ms（Prefill 全部改善，幅度与 G2 的 C 相当），但 `ttft_client_ms` = +42.05 / +34.77 / +30.60 / +44.03 / +44.58 ms（TTFT 全部反向变差）。

**由现有证据可确认的差异**：

| 维度 | D4b（证据） | G2（证据） |
|---|---|---|
| 请求窗口内容 | TTFT 1020.9–1089.5 ms，而其 Prefill 事件仅 44.6–79.0 ms（`descriptive_statistics.json`）；`measurement_validity.md` 明言完整路径含「内嵌审计」、「首次编译可能让事件间设备空闲」、外层事件含 COW 及其审计 | 测量模式：请求窗口（发送→首 Token）零 A 类审计零落盘（audit==2 断言）；TTFT 46.6–90.3 ms 与 Prefill 同量级；六段分解闭合 <5 ms |
| 形状预热 | runner（`exact_greedy_paired_gpu_fix_b5d4a/runner.py`，b5d4b launch.py:32 引用）无任何 SHAPE_WARM 步骤（grep 无匹配） | apply 后形状预热 B：pt[:E]+确定性随机尾在 E 分叉，max_new_tokens=2，先编译目标 extend/decode 形状；Triton 快照差分断言请求窗口零新增编译（40/40 成立） |
| 配对方式 | 单 Pending（C）×5 对 | 4 Pending × 5 轮 = 20 对；奇偶轮策略先后交替；配对差同时报告请求级与轮级等权两种口径 |
| 结果方向 | 5/5 对 Prefill 改善且 TTFT 反向 | 5/5 轮 ΔTTFT 与 ΔPrefill 同号为正；A/B 近零差对照成立 |

**未确认（不作归因）**：D4b TTFT 窗口内造成 +30.6~+44.6 ms 两臂差异的具体成分（审计/编译/主机提交空隙的各自占比）——D4b 证据未做窗口分解，其 `measurement_validity.md` 仅作定性警示；G2 的六段分解只证明 G2 自身窗口内各段占比（queue/execwin/delivery 见 `raw_results.csv`），不能反向归因 D4b。D4b 是否存在策略先后顺序均衡安排，其证据文件未记录，亦列为未确认。

## 6. 可用于 Evaluation 的中文结果段落

> **段落一（设置与协议）**：我们在同一严格兼容关系（B5-C4：7 个候选状态、4 个待恢复请求、K=2）下，对比 Exact 枚举与 Marginal Greedy 两种选集策略的真实恢复执行。两种策略在同一真实 Scheduler 进程内在线求解并经相同的验证与物理协调路径落地；每组对照（同轮同请求）在独立 GPU 生命周期中执行，共 5 轮 × 4 请求 × 2 策略 = 40 个生命周期，策略执行先后按轮交替以对冲时间漂移。计时采用三层口径：CUDA 事件（计算）、流式首 Token 墙钟及其六段分解（请求）、控制路径分段（协调）；正式请求窗口内零审计、零新增内核编译（经编译缓存快照差分验证），形状预热在计时前完成。
>
> **段落二（结果）**：Exact 选集 {N18, N27} 以两个深层分支状态替换 Greedy 选集 {N20, N18} 中的公共祖先状态后，模型化恢复成本从 130.80 ms 降至 48.23 ms。在两策略恢复前沿相同的请求 A/B 上，TTFT 与 Prefill 差值均在 ±0.2 ms 内（对照成立）；在恢复前沿相差 1088 个 Token 的请求 C/D 上，Exact 的实际 Prefill 计算量从 2003/2054 Token 降至 915/966 Token，TTFT 配对差中位数分别改善 34.7 ms 与 36.7 ms（CUDA Prefill 为 33.8 ms 与 36.4 ms）。四请求等权聚合下，五轮配对差全部为正，中位数为 17.7 ms（TTFT）与 17.7 ms（CUDA Prefill）。
>
> **段落三（边界）**：以上为 5 轮配对观测的中位数结果，我们不对其进行显著性检验宣称。Exact 策略存在个别更慢的单生命周期观测（5/40，最大幅度 15.2 ms），均如实保留并列入证据表；其收益来源于恢复前沿的结构性前移，随前沿差异消失（请求 A/B）而消失，与分配模型的预测一致。

---

**门禁自检**：所有数值可追溯至 `formal/raw_results.csv` 与逐臂 `summary.json`（复算报告 10/10）；40 臂与 20 对配对完整；逻辑分配（J/E/G_model）、Native 恢复计算（Prefill Token=len(pt)−E）与实测 TTFT 三层分列未混；原始结果零修改、负向观测零删除；判定 **PASS**。
