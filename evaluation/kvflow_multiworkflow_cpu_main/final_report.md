# Multi-Workflow CPU 五策略对比 — 最终报告

- Gate: **READY**（CPU 阶段完成；是否进入 GPU 另行决策）
- 实验执行：前任 agent，`execution_lock.json` 记录启动时刻 `1792582303870055077`（2026-10-21T11:31:43Z，即本地 19:31:43）
- 报告撰写：接续 agent（2026-10-21），仅做核验与离线分析，未重跑、未改任何冻结文件

---

## 1. 接管核验结论（§1）

| 核验项 | 结果 |
| --- | --- |
| 目录文件齐全（workloads.json 13.7MB、protocol.json、input_lock.json、execution_lock.json、run_cpu.py、test_cpu.py、cpu_tests.log、7 个输出文件） | ✅ 齐全 |
| `input_lock.json` 21 项 sha256 逐一重算 | ✅ **全部匹配，无 mismatch** |
| `execution_lock.json` 自洽（`inputs_sha256` = sha256(input_lock.json)；`implementation_sha256` = sha256(run_cpu.py)） | ✅ 一致 |
| 运行完整性 | ✅ 30 策略行、60 成对比较，结构与 K 覆盖完整 |
| 契约测试 | ✅ 17 项全 OK（前任 `cpu_tests.log`） |
| 反覆盖断言（`禁止覆盖或根据结果重跑`） | ✅ 生效，按规约**未重跑、未重新生成工作负载、未覆盖冻结文件** |

本轮新增文件仅两份：本报告与 `output_manifest.json`（15 个现有文件的 sha256 清单，供第三方复核）。所有 J 数值均为恢复成本模型输出（逻辑 ms），**不是 GPU 延迟**；全部非零项 `calibration_status=OUTSIDE_CALIBRATION_TARGET_RANGE`（所有 T=2048–8192 低于校准目标下限 32768，属外推）。全部候选 `state_status=CPU_PROPOSED`：CPU 前缀成立 ≠ EPOCH_ADMISSIBLE，不宣称 GPU 物化。

## 2. 实验设置与口径

| 结构 | 说明 | 候选数 | 当前 Pending | 未来声明 | K_min_frontier |
| --- | --- | --- | --- | --- | --- |
| A Sequential | 10 Agent 顺序，固定 4096 | 20 | 1（AGENT00） | 9 | 1 |
| B Independent Multi-Workflow | 3 工作流 × 2 阶段（2048/3072、4096/5120、6144/8192） | 12 | 3（各 W0_S0） | 3（各 S1） | 3 |
| C Shared/Branched | ROOT1024→AB/CD→四叶→JOIN(AND) | 16 | 4（A/B/C/D 叶） | 1（JOIN） | 4 |

预算 K∈{3,5}；五策略：LRU（预先声明合成访问事件）、Marconi Adaptation（token_position_L 代理评分，α=1.0）、KVFlow-inspired（声明 ASG/STE）、WFC、FlowState Marginal Greedy。决策语义按 §3：FlowState 只用决策时已声明当前 Pending；未来调用只在同一选集上**事后**评分（Frozen-Future J ≠ 真实在线完整 Workflow 延迟）。

## 3. 五策略选择与恢复成本对照表（30 行汇总）

`Jcur` = Current-Pending J（决策目标），`Jfut` = Frozen-Future J（事后口径），单位为模型逻辑 ms。完整逐 Pending 的 E/G 见 `cpu_comparison.csv` 的 `current_E_G`/`future_E_G` 列。

**K=3**

| 结构 | 策略 | 选中数 | Jcur | Jfut | 选中集合 |
| --- | --- | --- | --- | --- | --- |
| A | LRU | 3 | 154.35 | 1080.44 | AGENT09_P04096, AGENT09_P04032, AGENT08_P04096 |
| A | Marconi | 3 | 154.35 | 926.09 | AGENT09/08/07_P04096 |
| A | KVFlow | 3 | **0.00** | 1237.24 | AGENT00_P04032, AGENT00_P04096, AGENT01_P04032 |
| A | WFC | 1 | **0.00** | 1389.14 | AGENT00_P04096 |
| A | FlowState | 1 | **0.00** | 1389.14 | AGENT00_P04096 |
| B | LRU | 3 | 230.76 | 309.08 | W2_S1_P08192, W2_S1_P08128, W2_S0_P06144 |
| B | Marconi | 3 | 230.76 | 309.08 | （与 LRU 同集） |
| B | KVFlow | 3 | 236.25 | 623.85 | W0_S0_P01984, W0_S0_P02048, W1_S0_P04032 |
| B | WFC | 3 | **0.00** | 623.85 | W2_S0_P06144, W1_S0_P04096, W0_S0_P02048 |
| B | FlowState | 3 | **0.00** | 623.85 | （与 WFC 同集） |
| C | LRU | 3 | 582.03 | **0.00** | JOIN_P04096, JOIN_P04032, D_P07168 |
| C | Marconi | 3 | 348.23 | 154.35 | D_P07168, D_P07104, C_P06144 |
| C | KVFlow | 3 | 627.62 | 154.35 | AB_P01984, AB_P02048, A_P04032 |
| C | WFC | 3 | 471.87 | 116.23 | ROOT_P01024, CD_P03072, AB_P02048 |
| C | FlowState | 3 | **154.35** | 154.35 | D_P07168, C_P06144, B_P05120 |

**K=5**

| 结构 | 策略 | 选中数 | Jcur | Jfut | 选中集合 |
| --- | --- | --- | --- | --- | --- |
| A | LRU | 5 | 154.35 | 926.09 | AGENT09×2, AGENT08×2, AGENT07_P04096 |
| A | Marconi | 5 | 154.35 | **617.40** | AGENT09/08/07/06/05_P04096 |
| A | KVFlow | 5 | **0.00** | 1082.89 | AGENT00×2, AGENT01×2, AGENT02_P04032 |
| A | WFC | 1 | **0.00** | 1389.14 | AGENT00_P04096 |
| A | FlowState | 1 | **0.00** | 1389.14 | AGENT00_P04096 |
| B | LRU | 5 | 230.76 | **115.19** | W2_S1×2, W2_S0×2, W1_S1_P05120 |
| B | Marconi | 5 | 230.76 | **115.19** | （与 LRU 同集） |
| B | KVFlow | 5 | 2.49 | 623.85 | W0_S0×2, W1_S0×2, W2_S0_P06080 |
| B | WFC | 3 | **0.00** | 623.85 | W2_S0_P06144, W1_S0_P04096, W0_S0_P02048 |
| B | FlowState | 3 | **0.00** | 623.85 | （与 WFC 同集） |
| C | LRU | 5 | 348.23 | **0.00** | JOIN×2, D×2, C_P06144 |
| C | Marconi | 5 | 348.23 | **0.00** | D×2, C_P06144, JOIN_P04096, C_P06080 |
| C | KVFlow | 5 | 510.37 | 154.35 | AB×2, A×2, B_P05056 |
| C | WFC | 5 | 196.11 | 116.23 | ROOT, CD, AB, D_P07168, B_P05120 |
| C | FlowState | 4 | **0.00** | 154.35 | D/C/B 叶 + A_P04096（4/5 额度） |

预算使用形态：LRU/Marconi/KVFlow 恒满额选 K 个；FlowState 与 WFC 在边际≤0 时提前停机（A 中只选 1 个、B 中 3 个、C K=5 中 4 个）——A K=3 第一步选中 AGENT00_P04096 后，其余 18 个候选边际全为 0（`selection_traces.json`）。这不是同分规则差异，而是"零边际不驻留"的显式策略行为：省下槽位，但也放弃了对未来调用的静态覆盖（见 §4.2）。

## 4. 每类工作负载完整胜、平、负（60 对全覆盖）

60 对关系计数：**47 DIFFERENT_FRONTIER / 7 DIFFERENT_SET_SAME_FRONTIER / 6 SAME_SET_SAME_FRONTIER**。

### 4.1 Current-Pending J（决策口径）：FlowState 视角 **18 胜 / 6 平 / 0 负**

| 结构 | K | vs LRU | vs Marconi | vs KVFlow | vs WFC |
| --- | --- | --- | --- | --- | --- |
| A | 3 | 胜 −154.35 | 胜 −154.35 | 平（异集同前沿） | 平（同集） |
| A | 5 | 胜 −154.35 | 胜 −154.35 | 平（异集同前沿） | 平（同集） |
| B | 3 | 胜 −230.76 | 胜 −230.76 | 胜 −236.25 | 平（同集） |
| B | 5 | 胜 −230.76 | 胜 −230.76 | 胜 −2.49 | 平（同集） |
| C | 3 | 胜 −427.69 | 胜 −193.89 | 胜 −473.27 | 胜 −317.53 |
| C | 5 | 胜 −348.23 | 胜 −348.23 | 胜 −510.37 | 胜 −196.11 |

非 FlowState 的 36 对明细（result 为 policy_a 视角）：

| 结构 K | policy_a | policy_b | 关系 | Δ(a−b) | result_a |
| --- | --- | --- | --- | --- | --- |
| A 3/5 | LRU | Marconi | DIFFERENT_SET_SAME_FRONTIER | 0 | 平 |
| A 3/5 | LRU | KVFlow | DIFFERENT_FRONTIER | +154.35 | 负 |
| A 3/5 | LRU | WFC | DIFFERENT_FRONTIER | +154.35 | 负 |
| A 3/5 | Marconi | KVFlow | DIFFERENT_FRONTIER | +154.35 | 负 |
| A 3/5 | Marconi | WFC | DIFFERENT_FRONTIER | +154.35 | 负 |
| A 3/5 | KVFlow | WFC | DIFFERENT_SET_SAME_FRONTIER | 0 | 平 |
| B 3/5 | LRU | Marconi | SAME_SET（K5）/DIFFERENT_SET（K3）_SAME_FRONTIER | 0 | 平 |
| B 3/5 | LRU | WFC | DIFFERENT_FRONTIER | +230.76 | 负 |
| B 3/5 | Marconi | WFC | DIFFERENT_FRONTIER | +230.76 | 负 |
| B 3 | LRU/Marconi | KVFlow | DIFFERENT_FRONTIER | −5.49 | 胜 |
| B 5 | LRU/Marconi | KVFlow | DIFFERENT_FRONTIER | +228.27 | 负 |
| B 3/5 | KVFlow | WFC | DIFFERENT_FRONTIER | +236.25 / +2.49 | 负 |
| C 3 | LRU | Marconi | DIFFERENT_FRONTIER | +233.80 | 负 |
| C 3 | LRU | KVFlow | DIFFERENT_FRONTIER | −45.58 | 胜 |
| C 3 | LRU | WFC | DIFFERENT_FRONTIER | +110.16 | 负 |
| C 3 | Marconi | KVFlow | DIFFERENT_FRONTIER | −279.38 | 胜 |
| C 3 | Marconi | WFC | DIFFERENT_FRONTIER | −123.64 | 胜 |
| C 3 | KVFlow | WFC | DIFFERENT_FRONTIER | +155.74 | 负 |
| C 5 | LRU | Marconi | DIFFERENT_SET_SAME_FRONTIER | 0 | 平 |
| C 5 | LRU/Marconi | KVFlow | DIFFERENT_FRONTIER | −162.13 | 胜 |
| C 5 | LRU/Marconi | WFC | DIFFERENT_FRONTIER | +152.13 | 负 |
| C 5 | KVFlow | WFC | DIFFERENT_FRONTIER | +314.26 | 负 |

基线之间的排序并不稳定（如 C K=3：Marconi 胜 KVFlow/WFC，但 LRU 只胜 KVFlow；B 中 KVFlow K=3 输给 LRU/Marconi、K=5 反超），说明没有任何基线在当前口径上接近 FS 的稳定最优。

### 4.2 Frozen-Future J（事后口径，非决策目标）：FlowState 视角 **0 胜 / 8 平 / 16 负**

| 结构 | K | vs LRU | vs Marconi | vs KVFlow | vs WFC |（数值为对方 Jfut，FS 自己为表中 §3 所列）
| --- | --- | --- | --- | --- | --- |
| A | 3 | 负（1080.44） | 负（926.09） | 负（1237.24） | 平（1389.14） |
| A | 5 | 负（926.09） | 负（617.40） | 负（1082.89） | 平（1389.14） |
| B | 3 | 负（309.08） | 负（309.08） | 平（623.85） | 平（623.85） |
| B | 5 | 负（115.19） | 负（115.19） | 平（623.85） | 平（623.85） |
| C | 3 | 负（0.00） | 平（154.35） | 平（154.35） | 负（116.23） |
| C | 5 | 负（0.00） | 负（0.00） | 平（154.35） | 负（116.23） |

这是本轮最重要的**负面结果**，如实保留：LRU/Marconi 因按声明访问事件/位置代理保留了 JOIN 与 S1 状态，在 B、C 的未来口径上明显反超 FlowState；A K=5 上 Marconi 未来最优（617.40）。成因是结构性的口径差异，不是实现缺陷——LRU 的合成访问事件、Marconi 的代理、KVFlow 的声明 STE 都以声明方式"看到"了未来调用，而 §3 规定 FlowState 决策只能使用当前 Pending。代价是 FlowState/WFC 的零边际停机把槽位还给系统，静态上不再覆盖未来调用。任何 GPU 验证必须**同时记录两个口径**，不得只报当前 J。

## 5. §5 五个分析问题的回答

**Q1 各策略选择了什么？** 见 §3 表。模式清晰：LRU/Marconi 选"最近/最大位置代理"状态（A 中最近 Agent、C 中 JOIN）；KVFlow 选声明 STE 优先的早期/共享前缀（A 的 AGENT00 祖先、B/C 的 W0/AB 侧）；WFC 选公平覆盖的前缀（C 中 ROOT/AB/CD）；FlowState 按边际贪心选当前 Pending 的完整末端（B/C 中恰好是各 S0/各叶）。

**Q2 哪些差异真正改变了 Current-Pending J？** 全部 47 个 DIFFERENT_FRONTIER 对。三个机理：(i) A——LRU/Marconi 按新近度选近期 Agent，完全错过唯一当前 Pending（G=4096 → J=154.35）；(ii) B——LRU/Marconi 把额度花在 S1 近期状态、KVFlow 的 STE 偏好 W0 侧祖先，三者都 miss 部分当前 S0（230.76–236.25），K=5 时 KVFlow 才以 2 个祖先凑出 W1_S0 的 E=4032（剩 2.49）；(iii) C——K=3<K_min=4 人人都要留一个未覆盖，但留谁差别巨大：FS 按边际估值留下 A（J=154.35），WFC 留下的是"人人半覆盖"（196.11），LRU/Marconi 留两个叶（348.23），KVFlow 留 C/D 两个叶（510.37–627.62）。

**Q3 哪些差异来自恢复估值、哪些只是排序或同分规则？** 6 个 SAME_SET_SAME_FRONTIER（FS–WFC×4、B 的 LRU–Marconi×2）与 7 个 DIFFERENT_SET_SAME_FRONTIER（A 的 LRU–Marconi、KVFlow–WFC、FS–KVFlow 各×2，C K=5 的 LRU–Marconi）**不改变当前 J，纯属选择集/排序差异**。典型：A 中 KVFlow 用满 3 个额度（含 4032 祖先）与 FS 只用 1 个额度，当前 J 同为 0——但同一持平对在 Frozen-Future 上 KVFlow 反而更好（1237.24 vs 1389.14），且多占 2 个槽位：同分规则差异不改变当前恢复成本，却改变未来覆盖与物理预算占用。真正由估值驱动的是 47 个 DIFFERENT_FRONTIER 对；FS 的提前停机（A 选 1/3、C K=5 选 4/5）是"零边际不驻留"的估值行为而非同分规则，`selection_traces.json` 中 A K=3 首选后 18/20 候选边际为 0 可复核。

**Q4 Shared/Branched 重叠是否带来可识别收益与竞争？** 是。收益侧：WFC 用 ROOT/AB/CD 三个共享前缀在 C K=3 就让四个叶各得 E=2048/2048/3072/3072（J=471.87），KVFlow 的 AB 祖先给 A 凑到 E=4032/G=64——共享前缀确实一次覆盖多个 Pending。竞争侧：(i) K=3<K_min=4 时预算迫使至少一个 Pending 失去覆盖，选谁由估值决定，FS 的末端贪心在此最占优；(ii) K=5 时"叶全覆盖 + JOIN 无覆盖"（FS：Jcur 0 / Jfut 154.35）与"JOIN 保留"（LRU/Marconi：Jcur 348.23 / Jfut 0）构成真实权衡，**没有任何策略同时做到两个口径为 0**；(iii) 祖先不是免费驻留——本协议把祖先全部纳入统一可控域，KVFlow 选 4032 祖先就要消耗额度，这正是 C 中其落后的原因。

**Q5 什么条件值得进入 GPU？** 差异幅度大且方向稳定的场景：C 结构（全部 8 对 FS 胜、幅度 193.89–510.37 模型 ms、全部 DIFFERENT_FRONTIER）与 B K=3（当前前沿命中 0 vs 230.76/236.25）。A 的分离虽 100%，但单 Pending 场景解释平凡，只配作对照。前提条件：(a) 候选域物化审计——`kvflow_native_recovery_diagnostic` 已证明 4032 祖先会改变恢复语义与物理预算，PROPOSED→实际 GPU 状态、槽位、别名、驱逐必须逐项核查，未验证组合不得标 Runtime READY；(b) 全部 T 低于校准目标范围下限，GPU 实验要么先扩展校准、要么全部结果标注外推；(c) 决策语义与 CPU 相同（FlowState 只见当前 Pending，LRU/Marconi/KVFlow 用各自声明输入）；(d) 双口径（Current/Frozen-Future）同时记录。

## 6. 冻结输入与输出的可复核身份

| 层 | 文件 | 内容 |
| --- | --- | --- |
| 输入锁 | `input_lock.json` | 21 项 sha256（工作负载、protocol、生成器、tokenizer、optimizer/recovery_model/state_catalog、三个策略源、WFC/common、校准点），冻结于策略执行前 `1792582180500275913` |
| 执行锁 | `execution_lock.json` | 启动时刻 + input_lock 摘要 + run_cpu.py 摘要 |
| 输出清单 | `output_manifest.json` | 本轮生成，目录内 15 个文件的 sha256 |
| 原始输出 | `cpu_comparison.csv`(30 行)、`pairwise_comparison.csv`(60 行)、`candidate_priorities.csv`、`candidate_resource_ledger.csv`、`retention_ledger.json`、`selection_traces.json`、`cpu_results.json` | 全部为一次运行产物，反覆盖断言禁止重跑 |

复核路径：重算 `input_lock.json` 各摘要 → 核对 `execution_lock.json` → 用 `output_manifest.json` 核对输出未被改动 → 只读重放 `common.score` 即可从 `cpu_comparison.csv` 的选中集复算每个 J。

## 7. 有资格进入 GPU 的最小实验清单

1. **C Shared/Branched，K=3**（首选）：FS vs WFC vs LRU 三策略对照。CPU 上差距 193.89–473.27 模型 ms、全部 DIFFERENT_FRONTIER、四策略选择集互不相同，最能检验估值差异是否在真实恢复上成立。
2. **C Shared/Branched，K=5**：FS（Jcur 0/Jfut 154.35）vs LRU（348.23/0）——把"当前/未来"权衡从静态外推变成实测双口径数据。
3. **B Independent，K=3**：FS/WFC vs LRU/Marconi/KVFlow——多工作流当前前沿命中差异（0 vs 230.76/236.25）。
4. A Sequential 仅作对照组（分离平凡）。

前置门槛（不满足则不得启动）：§5 Q5 的 (a)–(d)。不自动启动 GPU。

## 8. 简短结论

**出现了足以支持进一步 GPU 验证的策略差异。** 在决策口径 Current-Pending J 上，FlowState 对四个基线 18 胜 6 平 0 负（6 个平局中 4 个是 WFC 复制其选择集、2 个是 KVFlow 以不同集合达到同一当前前沿），C Shared/Branched 全部 8 对获胜且幅度最大（至 510.37 模型 ms），K=5 时唯一做到当前 J=0。两条必须随行申明的保留：其一，全部数值为校准范围外的外推模型数，非 GPU 实测；其二，Frozen-Future J 上 FlowState 16 负 8 平 0 胜——LRU/Marconi 靠声明未来信息在 B/C 反超，这是决策语义的结构性代价而非缺陷，GPU 验证必须双口径同时记录。首选 GPU 实验：C 结构 K=3。
