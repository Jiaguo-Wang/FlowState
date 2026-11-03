# FlowState ICDE 2027：实验证据与科学结论简报

**审查边界：**只读取现有代码、原始记录及冻结汇总，未运行实验或策略。最新正式 GPU 实验的读取截点为 **2026-10-22 22:03:41（北京时间）**；该时点之后的进展不纳入本稿。STRONG 表示对明确限定的主张有直接证据，PARTIAL 表示只能支持部分主张，UNSUPPORTED 表示现有证据不能支持；不代表系统整体评级。文中 modeled ms 均为共同目标函数的预测单位，不是 GPU 测量。

## 1. Executive Findings

1. **Attention 命中不等于可执行恢复，是最强的系统动机。**固定 H=32K，改变实际 Recurrent 恢复位置，30 次 Qwen 实测的服务端 TTFT 均值从 32.495 ms 增至 1522.914 ms。Kimi TP=2 也验证了 Attention 保留、Recurrent 单独移除后恢复区间变长。两者支持 Hybrid 状态必须分组件建模，而非证明某种分配算法必然更优。[Qwen 原始数据](motivation/artifacts/replay_cost_20260819/raw_runs.csv)、[Kimi 配对记录](evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json)
2. **集合收益模型有实际区分度，但 Greedy 不是主要优胜者。**新 5 个 Strict Snapshot、20 个预算条件中，Greedy 与 WFC 全部持平；Greedy 仅 5/20 达到 Exact，1-Swap 为 18/20。Greedy 相对 StandaloneTopK 有 4 次严格改善，但不能据此声称普遍优于强简单策略。[逐条件结果](evaluation/greedy_swap_structure_e3_step1/condition_results.csv)
3. **Position-aware 的预测证据强，分配收益证据不足。**独立 held-out 聚合点上，MAPE 从 Gap-only 的 15.291% 降至 2.682%；但相同 20 条条件下，Exact、Greedy、1-Swap 的选集均未因模型切换改变。当前不能把预测改进直接写成已证实的分配或 GPU 收益。[模型选择](evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/model_selection.json)、[选择对照](evaluation/strong_baseline_position_audit_e5_step0/position_vs_gap.csv)
4. **Native 控制与恢复正确性已闭合若干受控实例，完整性能收益尚未闭合。**Shared/Branched K=3 的 12 个策略—Pending Arm 全部通过恢复和 Cold 输出；新的 60 臂性能实验截至截点仅 14/60 PASS，仍为 **IN_PROGRESS**。首轮数据仅可作 Preliminary，不能给出五次重复的最终结论。[Pilot](evaluation/c_shared_gpu_pilot_k3/final_report.md)、[正式执行状态](evaluation/c_formal_perf_gpu/arm_status_main.json)
5. **旧 43.06%/24.26% 可以复现，但只能作为限定的历史比较。**216 个生命周期、864 条请求记录完整；优势是逐快照相对改善率的平均，发生在协调之后，不包含控制成本；Marconi 有 2/24 负面快照，必须保留。[原始运行目录](rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/)、[逐快照核验](evaluation/legacy_runtime_evidence_p0_audit/snapshot_ttft.csv)

## 2. Claim–Evidence Matrix

| 拟支持主张 | 主要证据及实际数字 | 能证明／不能证明 | 评级；论文位置 |
|---|---|---|---|
| Hybrid cache 需要 Executable Recovery Frontier | [32K 原始记录](motivation/artifacts/replay_cost_20260819/raw_runs.csv)：6 个 G ×5 次；H 固定，TTFT 随恢复缺口显著变化 | 证明 FA 命中不能唯一决定恢复计算；不是纯 CUDA 时间或分配收益 | **STRONG**；Motivation |
| 恢复成本依赖 G 与 T | [冻结校准与 held-out](evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/)：MAE 287.514→31.180 ms，MAPE 15.291%→2.682% | 长上下文预测改进；不能证明短上下文预测准确或选集改善 | **STRONG**（预测）；Method/Evaluation |
| Workflow-conditioned valuation 有别于 Recency/STE | [合成 CPU 完整结果](evaluation/kvflow_multiworkflow_cpu_main/cpu_comparison.csv)：6 条条件中 FS 相对 LRU/Marconi 各 6 胜，KVFlow 4 胜2平 | 在声明的当前 Pending 目标下，需求与恢复深度改变取舍；不证明全未来工作流最优或官方系统性能优势 | **PARTIAL**；Evaluation |
| 动态边际更新能避免静态重复覆盖 | [E3](evaluation/greedy_swap_structure_e3_step1/condition_results.csv)：Greedy 对 StandaloneTopK 4胜16平 | 在合法 Strict 矩阵上有模型目标改善；尚非这四条差异的逐一 GPU 收益认证 | **STRONG**（模型）；Method/Evaluation |
| Greedy 普遍优于强简单策略或接近最优 | 新 Strict 20 条：与 WFC 全平；相对 Exact 15 负；[旧 573](evaluation/p1_strong_baseline_validation/per_case_results.jsonl)与 PCB/LDR/WFC 同集 | 反证现有样本足以支持该广泛主张；经典次模保证不能代替实证 | **UNSUPPORTED**；Limitations |
| Exact 相比强局部替换有稳定 GPU 联合收益 | [E5 三组四 Pending 总和](evaluation/exact_swap_runtime_e5_step1/four_pending_paired_sums.csv)：Exact−1-Swap 为 −13.399、+1.938、−3.102 ms | 2/3 组计算总和降低，但方向不一致；不能写“持续更低”或统计等价 | **PARTIAL**；Evaluation/Limitations |
| 严格选集能够落实为安全的物理恢复差异 | [C Pilot](evaluation/c_shared_gpu_pilot_k3/pending_policy_matrix.csv)、[E4 正确性](evaluation/native_resume_correctness_e4_phase1a_run01/final_report.md) | 有 Recurrent-only、FA 保护、Native Match/COW、Cold 输出证据；只认证实际执行的 Arm，不升级全部边或张量血缘 | **STRONG**（受控正确性）；Implementation/Evaluation |
| 降低恢复缺口带来生产端到端净收益 | [E1 原始计时](evaluation/lightweight_runtime_e1/gpu_raw_timings.csv)、[正式实验原始运行](evaluation/c_formal_perf_gpu/) | 部分 CUDA/TTFT 改善；控制、审计、预热和客户端边界仍不可混为生产延迟 | **UNSUPPORTED**（生产净收益）；Limitations |
| 方法适用于不止一种 Hybrid 架构 | [Kimi 最终合同](evaluation/rq5b_kimi_final_output/rq5b_frozen_20260923_090154/final_report.md)：TP=2 两 rank 状态控制、5 对恢复检查 | 支持 KDA/MLA 分离控制与恢复；不是 Qwen Phi 在 Kimi 上的校准或多基线性能复现 | **PARTIAL**；Evaluation |

**创新定位：**最扎实的贡献是可执行恢复语义及其可验证的状态控制闭环，以及将当前需求、恢复深度和重叠收益放进统一目标。固定矩阵的 max-coverage 型次模性是标准性质；Greedy 本身、Native COW 和原生驱逐 primitive 不应称为新算法或新 Runtime 发明。[理论边界](evaluation/allocation_structural_properties/final_report.md)

## 3. Allocation and Baseline Evidence

### 3.1 数据族、样本单位与可保留结论

| 数据族与来源 | 实际规模、比较及结果 | 证据边界 |
|---|---|---|
| [旧 OpenHands/AgentX Formal](evaluation/p1_strong_baseline_validation/per_case_results.jsonl) | 168×3=504 与23×3=69，共573条；CPU 读取冻结 Phi 目标，非重复 GPU 测量 | OpenHands 正收益支持至多单 Pending；AgentX 69 条均预算饱和。只能保留旧模型内部比较 |
| [B4-C 严格采集后的分配](evaluation/strict_compatibility_b4c/snapshot_budget_policy.csv) | g001–g004，4×4=16；224 对中68准入、156拒绝；Greedy 12/16 Exact | 这是另一批16条，**不是**下述20条。单共享候选结构可两分支 Top-K 精确求解，[B5-C0](evaluation/structural_exactness_gate_b5c0/final_report.md) |
| [E2 四个独立来源](evaluation/independent_runtime_allocation_e2/runtime_materialization_summary.csv)及[B5-C4](evaluation/multishare_allocation_gate_b5c4/policy_comparison.csv) | E2 每组7候选、4 Pending、K_min=4；准入数12/12/14/13，均0 UNKNOWN；另加B5-C4，构成5快照×4预算=20 | Qwen/H100 Runtime 物化支持的 Strict Epoch；CPU 求解，不是20个独立工作负载，也不是所有边逐一 Resume 认证 |
| [KVFlow Sequential](evaluation/kvflow_synthetic_stage1/cpu_comparison.csv) | 10 Agent，K=3/5；5策略全部同目标，J=1080.443/771.745；选集可不同 | 合成声明输入。最初评价涵盖10个调用，不能冒充10个当前 Ready；10末端 CPU 模型漏计10个物理祖先 |
| [Multi-Workflow CPU](evaluation/kvflow_multiworkflow_cpu_main/cpu_comparison.csv) | A顺序/B独立工作流/C共享分支，各K=3/5，30策略结果；20/12/16个 PROPOSED 状态 | 全候选计账、无免费祖先；只有C/K3后来完成对应 GPU 物化。A/B不能自动升级 Runtime 证据 |

旧 Formal 的关键缺口不是单纯少日志：[AgentX 审计](evaluation/gate1_step3_compatibility_retention_contract_audit/audit_report.md)确认 **70 条 Formal-only 正收益边不满足 Raw Token Prefix**。因此这些数字可解释旧逻辑目标，不能作为严格可执行分配收益。旧 OpenHands 也缺原始 Pending Token/配置等完整准入链，不能由 lineage 自动认证。

### 3.2 统一目标下的胜平负

下表为 **FlowState-Greedy 相对行中方法**的 W/T/L，按各族冻结数值规则，仅比较 J；没有混合成总胜率。[旧573原始结果](evaluation/p1_strong_baseline_validation/per_case_results.jsonl)、[新20原始结果](evaluation/greedy_swap_structure_e3_step1/condition_results.csv)

| 对照方法 | 旧OpenHands：504条件 | 旧AgentX：69条件 | 新Strict：20条件 |
|---|---:|---:|---:|
| LRU | 306/198/0 | 16/53/0 | 未形成同合同比较 |
| LFU | 504/0/0 | 12/57/0 | 未比较 |
| Marconi Adaptation | 386/118/0 | 16/53/0 | 输入未认证，不比较 |
| PCB、LDR（各自） | 0/504/0 | 0/69/0 | 15/0/5 |
| WFC | 0/504/0 | 0/69/0 | 0/20/0 |
| StandaloneTopK | 144/360/0 | 2/67/0 | 4/16/0 |
| Exact | 不作无界重新枚举 | 不作无界重新枚举 | 0/5/15 |
| Greedy+1-Swap | 未比较 | 未比较 | 0/5/15 |

新20条中 Exact/Greedy/1-Swap/WFC/Standalone/PCB/LDR 达到 Exact 的次数分别为 **20/5/18/5/5/5/5**。四次 Greedy 优于静态 TopK 发生于 w3/K3、w3/K4、w4/K4、B5-C4/K4，源于重复覆盖；不是泛化后的所有多共享树都会改善。w1/K3 的 Exact J=21.953375、1-Swap J=31.817384，表明固定首改善交换也会停在局部最优。[条件表](evaluation/greedy_swap_structure_e3_step1/condition_results.csv)

**为什么大量持平？**旧 OpenHands 近似可分离、旧 AgentX 预算已覆盖最佳前沿；新20条的 Greedy/WFC 虽评分不同，但每一步胜出候选一致，不能由低层排序不同推断最终选择不同。[573结构统计](evaluation/greedy_swap_structure_e3_step1/legacy_573_structure.csv)、[50步审计](evaluation/greedy_swap_structure_e3_step1/greedy_wfc_step_audit.csv)。B5-C4 的 Greedy 在K=2时 J=130.803，对 Exact 48.229；早期公共祖先占据预算，无法随预算增长撤回，最优集合不必嵌套。这限制 Greedy 的贡献叙述，不否定目标函数的价值。

### 3.3 Baseline 的真实身份与信息边界

| 方法 | 实际身份、决策输入与限制 |
|---|---|
| LRU / LFU | 内部 checkpoint 保留对照，不是完整原生缓存系统。旧LRU按访问、创建顺序及ID；LFU按频率、最近访问及ID。合成轨道的时间信息来自生成事件。缺元数据时不能移植到新Strict20条。[LRU代码](evaluation/controlled_multiworkflow_v1/policies.py)、[LFU代码](evaluation/rq3_frozen_snapshot_evaluator.py) |
| Marconi Adaptation | 受控适配，归一化 recency + α×归一化 compute/memory，旧实验α=1；旧 FLOP proxy 取增量长度，合成轨道取自身长度，不能混为同一输入合同。非完整官方系统；新20条缺可信原始元数据，UNVERIFIED。[策略](evaluation/sota_policies.py)、[元数据](evaluation/sota_metadata.py) |
| KVFlow-inspired | 受控适配：Ready STE=0，AND取max+1、OR取min+1，固定前缀祖先取最小STE；动态后缀优先淘汰，同分采用明确的ID规则。无Prefetch/Scheduling。真实OpenHands/AgentX缺可追溯ASG/Fixed Prefix声明，不能用LCP补造；现有完整比较来自合成轨道。[策略](evaluation/kvflow_inspired_retention_cpu/policy.py)、[真实数据接入记录](evaluation/kvflow_real_data_connection/final_report.md) |
| PCB / LDR / WFC | 内部强简单对照：PCB按Pending最佳收益候选排序去重；LDR按最深位置；WFC动态优先推进Pending数量、其次Token推进总量。WFC不使用Phi。[冻结实现与审计](evaluation/greedy_swap_structure_e3_step1/) |
| StandaloneTopK | 按候选独立收益总和静态排序，信息域与FS相同，但不扣除已覆盖收益。用于隔离集合依赖机制，不是外部SOTA系统 |
| Exact / 1-Swap | 独立枚举最优参考／实验性局部搜索。1-Swap固定遍历、首次严格改善、重复至局部最优；不可称生产求解器或全局最优。[E3实现与结果](evaluation/greedy_swap_structure_e3_step1/) |

**实际运行的求解器必须分开写。**生产 [GlobalOptimizer](flowstate/optimizer.py) 的 `select`、`plan_strict` 仍调用 Greedy；[B5-D1 Exact](evaluation/exact_planner_contract_gate_b5d1/planner.py) 是独立实验适配器，准入 n≤12、m≤4、K≤4、枚举数≤794。最新 C GPU [control.py](evaluation/c_formal_perf_gpu/control.py)读取冻结 `policy_plans.json` 后按当前物理身份完成计划，不是在该控制窗口重新运行全部在线选择算法；不能把该阶段耗时标成完整在线求解成本。

### 3.4 Position-aware 与 Workflow-conditioned 的独立性

冻结模型为 Φ=37.828150g+0.345974143gt−0.156201917g²，g=G/1024、t=T/1024。[正式模型](evaluation/FORMAL_RECOVERY_MODEL.md)。校准20点，独立 held-out 的非零G聚合点12个；原始 held-out 含210条（含零点及预热）。模型代码可接受短T，并不等于短T被校准。

20条条件×4求解方法的模型切换中，Exact、Greedy、1-Swap各20/20同集；Standalone仅 w4/K3 不同，涉及同深度候选、静态重复覆盖与排序。没有独立GPU对照证明这个差异更优。[80行模型对照](evaluation/strong_baseline_position_audit_e5_step0/position_vs_gap.csv)。因此目前 Position-aware 只能承担预测贡献，不能单独承担已验证的状态选择收益贡献。

合成C/K3的共同J为 FS154.349、Marconi348.234、WFC471.874、LRU582.035、KVFlow627.615；它说明“当前请求需求×可恢复深度”不等价于recency或STE，但这些短上下文值在长上下文校准范围外。未来静态评价还出现反向取舍，例如C/K3未来成本 FS154.349、LRU0；优化当前Pending不等于优化未来序列。[完整CPU结果](evaluation/kvflow_multiworkflow_cpu_main/cpu_comparison.csv)

## 4. GPU Performance and Runtime Overhead

### 4.1 已完成证据的范围

除Kimi外，下列主要Runtime结果使用 Qwen3.5-9B、H100 PCIe 单卡、TP=1及版本锁定/插桩的 SGLang v0.5.17；不是所有安装均可视为未修改的上游发行版。

| 实验及来源 | 样本／主要测量 | 可用结论及限制 |
|---|---|---|
| [32K恢复](motivation/artifacts/replay_cost_20260819/raw_runs.csv) | G=0/1/4/8/16/32K，各5次；均值32.495/64.454/210.838/411.194/804.369/1522.914 ms | 服务端TTFT，不是CUDA，也不含前置控制 |
| [旧24快照](evaluation/legacy_runtime_evidence_p0_audit/final_report.md) | 24×3策略×3次=216生命周期；每次4串行请求，共864；FS/LRU/Marconi均值94.025/164.705/129.097 ms | 逐快照相对改善均值43.06%/24.26%；胜负24/0、22/2。协调后1输出Token，缺当前严格Cold/Native/COW闭环，**QUALIFIED**历史比较 |
| [B5-D4b配对](evaluation/exact_greedy_paired_gpu_gate_b5d4b/paired_gpu_timings.csv) | 单一组合、Pending C、5对Greedy/Exact；G=1472/384，实际Prefill=2003/915 | CUDA78.348→46.099 ms，5/5改善；TTFT1039.121→1078.325 ms，5/5恶化；含控制完成13219.006→13135.119 ms，仅3/5改善 |
| [E1四臂](evaluation/lightweight_runtime_e1/gpu_raw_timings.csv) | 重/轻×Greedy/Exact各3次，另Smoke与Cold；Greedy控制4682.064→788.566 ms，Exact4926.071→806.993 ms | 分离重型审计确实降低控制；轻路径Exact CUDA43.647 vs76.846、TTFT73.169 vs107.556 ms，但审计全路径均值反而高92.071 ms |
| [E4-1A/1B](evaluation/native_resume_performance_e4_phase1b/gpu_raw_timings.csv) | 1A为8严格Arm+4Cold；1B为A/C×2策略×3测量（另Smoke4、Cold2） | A Exact/Greedy CUDA42.618/73.026 ms；控制824.074/775.516 ms；C同前沿CUDA37.460/37.062，负对照不支持算法普遍加速 |
| [E5 Exact/1-Swap](evaluation/exact_swap_runtime_e5_step1/four_pending_paired_sums.csv) | 8Smoke+4Cold+24测量；3组，每组4个独立Pending等权相加 | CUDA均值166.253 vs171.107 ms，2/3改善；既非并发墙钟也非稳定生产优势 |
| [Kimi TP=2](evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json) | Kimi-Linear-48B-A3B、2张H100；5对独立生命周期，H=16384，E=16384或8192；两rank最大恢复窗口均值71.749/831.941 ms | 差760.191 ms，5/5同向，10/10正确性及清理；计时为match至单输出完成，非纯CUDA/TTFT，排除准备与控制 |

Kimi 较早“16K→8K”的逻辑移除实际上还可命中14K；最终Phase3预先移除10/12/14K后才形成上述可比8K反事实。必须引用最终真实前沿，不沿用旧逻辑标签。[修正说明](evaluation/rq5b_kimi_final_output/rq5b_frozen_20260923_090154/final_report.md)

旧Marconi的两条负面快照为g015/round5、g184/round2，FS相对改善分别−26.52%、−10.57%。旧审计确认关键文件存在，但不能由文件齐全推出完整当前Strict认证；运行后续请求会新建状态，四串行请求也不是同一静止Epoch四次独立恢复。[逐快照记录](evaluation/legacy_runtime_evidence_p0_audit/snapshot_ttft.csv)

### 4.2 Shared/Branched 的物理状态与真实请求

[C/K3 Pilot](evaluation/c_shared_gpu_pilot_k3/final_report.md)成功部分为17个生命周期：1 Census、12严格Arm（FS/Standalone/WFC×4Pending）、4Cold；此前两次失败尝试保留，不能算作从未失败。16个候选均真实驻留，28槽池剩12槽；每个严格Arm保留3、真实移除13，非选祖先没有免费保留。12/12输出与8Token Cold一致。E=0的Arm不要求虚构COW；正E时验证相应复制与源保护。血缘继续 **PROVENANCE_UNVERIFIED**。

下表合并展示已经验证的恢复结构和**最新正式实验首轮原始测量**。每格仅 **n=1，Preliminary**；FS=Greedy，ST=StandaloneTopK，WFC为原冻结规则。总输入长度为H+32，因此Prefill Tokens=G+32，不应混同G。[Pilot恢复矩阵](evaluation/c_shared_gpu_pilot_k3/pending_policy_matrix.csv)、[正式首轮各Arm原始summary](evaluation/c_formal_perf_gpu/)

| Pending；H | 方法 | 实际 E / G / Prefill Tokens | CUDA Prefill ms | TTFT ms | Request Wall ms |
|---|---|---|---:|---:|---:|
| A；4096 | FS | 0 / 4096 / 4128 | 152.069 | 159.934 | 283.570 |
| A；4096 | ST | 0 / 4096 / 4128 | 150.114 | 157.763 | 283.099 |
| A；4096 | WFC | 2048 / 2048 / 2080 | 79.718 | 88.244 | 217.329 |
| B；5120 | FS | 5120 / 0 / 32 | 45.076 | 52.449 | 256.417 |
| B；5120 | ST | 0 / 5120 / 5152 | 197.483 | 209.977 | 424.022 |
| B；5120 | WFC | 2048 / 3072 / 3104 | 115.307 | 122.207 | 257.101 |
| C；6144 | FS | 6144 / 0 / 32 | 30.241 | 36.676 | 162.713 |
| C；6144 | ST | 6144 / 0 / 32 | 29.685 | 36.024 | 162.591 |
| C；6144 | WFC | 3072 / 3072 / 3104 | 116.017 | 129.549 | 347.666 |
| D；7168 | FS | 7168 / 0 / 32 | 51.649 | 62.258 | 281.246 |
| D；7168 | ST | 7168 / 0 / 32 | 33.884 | 41.186 | 174.191 |
| D；7168 | WFC | 3072 / 4096 / 4128 | 149.786 | 159.023 | 289.881 |

A体现FS牺牲单请求换取其他Pending的选择；D体现相同前沿不保证一次计时相同。不能只展示B，也不能把四次独立生命周期相加称为自然并发性能。

### 4.3 计时污染、预热与最新实验状态

[c_pair_timing_gpu](evaluation/c_pair_timing_gpu/)包含第一阶段20次（B/C×FS/ST×5）与形状预热4次（B×2策略×2）。按 **p01–p05原始summary**重算，B的FS−ST TTFT五对均为正，中位 **+1667.494 ms**，范围+1543.494～+1785.240 ms；C中位−24.897 ms，但有一对+77.329 ms。不能据此认定更短恢复导致模型计算更慢：原记录含首次形状编译与重型观测。

预热两对B的CUDA均值FS **30.769 ms**、ST **196.462 ms**，但TTFT仍约3249.927/1907.570 ms。移走编译后，审计污染仍在请求路径。**发现汇总口径问题：**[pair_analysis.csv](evaluation/c_pair_timing_gpu/pair_analysis.csv)前两条B记录实际引用 `w1/w2` 预热Arm，后面混入 `p03–p05`；不能直接当作五对同协议数据。本文只按原始Arm命名分组，不修改旧文件。[第一阶段原始样例](evaluation/c_pair_timing_gpu/p01_CB_FS/run/summary.json)、[预热原始样例](evaluation/c_pair_timing_gpu/w1_CB_FS/run/summary.json)

**正式60臂：IN_PROGRESS。**计划3策略×4Pending×5轮；截点14个主实验PASS，只有第一轮12臂完整，另2臂属于下一轮。门禁Arm不计为正式重复；最终汇总表未生成，[final_report.md](evaluation/c_formal_perf_gpu/final_report.md)仍有待填占位，不可当最终论文数字。截点状态文件SHA-256为 `3678351ee3204e5cd8df8c1aa1b74abf78ac65935c815c0a456c4fa34ef147bf`。

新协议把请求内重型Hash分离，使用分叉尾部的形状预热避免偷偷建立更深目标前缀；早期预热门禁失败也保留。首轮请求记录无新增被监测Triton缓存文件，并有实际E核对；这不是所有编译/autotune渠道完全为零的证明。[Runner](evaluation/c_formal_perf_gpu/runner.py)、[Hook](evaluation/c_formal_perf_gpu/bridge.py)、[协议](evaluation/c_formal_perf_gpu/protocol.json)

### 4.4 控制成本不能被“净收益”口径抹去

| 区间及来源 | 实际数字／包含关系 | 解释边界 |
|---|---|---|
| E4 Control RPC | 均值814.493 ms；已计子区间332.074，剩余482.419未归因 | 剩余不能猜成fsync或网络；内部Solve约0.129 ms只是子区间 |
| E4 Control-to-completion | 7469.333=RPC814.493+两者间隔243.077+Request6411.764 ms | 约7.5秒不是纯Serving请求；Audit-inclusive9428.766又含参考观察1855.752与间隔103.680 |
| 最新首轮Control RPC | 范围10628.444–12517.937 ms，含重型认证 | 不等于已部署轻量控制；不能与E4不同合同直接比较加速比 |
| 最新首轮控制内部均值 | 重观察2464.913、映射0.591、Epoch/冻结Plan认证165.039、apply重验1252.586、前FA审计953.939、协调880.096、后审计/恢复基线1719.609 ms | 协调含Native Eviction与postcheck，不能再重复加Eviction。父RPC与这些子区间间残差未完整归因，写盘/fsync单项为UNKNOWN |
| 最新首轮Epoch Wall | 范围56277.575–61516.500 ms | 含历史、预热及审计等，排除Engine启动；不是生产请求延迟或并发Goodput |

来源：[E4边界审计](evaluation/e4_control_timing_audit/final_report.md)、[正式首轮控制原始样例](evaluation/c_formal_perf_gpu/m1B_FS/run/summary.json)。该样例Native Match宿主区间约0.962 ms、COW CUDA约0.323 ms；不能把早期带Hash的match包装耗时归给Native复制。宿主与CUDA区间不可相加成一个无重叠总数。

对共享观察/协调路径的策略，观察、认证等大部分是共同成本，但保留/释放对象、实际恢复和求解可能不同；相对无额外控制策略则不能删去这些共同成本。最新控制窗口采用冻结选集，尚未测得完整在线选择到请求完成的低审计净收益。E1的轻量改进与新正式协议不是同一计时合同，不可拼接出一个虚构的生产系统。

## 5. Negative Results and Scientific Risks

- **选择能力与系统收益分离：**新Strict20条的Greedy=WFC、15次非最优、1-Swap仍2次非最优均需公开；Exact的目标最优不等于GPU总时间最优。E5的三组CUDA方向不一致，不能由模型J差9.864 ms证明实测优势。
- **预测外推：**长上下文校准的T为32K起；新Strict和合成主要在数K范围。其共同J可用于固定目标比较，但不能称为已校准准确GPU延迟。Position-aware下未改变Exact/Greedy选集的负面结果必须保留。
- **资源模型：**[4096/4032诊断](evaluation/kvflow_native_recovery_diagnostic/final_report.md)证明末端可合法恢复；移除4096后仍从4032恢复，8Token Cold通过。两个状态占两个槽，旧“10末端=10物理对象”的预算不成立。新16状态C合同修正了这一点，但K=3仍不是GPU HBM不足的证明。
- **来源与证据等级：**E2是Trace-derived Controlled Composition；合成ASG是预先声明的结构，不是观察到的自然Agent并发。EPOCH_ADMISSIBLE、RECONCILIATION_VALIDATED、RESUME_VALIDATED逐级不同；Native命中、Token一致、首次出现时间都不能独自证明跨Workflow张量血缘。
- **基线能力边界：**Marconi与KVFlow均是保留策略适配，不含完整官方系统能力；真实轨迹缺ASG/Fixed Prefix时保持不可比较。合成未来序列只能作事后评价，不能把实际未来输出输入当前决策。
- **统计与报告完整性：**5个快照×4预算不是20个独立工作流；同一合成C的重复不是外部泛化。旧24包含全部负面结果，最新正式实验未完成；c_pair汇总混组必须在论文引用时绕开，不能用报告文字替代原始记录。

## 6. Recommended Paper Figures / Tables

1. **Motivation图：**固定32K FA命中、横轴实际恢复缺口、纵轴服务端TTFT，画全部5次原始点；并列示意FA/Recurrent两个前沿。无需声称算法优势。
2. **预测表：**冻结Gap-only/Position-aware的held-out MAE/MAPE；旁列“Exact/Greedy选择变化0/20”，清楚分开预测贡献与决策贡献。
3. **分配表：**新20条完整J/Exact gap，标记1-Swap两次失败；旧573另表或附录且标Formal-only，不能合并胜率。
4. **Runtime正确性与性能表：**使用C的实际H/E/G、Prefill Tokens、Cold/COW证据；正式性能图须注明当前IN_PROGRESS。E1控制拆分与E5三组等权CUDA差值单列，避免挑选单个获益Pending充当全局收益。

## 7. Open Questions：提交前结论边界

1. **正式60臂是否完整闭合？**本稿只能确认截点14臂与一轮完整数据；最终有效分母、失败/异常、重复分布、配对汇总仍待原任务产物，当前不作最终性能结论。
2. **论文“FlowState”到底指目标、Greedy还是Exact？**生产Greedy、实验Exact和GPU冻结选集执行必须分别命名。当前证据不支持把三者统一称为同一个已验证在线最优系统。
3. **低审计完整控制后是否有净收益？**已有证据能区分CUDA计算、TTFT与控制窗口，但未证明稳定生产净收益；部分RPC残差和客户端/首次Decode开销仍UNKNOWN。不能靠事后扣除估算成本闭合该主张。

本轮仅形成证据整理；未修改任何冻结实验、生产代码或调度中的GPU任务，未安排或启动后续实验。
