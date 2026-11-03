# FlowState ICDE：全局实验进度与证据交接

审计日期：**2026-10-24，Asia/Shanghai**。本轮仅阅读代码、报告、清单、CSV/JSON与必要原始记录；没有调用求解器、测试、Engine、Docker或GPU，没有重新生成选集、拟合模型或修改旧文件。

## 1. Executive Summary：当前项目状态

**当前状态不是“所有论文主张已通过”，而是：受控恢复语义与状态控制证据较完整，小规模分配目标与求解器证据完整，一个合成共享负载的请求级性能矩阵已经完成，但通用在线系统及控制含入净收益尚未认证。**

### 与10月22日交接材料相比的两项实质更新

1. **C Shared/Branched、K=3 正式GPU矩阵已经完成。**本轮直接读取60个 `m*/run/summary.json`、60份清理记录及对应启动前资源记录：60/60状态PASS、8 Token Cold匹配、审计计数与请求窗口编译记录门禁通过；另有2个通过的门禁Arm。不能继续沿用旧简报的“14/60、IN_PROGRESS”。此前失败门禁仍保留，不包括在“62/62成功生命周期”的分母中。[完成账本](c_formal_perf_gpu/launch_main_complete.json)、[原始运行目录](c_formal_perf_gpu/)
2. **生产 `GlobalOptimizer.plan_strict()` 已在10月23日完成Hybrid Solver CPU集成。**合法Strict Epoch在 n≤12、Pending≤4、K≤4、枚举子集数≤794 时走Exact，否则走原Marginal Greedy；非法输入/求解异常不回退。20/20 Strict条件复现Exact，C/K3因n=16回退Greedy。**新生产入口尚没有真实GPU执行认证**；历史实验的外部指定Exact、冻结选集控制不能替它补证。[当前代码](../flowstate/optimizer.py)、[集成门禁JSON](paper_hybrid_solver_integration_gate.json)、[集成报告](paper_hybrid_solver_integration_gate.md)

### 当前可用于正文的核心事实

- 固定Attention驻留、改变Recurrent状态会改变实际恢复前沿及计算；Qwen与Kimi都有受控证据。
- 长上下文独立held-out数据支持Position-aware预测误差下降；**没有证据证明它在Strict20中改变Exact/Greedy选择**。
- Strict20中Greedy、1-Swap、Exact分别5/20、18/20、20/20达到冻结最优目标；Greedy与WFC20/20同集同J，不能省略。
- C/K3正式主矩阵中，四个独立Pending的等权TTFT，每轮配对FS−StandaloneTopK及FS−WFC均为负；单请求A对WFC稳定更慢，是分配取舍的一部分。
- 完整控制墙钟仍约11秒/Arm，并且正式C实验**加载冻结选集而非在线Select**。报告中的“约7次请求回本”是算术估算，不是测得的在线净收益。

### 证据阅读规则

**原始实测**、**CPU modeled objective**、**结构证明/分析**、**计划/未完成**分别标注。下文PASS是对应原实验合同通过，不自动等于当前更强Strict合同。历史项目复用已完成审计；仅对最新变更、主要数字及报告冲突下钻原始记录。没有可读原始证据的内容标记UNVERIFIED，而不以历史报告的READY补齐。

## 2. Experiment Inventory：按研究问题归类

环境简称：**Q**＝Qwen3.5-9B / 单H100 PCIe / TP=1 / 版本锁定SGLang v0.5.17；多数后期实验使用28个Recurrent槽，具体镜像及插桩以各执行锁为准，不能统称未修改的官方发行版。**K**＝Kimi-Linear-48B-A3B / 两张H100 PCIe / TP=2 / SGLang v0.5.17。**CPU**＝不含本项GPU执行，使用冻结Qwen Phi或已有Runtime证据；不同CPU计时版本不得混合。

### 2.1 物理驻留、可执行恢复与Strict证据生成

| 研究问题／实验 | 配置、样本与对照 | 结果及状态 | 能证明／不能证明；证据 |
|---|---|---|---|
| 32K Physical Hit是否足以描述恢复 | Q，固定H=32768，G=0/1024/4096/8192/16384/32768，各5次 | **PASS，原始实测**；TTFT均值32.495→1522.914 ms | FA命中不等于Recurrent执行前沿；不是分配胜率或CUDA纯计算。[raw_runs.csv](../motivation/artifacts/replay_cost_20260819/raw_runs.csv) |
| LMCache实际完整输入共享，A5 | Q，lmcache_10两个完整输入16338/21520 Token；共享状态9984；各1 Token Cold及移除反事实 | **PASS，局部恢复**；移除Recurrent后E9984→0，H9984不变；两续接输出均匹配 | 支持这两个输入，slot generation仍UNKNOWN；不能认证全部LMCache情景。[报告](submission_p0_lmcache_recovery/lmcache_runtime_report.md)、[原始证据](submission_p0_lmcache_recovery/runtime_evidence.json) |
| 旧OpenHands跨Workflow共享是否已认证，A8 | CPU只读168个历史快照；重建提示发现504条2432前缀关系 | **BLOCKED**；缺原始Runtime Token/配置闭环 | 不能把重建Token与旧lineage直接升级为共享；BLOCKED不等于不存在。[报告与清单](openhands_cross_workflow_compatibility_audit/) |
| B1/B1-C最小严格采集与跨输入恢复 | Q，g000：8候选、4Pending、32对，11准入/21拒绝；保留、移除、Cold独立生命周期 | B1 **PARTIAL**；B1-C局部 **PASS**；A请求H4352，E2432→0，G1920→4352，8 Token一致 | 具体边恢复验证，不推广另外10条边。B1组件合同弱于后续B4。[B1](strict_compatibility_b1_output/b1_openhands_g000_20261014_02/)、[B1-C](strict_compatibility_b1c_output/b1c_g000_20261014_r2/) |
| B4受控来源采样与同边界认证 | Q，g001–g006共6生命周期，84历史请求、24Pending身份核对；84候选、336对 | **PASS**；102准入/234拒绝/0UNKNOWN；正式入选g001–g004，56候选、224对、68准入 | 预注册主层配额排除g005/6，不是筛除失败；新增逐边RESUME_VALIDATED=0。六组均主层A，不能当三独立分层总体。[账本/清单/原Snapshot](strict_compatibility_b4b1_output/capture_g001_g006_run01/) |
| B5真实来源多共享是否可构造 | CPU审计5000原始OpenHands会话：1216个同任务多Session组；固定两个任务各两Session | B5-C1 **PARTIAL**，C2输入门禁 **READY**；LCP全体2438、AB3765、CD3566 | Trace-derived Controlled Composition，不是自然Fork或同时在线Ready。[C1](workload_structure_feasibility_b5c1/)、[C2锁与Token](multishare_cpu_input_gate_b5c2/) |
| 多共享能否自然物化，B5-C3 | Q，固定8条历史、4Pending不执行；1生命周期 | **READY**；7候选、28对，13准入/15拒绝/0UNKNOWN；实际共享位置2432、3712、3456/3520 | CPU LCP不等于物化点；同边界完整Strict Snapshot，不等于全部边已Resume。[物理Census/Strict Epoch](multishare_runtime_gate_b5c3/) |
| 独立来源复现，E2 | Q，w1–w4各1独立生命周期；各7候选/4Pending | **READY**；准入12/12/14/13，均0UNKNOWN，均有共享、K_min=4 | 四个受控组合；不是4个原生并发系统样本。[物化CSV](independent_runtime_allocation_e2/runtime_materialization_summary.csv)、[原Epoch/来源](independent_runtime_allocation_e2/) |

### 2.2 成本模型、分配质量与CPU求解

| 研究问题／实验 | 样本、方法 | 结果及状态 | 边界；源文件 |
|---|---|---|---|
| Phi是否需要目标位置T，RQ2 | Q，20个校准点；held-out 3T×5G×(2预热+12测量)=210次 | **PASS**；非零G的12聚合测试点，MAPE M0=15.291%、M2=2.682% | 测试未参与拟合；短T及Kimi未校准。[config/raw/predictions](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/)、[独立审计](rq2_model_audit_output/20260909_055110/) |
| 旧Formal能否区分联合分配，573 | CPU，OpenHands168×3=504；AgentX23×3=69；8种方法 | 旧模型内部比较 **PASS**，Strict可执行性 **不成立/缺证据**；FS=PCB=LDR=WFC同集573/573 | OpenHands可分离；AgentX预算饱和，不能检验共享紧预算优势。[逐条件](p1_strong_baseline_validation/per_case_results.jsonl)、[结构证书](greedy_swap_structure_e3_step1/legacy_573_structure.csv) |
| LMCache情景结构，A6 | CPU，16对完整Token；47 PROPOSED候选、63预算条件、378策略结果 | **STRUCTURAL_READY**；FS对WFC26胜37平，对Exact61平2负 | T=N为ASSUMED_FULL_FA，非Runtime物化；7组仅长度可容纳，9组超原配置context。[报告/矩阵](submission_p1_allocation_structure/) |
| 初始严格矩阵B2、解析B3 | B2：g000 8×4、K1–4；B3：4个合法前缀树解析案例 | B2 **DISCRIMINATIVE**于PCB/LDR；FS=WFC=Standalone全部K；K4 Greedy落后Exact | B3是机制正反例，不是新真实工作负载。[B2](strict_compatibility_b2/)、[B3](strict_compatibility_b3/) |
| B4-C与单共享精确化简 | CPU，g001–g004×K1–4=16；7策略 | **PASS**；Greedy12/16 Exact；B5-C0两分支Top-K16/16匹配Exact | 单共享B1+专属链是特殊可解结构，不是一般复杂共享。[B4-C](strict_compatibility_b4c/)、[B5-C0](structural_exactness_gate_b5c0/) |
| T1/T2树结构与证明 | 数学分析及冻结g001–004结构核验，无新DP | T1条件性证明；T2 **PARTIAL**：四正式Snapshot满足树合同，g000物理合同未定 | 标准次模性非新理论；DP未实现，额外兼容过滤/物理耦合可能改变适用性。[T1](allocation_structural_properties/)、[T2](allocation_structural_properties/t2/) |
| Strict20，E2+B5-C4/E3 | CPU，5快照×K1–4、7策略=140结果 | **PASS**；Exact20、Greedy5、1-Swap18、WFC5、Standalone5、PCB5、LDR5次达到Exact | 同快照多个K不是独立工作负载。全量失败和持平保留。[condition_results.csv](greedy_swap_structure_e3_step1/condition_results.csv) |
| Position-aware是否改变选择，E5-0 | CPU，同20条件、Exact/Greedy/1-Swap/Standalone，80条模型对照 | **完成**；前三种各20/20同集；Standalone仅w4/K3不同 | 不能用Phi自身J证明其选择在GPU更优；Marconi此族输入UNVERIFIED。[对照](strong_baseline_position_audit_e5_step0/position_vs_gap.csv) |
| Exact可行性，B5-D0/D1 | D0：5冻结Snapshot/20条件；D1：原g组+B5C3，12门内Exact、8超域Fallback、36拒绝测试 | **READY，CPU规划**；保持原Plan合同，n≤12/m≤4/K≤4/≤794 | D1这20条不是E2新Strict20；Python时间检查不是强制deadline。[D0](solver_feasibility_gate_b5d0/)、[D1](exact_planner_contract_gate_b5d1/) |
| 最新求解质量—成本微基准 | CPU Xeon Gold6426Y、Python3.10.12；Strict20加C/K3，104分组×120计时=12480行，另5轮热身 | **GREEDY_PARTIAL**；Strict全局中位Greedy0.0972、Exact0.2238、1-Swap0.2209、WFC0.0270 ms；C Greedy0.2587 ms | 不是在线P95；C Exact无证书。报告个别“生产默认”表述早于真正集成，见§4/§7。[原始计时](paper_solver_tradeoff_raw.csv)、[门禁](paper_solver_tradeoff_gate.json) |

### 2.3 Runtime控制、恢复正确性及请求性能

| 研究问题／实验 | 样本与实际执行 | 结果及状态 | 边界；证据 |
|---|---|---|---|
| R1→R2严格生产接线 | R1审计；R2-A纯规划16条件；R2-B Fake Runtime/CPU118项 | R1 **BLOCKED**原因明确；A/B **READY（CPU）** | 旧workflow谓词阻断已在Strict入口修复，Legacy仍独立保留；不能把Fake Runtime当GPU。[R1](runtime_control_closure_r1/)、[R2-A](runtime_control_closure_r2a/)、[R2-B](runtime_control_closure_r2b/) |
| R2-C0/C1a真实只读绑定与驱逐 | Q，g001/K2，12历史、保留B1/B3；新生命周期 | **READY**；C0只读；C1a10次真实移除，驻留12→2、空闲16→26，FA/Selected不变 | 同步控制窗口有效，非跨RPC长租约；C1a尚未恢复请求。[C0](runtime_control_closure_r2c0/)、[C1a](runtime_control_closure_r2c1a/) |
| R2-C1b恢复重验 | b1因观测代码读不存在Req.fill_ids失败；b2真实Req CPU43项+3子测试、1Strict+1Cold | b1 **BLOCKED**；b2 **READY**。实际H/E/G=4224/2432/1792，Prefill1962，COW6→27完成，8 Token一致 | 输入4394，Prefill不是G；失败不是已证实模型恢复错误。[b1](runtime_control_closure_r2c1b1/)、[b2原始输出/状态](runtime_control_closure_r2c1b2/) |
| B5-D2/D3新Epoch与Exact物化 | D2一个生命周期Shadow；D3实时Exact K2、1正式Pending及Cold | **READY（独立实验）**；D2 7×4/13准入；D3移除5项，B从AB共享状态恢复 | D2未协调；D3外部指定Exact，不能认证2026-10-23新Hybrid入口。[D2](exact_runtime_shadow_gate_b5d2/)、[D3](exact_reconciliation_correctness_gate_b5d3/) |
| B5-D4失败、修复与配对 | D4来源门禁错误、未正式恢复；D4a30/30 CPU；D4b5对Greedy/Exact独立生命周期 | D4 **BLOCKED**、D4a **READY**、D4b **READY**；C Prefill78.348→46.099 ms全改善，TTFT1039.121→1078.325 ms全恶化 | 血缘PROVENANCE_UNVERIFIED不等于恢复不合法；重控制完成仅3/5改善。[失败](exact_greedy_paired_gpu_gate_b5d4/)、[修复](exact_greedy_paired_gpu_fix_b5d4a/)、[10行计时](exact_greedy_paired_gpu_gate_b5d4b/paired_gpu_timings.csv) |
| E1分离重型审计与轻量控制 | Q，重/轻×Greedy/Exact各3测量；另正确性Smoke+Cold | **READY**；控制均值Greedy4682.064→788.566、Exact4926.071→806.993 ms | 具体入口正确性通过，不是通用增量状态管理器或无审计生产净收益。[四臂原始CSV](lightweight_runtime_e1/gpu_raw_timings.csv) |
| E4 w1/K2独立来源正确性与性能 | 1A：8严格Arm+4Cold；1B：A/C×2策略×3测量，另4Smoke+2Cold | **READY**；A Exact/Greedy CUDA42.618/73.026 ms；C同前沿37.460/37.062 | 最初资源BLOCKED目录保留；A正向、C负对照；单个Pending非四请求总收益。[1A](native_resume_correctness_e4_phase1a_run01/)、[1B](native_resume_performance_e4_phase1b/) |
| E5 w1/K3 Exact vs固定首改善1-Swap | 8Smoke+4Cold+24测量；三轮每轮四Pending×两策略 | **READY（执行）／PARTIAL（稳定优势）**；CUDA等权总和差−13.399、+1.938、−3.102 ms | 独立恢复总和，不是并发时间；不能宣称Exact持续胜过强局部替换。[原始差值](exact_swap_runtime_e5_step1/four_pending_paired_sums.csv) |
| KVFlow Sequential物化及诊断 | Q，10Agent固定4096；20实际状态（末端及4032祖先）；Agent00局部无逐出/逐出/Cold | Stage1 **PARTIAL**；合法续接E4096；仅逐出末端后E4032，8 Token一致 | 原10对象CPU预算未对应20槽；恢复有效≠基线公平比较成立。[Stage1](kvflow_synthetic_stage1/)、[诊断](kvflow_native_recovery_diagnostic/) |
| C Shared/Branched K3 Pilot | Q，1Census+12策略/Pending严格Arm+4Cold=17成功生命周期；此前2失败尝试保留 | **READY**；16对16物理映射、每Arm移除13、12/12输出正确 | 真正验证全部祖先计账、FS/ST/WFC不同前沿；不是性能重复。[Pilot](c_shared_gpu_pilot_k3/) |
| C配对计时诊断与形状预热 | Q，20正式诊断（B/C×2×5）+4预热诊断；历史失败仍保留 | **已完成诊断，结论受仪表污染限制**；预热B CUDA30.769/196.462 ms，但TTFT仍约3.25/1.91秒 | 不能混合预热与未预热配对；原因包括编译和请求内重型Hash。[原始Arm及计时](c_pair_timing_gpu/) |
| 最新C正式GPU矩阵 | Q，FS/ST/WFC×4Pending×5独立生命周期，交错顺序；另外2门禁 | **PASS：60主臂+2门禁**；见§3.4完整表 | 同一个合成需求集合；加载冻结选集，无在线Select、无并发Goodput。[protocol](c_formal_perf_gpu/protocol.json)、[60份summary](c_formal_perf_gpu/)、[全部CSV](c_formal_perf_gpu/lifecycle_results.csv) |
| Kimi Linear组件控制与恢复 | K，Phase1/2各Smoke+3控制重复；Phase3 5对、每对独立A/B生命周期 | **PASS于对应合同**；Phase3 H16K固定，E16K/8K，两rank一致；恢复窗口差均值760.191 ms | Phase1旧逻辑回退8K实际可到14K，最终Phase3才消除中间祖先；非Kimi Phi校准或完整Strict控制泛化。[最终报告](rq5b_kimi_final_output/rq5b_frozen_20260923_090154/final_report.md)、[5对原始记录](rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json) |

### 2.4 不能遗漏的历史开销、未执行实验及数据审计

| 研究问题／实验 | 已有实际产物与状态 | 交接限制 |
|---|---|---|
| 旧24快照Runtime策略比较 | 24×3策略×3次=216生命周期、864串行请求；FS/LRU/Marconi协调后TTFT均值94.025/164.705/129.097 ms；**QUALIFIED**历史证据 | 逐快照改善均值43.058%/24.264%，不是均值之比；24胜与22胜2负；1输出Token、非当前Strict Cold认证。[原始运行](../rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/)、[P0核验](legacy_runtime_evidence_p0_audit/) |
| RQ6开销及批量RPC优化 | 原CPU573×3=1719记录，平均构造+分配0.314 ms；原72GPU控制均值8398.847 ms，批量版72次为629.827 ms；旧门禁 **PASS** | 批量协调后重型统一验证另计；不能挪为当前Strict在线控制耗时。[旧raw](rq6_overhead_output/rq6_overhead_20260909_061500/raw/per_epoch_overhead.jsonl)、[批量raw](rq6_batched_control_output/rq6e_20260911_085820/raw/per_epoch_overhead.jsonl) |
| 旧control-inclusive恢复Epoch | 24×3×3，216 record+216 evidence、864请求，本轮核对数量/状态/均值；**PASS旧合同** | FS2078.838、LRU2130.623、Marconi1888.372 ms，均值FS比Marconi慢；bootstrap CI均跨零。[raw runs](resume_epoch_e2e_output/resume_epoch_final_20260928/runs/)、[统计](resume_epoch_e2e_output/resume_epoch_final_20260928/analysis/summary.json) |
| 优化后control-inclusive复测 | 同样216/864，本轮核对原记录；**PASS旧合同** | FS2425.641、LRU2952.253、Marconi1875.991 ms；对LRU15胜9负，对Marconi8胜16负；不能只报请求恢复更快。[raw](control_path_optimization_output/control_20260929/optimized_formal/runs/)、[统计](control_path_optimization_output/control_20260929/optimized_formal/analysis/summary.json) |
| 低探针Serving-path资格 | **BLOCKED_BY_CORRECTNESS_GATE，0/216生命周期、0/864请求** | 仅末态Census不能发现中途丢失/换槽再恢复，缺完整被动事件链；不是性能实验PASS。[资格证据](serving_path_e2e_output/serving_20260929/) |
| E4/E5统一计时复核 | 离线读取54生命周期，54PASS、0一致性问题，无新GPU | T0/T1只保留派生ms，T2–T4有原始ns；不补造服务端/写盘细分。[验证JSON](e45_unified_timing/unified_timing_verification.json)、[逐行CSV](e45_unified_timing/unified_timing_rows.csv) |
| KVFlow真实数据接入 | 策略与CPU拒绝测试已存在；真实OpenHands/AgentX缺可追溯ASG/Fixed Prefix声明 | **BLOCKED于真实数据接入**；不得拿LCP或Workflow标签代替。随后Synthetic是新证据类型，不是修复原轨迹。[接入审计](kvflow_real_data_connection/)、[Policy](kvflow_inspired_retention_cpu/policy.py) |
| 早期KVFlow距离对照 | CPU受控3候选/3需求，STE1/3/5、K1；KVFlow J2218.544、FS/Marconi/Exact968.603 ms | **PASS于简化STE信号比较**；未证明后续完整Fixed Prefix合同，更不是官方系统复现。[记录](rq3_kvflow_controlled_20260907_123928/) |
| 数据源适用性：FreeInference/GATC等 | FreeInference报告有464 Session/62354请求样本，block表示非完整目标Token；GATC样本审计实际读取0 | **PARTIALLY_SUITABLE / BLOCKED**；只能结构参考，不能新增严格快照。不同阶段网络/权限失败说明不合并为同一次错误。[源数据审计](new_dataset_suitability_audit/)、[GATC](gatc_real_session_sample_audit/)、[真实结构](real_multiworkflow_structure_audit/) |

## 3. Verified Quantitative Results：核心定量结果

### 3.1 Recovery Semantics：H、E、G与实际Prefill必须分开

在已验证Strict边界，T是冻结Planning Target，E由准入且保留的候选位置最大值决定，G=T−E。实际请求输入长度可能大于T，因此实际Prefill通常为 `len(input_ids)−E`，**不等于G**；在Cold生命周期H=0时，日志H−E=0不表示没有全量Prefill。

- Qwen 32K实验固定H，G=0/1K/4K/8K/16K/32K的服务端TTFT均值依次32.495/64.454/210.838/411.194/804.369/1522.914 ms，每格5次；这不是CUDA计时。[原始30行](../motivation/artifacts/replay_cost_20260819/raw_runs.csv)
- R2-C1b-2实际输入4394，H4224、E2432、G1792，实际Prefill区间[2432,4394)，长度1962；原生COW源6→工作27返回且源/目标内容核对，8Token Cold精确一致。请求后合法新状态与历史组件破坏分开判断。[resume_trace](runtime_control_closure_r2c1b2/resume_trace.json)、[输出](runtime_control_closure_r2c1b2/output_correctness.json)、[状态](runtime_control_closure_r2c1b2/state_integrity.json)
- Kimi Phase3固定MLA16K，公共准备移除10/12/14K KDA，A保留16K、B移除16K。5对恢复窗口均值A71.749/B831.941 ms，差760.191 ms；这是两rank最大match至单输出完成区间，不是单纯CUDA或整个控制过程。[paired_measurements](rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json)

### 3.2 冻结成本模型：精度、单调性及适用范围

部署公式：`Phi(G,T)=37.828150 g + 0.345974143 g t − 0.156201917 g²`，`g=G/1024, t=T/1024`，输出modeled ms。[源码](../flowstate/recovery_model.py)、[配置](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/config.json)

- 校准20聚合点，来源为位置矩阵及长Gap剖面；T为32K–128K。独立测试T=49152/81920/114688，每T五个G（含0），各2预热+12测量，共210原始记录。MAPE排除零G基线，在12个非零聚合点上计算，而不是把210行当210独立测试场景。
- Gap-only M0：MAPE15.291%、MAE287.514 ms；Position-aware M2：MAPE2.682%、MAE31.180 ms，最大相对误差8.179%。[逐点预测](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv)、[模型选择](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/model_selection.json)
- 固定T单调、非负、Phi(0,T)=0在冻结结构网格147/147通过。代码定义域0≤G≤T≤131072；定义域不等于全部位置均经实测校准，短T的数K工作负载属于外推；Kimi没有重拟合该模型。[独立审计](rq2_model_audit_output/20260909_055110/final_report.md)
- E5-0切换模型后，Exact/Greedy/1-Swap各20/20同集；Standalone19/20同集。不能由预测精度优势推导实际分配收益。[80行对照](strong_baseline_position_audit_e5_step0/position_vs_gap.csv)

### 3.3 分配：哪些是最优证书，哪些只是局部最优

| 数据族 | Exact / Greedy / 1-Swap / WFC达到Exact的条件数 | 结构归因与证书 |
|---|---|---|
| 新Strict20，5快照×4K | 20 / 5 / 18 / 5 | 各n=7，枚举至K4仅99子集；冻结独立枚举与新生产Exact均匹配。Greedy15个失败全从公共祖先首步不可撤回开始 |
| 旧B4-C16，g001–004×4K | Exact16，Greedy12；不混入新20 | 唯一跨Pending共享B1，其他专属；B5-C0两分支Top-K精确化简16/16匹配，非通用算法定理 |
| 旧573 Formal | 573个旧模型最优证书 | OpenHands504可分支持；AgentX66全前沿可达+3可分。没有无界149候选Exact枚举；无Strict Runtime认证 |
| C/K3，16候选 | **无Exact结果/无最优性证书**；Greedy=1-Swap，0次交换 | 在线Exact门因n>12拒绝，即使子集总数697；不能把准入限制称计算不可解。Greedy J154.349、WFC471.874、Standalone348.234 modeled ms |

证据：[Strict20逐选集/J/E/G/计时](greedy_swap_structure_e3_step1/condition_results.csv)、[结构汇总](paper_allocation_structure_audit.md)、[最新微基准](paper_solver_tradeoff_summary.csv)。新20条 Greedy对Standalone4胜16平，差异在w3/K3、w3/K4、w4/K4、B5-C4/K4，均是静态重复覆盖。对PCB/LDR各15胜5负；它们在K=K_min的5条达到Exact，说明复杂联合选择并非所有预算都必要。

w1/K3：Exact `{N18,N25,N27}`、J21.953375；固定首改善1-Swap `{N21,N22,N24}`、J31.817384。w2/K3也是局部陷阱；18/20最优不能提升为局部搜索保证。B5-C4/K2：Greedy `{N20,N18}` J130.803，Exact `{N18,N27}` J48.229；K1最优公共祖先在K2最优中消失。上述N是**冻结逻辑标签**，不是可跨Engine复用的物理Node/Slot。

CPU最新12480条计时的Strict20全局中位：Greedy0.0972、Exact0.2238、1-Swap0.2209、WFC0.0270 ms。Exact P95=0.5067 ms、最大3.14 ms，因此不能把“约0.5 ms级中位”写成最坏时延上限。输入JSON读取中位5.26 ms与纯求解是不同计时边界；微基准没有网络、GPU控制及并发承诺。[raw](paper_solver_tradeoff_raw.csv)、[环境/合同](paper_solver_tradeoff_gate.json)

### 3.4 最新60臂GPU：只用主矩阵，不混门禁

本表直接聚合60份原始 `m{1..5}{A..D}_{FS|SA|WFC}/run/summary.json`；每格5个独立生命周期，中位数ms。FS＝冻结FlowState-Greedy，ST＝StandaloneTopK。各次都重新物化同一16候选边界，保留3、驱逐13；三策略均使用同一Strict控制与8Token确定性生成。H/E/G/Prefill为各格相同的实际值。

| Pending；H | 策略 | E / G / Prefill Tokens | CUDA Prefill | TTFT | Request Wall |
|---|---|---|---:|---:|---:|
| A；4096 | FS | 0 / 4096 / 4128 | 151.354 | 159.312 | 286.878 |
| A；4096 | ST | 0 / 4096 / 4128 | 150.114 | 157.763 | 283.099 |
| A；4096 | WFC | 2048 / 2048 / 2080 | 79.718 | 87.322 | 216.825 |
| B；5120 | FS | 5120 / 0 / 32 | 30.808 | 37.630 | 167.446 |
| B；5120 | ST | 0 / 5120 / 5152 | 193.225 | 203.660 | 328.427 |
| B；5120 | WFC | 2048 / 3072 / 3104 | 115.732 | 123.509 | 251.626 |
| C；6144 | FS | 6144 / 0 / 32 | 34.820 | 42.737 | 171.790 |
| C；6144 | ST | 6144 / 0 / 32 | 30.454 | 37.623 | 162.591 |
| C；6144 | WFC | 3072 / 3072 / 3104 | 116.224 | 126.102 | 249.649 |
| D；7168 | FS | 7168 / 0 / 32 | 50.123 | 59.174 | 273.830 |
| D；7168 | ST | 7168 / 0 / 32 | 33.884 | 41.186 | 174.191 |
| D；7168 | WFC | 3072 / 4096 / 4128 | 154.627 | 163.213 | 289.892 |

来源：[主矩阵执行账本](c_formal_perf_gpu/arm_status_main.json)、[原始摘要示例](c_formal_perf_gpu/m1B_FS/run/summary.json)、[逐次CSV](c_formal_perf_gpu/lifecycle_results.csv)。CSV包括门禁，使用时必须按 `name.startswith('m')` 取正式60行。

**集合性能的合法聚合口径：**每轮对四个独立Pending取等权均值，再在5轮间报告中位和配对差；不是一个连续Workflow的完成时间，更不是吞吐。

| 指标 | FS | ST | WFC |
|---|---:|---:|---:|
| 四Pending等权TTFT，5轮中位 | 77.829 | 111.960 | 125.262 |
| 四Pending等权CUDA Prefill，5轮中位 | 69.396 | 103.910 | 116.656 |
| 等权TTFT五轮范围 | 69.961–79.418 | 106.524–113.530 | 124.522–125.377 |

FS−ST等权TTFT五轮差：**−33.408、−28.690、−40.180、−42.250、−34.112 ms**；FS−WFC：**−46.926、−46.688、−53.594、−55.301、−45.959 ms**。10个配对均负，支持此固定需求集合的请求级方向；不是统计显著性或跨工作负载泛化。[已有重聚合及限制](c_formal_perf_gpu/paper_pending_aggregate_audit.md)

取舍不能省略：A上FS−WFC配对中位 **+71.690 ms**；B上FS−ST **−163.144 ms**（五次均负），差异来自B状态被静态冗余D祖先挤出；C/D同前沿但FS−ST配对中位仍 **+6.623/+21.072 ms**。17/60臂decode CUDA>150 ms，分布FS9/ST6/WFC2。其时钟/功耗成因没有直接证据；同前沿不等于已完成统计等价检验，不可把正差删去或强制记Tie。

### 3.5 请求性能不等于控制含入在线收益

| 计时边界 | 已有直接结果 | 能否用于在线净收益 |
|---|---|---|
| 正式C请求TTFT | 发送→首Token；窗口内去掉重型Hash/落盘，保留Native Match/COW；形状预热后被监测Triton新增0 | 支持受控请求级比较；不代表未预热生产流量 |
| 正式C Control RPC | 主矩阵各20臂中位：FS11318.512、ST11245.886、WFC11300.036 ms | 含真实安全检查和审计；不能静默删除。FS−ST每轮等权控制差均正：+114.327/+261.458/+261.298/+16.888/+834.911 ms |
| C控制分段 | FS中位：观察2425.080、映射0.587、Epoch/冻结Plan认证159.504、Apply-time1237.969、前FA审计939.770、Reconcile869.085、后审计1844.382 ms | 父区间与子区间不能相加；Reconcile含真实Eviction及后检，不能再加Eviction。残差未完整计时归因 |
| C `epoch_wall_ms` | 各策略20臂中位FS57563.130、ST57294.335、WFC57146.289 ms | 含历史/预热/审计等实验阶段；不是连续在线E2E。四请求独立初始化，更不能摊作一次apply |
| E4/E5统一Control-to-completion | 同一生命周期T4−T0已可还原；E4均值约7469.333 ms，其中请求6411.764、RPC814.493、间隙243.077 | 是实测实验窗口，不能改名纯Serving；首Decode长区间及RPC残差未完全归因 |
| “约1.03秒必要控制、约7次B回本” | 来源是选取认证+协调两个分段及163ms请求差的除法 | **算术估算，非实测净收益**；没有可直接替代全部观察/新鲜度检查的轻量生产路径证明 |

证据：[C控制原始summary](c_formal_perf_gpu/m1B_FS/run/summary.json)、[真实调用](c_formal_perf_gpu/control.py)、[E4边界审计](e4_control_timing_audit/final_report.md)、[统一验证](e45_unified_timing/unified_timing_verification.json)。CUDA Event是设备时间线区间，可能含host提交空隙，不等于独立kernel总时间；TPOT/Decode/Prefill与Request Wall是嵌套指标，不能重复加总。

## 4. Runtime and Solver Status：实现与认证状态

| 能力 | 当前真实入口／合同 | 实现和认证等级 |
|---|---|---|
| Legacy选择 | `GlobalOptimizer.select→_select_candidates`，经旧Workflow/Lineage谓词评分 | **已实现**；保留历史行为。不能接收Strict对象或代表严格跨Workflow兼容 |
| Strict混合选择 | `plan_strict→epoch.validate→_exact_admission→Exact或原Greedy→StrictAllocationPlan` | **已进生产源码、已通过CPU门禁**；20/20新Strict Exact一致，C/K3回退一致；新入口GPU验证**未完成** |
| 实验Exact在线控制 | B5-D3、D4a/b、E1等按外部参数指定Exact；经生产 `reconcile_strict` | **独立实验真实GPU验证**；不是新Hybrid自动路由已上线 |
| Strict Epoch | [strict_epoch.py](../flowstate/strict_epoch.py)：ID唯一、全域同Epoch/配置、Token摘要、完整n×m、14证据条件、等大小、UNKNOWN拒绝 | **已实现+CPU验证+受控采集GPU证据**；它校验传入证据一致性，不自行观测GPU，不保证跨时刻新鲜 |
| 前沿 | `StrictAllocationEpoch.frontiers`仅使用EPOCH_ADMISSIBLE边，`max(L,0)` | 不调用旧workflow过滤；可保留公式需候选闭包、无外部Recurrent基底、补集完全驱逐等前提 |
| Plan合法性 | [strict_control.validate_plan](../flowstate/strict_control.py)重算T/E/G/J、验证摘要、预算及完整补集 | **已实现+CPU/GPU路径使用**；合法性不等于最优性证书 |
| 物理绑定 | `bind`逐Token唯一匹配当前节点；驻留Recurrent域必须等于候选域；槽与组件切片无别名 | **已实现+受控Q GPU验证**；不复用历史Node/Slot，不支持任意共享底层对象/多槽成本 |
| 连续隔离 | [R2-C0桥接](runtime_control_closure_r2c0/bridge.py)：单Scheduler同步调用、线程归属、空闲/无传输/禁异步功能 | **独立桥接真实GPU验证**；不是任意跨RPC租约。generation仍UNKNOWN，靠连续窗口与重新观测保证本轮新鲜 |
| 物理协调 | `StateController.reconcile_strict→StrictReconciler→StrictRuntimeBridge→SGLangAdapter` | 不重选；bind/preflight、每项前后postcheck、最终驻留恰为选集；失败记录账本+invalidate、停止，不回滚不重试。**Q受控GPU已验证** |
| Native移除/COW | [sglang.py](../flowstate/adapters/sglang.py)安全门禁后原生Recurrent-only primitive；请求匹配/复制由原生路径执行 | Q TP1有完整组件与Cold案例；Kimi TP2是另一套控制验证，不能声称通用Strict桥接TP2已经认证 |
| Heavy Audit | R2-C0/B5-D4a桥接，全树FA/逐槽内容Hash、Census、参考与后验 | 独立实验已验证；高开销是实测，不是线上成本最优实现 |
| Lightweight Control | [E1 control/light_capture/bridge](lightweight_runtime_e1/)；必要元数据仍遍历、引用/别名/记账/Apply-time不关闭，内容审计另RPC | **E1/E4/E5受控GPU验证**；不等于所有Runner已统一。无完整mutation_generation/被动变更日志证明 |
| 正式C Select | [control.py](c_formal_perf_gpu/control.py)：读取冻结 `policy_plans.json→from_frozen/prepare/finish→reconcile_strict` | **在线认证与落实，离线选择**；`strict_epoch_authentication_and_plan`不是在线选择耗时 |

### 当前Hybrid门禁的可追溯范围

[paper_hybrid_solver_integration_gate.json](paper_hybrid_solver_integration_gate.json)保存20条Exact结果、1条C回退、3类拒绝和6个规模边界结果；C回退J154.348955616，与冻结选集及E=(0,5120,6144,7168)一致。集成报告另称122 tests+16 subtests通过；本轮核实该报告及JSON，**未重新运行测试，未据报告补造独立测试日志**。

当前源码身份（SHA-256，交接时读取）：

| 文件 | 摘要 |
|---|---|
| `flowstate/optimizer.py` | `7dfa9aec307d1178d79d79ced0b5d42b0f6574754194c58d84db300c74e48176` |
| `flowstate/strict_epoch.py` | `5513f2703dd88e8f292c348baa07e198f30649c23c2df1a5699fe0066a1eb4a7` |
| `flowstate/strict_control.py` | `5ffeeaeef14606f3f1d280f41fa26a575b2f10c66ea5db28832939481a3ebab3` |
| `flowstate/recovery_model.py` | `f3fe216592ad62c26e5bf7936f907823745942f7f34b483b8dfbc2fbd8fda1f5` |

工作区已有 `optimizer.py` 和 `tests/test_strict_epoch.py` 的修改，本轮未产生这些修改。旧GPU执行锁记录的是旧代码身份；不能仅因当前函数名相同就声称新Hybrid在旧GPU运行中已执行。

## 5. Baseline Coverage：公平性与比较边界

| 方法 | 真实实现／可见信息 | 已比较范围 | 不能宣称 |
|---|---|---|---|
| LRU | [受控保留实现](controlled_multiworkflow_v1/policies.py)，last-access、creation及ID；不同合成轨道访问来自声明事件 | 旧573 CPU、旧24及旧E2E GPU、合成A/B/C CPU | 非完整官方缓存系统；新Strict20缺合法元数据时不能补分数 |
| LFU | [冻结适配](rq3_frozen_snapshot_evaluator.py)：频率降序、最近访问、ID | 旧573 CPU | 当前正式C GPU无LFU；不是完整原生全系统复现 |
| Marconi Adaptation | [sota_policies](sota_policies.py)、[metadata](sota_metadata.py)：归一recency+α归一FLOP/memory，α=1 | 旧573、B4-C、合成CPU；旧24/旧E2E GPU | 不是完整Marconi。旧增量span与合成自身长度proxy不相同；新Strict20输入不足UNVERIFIED；不纳入C正式GPU |
| KVFlow-inspired | [policy.py](kvflow_inspired_retention_cpu/policy.py)：ASG Ready/AND/OR STE、固定前缀祖先传播、动态后缀淘汰优先、ID同分 | 10Agent及A/B/C Synthetic CPU；Stage1局部物化/恢复 | 非官方完整服务系统，无Prefetch/Scheduler。尚无与FS在相同合法新Strict负载的正式KVFlow GPU性能对照 |
| WFC | [冻结WFC](p1_strong_baseline_validation/wfc.py)：动态推进Pending数量优先、Token推进其次；不用Phi | 旧573、B4-C、新Strict20、合成CPU；C正式GPU | 内部强简单策略，不是外部SOTA；Strict20持平不证明一般评分等价 |
| PCB/LDR | [冻结实现](p1_strong_baseline_validation/)：各Pending最佳收益推荐／最深位置，确定性去重排序 | 旧573、B4-C、新Strict20、A6 CPU | 未完成当前C正式GPU对照；不因旧573同集而声称新共享紧预算也最优 |
| StandaloneTopK | 静态独立总收益排序，同Phi/候选/预算，不更新覆盖 | 旧573、新Strict20、合成C补充CPU；C正式GPU | 内部机制对照；被深候选遮蔽的祖先仍占预算，是与Greedy的主要区别 |
| Exact / 1-Swap | 独立枚举Oracle／固定首改善局部替换；现Exact另已进核心CPU入口 | Strict20 CPU、E5 Exact/Swap GPU、B5/E1/E4 Exact/Greedy GPU | E5 Swap加载离线冻结选集；没有通用在线Swap，也没有C/K3 Exact证书 |

所有有效分配比较统一候选、已知Pending、准入关系、预算和评价J；但**这些是分数据族成立的合同**，不能跨旧Formal/新Strict/合成输入合并总体胜率。三种C正式策略同硬件/独立初态/统一控制协议；逻辑K=3确有13项物理移除，但物理池28槽仍有余量，不是HBM紧张实验。

关键胜平负（FS-Greedy相对对照，顺序W/T/L）：

| 数据族与分母 | LRU | LFU | Marconi | WFC | Standalone | PCB、LDR各自 |
|---|---|---|---|---|---|---|
| 旧OpenHands504，modeled J | 306/198/0 | 504/0/0 | 386/118/0 | 0/504/0 | 144/360/0 | 0/504/0 |
| 旧AgentX69，modeled J | 16/53/0 | 12/57/0 | 16/53/0 | 0/69/0 | 2/67/0 | 0/69/0 |
| 新Strict20，modeled J | 未比较 | 未比较 | UNVERIFIED | 0/20/0 | 4/16/0 | 15/0/5 |
| 合成A/B/C共6条件，当前Pending J | 6/0/0 | 未比较 | 6/0/0 | 2/4/0 | 不并入原五策略表 | 未比较 |

合成6条件FS对KVFlow为4胜2平；并不证明所有未来调用最优，C/K3未来静态评价FS154.349、LRU0 modeled ms，存在反向取舍。[完整CPU结果](kvflow_multiworkflow_cpu_main/cpu_comparison.csv)。正式C三策略不含LRU/Marconi/KVFlow，因此不能把其请求级优势写作已打败这些SOTA系统。

## 6. Claim-to-Evidence Matrix：论文可保留、需收窄及不支持

| 论文主张 | 支持实验与路径 | 证据等级 | 未解决问题／交接结论 |
|---|---|---|---|
| Physical Hit不等于Executable Hit | [32K](../motivation/artifacts/replay_cost_20260819/raw_runs.csv)、[A5](submission_p0_lmcache_recovery/runtime_evidence.json)、[Kimi](rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json) | **STRONG，可保留** | 限定相应模型/Cache配置与受控前沿，不外推所有Hybrid实现 |
| Position-aware预测更准确 | [heldout_predictions](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv) | **STRONG，可保留预测主张** | 短上下文与跨模型外推未验证；分配决策收益不能由误差自动推出 |
| Position-aware带来实际分配/GPU收益 | [模型切换](strong_baseline_position_audit_e5_step0/position_vs_gap.csv) | **UNSUPPORTED于当前Exact/Greedy20条** | 20/20同集；Standalone唯一变化无独立GPU优势认证 |
| 集合估值可以胜过独立排序 | [Strict20](greedy_swap_structure_e3_step1/condition_results.csv)、[C正式原始](c_formal_perf_gpu/) | **STRONG于已测条件，可保留** | 同链冗余→B恢复缺失→CUDA/TTFT增加有对应证据；不是所有工作负载都需要复杂分配 |
| Greedy优于强简单策略、普遍近优 | [旧573](p1_strong_baseline_validation/per_case_results.jsonl)、[Strict20](greedy_swap_structure_e3_step1/condition_results.csv) | **UNSUPPORTED广泛表述** | WFC20/20平、Greedy15/20不最优；应写适用域/取舍，经典次模保证不是新数学贡献 |
| Hybrid Solver已是生产核心能力 | [当前optimizer](../flowstate/optimizer.py)、[CPU门禁](paper_hybrid_solver_integration_gate.json) | **STRONG于实现与CPU；GPU为PARTIAL** | 新自动路由→实时Epoch→协调→请求尚未以同一入口完成GPU验证 |
| Recurrent-only协调安全有效 | [C1a](runtime_control_closure_r2c1a/physical_postcheck.json)、[b2](runtime_control_closure_r2c1b2/state_integrity.json)、[C Pilot/正式](c_formal_perf_gpu/) | **STRONG于受控Q配置** | 无跨RPC长期保护、并发活动请求或多槽/别名容量泛化；Kimi单独认证 |
| 共享预算取舍改善实际请求性能 | [C60臂](c_formal_perf_gpu/lifecycle_results.csv)、[等权分析](c_formal_perf_gpu/paper_pending_aggregate_audit.md) | **STRONG于此合成需求集合；总体泛化PARTIAL** | 每轮四独立Pending等权全量改善；A对WFC退化，n=5不作生产显著性结论 |
| Exact优于1-Swap的实际联合计算稳定成立 | [E5三组总和](exact_swap_runtime_e5_step1/four_pending_paired_sums.csv) | **PARTIAL，必须收窄** | 两负一正，均值−4.854 ms；不支持“持续更低”，更不是并发吞吐 |
| 完整在线控制含入净收益已实测 | [C控制](c_formal_perf_gpu/)、[旧E2E负例](resume_epoch_e2e_output/resume_epoch_final_20260928/analysis/summary.json)、[Serving门禁](serving_path_e2e_output/serving_20260929/) | **UNSUPPORTED** | 冻结选集、重审计和预热、非连续请求；7次回本只是估算 |
| 全面优于官方SOTA | [旧24审计](legacy_runtime_evidence_p0_audit/)、[KVFlow适配边界](kvflow_real_data_connection/) | **UNSUPPORTED，需收窄** | Marconi/KVFlow是适配，输入域不同，当前正式GPU仅FS/ST/WFC |

## 7. Open Gaps：未闭合问题及冲突处理

### 7.1 最影响可信度的三个问题

1. **在线系统身份与性能口径尚未闭合。**新Hybrid主入口只有CPU认证；C正式GPU执行冻结计划；完整观察/绑定/新鲜度的轻量替代未统一。把既有请求级收益写成在线Select及控制含入E2E收益，将跨越现有证据。
2. **范围、成本模型与基线的外推风险。**单一合成C需求集合不能代表自然并发工作流；Strict20多预算不独立；短T Phi未校准；Marconi/KVFlow不是完整系统复现，旧AgentX70条Formal-only边有Token不一致，旧573不能升级Strict。
3. **汇总与主张需要严格版本/分母管理。**门禁混入单格统计、快慢模式的事后解释、报告把估算写成净收益，以及Solver集成前后描述冲突，都可能让正确的局部实测被扩张成错误结论。下面记录差异，不覆盖旧报告。

### 7.2 冲突清单与本次采用的证据

| 冲突来源 | 具体差异 | 本报告处理 |
|---|---|---|
| [10月22日简报](../FLOWSTATE_PAPER_EVIDENCE_BRIEF.md) vs 当前完成账本 | 当时14/60、Greedy-only；现在60/60、Hybrid已进核心 | 正常时间演进；旧简报保留原截点，不当最新状态 |
| [solver_integration_audit](paper_solver_integration_audit.md)、[semantics_audit](paper_formulation_runtime_semantics_audit.md) vs [23日后续集成](paper_hybrid_solver_integration_gate.md) | 前两者写主库仅Greedy，后续代码增加Exact；tradeoff报告又曾提前把独立planner称生产默认 | 当前以代码与最新CPU门禁为准；旧GPU能力仍以当时入口为准 |
| [C final_report](c_formal_perf_gpu/final_report.md) vs [后续aggregate audit](c_formal_perf_gpu/paper_pending_aggregate_audit.md) | 前者称净收益为正/约7次回本；后者明确为算术估算、完整在线E2E未测 | 采用后者的范围限定；保留完整RPC，不能据挑选两个分段扣除其余成本 |
| [C pending_policy_matrix](c_formal_perf_gpu/pending_policy_matrix.csv)、[finalize.py](c_formal_perf_gpu/finalize.py) | 聚合 `gate_arms+main_arms`，C_B FS/ST各n=6；报告当成五轮数字37.2/202.5 ms | 本文只取m*主矩阵，n=5：37.630/203.660 ms；配对先筛主矩阵，受CSV一位小数舍入影响，正文用原始精度重聚合 |
| [C pair早期汇总](c_pair_timing_gpu/pair_analysis.csv) | 前两条B来自w1/w2预热Arm，后面来自p03–p05未预热 | 不直接引用为五对同协议；原始p*和w*分组分别可追溯 |
| C同前沿“平局”、机器慢模式 | 全量C/D FS−ST中位为正；报告用快模式子集解释。功耗/时钟原因未直接测量 | 报全量差和17/60分布；不剔除、不证明统计等价、不将相关性当成确定原因 |
| [四Pending汇总报告](c_formal_perf_gpu/paper_pending_aggregate_audit.md) | 表中FS Prefill最大写69.8，但其逐轮第5轮约70.5；正文单格也继承门禁中位 | 原始主矩阵五轮等权Prefill最大70.418 ms；不能把单格中位之和当作配对等权中位 |
| 旧Kimi Phase1 vs Phase3 | 逻辑16K→8K，实际尚有14K祖先 | Phase1只保留组件控制结论；8K回退引用Phase3实际匹配 |
| GATC D1/D2报告 | D2记录OS网络阻断；后续D1更新有401受限访问与FreeInference样本 | 不合并成一次失败；共同可确认是无GATC样本级恢复证据。未用这些报告创造新结果 |

### 7.3 其他明确未完成能力

- 通用在线Agent Ready/Fork/Join事件源、并发Admission下连续状态变化追踪、长期Pin/Lease、可靠Slot generation覆盖、非等大小/多槽/别名物理预算：尚未形成当前完整实现与GPU认证。
- Tree DP仅条件性分析；未实现。C/K3没有Exact最优性证书；不能从Greedy与1-Swap持平推导全局最优。
- 低审计请求计时已有直接数据；完整生产控制成本、同一Epoch多请求摊销、无额外控制策略的净收益、连续工作流Goodput均没有相应实验闭环。
- “重型Hash可移出请求窗口”有受控正确性证据；“所有观察及Apply-time检查可删去而不损失新鲜度”没有证明。这是两条不同主张。

## 8. Evidence Index：接管时的读取顺序与关键原始产物

以下是索引，不是待执行命令；**不要通过运行历史finalize或selector来读取结果**。

1. **当前核心与门禁**：`flowstate/optimizer.py`、`strict_epoch.py`、`strict_control.py`、`controller.py`、`adapters/sglang.py`；[Hybrid gate JSON](paper_hybrid_solver_integration_gate.json)及[报告](paper_hybrid_solver_integration_gate.md)。GPU旧代码身份以各 `implementation_lock*.json` 为准。
2. **最新GPU完整证据**：[protocol](c_formal_perf_gpu/protocol.json)、[implementation_lock_main](c_formal_perf_gpu/implementation_lock_main.json)、[source_identity](c_formal_perf_gpu/source_identity.json)、[policy_plans](c_formal_perf_gpu/policy_plans.json)、[main账本](c_formal_perf_gpu/arm_status_main.json)。每个 `m*/run/` 中的 `summary.json`、`reconciliation.json`、Native/请求记录与 `m*_cleanup.json`；`cold_reference_C_{A..D}.json`。原始文件优先于混门禁聚合表。
3. **Strict分配与Oracle**：[E2](independent_runtime_allocation_e2/)、[B5-C3物理Snapshot](multishare_runtime_gate_b5c3/strict_epoch_snapshot.json)、[B5-C4](multishare_allocation_gate_b5c4/)、[E3全表](greedy_swap_structure_e3_step1/condition_results.csv)、[solver microbenchmark](paper_solver_tradeoff_raw.csv)、[正确性门禁](paper_solver_tradeoff_gate.json)。
4. **成本模型与动机**：[32K raw](../motivation/artifacts/replay_cost_20260819/raw_runs.csv)、[model config](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/config.json)、`calibration_points.csv / heldout_raw.csv / heldout_predictions.csv / model_selection.json`、[独立审计](rq2_model_audit_output/20260909_055110/)。
5. **Strict闭环的阶段性证据**：R2-A/B CPU接口、R2-C0只读物理桥接、C1a真实移除、C1b-1失败、[C1b-2真实Req及Cold](runtime_control_closure_r2c1b2/)。失败日志不是可丢弃的旧产物；修复后新运行不能覆盖旧失败。
6. **控制计时与独立来源恢复**：[E1](lightweight_runtime_e1/)、[E4成功run01](native_resume_correctness_e4_phase1a_run01/)、[E4性能](native_resume_performance_e4_phase1b/)、[E5 Exact/Swap](exact_swap_runtime_e5_step1/)、[统一54行验证](e45_unified_timing/)。各实验Control-to-completion端点不能与C的epoch_wall互换。
7. **历史证据及限制**：[旧24 audit](legacy_runtime_evidence_p0_audit/)、[旧573](p1_strong_baseline_validation/)、[AgentX兼容/容量审计](gate1_step3_compatibility_retention_contract_audit/)、[旧E2E](resume_epoch_e2e_output/resume_epoch_final_20260928/)、[优化后E2E](control_path_optimization_output/control_20260929/optimized_formal/)、[未执行Serving资格](serving_path_e2e_output/serving_20260929/)。
8. **Synthetic与资源账本**：[KVFlow Policy](kvflow_inspired_retention_cpu/policy.py)、[真实数据接入失败](kvflow_real_data_connection/)、[Sequential](kvflow_synthetic_stage1/)、[4096/4032诊断](kvflow_native_recovery_diagnostic/)、[Multi-Workflow协议](kvflow_multiworkflow_cpu_main/protocol.json)、[候选账本](kvflow_multiworkflow_cpu_main/candidate_resource_ledger.csv)、[C Pilot](c_shared_gpu_pilot_k3/)。
9. **跨模型**：[Kimi冻结报告](rq5b_kimi_final_output/rq5b_frozen_20260923_090154/)、[Phase3配对原始记录](rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json)。不要用Qwen Phi预测替代Kimi实测或把旧两候选逻辑前沿冒充全树前沿。

### 最终交接回答

**Q1：哪些可靠证据可用于ICDE？** 可保留受控Physical/Executable Hit分离、Qwen长上下文Position-aware预测精度、Strict20完整分配与最优性对照、部分真实Recurrent-only/COW/Cold闭环，以及C/K3正式60臂中固定需求集合的请求级计算/TTFT改善。Hybrid Solver已实现并通过CPU门禁；Kimi补充组件与恢复语义验证。各自限定来源、样本、模型与计时边界。

**Q2：哪些主张仍缺证据？** 新Hybrid生产入口的GPU在线闭环；完整在线控制后净收益/并发Goodput；Position-aware对Exact/Greedy选择的独立收益；Exact稳定优于1-Swap的GPU总计算；通用多Workflow与跨模型性能泛化；全面胜过官方Marconi/KVFlow系统。

**Q3：最影响论文可信度的三个问题？** ①冻结选集/新Hybrid/独立实验入口混称同一个在线系统；②受控合成范围、短上下文Phi外推及适配基线被过度泛化；③汇总分母与计时边界不统一，把门禁混入重复、把估算或事后噪声解释写成实测净收益。三者的具体冲突和证据见§7，现有负面记录均保留。

本报告完成即停止；本轮只新增此交接文件，未修改论文、实现、冻结输入或实验结果。
