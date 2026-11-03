# FlowState 论文主张—实验证据一致性审计

日期：2026-10-29。**总体判定：PARTIALLY_SUPPORTED。** 严格恢复语义、固定 Epoch 的分配问题、小规模混合求解及受控物理协调具有证据；请求级性能仅在明确的负载和计时窗口内成立；稳定的完整在线控制周期净收益尚未成立。

## 1. 正式稿定位与本报告的审查边界

**当前正式稿的位置、版本和逐句内容：UNKNOWN。** 本轮检索项目（含隐藏文件、排除Git内部）的 `.tex/.docx/.typ`、正文相关Markdown，以及 `/home/wjg/code` 下四层和 `/home/wjg` 下三层相应文档，未找到可确认的论文正文。根目录 `FLOWSTATE_PAPER_EVIDENCE_BRIEF.md`、`evaluation/paper_experiment_handover_20261024.md` 是证据总结，不是正式稿。当前聊天未附可读取的正文文件或打开的文档。已向用户请求正式稿路径，报告生成时尚未取得。

因此本报告完成**实验依据、代码及候选主张的审计**，不能声称完成当前 Introduction、III–V 和实验章节的逐句认证。下表的章节是应核对的位置，不是已定位的论文原句。正文问题分为“现存报告/代码中的已确认冲突”和“取得正式稿后需核对的表述”；不将旧讨论或历史建议冒充最新正文。

本轮只新增本报告。没有启动GPU、Docker、模型、回放、策略求解、测试或正式验证器；只读取冻结产物并在内存复算必要统计。P0-4为其他任务的活动目录，以下状态是本次读取截点，不代表未来运行状态。

## 2. 总体裁决

|主张范围|证据等级|当前可写结论|
|---|---|---|
|Physical FA命中不等于Executable命中|已验证，限定受控配置|固定FA、移除Recurrent可使真实恢复前沿下降并产生重算|
|Position-aware模型预测能力|有限条件下支持|长上下文独立held-out上误差优于预注册Gap-only；不能搬用该精度到短上下文|
|Position-aware改变选择、带来独立分配收益|尚未验证；已有同集结果限制该主张|主要20条件及C/K3未观察到Greedy/Exact模型切换选集变化|
|集合收益与独立排序的区别|已验证于固定数学问题与有限Strict实例|存在重叠收益、静态冗余、非嵌套最优集合；不支持Greedy普遍胜简单规则|
|Exact–Greedy生产接线及受控恢复|已验证于已完成CPU/G1门禁|严格入口门内Exact、超规模Greedy；两路径8个GPU正确性臂通过|
|Recurrent-only协调、Native Resume/COW/Cold|已验证于实际认证的Arm|不扩大为全矩阵逐边认证、一般并发Serving或张量跨Workflow血缘|
|相对MC/KV适配策略的请求TTFT|有限条件下支持|同一合成C/K3冻结选集、五轮、60主臂的请求级配对结果|
|三策略完整在线周期稳定净收益|尚未验证，另有方向相反的完整周期结果|等待P0-4真实GPU闭环、边界核验及配对统计；不能由TTFT推导|
|原生真实Agent工具闭环、自然Ready/Fork|尚未验证|OpenHands为真实来源受控历史重放；C/K3为合成声明工作负载|

## 3. 主张—证据对照表

等级“已验证”只对该行限定对象成立，不表示系统整体或一般性主张已证明。J均为模型化额外恢复成本，CPU耗时与GPU时间分列。

|编号；应核对章节与主张|实验类型、规模、实际结果|直接证据|边界与建议|
|---|---|---|---|
|C01；Introduction/III：FA物理命中不能保证可执行恢复|真实GPU因果对照；Qwen3.5-9B/SGLang0.5.17/H100 PCIe。5个Fork配对：H=32768不变，E从32768→0，G从0→32768；服务端TTFT中位38.332→1507.858 ms。实际EXTEND为64→32832|[WP3A原始20行](../motivation/artifacts/wp3a_direct_eviction_20260819/raw_runs.csv)、[逐配对效应](../motivation/artifacts/wp3a_direct_eviction_20260819/paired_effects.csv)、[合同与结果](../motivation/artifacts/wp3a_direct_eviction_20260819/findings.md)|**已验证**。保留；这是特定配置下受控干预，不是所有Hybrid模型、所有Cache命中均重算。不能把G直接写成全部Prefill Tokens|
|C02；III：E、G、J定义与严格执行关系|代码：`frontiers`仅用显式EPOCH_ADMISSIBLE边，E=max合法L，G=T−E；完整矩阵，UNKNOWN拒绝。物理域闭合、补集真实驱逐由bind/postcheck强制|[strict_epoch.py](../flowstate/strict_epoch.py:99)、[strict_control.py](../flowstate/strict_control.py:141)、[语义审计](paper_formulation_runtime_semantics_audit.md)|**有限条件下支持**。保留公式并补静止边界、候选域闭合、等大小、无额外驻留Recurrent基础状态、固定T/配置；不能用Workflow祖先替代Token证据|
|C03；IV/实验：Position-aware预测更准确|真实GPU剖面及离线拟合；20个校准聚合点，12个非零gap held-out点，held-out不拟合。直接重算MAPE：M0=15.2912749%，M2=2.6824804%。目标校准32K–128K，held-out48K/80K/112K|[配置](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/config.json)、[原始held-out](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_raw.csv)、[12点预测](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/heldout_predictions.csv)、[选择记录](runtime_artifacts/recovery_model_freeze_20260826_154235_266020/model_selection.json)|**有限条件下支持**。保留该划分和聚合口径；12点不是12工作负载，不能称全部请求MAPE。不能把额外恢复成本Phi当完整TTFT|
|C04；Introduction/IV：位置感知项产生独立决策/性能优势|CPU模型切换：Strict20中Exact/Greedy/1-Swap各20/20同集；Standalone仅w4/K3不同。后续M0/M2对Strict20+C/K3共21条件，Greedy/Exact均不变；C/K3线性Gap-only也复现FS|[80行对照](strong_baseline_position_audit_e5_step0/position_vs_gap.csv)、[21条件](paper_recovery_model_short_context_audit/partC_conditions.csv)、[线性Gap-only轨迹](paper_ck3_simple_policy_check/gap_greedy_trace.json)|**尚未验证**，同集结果必须保留。收窄为预测改进；不能将C/K3请求优势独立归因于位置感知项。Standalone的唯一变化也无独立GPU优劣证据|
|C05；IV：单调次模与Greedy保证|固定非负b下F=Σmax b，具有标准单调次模性；纯基数精确边际Greedy有1−1/e收益保证。模型Phi在0≤G≤T≤131072单调|[完整证明及数值边界](allocation_structural_properties/final_report.md)、[Phi实现](../flowstate/recovery_model.py:59)|**已验证为条件性数学性质**。保留；不得称新数学理论；保证针对F，不是J乘法比、TTFT、HBM利用率或并发系统。浮点同分容差需注明|
|C06；IV/实验：联合分配优于独立状态排序|Formal573：168 OH×3预算+23 AgentX×3预算，四策略573/573同集同E/G/J。Strict20：5快照×4预算，Greedy对StandaloneTopK4胜16平；Greedy=WFC20/20，均仅5条件最优；1-Swap18/20|[历史结构逐条件](paper_allocation_structure_legacy573.csv)、[Strict140行策略表](greedy_swap_structure_e3_step1/condition_results.csv)、[结构审计](paper_audits/t1_structure_budget/t1_final_report.md)|**有限条件下支持**。可写存在性及结构条件，不能写普遍领先WFC/PCB/LDR。历史573不是Strict认证，也不是573独立工作流|
|C07；IV：Exact必要性、Greedy失败与质量|B5-C4/K2：Exact48.228729、Greedy130.803084 modeled ms；Strict20 Exact20最优，Greedy/WFC5，固定首改善1-Swap18。w1/w2 K3仍有1-Swap局部失败；K≥K_min只保证存在覆盖解|[Exact证据](multishare_allocation_gate_b5c4/exact_verification.json)、[Strict20逐条件](greedy_swap_structure_e3_step1/condition_results.csv)、[结构归因](paper_allocation_structure_audit.md)|**已验证于这些输入**。保留负面结果；不能说一般1-Swap都失败、Greedy达到K_min必最优、不同选集必有收益|
|C08；IV/V：当前系统使用Exact–Greedy|生产`plan_strict→validate→规模门禁→Exact/原Greedy→Plan`；Exact n≤12,m≤4,K≤4,子集≤794，错误不回退。CPU Strict20复现Exact；G1门内w1/K2四臂、超域C/K3四臂均在Scheduler内在线选择并真实协调恢复|[optimizer.py](../flowstate/optimizer.py:86)、[CPU集成](paper_hybrid_solver_integration_gate.md)、[G1原始核验指针](hybrid_solver_gpu_gate_g1/checks.json)、[G1报告](hybrid_solver_gpu_gate_g1/final_report.md)|**已验证于8个正确性臂**。保留混合实现，但Legacy `select`仍Greedy；不能据G1计时记录声称在线性能已验证。旧“Exact仅离线/strict永远Greedy”已过时|
|C09；IV/实验：求解成本可接受及最优证书|Strict20本地微基准每方法/条件120次，12480计时行含诊断：Greedy中位0.0972、Exact0.2238 ms，Exact经验P95 0.5067、max3.14 ms。另C/K3独立离线穷举697子集，Greedy唯一最优J154.348956，与PCB/LDR同集|[计时原始CSV](paper_solver_tradeoff_raw.csv)、[边界与环境](paper_solver_tradeoff_gate.json)、[C/K3证书](paper_ck3_simple_policy_check/exact_enumeration.json)、[简单策略表](paper_ck3_simple_policy_check/policy_comparison.csv)|**有限条件下支持**。CPU纯求解不是完整在线开销/硬deadline；C/K3证书不扩生产门禁。10月23日“无Exact证书”已被后续离线审计更新；不能继续沿用|
|C10；V：逻辑到物理映射、安全协调|当前Token全前缀唯一映射，拒绝重复Node、Slot/组件别名、非零引用、候选外Recurrent；逐项驱逐+postcheck，失败停止、不回滚。B5-D3保留2、驱逐5；E4每臂7→2；C/K3每臂16→3、驱逐13|[bind/preflight/postcheck](../flowstate/strict_control.py:141)、[B5-D3操作](exact_reconciliation_correctness_gate_b5d3/native_reconciliation_log.json)、[E4逐臂](native_resume_correctness_e4_phase1a_run01/resume_correctness.json)|**已验证于受控单池、等大小、静止配置**。不是一般组件共享、多槽、活动请求保护和容量压力系统的验证。Fail-stop是发现失败后隔离，非事务回滚|
|C11；V/实验：Native Resume、COW、FA保护、Cold|B5-D3同真实来源B：H/E/G4096/3712/384，8Token与Cold一致；E4 w1/K2共8严格臂+4Cold；E5 K3共8Smoke+24测量+4Cold，全部已记录通过|[D3输出](exact_reconciliation_correctness_gate_b5d3/output_correctness.json)、[E4认证](native_resume_correctness_e4_phase1a_run01/resume_correctness.json)、[E5报告/原始索引](exact_swap_runtime_e5_step1/final_report.md)|**已验证于实际臂**。EPOCH_ADMISSIBLE不自动RESUME_VALIDATED；请求后新工作状态允许变化；仍PROVENANCE_UNVERIFIED，不宣称已认证跨Workflow张量血缘|
|C12；实验：相对MC/KV的请求性能|T2-G2 GPU1独立批次60主臂+3Gate；合成C/K3、4Pending×5轮×3策略。原始summary复算FS−MC请求配对中位−148.932193 ms，FS−KV−148.749170 ms，各15胜5负；四Pending等权轮级差中位−114.127159/−105.727739 ms，五轮均负|[60主臂CSV](t2g1_three_policy_gpu/formal_gpu1b/formal_arms_main.csv)、[原始例](t2g1_three_policy_gpu/formal_gpu1b/m1A_FS/run/summary.json)、[全60臂报告/索引](t2g1_three_policy_gpu/formal_gpu1b/paper_evidence_final_audit.md)|**有限条件下支持**。保留全部A请求劣势；冻结选集应用，不是在线三策略完整系统性能；仅Marconi-Evict Adaptation/KVFlow-inspired，不是官方完整系统|
|C13；Introduction/实验：完整控制后稳定净加速|B5-D4b CUDA五对均改善、TTFT五对均恶化；E4 A的CUDA/TTFT改善但Control RPC三对均更高；E5四请求CUDA总和三差值有正有负；旧24快照完整Epoch FS对MC均值反而+549.649 ms|[D4b原始CSV](exact_greedy_paired_gpu_gate_b5d4b/paired_gpu_timings.csv)、[E4计时](native_resume_performance_e4_phase1b/gpu_raw_timings.csv)、[E5总和](exact_swap_runtime_e5_step1/four_pending_paired_sums.csv)、[P0-3逐Epoch](paper_runtime_epoch_bottleneck_audit/epochs_new.csv)|**尚未验证，存在反向结果**。不得概括为稳定净收益；对“每次/所有配置都更快”已有反例。必须等待P0-4闭环及边界认证|
|C14；Introduction/实验：真实Agentic工作流普遍适用|B5-C2完整OpenHands消息与数组；C3真实物化；E2四独立来源组合；AgentX六差异例缺原Prompt/Token且Formal/Raw冲突、4例Node别名|[C2来源锁](multishare_cpu_input_gate_b5c2/source_selection_lock.json)、[E2物化](independent_runtime_allocation_e2/runtime_materialization_summary.csv)、[AgentX审计](agentx_real_trace_gpu_feasibility_audit.md)、[最新缺口审计](agentic_workload_evidence_gap_audit.md)|**有限条件下支持真实来源受控重放**。不是原生Qwen Agent工具闭环、自然并发、真实Fork或在线Ready。C/K3合成来源另列；不能将不同来源条件混成胜率|

## 4. 端到端性能专项审计

### 4.1 必须分别命名的六种指标

|指标|实际对象与计时边界|不能替代|
|---|---|---|
|模型J|ΣPhi(T−E,T)，固定候选、T、严格矩阵及Phi|GPU恢复耗时、TTFT或完整周期|
|CUDA Prefill|插桩CUDA事件包围的实际Prefill区间，需说明同步与是否含提交空隙|客户端TTFT、全部GPU内核之和、端到端Wall|
|请求TTFT|目标请求提交→首个非空输出；控制可能此前已完成|控制+请求净时延|
|四请求TTFT等权聚合|独立生命周期内各请求TTFT的均值或和，再按轮配对|同Engine四请求串行完整周期、并发吞吐或Goodput|
|完整控制周期|共同观察开始→第4请求完成；中间选择、协调、传输、间隙按协议纳入|四个独立TTFT相加、单请求Wall；也不是Engine启动至清理总时间|
|Residual/请求间隙|完整周期−控制−请求执行和；用同一组端点定义|可任意删除的“审计时间”；未分类子项不能估计补齐|

当前C/K3：四个T分别4096/5120/6144/7168，完整输入各为T+32。T2正式数据的原生恢复关系：

|Pending|FS E/G/实际Prefill|MC E/G/实际Prefill|KV E/G/实际Prefill|
|---|---|---|---|
|A|0/4096/4128|1024/3072/3104|4032/64/96|
|B|5120/0/32|1024/4096/4128|2048/3072/3104|
|C|6144/0/32|1024/5120/5152|0/6144/6176|
|D|7168/0/32|1024/6144/6176|0/7168/7200|

这解释了A的明确负向取舍；**G=0仍有32个新输入Tokens需要Prefill**。OpenHands例中输入长度与T的差不同，不能把本表的“+32”当通用恒等式。

### 4.2 T2-G2：请求窗口有效，不是完整周期优势

本轮直接联接60个 `m*/run/summary.json` 与 `formal_arms_main.csv`：E/G/实际Prefill及状态/Cold字段一致；TTFT/CUDA字段按CSV显示精度一致，Control RPC两位小数舍入最大差0.004954 ms。统计只取60主臂，不混3个Gate、Smoke、GPU0中断批次或GPU1失败批次。

在同一合成C/K3、K3、冻结输入/预算、独立生命周期、形状预热和编译门禁下，FS相对两适配基线有上述请求级和轮级等权TTFT改善。A对MC/KV均5/5负向；B/C/D各5/5正向。不能写“所有请求改善”。

控制RPC原始中位：FS **11499.244845 ms**、MC **11060.672954 ms**、KV **11123.472019 ms**，在正式TTFT窗口之前。`c_formal_perf_gpu/control.py:75` 读取 `plans[policy]['selected_checkpoint_ids']`，其后 `planner.finish(prepared, selected)` 仅为计划构造/认证。T2-G1的CPU真实选择和G1的GPU在线混合门禁，不能追溯改变这60臂的执行路径。

已有 `paper_evidence_final_audit.md` 第4节/第6条写“控制开销不折损TTFT净收益”“控制开销不影响净收益”。**前半句只能限定为“不在所定义的请求TTFT窗口内”；后半句超出其证据。** 控制同量级不意味着每对严格抵消；不能将11秒归入生命周期外而据此证明完整控制周期净收益，也不能用20个请求对差中位减另一个父窗口中位构造净值。

命名统一为 **Marconi-Evict Adaptation**、**KVFlow-inspired**。前者保留适配驱逐机制但不等同原系统Attention合并/准入全栈，后者不含完整Prefetch/Scheduling；输入使用预先声明访问序和ASG/STE。C/K3上PCB/LDR以及线性Gap-only均与FS同集同J，这项负向创新性边界必须与SOTA适配比较并列说明。

### 4.3 已有计算、TTFT和完整路径的方向冲突

|实验与统计单位|恢复计算/TTFT|控制及完整路径|允许结论|
|---|---|---|---|
|B5-D4b，Pending C五对|CUDA均值78.348→46.099 ms，5/5改善；TTFT1039.121→1078.325 ms，5/5恶化|Control-to-completion均值13219.006→13135.119 ms，只有3/5改善|更深恢复降低该配置计算，但TTFT与完整路径没有一致改善|
|E4-1B，w1/K2 A三对|CUDA73.026→42.618；TTFT102.601→74.258 ms，均3/3改善；C同前沿负对照CUDA37.062→37.460|A Control RPC775.516→824.074 ms，三对均变长；完整路径7527.624→7352.864，仅2/3改善|有限请求收益成立，稳定全路径收益不成立|
|E5，w1/K3每组四次独立恢复总和|Exact−1-Swap的CUDA总和−13.399、+1.938、−3.102 ms；均值差−4.854 ms|独立生命周期之和不是一个自然工作流周期；控制/请求Wall另有波动|不能用均值负号声称稳定联合收益，也不能据三组判统计等价|

详细边界见 [E4计时审计](e4_control_timing_audit/final_report.md)：12个测量臂Control RPC均值814.493 ms，七个非嵌套服务阶段合计332.074，余额482.419 ms是**端点差额，内部分类UNKNOWN**。Control-to-completion均值7469.333，包含814.493控制、243.077控制后间隔和6411.764请求Wall；不能再加TTFT或CUDA事件。Audit-inclusive均值9428.766包含前置参考及间隔，但仍不包含Engine加载、全部历史和清理。首Decode长区间原因尚未由该数据分解；不能事后删成“预热”。

### 4.4 旧完整周期与P0-3：定位噪声不等于获得净加速

来源为旧/优化后两批各216个生命周期，24快照内三重复先均值，再24个快照配对；另有72个profiling生命周期。不能与T2的合成C/K3、Marconi-Evict或8Token请求混合。

优化后原始配对：FS−MC完整Epoch均值 **+549.649 ms**，控制 **+33.947**，四请求执行和 **−147.340**，Residual **+663.042**；三者代数闭合。FS−LRU完整Epoch均值−526.612，但不能略去快照间大幅正负波动或将其改称稳定优势。

Profiling会话Residual均值1126.2 ms，其中RPC墙钟1101.5（97.8%）；服务端计算43.4 ms（RPC的3.9%），before_submit+event_handoff占约95%。完整Epoch中>5s残余停滞同时出现在各策略。具体GIL/OS/共享机器原因 **UNKNOWN**；“观测到交接区间占主导”不等于证明其完全外生或全部可删除。剔除停滞仅能作诊断，不能替换正式统计。

依据：[逐Epoch](paper_runtime_epoch_bottleneck_audit/epochs_new.csv)、[profiling时间戳拆解](paper_runtime_epoch_bottleneck_audit/profiling_epochs.csv)、[审计原始统计](paper_runtime_epoch_bottleneck_audit/audit_data.json)。P0-3的“NO_CHANGE_NEEDED”是当时的任务决策，不是完整周期性能认证，也不能授权对冻结结果扣除估计开销。

## 5. P0-4实际状态及待验证代码风险

### 5.1 已完成与未完成

- `p0_control_inclusive_harness/out/cpu_gate_result.json`：**32/32 CPU检查PASS**；真实选择内核及纳秒计时恒等式已作CPU合同验证。报告状态为READY_FOR_GPU，列出7项GPU-only待验证。
- FS选择内核调用生产 `plan_strict`，MC调用 `MarconiEvictPolicy.select`，KV调用既有 `kvflow.select`；CPU记录复现冻结选集和J。这不是GPU实时观察、物理控制或完整周期实测。
- 新GPU包 `p0_control_cycle_gpu/` 已有观察、三策略在线选择、协调与四请求串行代码。**本轮读取时 `arms/` 为空，未找到该包的summary、输出、状态、Cold比较、清理或三策略周期统计。** 因此不能声称GPU已完成；其他未提供目录中的运行状态为UNKNOWN，不能据包名和代码存在补全。
- 未运行该包，也未改动P0-4文件。其启动器、实现锁与测量边界仍需相应任务验证，不能用本次静态阅读批准执行。

来源：[CPU报告](paper_control_inclusive_three_policy_cpu_gate.md)、[32项结果](p0_control_inclusive_harness/out/cpu_gate_result.json)、[GPU Runner](p0_control_cycle_gpu/runner.py)、[控制入口](p0_control_cycle_gpu/control.py)。

### 5.2 CPU报告不能覆盖的两个现存GPU测量风险

**风险一：审计写盘仍在控制周期中。** GPU `control.py:20` 导入旧 `exact_greedy_paired_gpu_fix_b5d4a.persistence.Journal`；`timed()`在第84行设置T0，第97/155/193/223/269行调用 `journal.record`，T3在第300行设置。真实Journal的 `record→save_new` 会JSON写盘、flush、fsync（旧 `persistence.py:5–17`）。它们不在各 `timed` 函数内，但处于T0–T3，进入stage gaps；最终complete写盘还在T3后、响应前。客户端 `_call` 同样有原响应持久化，`reconciliation.json`/`request_plan.json`也在请求前写。

CPU Harness中的 `OutOfWindowEvidence` 出窗合同，**尚不能视为这个GPU包已完成相同分离**。如后续报告声称“测量窗零Journal/fsync”，当前源码构成直接冲突；耗时大小现为UNKNOWN。整数恒等式闭合可以保留这些开销，但不能把它们标为纯生产控制或假称已出窗。

**风险二：形状预热的窗口标签不一致。** `runner.py:200–235` 在T3后、正式请求前执行四个形状预热并写“窗口外、不进控制周期”；但第382–390行恒等式纳入warm_total，第421–430行 `cycle_ms=(T4−T0)/1e6`，因此**真实T0–T4完整周期包含预热**。`control_cycle_ms=T3−T0`才不包含。不能同时使用“不含预热”解释和该cycle数值，也不能事后扣warm_total伪造连续生产窗口。

这些是待验证Harness的静态一致性问题，**不是已发生的GPU失败或已测性能损失**。本轮不修复；后续性能结论须采用与实际时间戳一致的冻结定义，并将审计、形状预热及其状态副作用说明清楚。

## 6. 正文一致性问题与修改建议（按严重性）

正式稿未取得，以下不能填入虚构页码/句号。已确认的原句来自现有报告；正文项以章节和待核对主张列出。

|严重性|具体位置/待核对表述|证据与风险|建议方向，不直接改文稿|
|---|---|---|---|
|P0|正式稿本身缺失|无法确认最新Abstract/Introduction/III–VI写了什么，不能给“全文一致”签名|补正式稿路径及内容摘要后才做逐句认证；当前报告注明此限制|
|P0|Introduction/贡献/实验：稳定端到端净收益；T2报告第6节第4条“控制开销不影响净收益”|T2只测请求窗口且冻结选集；旧完整Epoch对MC反向；P0-4无GPU产物|现在收窄为指定请求级配对TTFT，完整周期结论待P0-4；保留负向请求和完整路径|
|P0|V/实验：三策略全部在线Observe–Select–Reconcile性能已测|G1仅FS混合路径GPU正确性；T2-G2三策略正式性能加载冻结集合|逐实验标明在线求解或冻结应用，不能拼成同一次在线对照|
|P0|P0-4：“审计全部出窗、形状预热不进完整周期”|当前GPU包Journal同步写盘在窗内；T0–T4包含预热，详见§5.2|后续先核对实际边界与适用CPU门禁版本，不提前写成已测轻量完整周期|
|P1|III/IV：J代表真实TTFT；G代表实际Prefill|Phi为相对G0的额外成本，T不一定等输入长度；C/K3 Prefill=G+32，D3例不同|分别定义H/T/E/G、额外恢复区间与完整extend输入；J标modeled ms|
|P1|Introduction/IV：位置感知模型使当前分配更优|Strict20及C/K3主要算法同集；短上下文T低于校准支持域|保留长域预测改进，明确短侧外推；删除独立位置项已带来选择/GPU收益的归因|
|P1|IV：FlowState统一使用Greedy或统一保证Exact最优|当前strict混合、Legacy Greedy；C/K3超在线Exact门禁，但另有697子集离线证书|方法图、伪代码、实验图例分别说明路由、规模与证书；不把离线证书说成在线Exact执行|
|P1|IV：1−1/e是成本/延迟保证或新的理论贡献|仅固定非负max-benefit、基数约束下的经典收益保证；浮点容差另有边界|写成标准性质的应用，不提供J乘法比或Runtime保证|
|P1|V：物理命中、准入、实际恢复、张量血缘等同|EPOCH_ADMISSIBLE、RECONCILIATION_VALIDATED、RESUME_VALIDATED是不同证据；PROVENANCE_UNVERIFIED未解除|保留等级，不把同Token/首次观察当跨Workflow张量创建血缘|
|P1|III/V：K是HBM压力；共享组件都能安全释放；跨RPC租约持续有效|已验证为等大小唯一单槽域、连续单Scheduler窗口；池本有空闲；bind拒绝别名|列出受控物理合同，明确逻辑K与真实容量压力不同、Epoch外需重验|
|P1|实验：真实多Agent并发、AgentX自然分支已GPU复现|OH真实轨迹受控组合；C/K3 synthetic；AgentX六例BLOCKED|按来源分别命名，不将历史Formal、Strict捕获、Native恢复、性能混成一类|
|P1|实验：战胜原始Marconi/KVFlow完整系统|实际为两种策略级适配，共同Recurrent-only路径，不含完整原系统所有机制|全篇表格/摘要/图例固定用适配名称，披露各自额外输入与机制范围|
|P2|实验：简单策略都差；K1各策略都最优；Greedy优于WFC|Strict20 PCB/LDR仅K4的5条件最优，Greedy/WFC同集20/20；C/K3 PCB/LDR与FS同集|逐策略列分母。此前聊天中的结构段落“在K1各策略均…最优”若被采用，应改为对应的Greedy/WFC/Standalone/Exact；不是全部策略|
|P2|实验：20条件=20工作流、60主臂=60负载；固定首改善失败=所有局部搜索失败|Strict20仅5快照；T2为单合成实例×重复；1-Swap由固定首改善路径决定|修正统计单位与算法限定，保留18/20强局部基线结果|
|P2|旧总结“混合GPU未通过”“C/K3无最优证书”“正式60臂IN_PROGRESS”|对应10月22–24日截点旧文档；后有G1、CK3审计和T2 GPU1批次|明确版本时间，采用最新对应证据；不能让过时状态覆盖新结果，也不能让新结果追认旧执行路径|

短上下文精度尤其需要限定：原训练T为32768–131072，Strict/C使用T约3471–7168；代码允许的数学域不等于经验校准覆盖。现有短侧审计支持若干跨策略差值方向，但绝对预测不能继承长域2.68% MAPE，更不能据此重算所谓真实收益。

## 7. P0-4后的决策条件

### 现在即可修正，不依赖新GPU

1. 统一J、CUDA、TTFT、四请求聚合、完整周期与Residual名称及包含关系；删除稳定生产净收益推断。
2. 更新当前混合求解实际实现与G1资格，区分每批性能实验是否在线Select。
3. 保留适配基线名称、Strict20平局/失败、C/K3 PCB/LDR/Gap-only同集、E5混合方向与全部来源限制。
4. 正式稿一旦提供，按本报告编号定位实际原句；当前没有证据证明作者已把上述建议写入最新版。

### 必须等P0-4真实产物，不能由CPU代码代替

|待决问题|最低必要证据|通过后仍不可外推|
|---|---|---|
|同次GPU周期三策略是否真实在线Select|实时Epoch→实际内核→Plan→当前Binding→协调；冻结oracle只在选择后；输入身份、时刻可见性、实际调用及清理证据|不能推广成原系统全栈Marconi/KVFlow比较|
|四Pending串行周期是否保持正确|每请求Native来源/E/G/实际Prefill、COW/源保护、FA、Cold；检查前一请求新物化状态是否抬高后一请求前沿；每次预热和实际请求的影响均可追溯|不能用独立生命周期单请求认证代替串行同周期资格|
|T0–T4及“低审计”边界是否真实有效|逐段绝对单调时间戳和恒等式；Journal/fsync/预热/取证实际位置与协议一致；缺测残余保留UNKNOWN；共同CPU门禁与实际GPU实现版本对应|闭合恒等式不自动证明窗口代表生产、不自动证明净收益|
|是否观察到全周期收益|按冻结重复/策略顺序完成的同负载同预算配对，保留失败、负向、间隙和全部有效样本；Control、请求及Residual分别报告|少量合成C重复不是普遍生产性能/显著性/Goodput证据|

若完整周期FS仍更慢或方向不稳定，应保留请求计算收益，并明确“当前协议下未观察到稳定控制后净收益”；不能剔除审计或停滞后将诊断值替换正式结果。即使P0-4成功，原生Agent、自然Ready/Fork、跨模型泛化和真实HBM压力仍需各自证据，不会随该门禁自动补齐。

## 8. 本轮核验记录与可追溯性

优先材料已覆盖：历史Formal、B5-C2/C3/C4、D3/D4b、E2/E4/E5、Strict20、C/K3、混合Solver G1、T2-G1/G2、P0-3/P0-4及 [Agentic缺口审计](agentic_workload_evidence_gap_audit.md)。本轮直接读取生产optimizer/strict_epoch/strict_control/recovery_model、T2冻结控制入口和P0-4GPU源码，没有用旧证据总结代替这些实际路径。

关键离线复核：

- Strict20的140行表：Exact20/20，Greedy5/20，1-Swap18/20，WFC5/20，Standalone/PCB/LDR各5/20；未重新求解。
- 模型12个held-out预测行直接重算M0/M2 MAPE；读取20校准点、配置、原始held-out及模型选择；未重新拟合。heldout_summary的15行含三个G0基线，不能冒充15个非零gap误差点。
- T2正式60行与60份原始summary的关键字段/状态/Cold对照及40个比较差、五轮均值差重算；原始未舍入数字优先于CSV。未调用有落盘副作用的正式验证器。
- P0-4 CPU结果32项全部标PASS，登记7项GPU-only；GPU `arms/` 在读取时为空。没有执行启动器或探测GPU运行状态。
- 源码/证据读取身份：`flowstate/optimizer.py` SHA256=`7dfa9aec307d1178d79d79ced0b5d42b0f6574754194c58d84db300c74e48176`；P0-4 CPU结果=`d046e85f2fc7d889bab54832841ae8730176fb6853a6c37feebfd0a7ab6c95c7`；GPU runner=`cd7951396cfbb1f10a0b52e8cf1fb9b6eeac6bcd73c825d457446ef66842dcc1`；GPU control=`68118d0baf5e5261bb61e4c7b1b88bd1f73820722ba2a0f6bf216c31d3e0f5f0`。活动文件之后可能变化，当前审查不认证未来版本。

**交付结论：PARTIALLY_SUPPORTED。** 可保留严格恢复建模、有限集合分配差异、混合Solver及受控物理正确性；请求级比较须限定来源、策略适配与窗口；完整在线净收益待P0-4证据。当前正式稿未提供，逐句正文审计仍为UNKNOWN，本报告不签发“论文全文已核对一致”的结论。
