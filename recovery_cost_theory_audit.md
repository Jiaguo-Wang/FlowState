# FlowState 式（10）恢复成本模型专项审计

审计日期：2026-11-03。**最终判定：PARTIAL。** 现有参数的数学性质通过，20 个训练配置与 12 个非零缺口留出配置的预测结果可从原始测量复现；实际执行路径和固定缺口实验支持目标位置相关性。但标定量是首 Token 墙钟差分，尚未隔离位置效应的 GPU 组件来源，不能把三项多项式解释为严格硬件定律。留出集参与了模型族选择，应称“留出验证集”，而非完全未参与选择的最终测试集。

本轮仅进行文件读取、CPU 数值复算和冻结镜像源码读取；没有挂载 GPU、启动 Engine、重新拟合、修改生产代码或冻结数据。新增脚本、逐配置结果及源文件前后 SHA 清单见 [审计目录](evaluation/recovery_cost_theory_audit_20261103/)、[复算结论](evaluation/recovery_cost_theory_audit_20261103/recheck_results.json)、[源清单](evaluation/recovery_cost_theory_audit_20261103/source_manifest.json)。

## 1. 实际恢复执行路径

### 1.1 源码身份与证据边界

只读检查镜像 `sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`，包版本为 SGLang **0.5.17**。容器使用禁网、无 GPU 挂载方式运行标准库源码读取程序，未导入模型或执行推理。全文引用的 SGLang 行号均来自该镜像，根路径为 `/sgl-workspace/sglang/python/sglang/srt/`；完整文件 SHA、带原始行号的片段保存在 [native_sources.json](evaluation/recovery_cost_theory_audit_20261103/native_sources.json)。

其中 `unified_tree_core.py`、`mamba_component.py`、`schedule_batch.py`、`scheduler.py`、`model_runner.py` 五份源码摘要与 [已有 GPU source_identity](evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs/source_identity.json) 一致。旧标定配置记录了版本和启动参数，但未保存上述所有文件的逐文件摘要，因此不能声称已证明旧标定安装与当前镜像全部源码逐字节相同。旧测量中的实际 H/E/G 则可独立核对。

本地模型配置 [config.json](/home/wjg/models/qwen3.5-9b/config.json) 的文本部分为 32 层，其中 24 层 `linear_attention`、8 层 `full_attention`，每 4 层包含一个 Full Attention 层。已有 [runtime.log](evaluation/serving_control_gpu_gate_20261102/arms/A_shadow_validation/runtime.log:3) 的服务器参数指定 `attention_backend=fa3`，第 666–667 行记录 Triton 线性注意力和 Hybrid GDN 后端；因此下表同时检查两类前向实现。

### 1.2 从物理命中到真正的 Prefill

以下是被审计的实际调用路径，而非依据工作流祖先关系推断的恢复过程。

| 环节 | 源码位置与函数 | 核查结果 |
|---|---|---|
| 全组件匹配 | `mem_cache/unified_cache/unified_tree_core.py:624–676`，`_match_prefix_helper` | 聚合组件验证器；只有各组件验证通过才更新最佳可执行节点。 |
| H 与 E 分离 | 同文件 `685–686,741–751` | `full_kv_hit_length` 单独累计；返回的 `device_indices` 只覆盖最佳合法执行边界。物理 Attention 命中更深不等于 Recurrent 可从同位置恢复。 |
| Recurrent 合法性及 COW 请求 | `components/mamba_component.py:130–202`，`create_match_validator`、两级 `finalize_match_result` | 验证状态驻留；记录源状态并分配请求工作槽；设置 `mamba_cow_src_index`。此时不能仅凭字段已设置就认定复制完成。 |
| 请求与 Batch 区间 | `managers/schedule_batch.py:1222–1226,1249–1347,2281–2346`，`Req.get_fill_ids`、`init_next_round_input`、`ScheduleBatch.prepare_for_extend` | `get_fill_ids()` 返回截至当前 `extend_range.end` 的完整填充输入；真正送入模型的是 `get_fill_ids()[len(prefix_indices):]`，并校验 Prefix/Extend 长度一致。不存在本审计可依赖的 `Req.fill_ids` 假设。 |
| 原生执行 | `managers/scheduler.py:3528,3680–3694`，`Scheduler.run_batch` | 在本实验关闭 overlap 的路径，准备前向输入并调用 `model_worker.forward_batch_generation`。 |
| 实际复制 | `managers/schedule_batch.py:2639–2660`；`model_executor/model_runner.py:1486–1523` | 收集延迟 COW 源/目标索引，随后执行状态加载或 `mamba_pool.copy_from`；实际完成还须由运行日志和组件检查认证。 |
| Hybrid 前向 | `models/qwen3_5.py:620–667,688–742,830–890,1166–1189,1226–1235` | 后缀经过 Gated DeltaNet 层和 Full Attention 层，而非仅补算一个独立 Recurrent 模块。 |
| Conv/Temporal 初态 | `layers/attention/linear/gdn_backend.py:470–573,667–680`，`forward_extend` | 取得 Conv 与 Temporal 缓存，根据非零 Prefix 使用初态，向 Conv/GDN 前向传递状态及序列区间。 |
| Attention 上下文 | `layers/attention/flashattention_backend.py:1122,1320–1332,1453–1463`，`forward_extend` | 调用携带 `page_table`、`cache_seqlens`、查询区间等参数；Hybrid 后端按层路由，见 `hybrid_linear_attn_backend.py:885–946`。 |

设完整请求长度为 **N**，合法可执行前沿为 **E**，规划目标历史边界为 **T**，且 `E≤T≤N`。未分块时实际 Extend 为 **[E,N)**；若分块则各块区间的并集覆盖该后缀。其中 **[E,T)** 是历史补算，长度 **G=T−E**；**[T,N)** 是新增输入，长度 **N−T**。所以实际 Prefill 总量一般是 **G+(N−T)**，不能一律写成 G。

即使 Attention-side 的 H 已到 T，缺少同边界完整 Recurrent State 时，也不能跳过 `[E,T)` 的 Hybrid 前向。此前 E 以内的 FA-KV 可作为 Attention 上下文；后缀仍执行 Attention、Recurrent、投影和其他模型层，并可能按正常请求语义写入相应 KV。协调阶段“不修改 FA”与请求执行阶段“FA 完全不写入”是两种不同合同。

### 1.3 直接运行证据

- [R2-C1b-2 native_trace.json](evaluation/runtime_control_closure_r2c1b2/strict/native_trace.json) 的 `matches[0].native` 和 `execution[0]`：**H=4224、E=2432、历史 G=1792**；完整输入 **N=4394**，实际 Extend **[2432,4394)**，长度 **1962=1792+170**；源槽 6、工作槽 27；原生 `run_batch` 返回。观测字段的 CPU 合同见 [observation.py](evaluation/runtime_control_closure_r2c1b2/observation.py)、[Req API 审计](evaluation/runtime_control_closure_r2c1b2/req_api_audit.md)。
- [20261102 shadow_validation 的 post_request.json](evaluation/serving_control_gpu_gate_20261102/arms/A_shadow_validation/run/post_request.json)：四请求 **E=[0,5120,6144,7168]、G=[4096,0,0,0]、实际 Prefill=[4128,32,32,32]**。同一 [runtime.log:755–762](evaluation/serving_control_gpu_gate_20261102/arms/A_shadow_validation/runtime.log:755) 的正式 C_A–C_D（非 SHAPE_WARM）Prefill/请求记录确认新 Token 数、缓存数和 8 Token 输出长度。这是新尾部为 32 的协议，不能与标定协议的 64-token 尾部混为一种测量。
- 标定原始 CSV 每次均核对 `runtime_H=T`、`runtime_E=T−G`、`runtime_G=G`、输入长度 `T+64` 及安全字段；本次复算未以预期值替代实测 H/E/G。

### 1.4 T 的可观察作用与计时边界

**源码能够确认：** T/E 改变历史上下文和位置、Attention 缓存访问长度及页表/序列元数据；G 改变查询后缀长度；较大的后缀可能触发更多 Prefill 分块。Recurrent 初始化/COW、Batch 形状和同步也可能变化。这些是可检查的影响渠道，尚不能据此断言某个 kernel 独自造成位置效应，或断言实测时间严格服从某个复杂度公式。

标定入口 [recovery_profiler_128k.py:807–858](evaluation/recovery_profiler_128k.py:807) 在安全检查和 `client.ping()` 之后，调用 [measure_streaming_request:137–180](evaluation/sota_latency_runtime.py:137)。计时起点是调用 `engine.generate` 前，终点是调用者收到首个含输出 Token 的流块。它可能包含 IPC/排队、Native Match、COW/初始化、GPU 前向及输出交付；**不是独立 CUDA Event，也不是已隔离的纯服务端 kernel 时间**。旧环境文字将其称为内部 TTFT，论文应以此实际源码边界为准。

同 T 的零缺口基线消除了部分共同开销，但不能证明消除所有随 G 改变的初始化、分块、同步和新尾部 Attention 开销。原始数据不足以给这些部分分摊实测毫秒数，分别记为 **UNKNOWN**。

## 2. 原始数据与预测结果复算

### 2.1 来源、配置与独立性

主冻结目录：[recovery_model_freeze_20260826_154235_266020](evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/)。重要文件为 `config.json`、`candidate_models.json`、`calibration_points.csv`、`heldout_raw.csv`、`heldout_summary.csv`、`heldout_predictions.csv`、`model_selection.json`、`execution_order.json`、`server_command.txt`。

| 数据部分 | 配置与原始来源 | 本次核查规模 |
|---|---|---|
| 训练位置网格 | T={32768,65536,98304,131072}，G={4096,8192,16384,32768}；[position_matrix_raw.csv](evaluation/runtime_artifacts/recovery_position_audit_20260826_144654_852303/position_matrix_raw.csv) | 16 个非零训练点；包含各 T 的 G=0 基线共 20 配置、280 次记录。 |
| 训练长缺口补充 | T=131072，G={49152,65536,98304,131072}；[raw_trials.csv](evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133712_596813/raw_trials.csv) | 取其中 4 个点，合计训练 20 配置；该原始来源全量为 9 配置、126 次记录。 |
| 留出验证 | T={49152,81920,114688}；每个 T 的 G={0,T/8,T/4,T/2,3T/4} | 15 配置、210 次记录；其中 30 次预热、180 次测量。12 个非零缺口配置进入误差计算，3 个零缺口配置提供基线。 |

各配置均为 2 次预热和 12 次测量。本次从全部原始行重新分组，没有剔除不利测量；核对 case_id 不重复、PASS/安全字段、H/E/G 及输入长度。训练点可由两份原始来源的同 T 基线差分复现，留出摘要与预测文件也逐项匹配。配置级 MAE/MAPE 为 **12 个配置等权**，不是 180 次试验逐行 MAPE，也不是工作流胜率。

原环境：Qwen3.5-9B、H100 PCIe、TP=1、SGLang 0.5.17，`chunked_prefill_size=45056`、关闭 CUDA Graph 和 overlap、Recurrent 容量 24、静态显存比例 0.4；正式请求生成一个 Token。输入是在 T-token 历史后追加已生成的 1 Token 和确定性尾部 63 Tokens，因此 N=T+64。不能把该系数直接视为其他容量、批处理、图模式、模型或 GPU 的已校准参数。

**独立性核查：** 训练和留出的 T 集合、(T,G) 配置集合均不相交；候选参数文件在留出执行前后的 SHA 一致，未发现将留出测量用于参数拟合。可是 [grade_model/select_model:587–611](evaluation/recovery_model_freeze.py:587) 及后续选择流程使用留出误差选择 M0/M1/M2。因此这批数据是模型选择验证集，不能同时声称是完全未使用的最终泛化测试。相同模型、生成机制和输入族也限制了跨真实 Prompt 的外推。

### 2.2 基准差分和模型定义

本次独立计算：

\[
y_{T,G}=\overline{\mathrm{TTFT}}_{T,G}-\overline{\mathrm{TTFT}}_{T,0},\qquad
\mathrm{MAE}=\tfrac1{12}\sum_i|\hat y_i-y_i|,\qquad
\mathrm{MAPE}=\tfrac{100}{12}\sum_i|\hat y_i-y_i|/|y_i|.
\]

基线使用**相同 T** 的 12 次非预热均值，没有减错 T、重复减基线或使用不同协议的 CUDA 基线。G=0 不参加 MAPE 是既有定义，全部基线保存在 [heldout_baseline_recheck.csv](evaluation/recovery_cost_theory_audit_20261103/heldout_baseline_recheck.csv)。

- **M0 / Gap-only**：按 20 个训练点相同 G 的成本取均值，形成 8 个非零结点加零点的分段线性插值；不读取 T。它不是本仓库另一个旧版 `HistoricalRecoveryCostModel`，也不是本次新拟合的多项式。
- **M1**：既有无截距 `g,gτ` 线性回归。
- **M2**：既有无截距 `g,gτ,g²` 回归，即式（10）。本次只读系数并重新预测，没有运行回归。

| 模型 | MAE（ms） | MAPE（%） | 最大 APE（%） |
|---|---:|---:|---:|
| M0 Gap-only | 287.514084 | 15.291275 | 28.036457 |
| M1 Position-aware，无二次项 | 79.648757 | 4.635750 | 11.664183 |
| M2 冻结拟合精度 | 31.179583 | 2.682480 | 8.179469 |
| M2 当前生产舍入参数 | 31.179580 | 2.682480 | 8.179469 |

当前舍入参数与冻结全精度参数在这 12 点的预测最大差为 **0.000016600 ms**，不影响上述精度。完整配对预测、带符号误差、绝对误差与 APE 见 [heldout_prediction_recheck.csv](evaluation/recovery_cost_theory_audit_20261103/heldout_prediction_recheck.csv)。

| T（Tokens） | G（Tokens） | 实测增量 ms | M0 ms | M2 ms | M0 APE % | M2 APE % |
|---:|---:|---:|---:|---:|---:|---:|
|49152|6144|296.716|366.024|320.986|23.358|8.179|
|49152|12288|602.057|742.627|630.726|23.348|4.762|
|49152|24576|1226.296|1465.495|1216.466|19.506|0.802|
|49152|36864|1837.946|2353.241|1757.219|28.036|4.392|
|81920|10240|610.903|618.269|639.441|1.206|4.671|
|81920|20480|1237.384|1228.419|1247.641|0.725|0.829|
|81920|40960|2401.881|2766.833|2370.320|15.194|1.314|
|81920|61440|3418.063|4374.169|3368.038|27.972|1.464|
|114688|14336|1003.737|866.985|1041.466|13.624|3.759|
|114688|28672|2006.302|1702.572|2021.701|15.139|0.767|
|114688|57344|3823.063|4114.119|3798.477|7.613|0.643|
|114688|86016|5362.893|5779.762|5330.329|7.773|0.607|

M2 在 **10/12** 配置的绝对误差小于 M0；T=81920 的 G=10240、20480 两点反而分别多 **21.171、1.292 ms**，全部保留。不能写成逐点全面优于 Gap-only。

### 2.3 数据支持哪些形状

固定 G 的完整位置网格如下，单位 ms，值均为同 T 的 TTFT 增量均值。

| G | T=32768 | T=65536 | T=98304 | T=131072 |
|---:|---:|---:|---:|---:|
|4096|174.006|217.337|258.958|302.245|
|8192|364.189|450.745|537.807|622.906|
|16384|718.454|905.426|1081.735|1259.755|
|32768|1436.807|1763.329|2108.915|2449.544|

四种 G 下全部 **12/12 相邻 T 比较**随 T 增大而增加。例如 G=8192，从 T=32768 到 131072 成本由 364.189 增至 622.906 ms。故仅用 G 无法表示该数据中的系统性位置差异。[逐对结果](evaluation/recovery_cost_theory_audit_20261103/same_G_position_pairs.csv)

在三个原始来源分别按固定 T 排序，全部 **36/36 相邻 G 区间**成本增加。这些是来源内比较，含不同来源的重合配置，不能称为 36 个独立工作流。[逐区间结果](evaluation/recovery_cost_theory_audit_20261103/same_T_gap_steps.csv)

但相邻割线斜率的 **28 次变化中有 9 次上升**，所以原始均值并不逐区间严格凹；负二次项不能解释为所有区间边际成本必然下降。[完整斜率结果](evaluation/recovery_cost_theory_audit_20261103/finite_slope_changes.csv)

这些结果支持“位置相关、对缺口递增、低阶代理有较好验证精度”。M1 到 M2 的验证误差下降支持保留二次项的实用价值，不能证明三项形式是唯一正确模型。没有新拟合、显著性检验或独立组件剖析，也不声称已验证每个回归项的独立物理来源。

## 3. 参数、有效工作域与解析性质

实际参数来自 [recovery_model.py:39–49](flowstate/recovery_model.py:39)，单位为 g=G/1024、τ=T/1024，输出 ms：

\[
\theta_1=37.828150,\quad\theta_2=0.345974143,\quad\theta_3=-0.156201917.
\]

冻结全精度分别为 37.82814985789258、0.34597414281318883、−0.15620191741171993。实现 [estimate:58–71](flowstate/recovery_model.py:58) 在 G=0 返回精确零；[输入检查:94–105](flowstate/recovery_model.py:94) 拒绝 G<0、T<0、G>T 或 T>131072。

**声明域：** D={0≤G≤T≤131072}，即 0≤g≤τ≤128。这是函数允许域，不等于每个点都有 GPU 精度验证。非零训练覆盖 T≥32768、G≥4096，留出配置也在长上下文范围；短上下文和极小非零 G 仍属精度外推。

### 3.1 零点、非负和连续域单调性

直接代入有 Φ(0,T)=0。由于 θ3<0 且 g≤τ，

\[
\Phi=g(\theta_1+\theta_2\tau+\theta_3g)
\ge g[\theta_1+(\theta_2+\theta_3)\tau]\ge0,
\]

其中 θ2+θ3=**0.189772226>0**。固定 τ 的导数为

\[
D_g=\theta_1+\theta_2\tau+2\theta_3g
\ge\theta_1+(\theta_2+2\theta_3)\tau
\ge\mathbf{37.828150}>0,
\]

因为 θ2+2θ3=**0.033570309>0**。因此全连续三角域内严格随 G 递增，无需抽样推断。

独立顶点检查也一致：导数是仿射函数，闭三角域极小值在顶点取得。

| (g,τ) | ∂Φ/∂g（ms/Ki-token） |
|---|---:|
|(0,0)|37.828150000|
|(0,128)|82.112840304|
|(128,128)|42.125149552|

此外 ∂Φ/∂τ=θ2g≥0；固定 T 时 ∂²Φ/∂g²=2θ3=−0.312403834。后者是**模型自身的凹性**，不能反推所有原始测量严格凹。这些证明适用于域内连续变量，不依赖 Token 整数性。导数换算至 ms/token 还需除以 1024。

### 3.2 分配工作域覆盖与方法影响

本次读取历史 573 的 [matrices.jsonl](evaluation/gate1_step1_allocation_equivalence/matrices.jsonl)、Strict20 的 [strict_allocation_matrix.csv](evaluation/paper_strict_evidence_extraction/strict_allocation_matrix.csv)，枚举全部合法候选边和空集前沿，并补入 C/K3 三策略已有 `m1` 运行的实际点；去重得到 **1600 个 (G,T)**，最小 0、最大 T/G=**126847**，全部满足 D。见 [allocation_domain_points.csv](evaluation/recovery_cost_theory_audit_20261103/allocation_domain_points.csv)。这仅验证模型输入域；历史 Formal 边并未因此升级为 Strict 或 Native Resume 认证。

在固定 Epoch、固定兼容关系且无额外耦合约束下，上述单调性保证：更深合法 E 不会增加模型恢复成本，独立收益 `b_pc=Φ(T_p,T_p)−Φ(T_p−L_c,T_p)` 非负；最深合法保留状态决定每个 Pending 的模型收益，集合收益可写为 `Σ_p max_{c∈S} b_pc`（含零基线）。其单调覆盖/次模性质来自固定收益的最大值结构，**不要求 Φ 对 G 为凸，也不由这次 GPU 拟合证明真实多请求耗时具有同样性质**。本次没有发现需截断、替换成本函数的数学矛盾。

较短分配点在数学域内，却低于主要标定范围。已有 [短上下文汇总 partB_absolute.csv](evaluation/paper_recovery_model_short_context_audit/partB_absolute.csv) 的 10 项完整 Prefill 相对误差约 **10.72%–34.93%**，分母是完整实测 Prefill，预测含零缺口基线；它不是本次 Φ 增量 MAPE，不可与 2.682% 混报。跨 Kimi/TP=2 等也没有由这批系数自动获得精度认证。

## 4. 三项结构的机理解释及限制

### 4.1 可用的数学推导

对可微成本函数且 Φ(0,T)=0，定义有效边际成本 `r(z,T)=∂Φ(z,T)/∂z`，由微积分基本定理：

\[
\Phi(G,T)=\int_0^G r(z,T)\,dz.
\]

这是数学恒等表示，不等于证明 GPU 按 Token 独立加和。这里 z 表示固定 T 时逐步扩大缺口、纳入更早历史的长度，不能不加区分地解释为一次前向执行中的时间进度。

以 Ki-token 坐标 ξ=z/1024、τ=T/1024，若在有限域内用低阶近似

\[
r_{\rm Ki}(\xi,\tau)\approx a_0+a_1\tau+a_2\xi,
\]

则

\[
\Phi(G,T)\approx a_0g+a_1g\tau+\tfrac12 a_2g^2.
\]

对应 θ1=a0、θ2=a1、θ3=a2/2。若严格使用题设 Token 坐标 `r(z,T)≈a0+a1T+a2z`，则系数必须作单位换算：

\[
r_{\rm token}(z,T)=\theta_1/1024+
\theta_2T/1024^2+2\theta_3z/1024^2\quad(\mathrm{ms/token}).
\]

一条直观但更强的假设是：位置 s 上的有效密度近似 `c(s)=α+βs`，则

\[
\int_{T-G}^{T}c(s)\,ds=\alpha G+\beta GT-\tfrac12\beta G^2.
\]

它展示了线性项、交叉项和负二次项如何从历史区间产生，**只是建模动机**。若这条单一仿射密度规律严格成立，归一化系数还须满足 θ2+2θ3=0；实际为 **0.033570309**，并不满足。当前模型是更灵活的三参数近似，不能倒推为该规律的验证。

### 4.2 事实、假设与已获支持结论

| 层次 | 可以成立的表述 | 边界 |
|---|---|---|
| 执行事实 | 原生从合法 E 处理到请求终点，历史部分长度 T−E；Hybrid 层使用完整状态和 FA 上下文。 | 不能将 G 等同含新尾部的全部 Prefill Tokens。 |
| 数学推导 | 对有效边际成本作仿射近似可得到三项形式；当前参数在声明域满足零点、非负与单调。 | 近似前提不是硬件定理。 |
| 建模假设 | 上下文位置与有效边际开销相关，低阶交互项可以表征有限范围的变化。 | 未剥离 Attention、GDN、分块、初始化、调度等贡献。 |
| GPU 数据支持 | 固定 G 的位置差异稳定出现在均值中；本协议 M2 的配置级预测误差更低。 | 尚非纯 CUDA 因果归因，不是全模型/硬件泛化保证。 |
| 未获支持 | 某一个 kernel 严格产生 GT 项；二次系数具有唯一硬件含义；2.682% 适用于短上下文和所有 Serving 配置。 | 不应写入论文作为已验证结论。 |

## 5. Section IV-B 可以采用的技术结论

可以采用如下受限表述，不需修改公式：

> 从合法 Recurrent Checkpoint E 到历史目标 T 的恢复需要重执行 [E,T)；相同恢复缺口在不同目标位置下具有不同实测成本。因此，我们使用缺口与目标位置的低阶交互模型估计同位置零缺口基线之外的额外首 Token 延迟。三项形式可由有效边际成本的仿射近似得到，而非作为 GPU 硬件性能定律。冻结参数在 0≤G≤T≤131072 内满足零缺口成本为零、成本非负和对缺口单调递增。在 Qwen3.5-9B/H100/TP=1 的既定协议及 12 个非零缺口留出验证配置上，位置感知模型的 MAPE 为 2.682%，冻结 Gap-only 插值基线为 15.291%。该验证集未用于参数拟合，但用于候选模型选择。

同时应明确：该误差分母是**实测 TTFT 增量**；不是 CUDA Prefill 的误差，也不证明 Position-aware 必然改变选集、分配更优或获得端到端净收益。若正文使用“测试集”，应补充其模型选择用途；若使用“纯计算开销”，应收窄为上述实际测量口径。

## 6. 是否需要最小补充 GPU 实验

**维持经验恢复成本模型和有限域预测主张，不需要为本审计新增 GPU 实验。** 当前不存在必须重新拟合才能修复的复现问题，亦没有单调性反例。

**PARTIAL 的唯一机理缺口**是：TTFT 的位置效应尚未被拆分为同步 GPU 前向与请求侧非计算开销。若论文坚持“位置相关性源于 GPU 恢复计算本身”的更强主张，最小补证可复用已有配置：固定 G=8192，T=32768/98304，并各自包含 G=0 基线，共四种既有配置；在一致输入和协议下记录同步 CUDA Prefill 及 Match/COW/宿主首 Token 区间。只有进一步声称具体 Attention/GDN 因果来源时才需要组件级记录。该建议未执行，不扩展为新模型、全矩阵或参数搜索。

完全独立最终测试集是另一种泛化主张的额外条件；当前低成本修正是将 12 个配置准确标为模型选择用的留出验证集，而非要求本轮追加实验。

**结论：数学性质 PASS，原始数据与预测复现 PASS，GPU 组件级机理归因 PARTIAL；总体 PARTIAL。** 本轮到此停止，不启动 GPU、不改公式、不改论文正文。
