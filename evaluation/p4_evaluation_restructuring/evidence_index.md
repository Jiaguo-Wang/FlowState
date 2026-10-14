# 证据、口径与复现索引

下列路径相对于 `/home/wjg/code/FlowState`。不从论文或图像估计数值。最新论文正文和P4-G0未定位，逐段核对为UNKNOWN。

## Allocation 原始输入

| 代号 | 正式来源 | 使用范围 |
|---|---|---|
| OH | `evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl:1` 起，共504行 | LRU/LFU/Marconi/FlowState的原始目标、集合、Exact子集 |
| AX | `rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl:1` 起，共69行 | 同上；保留派生兼容与原生事件证据边界 |
| AB | `evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726/per_case_results.jsonl:1` 起，共573行 | without_set_dependent 对应 StandaloneTopK；full及去位置消融 |
| SB | `evaluation/p1_strong_baseline_validation/per_case_results.jsonl:1` 起，共573行 | 八方法共同评分、逐pending E/G、三强基线、K_min及选择顺序 |

`per_case_evidence.csv` 的 source 为原始正式集合/目标来源的准确文件与行号；structural_source 是P1-B行号，指向八方法统一评分与结构证书。每条都有dataset、snapshot_id、budget_ratio、method；联接不只用K，因为不同预算可能产生相同K。共4584条方法–条件记录，48个汇总组。

主矩阵使用P1-B已冻结的共同评分 `cost_ms`，并逐条核验OH/AX原始成本及AB的common_cost，差异≤1e-9 modeled ms；同时保留 `original_frozen_cost_ms`，不覆盖原值。这样避免跨历史浮点求和顺序把精确持平显示为约1e-13 ms增益。没有本轮调用Phi、selector或重建population；不是新baseline运行。强基线原P1-B目标差最大值就是0，并非扩大容差造平局。

## 矩阵定义

- 每行是单一dataset×budget×method；OpenHands分母168、AgentX23。三个预算的行只能描述性并列，不作为独立重复样本推断显著性。
- mean/median为每个snapshot的Σpending modeled recovery cost；并非每请求GPU时间。
- 差值为baseline−FlowState，正值表示FlowState成本更低；W/T/L用1e-9 modeled ms容差。FlowState自身行为全ties。
- Full Coverage：原正式判定为 C(S)≤1e-9（`evaluation/rq5_frozen_snapshot_ablation.py:427`）。本轮从冻结逐pending整数G核对所有G=0，并验证全部4584项与上述成本判定一致。不得用reaches_all_frontiers替代。
- OpenHands FlowState各预算为71/168、156/168、156/168，即42.3%、92.9%、92.9%；Marconi为4/168、32/168、69/168，即2.4%、19.0%、41.1%。AgentX所有方法全覆盖为0，原因包含不可消除的target residual，不能解读成所有方法毫无收益。
- selected_set等价按checkpoint ID集合比较，不要求选择顺序相同。
- K_min_frontier使用既有P1-B结构证书，目标是达到所有**候选可实现**的逐pending最深前沿，非保证G=0。零正前沿时K_min=0；本表不计算除以0的比值。
- 每行给出K及K_min的最小/最大值与K小于/等于/大于K_min的计数。物理pool竞争不能由此逻辑计数直接证明。

## Exact OPT 独立子表

`evaluation/rq3_formal_policy_evaluation.py:44,183,532`：Σ(i=0..min(K,n)) binom(n,i)≤100000时执行穷举；本轮只读取已有tractable=true且存在目标的结果。`exact_opt_cases.csv`保留全部482个来源行；`exact_opt_subset_quality.csv`按dataset/budget汇总。OpenHands 168/168、133/133、133/133；AgentX18/18、16/16、14/14。其余91条件无正式Exact结果，不计为不匹配，也不纳入Exact主曲线。

## 机制与结构

- P0：`evaluation/p0_allocation_mechanism_audit/final_report.md:15,39,91,99`；Case C `case_studies.md:166`，Case F `:423`；原矩阵在case_C.json/case_F.json。
- P1：`evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md:38` 起的数学充分条件与证明；不从482次观察推导一般最优性。
- P1-B：`evaluation/p1_strong_baseline_validation/final_report.md:9,28,78,95`；573全平、顺序差异、K_min及案例解释。
- P2：`evaluation/p2_shared_cohort_feasibility_audit/final_report.md:1` 起；原生fork/ready缺口、形状回放、容量证书。Figure C读取P1-B同口径证书，不新增求解。
- P3-G1：`evaluation/p3_physical_retention_budget_contract/final_report.md:1` 起；逻辑K与物理保留合同尚未联接，不将旧逻辑25%写成显存上限。

## Runtime 与基础模型

- RQ1：`motivation/artifacts/replay_cost_20260819/findings.md:1`，原始raw_runs.csv与replay_cost.csv同目录。此轮仅引用已审计刻画，不新绘拟合。
- RQ2：`evaluation/rq2_model_audit_output/20260909_055110/final_report.md:1,14`，recomputed_metrics.json同目录。此轮引用审计，不重新拟合。
- RQ4：`rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/runs/*/pending_requests.jsonl:1`起；每文件四条。`analysis_snapshot_level/paired_ttft_results.json:1`及bootstrap_results.json、per_round_results.json、repetition_stability.json。计时 `evaluation/sota_latency_runtime.py:137`；聚合 `evaluation/rq4_snapshot_level_analysis.py:150,286,383`。
- RQ6：`evaluation/rq6_batched_control_output/rq6e_20260911_085820/analysis/final_report.md:5,9,17`；本轮仅引用冻结控制表，不与E2E拼接。
- 初始E2E：`evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928/runs/*/record.json:1`，summary在analysis/summary.json。
- 优化E2E：`evaluation/control_path_optimization_output/control_20260929/optimized_formal/runs/*/record.json:1`，summary同上；跨批并列表在上层analysis/final_report.md:7。
- E2E代码：`evaluation/control_path_optimized_e2e.py:73,99,164,190,203,218`；初始`evaluation/resume_epoch_e2e.py`对应位置。
- serving资格：`evaluation/serving_path_e2e_output/serving_20260929/final_report.md:1`；未运行，不能提供性能图点。

所有数值来源和引用文件的读取前后SHA-256检查见source_integrity.json。此检查涵盖脚本列出的输入，不声称扫描了全仓库所有冻结文件。

## 复现与图注

只执行当前新目录中的CPU汇总脚本，不运行任何既有实验入口：

```bash
MPLCONFIGDIR=/tmp/flowstate-p4-mpl /usr/bin/python3 evaluation/p4_evaluation_restructuring/build_evaluation.py
```

环境：系统Python、matplotlib 3.7.5、NotoSansCJK字体；无GPU框架导入。脚本仅在本目录写CSV/JSON/PDF/PNG，在/tmp写matplotlib缓存。人工编写的章节建议不被脚本覆盖。

- Figure A：正式冻结总体每预算等权snapshot均值，两个面板各自纵轴；绿色曲线同时代表FlowState/PCB/LDR/WFC，完全重合而非视觉偏移。所有8方法完整值见allocation_tables.md/CSV。线段只连接已测三预算，不表示中间预算实验。
- Figure B：各强基线均573/573最终目标、集合、E/G相同；选择顺序不同9/12/21个条件，没有最终收益差异。573条件来自191个快照。
- Figure C：逻辑K不足以达到全候选前沿的占比及正收益共享结构占比；AgentX共享22/23却0/23受限，各预算分别计数。该图不是物理容量或原生并发证据。

图中没有新增拟合、估计缺失值、性能仿真或策略运行。所有UNKNOWN与未执行项保留在文档，不画作零。
