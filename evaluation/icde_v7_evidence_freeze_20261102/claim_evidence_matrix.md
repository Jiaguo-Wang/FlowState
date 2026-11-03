# v7.0 Section VI 主张—证据矩阵

版本：用户附件全文SHA见 `paper_version.json`；位置采用附件行号。数值索引见 `evidence_master.csv`；本文件不修改论文。

|位置与主张|原始证据/索引|复核结果|证据等级与结论边界|建议|
|---|---|---|---|---|
|VI-A，270：168 OpenHands、23 AgentX；573比例预算与764固定预算|`gate1_step1_allocation_equivalence/matrices.jsonl`、`p1_strong_baseline_validation/per_case_results.jsonl`、`parent_link_verification.json`|191快照、573/764条件正确；n/m范围可复核|已验证Formal输入统计；不是573/764独立自然工作流，更非Strict认证样本|保留；说明OpenHands为多Session受控组合、AgentX为离线逻辑矩阵|
|VI-A，274：主要GPU采用预先确定选集|`p0_control_cycle_gpu/arms/m*/run/summary.json`、`p0_control_cycle_gpu/control.py`|当前主数值来自Scheduler内在线三策略选择；结果再与冻结参照核对|该P0批次与T2-G2冻结选集协议不同；不能混写|改成“从当前Epoch在线选择，与预注册逻辑参照核对”；如指T2-G2须另列协议|
|VI-A，274：TTFT边界包含首次Decode|原runner的start/first_token时间戳；`pending_rows.timing_decomposition`|TTFT=提交到首次有效Token；首次独立Decode CUDA另列，不是TTFT求和的必含子项|已验证时间戳；不能泛指必然含一次完整Decode batch|保留精确定义，删除无条件“首次Decode”或解释为首Token生成/交付|
|VI-B，280：ΣTTFT 261.9；配对降低470.208/429.711 ms，64.23%/61.74%，五轮全改善|P0 15份原始summary；`audit_statistics.json/main15*`|261.902190；−470.207613/−429.710719；64.226378%/61.735126%|已验证请求级GPU、1固定C/K3、5轮；不含先前控制与预热，不是生产净收益|保留；标明合成受控Shared/Branched来源、n=5、R3中断续跑|
|VI-B，287：ΣCUDA Prefill中位233.5/700.4/648.5 ms|同一15份summary；`gpu_main15_reference.csv`，master B01_*_sum_cuda_median_ms|应为236.509439/699.810280/653.240814 ms|明确数值不一致；两种median聚合方式均不能复现文中三数|定向替换正文及对应表格；不需要新实验|
|VI-B，291：旧24快照43.06%/24.26%，24/24与22/24胜|`rq4_runtime_formal_output/.../runs.jsonl`，216运行/864请求|43.057902%/24.264401%；负例2个全部保留|限定支持：旧门禁、1输出Token、逐快照相对率平均；缺当前Strict/逐请求Cold及完整源环境闭包|保留数字并附旧协议限制；不并入现代Strict总体胜率|
|VI-C，295–299：20标定/12测试、M0/M2 MAPE/MAE/max error|`recovery_model_freeze_.../{calibration_points,heldout_predictions,heldout_raw}.csv`|20点；12非零G测试聚合点；210次含30预热；15.291275→2.682480%，287.514084→31.179583 ms，max28.036457→8.179469%|已验证Qwen长上下文额外恢复成本模型；测试未参与拟合；不支持短T/Kimi同等误差|保留；明确G/T单位Ki-token、长T范围和Φ为额外成本|
|VI-C，301：19/19非零成本差预测方向一致|`paper_recovery_model_short_context_audit/partB_pairs.csv`|24类比较中19非零，19方向均对|限定支持：来自T2-G2/C-formal/G2三套旧GPU协议，不是P0十五臂的19次独立实验|列来源与同前沿5格，不作统计显著性或普遍准确性结论|
|VI-C，301：额外恢复成本高估10.7%–34.9%|同审计`partB_absolute.csv`；master C05|10.720016%–34.929495%确实可复现，但分母为完整CUDA Prefill，不是G=0基线扣除后的额外成本|明确分母错配；不能用该范围充当Φ增量相对误差|改为“零缺口实测基线+Φ预测完整Prefill的误差”；若要写Φ增量误差须统一配对差口径|
|VI-C，302：M0/M2的21条件同Greedy/Exact选集|`partC_conditions.csv`、审计数据C_sensitivity|0/21选集改变，21/21序列一致|限定支持CPU敏感性；不能由MAPE改善推导这些实例上的分配收益；C/K3离线Exact不等于生产准入|保留反向结果；不得省略|
|VI-D，306–307：573同集同J及分离/饱和解释|原Formal评估行→Gate1矩阵→P1四策略→本轮K_min DP/J复算|OH 504无跨Pending正收益共享，90紧预算；AX 66共享但69预算全≥K_min；573同集同J|已验证冻结矩阵结构；不能外推“真实轨迹没有共享”或“收益重叠不起作用”|保留；把“合法”限定为该逻辑矩阵内，非Native认证|
|VI-D，308：764中10差异由公共/深分支竞争造成|`ten_difference_cases.csv`|9 AX K1/K_min2共享紧预算；1 OH g051-round5/K3无共享。10条均J不同；并非仅ID差异|计数已验证；对全部10条统一共享归因不成立|分别描述AX9条和OH例外；14条成对差异不能当14个条件|
|VI-D，309：396可枚举条件最优396/395/395/394|`exact_oracle_results.csv`、固定策略CSV|完全一致；99快照×4预算；另368条件N/A|已验证已存枚举证书与目标，门禁n≤12/m≤4/K≤4/≤794子集；非全部764最优证书|保留；标明规模准入和368未认证|
|VI-D，310–315：B5-C4/K2非嵌套组合、GPU C/D收益|C3原Epoch、E3结果、G2正式40行与40份summary|K1{N20} J228.054813；K2 Exact48.228729 vs Greedy130.803084；C/D E2432→3520，Prefill各少1088；ΔTTFT34.707/36.712 ms|已验证受控真实来源重放+GPU；不是原生自然Fork，N18/N27为本例逻辑标签；血缘未认证|保留；明确五轮配对与A/B负/持平结果；不能推广全部Strict边已恢复|
|VI-E，321–324：Strict20 Exact20、Greedy5、1-Swap18；CPU0.2238/0.2209|E3 140行、`paper_solver_tradeoff_raw.csv`|结果一致；两次失败为w1/K3和w2/K3；计时分别0.223828/0.220940 ms|已验证CPU；固定首改善的局部终集失败，不表示所有1-Swap路径失败|保留并声明规则、容差、聚合样本；20条件不是20工作流|
|VI-E，326–327：C/K3 Greedy0.2587、Formal764中位0.1045|`paper_solver_tradeoff_summary.csv`、预算敏感性策略CSV|Formal跨条件中位0.104510 ms；C/K3来自独立CPU微基准|计时协议不同，不能当在线Selection Call；不推出生产P95|保留各自协议；不与16.381 ms接口或Control混减|
|VI-F，331–332：旧216全部正确+新Exact/Greedy闭环|旧correctness216份；`paper_hybrid_gpu_gate_{a,b}/gate_*_result.json`|旧门禁216 PASS；新两条分支局部Cold/组件合同PASS|旧PASS不能升级为当前Strict、完整Cold及组件切片认证；新门禁每分支一例|保留但拆开认证级别，删除无范围的全局正确性暗示|
|VI-F，333–335：Kimi TP2固定MLA16K，8K gap多760.19/759.08 ms|Phase3五对+十份record.json|均值760.191236、中位759.080250，5/5正，两个rank与14项检查通过|已验证内部max-rank匹配到单输出窗口；不是TTFT/策略排名/完整周期，也非当前Qwen主Runner已支持TP2|保留，注明独立跨架构适配与指标窗口|
|VI-F，339–340：Selection16.381、Control10.139s、Full差+1.704/+0.781s|P0同一15摘要，main15_paired|均可复现；Full差+1704.020552/+781.348442 ms|已验证同协议完整实验周期，含审计/形状预热；不代表纯Serving不可避免成本|保留负面结果；不可用后来批次或Legacy131.904替换|
|VI-F，341–342：Legacy控制131.904、Epoch2425.641/2952.253/1875.991、CI含0|旧optimized_formal 216个record和冻结comparison.json|全部复现；共同+策略观察91.210358、协调39.525313、分配0.928711；另输入构建0.239705|已验证独立Legacy协议，非当前Strict路径；CI读取冻结bootstrap，未重新采样|保留但独立标协议；不串成10.139s→131.904ms的优化收益|
|全文8张“暂时无法在飞书文档外展示”表|用户附件；`paper_version.json`|只能核对周围文字，不能确认隐藏单元格与表头|UNKNOWN；不得根据推测代填后签READY|唯一最小补证需包含可见表格导出并按同一主清单封存|

## 总体判定

**NEEDS_TARGETED_FIX。** 核心TTFT、Formal/Strict分配质量、长上下文模型、G2和Kimi数字可追溯；存在可定位的正文数值/分母错误及隐藏表格缺口。无需新GPU或重新拟合模型，不能因此判定研究核心结果不可用；修复前也不能对完整Section VI签署READY_FOR_PAPER。
