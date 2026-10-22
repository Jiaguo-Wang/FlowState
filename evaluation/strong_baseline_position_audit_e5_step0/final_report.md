# E5-Step 0：强基线与Position-aware决策审计

**CPU审计PASS；Marconi公平复现PARTIAL/UNVERIFIED。** 五个冻结Snapshot、20个预算条件均可核验；Gap-only校准数据完整，本轮未重新拟合。Exact、Greedy、固定首改善1-Swap的选集及前沿在两模型下均20/20一致，不能宣称当前样本已证明位置感知模型带来这些求解器的分配收益。

范围为E2 w1–w4与B5-C4；预算K=1–4，严格边固定、UNKNOWN不放行。所有数值是模型化成本而非GPU时间。未启动GPU、未修改生产代码/冻结数据，20条件不是20个独立工作负载。

## 1. 全部策略选集差异矩阵

N数字为当前冻结Snapshot内的B5C3_NODE_数字缩写，不能跨Engine复用物理ID。括号内预算为同一逻辑Checkpoint数量；全部20条件Marconi均为UNVERIFIED，未编造选集。

|Snapshot/K|Exact|Greedy|1-Swap|WFC|StandaloneTopK|PCB|LDR|
|---|---|---|---|---|---|---|---|
|w1/1|N20|N20|N20|N20|N20|N27|N27|
|w1/2|N18,N21|N20,N21|N18,N21|N20,N21|N20,N21|N25,N27|N25,N27|
|w1/3|N18,N25,N27|N18,N20,N21|N21,N22,N24|N18,N20,N21|N18,N20,N21|N24,N25,N27|N24,N25,N27|
|w1/4|N22,N24,N25,N27|N18,N20,N21,N27|N22,N24,N25,N27|N18,N20,N21,N27|N18,N20,N21,N27|N22,N24,N25,N27|N22,N24,N25,N27|
|w2/1|N20|N20|N20|N20|N20|N25|N25|
|w2/2|N18,N21|N20,N21|N18,N21|N20,N21|N20,N21|N25,N27|N25,N27|
|w2/3|N18,N25,N27|N18,N20,N21|N21,N22,N24|N18,N20,N21|N18,N20,N21|N22,N25,N27|N22,N25,N27|
|w2/4|N22,N24,N25,N27|N18,N20,N21,N25|N22,N24,N25,N27|N18,N20,N21,N25|N18,N20,N21,N25|N22,N24,N25,N27|N22,N24,N25,N27|
|w3/1|N20|N20|N20|N20|N20|N25|N25|
|w3/2|N24,N27|N20,N24|N24,N27|N20,N24|N20,N24|N22,N25|N22,N25|
|w3/3|N24,N25,N27|N20,N24,N27|N24,N25,N27|N20,N24,N27|N18,N20,N24|N22,N24,N25|N22,N24,N25|
|w3/4|N22,N24,N25,N27|N20,N24,N25,N27|N22,N24,N25,N27|N20,N24,N25,N27|N18,N20,N24,N27|N22,N24,N25,N27|N22,N24,N25,N27|
|w4/1|N20|N20|N20|N20|N20|N27|N27|
|w4/2|N21,N24|N20,N24|N21,N24|N20,N24|N20,N24|N25,N27|N25,N27|
|w4/3|N24,N25,N27|N20,N21,N24|N24,N25,N27|N20,N21,N24|N20,N21,N24|N22,N25,N27|N22,N25,N27|
|w4/4|N22,N24,N25,N27|N20,N21,N24,N27|N22,N24,N25,N27|N20,N21,N24,N27|N18,N20,N21,N24|N22,N24,N25,N27|N22,N24,N25,N27|
|B5-C4/1|N20|N20|N20|N20|N20|N22|N22|
|B5-C4/2|N18,N27|N18,N20|N18,N27|N18,N20|N18,N20|N22,N24|N22,N24|
|B5-C4/3|N22,N24,N27|N18,N20,N27|N22,N24,N27|N18,N20,N27|N18,N20,N27|N22,N24,N25|N22,N24,N25|
|B5-C4/4|N22,N24,N25,N27|N18,N20,N22,N27|N22,N24,N25,N27|N18,N20,N22,N27|N18,N20,N21,N27|N22,N24,N25,N27|N22,N24,N25,N27|

完整两两分类见[policy_pair_matrix.csv](policy_pair_matrix.csv)：每条件21个无序策略对，共420对；同集同前沿140、异集同前沿0、异集异前沿280。这个计数不是280个独立收益实例。全部选集、逐Pending T/E/G、共同J、Exact差距沿用并核验E3，见[all_policy_results.csv](all_policy_results.csv)；不能只根据异集认定性能胜负。

|方法|达到共同Position-aware Exact目标的条件数|解释|
|---|---:|---|
|Exact|20/20|完整小规模子集枚举最优|
|Greedy|5/20|各Snapshot仅K1最优|
|Greedy+1-Swap|18/20|固定字典序首改善，严格下降；仅w1/K3、w2/K3仍次优|
|WFC|5/20|与Greedy全部20条件同集同前沿|
|StandaloneTopK|5/20|w3/K3、w3/K4、w4/K4、B5-C4/K4有额外静态冗余|
|PCB、LDR|各5/20|K4达到Exact，K1–3未达到；两者同集|

强基线结论：Exact相对原Greedy的15个优势条件中，13个被既有固定1-Swap完全消除，剩余仅w1/K3、w2/K3。不能继续把w1/K2的Exact对Greedy GPU优势当作Exact相对强局部替换的独立优势；该条件1-Swap与Exact完全同集。

## 2. 冻结Gap-only来源与逐条件对照

来源：`evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/`。`candidate_models.json`中的M0以G/1024为横轴，在0、4、8、16、32、48、64、96、128 Ki Tokens处线性插值；零点固定为0，同Gap校准成本取均值。协议见`evaluation/recovery_model_freeze.py:186–245,308–390`。20个校准点逐条回核两个原始summary，所有非零节点均值与冻结参数一致；不使用E2/E3选集或Held-out数据拟合。

独立性指校准测量独立于本轮分配测试，不声称M0与Position-aware使用互不重叠训练集。Position-aware沿用正式二次Phi。M0首个非零点为G4096、成本238.13663222916668 ms；其校准T为32768–131072，当前Snapshot的T约数千Tokens。首段插值和短T适用性缺少直接校准覆盖，因此本轮只能认证模型决策对照，不能将其直接当作短上下文预测准确性证明。

固定算法采用原E3 Exact枚举、原Greedy选择核心、相同1-Swap规则和StandaloneTopK，仅在独立审计进程替换成本接口；生产文件未变。Exact按1e-9容差、较少项再ID选择同分代表；Greedy及Swap沿用原容差/排序。PCB/LDR/WFC作为冻结强基线比较，不把PCB内部成本构造偷偷替换；WFC/LDR不读取Phi。

下表“同”指两模型选集、全部E/G相同；所有模型各自及交叉评分、完整选集、候选位置见[position_vs_gap.csv](position_vs_gap.csv)，共80行。

|Snapshot/K|Exact|Greedy|1-Swap|StandaloneTopK|
|---|---|---|---|---|
|w1/1|同|同|同|同|
|w1/2|同|同|同|同|
|w1/3|同|同|同|同|
|w1/4|同|同|同|同|
|w2/1|同|同|同|同|
|w2/2|同|同|同|同|
|w2/3|同|同|同|同|
|w2/4|同|同|同|同|
|w3/1|同|同|同|同|
|w3/2|同|同|同|同|
|w3/3|同|同|同|同|
|w3/4|同|同|同|同|
|w4/1|同|同|同|同|
|w4/2|同|同|同|同|
|w4/3|同|同|同|异集异前沿|
|w4/4|同|同|同|同|
|B5-C4/1|同|同|同|同|
|B5-C4/2|同|同|同|同|
|B5-C4/3|同|同|同|同|
|B5-C4/4|同|同|同|同|

全部20个Gap-only Exact最优集合均唯一，Position-aware Exact也属于这些最优集合，不是只因同分代表恰巧一致。

### 唯一模型切换差异：w4/K3 StandaloneTopK

两者均先选公共N20(L2432)和AB共享N24(L3520)。第三项Position-aware选CD共享N21(L3456)，Gap-only选AB共享N18(L3456)。M0下N18/N21独立总收益同分，ID规则选N18；但N18已被N24覆盖，集合边际为零。Position-aware独立收益N18=259.708957、N21=259.762398，微小排序差打破该同分。

|Pending|T|Position-aware E/G|Gap-only E/G|
|---|---:|---|---|
|A|3776|3520/256|3520/256|
|B|3549|3520/29|3520/29|
|C|3840|3456/384|2432/1408|
|D|3968|3456/512|2432/1536|

两个选集按Position-aware评价J=45.066824与122.775420，按M0评价J=68.661954与187.730270，均是前者更低。因此这是静态排序冗余与同分处理的交互，不是“两个模型分别偏好自己的全局最优选集”，也不是Position-aware目标位置项的干净独立消融。现有w4数据仅完成物化/准入，没有这两个策略选集的独立配对GPU恢复时间证据；E4 w1/A/C及B5-D4b不能替代。

## 3. 实现与公平性审计

|策略|实际规则和证据|本轮结论|
|---|---|---|
|Exact|E3 audit.py的全子集≤K枚举，较少项/字典序同分；统一common.score|可复核；离线目标最优，不等于在线净收益最优|
|Greedy|生产GlobalOptimizer._select_candidates，边际J下降、原同分/停止规则；E3独立矩阵评分|可复核，严格边不被旧Workflow谓词过滤|
|1-Swap|E3 audit.py:swap，已选/未选ID排序首个严格改善，重复至局部最优|可复核实验对照，不是生产策略；规则路径影响结果|
|WFC|p1_strong_baseline_validation/wfc.py：动态改善请求数优先、Token推进次之、ID同分|可复核；不读Phi，不能因20/20持平宣称与Greedy一般等价|
|StandaloneTopK|E3 audit.py：静态独立收益总和降序、ID同分、正收益截断|可复核；不更新重复覆盖收益|
|PCB|p1_strong_baseline_validation/pcb.py：每Pending最大独立收益推荐、静态收益排序去重|可复核；与LDR此批一致是观察|
|LDR|p1_strong_baseline_validation/ldr.py：每Pending最深推荐、位置排序去重|可复核；不是全候选按深度直接TopK|
|Marconi Adapt|sota_policies.py:96–140,216–258存在归一化recency＋alpha×归一化FLOP/bytes的代理；B4-C曾用冻结原始last_access_order、alpha=1|本批输入/适配公平性UNVERIFIED，不运行、不填代理分数|

已核验E3、E2及B5-C4 Manifest，P1 common/pcb/ldr/wfc摘要与B4-C冻结锁一致，140行原J与逐Pending结果全部重算一致。E3 Greedy/WFC步骤证据显示公共共享→二请求共享→另一子域→专属推进时赢家一致；部分低位候选在Phi与Token排序中不同，但没有改变获胜项。

Marconi并非没有代码：缺的是本批可公平使用的输入/合同。w1–w4 Strict候选不包含last_access_order、marconi_flop_saved等评分元数据；C4归约输入也未冻结这些字段。历史请求和Census可提供部分原始线索，但不等于完整原生访问时钟。此前B5-B0确认同Workflow访问更新遗漏跨Workflow匹配、父节点跨度代理与原树不同、Mamba FLOPs不能无说明套用于GDN、资源分母和原生驱逐资格不同、缺独立Bootstrap锁。证据：`evaluation/sota_baseline_gate_b5b0/marconi_input_gate.md`、`evaluation/strict_compatibility_b4c/final_report.md`。

最小适配工作包：规范化已发生的原生访问与树父节点；冻结GDN计算/内存分母及recurrent-only动作边界；独立历史Bootstrap或显式预设alpha合同；实现回归和事件完整性检查。工作量不是调用现成select即可完成，现有材料不足以给出可靠天数，也不授权将测试J用于调参。故本轮不声称已战胜Marconi官方系统或可信新适配。

## 4. 论文主张边界

可成立：这些严格Snapshot存在共享、重叠和预算截断；静态TopK及不可撤回Greedy可以损失模型化目标；固定1-Swap已是很强的对照，仍有两个局部陷阱；小规模Exact保证当前冻结目标最优。严格可执行前沿差异是潜在GPU对照依据，不自动等于性能提升。

不能成立：Position-aware已在Exact/Greedy分配上击败既有Gap-only；Exact普遍必要或1-Swap普遍不足；WFC与Greedy数学等价；单请求CUDA收益代表四Pending总体收益；已复现完整Marconi。旧573 Formal条件只能解释可分支持/预算饱和，不升级为Strict Runtime认证。

## 5. 唯一建议的下一项GPU对照

**w1/K3：冻结Exact vs 冻结首改善Greedy+1-Swap，四个Pending分别独立恢复。** 不建议为了Position-aware主张重复w1/K2，也不优先测试受同分/静态冗余混杂的w4/TopK案例。

Exact集合{N18,N25,N27}，J=21.953375；1-Swap集合{N21,N22,N24}，J=31.817384。Exact的A/B E=3392（G256/320），C/D E=4096/4160（G0/0）；1-Swap的A/B E=3648/3712（G0/0），C/D E=3712（G384/448）。这是双向取舍：Exact在A/B更浅，C/D更深，不能只挑C或D宣称总体胜利。

唯一尚缺的研究证据是：Exact相对强局部替换的模型化联合目标优势，是否在四需求的独立真实恢复中仍有对应方向。每Pending、每策略重建同一初始状态，先做8个恢复臂与4个独立Cold，共最多12个生命周期的功能/方向性Smoke；报告全部四请求Prefill及其预定义等权和，并另列控制成本，不称自然并发吞吐。

沿用已通过的w1 Token、Strict观察、Reconciliation、Native Match/COW/Cold与轻量路径。必要前提是CPU核验K3当前绑定和4个驱逐补集，把实验层K2/5项驱逐断言参数化，保持全部安全检查；从冻结选集构建合法Plan，不重选或开发生产求解器。按E4计时审计补齐持久化/首Decode边界，安全Smoke失败即停。若短期无法完成这些前提，不进行GPU，保留模型化结论即可。当前未执行此建议。

## 证据与复现

独立脚本：[audit.py](audit.py)，入口`/home/wjg/uv312/bin/python -B -m evaluation.strong_baseline_position_audit_e5_step0.audit`。完整结果：[results.json](results.json)，模型合同：[model_contract.json](model_contract.json)，Gap Exact同分穷举：[gap_exact_ties.json](gap_exact_ties.json)，来源摘要与不变性：[source_integrity.json](source_integrity.json)。脚本仅写本目录，复核冻结参数而不拟合。完成后停止。
