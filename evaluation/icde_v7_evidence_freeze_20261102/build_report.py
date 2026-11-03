"""汇总已核验结果，给出论文定稿的局部修复清单，不修改正文。"""
import csv
import hashlib
import json
import os
from pathlib import Path

OUT=Path(os.environ.get('FLOWSTATE_AUDIT_OUTPUT', str(Path(__file__).resolve().parent)))


def main():
    stats=json.loads((OUT/'audit_statistics.json').read_text())
    cases=list(csv.DictReader((OUT/'ten_difference_cases.csv').open()))
    version=json.loads((OUT/'paper_version.json').read_text())
    text=f'''# ICDE v7.0 Evaluation证据冻结与工作负载结构审计

**最终裁决：NEEDS_TARGETED_FIX。** 主要请求级TTFT、Formal/Strict分配统计、模型留出精度、受控分支GPU及Kimi恢复数据均能复核。当前不能签署READY_FOR_PAPER：正文三个CUDA Prefill数值与同批原始记录不符，短上下文误差分母表述不一致，附件还有8张表格不可见；部分来源和正确性范围需补充限定。这些问题可以用已有证据定向修复，未发现必须启动新实验才能解决的核心结果矛盾。

## 1. 版本、范围与复核方法

以用户本轮提供的v7.0全文为正式审计对象：`{version['path']}`，SHA-256=`{version['sha256']}`。全文{version['lines']}行，Section VI始于266行；本报告与主张矩阵以附件行号定位。**没有修改论文。**

直接核对573个原始Formal评估行及派生兼容/位置/收益矩阵、四策略记录；重新评价已保存选集的E/G/J并用Pending位掩码DP独立核对K_min。没有调用Optimizer产生新选集、没有重跑Exact/基线或拟合模型。396门内和Strict20的最优性使用已经保存的枚举证书和目标记录，本轮不新增枚举实验。

另直接读取P0十五臂原始摘要、G2四十臂摘要与CSV、C-formal独立六十臂摘要、历史216运行/864请求及216份旧正确性记录、旧Resume-Epoch 216份record、Kimi十份两rank原记录。校准误差从12个原预测点重新计算。源文件SHA读取前后相同，见[source_manifest.json](source_manifest.json)。原Parquet/AgentX全量轨迹的来源与Token缺口采用已有来源审计，**本轮没有重新解码整套轨迹或恢复遗失Token**。

|证据域|独立来源/快照|预算或重复|使用范围|
|---|---:|---|---|
|历史OpenHands Formal|168快照|3比例预算=504；另K1–4=672|逻辑分配，不是Strict GPU|
|历史AgentX Formal|23快照|3比例预算=69；另K1–4=92|逻辑矩阵；原Token与部分兼容/别名有缺口|
|Strict20|5个受控真实轨迹衍生快照|K1–4=20；七策略140结果|严格Epoch下CPU模型化分配|
|v7.0三策略主实验P0|1个合成Shared/Branched C/K3|5轮×3独立生命周期=15；每臂4串行Pending|请求级GPU与完整实验周期，原始TTFT时间戳复核|
|较早T2-G2|同类C/K3，独立协议|60单Pending正式生命周期|冻结选集协议；不混入P0数值|
|C-formal FS/SA/WFC|独立C/K3协议|60主臂+2门禁|另一组请求级基线，不是Marconi/KV五轮主表|
|B5-C4/K2 G2|1个真实来源受控组合|2策略×4Pending×5轮=40|在线Exact/Greedy及真实Native/COW/Cold|
|旧OpenHands GPU|24个冻结快照|3策略×3次=216生命周期/864请求|旧1输出Token TTFT与旧门禁|
|Legacy控制/完整Epoch|相同24个快照，另一采集协议|216旧优化运行|131.904ms与2425.641ms所属协议|
|Kimi Phase3|固定MLA16K、G0/8K|5对、10独立生命周期|两rank内部恢复窗口；非策略间TTFT|

同一快照的多个K不是独立工作流；同一C/K3的重复臂不是多个真实工作负载。不得合并上表的统计分母或计时窗口。

## 2. 最高优先级问题与建议修复

|问题|v7.0位置|现有证据|最小修复|
|---|---|---|---|
|CUDA Prefill三数不符|VI-B 287|文中233.5/700.4/648.5；同一P0五轮逐轮和的中位数为**236.509439/699.810280/653.240814 ms**|用同源15臂summary重导正文及表；不是换协议或新实验|
|短上下文误差分母错配|VI-C 301|10.720016%–34.929495%分母为完整实测CUDA Prefill，预测值为同Pending零缺口基线+Φ|改标签；不能称该百分比为Φ增量的相对误差，更不能称理论保守上界|
|GPU选集来源叙述不准|VI-A 274|P0主实验是Scheduler内在线选择并核对冻结参照；较早T2-G2才加载冻结选集|分开两个协议，不把“预注册参照”写成主实验只加载选集|
|第十个差异不是共享竞争|VI-D 308|OpenHands g051-round5/K3无跨Pending共享；J相差0.047745076924 modeled ms|AX9例与OH1例分述，不将十例统一归因公共祖先|
|旧PASS容易被升级为新Strict认证|VI-F 331|旧216通过旧检查，无逐请求Cold/完整存储切片及新Strict新鲜度闭包|明确旧门禁；严格链路用后续Hybrid/Native门禁单列支撑|
|表格未导出|VI的8处飞书占位|只有周围文字，表头/单元格UNKNOWN|补充静态表格CSV或完整导出，按本主清单逐格核对|

TTFT 470.208/429.711 ms、长上下文2.682%以及G2的34.71/36.71 ms没有因上述局部错误失效。不能因为存在错误而删掉负结果或更换样本，也不能把局部可修复问题泛称为所有结果不可用。

## 3. VI-D：573比例预算为什么同集

|来源|快照/条件|跨Pending正收益共享|收益重叠|K<K_min|共享且紧预算|四策略同集同J|
|---|---|---:|---:|---:|---:|---:|
|OpenHands|168/504|0|504|90|0|504|
|AgentX|23/69|66|33|0|0|69|

**OpenHands：可分离支持、同链冗余和这些预算下的排序一致。** 每个正收益候选最多服务一个Pending，同一Pending至多需其最佳代表；等大小基数预算可化为各Pending最佳收益Top-K。浅历史与深历史有重叠，但最深代表入选后较浅项的边际归零。原比例预算在这些具体数据上使PCB/LDR/WFC与Marginal Greedy截断后终集一致；90个紧预算条件说明“预算充足”不能单独解释全部OpenHands结果。不存在“任何按位置排序都一定最优”的一般结论，固定K3的小反例正好否定这种外推。

**AgentX：当前需求窄、相对总候选的比例预算足够。** 22/23逻辑快照有共享，达到当前全域最深前沿只需0–2项；候选n=6–149，而25/50/75%预算在69条件中均达到K_min。四策略实际都选中相同覆盖代表，额外候选没有正边际。不能由此推导“一般共享结构下Greedy最优”。

**不是因为完全没有集合效应。** 原冻结统计显示531/573条件沿已存Greedy轨迹出现边际抑制，509/573出现定义内稳健独立/边际排序逆转；这些变化没有在原预算边界改变终集。多数条件最小代表唯一，列表顺序或tie-breaking不能解释全部573同集。

这里的共享是**冻结收益矩阵的支持集**，不是对原始自然Workflow无共享/有共享的物理判断。历史OpenHands按Session/逻辑兼容建模，没有测出共享不代表真实System前缀不存在；AgentX逻辑边也不能替代实际Token前缀验证。

## 4. VI-D：764固定预算与全部10个差异

|来源/K|条件|逻辑共享|K<K_min|共享且紧预算|不同J|
|---|---:|---:|---:|---:|---:|
'''
    for key,r in stats['fixed'].items():
        text+=f"|{key}|{r['conditions']}|{r['shared']}|{r['tight']}|{r['shared_and_tight']}|{r['different_J']}|\n"
    text+='''
**754/764同集同J，10/764异集且异J；四策略间没有发现“异集但等J”的条件。** 成对差异CSV有14行，是10个条件中的多个基线比较，不能算14个工作负载。九个AX差异来自九个不同快照的K1，剩下一个OH条件也不是第十条共享轨迹。

### 全部差异条件（单位：modeled ms）

|来源/快照|n/m|K/K_min|共享|FS J|PCB J|LDR J|WFC J|结构归因|
|---|---|---|---|---:|---:|---:|---:|---|
'''
    for r in cases:
        explanation='共享累计收益优于深单请求推荐' if float(r['PCB_J_ms'])>float(r['FlowState_J_ms'])+1e-9 else '覆盖数优先误选浅共享状态'
        if r['population']=='OpenHands':explanation='无共享；不同T下同L收益排序与ID截断不同'
        text+=f"|{r['snapshot_id']}|{r['candidate_count']}/{r['pending_count']}|{r['K']}/{r['K_min']}|{'有，逻辑矩阵' if int(r['shared_candidates']) else '无'}|"+'|'.join(f"{float(r[k]):.6f}" for k in ('FlowState_J_ms','PCB_J_ms','LDR_J_ms','WFC_J_ms'))+f'|{explanation}|\n'
    text+='''
- AX g000/g007/g014：FS选择共享累计价值，PCB/LDR偏好深专属项；WFC与FS一致。其余六例相反：FS/PCB/LDR偏好深专属项，WFC因推进Pending数量优先选择浅共享项。两类方向都保留，不只展示一种基线的失败。
- AX的K2–4全部同集同J；共享22快照中，K1也只有9个发生紧预算分歧，另13个共享快照K1持平。共享本身不是差异的充分条件。
- OH g051-round5/K3：FS/PCB保留A5、B5、D4，J=307.027617198；LDR/WFC保留A5、B5、C5，J=307.075362275。C5和D4的L都为7808，但T分别7808和7999；Phi收益略不同，按ID截断与按收益截断出现分歧。没有跨Pending共享，不是1e−9容差或列表顺序造成。
- **K1只有一次选择。** AX九例证明跨请求独立收益汇总/覆盖数规则可能不同，不能用它们证明多步动态边际或复杂联合替换的独立优势。
- Formal固定K扫描没有非嵌套替换，既有四策略2292个相邻预算比较及门内Oracle297个比较均为0。该事实与Strict20非嵌套现象属于不同域，不冲突。

完整选集、E/G、支持集与源行见[ten_difference_cases.csv](ten_difference_cases.csv)。包含共享+紧预算的9个Formal快照没有因此获得Native Resume认证；AgentX缺原Token、部分Formal/Raw兼容关系不一致及旧物理别名，不能忠实迁移为当前严格GPU重放。原审计六个WFC失败例推荐GPU回放数为0，本轮没有制造替代历史。

### 396门内Oracle

99个快照（OpenHands90、AgentX9）×4预算=396条件，准入n≤12、m≤4、K≤4、Σ组合数≤794；另368条件全部N/A。读取冻结枚举选集后用原矩阵重新评价，最优数**FS396、PCB395、LDR395、WFC394**，与正文一致。

这是冻结Formal矩阵内的CPU最优证书，不是当前Engine可执行性或GPU收益。旧AgentX K1–8扫描使用特殊解析证书/不同离线枚举规则的184个条件，不能与此396直接合并为统一门禁或分母。OH g051 n20超本门禁，即使结构可解析，也不冒称已执行Oracle。

## 5. Strict20与受控分支GPU：可支持什么

Strict20来自四个独立来源受控组合w1–w4及B5-C4，共5个实际Runtime采集的严格快照，各n7/m4、K1–4。**它们是Trace-derived Controlled Composition，不是自然同时Ready/Fork，也不是主实验C/K3的合成ASG工作负载。** 对应关系是都形成层次共享支持、收益重叠和预算竞争；没有证据说明它们与AgentX九例具有相同真实执行历史。

|方法|Strict20达到Exact J|
|---|---:|
|Exact|20/20|
|Marginal Greedy|5/20，均K1|
|固定首改善Greedy+1-Swap|18/20|
|WFC / StandaloneTopK / PCB / LDR|各5/20|

五快照均K_min=4，K4时存在全域最深前沿覆盖集，但Greedy仍未必找到；“预算足够”只说明存在可达集合，不保证任何选择路径最优。既有最优集合证书20/20唯一，相邻预算15次转移中11次结构性非嵌套，不能以tie-breaking消除。1-Swap失败为w1/K3、w2/K3；从其固定规则产生的局部终集，1换1及2换2均不能改进，但从原Greedy存在不同的两步首交换顺序能到Exact。因此只说明**预注册固定首改善**存在局部陷阱，不是所有局部搜索都必败。

B5-C4/K2中，K1的N20 J=228.054813；K2 Exact={N18,N27}、J=48.228729，Greedy={N20,N18}、J=130.803084。从40个正式GPU原始记录和摘要核验：A/B前沿相同；C/D E2432→3520，实际Prefill分别2003→915、2054→966；五轮Greedy−Exact TTFT配对差中位C **34.707ms**、D **36.712ms**。四Pending等权差五轮中位17.736057ms（四舍五入17.74ms），具体全精度见`audit_statistics.json`。

A/B配对差中位分别−0.094/+0.129ms；B有−15.197/−12.800ms两轮Exact更慢，全部保留。旧checks.json或其他汇总中的B≈0.69/0.70为两组中位之差，不可替代配对差中位0.129。主文C/D 34.71/36.71正确。

此GPU证据支持该固定受控实例中组合分配减少真实恢复计算，不能由模型化J直接推出GPU时延，也不能把EPOCH_ADMISSIBLE或Native命中升级为跨Workflow张量血缘证明；来源血缘仍为PROVENANCE_UNVERIFIED。

## 6. GPU主实验、旧24快照与模型/CPU口径

### 6.1 v7.0五轮P0主实验

所有数值从同一15份原始summary重算；TTFT字段与first_token_ns−start_ns逐项相等。

|策略|四请求ΣTTFT五轮中位ms|四请求ΣCUDA Prefill五轮中位ms|Control T0–T3中位ms|Full T0–T4中位ms|Selection Call中位ms|
|---|---:|---:|---:|---:|---:|
'''
    for tag,label in [('fs','FlowState'),('mc','Marconi-Evict Adaptation'),('kv','KVFlow-inspired')]:
        r=stats['main15'][tag]
        text+=f'|{label}|'+ '|'.join(f'{r[k]:.6f}' for k in ('sum_ttft_median_ms','sum_cuda_median_ms','control_median_ms','full_median_ms','selection_call_median_ms'))+'|\n'
    text+='''
配对先逐轮相减，再取五轮中位：FS−MC ΣTTFT=−470.207613ms、FS−KV=−429.710719ms；逐轮相对率中位64.226378%/61.735126%，5/5请求总和改善。对应Full Cycle差为+1704.020552/+781.348442ms，正文1.704/0.781s成立。不能把策略组中位数相减当配对差，也不把ΣTTFT当四请求连续执行墙钟或吞吐。

主实验来自声明ASG/前缀/访问事件的合成Shared/Branched C/K3，在真实Qwen中物化并恢复；不是原始OpenHands自然并发。策略均为共同候选域受控保留规则；Marconi-Evict Adaptation和KVFlow-inspired不代表官方完整Serving系统。R3的MC因资源门禁修复晚约12分钟续跑，应随文披露；不是删掉该轮的理由，也不能宣称五轮统计显著性。

与较早T2-G2/正式单Pending60臂、FS-SA-WFC的C-formal60臂、20261101三策略15臂、20261102 Serving两臂严格分开。本轮也读取C-formal60份摘要，其C_B FS−SA中位约−163.2ms、FS−WFC约−84.4ms；不把这些数字混入v7.0的Marconi/KV表。

### 6.2 旧24个OpenHands快照

216不同run_id、864请求、216份旧correctness和九类关键文件均在；直接重算逐请求→运行均值→快照内三重复均值→24快照等权。逐快照相对率平均为LRU **43.057902%**、Marconi **24.264401%**；胜/平/负24/0/0和22/0/2。两负例g015-round5、g184-round2必须保留。

可保留为旧版受控请求级TTFT：单输出Token、控制先完成、旧Marconi采用跨度代理/α=1；历史源环境不能完整重建全部执行源码，旧正确性没有逐请求Cold或当前Strict组件切片/连续控制认证。不能把216/864加入当前Strict“全覆盖正确性”分母，或写成官方Marconi系统/端到端性能胜率。

### 6.3 成本模型和求解时间

20个GPU标定点（16位置矩阵+4长gap）；独立测试15个目标/gap格含3个G0，性能误差只对12个非零G格；每格2预热+12测量，共210次，未参与拟合。T校准32K–128K，测试48K/80K/112K。M0/M2 MAPE=15.291275%/2.682480%，MAE=287.514084/31.179583ms，最大相对误差28.036457%/8.179469%。这些数字与正文相符。

短上下文T≈3471–7168，低于T最小标定32768，G小于4096也缺校准覆盖；是外推，不适用长上下文MAPE。19/19方向一致来自T2-G2、C-formal、G2三套独立GPU协议的19类非零配对比较；另外5类同前沿对照完整保留，C-formal同前沿残余6.6/21.1ms不能被静默筛掉后宣称完全相等。

10格的10.720016%–34.929495%是**零缺口基线+Φ预测完整Prefill时间**相对于完整实测Prefill的误差；不是纯增量Φ误差。21个模型敏感性条件Greedy/Exact选集变化0，不能由预测精度改善声称这些实例的分配选择收益。

CPU Exact0.223828、1-Swap0.220940、Greedy0.097175ms来自Strict20独立微基准：每方法20条件×120次混合计时中位，准备不计入，不能当线上P95。Formal764的0.104510ms是各条件批均调用中位的跨条件中位。C/K3的0.2587ms属于另一独立CPU分组；均不能与P0 Selection Call16.381ms混减或把模型化J减CPU时间推导GPU收益。

## 7. VI-F正确性与开销的允许边界

- 新Hybrid门禁分别核验小Exact与大Greedy路径，每分支一个局部GPU恢复案例，Cold匹配、物理状态协调及各自21/22项检查通过；不能扩为全部模型与请求全认证。
- Kimi Phase3固定MLA16K，A E16K/G0、B E8K/G8K，5对10次独立生命周期；两rank及每条件14项检查通过。max-rank内部匹配到单输出的时延差均值**760.191236ms**、中位**759.080250ms**，5/5为正。不是客户端TTFT、策略间加速、Kimi成本模型校准或当前Qwen主Runner已支持TP2的证据。
- 正文Legacy Control=**131.904087ms**，原216个优化协议record直接均值一致：共同观察66.225382+策略观察24.984976=91.210358ms，Allocation0.928711，Reconciliation39.525313，另输入构建0.239705ms。不能把分项缺的0.240ms当测量丢失，也不能串成P0的10.139s被优化至131.904ms。
- 同一Legacy协议完整Epoch均值FS2425.640814、LRU2952.253、Marconi1875.991ms；冻结bootstrap的两组95%CI均跨0，LRU约[−1057.44,2214.94]、Marconi约[−1784.9,699.4]ms。只核对已存CI，不新增bootstrap实验。
- 当前完整控制含研究认证、序列化、日志、形状预热及请求间隙。v7.0保留未证明稳定完整周期净收益的结论是正确的；不能把全部Control或其差额称为Serving不可避免的计算/协调成本。
- 后续20261101 SUCCESS_FULL_CYCLE与20261102精简接口两生命周期支持各自协议的正确性。两次Serving门禁仍保留全部SHA，基线子对象606字节不代表完整包606字节，整包仍约15.4MB；未证明生产或统计性能收益。它们只能作更新的限定补充，不替换v7.0主表或Legacy控制数值。

## 8. 修复清单与唯一最小补证动作

**唯一最小补证动作：提交含8张完整表格的v7.0静态导出（表格CSV或可见表格正文），以本次冻结源清单统一重导Section VI表/文并做逐单元格对拍。** 这是论文证据闭合，不是新实验；不需要新GPU、重跑573/764、补构工作负载、重新拟合Phi或改求解器。

在该动作中同时修正：三项CUDA数值；短上下文分母；P0在线选集来源；AX9/OH1分述；旧PASS与新Strict认证范围；固定首改善1-Swap而非泛指所有局部搜索。可采用的精确TTFT定义是“请求提交至首个有效输出Token到达”，不无条件加入一次完整首次Decode batch。全文保留负面结果、来源标签及R3中断说明。

修复前：NEEDS_TARGETED_FIX。若8张表均与已锁原始数据一致并完成这些限定，可进入最终复核；目前无证据要求为追求正向结论重做实验。若表格另含尚未提供的新统计，则该新增部分保持UNKNOWN，不推测其数值。

## 9. 交付与复算说明

- [evidence_master.csv](evidence_master.csv)：83行指标/口径/来源/正文位置，含NUMERIC_MISMATCH、DENOMINATOR_MISMATCH、QUALIFIED与UNKNOWN状态。
- [workload_structure_summary.csv](workload_structure_summary.csv)：573比例+764固定+20Strict条件=1357行；各域/策略集合不同，不混为总体胜率。
- [ten_difference_cases.csv](ten_difference_cases.csv)：全部10差异条件、四策略选集、J、E/G、实际支持集及源行。
- [claim_evidence_matrix.md](claim_evidence_matrix.md)：逐项论文主张与证据允许范围。
- [gpu_main15_reference.csv](gpu_main15_reference.csv)：原始P0五轮逐Pending同源参考，供导出表格后对拍；不能代称已看到原表单元格。
- [source_manifest.json](source_manifest.json)、[parent_link_verification.json](parent_link_verification.json)：读取前后SHA与573条原始Formal评估行到派生矩阵的逐格联接。
- `audit.py → complete_audit.py → finalize.py → parent_links.py`：本轮只读统计脚本；日志与JSON统计均在本目录。未启动GPU、未执行历史实验、未修改论文或生产源码。

**审计完成后停止，没有自动开展补证或新实验。**
'''
    text=text.replace('这是冻结Formal矩阵内的CPU最优证书','这是冻结Formal矩阵内的CPU最优证书')
    (OUT/'final_audit_report.md').write_text(text)
    print('审计报告生成完成：NEEDS_TARGETED_FIX。')


if __name__=='__main__':main()
