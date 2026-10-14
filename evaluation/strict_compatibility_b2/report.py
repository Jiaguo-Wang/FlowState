"""从本轮 CPU 结果生成中文报告，不重新选择策略。"""
import json
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text());m=json.loads((OUT/'matrix.json').read_text())
def short(cid):
    return cid.replace('RQ3_G000_','').replace('_TURN_00','')
rows=[]
for a in r['records']:
    eg='；'.join(f"{chr(65+i)}:{p['E']}/{p['G']}" for i,p in enumerate(a['per_pending']))
    rows.append(f"| {a['K']} | {a['strategy']} | {','.join(map(short,a['selected']))} | {a['F_ms']:.9f} | {a['J_ms']:.9f} | {a['gap_vs_opt_ms']:.9f} | {eg} | {'是' if a['same_set_as_flowstate'] else '否'} | {'是' if a['tie_with_different_set'] else '否'} |")
mat=[]
for c in m['input']['candidates']:
    b=m['benefit_matrix_ms'][c['identifier']]
    mat.append(f"| {short(c['identifier'])} | {c['position']} | "+' | '.join(f'{v:.9f}' for v in b)+f" | {sum(b):.9f} |")
steps=[]
for step in r['traces']['4/FlowState']:
    steps.append(f"- 已选 {','.join(map(short,step['prefix'])) or '空集'}：选择 {short(step['chosen'])}；全部剩余边际收益 modeled ms："+'；'.join(f'{short(c)}={v:.9f}' for c,v in step['marginal_ms'].items())+'。')
text='''# Phase B2：严格准入矩阵下的最小 CPU 分配比较

## 判定

**DISCRIMINATIVE（单 Snapshot、局部预算范围）**。在固定 group 0 / round 2 的 8×4 准入矩阵上，FlowState 在 K=1、2、3 时严格优于 PCB/LDR，并与 Exact OPT 一致；K=4 时反而落后 PCB/LDR/Exact OPT 58.401745393 modeled ms。WFC 与 StandaloneTopK 在全部四个预算下与 FlowState 的最终集合、E/G 和目标一致。

因此，本例能区分共享收益选择与逐请求最深推荐，但**不能证明 set-dependent marginal 或位置感知模型相对 WFC/StandaloneTopK 的独立必要性**。也不能证明全局贪心总是最优。Marconi 因当前 B1 缺少访问元数据单列不可比较，不将其缺失写成持平。结论只依赖已实际执行的强基线 PCB、LDR、WFC，未完成七策略全部可比的主张。

## 1. 输入身份与语义边界

唯一分配输入为 `evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json` 的 positive_arm：8个唯一逻辑候选、4个已知Pending、32个唯一配对，11条EPOCH_ADMISSIBLE、21条REJECTED、0条UNKNOWN。各候选Token长度、L与物理prefix_tokens一致；准入边的全部conditions为真，直接比较已有Token数组，核对L≤T。未重新生成Token IDs，未使用未来消息、Pending输出或B1-C执行后的状态构造收益。

T向量按A/B/C/D为4352/3968/4160/4032。完整性要求是B1明确的受控准入合同，不自动升级为任意部署下完整执行认证。`compatibility_benefit.csv` 和 `matrix.json`保留原条件。B1-C只将B1→A标为RESUME_VALIDATED；其余10条仍不具有由B1-C提供的逐边恢复认证。B1原有actual_resume_verified字段原样保留，不抹去同Workflow历史验证。

B1-C三条件PASS及B1摘要一致性已核对，仅用于认证等级。没有用B1-C隔离后的候选集替代B1完整候选集。输入与源代码前后SHA-256完全一致，见input_integrity.json；P1-B的四个实现文件与implementation_lock.json逐项相符。

逻辑预算K=1–4，单位是等大小保留项。本轮包装器memory_bytes=1仅编码逻辑计数，不是实测字节。原池28槽、8驻留、20空闲；本轮没有执行驱逐或容量约束，不是GPU物理压力实验。

## 2. 共同目标与完整收益矩阵

使用未修改的 `flowstate/recovery_model.py:40–80` 与 `evaluation/FORMAL_RECOVERY_MODEL.md`：g=G/1024，t=T/1024，Phi=37.828150g+0.345974143gt−0.156201917g²，单位modeled ms。无拟合、外推或参数调整。

准入边b=Phi(T,T)−Phi(T−L,T)，其他边b=0。F(S)=逐Pending最大b之和；E为所选准入候选最大L（默认0），G=T−E；J为逐Pending Phi之和。J(空集)=622.330421990672 modeled ms，F(空集)=0。穷举256个集合验证J=J(空集)−F，误差小于1e−9ms；所有收益非负。

缩写A1表示完整ID RQ3_G000_A_TURN_001，其他同理。Pending A表示RQ3_G000_A_PENDING_TURN_003，其他同理。

| 候选 | L | 对A收益 | 对B收益 | 对C收益 | 对D收益 | 独立总收益 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(mat)+'''

## 3. 真实策略定义与适配

- **FlowState**：复用 `flowstate/optimizer.py:38–141` 的原select循环：重算成本差、容差1e−9ms、候选ID同分规则、非正收益停止。独立MatrixOptimizer仅覆盖 `_recovery_cost`，用准入矩阵计算E/G和成本；不修改生产代码，不虚构统一Workflow。旧is_compatible不参与矩阵评分。另以独立显式max-benefit边际计算验证四个预算的原循环输出。
- **PCB**：直接调用 `evaluation/p1_strong_baseline_validation/pcb.py:5–23`；逐Pending取Phi独立收益最大的候选，正收益>0，按收益/候选ID/PendingID排序，去重至K。它不按多Pending收益之和重新推荐。
- **LDR**：直接调用同目录 `ldr.py:4–18`；逐Pending推荐最深候选，按L/候选ID/PendingID排序去重，不调用Phi做选择。
- **WFC**：直接调用 `wfc.py:4–20`；动态按可推进Pending数q、推进Token总量r、候选ID排序，q=0停止。不调用Phi做选择。
- **StandaloneTopK**：独立矩阵适配 `evaluation/rq3_sanity_structure_audit.py:1116–1131`：F(单候选)>1e−9，按独立收益降序/ID升序取K，返回ID排序集合；不更新边际。只替换旧谓词输入，不改变静态选择规则。
- **Exact OPT**：穷举所有大小≤K的子集，分别检查9、37、93、163个集合；保留全部最优集合。成本同分1e−9ms时，报告代表优先较小集合，再按ID字典序；不将代表选择规则当作生产Exact规则。四个预算的最优集合均唯一。
- **Marconi Adaptation：NOT_COMPARABLE**。`evaluation/sota_policies.py:100–150` 使用归一化recency+alpha×归一化FLOP/byte效率，alpha由 `evaluation/sota_metadata.py:19` 固定为1；增量同Workflow祖先span定义见该文件106行起。原collector在 `rq3_openhands_neutral_collector.py:483–510` 基于实际命中的e更新last_access_order，再在803–815行提供元数据。B1没有保存逐候选last_access_order或完整历史请求e；仅有物化census不能还原访问事件。B1-C是另一个生命周期，即使输入相同也不拼接其访问记录冒充B1。未构造替代评分、未运行Marconi；CSV保留四行不可比较项。

所有已运行策略使用同一8个候选、4个Pending和矩阵；LDR/WFC在决策时不读Phi，但最终统一评分。PCB/LDR/WFC直接导入冻结模块，无文件改写。数值比较容差1e−9 modeled ms；不同顺序不视为不同集合。

## 4. 完整分配结果

F为收益、J为成本、最优差距为J−J_OPT，均为modeled ms。E/G单位tokens，按A/B/C/D列出。完整逐请求成本在results.json与allocation_results.csv。

| K | 策略 | 所选顺序/集合 | F | J | 最优差距 | 各Pending E/G | 集合等于FlowState | 同收益异集合 |
|---|---|---|---:|---:|---:|---|---|---|
'''+ '\n'.join(rows)+'''

所有已运行结果均未出现“同收益但不同集合”的配对；确有选择顺序不同，例如K=2 FlowState为B1,A2，StandaloneTopK返回A2,B1，集合及目标完全相同。不可将其计为胜负。

## 5. 共享收益、预算与失败机制

B1覆盖4个Pending，含3条跨Workflow准入边；其他7个候选各仅对自己Workflow有正收益。K=1时B1独立收益364.177368680，高于最深单请求A2的164.197398332，因而共享关系确实进入并改变了决策。PCB/LDR只推荐各请求最深项A2/C2/D2/B2，不推荐浅层B1，失去低预算共享收益。FlowState在K=1/2/3相对PCB/LDR降低成本199.979970348 / 116.306743283 / 30.178200543 modeled ms。

达到全部候选最大前沿的K_min_frontier=4；必须同时持有A2/B2/C2/D2。K<4是真正的逻辑前沿取舍，绝不意味着物理池耗尽。K=4时该四项恢复成本为0；贪心已选B1，后续不交换，最终只能再选三个深层项，B保留在2432而非3968，剩余1536-token缺口，成本58.401745393ms。该失败是本例正式结果的一部分；未增加交换步骤修复算法。

FlowState选择过程中全部剩余候选边际如下：

'''+ '\n'.join(steps)+'''

A2选入后A1边际为0；C2选入后C1边际为0；D2选入后D1边际为0。全深层最优集合上B1也是零边际；但在FlowState的K=4最终集合上B1仍为B提供正收益，不能误写成已选的完全冗余项。K=4失败是被早期选择占用预算且无替换，不是忽略零边际终止。

WFC先按q=4选择B1，再按剩余Token推进选A2/C2/D2；本实例的位置与Phi边际排序一致。StandaloneTopK的静态F排序前四也是B1/A2/C2/D2；尽管存在覆盖重叠，更新边际并未改变本次前四选择。持平是本矩阵排序结构的观测，不是三种算法一般数学等价。

## 6. 验证、复现与停止边界

7项CPU测试全部通过：拒绝边为零、B1跨Workflow正收益不被旧谓词过滤、空集与256集合F/J等价、重复覆盖不相加、全部子集和增量边的边际一致、原贪心与独立边际选择一致、所有策略预算约束及穷举最优性。日志见tests.log。未执行GPU、Engine、Replay或任何旧策略总体重评估。

复现命令（仅CPU）：

```bash
/home/wjg/uv312/bin/python -B -m evaluation.strict_compatibility_b2.evaluate
/home/wjg/uv312/bin/python -B -m unittest evaluation.strict_compatibility_b2.test_evaluate -v
/home/wjg/uv312/bin/python -B -m evaluation.strict_compatibility_b2.report
```

原始来源、脚本与报告哈希见evidence_manifest.json，冻结输入检查见input_integrity.json。本轮所有产物独立位于evaluation/strict_compatibility_b2/，未修改旧504或573条件、B1/B1-C及生产代码。

**最终回答**：这张真实Runtime支持的准入矩阵有可解释的共享与预算区分力，FlowState相对PCB/LDR在K=1–3获益、K=4失败；对WFC/StandaloneTopK没有额外收益。它证明这一个模型化问题值得区分分析，不证明集合边际算法必要或总体性能更好。Marconi元数据缺口必须保留。完成本轮CPU比较后停止，不启动Runtime策略比较。
'''
(OUT/'final_report.md').write_text(text)
paths=[p for p in OUT.iterdir() if p.is_file() and p.name!='evidence_manifest.json']
manifest={'verdict':'DISCRIMINATIVE','说明':'单Snapshot，K=1–3胜PCB/LDR，K=4负；WFC与StandaloneTopK全持平；Marconi不可比较','files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('报告已生成')
