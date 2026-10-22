"""从已核验的CPU结果生成中文报告，不重新执行策略。"""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text());v=json.loads((P/'validation.json').read_text());integrity=json.loads((P/'input_integrity.json').read_text())
def write(n,s):(P/n).write_text(s.strip()+'\n')
def short(ids):return ','.join(c.replace('B5C3_NODE_','N') for c in ids)
names=['FlowState','PCB','WFC','LDR','StandaloneTopK','Exact OPT']
table='|K|'+'|'.join(names)+'|\n|---|'+'---:|'*6+'\n'
sets='|K|策略|选集／选择顺序|J（模型化ms）|Exact差距（模型化ms）|\n|---|---|---|---:|---:|\n'
for k in range(1,5):
 table+='|'+str(k)+'|'+'|'.join(f"{next(x for x in r['rows'] if x['K']==k and x['policy']==name)['J_ms']:.9f}" for name in names)+'|\n'
 for x in r['rows']:
  if x['K']==k:sets+=f"|{k}|{x['policy']}|{short(x['selection_order'])}|{x['J_ms']:.9f}|{x['gap_to_exact_ms']:.9f}|\n"
benefits='|节点|L|正支持|A|B|C|D|\n|---|---:|---|---:|---:|---:|---:|\n'
for c in r['input']['candidates']:
 cid=c['identifier'];benefits+=f"|{short([cid])}|{c['position']}|{','.join(r['supports'][cid])}|"+'|'.join(f'{b:.9f}' for b in r['benefit_matrix_ms'][cid])+'|\n'
write('input_integrity.md',f'''# 输入、版本与预算合同

**PASS**。只读取B5-C3和B4-C冻结证据，前后SHA全部一致。完整路径与摘要见input_integrity.json；B5-C3及B4-C各自evidence_manifest逐文件检查通过。

来源：`evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json`、`physical_state_census.json`、`sharing_support_matrix.json`。七个唯一候选、四个唯一Pending，28个完整无重复配对，13准入、15拒绝、0未知。逐边核对Token前缀、L≤T、Epoch及配置身份。正收益与B5-C3单边记录一致；拒绝边收益为零。四个T按A/B/C/D为4160/4096/3904/3566，不使用后续执行状态。

## 等大小证据

每个候选有一个独占Slot及完整Conv/Temporal组件。以冻结physical_state_census.json → audit_before.component_slices → byte_ranges求和，每项为 **{integrity['equal_checkpoint_bytes']}字节**：Conv为24×49152=1179648字节，Temporal为24×2097152=50331648字节。七个Slot互异。逻辑成本均为1个完整Checkpoint；K=1、2、3、4，无加权预算或候选删除。以上是Recurrent切片大小，不包含Attention、不声称归还Slot释放预分配HBM。Node/Slot仅用于历史证据，不能充当未来生命周期映射。

## 锁定策略

PCB/LDR/WFC/common.py直接复用`evaluation/p1_strong_baseline_validation/`，当前SHA与B4-C/input_integrity.json一致。

生产optimizer.py在R2阶段已变化；本轮从Git提交`{integrity['historical_optimizer_commit']}`读取历史文件，核对SHA `{integrity['historical_optimizer_sha256']}` 后隔离保存为frozen_optimizer.py。执行原Greedy循环，仅按B4-C的MatrixOptimizer方式替换成本接口，使严格矩阵直接决定前沿。未调用后来改变的生产选择入口，未修改任何生产文件；占位Workflow字段不参与评分。

StandaloneTopK完全沿用B4-C/evaluate.py定义：静态总收益降序，ID升序，收益大于1e-9，截取K；最终输出ID排序。results中的selection_order对该策略表示静态排名次序，而非动态决策。Exact没有决策顺序，展示字典序代表集合。PCB/LDR保存原推荐记录；WFC保存原动态排序。

Phi源码摘要与B4-C一致，元数据逐项一致：a=37.828150，b=0.345974143，c=-0.156201917，G和T以1024归一化。未拟合。比较容差为B4-C既有1e-9 ms；Greedy仍沿用原源码的非正收益停止、容差及ID同分规则。

## 独立核验与复现

`validate.py`从原始边和Token位置出发，以Decimal独立实现冻结公式；128个子集、24个策略结果、448个加一候选的边际变化、4个Exact预算全部通过。详见validation.json。没有导入或调用Runtime启动接口。

在仓库根目录依次运行：
```
/home/wjg/uv312/bin/python -B -m evaluation.multishare_allocation_gate_b5c4.evaluate
/home/wjg/uv312/bin/python -B -m evaluation.multishare_allocation_gate_b5c4.validate
/home/wjg/uv312/bin/python -B -m evaluation.multishare_allocation_gate_b5c4.report
```
这些入口只写本轮新目录，不修改冻结源。
''')
write('candidate_structure_audit.md',f'''# 候选结构与冗余

完整保留七个候选；以下数值为单边模型化收益ms，N为B5C3_NODE_的简写。

{benefits}

N21和N27的准入及正收益支持都恰为C/D。N27位置3520比N21的3456更深，在C/D收益分别严格增加2.438766398和2.438073846 ms，在A/B同为零。因此相同逻辑成本下N27逐坐标支配N21。包含N21而不含N27的集合可替换为N27且不变差；严格单边支配不保证对已充分覆盖的集合有严格增益。二者都被选择时N21的删除贡献为零。

穷尽42个有向候选对，唯一严格支配是N27→N21；21个无序对中没有完整收益向量相同的候选。条件边际同分仍会出现，例如N18/N27都保留后，N20与N21都为零边际。完整替代诊断见results.json，每次替代固定其他选集，不产生新策略。

K_min_frontier=**4**。全候选前沿为A=4160、B=4096、C=3904、D=3520，分别必须由N22/N24/N25/N27达到，所以这四项既必要又充分。D的T=3566仍有46-token残差，J(C)=1.753119344 ms；预算饱和不等于零Gap。

正支持域为ABCD、AB、CD、A、B、C，其中CD重复两次。全部两两嵌套或不交，且与真实Token前缀树一致。不存在非层次交叉支持。该固定快照在等大小、纯基数、固定非负收益、单调位置代价和无额外物理耦合的限制下适合按树及最近保留祖先状态做精确DP；本轮没有实现DP，不主张一般Runtime都满足这些前提。
''')
write('allocation_difference_analysis.md',f'''# 完整结果与机制归因

一个快照、四个重复预算条件，不是四个独立工作流。单位均为模型化恢复成本ms。

{sets}

每个Pending的T/E/G/成本、全部Greedy候选边际、WFC排序、PCB/LDR推荐及静态排名见results.json和policy_comparison.csv。选集中的每项删除贡献及一换一替代诊断也已完整保存。Exact枚举全部128子集；各预算内检查8/29/64/99个集合，四个预算的最优集合均唯一（1e-9容差）。

## Greedy与Exact

Greedy依次选择N20、N18、N27、N22，新增收益分别为364.116170072、97.251729525、82.574355071、17.134832583 ms。

- K1：N20覆盖四请求，总独立收益最高；Greedy=WFC=StandaloneTopK=Exact。
- K2：Greedy已选N20无法回退，选N18后J=130.803083830。Exact直接选N18+N27，J=48.228728759；用N27替代N20改善82.574355071。此时深层两分支覆盖比全局浅祖先更有效。
- K3：Greedy选N20+N18+N27后，N20删除贡献已经为零；它占用了预算。Exact选N22+N24+N27，改善31.817384152。该差距涉及用两个深层AB专属状态替代N20及AB共享节点，不能把一次不可回退解释成全部差距的单独因果消融。
- K4：Greedy再选N22，保留零贡献N20；Exact用N22/N24/N25/N27达到全候选前沿。差距29.340776833等于B额外恢复成本14.682551570与C额外恢复成本14.658225263之和。Greedy并没有因为K达到K_min而自动饱和。

## 强简单策略与互补现象

PCB/LDR分别推荐各Pending最深项，并按既有分数排序：N22、N24、N25、N27。两者本轮所有预算相同，K4才Exact。WFC按改善人数、总Token推进、ID排序，始终与Greedy选集及顺序一致，仅K1 Exact。因此PCB/WFC在K2和K3均失败，二者互补覆盖Exact的旧实例现象在此不成立。

FlowState相对PCB及LDR均为3胜0平1负；相对WFC为0胜4平0负；相对StandaloneTopK为1胜3平0负；相对Exact为0胜1平3负。这些是同一快照四个条件的描述计数，没有显著性或总体推断。

## 静态重复覆盖

StandaloneTopK的静态顺序为N20、N18、N27、N21。前三步与Greedy相同。K4继续选同属CD的浅节点N21，因N27已经提供更深前沿，真实集合新增收益为0；Greedy更新边际后选N22，改善17.134832583 ms。此结果在固定同一Phi下支持动态边际更新相对该静态规则的作用，但不能作为位置感知项的独立消融证据。

多共享确实产生非平凡组合：最优集合随预算并不嵌套（K1的N20在K2最优解消失），不可回退Greedy因此受限。没有证据证明Greedy普遍优于工作流感知规则；树形结构也可能容许精确求解。
''')
write('final_report.md',f'''# Phase B5-C4 最终报告

**Gate：READY**。这是输入、版本、枚举与比较可复核的门禁，不表示Greedy取胜。

## 主结果

J(S)，单位为模型化恢复成本ms：

{table}

七候选全部保留，逻辑成本相同；13准入/15拒绝/0未知；冻结Phi、Token、来源及配置未改变。历史Greedy按B4-C SHA恢复隔离执行，PCB/WFC/LDR按原锁调用，无UNVERIFIED策略。

## 结论

1. 多共享结构打破此前PCB/WFC互补覆盖Exact的观察：K2/K3两者均非最优。
2. FlowState仅K1 Exact；K2/K3/K4分别多支出82.574355071、31.817384152、29.340776833模型化ms。不可回退的浅共享选择及后续重复覆盖占用预算是可观察的失效机制。
3. FlowState与WFC四条件完全一致。相对StandaloneTopK仅K4更好；该静态策略选中被N27支配的N21，造成零边际重复。
4. K_min=4；树形支持成立。当前事实支持重新评估是否应采用适合受限树结构的精确求解器，不能继续以这些实例论证现有Greedy的普遍优势。本轮不实现、不调优任何新求解器。

## 验证与范围

独立Decimal核对128子集、24策略结果和448个边际变化全部PASS。来源摘要前后不变，详见input_integrity.json、validation.json、exact_verification.json；完整数值及归因见policy_comparison.csv、results.json、allocation_difference_analysis.md。

本轮只有一个Trace-derived Controlled Composition快照；四预算不是四独立样本。EPOCH_ADMISSIBLE不自动成为RESUME_VALIDATED。没有GPU运行、Eviction或Reconciliation，不报告TTFT、端到端收益或真实物理容量竞争。没有进行Phi位置项消融，也没有证明一般非树共享结构的算法性质。

已完成，停止并等待审查，不进入下一阶段。
''')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}
(P/'evidence_manifest.json').write_text(json.dumps({'files':manifest,'说明':'本轮独立CPU产物摘要'},ensure_ascii=False,indent=2)+'\n')
print('报告完成，READY。')
