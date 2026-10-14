"""整理固定解析案例与B2证据，生成中文机制报告。"""
import json
import hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
r=json.loads((OUT/'results.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
b2=json.loads((ROOT/'evaluation/strict_compatibility_b2/results.json').read_text())
sections=[]
for spec in protocol['cases']:
    lines=[f"### {spec['id']}（ANALYTICAL EXAMPLE）",'',f"共同前缀长度={spec['shared']}；目标T={spec['targets']}；候选（所属分支,L）={spec['checkpoints']}。",'', '| K | 策略 | 选择顺序/集合 | J（modeled ms） | F（modeled ms） | 最优差距 |','|---|---|---|---:|---:|---:|']
    for case in [x for x in r['cases'] if x['id']==spec['id']]:
        for n,p in case['policies'].items():lines.append(f"| {case['K']} | {n} | {','.join(p['selected'])} | {p['cost_ms']:.9f} | {p['F_ms']:.9f} | {p['gap_to_opt_ms']:.9f} |")
        lines += ['',f"K={case['K']} 的FlowState逐轮计算："]
        for step in case['flowstate_steps']:lines.append(f"- 当前集合={step['selected']}，边际收益={step['gains_ms']}；选择={step['chosen']}。")
        lines += [f"- 全部最优集合={case['opt_ties']}；穷举可行集合数={case['checked_subsets']}。"]
    sections.append('\n'.join(lines))
text=r'''# Phase B3：分配机制区分度分析

## 结论

1. **FlowState与StandaloneTopK不同**：至少两个选择名额下，已选集合对高独立收益候选形成更大覆盖重叠，足以反转剩余候选的边际排序；若预算截断发生在该处，可改变最终收益。K=1两者同分规则之外相同。重复链即可出现，不必跨Workflow共享。
2. **FlowState与WFC不同**：WFC的改善请求数优先或推进Token总量排序，与恢复成本加权推进排序发生冲突；不同T、当前E和推进长度均可使每Token收益不同。覆盖数冲突不等于位置感知模型贡献，必须分开验证。
3. **合法前缀树允许这些差异**：本报告四个固定小案例全部从明确整数Token串生成边，兼容支持集合满足树形包含/互斥结构，不使用任意覆盖图。
4. **B2持平**：共享B1的独立收益和覆盖数均遥遥领先；之后深层独立候选的静态收益、动态边际、Token推进排序一致，预算1–4未触及排序差异。此为实例结构，非算法一般等价。
5. **值得采集的最小结构**：同边界至少两个已知Pending、有效完整候选、可审计Token前缀和资源合同；另外至少存在“重叠候选与竞争候选”或“覆盖数/推进长度/目标位置之间的结构冲突”。仅增加共享边或K_min不足以证明机制区分力。

所有数值均为CPU模型化结果。四个案例是解析构造，不是运行过的真实Workflow或Runtime物化状态，不作为论文性能收益。B2为已有真实Runtime支持的准入矩阵，仅一条跨Workflow边有B1-C独立恢复反事实；其他边不自动获得逐边恢复认证。

## 1. 精确定义与判别条件

固定同一快照及显式准入矩阵，u_p(S)=max_{c∈S} b_pc（空集为0），F(S)=Σ_p u_p(S)。成本J=J(空集)−F。所有候选等大小，约束|S|≤K；没有强制填满无收益项。

**FlowState**按 Δ(c|S)=Σ_p max(0,b_pc−u_p(S)) 选择，按生产实现的1e−9 modeled ms容差处理同分并以候选ID破同分；最大收益非正时停止。

**StandaloneTopK**按 a_c=Σ_p b_pc 静态排序，保留a_c>1e−9的前K项，收益相同按候选ID。它最终返回ID排序集合，返回顺序不是决策顺序。

**WFC**定义 d_pc(S)=max(0,L_c−E_p(S))（非准入边为0），q_c=Σ_p 1[d_pc>0]，r_c=Σ_p d_pc；按(q_c,r_c)降序、候选ID升序选择，q=0停止。不读取Phi。

### 1.1 静态独立收益与动态边际

恒等式：

Δ(c|S)=a_c−O_c(S)，其中 O_c(S)=Σ_p min(b_pc,u_p(S))。

因此在共同前缀S上，如果静态策略选x，而存在y满足

`a_x ≥ a_y` 且 `O_x(S)−O_y(S) > a_x−a_y`，

则Δ_y>Δ_x；还须y在所有剩余候选中最大（或者另一个最大者不是x）才能断言下一步决策不同。实际实现需跨越容差，不将浮点近似同分认作机制差异。若y不是静态最终TopK成员，且预算恰在此截断、两种集合的F严格不同，才能得到最终目标差异。不同中间顺序并不充分。

可证明的充分等价条件：所有剩余候选相对S的O相同，则边际与独立排序相同；特别是正收益支持两两不相交时，选择不会压低其他候选收益，整个过程等价（排除阈值附近的数值边界）。每条独立链有多个冗余候选并不满足“两两不相交”，这正是静态TopK可能失败的地方。

### 1.2 WFC与恢复成本边际

在共同前缀S上，若WFC按(q,r,ID)选x，而某y有Δ_y>Δ_x并达到FlowState最大值，二者下一步不同。两类严格冲突为：

- q_x>q_y，但Δ_x<Δ_y：覆盖更多请求，却每个请求推进太少。
- q相同、r_x>r_y，但Δ_x<Δ_y：Token推进总量较大，却成本权重较小。

若对全部正推进候选每Token收益均为同一常数w，则Δ=w r。但这仍不能保证WFC等价，因为WFC先比较q！还需最大r的候选也处在最大q集合内，且同分规则一致。对于只有q=1的竞争，统一w才足以让两者排序相同。冻结二次模型一般没有统一w。

令t=T/1024、e=E/1024、d=推进量/1024；冻结Phi系数a=37.828150、b=0.345974143、c=−0.156201917，则单请求边际严格为：

`Δ_p=d [a+(b+2c)t−2ce−cd]`。

其中b+2c=0.033570309>0。即使q相同，T、已有前沿和推进量都会影响单位Token收益。独立收益为 `a l+(b+2c)t l−c l²`。冻结模型在0≤G≤T≤131072单调，不允许任意填入收益数字。

区分“边际更新”与“位置感知”的实验应各自控制因素：相同T的重复覆盖例能隔离集合更新；q相同且严格Token排序与Phi排序反转的例能说明非统一恢复权重；只比较q优先造成的差异不足以证明位置感知模型必要。

## 2. 前缀树合法性及状态假设

同一执行配置下，每个Checkpoint是一条完整Token路径上的位置L；Pending输入是树中一条路径，T固定。边当且仅当候选Token串是Pending的逐Token前缀且L≤T。两个节点的支持集合只可能包含或互斥；不能使用“分别覆盖AB、BC而不覆盖AC”的任意交叉覆盖来制造结论。

四例实际编码：共同前缀为Token10重复shared次，A后缀为Token20重复，B后缀为Token21重复，完整Pending长度T+1；候选直接截取相应路径或共同前缀。保存的 `analytical_token_trees.json` 含全部整数Token数组。它们是定义合法的模型输入Token序列，但不是自然语言Agent轨迹或模板一致的真实请求证据。

每个候选假定在同一冻结边界之前已完成前缀执行，具有独立完整conv+temporal状态、同模型/位置/配置，无活动引用；FA驻留覆盖各T，无候选外可恢复状态。所有当前输入已经已知，不需要未来工具输出。候选是完整状态的解析占位，不宣称实际GPU张量已存在。持有同一路径不同位置的完整检查点在状态语义上合法；自动缓存是否自然保留它们，仍需另行Runtime证据。预算仅为逻辑保留上限，允许候选数大于K的前提是边界前具有合法暂存空间，未宣称物理配额已执行。

下面每例至多2个Pending、3个候选、K≤2。“小”指针对该机制的紧凑示例，不宣称穷举证明了所有可能实例中的全局最小规模。案例及预算先写入protocol.json再计算；没有随机搜索或根据输出修改参数。

## 3. 全部固定解析实例

'''+ '\n\n'.join(sections)+r'''

### 机制解释与边界

- **S1**：K=1持平；K=2静态TopK把A1和A2同时保留，A1在A2之后为零边际。FlowState/WFC转向B1，降低成本76.549670140ms。2个Pending、3候选是展示“冗余占位挤掉另一请求”的紧凑结构；该收益无需跨Workflow共享，也不依赖不同T。
- **S2**：K=1时WFC按q=2选浅共享R，FlowState按成本选深层A1，差距78.111689310ms。K=2集合持平。它证明覆盖数量优先可能不合理，不证明位置感知项的独特价值：常数每Token成本也能出现该冲突。
- **S3**：两候选q均为1；A1推进4224比B1的4096更长，但B的T=65536使独立恢复收益更高；FlowState选B1，WFC选A1，K=1严格差距3.135617993ms（以results.json未舍入值为准）。K=2持平。没有ID同分伪差异。该差异由当前已知目标位置的成本权重产生；大目标仍有不可消除残差，不把小局部收益包装为整体大幅改善。
- **S4**：K=1共同选择共享R且最优；K=2贪心无法撤销R，保留R+A1，B残留1024-token缺口，成本39.055844655ms；Exact选择A1+B1，成本0。FlowState、WFC、StandaloneTopK均失败。允许“共享优先”先好后坏，不执行交换修复。

以上赢家与失败者均完整报告，不以结果选择例子或省略预算。结果记录保留所有最优集合，S4的非最优贪心集合不能称为Exact匹配。

## 4. B2为何无法区分

B2输入和结果仅从 `evaluation/strict_compatibility_b2/matrix.json` 与 `results.json` 读取，不重评估旧Formal总体。

B1位置2432，覆盖A/B/C/D全部4个Pending；其余7项各只帮助一个Pending。B1独立总收益364.177368680，远大于A2的164.197398332；WFC也以q=4将其置首。因此K=1三者相同。

随后可取的四个最深项A2/C2/D2/B2的：

| 候选 | 独立收益 modeled ms | 已选B1后推进tokens | 已选B1后边际 modeled ms |
|---|---:|---:|---:|
| A2 | 164.197398332 | 1920 | 73.135615338 |
| C2 | 156.808842402 | 1728 | 65.762008685 |
| D2 | 151.890551426 | 1600 | 60.853683895 |
| B2 | 149.433629831 | 1536 | 58.401745393 |

三列严格同序，且深层项正收益支持互斥，选中一个不会改变另外三个深层项的边际。A1/C1/D1位置更浅；静态排名前四已被B1/A2/C2/D2占据，不会在预算1–4挤入重复链。动态更新虽然把A1、C1等置零，但没有影响最终入选边界。

位置T分别4352/3968/4160/4032；冻结Phi在这里没有制造相对Token推进的排序反转。WFC在B1之后所有正推进候选q=1，因此r排序也相同。故三者在四个预算都持平；这既不是“没有共享”，也不是“K总足够”，而是共享候选与剩余竞争项恰好具有一致排序。

K_min_frontier=4；K=4 Exact/PCB/LDR选A2/B2/C2/D2并达到0成本，三种共享优先策略仍保留B1，损失58.401745393ms。该反例进一步说明“真实共享+逻辑预算取舍”并不自动证明贪心全局分配优势。未从这一单快照推断其他快照性质。

## 5. 后续真实采样的预注册草案（仅设计，不执行）

### 5.1 共同准入条件

正式采集前冻结数据源版本、时间窗口、按原始Session/事件顺序遍历的规则、Tokenizer/模型/模板身份和状态合同。Pending必须在边界前已知，至少两个；候选必须来自已发生执行的完整Checkpoint，逐Token身份可核查、同一静止边界驻留、必要组件与引用安全明确。不得因文本相似、项目相同或逻辑父子直接生成边；不得事后插入Checkpoint位置。保留所有被拒边及UNKNOWN原因。

只有原生或受控因果合法证据达到上述条件，才称真实Snapshot。旧Formal矩阵仅作历史线索，不计为严格认证。先用现有CPU元数据审查有无潜在边界，再决定是否有价值进行最小GPU状态验证；CPU解析成功本身不是GPU采集通过。

### 5.2 三个纯结构分层

按以下顺序给每个边界指定首个符合的分层，不计算任何策略胜负作为准入条件：

1. **重复链与外部竞争**：存在同支祖先/后代两个候选，另有帮助其他Pending的候选；至少两个Pending且至少3个候选。用于检验静态重复覆盖。
2. **共享浅层与专属深层竞争**：存在至少覆盖两个Pending的共同祖先，及至少一个分支更深的独占候选；最深前沿覆盖需不止一个保留项。用于检验q优先与深度权衡，也保留贪心锁定失败。
3. **不同目标/前沿的并列竞争**：存在支持互斥、正推进的至少两个候选，Pending目标T不同，候选位置L或当前基准E不同。用于检验恢复权重是否实际改变决策，不要求先发现Phi排序反转。

这些是必要候选结构，不是成功保证；分层仍可能全部持平。不得按收益比、FlowState胜率或Exact差距选入，不调整位置、消息长度或现成候选制造冲突。记录每层实际T/L/覆盖与预算分布，以及自然出现频率。

### 5.3 固定规模、预算和停止规则

建议在另行批准和锁定来源后，最多顺序检查30个根Session或100个自然Allocation边界（任一上限先到即停）。按事件顺序每层接收最先4个合格边界，合计上限12；每个根Session最多纳入1个，以免将同根多个轮次视为独立样本。未满额就报告缺口，不延长窗口找胜例，不重新筛历史样本。层归属和排除记录均公开。

CPU逻辑比较预先固定对每个入选快照运行K=1…min(4,候选数)，以及K_min_frontier作为单列结构参考；K_min来自覆盖结构，不根据策略胜负定预算。四个预算是同一Snapshot的重复条件，不是四个独立样本。若计划GPU保留压力验证，必须在采集前单独冻结物理池/工作/COW/保留配额合同，不能把上述CPU逻辑K冒充真实显存竞争，不能事后降低K。

到达上限即停止，不以统计显著或FlowState胜利作为停止条件。Token身份、因果边界、组件完整性或安全驱逐发生冲突时，停止该实例并保留原因；若同一关键接口缺失阻断全体，停止采集报告BLOCKED。所有入选实例的胜、平、负、Greedy最优差距均报告；不丢弃对FlowState不利的结果。

### 5.4 投入GPU的最低门槛

至少一个真实可追溯边界具备共同准入条件和上述任一结构，且原Runtime能够在安全边界保存并读取所需状态。验证静态冗余通常需要2个Pending/3个候选/K=2；验证覆盖数冲突或目标位置权重可以只需2个Pending/2个候选/K=1。若目标是“多Workflow共享下的集合更新”，应优先要求同时具有共享祖先、至少两条更深分支和预算约束，而不是只用独立重复链作替代证据。

应先确认真实状态来源与资源合同，再投入GPU核验该边界；不要求CPU预评估显示FlowState获胜。GPU只解决状态可执行性和之后另行授权的运行时问题，不能补救缺乏结构区分力的采样设计。本轮不启动任何采集或Pilot。

## 6. 实现、核对与证据路径

- 冻结参数与原生产选择循环：`flowstate/recovery_model.py`、`flowstate/optimizer.py`，均未修改。
- WFC：直接调用冻结 `evaluation/p1_strong_baseline_validation/wfc.py`；`common.py`共同成本接口。哈希与implementation_lock.json核对。
- StandaloneTopK遵循 `evaluation/rq3_sanity_structure_audit.py:1116–1131` 的静态收益、正收益阈值、ID规则，仅将兼容关系替换为解析Token前缀生成矩阵。
- `protocol.json`：计算前固定的4案例、2预算各例；没有随机搜索、自动调参或迭代选样。
- `analyze.py`：独立MatrixOptimizer只覆盖成本读取，复用生产select；不修改旧谓词，不虚构Workflow身份来绕过旧规则。
- `analytical_token_trees.json`：全部解析Token路径和状态假设；`results.json`：逐候选收益、选择过程、E/G、成本、所有最优集合。
- `validation.json`：8个case-budget检查通过；全子集F/J等价、全部边际与重算一致、预算约束、逐轮最大边际及Exact穷举；B2和源文件SHA-256前后一致。

首次运行时全子集验证误用当前K约束的评分接口而停止；修正独立验证调用的检查容量为候选总数，正式选择仍使用原K。案例、参数、策略均未调整。未将该工具问题当成算法结果。

CPU复现入口：`/home/wjg/uv312/bin/python -B -m evaluation.strict_compatibility_b3.analyze`，随后同路径模块`report`生成报告。完成后停止，无GPU、Engine、Replay、策略生产代码修改或新真实数据采集。
'''
(OUT/'final_report.md').write_text(text)
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('中文报告生成完成')
