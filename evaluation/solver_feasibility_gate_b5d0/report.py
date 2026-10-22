"""汇总固定问题的实测CPU成本与有条件精确求解路线，不实现DP。"""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
rows=json.loads((P/'timing_summary.json').read_text());structure=json.loads((P/'structure_audit.json').read_text());env=json.loads((P/'environment.json').read_text())
def write(n,s):(P/n).write_text(s.strip()+'\n')
table='|快照|n|m|K|搜索子集|准备中位ms|矩阵中位ms|求解中位ms|求解本地P95 ms|三阶段本地P95 ms|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for r in rows:table+=f"|{r['group']}|{r['n']}|{r['m']}|{r['K']}|{r['subsets']}|{r['prepare_ms_median']:.3f}|{r['benefit_ms_median']:.3f}|{r['solve_ms_median']:.3f}|{r['solve_ms_p95']:.3f}|{r['total_ms_p95']:.3f}|\n"
st='|快照|候选/请求|共享候选/不同共享域|前缀位置判定与严格边差异|Laminar|一槽一候选且槽互异|\n|---|---|---|---:|---|---|\n'
for s in structure:st+=f"|{s['group']}|{s['n']}/{s['m']}|{s['shared_candidates']}/{s['unique_shared_supports']}|{s['extra_filter_differences']}|{s['laminar']}|{s['one_slot_each'] and s['unique_slots']}|\n"
write('solver_problem_contract.md',f'''# 冻结问题合同

## 目标及边界

固定Epoch、候选C、已知Pending P、T、严格边与Phi：E_p(S)=max({{L_c:c∈S且(c,p)准入}}∪{{0}})，J(S)=ΣPhi(T_p−E_p(S),T_p)，F(S)=J(∅)−J(S)=Σ_p max_{{c∈S}} b_pc。b_pc仅准入边可为正，REJECTED为0；UNKNOWN必须先拒绝形成完整实验输入，不能通过换求解器消除证据缺口。优化约束|S|≤K，K=1–4。

模型沿用position_aware_quadratic_v1，a=37.828150、b=0.345974143、c=−0.156201917，G/T按1024归一化。目标是模型化恢复成本，不是CPU求解时间、GPU TTFT或物理显存效率。所有需求来自冻结的Controlled Pending；没有读取后续请求结果作为新增在线信息。

## 五个固定输入

{st}

B4-C的48+64+80+32条和B5-C3的28条共252配对逐项检查。没有新矩阵、候选删减或预算调整。严格矩阵来自Runtime证据，不意味着逐边RESUME_VALIDATED。B4-C四个快照来自受控回放；B5-C3是Trace-derived Controlled Composition。20预算条件只属于5个固定快照，不是20个独立工作负载。

B4-C等大小合同由冻结load及B5-C0逐项核查支持；B5-C4通过实际Conv/Temporal切片核验七候选各51511296字节、一个独占槽。逻辑K并不代表物理池耗尽。Alias或其他联合约束若存在，不能继续使用这个纯基数可行域。

## 单共享与多共享

B5-C0的四组只有一个非单例正支持候选；固定是否选它后，两分支Top-K具有已证明的精确性（exact_decomposition_proof.md）。B5-C3则有四个共享候选、三个不同共享域，不能套用单共享公式。N27支配N21但全部七候选仍在本轮搜索。

Greedy给定一次执行只能不断增加已有选集。B5-C4最优集合K1={{N20}}、K2={{N18,N27}}、K3={{N22,N24,N27}}、K4={{N22,N24,N25,N27}}，并不嵌套。首步最优不能保证是后续预算最优解的子集；原Greedy没有回退机制。这里只是冻结反例，不声称所有共享结构均使Greedy失败。

证据：B4-C/results.json及evaluate.py；B5-C0/verification.json、exact_decomposition_proof.md；B5-C4/results.json、exact_verification.json、validation.json。完整路径和SHA见integrity.json与report_sources.json。
''')
write('exact_solver_feasibility.md',f'''# 直接Exact独立性、完整性和有限计时

## 审计结论

B4-C/evaluate.py的subsets/evaluate与B5-C4/evaluate.py均遍历大小≤K的全部候选子集，用共同J评价并按1e-9容差、较小集合、字典序确定代表。候选不经过Greedy筛选，Exact并不依赖Greedy输出。B4-C按最大K=4预计算并缓存各子集分数，随后各K查询；B5-C4检查全部128个子集。它们不是同一计时边界。

B5-C0通过独立两分支数学化简和Decimal收益验证核对16条件；B5-C4通过独立Decimal公式核对全部128子集、24策略结果、448边际变化，构成数值独立验证。浮点1e-9比较属于冻结数值合同，不等于任意真实数位精确性。

本轮measure.py的exact函数复制既有枚举/评分/同分定义，没有剪枝、增量优化或新策略。每次仅求当前K，全部生成子集、common.score评分、取最小与确定同分代表均计入solve。不使用跨预算缓存。20条件、60次预热、600次计时的所有结果都与冻结Exact目标及代表集合一致；同次运行一次性检查所有可行子集J=J0−F。

## 计时合同

环境：Intel Xeon Gold 6426Y，Python3.12.13，/home/wjg/uv312/bin/python，单进程固定CPU0，GC保持启用。未改变其他任务，无频率与后台负载隔离。环境完整记录见environment.json。

每个条件3次预热、30次正式重复；每轮固定循环轮转条件顺序。perf_counter_ns三个连续区间：

1. prepare：内存中的完整Snapshot JSON反序列化、ID/配对域/状态检查及不可变Epoch对象构建。
2. benefit：使用冻结Phi构建全部n×m独立收益矩阵。
3. solve：当前K的所有子集生成、原common.score前沿与Phi成本评分、选出最优及同分代表。

历史求解器直接调用J评分，不消费第2步矩阵，因此第2步是单独记账的准备成本；没有用矩阵缓存降低求解时间。磁盘读入、SHA、一次性Token逐边结构审计、结果正确性核对、写盘在计时之外。prepare并非完整生产StrictEpoch认证，三个区间之和也不是在线控制总成本。

全部重复值见raw_timings.json；均值/中位/最小/最大/本地P95见timing_summary.csv。P95是30次重复升序第29项，不是线上P95，也不是置信区间；同一条件重复不增加工作负载样本量。不同阶段P95不能相加代替total的P95。

{table}

## 可用范围

搜索规模S(n,K)=Σ_{{i=0}}^{{min(n,K)}} C(n,i)。直接J评分时间上界O(mK·S)，本测量保存全部评分及逐请求信息，空间约O((m+K)·S)，不是内存优化版本。K接近n时为指数增长；不能按n20/K4外推n20/K20或更大请求规模。

当前B5-C3每预算求解中位0.041–0.399ms、K4本地P95约0.472ms，确有进入后续小规模Runtime控制验证的机会。g001/K4中位3.315ms；g002/K4为10.925ms；g003/K4为31.887ms、本地P95约37.313ms，是否可接受取决于尚未提供的在线控制期限。

输入准备中位约2.8–38ms不可忽略；最大三阶段本地P95约77.010ms。不能用单独solve数字声称整个控制链亚毫秒，也不能从本轮推断Exact比Greedy更快：本轮没有相同计时合同下的Greedy速度比较。此前T2包含缓存建立和查询，应保留其原合同，不直接与这里单预算无缓存计时做速度相减。
''')
write('tree_dp_applicability.md',f'''# 前缀树DP适用性审计

## 当前证据等级

{st}

- **已证明适用（限定数学输入）**：这五个冻结矩阵中，准入边恰等于完整Token前缀且L≤T；无额外配对删边；收益随同路径深度非减。固定纯基数目标满足下面的精确DP前提。
- **需要额外假设（一般Runtime）**：新Epoch也必须满足相同配置、完整驻留、稳定边界、位置坐标一致、无未建模基础状态、没有跨候选物理可行性耦合。不能从五个实例推出整个Strict API自动保证这些性质。
- **当前证据不足**：DP实现正确性、构树/验证/求解耗时、在线超时和隔离行为均未验证；没有DP生产READY结论。
- **明确不适用（以下简化递推）**：配对特定过滤使某祖先对部分后代失效、收益不满足深度支配、联合资源约束或非等大小成本且仍使用此纯基数递推。Laminar支持本身不足以保证下面的一标量祖先状态能处理任意收益。

本轮252条配对中额外过滤差异为0；是原始Token/位置与冻结准入逐项比较的观察，不是删除不一致边之后的结果。树要把Pending挂在T处而非完整输入末端，否则会把L>T错误计入。候选是否完整、引用安全等全局资格删除一般保持树结构；独立删候选–请求边不一定保持。

## 状态、递推与证明（条件性结论，不实现）

压缩Token trie保留根、分叉、候选位置和每个Pending的T终端。N为压缩节点数，n候选、m需求。多相同Token位置的候选若物理语义等价，可保留确定性代表处理；不能在未经等价检查时合并。本批没有同位置多身份问题。

设虚拟祖先⊥深度0且不计费。D(v,a,k)表示v子树选恰k个候选，子树外最深已选有效候选为a时的最小恢复成本。P_v是恰在v终止的Pending，Q_v(a)=Σ_{{p∈P_v}}Phi(T_p−L_a,T_p)。

不选v：Q_v(a)+min_{{Σk_i=k}}Σ_iD(u_i,a,k_i)。

选v（仅v为候选且k≥1）：Q_v(v)+min_{{Σk_i=k−1}}Σ_iD(u_i,v,k_i)。

取两者最小，叶子空卷积只有0预算可行；根结果为min_{{0≤k≤K}}D(root,⊥,k)。内部节点的Pending必须当场结算，不可由更深后代改善。

正确性由子树高度归纳：任何集合唯一分解为选/不选v及各孩子预算；给定最近祖先后，不同孩子需求互斥且成本可加。最深保留祖先支配更浅祖先，外部不需要保存其他状态；每种合法分解被卷积覆盖，最优子解的组合亦合法，所以结果是Exact。

Phi依赖T不破坏最优子结构：T是每个需求的固定参数，只进入Q；关键是固定T时随G非减，故更深有效祖先不更差。冻结Phi在0≤G≤T≤131072上对G的导数为a+bT/1024+2cG/1024，最小至少a+(b+2c)T/1024>0。因此本模型满足严格深度单调性，不需要不同Pending共享相同T。

每节点至多n+1祖先状态，K+1预算；逐子节点卷积的保守时间O(N(n+1)K²+m(n+1))、空间O(N(n+1)(K+1))，另加读取Token、构树和兼容证据验证。N=O(n+m+1)在压缩树及去重等价位置后成立；原始Token总长度并非此复杂度免费包含。该上界不是实测性能，也不声称理论创新。

## 同支持和物理条件

N21/N27均支持CD，但N27更深、同成本且逐坐标支配N21。仅在固定目标/无其他副作用约束下，可以在未来求解器预处理中排除被支配项，不改变最优值；若需要复现所有同分最优集合或精确ID选择，仍须保留身份或明确同分合同。本轮没有删除N21。

五快照全部一候选一槽、槽号互异；B5-C3还具有完整组件切片无别名证据，B4的完整性见既有门禁。树形逻辑问题无耦合不代表未来Runtime可安全逐项驱逐。若存在共享组件、Pin、活跃引用、Slot generation变动或强制联选，须先明确定义可行域，不应继续把本DP当正确求解器。

依据既有T1/final_report.md第4.1–4.4节，当前结构复核保存在structure_audit.json；本轮没有实现DP。
''')
write('solver_route_decision.md','''# 求解器路线决策

## 唯一优先路线

**下一阶段优先做小规模直接Exact的CPU集成与合同验证设计；不优先开发完整树DP，不再默认Greedy是本批的最佳求解路线。** 本轮仅推荐，不执行下一阶段。

已测可信优先域是B5-C3（n7,m4,K1–4），其次g004（n8）和g001（n12），最大搜索794子集。可把n≤12、m≤4、K≤4且S≤794作为下一阶段保守候选准入上限，但其中未实际出现的规模/Token大小仍需验证，不能宣称已测全域。上限来自工作量与测量范围，不来自策略胜负。

n16/n20,K≤4本轮可正确完成Exact，成本约11/32ms的K4求解中位值需要结合控制期限评估；暂不默认加入延迟敏感在线路径。只按n阈值不够，必须同时检查m、K、搜索规模、输入准备与证据认证成本。

## 条件性后续路线

若下一阶段出现比有限枚举域更大的、已证明纯Token祖先兼容且无资源耦合的Epoch，则值得开发并验证树DP。先以全部小规模子集Exact作独立oracle，核对目标、预算、同分合同，再计构树及求解开销。当前只证明递推适用，不批准未实现DP上线。单共享两分支公式可作为结构特化与测试oracle；不是所有多共享数据的通用替代。

Greedy仅作为完整合法Epoch上的受控Fallback：搜索规模超出已验证域、DP未实现或结构前提不满足但普通固定非负收益/纯基数问题仍成立、或Exact达到预先冻结的时间限制。UNKNOWN、物理别名、状态陈旧或非法可行域不是自动切换Legacy Greedy的理由，应拒绝分配或重新观察。不得改变Strict Compatibility来让Fallback运行。

## 在线时间上限

需要预先声明端到端控制期限、Exact可用子预算及剩余Fallback时间；当前没有Runtime SLA，不能根据本次一次本地P95直接承诺截止时间。生产前应有搜索规模上限和单调时钟deadline。超时中断未穷尽不能标Exact；返回Greedy或已完整验证的可行解时，必须标注非最优证据等级。Fallback本身也消耗时间，应预留而非超时后无限追加。

当前measure.py无生产取消/超时机制，只对已冻结小问题有界运行。要接入Runtime仍需纯规划接口、相同Epoch摘要、同分输出合同、异常隔离以及独立安全Reconciliation；CPU结果不能验证这些物理后置条件。Exact不需要未来请求信息，也不能解决模型或兼容性不合法问题。

## 论文影响

将固定目标和求解器分开：目标仍为严格准入下的模型化恢复成本；描述当前Greedy的标准近似性质及已观测失败，并明确受限结构可精确求解。未来若正式采用Exact/DP，应更新算法、复杂度、版本与实验，而不能用本轮求解时间替换旧Greedy结果。

保留PCB/WFC/StandaloneTopK全部负面、胜出与持平记录。B5-C4 Greedy与WFC全平、K4落后PCB、K2/K3各简单规则落后Exact的事实不变。不能把精确优化共同Phi解释为真实GPU更快，也不能将树DP条件性理论描述为已验证系统实现。
''')
write('final_report.md',f'''# Phase B5-D0 最终报告

**Gate：READY（求解器路线审计，不是新求解器上线或Runtime性能门禁）**。

推荐下一阶段优先验证**小规模直接Exact**，树DP作为规模扩大且结构合同通过时的后续路线；Greedy保留为合法输入上的有界Fallback。

## 实测证据

冻结输入：B4-C四快照与B5-C3一个快照，共20个相关预算条件；n=7/8/12/16/20，m均为4，K=1–4。600次有限CPU测量及60次预热全部匹配冻结Exact目标和代表选集。输入/历史结果未修改。

{table}

B5-C3直接Exact求解中位0.041–0.399ms，K4本地P95为0.472ms；g003/K4中位31.887ms、本地P95为37.313ms。两者都不能代表线上P95。准备输入的成本单独报告，最大三阶段本地P95约77.010ms。未测GPU或Greedy速度，不声称系统性能优势。

## 结构与结论边界

252条关系逐项比较，五快照严格准入均与Token前缀且L≤T一致；支持族均Laminar，一候选一槽且槽互异。当前数学问题满足带最近祖先状态的精确树DP条件；DP尚未实现，实际耗时与一般Runtime适用性未验证。

小规模优先域为已测n7/8/12、m4、K≤4，最多794子集；推广为其他同规模输入仍需完整合同验证。n16/20需明确在线控制期限后决定。时间上限应预注册，不把未穷尽结果称Exact；语义不合法不能用Greedy降级掩盖。

当前工作足以确定路线，但不足以直接替换生产优化器。阶段间没有拼接GPU收益、没有删除PCB/WFC负面或持平结果、没有改动旧选集。完成后停止，等待审查。

## 产物

solver_problem_contract.md：目标与输入合同；exact_solver_feasibility.md：枚举独立性、计时边界和完整分布；tree_dp_applicability.md：条件性递推与证明；solver_route_decision.md：优先路线及Fallback边界。measurement_protocol.json、environment.json、raw_timings.json、timing_summary.csv、structure_audit.json和integrity.json保存原始证据。measure.py是独立CPU入口，已存在原始计时时拒绝覆盖。
''')
paths=[ROOT/'evaluation/allocation_structural_properties/final_report.md',ROOT/'evaluation/allocation_structural_properties/t2/audit.py',ROOT/'evaluation/structural_exactness_gate_b5c0/exact_decomposition_proof.md']
(P/'report_sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},ensure_ascii=False,indent=2)+'\n')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}
(P/'evidence_manifest.json').write_text(json.dumps({'files':manifest},ensure_ascii=False,indent=2)+'\n')
print('五份审计报告已生成。')
