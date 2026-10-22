"""只读取本轮审计和计时结果，生成中文报告，不调用求解器。"""
import csv
import json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'structure_audit.json').read_text())
rows=list(csv.DictReader((HERE/'solver_timing.csv').open()))
assert len(rows)==16 and len(s['timing_samples'])==480
assert all(len([r for r in s['timing_samples'] if r['group']==g and r['K']==k])==30 for g in range(1,5) for k in range(1,5))
lines=['# Phase T2：结构适用性与求解开销审计','', '**Final Gate：PARTIAL。** g001–g004 的固定逻辑分配问题满足 T1 树形动态规划合同；g000 的完整合同仍为 UNDETERMINED。四个正式快照全部 16 个预算条件的既有 Greedy / Exact 求解计时和冻结结果一致性核对完成。没有实现 Tree DP，没有更换默认算法，没有运行 GPU 或重新采集。','',
'## 1. 输入身份与本轮范围','',
'读取 T1、B4-B1 最终执行锁与根 Manifest、B4-C 输入摘要及产物摘要。608 条摘要引用核验通过，244 项执行锁中包括完整源数据及模型文件；相同文件在多个 Manifest 中出现不作为不同证据样本。执行前后逐文件 SHA-256 相同，B4-B0 最终锁与 B4-B1 保存的锁一致。详情见 structure_audit.json/integrity。没有调用 Docker 检查当前镜像或启动 Runtime；这里核对的是冻结执行身份及磁盘内容。',
'g000 仅读取 B1 positive_arm 的原始 Token、状态与配对记录，并记录该文件当前 SHA；不使用 evicted_arm 混合状态。g001–g004 用于结构与计时。根 Manifest 的完整性核对涉及 g005/g006 文件，但不分析其结构、不计时、不补位。没有读取后续状态回填原 Snapshot。','',
'## 2. 逐 Snapshot 结构结论','',
'VALIDATED 均限定于当前静止边界、固定候选、同一 TP=1 配置及纯逻辑基数目标，不表示生产环境的全部恢复关系或物理资源合同已认证。','',
'|组|候选/配对|准入/拒绝|Token 必要性/充分性|物理组件合同|完整树合同|', '|---|---:|---:|---|---|---|']
for x in s['snapshots']:
 counts=Counter(e['status'] for e in x['pairs'])
 lines.append(f"|{x['group']}|{x['candidates']}/{len(x['pairs'])}|{counts['EPOCH_ADMISSIBLE']}/{counts['REJECTED']}|VALIDATED / VALIDATED|{x['checks']['physical']}|{x['checks']['tree_contract']}|")
lines += ['', '### 2.1 必要性、充分性与拒绝原因','',
'逐条使用冻结的完整整数 Token 数组，而非 Digest 推断前缀。先检查数组摘要和位置长度，再比较候选序列与 Pending 截断到 T 后的前缀。逐边验证准入⇒前缀祖先，以及前缀祖先且候选合格⇒准入。g001–g004 所有候选的配置身份、物理状态与安全条件均成立，所以在这四张矩阵上还可验证不附加过滤的双向等价。',
'重新读取配置对象并核对其摘要；使用 B4-B0 原有 build_edges 只读函数核对完整条件向量，与冻结向量逐项相同。检查真实 Node Token 数组、位置、slot 映射、conv/temporal 存储形状和索引范围、引用与别名。静止边界前后树和池记账相同。源码：evaluation/strict_compatibility_b4b0/contracts.py:36–81，capture.py:50–106。全配对及失败条件保存于 structure_audit.json/snapshots/*/pairs。','',
'|组|仅 Token 不匹配|Token 不匹配且 L>T|配置/组件/其他拒绝|前缀成立却未准入|','|---|---:|---:|---:|---:|']
for x in s['snapshots']:
 c=Counter(tuple(e['failed_conditions']) for e in x['pairs'] if e['status']=='REJECTED')
 only=c[('exact_token_prefix',)];both=c[('L_le_T','exact_token_prefix')]
 lines.append(f"|{x['group']}|{only}|{both}|{sum(c.values())-only-both}|{sum(not e['sufficient'] for e in x['pairs'])}|")
lines += ['', '拒绝类别互斥，L>T 同时 Token 不匹配的配对不重复计数。全部五组没有前缀符合但被额外配对条件排除的记录，也没有准入却不符合前缀的记录。UNKNOWN 配对数为零不代表所有更强物理合同字段齐全。','',
'**g000 的限制：** B1 run_pilot.py:242–270 中 physical_ok 仅检查 FA、Recurrent 驻留布尔值与非空 slot；并未像 B4 一样逐组件核对形状及槽位范围。其配置为一次加载的公共身份，缺逐对象配置摘要，旧记录也未明确保存等大小 memory_bytes。已有记录能支持 token 结构及当时的准入逻辑，却不足以把完整物理条件标为 VALIDATED。不能把 B1-C 一条跨 Workflow 恢复验证推广到全部候选，亦不能通过 B4 的新记录替旧 g000 补证。','',
'### 2.2 Laminar 与深度支配独立核查','',
'支持集合按冻结 Phi 与准入关系计算 b>0。表中 EQUAL 为相同支持（不是相同 Token 前缀），NESTED 为严格非空相交包含，DISJOINT 为不相交；分类优先相等再不交再包含。所有候选支持集、每个候选对类型、每个同请求祖先/后代收益对均在 JSON 保存。','',
'|组|共享候选|EQUAL|NESTED|DISJOINT|Laminar 违例|深度支配检查/违例|相同前缀与位置重复项|','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in s['snapshots']:
 c=Counter(t['kind'] for t in x['pair_types'])
 lines.append(f"|{x['group']}|{x['shared_candidates']}|{c['EQUAL']}|{c['NESTED']}|{c['DISJOINT']}|{len(x['laminar_violations'])}|{len(x['depth_checks'])}/{sum(not d['valid'] for d in x['depth_checks'])}|{len(x['duplicates'])}|")
lines += ['',
'五组均没有 Laminar 反例、深度收益下降或相同前缀/位置的逻辑重复项。共享候选均为各组 B_TURN_001，其余正收益支持为单请求。Token trie 存在、支持族 laminar、深度支配和完整合同是分别检查的四项；本报告不是仅凭 laminar 就推定 DP 可用。',
'对 g001–g004，Token/位置的双向等价、配置身份、完整驻留与等大小候选均有记录；B4-C 的约束仅为固定 K，没有额外联合选择约束。因此可以无损映射为 T1 的终端截断 trie 和最近已选祖先状态，**该逻辑问题适用 T1 精确 DP 定理**。没有实际构建或运行 DP 求解器。',
'尚未认证的内容包括 slot generation、逐边实际恢复及所有选集的物理执行。本结论不声称 K 为 HBM 上限，也不声称每个模型最优集合已经安全落地到 Runtime。树合同对模型问题成立与物理控制闭环完备是不同结论。','',
'## 3. 计时实现与公平性边界','',
'只使用 evaluation/strict_compatibility_b4c/evaluate.py 的 MatrixOptimizer（继承生产 GlobalOptimizer.select）和既有 Exact 穷举语句。通过 AST 原样抽取 evaluate 中 all_sets、scores 与 available、optimum、ties、exact 六个赋值语句，没有重写枚举器、tie-breaking 或选择顺序。Greedy 的排序、容差、停止规则及结果包装保持生产路径。','',
'每组先将输入、对象和冻结收益矩阵载入内存，验证矩阵一致；IO、JSON 解析、Snapshot 构造和矩阵校验不计时。两种现有求解路径实际都调用同一 common.score，按前沿重新计算 Phi，并分配 per_pending 字典。它们并未消费共同的预计算 b 矩阵。为避免改变既有缓存和实现，本轮保留此差异，不将时间解释为抽象算法操作数或优化矩阵内核速度。common.py:59–71 给出共同计分路径。','',
'**Exact 的三个时间口径必须区分：**','',
'- prepare：按冻结代码一次构造全部 |S|≤4 子集并计算其成本缓存，纳入求解工作，不当作输入加载排除。',
'- query：在本次缓存中筛出 |S|≤K，寻找最小成本、完整同分集合和按大小/ID 的代表；只有它不等于完整精确求解。',
'- cold：同次 prepare+query，表示某 K 在无已有子集成本缓存时调用现有四预算实现的成本；K<4 仍包含既有实现的 K4 缓存准备，不能称为该 K 的最小可实现枚举成本。',
'- amortized：prepare/4+query，仅是同组四预算共用缓存时的记账均摊，不是一次实际调用延迟。四个均摊量之和才等于该批次完整工作。','',
'每组执行 3 次预热和 30 次重复批次；每次重建 Exact 缓存，并对每个 K 独立调用 Greedy / Exact 查询。缓存仅在同一四预算批次内复用，保持 B4-C 行为。两方法先后顺序按重复编号交替。计时后核对完整返回序列、J 与冻结结果，每次均一致。每条件 30 次计时，480 条预算记录；它们不是新增 Snapshot 或独立工作负载。未将正确性检查、写盘放入求解时间。',
'使用 perf_counter_ns；P95 为最近秩法第 ceil(0.95×30) 项。普通 Python GC 保持启用，不清除操作系统页缓存，不做独立进程冷启动；“cold”仅指没有上一批次子集成本缓存。Python exec 包装、对象分配、字典构建、线程调度与 GC 影响包含在实测中。没有根据有利结果剔除计时样本。','',
'CPU：Intel Xeon Gold 6426Y；解释器 /home/wjg/uv312/bin/python，版本 '+s['environment']['python'].split('\n')[0]+'。进程固定到允许的 CPU '+str(s['environment']['benchmark_affinity'])+'；OMP/OPENBLAS/MKL/NUMEXPR 线程设置均为 1。无并行求解，未保证 SMT 兄弟线程或其他用户负载隔离，未固定 CPU 频率。完整环境及 480 条原始计时见 structure_audit.json。','',
'## 4. 全部求解时间与质量','',
'下表时间单位为 ms，均为中位数；J 和损失单位为 modeled ms，与 CPU 时间是不同指标。G/E 比为 Greedy/cold Exact 的中位数比。完整 Mean/P50/P95、每条件枚举范围与重复数见 solver_timing.csv。','',
'|组|K|候选|Greedy|Exact cold|Exact 均摊|G/E|Greedy J|Exact J|损失|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 f=lambda k:float(r[k])
 lines.append(f"|{r['group']}|{r['K']}|{r['candidates']}|{f('greedy_median_ms'):.6f}|{f('cold_median_ms'):.6f}|{f('amortized_median_ms'):.6f}|{f('greedy_over_exact_cold_median'):.4f}|{f('greedy_J_ms'):.6f}|{f('exact_J_ms'):.6f}|{f('optimality_gap_ms'):.6f}|")
lines += ['', '### 4.1 Mean / P50 / P95','', '|组/K|Greedy Mean/P50/P95|Exact cold Mean/P50/P95|Exact query Mean/P50/P95|','|---|---|---|---|']
for r in rows:
 def triple(prefix):return ' / '.join(f"{float(r[prefix+'_'+v+'_ms']):.6f}" for v in ['mean','median','p95'])
 lines.append(f"|{r['group']}/{r['K']}|{triple('greedy')}|{triple('cold')}|{triple('query')}|")
lines += ['', '### 4.2 缓存准备与枚举范围','', '|组|一次准备的子集数|K1/K2/K3/K4 实际可行子集数|准备 Mean/P50/P95 ms|','|---|---:|---|---|']
for g in range(1,5):
 rr=[r for r in rows if r['group']==f'g{g:03d}'];r=rr[0]
 lines.append(f"|{r['group']}|{r['cached_subsets']}|{' / '.join(v['eligible_subsets'] for v in rr)}|{' / '.join(f'{float(r["prepare_"+v+"_ms"]):.6f}' for v in ['mean','median','p95'])}|")
lines += ['',
'所有可行子集包含空集，范围与原 B4-C 相同。以上 K1–3 的 cold 时间受 K4 预计算支配，不能据此推断单 K 枚举的渐近复杂度。也不能用 query 与 Greedy 直接比较，因其前者已经支付过全部子集成本。均摊结果提供符合原四预算批次的另一种口径。','',
'## 5. 质量—成本权衡与科研判断','',
'**已测量：** Greedy 在 12/16 条预算条件匹配 Exact；四个失败条件为 g001/K4（68.218394974）、g002/K3（49.016811025）、g002/K4（82.987847211）、g004/K4（48.608817340 modeled ms）。与 B4-C 完全一致。没有将不同集合但同目标计为质量差异。',
'四组候选规模仅 8–20，最大枚举缓存为 6196 项。当前机器上完整既有 Exact 的各条件中位数约 0.595–31.734 ms，Greedy 约 0.040–0.344 ms。四预算均摊 Exact 约 0.155–8.887 ms/条件。可见实际实现的求解时间—质量权衡：Greedy 更快，但其中四条有非零最优性损失；g003 各预算 Greedy 已精确，Exact 没有新增模型收益。',
'这些规模下 Exact 对 CPU 离线审计是可承受的实测计算量；“生产可接受”仍无可判定的控制周期或延迟预算。不能把约 32 ms 中位数视作任何服务 SLA 都可接受，也不能把中位数当尾延迟保证。本轮不能证明 Exact 可扩展至任意候选规模，更不能证明未实现的 Tree DP 比 Greedy 快或慢。',
'Greedy 默认选择目前有三类依据：经典近似保证、这四张矩阵中较低的实测 Python 求解开销，以及 12/16 的有限实例最优匹配。仍缺在线候选规模分布、决策周期、并发争用、Runtime 总控制成本、质量损失到 GPU 延迟的映射，以及利用精确树结构的工程代价。故**不足以证明 Greedy 在生产环境普遍优于精确求解器**，也不足以支持立即替换默认算法。','',
'## 6. T1 性质与 Contribution 2 的限定','',
'|类别|可进入论文的内容|不得扩大为|','|---|---|---|',
'|标准优化理论应用|固定 max-benefit 的单调次模性及经典 Greedy 保证|新的次模理论贡献、请求延迟近似保证|',
'|问题结构推论|可分支持精确化简；K_min 饱和定理；纯前缀合同下的精确树 DP|所有 Runtime 都满足纯树合同、Greedy 必然精确|',
'|严格实例观察|g001–g004 双向前缀等价、laminar、深度支配、固定逻辑树合同；实测求解开销与四个失败|一般负载分布或系统净收益|',
'|尚未证明的系统主张|生产可扩展性、控制开销后净收益、逐边恢复和选集安全执行|利用逻辑结构检查替代 Runtime 认证|',
'',
'Contribution 2 可限定为：在固定、经过准入核查的可执行状态视图上，以共同恢复成本定义集合相关分配，并使用具经典保证的 Greedy；同时披露可精确化简的子类及贪心失败边界。当前四个严格实例的纯前缀树条件成立，因此不能把这些实例描述成没有可利用结构、只能近似求解的一般覆盖问题。可以报告 Greedy 的实测开销优势，但必须指明现有 Python 实现、缓存口径及小规模范围。位置感知项的独立分配收益仍未由本轮计时或结构检查证明。',
'不直接修改正式论文，不替换生产默认算法；本轮不提出或执行新的求解器实验。','',
'## 7. 证据索引与复现入口','',
'- 结构输入：B4-B1 根目录 g001–g004/snapshot.json；g000 使用 evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json 的 positive_arm。逐边 JSON 保留实际路径和 SHA。',
'- 物理准入：evaluation/strict_compatibility_b4b0/contracts.py:50–81；原配置与同边界来源见 capture.py:50–106。g000 较弱检查见 strict_compatibility_b1/run_pilot.py:242–270。',
'- Greedy：flowstate/optimizer.py:34–135；B4-C 矩阵适配 evaluate.py:26–29；共同评分 p1_strong_baseline_validation/common.py:59–71。',
'- Exact：strict_compatibility_b4c/evaluate.py:24,62–71 的缓存与选择语句，使用 AST 原样抽取；收益矩阵与冻结策略结果来自同目录 results.json。',
'- 本轮 audit.py 为独立 CPU 入口；report.py 只读本轮结果生成报告。原始正式代码和产物保持不变。',
'',
'复现命令（输出采用独占创建，当前已有产物时不会覆盖；如需再次执行，必须另行审查输出位置）：',
'```bash',
'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/wjg/uv312/bin/python -B evaluation/allocation_structural_properties/t2/audit.py',
'```','',
'## 8. 最终判定','',
'**PARTIAL。** g001–g004 的双向 Token 兼容、完整边界准入、laminar、深度支配与纯基数树形模型合同已验证；g000 完整物理组件和逐对象身份/等大小证据不足，保留 UNDETERMINED。四个正式快照的既有求解器计时与质量核对完成，输入摘要未变化。',
'Greedy 有已测量的小规模实现开销优势和经典模型收益保证，但作为生产默认求解器的普遍优越性证据不充分。Contribution 2 应明确结构可解边界与近似权衡。不能成立的主张包括：全部 Runtime 纯树形、Greedy 始终最优、Exact 任意规模可扩展、EPOCH_ADMISSIBLE 等同逐边恢复认证、模型质量等同 GPU TTFT 或端到端净收益。任务完成后停止。']
(HERE/'final_report.md').write_text('\n'.join(lines)+'\n')
