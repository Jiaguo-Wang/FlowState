"""根据已完成的 CPU-only 验证生成完整报告，不执行或修改任何策略。"""
from collections import Counter
from common import *

rows=jsonl(OUT/'per_case_results.jsonl');summary=read(OUT/'summary.json');validation=read(OUT/'validation.json');gate=read(OUT/'pcb_gate.json')
assert validation['status']=='PASS' and gate['status']=='PASS'
assert all(abs(c['baseline_minus_flowstate_ms'])<=EPS for r in rows for c in r['comparisons'].values())
assert all(c['same_set'] and c['same_EG'] for r in rows for c in r['comparisons'].values())
status='WORKLOAD_NOT_DISCRIMINATIVE'

def fm(x):return f'{x:.6f}'
def find(s,k):return next(r for r in rows if r['snapshot_id']==s and r['k']==k)

chosen=[
('无共享但预算真正受限的典型 tie',find('rq3-openhands-main-g000-round2',2)),
('位置感知收益改变顺序但未改变最终集合',find('rq3-openhands-main-g051-round5',5)),
('共享总收益与逐请求推荐改变顺序',find('rq3-agentx-g000-epoch109',27)),
('WFC 覆盖数量优先选取很浅共享项',find('rq3-agentx-g005-epoch8',2)),
('原 StandaloneTopK 严格胜例之一',find('rq3-agentx-g006-epoch35',8)),
('原 StandaloneTopK 严格胜例之二',find('rq3-agentx-g018-epoch30',7)),
('无任何正收益候选，四种方法都提前停止',find('rq3-agentx-g012-epoch11',2)),
]
case_lines=['# 真实案例与顺序分歧','', '七例按机制覆盖选取，全部属于 objective ties；没有挑选不存在的正例。以下所有值均为 modeled ms。完整候选、兼容矩阵、收益及全部逐步 q/r 见 per_case_results.jsonl。','']
case_index=[]
for i,(reason,r) in enumerate(chosen,1):
    idx=next(j for j,x in enumerate(rows,1) if x['snapshot_id']==r['snapshot_id'] and x['k']==r['k'])
    case_index.append(dict(case=i,reason=reason,snapshot_id=r['snapshot_id'],k=r['k'],result_line=idx,source_file=r['source_file'],source_line=r['source_line']))
    case_lines += [f'## 案例 {i}：{r["snapshot_id"]}，K={r["k"]}', '',reason+'。', '',f'来源：`{r["source_file"]}:{r["source_line"]}`；新结果 `per_case_results.jsonl:{idx}`。', '',f'预算结构：{r["budget_structure"]}', '', '| 方法 | 选择顺序（原始 ID） | C(S), modeled ms |','|---|---|---:|']
    for name in ('FlowState','PCB','LDR','WFC','StandaloneTopK','LRU','LFU','Marconi'):
        v=r['policies'][name];case_lines.append(f'| {name} | '+ ' → '.join('`'+c+'`' for c in v['selected'])+f' | {fm(v["cost_ms"])} |')
    case_lines += ['', 'FlowState、PCB、LDR、WFC 的下列逐 pending 数值完全相同，合并展示：','', '| pending 原始 ID | T | E | G | C_p, modeled ms |','|---|---:|---:|---:|---:|']
    for p in r['policies']['FlowState']['per_pending']:case_lines.append(f'| `{p["continuation_id"]}` | {p["T"]} | {p["E"]} | {p["G"]} | {fm(p["cost_ms"])} |')
    for name in ('PCB','LDR','WFC'):
        d=r['comparisons'][name]['first_divergence']
        if d:
            case_lines += ['',f'{name} 与 FlowState 首次顺序分歧：第 {d["step"]} 步；共同前缀 `{d["common_prefix"]}`。','', '| 方法 | 本步 checkpoint | q | r, tokens | 当前 marginal, modeled ms |','|---|---|---:|---:|---:|']
            for label,field in [('FlowState','FlowState'),(name,'baseline')]:
                v=d[field]
                if v:case_lines.append(f'| {label} | `{v["checkpoint_id"]}` | {v["q"]} | {v["r"]} | {fm(v["marginal_ms"])} |')
        else:case_lines+=['',f'{name} 与 FlowState 顺序也相同。']
    case_lines += ['', '所有新增 baseline 的逐 pending ΔC 向量均为零，总差值为零。选择中间过程的收益差不等于最终 objective 差。','']
(OUT/'case_studies.md').write_text('\n'.join(case_lines)+'\n');write('case_index.json',case_index)
order_cases=[dict(population=r['population'],snapshot_id=r['snapshot_id'],k=r['k'],baseline=n,first_divergence=c['first_divergence']) for r in rows for n,c in r['comparisons'].items() if c['order_different']]
write('order_difference_cases.json',order_cases)

report=['# P1-B CPU-only 强基线验证','', '**WORKLOAD_NOT_DISCRIMINATIVE**','', '日期：2026-10-09（Asia/Shanghai）。PCB、LDR、WFC 正确性验证通过，但当前全部 573 个冻结 snapshot-budget cases 上，三种方法与 FlowState 的最终 selected set、逐 pending E/G 和 modeled executable-recovery cost 完全相同。三种新 baseline 的严格正例与负例均为 0；不是靠四舍五入或提高容差得到的持平，原始浮点成本差最大值就是 0.0。','', '本轮只执行 CPU 离线选择与评分，不是 GPU TTFT、request execution 或 E2E latency 实验。未改变正式总体、budget、候选集、eligibility、Phi、compatibility 或 FlowState。未运行旧策略 selector，未启动引擎、GPU、请求或 trajectory replay。','', '## A. PCB 等价性强制门禁','', '| 核验项 | 结果 |','|---|---|','| OpenHands | 168 snapshots，504 cases，全通过 |','| AgentX | 23 snapshots，69 cases，全通过 |','| PCB vs FlowState objective | 573/573 相同，最大绝对差 0.0 modeled ms |','| 最终 selected set | 573/573 相同 |','| 逐 pending E/G | 573/573 相同 |','| 集合不同但相同前沿 | 0 |','| AgentX 达到全候选可实现前沿 | 69/69 |','| 门禁期间输入哈希 | PASS |','', '执行顺序：先仅创建 common.py、pcb.py、phase1.py 并运行 PCB；pcb_gate.json 记录 PASS 后才创建 LDR/WFC。evaluate.py 开始时验证 PCB、输入适配器和 P1 定义文件的 SHA-256 与门禁一致。策略实现摘要在正式评价前锁定为 implementation_lock.json，完成后再次核验，未按结果调整任何规则。','', '唯一实现依据为 `evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md` Section B。PCB 精确收益 tie 按 checkpoint ID，跨 pending 再按 pending ID；正收益为 >0；不会用 1e-9 门槛过滤推荐。三种策略的结果比较统一采用正式 1e-9 modeled ms 容差。此区分来自冻结设计，未重新设计 tie-breaking。','', '## B. 强基线全量结果','', '胜/平/负均从 FlowState 视角；paired difference = baseline C(S) − FlowState C(S)。统计单位区分 snapshot 与 budget case；三个预算是相同 snapshots 的重复条件，未当成独立样本计算显著性。','', '| 数据集 | 预算 | cases / snapshots | vs PCB 胜/平/负 | vs LDR | vs WFC | 三者 mean / median paired difference, modeled ms |','|---|---|---|---|---|---|---|']
for pop,groups in summary.items():
    for ratio,g in groups.items():
        label='全部预算（描述性汇总）' if ratio=='all' else f'{float(ratio):.0%}'
        vals=[]
        for n in ('PCB','LDR','WFC'):
            v=g['comparisons'][n];vals.append(f'{v["win"]}/{v["tie"]}/{v["loss"]}')
        report.append(f'| {pop} | {label} | {g["cases"]} / {g["snapshots"]} | '+' | '.join(vals)+' | 0.000000 / 0.000000 |')
report += ['', '下面四种方法的 mean/median 完全相同，合并为一列；这里是每个 allocation case 的 C(S)，不是每请求 runtime latency。','', '| 数据集 | 预算 | FlowState = PCB = LDR = WFC mean C(S) | 相同 median C(S) | 各方法达到全候选前沿 | K 真正限制全候选前沿 |','|---|---|---:|---:|---:|---:|']
for pop,groups in summary.items():
    for ratio,g in groups.items():
        c=g['policy_costs']['FlowState'];label='全部预算' if ratio=='all' else f'{float(ratio):.0%}'
        for n in ('PCB','LDR','WFC'):assert g['policy_costs'][n]==c
        report.append(f'| {pop} | {label} | {fm(c["mean"])} | {fm(c["median"])} | {c["reaches_all_frontiers"]}/{g["cases"]} | {g["binding_cases"]} |')
report += ['', '| 数据集 | baseline | objective-equivalent | 相同 selected sets | 不同 selected sets 但 E/G 相同 | 选择顺序不同 |','|---|---|---:|---:|---:|---:|']
for pop,groups in summary.items():
    for n,c in groups['all']['comparisons'].items():report.append(f'| {pop} | {n} | {c["tie"]} | {c["same_set"]} | {c["different_sets_same_EG"]} | {c["order_different"]} |')
report += ['', '分预算同集合和 objective-equivalent 的数量就是各预算全部 cases（OpenHands 168，AgentX 23）；不同集合但同 E/G 的数量在每组都是 0。所有原始 per-continuation T/E/G/C、选择顺序与候选矩阵均保存于 per_case_results.jsonl。','', '### 冻结参考结果（仅读取并复核评分，未重跑 selector）','', '| 数据集 | 预算 | StandaloneTopK mean / median | LRU mean / median | LFU mean / median | Marconi Adaptation mean / median |','|---|---|---|---|---|---|']
for pop,groups in summary.items():
    for ratio,g in groups.items():
        if ratio=='all':continue
        report.append(f'| {pop} | {float(ratio):.0%} | '+' | '.join(f'{fm(g["policy_costs"][n]["mean"])} / {fm(g["policy_costs"][n]["median"])}' for n in ('StandaloneTopK','LRU','LFU','Marconi'))+' |')
report += ['', '这些参考结果保留其原有价值信号和 adaptation 边界。新方法不包装成完整 KVFlow、TOPAS 或 Marconi。','', '## C. 工作负载区分力与 PCB 持平原因','', '### 无共享不等于预算足够','', '- OpenHands 全部 504 cases 均满足独立正收益支持的可分结构。90 cases 的 K<4：预算真正限制服务所有四个 pending，但 PCB 的逐请求最佳 Top-K 仍精确等价；不能把这 90 个 ties 归因于预算宽松。','- 其余 414 cases 的 K≥4，所有方法都保留四个 pending 各自的最深可用项，达到全候选前沿。','- AgentX 中 63 cases 存在正收益共享且 K≥有效 pending 数；3 cases 没有任何正收益候选（g012 的三个预算），四种方法都返回空集；另外 3 cases 为 K=1、两个有效 pending，共同推荐同一个唯一正收益 checkpoint，去重只占一个单位。AgentX 的这三类原因分别记录。','', '预算结构采用“达到所有候选可实现的逐 pending 最深前沿所需的最少 checkpoint 数”定义是否真正受限。每个候选只标记它能达到哪些 pending 的全候选最深位置，再做有限 bitmask 覆盖数分析；这是评价后的结构证书，不是新 baseline，也不向任何选择器提供答案。OpenHands 全部需 4 个；AgentX 27 cases 需 2 个、39 cases 需 1 个、3 cases 需 0 个，全部 ≤K。','', '### 当前能区分与不能区分的内容','', '- 可以区分旧静态冗余保留和这些 workflow-aware 方法；P0 对 StandaloneTopK 的 144/504、2/69 胜例仍成立，但不是 FlowState 相对强简单方法的独特收益。','- 不能在最终 modeled objective、selected set 或 E/G 上区分 FlowState、PCB、LDR、WFC：全量完全相同。','- 可以观察部分中间顺序差异：OpenHands LDR/WFC 各 3 cases（1 snapshot×3 budgets）；AgentX PCB/LDR 各 9 cases（3 snapshots×3 budgets），WFC 18 cases（6 snapshots×3 budgets）。这不是新的独立性能样本。','- 没有严格正差、负差或接近零但为正的改善案例；全部差值精确为零。没有人为挑选低收益阈值，也没有删除 ties。','', '## D. 机制与真实案例','', '完整七例见 [case_studies.md](case_studies.md)，原始来源行号见 case_index.json；所有顺序差异而不只是代表例见 order_difference_cases.json；strict_cases.json 为空数组。','', '### 位置感知估值：顺序有效，但最终选择没有额外收益','', '`rq3-openhands-main-g051-round5` 的 K=5/10/15 三个 cases：C 与 D 最佳 checkpoint 的 L 均为 7808，但 T_C=7808、T_D=7999。共同前缀已保留 A、B 后，FlowState 对 D 的 marginal 为 299.520864654 modeled ms，对 C 为 299.473119577 modeled ms，因此先选 D；LDR/WFC 在相同长度或 q/r 下按 ID 先选 C。最终 C、D 都能留下，objective 差为零。','', '**Position-aware Cost Model 在当前 allocation population 下尚无相对 LDR 的额外最终选择收益证据。** 这不否定恢复模型的拟合准确性，也不否认它改变了该例排序；只能说明现有预算边界没有把此排序差异转化为最终代价改善。','', '### 共享总收益与 per-continuation 推荐的区别','', '`rq3-agentx-g000-epoch109`，K=27：FlowState 首选 req0000，q=2、r=75520，marginal=3481.852412350 modeled ms；PCB/LDR 首选 req0005，q=1、r=51328，marginal=2470.221830882 modeled ms。两种顺序最终都保留 req0000 与 req0005，E/G/C 相同。此类顺序差异覆盖 g000、g007、g014 的三个预算，共 9 cases。','', '### WFC 的 coverage-first 规则确实不同','', '`rq3-agentx-g005-epoch8`，K=2：WFC 首选 req0000（L=384，q=2，r=768，marginal=29.101911990 modeled ms）；FlowState 首选 req0001（L=43840，q=1，r=43840，marginal=1982.352784905 modeled ms）。WFC 没有偷偷使用 Phi，否则不会作出该 q 优先选择；最终两个 checkpoint 都保留，成本仍完全相同。','', '原来相对 StandaloneTopK 的两个严格胜例也没有保留对强基线的优势：','', '- g006/K8：FlowState 先 req0011（q=1，r=103104，marginal 5807.509134874），WFC 先共享 req0001（q=2，r=96896，marginal 4668.125454620）；最终都保留这两个，C=6832.743154525 modeled ms。','- g018/K7：FlowState 先 req0012（q=1，r=105344，marginal 5963.868027112），WFC 先共享 req0001（q=2，r=94720，marginal 4548.133889245）；最终都保留这两个，C=6745.074698151 modeled ms。','', '因此 **global set-dependent allocation 在当前冻结总体上没有相对 PCB/LDR/WFC 独立可观察的最终 modeled-cost 增益**。这不是“永远不需要全局优化”的证明；只是原实验总体与预算缺少能惩罚不同排序的有效竞争。','', '## E. 正确性与可复现性','', '| 核查 | 结果 |','|---|---|','| PCB 强制门禁 | 573/573 PASS，先于 LDR/WFC 实现 |','| 定义边界测试 | 10/10 PASS：空输入、K=0、零/不兼容、精确 tie、共享 ID 去重后继续扫描、部分前沿改善、q 优先、位置收益反转、Phi 隔离、非法集合拒绝 |','| 输入排列不变性 | 1719/1719 PASS；反转候选及 pending 顺序不改变选择顺序 |','| 新增策略独立高精度评分 | 1719/1719 PASS；Decimal 公式、独立位置枚举与同一 Phi 结果一致 |','| LDR/WFC 无 Phi 决策 | 1146/1146 PASS；决策期间将模型调用替换为立即失败的 guard |','| 所有旧参考集合评分复现 | 2865/2865 PASS；只评分冻结集合，不运行旧 selector |','| K、候选合法性、compatibility、逐 pending E/G、成本求和 | PASS |','| 冻结源、P0、P1 定义及本轮所读核心文件哈希 | PASS |','| 新策略实现锁定摘要 | PASS |','', '统一 scoring 采用冻结兼容矩阵上的 E=max L、G=T−E、C=sum Phi，并直接调用未修改的 RecoveryCostModel。未调用重建 population 的 builder，也未伪造 AgentX 的原始 lineage。这个纯数据评分适配器的全部旧集合结果已经逐 pending 对照正式 scorer 的冻结输出；另以独立 Decimal 公式核验新策略。','', '防污染：selector 接口只接收不可变 Epoch(candidates,pending,K)，候选字段仅 ID/L/compatible IDs，pending 仅 ID/T；不包含 FlowState selected set、旧 objective 或测量 latency。T 虽在正式 per-continuation 结果中保存，仍作为冻结输入读取；与所有旧策略同名字段逐项核对，OpenHands 另对 canonical snapshot 的 min(anchor,FA frontier) 核对。AgentX 的 compatibility 与 formal_snapshot_manifest.json 逐项一致，并核验正式 common_input_verified 与跨策略 input digest。旧结果只用于选择完成后的比较。','', '数值：PCB 的选择按冻结定义使用精确分数排序；LDR/WFC 使用整数位置/推进。正式比较容差为 1e-9 modeled ms，独立高精度核查不参与选择，也不修改模型。实测最终差值恰为零，持平不依赖扩大 epsilon。','', '主要源码：`pcb.py:5`、`ldr.py:4`、`wfc.py:4`；公共输入与评分 `common.py`；强制门禁 `phase1.py`；全量评估与结构证书 `evaluate.py`；测试 `test_contracts.py`。准确源码摘要见 implementation_lock.json，P1 文档摘要见 pcb_gate.json。','', '输入来源：','', '- OpenHands：`evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl` 及 `rq3_openhands_main_formal_20260904_001017/snapshots/`。','- AgentX：`rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl` 与 `formal_snapshot_manifest.json`。','- StandaloneTopK：`evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726/per_case_results.jsonl` 的 `without_set_dependent`。','- 定义、P0 与旧核心语义的逐文件摘要记录在 source_integrity_before.json；每个新结果都有正式 source_file/source_line、snapshot_digest、decision input_digest 和 ablation 来源行。','', '可复核命令仅供复现本轮 CPU 分析：先运行 phase1.py，检查 pcb_gate.json 为 PASS，再运行 test_contracts.py、evaluate.py、report.py；设置 PYTHONDONTWRITEBYTECODE=1。这些步骤不调用 GPU。运行环境为 `/home/wjg/uv312/bin/python`。本轮已经完成，不自动启动额外流程。','', '## F. 论文含义与下一步','', '必须弱化的 claim：','', '- “当前 population 证明一般 Global Executable-State Allocation 优于简单 workflow-aware/per-continuation retention”：不成立。当前四种方法最终选择相同。','- “位置感知 valuation 在该 population 提供独立的 allocation cost 改善”：无证据；目前只观察到排序改变。','- “对 Marconi 或 StandaloneTopK 的全部改善证明完整 marginal allocator 的必要性”：不成立；强简单基线也达到同一 objective。','- “AgentX 的两个 StandaloneTopK 胜例证明同时多分支 marginal 的独特性能价值”：不能保留；新 baseline 同样达到该前沿。','', '可保留：原有有限策略对比、原 objective 下的去冗余解释、已有 Exact OPT 匹配事实、共享收益存在与排序影响；不能外推成本等价为 runtime/control latency 等价。当前未提供论文正文原文件，以上指向主张类型，不伪造 Section 6.4 的逐字引文。','', 'Section 6.4 最小英文补充：','', '> On all 573 frozen snapshot-budget cases (504 OpenHands and 69 AgentX), FlowState, Per-Continuation Best-Checkpoint, Lineage-Aware Deepest Retention, and Workflow-Aware Frontier Coverage select identical checkpoint sets and achieve identical modeled recovery costs. Some selection orders differ, but these differences do not change the final executable frontiers. Thus, the current allocation population does not establish an additional modeled-cost benefit from position-aware valuation or global set-dependent allocation over these workflow-aware retention baselines.','', '**是否值得新增真实受限共享 cohort：有条件值得，但当前不启动。** 科研价值在于补足当前缺失的“共享祖先与分支专属深 checkpoint 的有限预算取舍”，而不是保证 FlowState 获胜。新 cohort 需要：多个真实且当前 pending 的分支；真实可验证的父子继承；同时存在共享和分支专属的有效驻留候选；K 小于同时实现各分支最深前沿所需的 checkpoint 数；候选收益重叠使不同排序在预算边界影响最终 E/G。仅增加零收益候选、无可恢复 pending、lineage 深度或不可消除 residual 不够。','', '未来应事先固定来源、时间窗口、policy-blind 纳入规则及预算协议，独立保留新 cohort，完整报告所有门禁通过样本；不得回筛当前正式总体、按胜负挑选或事后降低预算。若数据来源自然缺乏所需结构，应保留本轮无区分力结论。GPU 仅在另行验证这些新策略真实 runtime/control cost 时才有必要；本轮没有该结果，也没有启动采集、GPU 或 runtime 优化。','', '当前三种算法实现与证据足以判断当前 population 的区分力；无需为制造收益继续调算法。结论选择 WORKLOAD_NOT_DISCRIMINATIVE，而不是把全部 PASS 的正确性包装为性能优势。','']
(OUT/'final_report.md').write_text('\n'.join(report)+'\n')
assert input_hashes()==read(OUT/'source_integrity_before.json')
validation['conclusion']=status
validation['implementation_unchanged_after_reporting']=all(sha(OUT/n)==v for n,v in read(OUT/'implementation_lock.json').items())
assert validation['implementation_unchanged_after_reporting']
write('validation.json',validation)
write('artifact_manifest.json',dict(status=status,说明='独立 P1-B 产物摘要；清单本身不进入摘要。',files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}))
print(status)
print('输入摘要核对通过：',validation['input_hashes_unchanged'])
