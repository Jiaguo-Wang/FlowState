"""把已核对的冻结结果生成中文报告、案例材料和两张独立矢量图。"""
from pathlib import Path
import json, statistics, html, hashlib, xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
rows=[json.loads(s) for s in (OUT/'all_cases.jsonl').read_text().splitlines()]
summary=json.loads((OUT/'summary.json').read_text())
EPS=1e-9

def dump(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def fmt(x):return f'{x:.3f}'
def key(r):return (r['population'],r['snapshot_id'],r['k'])
def positive_median(b):
    rr=sorted([r for r in rows if r['population']=='OpenHands' and r['improvement_ms'][b]>EPS],key=lambda r:(r['improvement_ms'][b],r['snapshot_id'],r['k']))
    return rr[(len(rr)-1)//2]
def find(sid,k):return next(r for r in rows if r['snapshot_id']==sid and r['k']==k)
chosen=[
 ('A','OpenHands 对 Marconi 最大绝对改善；同时也是对 StandaloneTopK 最大改善',max([r for r in rows if r['population']=='OpenHands'],key=lambda r:r['improvement_ms']['Marconi'])),
 ('B','OpenHands 对 Marconi 严格改善排序的下中位案例',positive_median('Marconi')),
 ('C','OpenHands 对 StandaloneTopK 严格改善排序的下中位案例',positive_median('StandaloneTopK')),
 ('D','AgentX 第一个严格优于 StandaloneTopK 的案例',find('rq3-agentx-g006-epoch35',8)),
 ('E','AgentX 第二个严格优于 StandaloneTopK 的案例',find('rq3-agentx-g018-epoch30',7)),
 ('F','AgentX 不同集合但 objective 相同：按预算、ID 升序首例',find('rq3-agentx-g000-epoch109',27)),
 ('G','OpenHands 对 Marconi 持平：按预算、ID 升序首例',find('rq3-openhands-main-g009-round3',3)),
]
english={
'A': 'In this OpenHands snapshot, StandaloneTopK spends all four slots on one workflow. After retaining its deepest checkpoint, FlowState assigns the remaining slots to three other workflows, reducing modeled recovery cost by 753.200 ms relative to StandaloneTopK and 582.242 ms relative to Marconi Adaptation. These are modeled allocation costs, not measured runtime latencies.',
'B': 'This median-positive Marconi comparison yields a 115.491 ms modeled-cost reduction, while FlowState and StandaloneTopK have identical objectives. The example therefore supports a difference in checkpoint valuation relative to Marconi, but does not demonstrate an additional benefit from set-dependent updates.',
'C': 'At the first divergent selection, the statically preferred checkpoint has zero additional benefit relative to the retained deeper checkpoint. FlowState instead covers another pending workflow, reducing the final modeled recovery cost by 174.070 ms relative to StandaloneTopK.',
'D': 'After retaining req0011, req0010 adds no recovery benefit. FlowState instead selects the shared ancestor req0001, whose incremental benefit at this step is confined to the other pending continuation. The resulting 2333.666 ms improvement supports redundancy-aware reallocation, not a claim that simultaneous gains on multiple branches caused this win.',
'E': 'The second strict AgentX case exhibits the same mechanism: after req0012, FlowState selects req0001 for a 2276.008 ms gain on the uncovered continuation, whereas the next static choice adds zero benefit. Although req0001 has positive standalone benefit for both continuations, only one contributes to its current marginal gain.',
'F': 'A shared checkpoint contributes 1740.926 ms to each of two continuations and is preferred over an alternative with a larger individual but smaller aggregate benefit. Nevertheless, FlowState and StandaloneTopK attain identical executable frontiers and the same final objective; the additional static selections are redundant.',
'G': 'FlowState and Marconi Adaptation attain the same modeled recovery objective in this snapshot. Different checkpoint choices should not be interpreted as a performance improvement when their per-continuation executable frontiers coincide.',
}
case_doc=['# 代表性案例：冻结分配结果机制审计','', '所有成本单位均为模型预测 ms，不代表 runtime latency。案例在全量 573 cases 审计完成后按以下规则确定。C 编号只用于本案例内映射；P 编号顺序与冻结 per-continuation 记录一致。','']
case_index=[]
for label,rule,r in chosen:
    cmap={c:f'C{i+1}' for i,c in enumerate(sorted(r['candidates']))};pmap={p:f'P{i+1}' for i,p in enumerate(r['pending_order'])}
    r=dict(r);r['case_label']=label;r['selection_rule']=rule;r['checkpoint_aliases']=cmap;r['pending_aliases']=pmap;r['paper_explanation']=english[label]
    dump(f'case_{label}.json',r)
    case_index.append(dict(label=label,rule=rule,snapshot_id=r['snapshot_id'],k=r['k'],budget_ratio=r['budget_ratio'],source_file=r['source_file'],source_line=r['source_line'],improvement_ms=r['improvement_ms']))
    case_doc += [f'## 案例 {label}：{r["snapshot_id"]}', '',f'选择规则：{rule}。预算 {r["budget_ratio"]:.0%}，K={r["k"]}；候选 {r["candidate_count"]}，待续请求 {r["pending_count"]}。', '',f'冻结来源：`{r["source_file"]}:{r["source_line"]}`。完整矩阵、全部候选每步 marginal、血缘/metadata 和原始 ID 见 [case_{label}.json](case_{label}.json)。','', '| 请求 | 原始 continuation ID | T | workflow（冻结编码） |','|---|---|---:|---|']
    for p in r['pending']:case_doc.append(f'| {pmap[p["continuation_id"]]} | `{p["continuation_id"]}` | {p["target_tokens"]} | `{p["workflow_id"]}` |')
    case_doc += ['', 'OpenHands 的四个 pending 对应独立 workflow、不同 lineage；AgentX 的表中 workflow 是正式兼容编码，真实完整 lineage 路径未被本轮读取的正式 manifest 保存，标记 UNKNOWN。AgentX 使用 manifest 的正式兼容集合，不能从 ID 字面推断真实血缘。','', '| 候选 | 原始 checkpoint ID | L | 可用于 objective 的 pending | 各 pending 的 standalone benefit（按 P 顺序） |','|---|---|---:|---|---|']
    for c,v in r['candidates'].items():
        case_doc.append(f'| {cmap[c]} | `{c}` | {v["token_pos"]} | '+', '.join(pmap[p] for p,val in r['compatibility'][c].items() if val)+' | '+', '.join(fmt(v) for v in r['benefit_matrix_ms'][c])+' |')
    case_doc += ['', '| 策略 | 选中集合（编号） | C(S), ms | 相对 FlowState 多出的成本, ms |','|---|---|---:|---:|']
    for name,v in r['policies'].items():case_doc.append(f'| {name} | '+', '.join(cmap[c] for c in v['selected'])+f' | {fmt(v["cost_ms"])} | {fmt(r["improvement_ms"].get(name,0.))} |')
    case_doc += ['', 'StandaloneTopK 完整正收益静态排序：'+ ' → '.join(cmap[c] for c in r['standalone_ranking'])+'。', '', '| 贪心步 | 选中 checkpoint | 当前 marginal, ms | 选择前 u（按 P 顺序）, ms | 本步逐 pending 新增收益, ms |','|---|---|---:|---|---|']
    for s in r['flowstate_steps']:case_doc.append(f'| {s["step"]} | {cmap[s["chosen"]]} | {fmt(s["delta_ms"])} | '+', '.join(fmt(v) for v in s['u_before_ms'])+' | '+', '.join(fmt(v) for v in s['per_pending_gain_ms'])+' |')
    if r['first_divergence']:
        d=r['first_divergence'];case_doc += ['',f'首个分歧：第 {d["step"]} 步；共同 retained set = '+str([cmap[c] for c in d['retained_set']])+'。']
        for name in ('flowstate','standalone'):
            v=d[name];case_doc.append(f'- {name}：'+('停止选择，剩余候选 marginal 为零。' if v is None else f'{cmap[v["checkpoint_id"]]}；独立收益 {fmt(v["standalone_ms"])} ms；相对上述 retained set 的 marginal {fmt(v["marginal_ms"])} ms。'))
    else:case_doc += ['', 'FlowState 与 StandaloneTopK 选中集合相同，无集合分歧。']
    case_doc += ['', '| 请求 | 策略 | E | G | C_p, ms | baseline − FlowState, ms |','|---|---|---:|---:|---:|---:|']
    for name,v in r['policies'].items():
        for i,p in enumerate(v['per_pending']):case_doc.append(f'| {pmap[p["continuation_id"]]} | {name} | {p["E"]} | {p["G"]} | {fmt(p["cost_ms"])} | {fmt(v.get("delta_per_pending_ms",[0.]*r["pending_count"])[i])} |')
    mar=r['policies']['Marconi']['selected'];zero=r['final_removable_zero']['Marconi']
    case_doc += ['', 'Marconi 价值信号：'+('原始 last_access_order、marconi_flop_saved、alpha 和归一化 utility 均见本案例 JSON；以原始 utility 重新排序，选中集合核对通过。' if r['population']=='OpenHands' else 'UNKNOWN：正式逐案例文件未保留原始排序 metadata；仅核对冻结输出与共同成本，不臆测内部排序原因。'), '', 'Marconi 最终集合中可删除而不增加成本的 checkpoint：'+(', '.join(cmap[c] for c in zero) if zero else '无')+'。对每个 c，检查的是 Δ(c | S_Marconi \\ {c})；这里的 retained set 由本节完整选中集合明确给出，绝不把它当成 Marconi 内部选择顺序。','',f'英文论文解释：{english[label]}','']
(OUT/'case_studies.md').write_text('\n'.join(case_doc)+'\n');dump('case_index.json',case_index)

names={'A_overlap':'A：同一 pending 的候选收益重叠','A_standalone_redundant':'A：StandaloneTopK 最终集合存在零删除边际','A_marconi_redundant':'A：Marconi 最终集合存在零删除边际','B_independent_workflows':'B：确认存在独立工作流','B_budget_binding_independent':'B：K 少于正收益独立工作流数','C_structural':'C：正式结构允许 checkpoint 覆盖多个 pending','C_positive':'C：至少一个 checkpoint 对多个 pending 有正独立收益','C_multiple_positive_step':'C：实际贪心某一步对多个 pending 有正新增收益','C_selected_essential':'C：选中共享 checkpoint 的删除会增加成本','C_joint_rank_reversal':'C：联合求和与最大单项贡献发生相对排序反转','D_standalone_diff_tie':'D：与 StandaloneTopK 集合不同但成本相同','D_marconi_diff_tie':'D：与 Marconi 集合不同但成本相同','E_marconi_tie':'E：与 Marconi 成本相同','E_marconi_small_positive':'E：对 Marconi 改善在 (1e-9, 1] ms','E_marconi_adverse':'E：对 Marconi 成本更高'}
report=['# P0 分配机制与案例审计','', '**PARTIALLY_SUPPORTED**','', '已冻结结果支持去冗余后的跨待续请求预算分配；OpenHands 的独立工作流竞争证据明确。AgentX 有真实共享收益和联合排序效应，但不能把两个相对 StandaloneTopK 的严格胜例归因于同时覆盖多个分支的新增收益。以下统计单位均为 snapshot-budget case，三个预算不是独立 snapshots；本报告不把 573 cases 当成 573 个独立统计样本做推断。所有差值为 baseline C(S) − FlowState C(S)，单位为模型预测 ms，不是 GPU latency。','', '## 输入与只读边界','', '- OpenHands：168 snapshots × 3 budgets = 504 cases。正式结果：`evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl`；原始 snapshots：`evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/snapshots/`。','- AgentX：23 snapshots × 3 budgets = 69 cases。正式结果：`rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl`；兼容性 manifest：同目录 `formal_snapshot_manifest.json`。','- 两个总体的完整 StandaloneTopK 决策均存在：`evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726/per_case_results.jsonl`，使用 `without_set_dependent`；没有临时新增 baseline。','- 全量解释数据：[all_cases.jsonl](all_cases.jsonl)。每行包含来源文件及行号、候选与兼容矩阵、T/L/E/G、所有策略成本、全部 standalone benefit、每个贪心前缀的所有剩余候选 marginal、u、首个分歧与最终零删除边际。','- 分析脚本：[analyze.py](analyze.py)；报告与矢量图脚本：[report.py](report.py)。没有修改原始结果、策略或候选总体。','', '## 全量收益分布','', '| 数据集 | baseline | 胜 / 平 / 负 | 平均差值, ms | 中位差值, ms | P25 / P75, ms | 最大差值, ms | 不同选中集合 |','|---|---|---|---:|---:|---|---:|---:|']
for pop,s in summary.items():
    for b,v in s['comparisons'].items():
        q=v['improvement_ms'];w=v['wtl'];report.append(f'| {pop} | {b} | {w["win"]} / {w["tie"]} / {w["loss"]} | {fmt(q["mean"])} | {fmt(q["median"])} | {fmt(q["p25"])} / {fmt(q["p75"])} | {fmt(q["max"])} | {v["different_sets"]} |')
report += ['', '| 数据集 | 预算 | vs Marconi 胜/平/负 | vs LRU | vs LFU | vs StandaloneTopK |','|---|---|---|---|---|---|']
for pop,s in summary.items():
    for ratio,entry in s['by_budget'].items():report.append(f'| {pop} | {float(ratio):.0%} | '+' | '.join('/'.join(str(entry[b]['wtl'][k]) for k in ('win','tie','loss')) for b in ('Marconi','LRU','LFU','StandaloneTopK'))+' |')
report += ['', 'OpenHands 的三预算 W/T/L 与冻结 `rq3_sanity_structure_audit_20260904_121500/win_tie_loss.json` 完全一致；四个正式策略在两个总体、三个预算下的平均 C(S) 均与原 aggregate_results.json 一致。没有负例。近零正收益采用预先明确的描述性区间 (1e-9, 1] ms：两个总体对 Marconi 都没有此类 case；没有更改正式 1e-9 ms tie 容差。','', '## 机制分类及分布','', '分类允许重叠，不可相加为 100%。A 的单位是同一个可兼容 pending 的收益覆盖；OpenHands 同时有完整 lineage 路径佐证，AgentX 以正式祖先兼容集合为准。B 的严格容量竞争条件是 K 小于正收益独立工作流数；K≥工作流数时，静态 TopK 仍可能因冗余占用而漏掉其他工作流。AgentX 未在本轮恢复真实完整 workflow/lineage metadata，因此 B 显示 UNKNOWN；逐案例 JSON 中 B=null、汇总 unknown_count=69，不能把未确认当成不存在。','', '| 数据集 | 分类 | 数量 / 占比 | vs Marconi 平均 / 中位 [最小, 最大], ms | vs StandaloneTopK 平均 / 中位 [最小, 最大], ms |','|---|---|---|---|---|']
for pop,s in summary.items():
    for tag,v in s['classifications'].items():
        if pop=='AgentX' and tag.startswith('B_'):report.append(f'| {pop} | {names[tag]} | UNKNOWN | UNKNOWN | UNKNOWN |');continue
        def cell(q):return '不适用' if not q['n'] else f'{fmt(q["mean"])} / {fmt(q["median"])} [{fmt(q["min"])}, {fmt(q["max"])}]'
        report.append(f'| {pop} | {names[tag]} | {v["count"]} / {v["fraction"]:.2%} | {cell(v["vs_marconi_ms"])} | {cell(v["vs_standalone_ms"])} |')
report += ['', '完整 P25/P75/P95 分布见 [summary.json](summary.json)。分类是证据描述，不是额外策略实验：C 联合排序反转要求实际选中 c 的总 marginal 大于某个候选 d，但 c 的最大单个 pending 新增收益小于 d 的最大单项。只比较已经计算的贡献向量，没有构造或运行新 baseline。AgentX 该现象为 9 cases（3 snapshots × 3 budgets），均与 StandaloneTopK 最终成本持平；它说明联合求和影响相对排序，不证明相对 StandaloneTopK 的独特最终收益。','', '## 持平原因','', '| 数据集 | 与 StandaloneTopK 持平原因 | cases | 达到所有候选可实现的最大前沿 | C(S)=0 |','|---|---|---:|---:|---:|']
for pop,s in summary.items():
    for reason,v in s['ties'].items():report.append(f'| {pop} | {reason} | {v["count"]} | {v["reaches_all_candidate_frontier"]} | {v["zero_cost"]} |')
report += ['', '- OpenHands：360/504（71.43%）持平，其中 73 集合相同，287 集合不同；后者全部逐 pending 的 E/G 相同，静态额外保留不会改善目标。267 个不同集合 tie 已零成本，其余 20 个仍有非零但候选无法消除的残余 gap。','- AgentX：67/69（97.10%）持平，其中 45 集合相同、22 集合不同。67 个都达到所有现有候选可实现的前沿，但全部 C(S)>0；不能把非零不可恢复残余误称为分配失败，也不能把不同集合误称为收益。','- 所有不同集合 objective ties 都具有逐 pending 相同前沿；没有通过不同 pending 成本正负抵消制造的 tie。','', '## 机制归因与论文边界','', '1. **相对 Marconi 的主要证据是价值信号与保留位置不同。** OpenHands Marconi 用归一化最近访问值 + alpha×归一化增量计算效率；FlowState 用同一恢复模型下当前可执行前沿的增益。386 个胜例中 245 个与 StandaloneTopK 成本持平；不能把对 Marconi 的全部收益归因于 marginal updates。323 个 Marconi 胜例还包含最终可删除的冗余状态，但这是最终集合诊断，不是 Marconi 内部顺序证据。AgentX 16 个 Marconi 胜例中 14 个与 StandaloneTopK 持平；其原始 utility metadata 缺失，不能补写排序动机。','2. **相对 StandaloneTopK 的额外收益可归因于冗余消除后的重分配。** 全部 OpenHands 144 个、AgentX 2 个严格胜例的首个分歧处，静态下一候选相对共同 retained prefix 的 marginal 为零；FlowState 的替代选择对另一个 pending 有正收益。OpenHands 还明确跨越独立 workflows；这不是因为改变了 standalone score 或 K。','3. **Global allocation 在 OpenHands 的主要可观察作用是独立 workflow 预算竞争。** 四个 workflows 无 cross-continuation sharing；504 cases 都有同一 pending 的收益重叠，90 cases 的 K 少于可服务工作流数。其余 cases 仍可通过避免静态冗余占用而覆盖更多 workflows。不能从此推广“AgentX 也主要是独立 workflows”，因为真实完整血缘未保留在读取的正式 manifest 中。','4. **AgentX 有共享结构和共享收益，但两胜例不证明同时多分支新增收益带来的严格优势。** 69 cases 有结构共享，66 有正收益共享，48 在实际贪心某一步同时增加多个 pending 的收益，9 有联合排序反转；两个严格胜例都在第二步选择共享祖先，但此时其在已覆盖分支上的 marginal 已为零。共享可用性参与了恢复另一个分支，却没有被隔离为独立因果因素；StandaloneTopK 本身也计算所有 pending 的联合独立收益。','5. **当前证据边界。** 支持冻结 allocation objective 下的冗余感知全局分配；不支持“共享联合收益普遍导致相对独立 TopK 严格改善”、所有阶段/任意拓扑最优或运行时加速的泛化结论。当前不必为冗余重分配 claim 新增 baseline；若论文坚持共享建模本身带来独特优势，需要另行控制共享因素的证据，当前未隔离该因素。本轮不启动 P1。','', '## 七个代表性案例','', '| 案例 | snapshot | K | vs Marconi, ms | vs StandaloneTopK, ms | 选择依据 |','|---|---|---:|---:|---:|---|']
for label,rule,r in chosen:report.append(f'| {label} | {r["snapshot_id"]} | {r["k"]} | {fmt(r["improvement_ms"]["Marconi"])} | {fmt(r["improvement_ms"]["StandaloneTopK"])} | {rule} |')
report += ['', '全部 candidates、原始 IDs、compatibility、selected sets、贪心过程与逐 pending 成本见 [case_studies.md](case_studies.md) 和 case_A.json 至 case_G.json。每个 baseline 的逐 pending ΔC 之和均核对为 aggregate ΔC。两个 AgentX 严格案例的机制相同，仍逐一列出，避免用单例替代完整审查。','', '## 论文与组会建议','', '- 论文图 1：[figure_marginal.svg](figure_marginal.svg)。显示案例 A/D/E 首个分歧时，独立高分候选的当前 marginal 已归零。三个面板使用独立标注的尺度，不比较未归一化柱长。','- 论文图 2：[figure_cost.svg](figure_cost.svg)。显示案例 A/B/D/F 的逐 pending 成本，包括中等 Marconi 收益但静态策略持平，以及 AgentX 的高残余成本 tie。最多这两张，不自动编辑论文。','- 组会第 1 页：两个总体、四 baseline 的全量 W/T/L 与 ties；强调不是 GPU latency。','- 组会第 2 页：案例 A/D/E 的首次分歧和余量重分配，使用图 1。','- 组会第 3 页：案例 F 的联合收益存在但最终 tie，配合图 2、未知 metadata 和 claim 边界。','', '可增加的最小英文分析段落：','', '> On the frozen allocation cases, FlowState improves modeled recovery cost over StandaloneTopK in 144/504 OpenHands cases and 2/69 AgentX cases, with ties in all remaining cases. At the first divergent selection in every strict win, the static choice has zero marginal benefit relative to the common retained prefix, whereas FlowState selects a checkpoint that benefits another pending continuation. The two AgentX wins involve shared ancestors, but each divergent step contributes new benefit to only one continuation; they therefore support redundancy-aware reallocation rather than an isolated advantage from simultaneous multi-branch gains. Different selected sets alone are not evidence of improvement: 287 OpenHands cases and 22 AgentX cases retain identical per-continuation executable frontiers despite different selections.','', '## 验证与证据定位','', '- 573 cases、2865 个策略 objective、573 次现有 GlobalOptimizer CPU 回放、573 个 StandaloneTopK 集合全部核对通过。OpenHands 504 个 Marconi 集合由冻结原始 utility 重新排序核对通过。','- 使用原有 GlobalOptimizer.select，成本钩子仅以冻结兼容矩阵和原 RecoveryCostModel 计算同一 C(S)；没有产生新总体，也没有修改 policy。按正式记录检查 eligibility、K、candidate IDs、T、compatibility 和 digest；AgentX common_input_verified 与各策略 input digest 一致。','- StandaloneTopK 的逐项收益求和排序与消融记录的总成本差展示排序在这 573 cases 中顺序相同。正式 1e-9 ms 比较容差保留；greedy 停止条件仍来自现有代码，不额外设阈值。','- Section 4 分解核对：b(p,c)=Phi(T,T)−Phi(T−L,T)（仅对正式可兼容候选）；u_p=max b；C(S)=C(空集)−sum u_p；Delta=sum max(b−u,0)。逐 pending 成本和冻结 E/G 全部一致。','- 原实现位置：`flowstate/optimizer.py:41`（选择）、`:95`（marginal）、`:110`（tie）、`:124`（停止）；`flowstate/recovery_model.py:60`（Phi）；`flowstate/state_catalog.py:60`（compatibility）；`evaluation/sota_policies.py:100` 与 `:253`（Marconi utility）；`evaluation/rq3_sanity_structure_audit.py:1116`（StandaloneTopK）；`evaluation/rq5_frozen_snapshot_ablation.py:196`（冻结展示顺序）；`evaluation/rq3_agentx_formal_policy_evaluation.py:1101`（正式 manifest 写出字段）。','- Frozen OpenHands W/T/L 报告：`evaluation/runtime_artifacts/rq3_sanity_structure_audit_20260904_121500/STEP_13G_A_FINAL_REPORT.md:232`。全量源行号逐例记录在 all_cases.jsonl 的 source_file/source_line，七例同时列在案例文档中。','- UNKNOWN：AgentX 原始 Marconi metadata 与完整真实 lineage 路径；未从 ID 或 encoded workflow 推断补齐。正式兼容集合、checkpoint L、pending T 和全部结果仍可完整审计。','- 文件 SHA-256 的前后结果与验证门禁见 [integrity.json](integrity.json)。仅独立分析目录有新产物；未启动 GPU、Engine、trajectory 或后续开发。','']
(OUT/'final_report.md').write_text('\n'.join(report)+'\n')

# 图 1：冻结首次分歧的独立收益与当前边际，不绘制推断数据。
COLORS={'FlowState':'#28658e','Marconi':'#b56c36','StandaloneTopK':'#858585'}
def start(w,h):return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<rect width="{w}" height="{h}" fill="white"/>','<g font-family="Noto Sans CJK SC, sans-serif" fill="#222">']
def text(svg,x,y,s,size=14,anchor='start'):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
def rect(svg,x,y,w,h,color):svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0,w):.2f}" height="{h}" fill="{color}"/>')
def finish(svg,name):
    svg.extend(['</g>','</svg>']);content='\n'.join(svg);ET.fromstring(content);(OUT/name).write_text(content)
svg=start(980,700);text(svg,25,32,'首次分歧：静态高分不等于当前新增收益',21)
rect(svg,30,53,20,12,'#999');text(svg,58,65,'独立收益',13);rect(svg,170,53,20,12,'#28658e');text(svg,198,65,'相对共同 retained prefix 的当前边际',13)
figdata=[]
for n,label in enumerate(('A','D','E')):
    r=next(v for l,_,v in chosen if l==label);d=r['first_divergence'];y=110+n*195
    text(svg,25,y,f'{label}  {r["snapshot_id"]}   K={r["k"]}   第 {d["step"]} 步',16)
    prefix=', '.join(c.split('::')[-1] for c in d['retained_set']);text(svg,25,y+24,'已保留：'+prefix,12)
    maximum=max(d[n]['standalone_ms'] for n in ('flowstate','standalone'))*1.15;x0=280;scale=600/maximum
    for j,name in enumerate(('standalone','flowstate')):
        v=d[name];yy=y+49+j*51;alias=v['checkpoint_id'].split('::')[-1]
        text(svg,25,yy+12,('静态下一项：' if name=='standalone' else 'FlowState：')+alias,12)
        rect(svg,x0,yy,v['standalone_ms']*scale,12,'#999');text(svg,x0+v['standalone_ms']*scale+5,yy+11,fmt(v['standalone_ms']),11)
        rect(svg,x0,yy+16,v['marginal_ms']*scale,12,'#28658e');text(svg,x0+v['marginal_ms']*scale+5,yy+27,fmt(v['marginal_ms']),11)
    yy=y+155;svg.append(f'<path d="M{x0},{yy} H880" stroke="#555" fill="none"/>')
    for j in range(4):
        xx=x0+j*200;text(svg,xx,yy+17,fmt(maximum*j/3),11,'middle')
    text(svg,960,yy+17,'ms',11,'end');figdata.append(dict(case=label,source_file=r['source_file'],source_line=r['source_line'],first_divergence=d))
finish(svg,'figure_marginal.svg');dump('figure_marginal_data.json',figdata)

# 图 2：每个 pending 的共同模型成本，包含正例和持平例。
svg=start(1060,770);text(svg,25,33,'逐 pending 成本分解：正收益、中等收益与持平',21)
for j,name in enumerate(COLORS):rect(svg,25+j*240,53,18,12,COLORS[name]);text(svg,50+j*240,65,name,13)
figdata=[]
for n,label in enumerate(('A','B','D','F')):
    r=next(v for l,_,v in chosen if l==label);x=35+(n%2)*520;y=110+(n//2)*320
    text(svg,x,y,f'{label}  {r["snapshot_id"]}',14);text(svg,x,y+22,f'K={r["k"]}；模型恢复成本（ms）',12)
    maximum=max(p['cost_ms'] for name in COLORS for p in r['policies'][name]['per_pending'])*1.12 or 1.;base=y+226;height=160;left=x+55;width=410;np=r['pending_count'];group=width/np
    for j in range(4):
        value=maximum*j/3;yy=base-height*j/3;svg.append(f'<path d="M{left},{yy} H{left+width}" stroke="#e3e3e3" fill="none"/>');text(svg,left-5,yy+4,f'{value:.0f}',11,'end')
    for i in range(np):
        for j,name in enumerate(COLORS):
            v=r['policies'][name]['per_pending'][i]['cost_ms'];barw=min(22,group/5);xx=left+group*(i+.5)+(j-1.5)*barw
            rect(svg,xx,base-height*v/maximum,barw-2,height*v/maximum,COLORS[name])
        text(svg,left+group*(i+.5),base+20,f'P{i+1}',12,'middle')
    text(svg,x,y+276,'相对 Marconi / StandaloneTopK：',12);text(svg,x,y+295,f'{fmt(r["improvement_ms"]["Marconi"])} / {fmt(r["improvement_ms"]["StandaloneTopK"])} ms',12)
    figdata.append(dict(case=label,pending_order=r['pending_order'],source_file=r['source_file'],source_line=r['source_line'],costs={name:[p['cost_ms'] for p in r['policies'][name]['per_pending']] for name in COLORS}))
finish(svg,'figure_cost.svg');dump('figure_cost_data.json',figdata)
print('报告、七个案例和两张 SVG 已写入独立分析目录。')
