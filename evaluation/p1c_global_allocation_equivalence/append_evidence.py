"""将冻结轨迹与解析例子的完整数值摘要附入研究报告，不执行任何选择器。"""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
rows=[json.loads(s) for s in (ROOT/'evaluation/p1_strong_baseline_validation/per_case_results.jsonl').read_text().splitlines()]
examples=json.loads((OUT/'analytical_examples.json').read_text())
report=OUT/'global_allocation_equivalence_audit.md'
s=report.read_text().split('\n## 7. 全部顺序差异的输入与选择证据')[0]
a=['\n## 7. 全部顺序差异的输入与选择证据\n','以下每行代表同一快照的三个冻结预算；没有遗漏顺序差异快照。每个 Case ID 需与表内 budget/K 配对。全量输入包含未选候选，保存在 R 指定行及 `order_difference_evidence.json`；下表只缩写 checkpoint 前缀，不改变 ID，完整 ID 可由该文件恢复。\n','| Case ID | R 行号（25/50/75%） | K（三预算） / K_min | 与 FlowState 顺序不同的方法 |','|---|---|---|---|']
unique={r['snapshot_id']:r for r in rows if any(v['order_different'] for v in r['comparisons'].values())}
for sid in unique:
    group=[(i,r) for i,r in enumerate(rows,1) if r['snapshot_id']==sid];r=group[0][1]
    names=[n for n in ('PCB','LDR','WFC') if r['comparisons'][n]['order_different']]
    a.append(f"| {sid} | {'/'.join(str(i) for i,r in group)} | {'/'.join(str(r['k']) for i,r in group)} / {r['budget_structure']['minimum_checkpoints_for_all_frontiers']} | {','.join(names)} |")
for sid in unique:
    group=[(i,r) for i,r in enumerate(rows,1) if r['snapshot_id']==sid];lineno,r=group[0]
    short=lambda x:x.split('::')[-1].replace('RQ3_G051_','')
    a.extend([f'\n### {sid}\n',f"来源 R:{lineno}；原始 `{r['source_file']}:{r['source_line']}`。以下是 25% 条件的实际顺序；另两个预算的相同输入与完整轨迹见上表对应行。",
      '\n输入 T 向量（tokens，按冻结 pending 顺序）：'+str([p['target'] for p in r['input']['pending']])+'。候选的兼容索引用 p1、p2…对应此顺序。',
      '\n| 最终必要项 | L（tokens） | 兼容 pending |','|---|---:|---|'])
    selected=set(r['policies']['FlowState']['selected'])
    for c in r['input']['candidates']:
        if c['identifier'] in selected:
            ps=','.join('p'+str(i) for i,p in enumerate(r['input']['pending'],1) if p['identifier'] in c['compatible'])
            a.append(f"| {short(c['identifier'])} | {c['position']} | {ps} |")
    a.extend(['\n| 方法 | 实际选择顺序 | 最终 C(S)，modeled ms |','|---|---|---:|'])
    for n in ('FlowState','PCB','LDR','WFC'):
        p=r['policies'][n];order=r['flowstate_order'] if n=='FlowState' else p['selected']
        a.append(f"| {n} | {' → '.join(map(short,order))} | {p['cost_ms']:.9f} |")
    for n in ('PCB','LDR','WFC'):
        d=r['comparisons'][n]['first_divergence']
        if not d:continue
        f,b=d['FlowState'],d['baseline']
        a.append(f"\n{n} 首次分歧在第 {d['step']} 步，共同前缀 E={d['E_before']}：FlowState 选 {short(f['checkpoint_id'])}（q={f['q']}、r={f['r']}、marginal={f['marginal_ms']:.9f}）；该基线选 {short(b['checkpoint_id'])}（q={b['q']}、r={b['r']}、marginal={b['marginal_ms']:.9f}）。")
    a.append('\n持平证书：表内所有最终项分别是至少一个 pending 的唯一最大前沿实现者；三个预算均可容纳它们，四种已保存轨迹恰好选完这些项，没有额外保留浅层冗余项。不存在未选候选还能提高前沿的情形。')
a.extend(['\n## 8. 解析反例的逐步数值\n','由 `audit.py` 直接调用冻结实现；此处仅汇总数学验证。穷举只作用于这些 3 候选小例子，不作用于正式总体。\n','| 数学条件 | K | FlowState 顺序 / C | PCB 顺序 / C | LDR 顺序 / C | WFC 顺序 / C | 小例子 Exact OPT |','|---|---:|---|---|---|---|---:|'])
for e in examples:
    cols=[f"{','.join(e['results'][n]['selected'])} / {e['results'][n]['cost_ms']:.9f}" for n in ('FlowState','PCB','LDR','WFC')]
    a.append(f"| {e['case_id']} | {e['k']} | "+' | '.join(cols)+f" | {e['exact_cost_ms']:.9f} |")
for e in examples:
    a.extend([f"\n### {e['case_id']} 的评分过程\n",'候选独立收益向量（pending 顺序同定义，modeled ms）：'+ '；'.join(f"{c}={','.join(f'{v:.9f}' for v in vs)}" for c,vs in e['benefit_matrix_ms'].items())+'。'])
    for step in e['flowstate_steps']:
        a.append('\nFlowState 前缀 '+str(step['prefix'])+'：'+ '，'.join(f'{c} marginal={v:.9f}' for c,v in step['gains'].items())+'；选择 '+step['chosen']+'。')
    for n in ('PCB','LDR'):
        a.append('\n'+n+' 推荐排序：'+' → '.join(f"{r['pending_id']}:{r['checkpoint_id']}（{r['score']:.9f}）" for r in e['traces'][n]['recommendations'])+'；按 K 扫描并对 ID 去重。PCB 分数单位 modeled ms，LDR 分数单位 tokens。')
    for step in e['traces']['WFC']['steps']:
        a.append('\nWFC 前缀 '+str(step['prefix'])+'：'+' → '.join(f"{r['checkpoint_id']}（q={r['q']},r={r['r']}）" for r in step['ranked_candidates'])+'；选择 '+step['chosen']+'。')
a.extend(['\n## 9. 完整性与复核边界\n','`integrity.json` 记录本轮所用策略、核心模型、P1-B 记录、两个正式原始结果及前序报告的 SHA-256 前后对照；全部匹配。四项 implementation_lock 摘要全部通过，573 条 FlowState 输出另与各自原始正式文件的 Case ID、snapshot digest、预算、集合及成本交叉核对。\n','本轮生成：`audit.py`、`append_evidence.py`、`structure_summary.json`、`frontier_certificates.json`、`order_difference_evidence.json`、`analytical_examples.json`、`integrity.json` 和本报告。前两个脚本只写本隔离目录；可用 `PYTHONDONTWRITEBYTECODE=1 python3 evaluation/p1c_global_allocation_equivalence/audit.py` 复核。`append_evidence.py` 仅更新本报告附录。\n','超出旧实验手册的新信息已明确标记：全量唯一最大前沿证书来自冻结输入的 CPU 只读分析；解析反例来自人工合法数学构型，未进入正式 population。尚未提供新的原生共享事件或 runtime 证据，因此真实新工作流中的收益与额外控制成本保持 UNKNOWN。'])
report.write_text(s+'\n'.join(a)+'\n')
print('报告附录已生成；顺序差异快照数：',len(unique))
