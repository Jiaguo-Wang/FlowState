"""根据只读复核产物生成中文审计报告，不运行策略。"""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(n):return json.loads((OUT/n).read_text())
def put(n,s):
    p=OUT/n
    with p.open('x') as f:f.write(s)
def short(cid):return cid.split('_',2)[-1]
def main():
    ss=read('structure_statistics.json');cc=read('condition_review.json');integ=read('integrity_review.json')
    lines=['# 工作负载结构与判别力审计','\n仅g001–g004四个受控已知输入回放快照；16个相关预算条件。全部数值来自冻结B4-C results.json，不重新计算Phi、不调用策略。容差沿用evaluate.py的EPS=1e-9 modeled ms。K_min独立穷举仅检查前沿集合覆盖，不运行新的Exact收益求解。','\n## 结构总表','\n|快照|候选|Pending|准入/正收益边|跨Workflow正收益边|共享候选数|有重叠Pending数|K_min|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in ss:lines.append(f"|{s['group']}|{s['candidate_count']}|4|{s['admissible']}/{s['positive_edges']}|{s['cross_workflow_positive_edges']}|{len(s['shared_candidates'])}|{sum(len(v)>1 for v in s['positive_candidates_by_pending'].values())}|{s['K_min']}|")
    lines+=['\n合计56候选、16 Pending、224配对：68准入且正收益、156拒绝、0未知；12条跨Workflow正收益边。4/4快照均有共享与非平凡重叠，12/16条件（每组K1–3）同时具有共享、重叠和紧逻辑预算。K4均饱和，但不保证Greedy找到完整前沿。这里没有GPU容量压力证明。','\n## 逐候选支持与替代关系']
    for s in ss:
        lines+=['\n### '+s['group'],'\n主层A，附加标签A/B/C；不是三个独立分层样本。唯一多请求候选为B_TURN_001，位置2432，支持四个Pending；其余候选均仅支持一个Pending。','\n|候选（省略组前缀）|正收益Pending数|支持Pending|','|---|---:|---|']
        for c,v in s['positive_support'].items():lines.append(f"|{short(c)}|{len(v)}|{', '.join(short(p) for p in v)}|")
        lines+=['\n|Pending|正收益候选数|全候选最大前沿|唯一最大前沿候选|','|---|---:|---:|---|']
        for p,v in s['positive_candidates_by_pending'].items():lines.append(f"|{short(p)}|{len(v)}|{s['maximum_frontiers'][p]}|{','.join(short(c) for c in s['maximum_frontier_candidates'][p])}|")
        lines.append(f"\n冻结收益向量逐维比较确认{s['group']}有{len(s['domination_pairs'])}对严格支配关系，完整方向及ID见structure_statistics.json/domination_pairs。每条专属链的最深候选支配该链较浅项，但不逐维支配覆盖其他请求的共享B1。不能把共享和同请求替代混为一谈。")
    lines+=['\n## 全部预算的共同目标复核','\n单位：modeled ms。完整最终ID、Exact差距见condition_policy_review.csv（80行）；原始7策略全部结果仍位于B4-C。不同选择顺序不等于不同集合。','\n|条件|FlowState|PCB|WFC|LDR|Exact|达到Exact的PCB/WFC|','|---|---:|---:|---:|---:|---:|---|']
    for c in cc:lines.append('|'+c['group']+'/K'+str(c['K'])+'|'+'|'.join(f"{c['costs'][p]:.9f}" for p in ['FlowState','PCB','WFC','LDR','Exact OPT'])+'|'+','.join(p for p in ['PCB','WFC'] if p in c['optimal_policies'])+'|')
    lines+=['\nFlowState 12/16最优；PCB 8/16；WFC 8/16；两者最优条件互补，联合覆盖16/16。这个事后事实不是可部署的Best-of策略，不知道最优解时不能按结果选择基线。所有7个冻结策略之间，同目标不同最终集合的配对数为0（忽略顺序）；Exact代表集合与策略顺序不同不构成胜负。','\n全部Greedy非最优条件：']
    for c in cc:
        if c['gaps']['FlowState']>1e-9:lines.append(f"- {c['group']}/K{c['K']}：落后Exact {c['gaps']['FlowState']:.9f} modeled ms；集合{c['selected']['FlowState']}。")
    lines+=['\n## 为什么简单策略能达到Exact','\n**冻结结构已证实，以下化简为条件性数学推论。** 每组只有一个覆盖四请求的共享B1，其他候选按请求可分，最深专属项在该请求收益上支配浅层项。任意可行解中，同一专属链的多个项可由最深项替代而不降低F。因此最优解只需考虑“无B1的至多K个最深专属项”或“有B1及至多K−1个最深专属项”两类。固定是否含B1后，剩余收益按请求可分，选最大单请求剩余收益即可。这里是解释结构，不实现或运行新求解器。','\n四个请求各自最大前沿均由不同唯一深层项达到，所以任何达到全候选前沿的集合都必须含这四项：K_min≥4；四项集合构成见证给出K_min≤4。本批全候选前沿恰为T，J(C)=0。这并非任意工作负载都零残差。','\nPCB按每请求最佳项推荐并排序，故能够体现无共享项的解族；WFC在本批总会优先覆盖四请求的B1，再处理专属项。WFC的Token推进排序不一般等价于Phi边际排序：本批后三步在g001/002/004与Greedy一致是冻结观察，不是普适定理。两解族覆盖所有可行最优形态，并不证明PCB/WFC各自一定得到该族最优。','\ng001/K1–3、g002/K1–2、g004/K1–3由共享优先族获胜，WFC达到Exact；g003/K1–4、g002/K3–4、g001/K4、g004/K4由PCB达到Exact。g002/K3是紧预算下Greedy也会失败的直接例子：早选B1且无交换，最终不如A4/C4/D4，差49.016811025。K4时三组Greedy保留B1挤占第四个深层项。','\ng003/K1：WFC选择B1覆盖4请求、推进9728 Token；FlowState选择A5只覆盖1请求但推进10816 Token，冻结收益分别365.672296502与420.732039636。故覆盖数量优先本身即可解释分歧，不是位置项独立消融证据。该组PCB与Greedy一致并达到Exact。','\n## 判别力与主张边界','\nFlowState对PCB：8胜4平4负；对WFC：4胜12平0负；对StandaloneTopK：5胜11平。所有严格差条件保留在condition_review.json；FlowState没有任何条件同时严格优于PCB和WFC。四个组有可解释的局部判别力，不能说完全无差别；但不支持“全局集合分配普遍优于强简单策略”或独特必要性。','\n当前结构缺少多个有不同非单例支持域、相互嵌套的共享候选，以及在紧预算下这些共享层与专属深层真正竞争的多层实例；也缺少能隔离Phi位置项与简单Token排序差异的预注册条件。这是尚未覆盖的检验维度，不保证FlowState获胜。合法Token树通常是嵌套/不交覆盖，不应为复杂性构造不合法交叉矩阵。','\n没有新增性能实验、策略组合或结构搜索；没有将4快照的16预算视为16独立工作流。已有目录中的audit_workload_structure.py及workload_structure_metrics.json保持原样，本轮直接消费冻结收益并独立交叉核对主要计数。']
    put('workload_structure_audit.md','\n'.join(lines)+'\n')
    idx=['# 冻结证据索引','\n仓库：`/home/wjg/code/FlowState`。本轮仅CPU文件解析，未运行策略、GPU、Docker或Replay。','\n## 权威来源','\n|来源|作用/身份|','|---|---|',
    '|evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/manifest.json、batch_summary.json、execution_lock.json|根清单及正式入选g001–g004；g005/006不纳入比较|',
    '|同根g001–g004/snapshot.json、summary.json、manifest.json|实际Token、准入矩阵、物理驻留、来源组及同边界；各snapshot摘要与summary一致|',
    '|evaluation/strict_compatibility_b4c/results.json|权威已冻结收益、选择集合、J、Exact所有最优集合、各步trace；本轮不导入evaluate模块|',
    '|同目录evidence_manifest.json、input_integrity.json、evaluate.py|冻结结果完整性；EPS=1e−9；历史生产输入版本|',
    '|evaluation/sota_baseline_audit_b5a/source_lock.json、final_report.md、adaptation_contract.md|仅作为已有源码审计；输入充分性主张本轮独立核对|',
    '|B5-A raw/marconi_repo|HEAD=08016617b1524e6bf6ac29b680641cc945bda7f0，git工作树无差异|',
    '|B5-A raw/kvflow_repo|HEAD=7ef897eaedc16626a281ccfe0d453fae80251e8f，git工作树无差异；论文版对应UNVERIFIED|',
    '|evaluation/strict_compatibility_b4b0/capture.py:52–88；evaluation/rq3_openhands_neutral_collector.py:480–510|实际采集与受控recency来源|',
    '|evaluation/sota_metadata.py:64–77、107起|步骤固定1和同Workflow增量长度代理，不是原生SOTA输入|',
    '|/home/wjg/models/qwen3.5-9b/config.json|真实模型层数/维度；SHA需与各snapshot配置一致|',
    '|evaluation/runtime_control_closure_r2a/final_report.md|当前optimizer与B4-C历史版本变化的来源：严格生产接口接线，已有16/16回归|',
    '|evaluation/runtime_control_closure_r2c1b2/final_report.md|只支持g001一条跨Workflow恢复，不推广到全部68边|',
    '\n## 已完成摘要核验']
    for c in integ['checks']:idx.append('- '+json.dumps(c,ensure_ascii=False))
    idx+=['\nB4-C历史输入摘要存在一项差异，必须显式保留：'+json.dumps(integ['historical_input_hash_mismatches'],ensure_ascii=False),
    '\n这是当前optimizer已在R2-A接入Strict Epoch的版本变化；冻结B4-C结果本身和四张snapshot摘要未变。故本轮不使用当前optimizer重跑或替代原始结果。所有结果与其历史执行合同关联，而非宣称当前源码与历史完全相同。',
    '\nB5-A source_lock记录commit但不是全面文件SHA清单；本轮核验本地commit和clean状态，关键输入即时SHA见source_hashes.json。SHA保证内容一致性，不是独立真实性认证。不下载或刷新仓库/论文。',
    '\n根执行锁作为历史环境身份保留；本轮不重新认证当前GPU、镜像或扫描全部模型权重。只核验本次用到的冻结数据和配置，不能声称完成动态环境复验。',
    '\n已有同名输出目录包含早先结构脚本与metrics。本轮未覆盖它们；新增audit.py只读冻结收益、补充完整支持统计/Native事件映射。两个版本主要结构计数一致，旧脚本未在本轮执行。',
    '\n复核入口：`/home/wjg/uv312/bin/python -B -m evaluation.sota_baseline_gate_b5b0.audit`。仅写本目录派生统计，不实现策略；K_min覆盖穷举由任务明确允许。']
    put('evidence_index.md','\n'.join(idx)+'\n')
if __name__=='__main__':main()
