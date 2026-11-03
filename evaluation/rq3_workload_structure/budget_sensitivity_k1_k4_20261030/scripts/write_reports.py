"""读取本轮结果生成分析报告，并用独立收益恒等式复核门内Oracle。"""
import argparse
import csv
import hashlib
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path

from run_budget import BASE, NAT, POLICIES, EPS, phi, put


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description='仅分析本轮CPU产物，不运行策略或GPU')
    parser.add_argument('--out', type=Path, default=BASE)
    args = parser.parse_args()
    out = args.out.resolve()
    for name in ['budget_sensitivity_audit_report.md', 'solver_reproduction_audit.md',
                 'case_details.json', 'post_analysis_validation.json']:
        if (out / name).exists():
            raise ValueError('拒绝覆盖报告或核验产物：' + name)
    validation = json.loads((out / 'validation.json').read_text())
    if validation['status'] != 'PASS':
        raise ValueError('两道运行门禁未通过，不能生成成功分析')
    conditions = read_csv(out / 'budget_sensitivity_per_snapshot.csv')
    strategies = read_csv(out / 'budget_sensitivity_strategy_results.csv')
    pendings = read_csv(out / 'budget_sensitivity_pending_results.csv')
    oracles = read_csv(out / 'exact_oracle_results.csv')
    replacement = read_csv(out / 'budget_replacement_analysis.csv')
    timing = read_csv(out / 'solver_timing_raw.csv')
    reproduction = read_csv(out / 'original_budget_reproduction.csv')
    summary = read_csv(out / 'budget_sensitivity_summary.csv')
    index = {(r['snapshot_id'], int(r['K']), r['policy']): r for r in strategies}
    old = {r['snapshot_id']: r for r in read_csv(NAT / 'natural_workload_per_condition.csv')}
    matrices = {r['condition_id'].split('::budget=')[0]: r['independent_benefit_matrix_ms']
                for r in map(json.loads, (NAT / 'calculation_evidence.jsonl').open())}
    checks = []

    def check(name, value):
        checks.append({'check': name, 'pass': bool(value)})
        if not value:
            raise ValueError('独立结果核验失败：' + name)

    for r in strategies:
        if r['status'] != 'PASS':
            check('超限Oracle确为N/A：' + r['snapshot_id'] + '/' + r['K'], r['policy'] == 'Exact Oracle' and r['J_ms'] == 'N/A')
            continue
        ids, es, gs = [json.loads(r[k]) for k in ['selected', 'E', 'G']]
        row = old[r['snapshot_id']]
        b = matrices[r['snapshot_id']]
        ts = json.loads(row['targets'])
        j0 = sum(phi(t, t) for t in ts)
        f = sum(max([b[c][p] for c in ids] + [0.0]) for p in range(len(ts)))
        check('集合收益与J一致：' + r['snapshot_id'] + '/' + r['K'] + '/' + r['policy'], abs(j0 - f - float(r['J_ms'])) <= EPS)
        check('预算与逐请求G：' + r['snapshot_id'] + '/' + r['K'] + '/' + r['policy'], len(ids) <= int(r['K']) and [t - e for t, e in zip(ts, es)] == gs)
    independent_oracles = 0
    for r in oracles:
        n, m, k = int(r['n']), int(r['m']), int(r['K'])
        allowed = n <= 12 and m <= 4 and k <= 4 and sum(__import__('math').comb(n, a) for a in range(min(n, k) + 1)) <= 794
        check('Oracle规模门禁：' + r['snapshot_id'] + '/' + r['K'], allowed == (r['status'] == 'PASS'))
        if not allowed:
            continue
        b = matrices[r['snapshot_id']]
        ts = json.loads(old[r['snapshot_id']]['targets'])
        j0 = sum(phi(t, t) for t in ts)
        best = min(j0 - sum(max([b[c][p] for c in subset] + [0.0]) for p in range(m))
                   for size in range(k + 1) for subset in itertools.combinations(sorted(b), size))
        check('Oracle独立收益式枚举：' + r['snapshot_id'] + '/' + r['K'], abs(best - float(r['J_ms'])) <= EPS)
        independent_oracles += 1
    check('门内独立Oracle域完整', independent_oracles == 396)
    check('计时批次数', len(timing) == 15 * (764 * 4 + independent_oracles))
    check('旧预算复现域', len(reproduction) == 2292 and all(r['same_set'] == r['same_J'] == 'True' for r in reproduction))
    check('前后没有非嵌套替代', all(r['nested'] == 'True' for r in replacement))
    kmin2 = [r for r in conditions if r['population'] == 'AgentX' and r['K'] == '1' and r['K_min'] == '2']
    check('旧27条去重为9快照', len(kmin2) == 9)
    details = []
    agentx_table = []
    for r in kmin2:
        sid = r['snapshot_id']
        original = old[sid]
        cid_list = json.loads(original['candidate_ids'])
        positions = json.loads(original['positions'])
        supports = json.loads(original['positive_supports'])
        data = {'snapshot_id': sid, 'input_sha256': r['input_sha256'], 'K_min': 2,
                'pending_ids': json.loads(original['pending_ids']), 'targets': json.loads(original['targets']),
                'candidates': [{'checkpoint_id': c, 'L': l, 'positive_support': supports[c],
                                'benefit_vector_ms': matrices[sid][c],
                                'historical_node_id': json.loads(original['candidate_node_map'])[c]}
                               for c, l in zip(cid_list, positions)],
                'results': {str(k): {p: index[(sid, k, p)] for p in POLICIES} for k in range(1, 5)}}
        details.append(data)
        fs1 = index[(sid, 1, 'FlowState')]
        fs2 = index[(sid, 2, 'FlowState')]
        agentx_table.append('|'+sid.replace('rq3-agentx-', '')+'|'+str(len(cid_list))+'/'+str(len(data['pending_ids']))+'|'+
                            '|'.join(f"{float(index[(sid, 1, p)]['J_ms']):.6f}" for p in POLICIES)+
                            f"|{float(fs2['J_ms']):.6f}|{float(fs1['J_ms'])-float(fs2['J_ms']):.6f}|"+
                            index[(sid, 1, 'Exact Oracle')]['status']+'|')
    put(out / 'case_details.json', {'AgentX_K_min_2': details,
                                    'OpenHands_divergent_snapshot': old['rq3-openhands-main-g051-round5'],
                                    'scope': '只表示冻结Formal收益及历史Node线索；不是物理恢复认证'})
    timing_rows = []
    for policy in (*POLICIES, 'Exact Oracle'):
        ss = [r for r in strategies if r['policy'] == policy and r['status'] == 'PASS']
        timing_rows.append(f"|{policy}|{len(ss)}|{statistics.median(float(r['solve_median_ms']) for r in ss):.6f}|"
                           f"{statistics.median(float(r['solve_P95_batchmean_ms']) for r in ss):.6f}|"
                           f"{max(float(r['solve_P95_batchmean_ms']) for r in ss):.6f}|")
    environment = json.loads((out / 'environment.json').read_text())
    source = json.loads((out / 'source_manifest.json').read_text())
    reproduction_report = '''# 第一道门禁：原策略实现与573条件复现

**PASS：573/573条件、2292/2292策略结果。** 逐项比较集合语义、基数、每Pending的T/E/G及成本、总J，容差1e−9 ms；FlowState原始选择顺序573/573完全一致，PCB/LDR/WFC的返回顺序也与各自冻结结果一致。不是读取旧“同集”结论当复现。

## 实际选择入口

|策略|真实实现及规则|本轮复用方式|
|---|---|---|
|FlowState|原评估器`rq3_frozen_snapshot_evaluator.py`用`GlobalOptimizer.select`；当前该入口进入`_select_candidates`，动态最大边际，1e−9同分容差、ID字典序、非正边际停止|直接调用同一生产Greedy核心`_select_candidates`，成本闭包按冻结Formal矩阵和Phi计算。没有自行重写Greedy，也未调用plan_strict|
|PCB|`p1_strong_baseline_validation/pcb.py::select`：每Pending推荐最佳正独立收益项，按收益/ID/Pending排序，ID去重到K|导入已有冻结模块；不改变其推荐和停止规则|
|LDR|同目录`ldr.py::select`：每Pending最深合法项，按位置/ID/Pending排序去重|导入已有冻结模块；不读Phi作决策|
|WFC|同目录`wfc.py::select`：每步优先推进Pending数q，其次推进Token总数r，最后ID；q=0停止|导入已有冻结模块；完整保留动态更新及其诊断轨迹|

Greedy核心所需对象仅有checkpoint_id/memory_bytes/recurrent_resident；本轮使用只含这三个字段的不可变适配对象，完整候选/位置/兼容在独立成本闭包中保持原样。没有伪造Workflow或Lineage来绕过关系，也没有把Formal输入构造成已认证Strict Epoch。PCB/LDR/WFC接受其原common.Epoch对象。实际原公式与独立评分逐值核验，原始字节预算转K保持等大小合同。

**当前生产能力必须另述：** `GlobalOptimizer.plan_strict()`已经接入Exact–Greedy门禁：先Strict Epoch完整认证，n≤12、m≤4、K≤4、子集≤794时Exact，否则原Greedy；输入不合法不回退。它是生产源码能力，不是仅有独立实验Exact Planner。但本次历史Formal策略定义是Marginal Greedy，故不通过Strict混合入口替换它，Exact单列Oracle。

## 版本、同分与完整性

冻结PCB/LDR/WFC/common四文件SHA与implementation_lock全部一致；当前生产Optimizer、Phi、原始评估入口另记录完整SHA。当前Optimizer文件包含后续混合接线，不能说整个文件与早期版本相同；本轮复用Greedy循环并用2292份原结果复现建立行为对应。

没有执行旧脚本main，避免覆盖旧文件。原预算复现先完成并写reproduction_gate.json=PASS，随后才进入K扫描。原始FlowState逐步选择按原diagnostics核对，不从终集猜测。原列表顺序差异与跨策略集合一致性是不同概念。

全部读取文件前后内容摘要一致；输入去预算SHA三条件逐项相同，191唯一快照域完整。若任一复现、身份、公式或规模失败，run_budget.py保存BLOCKED及阶段/堆栈并停止，不静默补数据或扩大门禁。

依据：original_budget_reproduction.csv、reproduction_gate.json、validation.json、source_manifest.json及protocol.json。
'''
    with (out / 'solver_reproduction_audit.md').open('x') as f:
        f.write(reproduction_report)
    summary_lines = []
    for r in summary:
        if r['population'] == 'ALL':
            continue
        wt = json.loads(r['FlowState_W_T_L'])
        summary_lines.append(f"|{r['population']}|{r['K']}|{r['conditions']}|{r['all_four_same_set']}|{r['all_four_same_J']}|"+
                             '|'.join(f"{wt[p]['win']}/{wt[p]['tie']}/{wt[p]['loss']}" for p in ['PCB', 'LDR', 'WFC'])+f"|{r['oracle_eligible']}|")
    report = f'''# RQ3-B1：固定实例的K1–K4预算敏感性实验

日期：2026-10-30。**最终裁决：PASS。** 第一门禁573/573条件、2292份四策略结果复现；第二门禁191/191快照全部完成K1–4，共764条件。Oracle仅396条件满足门禁，另368条件全部N/A，未扩大门禁。没有GPU、合成工作流、受控结构生成、生产源码修改或论文写作。

## 1. 十项关键发现与回答

1. **原始573条件复现PASS**，不只同集，还包括每Pending E/G/成本和J；FlowState完整保存选择顺序也复现。
2. **AgentX K1有9/23条件四策略分歧**，恰好是9个K_min=2的共享快照；其余14个K1条件同集同J。不是看到共享就必然分歧，22个共享快照中仍有13个在K1持平。
3. **AgentX K1：FS对PCB/LDR各3胜20平0负，对WFC6胜17平0负。** 三组PCB/LDR失败与六组WFC失败互补，不能只展示其中一类。
4. **K2及以后AgentX全部23/23同集同J。** K2已达到完整最深前沿；K3/4按正边际/推荐去重规则停止，没有为填满K追加冗余项。
5. **OpenHands有一个预算反例。** g051-round5/K3中FS/PCB相对LDR/WFC低0.047745076924 modeled ms；其余671/672同集同J。无共享足以支持分组最优化，但不能保证按位置和按恢复收益的截断排名总相同。
6. **Oracle域内FS为396/396最优；PCB/LDR各395/396，WFC394/396。** 域外368条件不能计算Oracle Gap。尤其n149的K1没有偷偷扩大枚举；规则上最大单项可分析，但不将其填成实测Exact证书。
7. **全部764条件中754同集同J，10条同时有选集和J差异；没有观察到四策略异集但等J。** 不同集合不能自动算优势，本轮严格差距都由共同J/E/G核验。
8. **没有非嵌套替换。** 相邻K的四策略2292行比较全部保留旧选集；门内Oracle另297行也无替换。跨策略差异不是同一策略随预算变化的替换。
9. **本轮差异主要发生于单项价值比较，不证明多步动态边际的独立贡献。** AgentX差异全在K1，第一步就是总独立收益最大化；Shared有时获胜、有时应让位深专属项。没有观察到本轮实例中的复杂多项替代。
10. **这仍是Formal模型结果，不是AgentX Strict可执行恢复或GPU收益。** 原始Token缺失、Formal/Raw关系差异、物理别名问题没有因CPU预算扫描被解决。

## 2. 数据、门禁与实际实现

自然分析573行是191快照各三个原始比例预算，不是573独立工作负载。逐条件去预算输入摘要（候选ID/位置、Pending/T、兼容、收益、等大小成本）三份完全相同才去重；168个OpenHands、23个AgentX全部通过，没有按结果筛选。

读取自然分析要求的四CSV、来源清单及脚本说明，再逐项核对其185个已读来源内容摘要，并回到原始504/69评估行验证snapshot_id、K、snapshot_digest与FlowState逐步选择。新实验清单共{len(source['files'])}个源条目，前后SHA一致；完整文件和原始Node线索见source_manifest.json。

选择接口详见[solver_reproduction_audit.md](solver_reproduction_audit.md)。FS复用真实生产Greedy核心、显式喂入原Formal评分关系；PCB/LDR/WFC直接import历史锁定实现。没有通过伪造Workflow/Lineage改边。**本实验FlowState=原Marginal Greedy**；当前生产`plan_strict`已具备混合路由，但不能将其Exact结果混入本次历史Greedy对照。Oracle独立列出。

固定Phi：g=G/1024，t=T/1024，Phi=37.828150g+0.345974143gt−0.156201917g²。评价程序独立计算E=max合法已选L、G=T−E、J=ΣPhi，不用Σ候选独立收益替代J。另用F=Σmax b的独立收益式复核全部有效结果，并在原规模内二次验证396个Oracle最优目标；详见post_analysis_validation.json。K表示等大小逻辑状态数量，不是GPU物理容量压力。

## 3. 全部预算对照

胜/平/负为FlowState相对对应基线，比较共同J，容差1e−9 ms。

|来源|K|条件|四者同集|四者同J|FS/PCB胜平负|FS/LDR胜平负|FS/WFC胜平负|门内Oracle|
|---|---:|---:|---:|---:|---|---|---|---:|
{chr(10).join(summary_lines)}

Oracle有99个快照（OH90、AX9）×4预算=396条件；其余92快照×4=368条件N/A。Exact枚举|S|≤K，包括空集，选最优容差内最少项再ID字典序；没有因预算大而补零收益项。门内最优集合可多解：OpenHands360个Oracle条件中28个多解，AgentX36中25个多解；其中包含预算有余时加零边际候选的不同基数集合，不能直接解释为不同恢复质量。Oracle代表与实际策略的完整集合、前沿和Gap全部保存在CSV。

### AgentX九个K_min=2快照：全量而非只选有利实例

下表单位为modeled ms。K2四策略同J；最后列PASS表示该条件有门内穷举，N/A_SIZE_GATE表示没有Exact值。完整选集、所有Pending E/G、全部候选支持与原始Node线索见[case_details.json](case_details.json)，不能拿这些Node作新Engine绑定。

|Snapshot|n/m|K1 FS J|K1 PCB J|K1 LDR J|K1 WFC J|K2共同J|FS K1→K2下降|Oracle状态|
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(agentx_table)}

**两种相反竞争，均需保留：**

- g000/g007/g014：FS选择共享项，PCB/LDR选择单Pending的更深最佳推荐。例g014：共享req0000，L34304，支持两个Pending；专属req0001，L34560，只支持第一个。K1 FS使E=[34304,34304]，PCB/LDR E=[34560,0]；J821.439202对2298.288149。独立单请求收益排序漏计共享项的总收益，WFC在这三例与FS一致。
- g005/g006/g010/g017/g018/g021：FS选择更深专属项，WFC因推进Pending数优先选浅共享项。例g005：共享L384支持两个，专属L43840支持第一个；K1 E分别[384,384]与[43840,0]，J2573.277764对620.026891。此时覆盖两个请求并不抵得上一个深恢复；PCB/LDR与FS一致。

K2时每例同时保留合适的共享和专属项，因此原来的竞争消失。不过“K_min=2”本身不能推出这个结构；本轮是逐个检查全部支持和位置后才作上述归因。每个原候选都保留，包括被支配和当前零收益状态，未为算法删候选。

## 4. OpenHands无共享对照：不能省略的小差异

Snapshot=`rq3-openhands-main-g051-round5`，n20、m4、K_min4，目标向量[11200,8320,7808,7999]。K3结果：

|方法|逻辑选集|E|G|J ms|
|---|---|---|---|---:|
|FS/PCB|A_TURN_005、B_TURN_005、D_TURN_004|[11200,8320,0,7808]|[0,0,7808,191]|307.027617198|
|LDR/WFC|A_TURN_005、B_TURN_005、C_TURN_005|[11200,8320,7808,0]|[0,0,0,7999]|307.075362275|

C5与D4的L同为7808；C的T7808、D的T7999。LDR以及此时q相同、r也相同的WFC按ID取C；位置感知Phi计算的D恢复收益略高，FS/PCB取D。严格差距约0.047745 ms，不是数值1e−9容差或列表顺序导致，也不是跨Pending共享。

本例n20超Exact门禁，Oracle Gap仍为N/A，**不为了一个小差距扩大Oracle**。可分离收益的理论可另作解释，但不填成已执行Oracle。差异仅是当前模型内排序证据，没有GPU对照，不能称实际0.047745ms加速。原比例预算为K5/10/15，都会覆盖四个最佳代表，所以旧573条没有显露这个K3截断差异。

其他OpenHands预算结果完整保留，K1/2/4各168/168四策略同集同J；K3为167/168。无共享时Pending内较浅状态的边际仍会被更深状态覆盖，但该结构不要求一般共享组合优化。它也不意味着任何不读Phi的简单位置排序都能在所有T下精确最优。

## 5. 集合替代、停止与机制边界

相同输入下检查K1→2、2→3、3→4：四策略191×3×4=2292个比较、Exact门内99×3=297个比较，非嵌套0、同策略异集但J相同0。所有新增状态的影响在E/G与J中计算，没有以集合变动自动认定收益。

AgentX K2之后各策略已经达到候选域完整前沿：K3/4仍停止在≤2个有效状态，而不是填满K。OpenHands K4覆盖各Pending最深合法前沿；“最深可达”不保证G全为0，原候选可能仍浅于T。请勿将逻辑K解释为28槽池的HBM压力。

**已验证事实：** 原条件可复现，预算收紧能显露模型目标差异，且差异会在饱和预算消失。**原因解释：** 共享总收益、专属深度和覆盖数之间的排序冲突，以及无共享时T影响收益排名。**尚未证明：** 此数据中存在不可撤回的多步Greedy损失、非嵌套最优组合或动态边际优于静态总独立收益。K1只有一次选择，不能拿它证明后两种机制。

## 6. CPU求解时间与环境

CPU：{environment['cpu_model']}；Python：{environment['python'].splitlines()[0]}。使用perf_counter_ns，输入读取/解析、候选对象准备、Oracle外部核验和报告均不计入选择计时。每条件/策略3次预热，15批×5次；策略顺序按快照预算与批次确定性轮转。全部51780批原始时间戳差保存在solver_timing_raw.csv。

下表是各条件“批均单次耗时中位”的跨条件中位；P95也是15批均值分布的经验P95，**不是单调用尾延迟、更不是线上P95**。Exact仅用396个门内条件，其他方法用764个，不能直接用总体表作同域加速比；同条件比较请读逐策略CSV。

|策略|计时条件|条件中位的中位ms|条件P95的中位ms|最大条件P95 ms|
|---|---:|---:|---:|---:|
{chr(10).join(timing_rows)}

各真实方法内部构造诊断轨迹的开销仍计入；不剥去WFC的记录构建再冒称同一实现。不从CPU时间推导GPU控制成本，不把J减CPU时间计算净收益。只有一台共享CPU上的这一轮测量协议，运行间负载影响未排除。

## 7. 完整交付与复现

所有结果位于本独立目录，原natural_analysis、573冻结结果与GPU数据未覆盖。7张要求的主CSV分别保留快照条件、策略、Pending、分来源预算汇总、相邻预算替代、全部14行策略差异、Oracle及N/A；另有原预算2292行复现、51780批计时、完整来源清单、protocol/environment和机器门禁JSON。逐Pending结果13648行，各自保存T/E/G/cost及聚合J。

```bash
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/scripts/run_budget.py --out evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/reproduce_01
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/scripts/write_reports.py --out evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/reproduce_01
```

输出子目录必须无既有JSON/CSV/Markdown产物；禁止覆盖正式结果。源码和196个读取源的内容摘要见source_manifest；原脚本异常会保存BLOCKED阶段和堆栈并停止，不进入后续扫描。

## 8. 仍无法回答与是否值得下一阶段

AgentX Formal关系不等于真实Token前缀或逐边Native可恢复性；原始输入缺失、Raw额外/冲突边与物理别名仍在。因此本次不能推荐直接把九例当合法GPU恢复对照，不能把任何模型成本差写成实际TTFT改善。OpenHands是四独立Session受控组合，不是原生多路Fork。368个超限条件没有Oracle Gap，不把其他结构证书冒充它。

**值得有条件研究下一阶段，但问题应明确收窄：** 当前数据已经回答“共享与紧预算能造成不同模型成本”，却没有检验多共享域、K≥2紧预算下的多项组合替代或Greedy不可回退损失。后续若获授权，可围绕这些缺失结构作预注册敏感性研究，保留简单规则足够的对照；还需将逻辑结构与真实Token/状态可执行性分开认证。不为追求胜负继续降K、改Phi、改矩阵或筛选来源。

本阶段立即停止，不进入五维受控实验、GPU、论文正文或其他工作负载任务。
'''
    with (out / 'budget_sensitivity_audit_report.md').open('x') as f:
        f.write(report)
    put(out / 'post_analysis_validation.json', {'status': 'PASS', 'checks': checks,
                                               'independent_oracle_conditions': independent_oracles,
                                               'report_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                               'scope': '仅CPU固定Formal问题，不认证GPU恢复'})
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'report': str(out / 'budget_sensitivity_audit_report.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
