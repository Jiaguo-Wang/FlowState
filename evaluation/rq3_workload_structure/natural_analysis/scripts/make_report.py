"""核验独立计算产物，生成来源清单、补充排序检查与中文审计报告。"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze import BASE, ROOT, EPS, phi, write_csv


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def yes(row, key):
    return row[key] == 'True'


def main():
    parser = argparse.ArgumentParser(description='仅验证已有CPU统计并生成报告，不执行实验')
    parser.add_argument('--input-dir', type=Path, default=BASE)
    args = parser.parse_args()
    out = args.input_dir.resolve()
    if out != BASE and BASE not in out.parents:
        raise ValueError('报告必须写入本次独立目录')
    for name in ['natural_workload_audit_report.md', 'source_manifest.md', 'ordering_and_boundary_audit.csv', 'analysis_validation.json']:
        if (out / name).exists():
            raise ValueError('拒绝覆盖已有产物：' + name)
    rows = load_csv(out / 'natural_workload_per_condition.csv')
    agreement = load_csv(out / 'strategy_agreement_analysis.csv')
    replacement = load_csv(out / 'budget_replacement_analysis.csv')
    summary = load_csv(out / 'natural_workload_structure_summary.csv')
    manifest = json.loads((out / 'source_manifest.json').read_text())
    calculations = [json.loads(x) for x in (out / 'calculation_evidence.jsonl').open()]
    calc = {r['condition_id']: r for r in calculations}
    old_before = json.loads((ROOT / 'evaluation/p1_strong_baseline_validation/source_integrity_before.json').read_text())
    artifact = json.loads((ROOT / 'evaluation/p1_strong_baseline_validation/artifact_manifest.json').read_text())
    integrity = []
    for item in manifest['source_files']:
        expected = old_before.get(item['path'])
        if Path(item['path']).parent == ROOT / 'evaluation/p1_strong_baseline_validation':
            expected = artifact['files'].get(Path(item['path']).name, expected)
        status = 'MATCH' if expected == item['sha256_before'] else 'NO_PRIOR_LOCK' if expected is None else 'MISMATCH'
        integrity.append({'path': item['path'], 'current_sha256': item['sha256_before'],
                          'expected_prior_sha256': expected or 'UNKNOWN', 'status': status})
    if any(r['status'] == 'MISMATCH' for r in integrity):
        raise ValueError('存在与原冻结内容锁不一致的已读取来源')
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'pass': bool(condition)})
        if not condition:
            raise ValueError('独立统计核验失败：' + name)

    check('逐条件总数及身份唯一', len(rows) == 573 == len({r['condition_id'] for r in rows}))
    check('两来源条件数', Counter(r['population'] for r in rows) == {'OpenHands': 504, 'AgentX': 69})
    check('策略六对全覆盖', len(agreement) == 573 * 6)
    check('每快照三预算全量比较', len(replacement) == 191 * 3 * 4)
    check('跨预算输入一致', all(yes(r, 'same_input_except_budget') for r in replacement))
    check('比较预算严格递增', all(yes(r, 'strictly_increased_K') for r in replacement))
    check('原始输入摘要首尾一致', all(r['sha256_before'] == r['sha256_after'] for r in manifest['source_files']))
    check('原始来源联接完整', len(manifest['original_openhands_sessions']) == 672 and len(manifest['original_agentx_traces']) == 23)
    check('四策略最优性证书均已产生', all(r['formal_optimality_certificate'] != 'UNKNOWN' for r in rows))
    ordering = []
    for r in rows:
        matrix = calc[r['condition_id']]['independent_benefit_matrix_ms']
        ids, positions, pids = [json.loads(r[k]) for k in ['candidate_ids', 'positions', 'pending_ids']]
        pos = dict(zip(ids, positions))
        recommendations = []
        for p, pid in enumerate(pids):
            best = max((matrix[c][p] for c in ids), default=0.0)
            if best <= EPS:
                continue
            candidates = [c for c in ids if abs(matrix[c][p] - best) <= EPS]
            cid = min(candidates)
            recommendations.append((cid, pid, best, pos[cid]))
        benefit_order = sorted(recommendations, key=lambda v: (-v[2], v[0], v[1]))
        depth_order = sorted(recommendations, key=lambda v: (-v[3], v[0], v[1]))

        def truncate(items):
            selected = []
            for cid, _, _, _ in items:
                if cid not in selected and len(selected) < int(r['K']):
                    selected.append(cid)
            return selected

        gain, depth = truncate(benefit_order), truncate(depth_order)
        selected = json.loads(r['selection_by_policy'])
        check('独立收益推荐复核：' + r['condition_id'], set(gain) == set(selected['PCB']))
        check('独立深度推荐复核：' + r['condition_id'], set(depth) == set(selected['LDR']))
        ordering.append({'condition_id': r['condition_id'], 'population': r['population'],
                         'benefit_recommendation_order': [v[0] for v in benefit_order],
                         'depth_recommendation_order': [v[0] for v in depth_order],
                         'full_recommendation_order_equal': [v[0] for v in benefit_order] == [v[0] for v in depth_order],
                         'same_selected_at_observed_K': set(gain) == set(depth),
                         'all_frontiers_equal_targets': list(json.loads(r['all_candidate_frontiers']).values()) == json.loads(r['targets']),
                         'native_branch_status': 'UNKNOWN'})
    write_csv(out / 'ordering_and_boundary_audit.csv', ordering)
    write_csv(out / 'source_integrity_crosscheck.csv', integrity)

    def stats(pop):
        return {r['metric']: int(r['value']) for r in summary if r['population'] == pop}

    oh, ax, total = (stats(p) for p in ['OpenHands', 'AgentX', 'ALL'])
    unique = {r['snapshot_id']: r for r in rows}
    traces = manifest['original_agentx_traces']
    subagents = [t['raw_type_counts'].get('subagent', 0) for t in traces.values()]
    raw_call_count = sum(t['raw_type_counts'].get('calls_with_block_ids', 0) for t in traces.values())
    prior_counts = Counter(r['status'] for r in integrity)
    report = f'''# 自然来源工作负载结构独立审计

日期：2026-10-30。**裁决：573条件的Formal统计核验PASS；真实执行分支结构与可执行共享认证仍为PARTIAL。** 本报告不是论文正文，不运行GPU、不调用生产Solver、不生成新工作负载或预算扫描。全部统计从原始逐行输入、候选/目标/兼容关系及冻结策略结果重新计算，没有读取旧报告汇总作为数值输入。

## 1. 最重要的十项发现

1. **四策略最终同集、同E/G、同J：573/573；列表顺序却仅543/573相同。** OpenHands有3条、AgentX有27条次序不同。不能用列表顺序报告分配不同，也不能把同集扩大为过程完全相同。
2. **OpenHands的504条全部正收益可分离，均有候选重叠。** 没有候选正收益支持多个Pending；但每个Pending有多轮历史候选，504/504存在重叠和严格支配，且504/504记录到独立收益被已选集合压低。这不是“没有集合效应”。
3. **OpenHands并非全部预算充足。** 90/504的K<K_min=4，仍然四策略同集。可分离收益使此域可化简为每Pending最佳代表的Top-K；不是“预算大”一个因素解释全部平局。
4. **AgentX原预算确实饱和。** 66/69有正收益共享，来自22/23快照；69/69的K≥K_min，且已选集合达到完整候选域最深前沿。K_min分布为0/1/2，对应3/39/27条件；原预算按总候选数计算，许多候选对当前目标根本无正收益。
5. **共享支持与重叠是不同指标。** AgentX只有33/69有两个有效候选支持相同Pending；其余有共享也可能只有一个有效共享候选。全部共享候选的正收益支持大小都是2，不等于原轨迹只有两条分支。
6. **“重叠没有改变边际”被数据否定。** 两来源合计531/573记录到剩余候选边际小于独立收益；按本文稳健判据，509/573出现剩余候选独立收益与边际收益的严格序逆转。然而最终四策略依然相同，说明边际变化与最终预算截断差异不可混同。
7. **预算间未观察到替换。** 同输入的191快照×3预算对，共573个不重复快照—预算对；四策略共2292行，全部嵌套、0次移除旧候选。101/191快照三个预算下选集完全不变，其中AgentX23/23。结论只限原0.25/0.5/0.75预算，不覆盖更紧K。
8. **同集不等于最优集合唯一。** 容差下565/573具有唯一最小基数最优代表；其余8条OpenHands紧预算条件存在同基数等值替代。另387/573预算未填满，可加入零边际候选而保持J。因此共同停止及tie-breaking可固定代表，但不是大多数平局的根本原因。
9. **不能把这个矩阵的分离性直接称为自然工作流没有共享。** OpenHands按四个独立Session受控组合，旧Legacy谓词要求Workflow ID相同，直接限制跨Session兼容。AgentX Formal边也经过逻辑编码，不是Strict可执行边。两来源的真实分支数、公共Token历史及自然Ready关系不能由此恢复。
10. **原始复杂性并非全都缺失，但不能强行对齐到决策图。** 23条AgentX原始trace合计{sum(subagents)}个`type=subagent`容器、{raw_call_count}条带块身份调用，单trace容器数{min(subagents)}–{max(subagents)}；672个OpenHands来源保留完整消息。这些不等于某个Epoch的Fork/Ready/Token祖先图；旧物理记录还显示AgentX11/23快照有Node别名，Formal/Raw关系不一致，不能当真实恢复证据。

## 2. 数据来源、身份与完整性

独立清单见[source_manifest.md](source_manifest.md)，机器可读完整文件摘要与原始记录联接见[source_manifest.json](source_manifest.json)。本轮实际读取185个来源文件，均记录前后SHA-256且未变化；与历史冻结锁交叉核对：{prior_counts.get('MATCH', 0)}个MATCH、{prior_counts.get('NO_PRIOR_LOCK', 0)}个没有该旧锁条目、0个MISMATCH。没有旧锁不等于文件无效，来源范围和本轮摘要已单独保存。

|层级|来源与用途|实际粒度/限制|
|---|---|---|
|原始OpenHands消息|`/home/wjg/data/agentic_coding_trajectories/sessions.parquet`，按本次快照Session ID联接|总15000条、包含多个来源；本轮实际联接672个OpenHands Session。每个快照四Session是受控分组，不是原生四路同时Ready|
|原始AgentX|`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl`，按23个trace ID及原始行定位|局部块身份、长度、调用/subagent容器；23条均未发现完整Prompt/messages/input_token_ids/token_ids字段。完整trace的容器数量包含Epoch之后记录，只用于描述性来源审计，不作当时决策输入|
|原始OpenHands分配快照|`evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/snapshots/*.json`及population_manifest|168个canonical快照，含候选位置、Session/lineage、驻留和历史Node身份；没有当前Strict所需的完整Token/组件认证|
|原始Formal评估行|`rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl`及`rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl`|504+69行。直接读取diagnostics的兼容、位置、目标、单边收益、FlowState逐步选择和原策略输出|
|四策略冻结输出|`evaluation/p1_strong_baseline_validation/per_case_results.jsonl`|573行；读取PCB/LDR/WFC输出及轨迹，并和原FlowState行精确联接；不是用其summary结论代替计算|
|交叉证据|Gate1 matrices、AgentX formal_snapshot_manifest、旧runtime_snapshots|逐格验证收益/兼容/预算；不用于制造新关系。P0 all_cases仅作原始canonical文件定位|

四策略逐Pending的J/E/G均独立重算；最大J差 **{manifest['max_J_recalculation_error_ms']:.3g} ms**。收益逐格与原diagnostics、冻结四策略输入及Gate1矩阵交叉相等。原始行号、snapshot_digest、候选/目标、去预算输入摘要在逐条件CSV中完整保留。没有从Workflow ID补造新关系：只核查旧谓词确实生成了它声称的旧矩阵。

所有自然来源统计均为**冻结Formal问题**，不是573条Strict Snapshot、573个自然并发边界或GPU性能样本。模型化ms不解释为TTFT。Node仅为旧生命周期身份，不能跨Engine复用。

## 3. 指标定义与复算方法

- 单边收益：`b[p,c]=Phi(T,T)−Phi(T−L,T)`，拒绝边为0；固定Phi系数37.828150、0.345974143、−0.156201917，G/T先除1024；代码独立实现，不import生产模块。
- 正收益支持：`P_c={{p:b[p,c]>1e−9 ms}}`；共享指支持大小≥2。共享比例分母为正收益候选数，分母0写UNKNOWN，不填0%。
- 候选重叠：不同正收益候选的支持交集非空；另存交集Jaccard、`sum_p min(b_pc,b_pd)`的重复收益、每Pending正收益候选数。相似收益仅为描述性阈值：同一正收益Pending相差不超过两者较大值的5%，不参与选择，未根据策略胜负调参。
- 严格支配：等成本下候选向量各分量不低且至少一维严格更高；零支持候选不计入“被支配正收益候选”。相等向量另计。没有删除任何候选。
- 集合收益：`F(S)=sum_p max b_pc`；前沿E用原兼容关系下最大L；J直接按`sum Phi(T−E,T)`重算，并核对冻结逐Pending记录。
- 边际：只沿**已保存的**FlowState选择轨迹重算全部剩余候选，不另跑新策略。序逆转采用代码中的双绝对差>1e−9ms，且独立差×边际差<−1e−9ms²的稳健判据；它是本次定义的可复算排序变化指标，不是假定所有微小浮点差都具有机制意义。
- K_min：对各Pending全域最深前沿建立覆盖位掩码，以Pending域DP求最小基数和见证，并统计最小覆盖数；当最深实现者唯一时，另用这些唯一候选并集大小独立复核。最多8个Pending，状态数≤256，不枚举149候选的所有子集，不实现新的分配策略。
- 可分离最优证书：各Pending最佳独立收益Top-K上界；最优代表多解数仅在最多4个Pending组上组合计数。共享且预算饱和时，以J(全部候选)作可达下界证书。573条四策略均达到相应上界/下界；不是只因为四者相等就称最优。
- 比较预算：候选ID/位置、完整兼容矩阵、Pending/T、收益及大小的去预算SHA相同才允许比较。对每快照三预算全部两两比较，分别报告四策略。
- 真实分支指标：原生分支数、公共Token历史长度/比例、分支执行深度及Pending真实依赖均写UNKNOWN。`sa/fa`派生名称、logical lineage、支持集、hash块LCP均不当作这些指标的替代。

## 4. 分来源的完整结构统计

|指标（条件数，另注除外）|OpenHands|AgentX|总体|
|---|---:|---:|---:|
|快照/预算条件|168/504|23/69|191/573|
|存在跨Pending正收益共享|0|66|66|
|存在候选收益重叠|504|33|537|
|存在严格支配正收益候选|504|24|528|
|存在5%相近收益对|195|18|213|
|K<K_min|90|0|90|
|K<正收益候选数|504|4|508|
|尚有未选正收益候选|504|24|528|
|尚有未选正**边际**候选|90|0|90|
|达到完整候选前沿|414|69|483|
|未填满K|323|64|387|
|轨迹中边际被已有集合压低|504|27|531|
|稳健独立/边际排序逆转|494|15|509|
|四策略同最终集合/J/E/G|504|69|573|
|四策略同列表顺序|501|42|543|
|至少一次Greedy最大边际并列|82|0|82|
|唯一最小基数最优集合|496|69|565|
|发现同基数等J替代集合|8|0|8|
|预算可加零边际候选且J不变|323|64|387|
|所有预算下FS选集不变（快照）|78|23|101|
|预算增加后移除旧候选（快照）|0|0|0|

这些是本轮读取结果后生成的表，全部逐条件值见CSV；同一快照三个K为重复条件，不能将共享频率或候选数跨K累计当独立工作流规模。

**候选域与预算：** OpenHands n=8–20、m=4、K=2–15、K_min恒4；AgentX n=6–149、m=2–8、K=1–111、K_min=0–2。OpenHands去重快照合计2276个候选、2264个正收益候选；AgentX合计568个逻辑候选、84个正收益候选，其中30个共享，正收益共享比例30/84=35.71%。该比例是跨快照候选加权值，不等于平均每快照比例；3个零收益条件的比例UNKNOWN。

**全域饱和并不等于G=0：** OpenHands有36条件、AgentX全部69条件的J(全候选)>0。K_min定义为达到候选域可达最深状态，不是保证每个请求完全无需恢复。预算“大于有效候选数”“覆盖可达最深前沿”和“填满物理容量”不能混用。

## 5. 为什么四策略同集：逐假设结论

|假设|本轮证据|裁决|
|---|---|---|
|缺少复杂共享|OH支持全单例；AX共享候选全为二员支持，且支持域没有发现非层次交叉；但原轨迹不等于这个Formal域|对**入模收益结构**有支持；不能推出真实Agent工作流本身没有复杂分支|
|明显收益支配|OH504/504、AX24/69存在；更深同支持候选压过浅层；全域前沿实现者可明确列出|是压缩有效决策的因素；但不是共享问题必然可解的一般充分条件|
|预算足够|AX69/69已饱和，23快照预算均不改变选集；OH仍有90紧预算|足以解释AX原预算的退化，不足以单独解释OH全部条件|
|重叠没有改变边际选择|531条件边际被压低、509有稳健序逆转；终集依然相同|“没有边际变化”不成立；实际是变化未造成四规则在这些预算下终集分歧|
|真实分支结构简单|原始AgentX有135个subagent容器，原生产决策图/Token/Ready证据不足；OH分组本身是人为编排|**待验证**，不能从兼容支持或相同选集推论|
|J相同或tie-breaking造成平局|573同集同J；仅8条最小基数多解，82条有中间并列；多数最小代表唯一|tie-breaking解释少量代表选择/顺序，不能解释主要结构性一致|

### OpenHands：可分离 + 支配 + 已观察截断排名一致

每个候选最多支持一个Pending，因此同一Pending只需要其最佳代表；其余历史状态的边际会在更深代表入选后归零。四请求间最佳收益Top-K给出精确上界。独立按收益和位置构造推荐序、在原K截断后均与冻结PCB/LDR终集相同，复核见[ordering_and_boundary_audit.csv](ordering_and_boundary_audit.csv)。

这不是“所有位置排序总与Phi排序等价”的定理：不同T可改变排序，原结果已有顺序差异。此处只是每个原K的成员集合一致；更浅同链的排序逆转也不必改变组内最佳代表。504条均未保留零删除贡献项，观察到的动态更新用于避开同链冗余，但不能据这些平局声称优于全部简单策略。

### AgentX：当前需求窄 + 总候选比例预算 + 前沿实现者足够少

总候选n可很大，但绝大多数与当前目标无正收益；达到当前最深前沿只需0–2项。原K按总候选比例取0.25/0.5/0.75，69条都足够，四种规则最终恰好选中了唯一的最小前沿覆盖代表。此时继续保留其他正独立收益项也没有边际增益。

**该结论只对原预算成立**。本轮不扫描更紧预算，不引入旧扩展扫描作为新增样本，不以0次替代推断整个解路径总嵌套；现有统计没有检验紧共享预算下规则是否分歧。

## 6. 真实结构与物理身份缺口

OpenHands原始消息可追溯，但每快照是四个独立Session组合；canonical中的lineage是同Session标签，旧`state_catalog.is_compatible`还要求Workflow ID相同。故“OH无跨Pending共享收益”是**在该输入编排与兼容定义下**的事实，不能说明不同Session真实Token没有公共System或任务前缀。原快照只存旧驻留/Node摘要，非新Strict完整Token/组件合同。

AgentX原始数据有调用和subagent容器，完整请求容器深度为1或2，但没有完整Prompt/Token。派生flat-agent、seam与lineage不能直接作为原生Fork记录。将原始subagent数量直接报告为“分配时分支数”会混入时刻和定义差异，因此本轮保留UNKNOWN。原始完整trace的调用和容器计数见source_manifest中的逐trace记录；不把事后调用作为当时选择输入。

在去重的23个AgentX旧Runtime快照上，本轮按完整ID联接重算Formal与Raw关系，得到70条Formal-only、413条Raw-only；它们不是新增Strict边，也没有乘以预算当独立边。11个快照发现多个逻辑候选共Node。故此处的K与K_min是**逻辑Formal值**，不能用这些旧记录直接构造安全物理驱逐或真实GPU恢复比较。原始Token缺失也使首次分歧位置、真实公共历史长度与完整组件别名无法补认证。

## 7. 产物、复现与验证范围

|产物|行数或内容|
|---|---|
|natural_workload_per_condition.csv|573行，每行有condition_id、来源行号、输入摘要、完整候选/目标/支持/策略集合及未知字段|
|natural_workload_structure_summary.csv|分OpenHands、AgentX、总体的长表；指标单位与条件/快照分母明确|
|strategy_agreement_analysis.csv|3438行=573条件×四策略六对；分别比较集合、顺序、J与E/G|
|budget_replacement_analysis.csv|2292行=191快照×三个预算对×四策略；全部先验检查去预算输入相同|
|calculation_evidence.jsonl|573条独立收益矩阵、原Greedy轨迹再评价、终态边际、K_min覆盖位及逐策略E/G/J|
|source_manifest.json / .md|185个读取来源的前后内容摘要、672个Session和23个trace的真实来源匹配；无原始消息大体积复制|
|source_integrity_crosscheck.csv|当前摘要与可用旧冻结锁的逐文件对照，缺旧锁明确标识|
|ordering_and_boundary_audit.csv|573行收益/位置推荐及已观察预算截断核查|
|analysis_validation.json|独立产物约束与每条件PCB/LDR推荐校验；不是GPU或新Strict门禁|

使用已有`/home/wjg/uv312/bin/python`（已有pyarrow；本轮未安装依赖）。复现必须指定一个**不存在**的新子目录，原结果和来源均不覆盖：

```bash
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/natural_analysis/scripts/analyze.py --out evaluation/rq3_workload_structure/natural_analysis/reproduce_01
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/natural_analysis/scripts/make_report.py --input-dir evaluation/rq3_workload_structure/natural_analysis/reproduce_01
```

不导入当前Optimizer或任何策略选择函数，不调用旧审计脚本，不执行GPU。K_min的位掩码计算与最优性上界证书是必要离线统计，不是新分配算法实验。数据源内容发生变化或输出目录已有结果时直接拒绝，不替换来源、不自行重试。

## 8. 当前不能解释的问题及下一阶段值得研究的方向

**仍无法回答：**真实原生Workflow在决策时的分支数/分叉Token/深度差、自然Ready同步性、完整输入下真正可执行共享比例，以及旧Formal策略差异能否转成实际恢复。这些缺口不能由本次数学最优性证书或所有策略同集补足。

仅建议供后续人工审查的结构问题，不实施：

1. 共享存在且K<K_min时，公共浅层与深层专属/分支状态如何竞争；必须与可分离紧预算对照，避免把“紧”本身当复杂性的充分条件。
2. 在相同共享支持下改变深度和收益重叠，区分静态独立排序、动态边际与覆盖数量规则；既保留可能同集的情形，也允许结果无差异。
3. 预算从不足到饱和时，最优集合是否非嵌套、早期公共状态是否需要被替换；不能由本轮原预算0替代推断其不可能。
4. 用真实Token与完整状态身份认证支持关系，再与逻辑矩阵的结论对照；不能靠Workflow标签或hash合成历史制造共享。

本阶段到此停止。没有进入受控结构敏感性实验，没有运行更紧K扫描、GPU或论文写作。
'''
    with (out / 'natural_workload_audit_report.md').open('x') as f:
        f.write(report)
    source_text = '# 原始冻结来源与本轮内容身份\n\n本清单由实际读取的逐文件摘要生成，不使用旧汇总数字作为统计输入。原始记录行映射见逐条件CSV和JSON清单。\n\n|路径|用途|字节数|SHA-256（前后相同）|\n|---|---|---:|---|\n'
    for item in manifest['source_files']:
        source_text += f"|{item['path']}|{item['role']}|{item['bytes']}|`{item['sha256_before']}`|\n"
    with (out / 'source_manifest.md').open('x') as f:
        f.write(source_text)
    with (out / 'analysis_validation.json').open('x') as f:
        f.write(json.dumps({'status': 'PASS', 'checks': checks, 'passed': len(checks),
                            'prior_lock_status': dict(prior_counts),
                            'report_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'scope': '冻结Formal离线统计，不是新Strict或GPU认证'}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(out / 'natural_workload_audit_report.md'), 'passed': len(checks),
                      'prior_lock_status': dict(prior_counts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
