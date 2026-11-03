"""根据只读复核结果生成中文交接表与结构归因，不执行策略。"""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
details = json.loads((OUT / 'evidence_details.json').read_text())
v = json.loads((OUT / 'verification.json').read_text())
sources = json.loads((OUT / 'source_hashes.json').read_text())
sr = list(csv.DictReader((OUT / 'structure_metrics.csv').open()))
solvers = list(csv.DictReader((OUT / 'solver_comparison.csv').open()))

def link(path):
    return '[' + Path(path).name + '](' + str(ROOT / path) + ')'

def short(ids):
    if isinstance(ids, str):
        ids = json.loads(ids)
    return '{' + ','.join(x.replace('B5C3_NODE_', 'N') for x in ids) + '}'

def write(name, text):
    (OUT / name).write_text(text + '\n')

# 全策略配对关系将选集、前沿和目标三个维度分别记录。
pairs = []
for case, d in details.items():
    for c in d['conditions']:
        for a, b in itertools.combinations(c['policies'], 2):
            x, y = c['policies'][a], c['policies'][b]
            ex = [p['E'] for p in json.loads(x['per_pending'])]
            ey = [p['E'] for p in json.loads(y['per_pending'])]
            eq = abs(float(x['J_ms']) - float(y['J_ms'])) <= 1e-9
            ss = set(json.loads(x['selected'])) == set(json.loads(y['selected']))
            pairs.append({'case': case, 'K': c['K'], 'policy_a': a, 'policy_b': b, 'same_set': ss, 'same_E': ex == ey, 'same_J': eq, 'J_a_minus_b_ms': float(x['J_ms']) - float(y['J_ms']), 'category': 'SAME_SET_SAME_COST' if ss and eq else 'DIFFERENT_SET_SAME_COST' if eq else 'DIFFERENT_COST'})
with (OUT / 'pairwise_relations.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)

manifest = ['# T1 数据来源与冻结身份', '', '审计日期：2026-10-26。仅补充离线统计；未运行策略实验或 GPU。审计门禁与历史 Runtime 认证门禁不是同一个概念。', '',
'## 来源及口径', '',
'| 来源 | 实际行数/条件 | 身份与用途 |', '|---|---:|---|',
'| P1 `per_case_results.jsonl` | 573 | OpenHands 504、AgentX 69；包含强基线及明确收益矩阵，不依据 Workflow 标签重造关系 |',
'| 原 OpenHands `per_snapshot_results.jsonl` | 504 | 按 P1 source_file/source_line 核对 snapshot_id、k、snapshot_digest、FlowState 选集与成本 |',
'| 原 AgentX `per_case_results.jsonl` | 69 | 同上；非 Strict Epoch |',
'| Gate1 `matrices.jsonl` | 573 | 独立交叉核对每格收益、等大小成本与预算 |',
'| E3 `condition_results.csv` | 140 | 5 个快照 × 4 个预算 × 7 策略；CSV 行号包含表头，JSONL 从 1 开始 |',
'| E3 `selection_traces.json` | 50 | 各预算逐轮轨迹；5 个 K4 路径共20步，较小 K 重复其前缀 |',
'| E2 w1–w4 / B5-C3 Strict Epoch | 5 | 每个7候选、4 Pending、28三态关系；逐准入边核验原始Token前缀、Epoch和配置摘要 |',
'| G2 `formal/raw_results.csv` | 40 | 单个 B5-C4/K2；5轮 × 4 Pending × 2策略；只读对应，不计入 Strict20 CPU 条件 |', '',
'## 校验与版本', '',
f'- 四份冻结清单共检查 {len(v["manifests"])} 个文件条目，SHA-256 全部吻合；未发现缺失或版本不一致。完整 expected/actual 见 [verification.json](verification.json)。',
f'- 本次 {len(v["checks"])} 项断言全部通过；包括逐格收益、逐 Pending E/G/J、冻结边际轨迹、原始行身份、K_min 独立证书与读取前后 SHA 一致性。',
'- Phi 使用冻结 `position_aware_quadratic_v1`，系数 a=37.82815、b=0.345974143、c=−0.156201917；G/T 先除1024，输出 modeled ms。容差为绝对 1e−9 ms。',
'- 本次不导入当前生产 Optimizer，不因当前代码变化重算策略选集。历史策略版本由冻结清单认证；E3 首改善 1-Swap 规则来自其 protocol.json。',
'- 先前存在的 `t1_extract.py` 未作为本次统计入口或证据使用；本次入口为 [audit_frozen.py](audit_frozen.py) 与 [write_reports.py](write_reports.py)。', '',
'## 不可混同的身份', '',
'- 历史573：只证明给定 Formal 矩阵上的代数性质与策略结果。真实 Token、完整同边界状态认证缺失，Strict 支持、Strict K_min、实际 Runtime 收益均为 UNKNOWN。本表给出的历史结构数值明确是 Formal 值。',
'- AgentX 既有 Gate1 Step3 审计记录70条 Formal-only 新增关系（对应该次原始/形式化域比较，不能乘以预算当成独立边）；不能把历史 Formal 最优性作为真实可恢复性。',
'- Strict20：5 个受控来源快照，20 个预算条件不是20个独立工作流；EPOCH_ADMISSIBLE 不意味着逐边 RESUME_VALIDATED。所有Node编号均为各快照内部身份。',
'- K 为等大小 Recurrent Checkpoint 逻辑单位。E2冻结预算合同要求独占等大小槽、无别名；B5-C4记录每项51,511,296 bytes。原采集Epoch的 logical_budget="未定义；本轮不分配"，后续CPU协议明确K=1..4，不反向修改采集记录。', '',
'## 源文件索引与内容摘要', '', '| 文件 | SHA-256 | 字节数 |', '|---|---|---:|']
for p, x in sources.items():
    manifest.append(f'| {link(p)} | `{x["sha256"]}` | {x["bytes"]} |')
write('source_manifest.md', '\n'.join(manifest))

case_text = ['# Strict20 逐案结构归因', '',
'所有 J/边际均为 modeled ms；E/G 为 Tokens。N18 等简称只在各自快照内有效。下面15个失败条件全部保留；完整7策略与140行结果见 solver_comparison.csv。', '',
'## 完整结构', '', '| 快照 | 候选位置与正收益支持 | K_min及独立覆盖见证 |', '|---|---|---|']
for case, d in details.items():
    pid = {p['identifier']: 'ABCD'[i] for i, p in enumerate(d['input']['pending'])}
    desc = []
    for c in d['input']['candidates']:
        support = d['metrics']['positive_supports'][c['identifier']]
        desc.append(f"{short([c['identifier']])}@{c['position']}→{''.join(pid[p] for p in support)}")
    case_text.append(f"| {case} | {'；'.join(desc)} | {d['metrics']['K_min']}：{short(d['metrics']['K_min_witness'])} |")
case_text += ['', '所有支持族均 nested/disjoint，未发现非层次交叉。w3 的 AB、CD 各有两个同支持候选；w4、B5-C4 的 CD 有两个同支持候选。这些同域候选的收益互相重叠，不能简单相加。', '',
'## 全部 Greedy 非最优条件', '', '| 条件 | Exact选集 / J | Greedy选集 / J | 1-Swap选集 / J | Greedy−Exact |', '|---|---|---|---|---:|']
for case, d in details.items():
    for c in d['conditions']:
        p=c['policies']; e,g,s = [p[a] for a in ['Exact','Greedy','Greedy+1-Swap']]
        if float(g['J_ms'])-float(e['J_ms']) <= 1e-9: continue
        case_text.append(f"| {case}/K{c['K']} | {short(e['selected'])} / {float(e['J_ms']):.6f} | {short(g['selected'])} / {float(g['J_ms']):.6f} | {short(s['selected'])} / {float(s['J_ms']):.6f} | {float(g['J_ms'])-float(e['J_ms']):.6f} |")
case_text += ['', '## 每个失败条件的实际逐步选择、前沿及替代关系', '',
'Exact仅有最优集合，没有天然的“第一步”。这里的首次分歧定义为：Greedy轨迹首次选中不属于冻结Exact代表集合的候选，而不虚构Exact逐轮轨迹。15例均为第1步N20；这不等于任何共享祖先必然应该排除。K1五例N20恰为最优。']
for case, d in details.items():
    for c in d['conditions']:
        p=c['policies']; e,g,s = [p[a] for a in ['Exact','Greedy','Greedy+1-Swap']]
        if float(g['J_ms'])-float(e['J_ms']) <= 1e-9: continue
        case_text += ['', f"### {case}/K{c['K']}", '', '| 步 | 实际选择 | 当轮边际 ms | 改善Pending数q | Token推进r |', '|---:|---|---:|---:|---:|']
        for t in c['trace']:
            a=next(x for x in t['ranked'] if x['checkpoint_id']==t['greedy_chosen'])
            case_text.append(f"| {t['step']} | {short([t['greedy_chosen']])} | {a['gain_ms']:.9f} | {a['q']} | {a['r']} |")
        case_text += ['', '| Pending | Exact E/G | Greedy E/G | 1-Swap E/G |', '|---|---|---|---|']
        for i,label in enumerate('ABCD'):
            pe,pg,ps=[json.loads(x['per_pending'])[i] for x in [e,g,s]]
            case_text.append(f"| {label} | {pe['E']}/{pe['G']} | {pg['E']}/{pg['G']} | {ps['E']}/{ps['G']} |")
        if c['K']==2:
            reason='首步公共祖先占据一个槽；预算只剩一项，无法同时保留两个深层分支。冻结Exact选择两个深层分支；一次祖先换出即可修复。'
        elif c['K']==3:
            reason='两个分支保留后，公共祖先已没有额外前沿贡献，仍占用一个名额。Exact将预算用于深层专属/分支组合。'
        else:
            reason='K已达到K_min，但公共祖先及部分被替代分支仍占据选集名额；逐步只添加的Greedy无法释放这些名额。Exact达到全域前沿，1-Swap也完成修复。'
        case_text += ['', reason]
        for st in c['swap_trace']:
            for step in st['steps']:
                case_text.append(f"- 冻结替换：{short([step['remove']])}→{short([step['add']])}，J {step['before_J']:.6f}→{step['after_J']:.6f}。")
        if 'fixed_local_set_neighborhood_min_J' in c:
            ns=c['fixed_local_set_neighborhood_min_J']
            case_text += ['',f"固定首改善最终停点J={float(s['J_ms']):.9f}；其全部一换一邻居最小J={ns['1']:.9f}，二换二最小J={ns['2']:.9f}，三换三最小J={ns['3']:.9f}。前两类均更差，不能以严格下降局部步离开。这个最终停点需要三个名额同时重配才能改善。",
            '但这不是“所有1-Swap路径都失败”：冻结local_trap_diagnostics.json保存从原Greedy出发的另一条两步严格改善路径，可以达到Exact。首改善ID顺序影响进入哪个局部盆地，不得将两例推广为任意局部搜索必败。']
case_text += ['', '## B5-C4/K1→K2 与 G2 对应', '',
'K1 Exact={N20}，J=228.054813；K2 Exact={N18,N27}，J=48.228729，最优代表集合非嵌套。Greedy固定前缀选集{N20,N18}，J=130.803084，差82.574355。N18覆盖AB、N27覆盖CD；N27位置3520且同支持收益支配N21位置3456，主结果仍保留N21，没有删候选。', '',
'G2原始CSV共40臂（5轮×4 Pending×2策略），全部记录Cold匹配。其冻结选集、模型E/G与本次B5-C4/K2一致；这是该条件的实际恢复对应，不能据此认证其余Strict20条件。', '',
'| 策略/请求 | E | G=T−E | 实际Prefill Tokens | CUDA Prefill五轮中位ms | TTFT五轮中位ms |', '|---|---:|---:|---|---:|---:|']
for key,r in v['g2'].items():
    case_text.append(f"| {key} | {r['E']} | {r['G_model']} | {r['prefill_tokens']} | {r['cuda_median_ms']:.3f} | {r['ttft_median_ms']:.3f} |")
case_text += ['', '**源报告冲突：** G2 final_report.md §4 将“C/D”的G写为Exact[46,384]、Greedy[1134,1472]，与逐请求CSV相反。正确按C、D顺序应为Exact[384,46]、Greedy[1472,1134]。其Prefill数组[915,966]和[2003,2054]顺序正确。本审计保留原文件，采用CSV并显式说明文字错序。', '',
'实际Prefill包含T之后的输入尾部，因此不等于G。A/B两策略前沿相同，仍保留其小幅负面/波动结果；C/D前沿不同且设备时间下降。该数据不是完整在线净收益的证明，也不是Exact相对1-Swap的证据：此K2的1-Swap选集已经等于Exact。']
write('case_attribution.md', '\n'.join(case_text))

final = ['# FlowState T1：工作负载结构与预算敏感性审计', '',
'**判定：PASS（冻结CPU结构与结果审计通过）。** 历史573条件仅保留Formal模型解释；不因此获得Strict Runtime认证。未运行GPU、未重新执行分配策略、未修改算法或冻结文件。', '',
'## 1. 可以写入论文的核心发现', '',
'1. **历史平局有具体的充分条件。** OpenHands 504/504正收益支持可分离；其中90个预算仍紧张，但可用分组Top-K精确化简。AgentX 69/69预算达到完整前沿所需K_min，其中66个条件有共享。四策略FlowState/PCB/LDR/WFC在573/573选集、E/G、J均相同，不只是目标打平。',
'2. **Strict20中层次共享足以产生不可回退损失。** 五个快照均含全局公共祖先和深层分支，K_min=4。Greedy/WFC仅K1达到Exact，其余15条件均有差距；K4虽预算充分，Greedy仍未找到存在的全域覆盖解。非层次交叉共享不是出现这种现象的必要条件。',
'3. **局部替换很强但不是普遍保证。** 固定首改善1-Swap修复15个Greedy失败中的13个，达到18/20最优；w1/K3、w2/K3仍分别差9.864009、9.802104 modeled ms。这是特定冻结遍历规则的失败，不是所有1-Swap路径的失败。',
'4. **支持联合取舍，不支持“Greedy独有优势”。** Greedy与WFC在Strict20全部20条件同集同成本；与StandaloneTopK相比仅4条件严格更优、16条件持平。Exact提供目标最优参照；本数据不能把联合分配价值等同于Marginal Greedy优于全部简单基线。', '',
'## 2. 建议 VI-C 的核心表', '',
'建议用下面一张分块表，保留各预算，不以平均目标抹去失败。历史行只呈现结构解释；Strict行呈现求解最优条件数。', '',
'| 数据/预算 | 条件数 | 共享条件 | K<K_min | Greedy最优 | 1-Swap最优 | WFC最优 | Exact最优 |', '|---|---:|---:|---:|---:|---:|---:|---:|',
'| OpenHands Formal | 504 | 0 | 90 | 504（代数证书） | 未运行 | 504（代数证书） | 未枚举 |',
'| AgentX Formal | 69 | 66 | 0 | 69（完整前沿证书） | 未运行 | 69（完整前沿证书） | 未枚举 |']
for k in range(1,5):
    rr=[r for r in solvers if r['dataset']=='Strict20' and int(r['K'])==k]
    nums=[sum(r['optimal']=='True' for r in rr if r['policy']==p) for p in ['Greedy','Greedy+1-Swap','WFC','Exact']]
    final.append(f'| Strict20 / K={k} | 5 | 5 | {5 if k<4 else 0} | {nums[0]} | {nums[1]} | {nums[2]} | {nums[3]} |')
final += ['', '分母是预算条件，不是独立工作负载：历史对应168个OpenHands快照、23个AgentX快照；Strict20仅5个快照。各快照n=7、m=4；Historical n=6–149、m=2–8。完整指标见 [structure_metrics.csv](structure_metrics.csv)，结果见 [solver_comparison.csv](solver_comparison.csv)。', '',
'### Strict20全策略目标与选集关系', '', '| 策略 | 达到Exact目标 | 与Exact同集 | 同目标但异集 | 成本不同 |', '|---|---:|---:|---:|---:|']
for p,x in v['summary']['strict_optimal'].items():
    final.append(f"| {p} | {x['optimal']}/20 | {x['same_set_as_exact']}/20 | {x['ties_different_set']}/20 | {20-x['optimal']}/20 |")
final += ['', '该表只以冻结Exact代表集合为参照，不宣称最优集合唯一。所有策略间同集、同前沿、同成本三个维度另见 [pairwise_relations.csv](pairwise_relations.csv)；不能用ID不同代替收益差异。', '',
'## 3. 结构条件、数学结论与适用边界', '',
'### PROVED：可分支持的精确化简（充分条件）', '',
'固定非负 b、等大小、纯基数预算，若每个候选的正收益支持≤1，令q_p为该Pending所有专属候选的最高收益，空集合取0。任何选集对同一Pending只需保留收益最大的一个，其余删除不损失F。不同Pending的正收益候选域互不相交，因此选择最大的至多K个q_p及各自实现者恰为全局最优。零收益、空域和K超过有效Pending数量均不改变结论。',
'该定理不说明任意PCB/LDR规则都会选到各组最佳；本次573的具体选集确实达到上述证书。支持可分离不是简单策略最优的必要条件，AgentX饱和共享和Strict20/K1均为反例。', '',
'### PROVED：预算充分保证“存在”全域覆盖解', '',
'令E*_p=max_{c∈C兼容p}L_c∪{0}。K_min是覆盖所有正E*_p的最少候选数，每个候选只覆盖它能实现最深前沿的Pending。K≥K_min时见证集合可行；任意集合E_p≤E*_p，Phi关于G单调不减，故J_OPT=J(C)。这不保证Greedy找到见证，也不要求E*_p=T_p。',
'本次K_min先用Pending掩码DP精确计算（最多8个Pending、256种掩码，不枚举149候选的分配集合），再用独立下界验证：所有最大非空前沿覆盖域两两不交，每块至少需一个候选，见证恰达到此下界。逐条件见证与全域前沿保存在结构CSV。',
'Strict20中w3、w4、B5-C4的J(C)分别仍为1.142854、1.105138、1.753119 ms；预算饱和不等于零Gap。', '',
'### OBSERVED：共享、重叠与紧预算的组合', '',
'- 五个Strict快照的共享候选数依次为w1=3、w2=3、w3=5、w4=4、B5-C4=4（各7候选）。支持均为嵌套或不交，crossing_pairs=0。',
'- 共享比例本身不能预测最优性差距；同一支持域的不同深度候选贡献重叠，F按Pending最大收益而非独立收益求和。',
'- K1每个方法只能选一项，Marginal Greedy与独立总收益排序等价；五例均选公共祖先。K2允许两个分支联合替换公共祖先，五例Exact均排除K1祖先，形成非嵌套代表解。K3/K4则竞争分支与更深专属状态。',
'- 15个Greedy失败逐案核对均在首步选择了不属于冻结Exact代表集合的N20；K3时它已被两个分支完全覆盖而仍占名额。后续损失不能全归于一个原因：K4还有被深层状态替代的分支冗余；w1/w2K3的局部搜索则停在另一组无N20的组合中。',
'- WFC按当前能推进的Pending数量q、总Token推进r排序，Greedy按Phi边际。50条冻结步骤（20个不同K4路径步骤）两者赢家一致；非赢家排序仍可不同，不能证明两评分函数数学等价。', '',
'### CONDITIONAL / OPEN：更强结构与求解器', '',
'真实Token祖先关系在同配置、完整状态及无额外过滤/物理耦合条件下可形成树结构；本次观测到的laminar支持不单独证明任意Runtime可直接使用树DP。额外兼容过滤、物理别名、异构大小可能破坏简化。本轮不实现新求解器，也不推出运行时的总体复杂度/收益结论。', '',
'## 4. 关键失败与真实恢复对应', '',
'- B5-C4/K2：Greedy {N20,N18} J=130.803084，Exact/1-Swap {N18,N27} J=48.228729，差82.574355 modeled ms。联合分支替换有明确前沿效应，而不是仅改ID。',
'- w1/K3：固定首改善停在{N21,N22,N24}，J=31.817384；Exact {N18,N25,N27}，J=21.953375。w2/K3同类结构，J=31.767642 vs21.965538。对这两个最终停点，一换一及二换二所有邻居均更差；三换三可改善。不过从原Greedy存在不同的一换一路径可达到Exact。',
'- [case_attribution.md](case_attribution.md)完整记录全部15例的实际Greedy步骤、边际、选集、每Pending E/G、替换序列与局部邻域最低值，没有猜测Exact步骤。',
'- G2仅核对B5-C4/K2：40臂、5轮，A/B两方法E=3712；C/D由Greedy E=2432变为Exact E=3520。C的CUDA Prefill中位74.342→40.574 ms，D为78.241→42.231 ms（精确值见逐请求表）；A/B近零对照保留负面波动。该有限实测独立于模型目标，不能外推成20条件或生产端到端收益。',
'- G2原报告C/D Gap数组错序，已在归因报告显式登记；本次采用原始CSV，不覆盖历史文件。', '',
'## 5. 论文可保留与不可支持的主张', '',
'**可以保留：** 固定严格可执行关系下，恢复状态价值存在重叠，紧预算下联合选择两个深层分支可优于不可回退祖先选择；层次共享已足以产生非嵌套最优解。简单规则在可分支持或已经覆盖全域前沿时能够精确最优；局部替换修复大部分但非全部本数据中的Greedy损失。', '',
'**应收窄：** “需要联合Checkpoint分配”指必须评价重叠和跨Pending取舍，不能表述为必须使用FlowState Greedy，更不能把经典次模性质称为新数学贡献。Exact在这些有限规模上提供最优目标，1-Swap已覆盖18/20，WFC与Greedy20/20一致，均须写入Evaluation。', '',
'**不能证明：** 共享必然造成困难、紧预算必然Greedy次优、预算充足保证Greedy最优、层次支持必然需指数求解、1-Swap所有顺序都会失败、Position-aware位置项独立带来分配/实际GPU优势、历史573已经严格认证、20条件为20个独立工作流、模型J就是TTFT、全系统在线净收益。', '',
'## 6. 审计完整性与停止', '',
f'共产生593行结构指标、2432行求解对照（573×4+20×7），以及420行Strict策略配对关系。{len(v["checks"])}项复核通过，冻结清单无不一致。历史代数最优证书在CSV中与冻结Exact结果明确区分，不虚构历史Exact选集。',
'必要原始文件可读，结构/结果推断未因缺证据中断；历史Strict认证缺口按UNKNOWN保留。只读复核脚本和源码摘要见 [source_manifest.md](source_manifest.md)。本任务到此停止，不开展后续实验或修改论文正文。']
write('t1_final_report.md', '\n'.join(final))
print('报告写入完成；全部新文件位于指定审计目录。')
