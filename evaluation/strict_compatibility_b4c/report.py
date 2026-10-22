"""只汇总本轮正式结果，列出全部胜平负与机制，不启动实验。"""
import json
import hashlib
from collections import Counter
from pathlib import Path
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text());xs=r['snapshots'];rows=[a for x in xs for a in x['rows']]
policies=['FlowState','Exact OPT','PCB','LDR','WFC','StandaloneTopK','Marconi Adaptation']
comparisons={}
for policy in policies[1:]:
 rs=[x for x in rows if x['policy']==policy];counts=Counter(x['FlowState_WTL'] for x in rs)
 comparisons[policy]={'W':counts['W'],'T':counts['T'],'L':counts['L'],'cases':[{'id':f"{x['group']}/K{x['K']}",'outcome':x['FlowState_WTL'],'difference_ms':x['baseline_minus_flowstate_ms'],'same_set':x['same_set_as_flowstate'],'different_set_same_objective':x['different_set_same_objective']} for x in rs]}
(OUT/'all_comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n')
summary='\n'.join(f"| {p} | {v['W']} | {v['T']} | {v['L']} |" for p,v in comparisons.items())
table=[];case_table=[]
for x in xs:
 for k in range(1,5):
  rs={a['policy']:a for a in x['rows'] if a['K']==k}
  table.append('| '+x['group']+f' / K{k} | '+' | '.join(f"{rs[p]['J_ms']:.6f}" for p in policies)+' |')
  case_table.append('| '+x['group']+f' / K{k} | '+' | '.join(f"{rs[p]['FlowState_WTL']} ({rs[p]['baseline_minus_flowstate_ms']:+.9f})" for p in policies[1:])+' |')
losses=[a for a in rows if a['policy']=='FlowState' and a['gap_to_exact_ms']>1e-9]
loss_text='\n'.join(f"- {a['group']}/K{a['K']}：差距 {a['gap_to_exact_ms']:.9f} modeled ms，选择 {a['selected']}。" for a in losses)
mechanism=['# 结构与机制归因','','全部数字为冻结Phi下的modeled ms，不是Runtime时延。主层均A，A/B/C为非互斥标签。','']
for x in xs:
 mechanism += [f"## {x['group']}",'',f"冻结标签={x['structure']}；K_min={x['K_min']}；目标与候选见results.json/input。",'', '### 全部FlowState选择步骤（K=4的前缀同时覆盖K=1–3）']
 for step in x['traces']['4/FlowState']:
  mechanism.append(f"- 已选{step['prefix']}；选择{step['chosen']}；剩余边际={step['marginals_ms']}。")
 mechanism += ['', '### 每个预算的比较与静态冗余证据']
 bm=x['benefit_matrix_ms']
 def benefit(s):return sum(max([bm[c][i] for c in s]+[0]) for i in range(4))
 for k in range(1,5):
  rr={a['policy']:a for a in x['rows'] if a['K']==k}
  mechanism.append(f"- K={k}："+'；'.join(f"{p}={v['selected']}，J={v['J_ms']:.9f}" for p,v in rr.items()))
  selected=rr['StandaloneTopK']['selected']
  redundant=[c for c in selected if abs(benefit(selected)-benefit([a for a in selected if a!=c]))<=1e-9]
  mechanism.append(f"  静态集合中移除后F不变的候选={redundant}；只报告共同目标差，不把顺序/集合变化当胜利。")
mechanism += ['', '## 解释边界','',
'g003：WFC第一步按覆盖数量选择B1(q=4,L=2432)，FlowState选择A5(q=1,L=10816)。B1的推进总量9728也小于A5的10816，因此该差异不能单独归因于位置感知Phi项；覆盖数优先本身已经足以造成选择分歧。冻结Phi下A5收益420.732039636，高于B1的365.672296502。WFC先选共享祖先的预算占用持续影响后续K。',
'g001/K4、g003/K3与K4、g004/K3与K4：StandaloneTopK保留已被更深同链候选支配的项；动态边际将这些项置零并转向仍有收益的Pending。其余11条件目标持平，不可声称所有场景都需要动态更新。',
'共享边确实影响边际：g003的B1在空集收益365.672296502，A5后274.107221141，A5+B5后182.567061243，A5+B5+C5后91.121580107。其他三组首先选择共享B1，它降低每个Pending后续深层项的剩余收益。该事实证明共享覆盖进入目标；本次相对静态策略的严格收益主要由同链冗余挤占解释，不能单独归因于跨分支复杂重叠。',
'g001/K4、g002/K3和K4、g004/K4是Greedy的全部非最优条件：早期共享B1保留后不做交换。g002/K3时Exact已可通过A4/C4/D4放弃B而得到更低成本；K=4时四组均可保留全部最深项达到0，但三个共享优先Greedy集合仍有缺口。',
'没有运行Gap-only策略或位置模型消融，不证明位置感知模型独有的分配收益。不能将WFC数量优先的失败写成Phi校准带来的独立改善。']
(OUT/'mechanism_attribution.md').write_text('\n'.join(mechanism)+'\n')
report='''# Phase B4-C：严格Compatibility下的正式CPU分配评估

## 结论与门禁

**PASS**。仅使用g001–g004四个正式入选Snapshot，完成4个逻辑预算×7种策略，共112条结果。g000、g005、g006没有参与策略比较。原始根及各组Manifest、Snapshot摘要和最终执行锁244项核验通过；输入及生产源码保持不变。

FlowState相对WFC为4胜12平，相对StandaloneTopK为5胜11平；相对PCB/LDR各8胜4平4负。FlowState在12/16条件匹配Exact OPT，另外4个条件确实失败，未删除或修复。

这些是4个快照的16个重复预算条件，不是16个独立样本；不作总体显著性推断。四个主层均为A且都附有B/C标签，不拆成三个独立分层总体。

## 输入、定义与实现

来源：`evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g001..g004/snapshot.json`；56个候选、16个Pending、224配对（68准入、156拒绝、0未知），其中12条跨Workflow准入。仅EPOCH_ADMISSIBLE产生b，拒绝边为0；不调用旧Workflow-ID谓词重新过滤准入边。任何UNKNOWN都会使评估门禁失败，未默认为合法或零收益。准入不是RESUME_VALIDATED，B4-B1没有逐边恢复认证。

Phi固定为position_aware_quadratic_v1：g=G/1024、t=T/1024；37.828150g+0.345974143gt−0.156201917g²，无重新拟合。共同E=max所选准入L，G=T−E，J=ΣPhi，F=J(空集)−J；恢复成本单位modeled ms。

- FlowState复用生产GlobalOptimizer.select，仅独立子类覆盖成本接口为矩阵评分；同分容差1e−9ms、ID破同分及正边际停止规则保持不变。预算使用原等大小memory_bytes，K是逻辑项数。
- PCB/LDR/WFC直接导入冻结P1-B实现，四文件哈希与implementation_lock一致。PCB独立推荐最优成本项，LDR独立推荐最深项，WFC动态按改善数/Token推进/ID选择。
- StandaloneTopK沿用rq3_sanity_structure_audit.py:1116的独立总收益排序、1e−9正收益阈值和ID规则，仅用严格矩阵代替旧兼容查询。
- Marconi Adaptation直接调用sota_policies.MarconiStylePolicy.select。严格沿用rq3_frozen_snapshot_evaluator.py:835起的**原始last_access_order浮点值**输入，不擅自改成LRU名次；alpha=1，FLOP增量及memory_bytes直接读取冻结metadata。决策不需要兼容矩阵，但最终统一在同一矩阵评价。其元数据是受控代理，不冒称原生完整Marconi或跨Workflow全树LRU。
- Exact OPT穷举大小≤K全部子集，K=4四组分别检查794、2517、6196、163个集合；保留所有最优集合。同分代表以较小集合、ID字典序选取，不能把代表集合不同误报为性能差异。
- K_min按达到全候选最大前沿所需最少项数计算，四组均为4；K1–3限制逻辑前沿保留，K4足够达到全候选前沿。本批全候选前沿恰等于各T，不将此性质推广到所有负载。没有物理池压力主张。

## 完整共同成本表

单位modeled ms；完整112行Checkpoint IDs、F/J、Exact差距、每Pending的T/E/G/成本见snapshot_budget_policy.csv和results.json。

| 条件 | FlowState | Exact OPT | PCB | LDR | WFC | StandaloneTopK | Marconi Adaptation |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

## 全部胜平负

从FlowState视角，W表示基线J更大，L表示基线J更小；严格差异阈值1e−9ms。不同集合或顺序而共同目标持平仍记T。

| 对比对象 | W | T | L |
|---|---:|---:|---:|
'''+summary+'''

下面每个括号为“对比对象J−FlowState J”，正值支持FlowState。包含全部严格负例和持平条件，没有只汇总均值。

| 条件 | Exact OPT | PCB | LDR | WFC | StandaloneTopK | Marconi Adaptation |
|---|---|---|---|---|---|---|
'''+ '\n'.join(case_table)+'''

FlowState全部非最优案例：

'''+loss_text+'''

全部同收益异集合标识、Exact全部最优集合、每次选择顺序均保存在结果JSON；all_comparisons.json列出所有配对差异。不能将Exact的不同同分代表或策略返回ID排序误认成分配优势。

## 机制归因

1. **静态重复覆盖**：g001/K4、g003/K3–4、g004/K3–4的五个胜例来自静态TopK保留已被更深候选覆盖的同链Checkpoint，动态边际将其归零并将预算用于其他Pending。逐候选边际与去除零贡献项证据见mechanism_attribution.md。
2. **覆盖数优先偏差**：g003/K1–4是对WFC的全部四个胜例。首步WFC选共享B1(q=4，推进9728 Token)，FlowState选深层A5(q=1，推进10816 Token)，共同Phi下收益420.732039636对365.672296502。后续WFC不替换先选祖先，持续占用预算。
3. **共享边际真实参与计算**：B1对四个Pending的共享收益随着深层项被选入而下降；g003下降序列365.672296502→274.107221141→182.567061243→91.121580107。其他三组先选共享B1，再对各请求剩余收益分配。该共享关系来自准入矩阵，非虚构Workflow身份。
4. **Greedy失败**：g001/K4、g002/K3–4、g004/K4保留了早选共享祖先，无交换机制，落后Exact/PCB/LDR。这不是实现错误，不在本轮修改算法。g002/K3的最优集合甚至无需先保留共享祖先。
5. **位置感知独特收益尚未隔离**：g003的WFC差异仅覆盖数优先就足以解释，连总Token推进也优先A5；本轮没有Gap-only消融，不能把胜例归因于位置感知项的独立贡献。

## 验证与证据边界

5项CPU测试通过：正式范围排除g000/005/006；Decimal独立成本计算与全部策略约束；拒绝边为0且跨Workflow准入未过滤；每步边际与max-benefit公式一致；Marconi原始recency归一化排序独立核对。所有大小≤4集合均验证F/J等价，Greedy每步收益与独立重算核对；Exact覆盖完整可行集合。

复现：项目根下运行 `/home/wjg/uv312/bin/python -B -m evaluation.strict_compatibility_b4c.evaluate`、同路径模块`test_evaluate`（unittest）及`report`。原文件哈希见input_integrity.json，交付摘要见evidence_manifest.json。本轮没有GPU、Docker、SGLang、回放或新的Runtime Snapshot。

## 是否值得进入严格Runtime Policy Comparison

**有条件值得，但本轮不执行。** 四个预注册新快照已经出现共同目标上的可解释严格差异，能检验动态去冗余与覆盖数优先的实际后果；也同时提供Greedy失败和持平条件。后续若获授权，应保留同一四快照全部K和既定策略，不只运行g003或正差条件，并验证每个保留集合的安全实现、输出与控制开销。

目前只能证明该受控小样本严格矩阵上的模型化分配差异，不能证明GPU TTFT、控制开销后净收益、总体优势或物理内存紧张。候选准入不等于逐边恢复认证，真实Runtime选集实现仍是下一阶段门禁。没有必要为追求更好结果追加样本或重选结构；完成本轮后停止。
'''
(OUT/'final_report.md').write_text(report)
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}
(OUT/'evidence_manifest.json').write_text(json.dumps({'status':'PASS','files':manifest},ensure_ascii=False,indent=2)+'\n')
print(summary)
