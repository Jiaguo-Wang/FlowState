"""把已完成数值核验与冻结轨迹整理为中文报告。"""
import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def read(n):return json.loads((OUT/n).read_text())
def put(n,s):(OUT/n).write_text(s+'\n')
def short(c):return c.split('_',2)[-1]
def main():
    contracts=read('structural_contract.json');rows=read('branch_details.json');failures=read('failure_details.json');verification=read('verification.json')
    a=['# 结构合同重新确认','\n结论：四组全部满足单共享候选条件。只以B4-B1原始snapshot及B4-C冻结results.json确认，不以B5-B0结论替代检查。','\n## 权威来源和检查方式','\n仓库根 `/home/wjg/code/FlowState`。快照根为 `evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/`；对应g001–g004的snapshot.json、summary.json及manifest.json。结果根 `evaluation/strict_compatibility_b4c/`，使用results.json的benefit_matrix_ms、input、rows和traces，核验evidence_manifest.json。原始清单全部通过，路径及SHA见verification.json。','\n不导入生产Optimizer，不重新执行B4-C或Baseline。当前optimizer与B4-C历史版本的变化不进入本次数值核验。没有使用g000、g005/g006或旧573条件。','\n|条件|实际核验|','|---|---|','|唯一共享|按严格正收益b>0统计支持，不用Workflow ID推断；每组仅B1支持4请求|','|其余候选|均支持恰好1个Pending，没有第二个共享候选|','|非负固定收益|全部条目有限且≥0；拒绝边严格为0|','|等大小|每组所有候选memory_bytes=51,511,296；K是逻辑项数而非HBM压力|','|逐请求最大收益|对两个分支实际选集直接按Σmax计算F，与分支公式一致；逐轮日志边际也独立复核|','|同Epoch与Phi|候选/请求barrier_id与configuration_digest一致；位置/T与冻结输入一致；全部收益用冻结参数Decimal算术核对，误差≤1e−9|',
    '\nPhi只用于核验既有值，没有拟合或改变矩阵。参数为a=37.82815、b=0.345974143、c=−0.156201917，g=G/1024、t=T/1024，Phi=a g+b g t+c g²；冻结模型名position_aware_quadratic_v1。','\n## 逐快照合同与q/r']
    for c in contracts:
        a+=['\n### '+c['group'],f"\n{c['candidate_count']}候选、{c['pending_count']}Pending；共享候选{c['shared']}。快照SHA：`{c['snapshot_sha256']}`；Barrier：`{c['barrier_id']}`。",'\n|Pending|专属最佳代表|q（ms）|r（ms）|max(0,q−r)（ms）|','|---|---|---:|---:|---:|']
        for q in c['qr']:a.append(f"|{short(q['pending'])}|{short(q['best_exclusive'])}|{q['q']:.12f}|{q['r']:.12f}|{q['residual']:.12f}|")
    a+=['\n四组K_min=4：各请求最大前沿对应不同的唯一专属最深项，共享B1不能达到这些最深前沿；四项既必要又充分。本批J(C)=0，不推广到其他负载。','\n准入边是EPOCH_ADMISSIBLE；本轮不追加逐边恢复认证。没有改变冻结收益、候选身份或严格关系。']
    put('structural_assumptions.md','\n'.join(a))
    f=['# 四个Greedy失败条件的证据归因','\n归因完整：冻结results.json提供每步prefix、chosen和全部marginals_ms。核验prefix与最终选择顺序一致、每个已记录边际等于冻结矩阵的F(S∪{c})−F(S)，不猜测缺失轨迹。完整原始步骤复制在failure_details.json。','\nExact是无序集合，没有“Exact第一步”的定义。下面“首次偏离”指Greedy第一次选入不属于任何冻结最优集合的候选，并且该分支最优值已经严格低于全局最优。四个条件均发生在第1步B1，不是错误地给穷举解附加时间顺序。']
    for x in failures:
        r=x['branches'];f+=['\n## '+x['group']+'/K'+str(x['K']),f"\n首步B1收益{x['shared_first_gain']:.12f}，最大专属独立收益{x['best_exclusive_first_gain']:.12f}：B1是当前单步最优，但不属于任何全局最优集合。保留B1分支成本J1={r['J1_ms']:.12f}；不保留分支J0={r['J0_ms']:.12f}。Greedy在已固定B1的分支内达到最优，错误来自不可撤销的分支选择，而不是后续专属排序没有求好该分支。",'\n|步骤|冻结选择|选择时边际（ms）|','|---|---|---:|']
        for i,s in enumerate(x['frozen_steps'],1):f.append(f"|{i}|{short(s['chosen'])}|{s['marginals_ms'][s['chosen']]:.12f}|")
        f += [f"\n冻结Exact全部最优集合：`{x['exact_sets']}`。PCB处于F0并达到Exact；WFC处于F1，与Greedy成本相同。",'\n|Pending|Greedy E/G|Exact E/G|Greedy成本|Exact成本|额外成本（ms）|','|---|---|---|---:|---:|---:|']
        for p in x['per_pending_cost_difference']:f.append(f"|{short(p['pending'])}|{p['greedy_E']}/{p['greedy_G']}|{p['exact_E']}/{p['exact_G']}|{p['greedy_cost_ms']:.12f}|{p['exact_cost_ms']:.12f}|{p['extra_cost_ms']:+.12f}|")
        f.append(f"\n总额外成本：{r['FlowState_gap_to_exact_ms']:.12f} modeled ms。")
        if x['group']=='g002' and x['K']==3:f.append('\ng002/K3不是简单的一个请求遗漏：Greedy保留B1让B获益91.081715366，但因预算占用没有保留D4，D额外付出140.098526391；净损失49.016811025。Exact选择A4/C4/D4，主动让B从0恢复。共享候选有真实正贡献，仍可能不值得占用一个槽。')
        else:f.append('\nK4时四个专属最深项可使成本为0。Greedy为B1占用一个槽，剩余只能保留三个最深项；最后未被推进的请求仍从2432恢复。B1并非最终零收益冗余：它仍服务该请求，但收益不及被挤出的最深项。')
    f+=['\n## 同分及推论边界','\nA/D或B/D剩余收益同分时，冻结Greedy按既有ID规则选代表；换同收益代表只会改变哪条请求留下缺口，不消除这四个条件的目标差距。没有调整同分规则来修复Greedy。','\n本批PCB所有16条件均达到F0分支最优，WFC所有16条件均达到F1分支最优；这是冻结数值观察，不是WFC在任意单共享实例中按Token推进都等价于Phi排序的定理。两分支本身可精确求解，不意味着Greedy知道应选哪个分支。','\n这些是模型化恢复成本的归因，不是GPU TTFT、物理内存竞争或位置感知Phi项独立消融证据。']
    put('greedy_failure_analysis.md','\n'.join(f))
    z=['# Phase B5-C0 最终报告','\n## 最终门禁：READY','\n四组结构合同全部满足；两分支精确性命题在明确假设下成立；16/16条件与冻结Exact目标一致；四个Greedy失败均有原始逐轮日志及逐请求成本证据。没有修改生产算法或冻结产物。',f"\n最大绝对目标误差：{verification['maximum_absolute_error_ms']:.16g} modeled ms，低于冻结比较容差1e−9 ms。浮点减法产生的约1e−13偏差未用于改变胜负。",'\n## 核心结论','\n仅一个共享候选且其余候选正收益支持互斥时，固定是否保留共享项后，额外收益可分。无共享分支按q取至多K项；有共享分支先计Σr，再按max(0,q−r)取至多K−1项。两个分支穷尽可行域，所以取较大收益是全局最优。K=0单独返回空集。','\n该证明不需要专属候选必须形成Token链；正收益支持结构已经足够。完整假设、空集、零收益、饱和、同分及复杂度见exact_decomposition_proof.md。多共享候选或物理耦合选择约束不能直接沿用本命题。','\n## 16个条件的分支结果','\n单位modeled ms；J越低越好。完整F0/F1、两分支代表集合、各策略集合与差距见branch_verification.csv。','\n|条件|不含B1的J0|含B1的J1|最优分支|冻结Exact J|Greedy额外成本|','|---|---:|---:|---|---:|---:|']
    for r in rows:z.append(f"|{r['group']}/K{r['K']}|{r['J0_ms']:.9f}|{r['J1_ms']:.9f}|{r['winning_branch']}|{r['frozen_exact_J_ms']:.9f}|{r['FlowState_gap_to_exact_ms']:.9f}|")
    z+=['\nF0、F1各获胜8个条件，没有分支同分。PCB在本批对应F0分支最优，WFC对应F1分支最优，因此各8次达到Exact并互补覆盖16条件。不是事后构建可部署的PCB/WFC组合策略，也不声称WFC具有一般分支最优保证。','\n## Greedy失败的精确含义','\ng001/K4、g002/K3、g002/K4、g004/K4均在首步选择B1。单步共享总收益更大，但这四个条件下整个含B1分支的最优值低于不含B1分支。Greedy随后已求得含B1分支最优，仍无法撤销首步占用，因此分别多付68.218394974、49.016811025、82.987847211、48.608817340 modeled ms。','\ng002/K3还表明失败不依赖预算饱和：K=3<K_min=4时，放弃共享B1并保留A4/C4/D4更好。K4三个失败是未给第四个请求保留最深项。完整E/G与净损失分解见greedy_failure_analysis.md。','\n## 证据和执行边界','\n脚本verify.py仅使用Python标准库读取冻结收益矩阵，直接计算题设两分支公式；未调用生产Optimizer、任何Baseline或新的穷举收益求解器。Decimal计算只校验既有Phi收益条目，没有重拟合或生成新收益矩阵。没有算法性能计时。','\n报告输入为四个Snapshot的16个相关预算条件，不是16个独立工作负载。CPU精确化简仅证明特殊结构，不意味着所有Workflow可精确Top-K，也不构成Runtime加速或SOTA优越性证据。','\n可复核入口：`/home/wjg/uv312/bin/python -B -m evaluation.structural_exactness_gate_b5c0.verify`。原始清单、来源SHA和数值检查见verification.json；生成报告由report.py完成。','\n已交付structural_assumptions.md、exact_decomposition_proof.md、branch_verification.csv、greedy_failure_analysis.md、final_report.md及独立复核脚本/明细。本轮完成后停止，不进入下一Phase。']
    put('final_report.md','\n'.join(z))
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}
    put('evidence_manifest.json',json.dumps({'files':files,'说明':'仅本轮独立分析产物'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
