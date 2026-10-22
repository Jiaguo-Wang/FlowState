"""生成完整中文报告，保留替换失败、原策略持平及旧证据限制。"""
import csv,json,statistics
from .audit import OUT,read,save,EPS,POLICIES

def main():
 x=read(OUT/'results.json');rows=x['rows'];traps=read(OUT/'local_trap_diagnostics.json');legacy=read(OUT/'legacy_summary.json');traces=list(csv.DictReader((OUT/'greedy_wfc_step_audit.csv').open()))
 main=['# E3-Step1：Greedy局部替换与联合分配结构审计','','**结论：PASS。** 此门禁表示审计、冻结回归和计时合同通过，不表示1-Swap全部达到Exact。','','## 完整结果','','5个冻结Snapshot、20个预算条件、7种策略，共140行。E2的16个条件中1-Swap有14个达到Exact；B5-C4四个条件全部达到Exact。原Greedy总共5/20最优，1-Swap提高至18/20；原15个非最优条件全部改善，其中13个完全修复、2个仍有严格差距。没有修改生产求解器或冻结输入。','','| 实例 | K | Exact | Greedy | Greedy+1-Swap | WFC | StandaloneTopK | PCB | LDR | 替换数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name in ('w1','w2','w3','w4','B5-C4'):
  for k in (1,2,3,4):
   z={r['policy']:r for r in rows if r['case']==name and r['K']==k}
   main.append('| '+name+' | '+str(k)+' | '+' | '.join(f"{z[p]['J_ms']:.6f}" for p in POLICIES)+' | '+str(z['Greedy+1-Swap']['swaps'])+' |')
 main+=['','单位均为modeled ms。完整选集、各Pending E/G、Exact Gap、逐条件计时与次数见`condition_results.csv`；替换轨迹见`swap_traces.json`。20个预算条件不是20个独立工作负载。','','## 两个不能被本固定1-Swap修复的条件','','| 实例 | K | 局部最优J | Exact差距 | 离开该局部集合至少同步替换项数 |','|---|---:|---:|---:|---:|']
 for t in traps:main.append(f"| {t['case']} | {t['K']} | {t['local_J']:.9f} | {t['exact_gap_ms']:.9f} | {t['min_simultaneous_replacements_to_improve']} |")
 main+=['','两个局部集合都是{Node21,Node22,Node24}，即保留CD共享状态及AB的两个专属深状态；Exact为{Node18,Node25,Node27}，即保留AB共享状态及CD的两个专属深状态。逐邻域枚举证明所有12个1换1邻居均不严格改善；全部同大小可行集合诊断还显示，不存在更好的2换2邻居，严格改善需要同时替换3项。这是该最终集合的性质。','','但不能推论所有1-Swap规则均失败：从原Greedy的{18,20,21}出发，先20→25，再21→27，两步均严格改善即可到达Exact。该可达性仅作事后机制诊断，未替换预注册首改善规则、未重新调参或报告另一个策略的成绩。因此需要区分“固定首改善算法失败”与“存在能成功的局部路径”。','','## Greedy与WFC为何持平','','Greedy逐轮最大化Phi恢复成本的动态下降量，容差1e-9，同分按ID；WFC不读Phi，先最大化前沿严格推进的Pending数q，再最大化总Token推进r，最后按ID。两者都动态更新前沿，但评分不同。E2的16/16条件不仅最终目标相同，逐轮所选ID及前沿亦相同；完整40步（四组各1+2+3+4步）的所有候选分数见selection_traces.json。','','四组在第一步均选择公共状态Node20：q=4，推进9728 Token，且真实Phi边际也最大。第二步选择推进较大的二请求子域状态：w1/w2为CD Node21，w3/w4为AB Node24。第三步选另一个二请求子域；其q=2且Phi边际仍高于所有专属q=1候选。第四步剩余有效竞争主要是q=1专属推进，实际最大Token推进候选也具有最大成本下降。嵌套/不相交支持和本组具体深度共同造成赢家一致，不是一般评分等价定理。','','尤其是w2/K4第4步，Node22与Node27都推进320 Token，但Phi收益分别约12.204724和12.211481，WFC按ID将22排在27前，成本排序却相反；二者均不及获胜Node25。w4的同推进AB/CD浅候选也有成本排序差异。因此“低位排序不同，但没有改变最大项”比“所有评分相同”更准确。详见逐轮表及全部分数。','','## 旧573条条件：结构最优证书','','| 来源 | 条件/快照 | 存在共享的条件 | 存在收益重叠的条件 | K<K_min | Greedy与PCB/LDR/WFC同集 |','|---|---|---:|---:|---:|---:|']
 for name,v in legacy['summary'].items():main.append(f"| {name} | {v['conditions']}/{v['snapshots']} | {v['shared_conditions']} | {v['overlap_conditions']} | {v['tight_budget']} | {v['conditions']} |")
 main+=['','OpenHands全部504条件的正收益支持按Pending可分：每个候选最多服务一个Pending。每个Pending只需其最高收益候选，按这些独立最大收益取Top-K即可精确求解；本轮直接验证冻结Greedy收益等于这一上界。虽然504条都存在同链重叠、90条存在预算不足，重叠本身并不意味着复杂联合选择必要。PCB/LDR/WFC在这些具体数据中与Greedy一致；不能据此把LDR普遍最优化。','','AgentX共69条件，66条存在共享、33条存在重叠，但全部K≥K_min，冻结Greedy已经达到全候选最佳前沿。逐Pending唯一最大前沿实现者的并集给出最小预算及匹配下界，本轮逐条复核，无需枚举最多149候选的子集。由全候选上界可知这些选集在旧模型内最优，任何严格降低J的局部替换均不存在；本轮未对旧实例运行新策略或Exact。','','以上只解释旧Formal矩阵上的573/573持平。旧矩阵不是严格Runtime认证，AgentX此前Formal-only兼容边问题未修复，也不能将本轮代数最优证书升级为真实恢复正确性。统计复用Gate1-Step1，逐项复核支持、重叠、K_min、共同Phi和成本；来源/Case ID全部保留在legacy_573_structure.csv。','','## CPU求解成本','','共同解析、身份检查、输入对象及收益矩阵构建在计时外；同一输入和J、3次预热、31次重复、每轮方法循环移位。Greedy+1-Swap包含重新执行Greedy，另报从已有Greedy集合出发的Swap-only。每方法620次是20个不同条件的混合统计，不能作为某个在线请求的时延分布；逐条件值才是对应输入的比较。','','| 方法 | 均值ms | 中位ms | 范围ms |','|---|---:|---:|---|']
 for t in x['summary']:main.append(f"| {t['policy']} | {t['mean_ms']:.6f} | {t['median_ms']:.6f} | {t['min_ms']:.6f}–{t['max_ms']:.6f} |")
 main+=['','这是当前Python实现的CPU微基准，PCB/WFC等原函数附带原生诊断记录的构造成本仍包含在函数计时内。StandaloneTopK的共同独立收益矩阵在计时前已构建；Greedy/Exact使用共同直接J评分，不缓存枚举目标。没有把Exact求解与Greedy完整Runtime规划混比，也未测输入认证、Plan绑定、控制RPC或GPU成本。计时不构成生产P95、deadline或端到端收益。','','## 是否调整FlowState求解器','','建议调整求解器路线，但本轮不实施：已准入的小规模完整Strict Epoch优先保留Exact路线；本轮7候选的Exact代价处于亚毫秒量级，并有全局最优认证。Greedy+1-Swap可作为实验性非最优改进候选，不能替代Exact保证；本次只改善13/15个失败至最优，且首改善路径敏感，不能根据两个失败例临时调换遍历顺序。','','大规模或超出Exact限制时，继续明确标注Greedy的非最优Fallback性质；是否引入局部搜索需另行冻结规则和开销门禁。可分支持或已达到全候选前沿时，简单方法已经足够，没必要为了统一叙事增加复杂度。本轮未证明Greedy相对WFC的独立收益，未证明位置项的独立消融收益；不应以Exact优化同一目标就声称GPU性能提升。','','## 限制与停止','','仅5个严格快照，其中E2四个独立来源受控组合、B5-C4一个历史发现案例；多K是重复条件。EPOCH_ADMISSIBLE不等于RESUME_VALIDATED，来源血缘仍PROVENANCE_UNVERIFIED。没有修改任何生产算法、Phi或冻结输入，没有GPU、策略物理应用或E4。PASS允许保留上述负面和持平结果。']
 (OUT/'final_report.md').write_text('\n'.join(main)+'\n')
 detail=['# 差异案例与逐步证据','', 'Node编号仅为各自冻结Epoch内的简称，不跨Engine绑定。所有成本为modeled ms。','']
 swaps=read(OUT/'swap_traces.json')
 for t in swaps:
  detail+=['## '+t['case']+'/K'+str(t['K']),'',f"替换次数：{t['swaps']}；邻居评分次数（含最终停止扫描）：{t['neighbor_evaluations']}。",'']
  for i,s in enumerate(t['steps']):detail.append(f"{i+1}. 移出{s['remove']}，移入{s['add']}；J {s['before_J']:.9f} → {s['after_J']:.9f}；下降{s['gain_ms']:.9f}。")
  if not t['steps']:detail.append('无严格改善；保持原Greedy集合。')
  r=next(r for r in rows if r['case']==t['case'] and r['K']==t['K'] and r['policy']=='Greedy+1-Swap');detail.append(f"\n最终集合：{r['selected']}；J={r['J_ms']:.9f}；Exact Gap={r['gap_exact_ms']:.9f}。\n")
 detail+=['## 两个局部陷阱的完整邻域','', 'local_optimum_neighbors.csv保留w1/K3、w2/K3全部12个一换一邻居及J差值。local_trap_diagnostics.json保留从原Greedy到Exact的另一条严格改善路径；这不是新评分或事后替换正式策略结果。']
 (OUT/'difference_cases.md').write_text('\n'.join(detail)+'\n')
 select=['# Greedy/WFC逐轮最大项证据','', '所有候选的q、r、Phi边际完整保留在selection_traces.json。下表展示K4的完整路径，较小K是同一前缀；不能重复当作独立证据。','', '| 实例 | 步 | 共同赢家 | q | Token推进 | Phi边际ms | 对第二大Phi边际的优势ms | 非赢家排序逆序数 |','|---|---:|---|---:|---:|---:|---:|---:|']
 for t in traces:
  if t['K']=='4':select.append(f"| {t['case']} | {t['step']} | {t['chosen']} | {t['q']} | {t['r_tokens']} | {float(t['gain_ms']):.6f} | {float(t['cost_margin_to_second_ms']):.6f} | {t['rank_reversals_below_winner']} |")
 (OUT/'greedy_wfc_rule_audit.md').write_text('\n'.join(select)+'\n')
 print('PASS：20条件、140行、2个局部陷阱、573条结构证书；报告已生成')
if __name__=='__main__':main()
