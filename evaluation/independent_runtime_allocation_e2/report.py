"""依据全部结果生成中文归因，不按策略胜负筛选实例。"""
from .evaluate import OUT,read,save,EPS
import itertools

def main():
 a=read(OUT/'aggregate_results.json');rs=a['results'];policies=['FlowState-Exact','FlowState-Greedy','StandaloneTopK','PCB','LDR','WFC'];differences=0;wins={p:0 for p in policies};ties={p:0 for p in policies};conditions=0
 lines=['# 工作负载结构与机制归因','', '全部四组均按原顺序保留。以下成本单位为modeled ms，非实测GPU耗时。','']
 summary=['| 组 | 候选 | 准入/拒绝/未知 | 共享候选 | K_min | 结果 |','|---|---:|---|---:|---:|---|']
 for m in a['materialization']:summary.append(f"| {m['group']} | {m['candidates']} | {m['admissible']}/{m['rejected']}/{m['unknown']} | {m['shared_candidates']} | {m['K_min']} | {m['status']} |")
 for group,r in rs.items():
  s=read(OUT/group/'strict_epoch_snapshot.json');positions={c['checkpoint_id']:c['token_position_L'] for c in s['candidates']};labels={p['continuation_id']:p['label'] for p in s['pendings']}
  lines+=['## '+group,'',f"K_min={r['K_min']}；正收益支持层次化={r['laminar']}。所有{r['n']}个候选均保留，未进行支配剪枝。",'','| 候选 | L | 正收益支持 |','|---|---:|---|']
  for c,support in r['supports'].items():lines.append(f"| {c} | {positions[c]} | {','.join(labels[p] for p in support)} |")
  lines+=['','| K | Exact | Greedy | StandaloneTopK | PCB | LDR | WFC |','|---|---:|---:|---:|---:|---:|---:|']
  for k in range(1,5):
   rows={x['policy']:x for x in r['rows'] if x['K']==k}
   if not rows:continue
   conditions+=1;values=[rows[p]['J_ms'] for p in policies if p in rows]
   differences+=max(values)-min(values)>EPS
   lines.append('| '+str(k)+' | '+' | '.join(f"{rows[p]['J_ms']:.6f}" if p in rows else 'NOT_TRACTABLE' for p in policies)+' |')
   if 'FlowState-Exact' in rows:
    j=rows['FlowState-Exact']['J_ms']
    for p in policies:
     if p in rows:
      wins[p]+=rows[p]['J_ms']>j+EPS;ties[p]+=abs(rows[p]['J_ms']-j)<=EPS
  lines+=['','逐预算组合及冗余：','']
  for k in range(1,5):
   rows={x['policy']:x for x in r['rows'] if x['K']==k}
   if not rows:continue
   lines.append(f"- K={k}："+'；'.join(p+'='+','.join(v['selected'])+f"（J={v['J_ms']:.6f}）" for p,v in rows.items()))
   for p in ('FlowState-Greedy','StandaloneTopK','WFC'):
    v=rows[p];zero=[x['checkpoint'] for x in v['steps'] if abs(x['marginal_ms'])<=EPS]
    if zero:lines.append(f"  {p}按其选择顺序出现零边际候选：{','.join(zero)}。静态独立收益不扣除已有覆盖；最终目标仍按每个Pending的最大收益评价。")
  ex=r['exact'];nonnested=[];mandatory=[]
  for x,y in zip(ex,ex[1:]):
   if 'optimal_sets' not in x or 'optimal_sets' not in y:continue
   canonical=lambda z:min(z['optimal_sets'],key=lambda v:(len(v),tuple(v)))
   if not set(canonical(x))<=set(canonical(y)):nonnested.append([x['K'],y['K']])
   if not any(set(u)<=set(v) for u in x['optimal_sets'] for v in y['optimal_sets']):mandatory.append([x['K'],y['K']])
  lines+=['',f'代表Exact集合不嵌套的相邻预算：{nonnested}；枚举全部同分最优集合后，仍不存在嵌套最优延伸的预算：{mandatory}。后者说明不可撤回的Greedy选择存在结构性限制，不能用不同集合本身宣称性能差异。','']
  lines.append('当K<K_min时不能保留达到全部候选最佳前沿的完整集合，存在有效逻辑取舍；K≥K_min时Exact可达到该前沿，但Greedy先前选择可能继续占用预算。共享祖先和专属/子集深层候选的收益重叠，导致初始覆盖优势与最终组合收益不一致。具体选集、边际值及各Pending E/G见allocation_results.json和allocation_comparison.csv。')
 lines+=['','## 适用范围','', '层次化支持与路径上最大恢复位置符合已有条件性树DP讨论，但本轮没有实现或验证新的DP求解器。相同目标值表示在本固定输入与预算下简单策略足够；不代表一般算法数学等价。位置感知Phi始终固定，没有Gap-only消融，不能单独归因于位置项。']
 (OUT/'workload_structure_analysis.md').write_text('\n'.join(lines)+'\n')
 shared=sum((m['shared_candidates'] or 0)>0 for m in a['materialization']);gate='READY' if len(rs)==4 and all(m['status']=='READY' and m['cleanup']=='PASS' for m in a['materialization']) else 'PARTIAL'
 text=['# FlowState E2 最终报告','',f'Gate：**{gate}**。{len(rs)}/4组完成严格合同和分配比较；{shared}/4组形成真实驻留且满足EPOCH_ADMISSIBLE的跨Pending共享。','']+summary+['',f'{conditions}个Snapshot—预算条件中，{differences}个条件至少两种方法的最终J严格不同。四预算是同一组的重复条件，不作为独立工作负载样本。','','| 方法 | 达到Exact | 严格落后Exact |','|---|---:|---:|']
 for p in policies:text.append(f'| {p} | {ties[p]} | {wins[p]} |')
 text+=['','| 对比（前者相对后者） | 胜 | 平 | 负 |','|---|---:|---:|---:|']
 for left,right in [('FlowState-Exact','FlowState-Greedy'),('FlowState-Greedy','StandaloneTopK'),('FlowState-Greedy','PCB'),('FlowState-Greedy','LDR'),('FlowState-Greedy','WFC')]:
  counts=[0,0,0]
  for r in rs.values():
   for k in range(1,5):
    z={x['policy']:x['J_ms'] for x in r['rows'] if x['K']==k}
    if left not in z or right not in z:continue
    diff=z[left]-z[right];counts[0 if diff < -EPS else 2 if diff > EPS else 1]+=1
  text.append('| '+left+' / '+right+' | '+' | '.join(map(str,counts))+' |')
 text+=['','## 能证明什么','', '在四组固定真实来源受控组合中，历史请求可自然物化共享Recurrent状态，严格矩阵可支持统一模型目标下的分配比较。结果完整展示Exact、Greedy与简单规则的相同和差异；支持重叠收益与预算截断会影响集合分配质量，而非仅比较选集名称。详见workload_structure_analysis.md逐组全部成本与选集。','','## 不能证明什么','', '本轮未派发任何Pending，Native Pending匹配和逐边恢复均未测试，不能把准入升级为RESUME_VALIDATED。四组物理来源血缘均为PROVENANCE_UNVERIFIED。没有策略GPU执行、驱逐或性能配对，不能从modeled ms推断TTFT或净端到端收益。逻辑K不是HBM压力。全部组合是Trace-derived Controlled Composition，不是自然并发Agent工作负载。','','## 完整性与复现','', '协议在GPU前冻结，CPU回归复现B5-C4的24个历史策略条件。每组capture_manifest.json锁定原始采集文件；每组allocation_results.json保存支持集、Exact全部同分集合、选择步骤及原生Baseline记录。source_identity.json和execution_lock.json记录执行前来源/代码身份，gpu_cleanup.json逐组保留资源清理。旧cpu_run.log末行追加差异已在来源报告中以精确字节证明解释，未改旧数据。','','执行结束后停止；未进入E3/E4或扩充来源。']
 (OUT/'final_report.md').write_text('\n'.join(text)+'\n')
 save(OUT/'summary_statistics.json',{'gate':gate,'conditions':conditions,'different_objectives':differences,'exact_ties':ties,'worse_than_exact':wins})
 print('\n'.join(text[:24]))
if __name__=='__main__':main()
