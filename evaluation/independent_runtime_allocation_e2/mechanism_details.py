"""从逐条件结果补充可复核的具体机制归因与证据等级。"""
from .evaluate import OUT,read,save,EPS

def main():
 a=read(OUT/'aggregate_results.json');details=[];lines=['','## 具体机制证据补充','']
 for group,r in a['results'].items():
  rows={(x['K'],x['policy']):x for x in r['rows']};by=r['benefits']
  dom=[]
  for x,v in by.items():
   for y,w in by.items():
    if x!=y and r['supports'][x]==r['supports'][y] and all(i>=j-EPS for i,j in zip(v,w)) and any(i>j+EPS for i,j in zip(v,w)):dom.append([x,y])
  g2=rows[2,'FlowState-Greedy'];e2=rows[2,'FlowState-Exact'];g3=rows[3,'FlowState-Greedy'];j0=g3['plan']['recovery_cost_ms']+g3['plan']['benefit_ms']
  removal={c:j0-sum(max([by[t][j] for t in g3['selected'] if t!=c]+[0.]) for j in range(4))-g3['J_ms'] for c in g3['selected']}
  details.append({'group':group,'equal_support_dominance':dom,'greedy_K3_removal_cost_ms':removal,'greedy_K2_excess_ms':g2['J_ms']-e2['J_ms']})
  lines.append(f"- {group}：K2 Greedy的J={g2['J_ms']:.6f}，Exact={e2['J_ms']:.6f}，差额{g2['J_ms']-e2['J_ms']:.6f} modeled ms。Exact同时保留AB/CD两个子域状态；Greedy保留公共祖先及一个子域状态。K3 Greedy选集中删除公共祖先B5C3_NODE_20的目标增量为{removal.get('B5C3_NODE_20',float('nan')):.9f}，说明该项此时已被更深覆盖替代，但不可回退的Greedy仍为其占用预算。")
  if dom:lines.append('  相同支持集合下的严格支配（支配者→被支配者）：'+str(dom)+'；主比较仍保留全部候选。')
 lines+=['','w3的StandaloneTopK在K3选择已被更深AB状态替代的Node18，边际为0，J=103.251492；Greedy转选CD状态，J=25.559190。K4两者J分别25.559190与10.909073。w4在K4同样出现Node18静态冗余，StandaloneTopK J=45.066824，Greedy J=25.521474。w1/w2未出现改变最终结果的静态排序冗余，Greedy与StandaloneTopK全部预算持平。','','WFC与Greedy在16/16条件均得到相同选集和E/G。该观察不能证明动态恢复收益评分相对WFC有独立优势，也不证明两规则一般等价。四组K1均以公共共享状态达到Exact；K4均由PCB/LDR的最深需求覆盖达到Exact。K2/K3共8个条件，Exact严格优于本轮全部五种其他规则。','','四组K_min=4：K1–K3共12个条件存在逻辑预算截断，K4无需复杂联合求解也可由PCB/LDR达成全候选最佳前沿。w1/w2此时四个G均为0；w3/w4仍有未被任何候选覆盖的Gap（分别对应B/D的15/15 Token、B的29 Token），因而最优J仍为1.142854、1.105138。不能把K_min等同于零Gap。','','每组物理池均为28槽，7槽驻留、21槽空闲，无本轮逻辑预算引起的物理驱逐。真实共享与逻辑紧预算已经建立；真实HBM不足或在线容量压力没有建立。']
 save(OUT/'mechanism_details.json',details)
 with (OUT/'workload_structure_analysis.md').open('a') as f:f.write('\n'.join(lines)+'\n')
 with (OUT/'final_report.md').open('a') as f:f.write('\n## 结构解释与简单规则适用条件\n\n四组K1时公共共享状态已经最优；K2/K3共8个条件Exact严格优于本轮全部其他规则。Greedy首步保留公共祖先，此后不能撤回，在K3该祖先已变为零删除贡献却仍占预算。w3/K3、w3/K4、w4/K4进一步出现静态TopK重复覆盖，动态更新可以避免该冗余；但Greedy与WFC始终持平。K4时四组PCB/LDR均达到Exact，说明预算充足时简单方法已经足够。\n\n四组均28槽、7驻留、21空闲，未发生策略驱逐。全部32条历史请求完成，16个Pending均未提交；清理后各GPU检查无计算进程、显存14 MiB。日志中的退出CUDA IPC告警原样保留，未据此省略独立容器/GPU清理检查。\n')
if __name__=='__main__':main()
