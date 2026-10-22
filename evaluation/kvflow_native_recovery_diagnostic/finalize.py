"""汇总已执行证据，不补造缺失结果，不启动Runtime。"""
import hashlib,json,time
from pathlib import Path
from evaluation.exact_reconciliation_correctness_gate_b5d3.protocol import compare_outputs
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[];cleanup=read(OUT/'gpu_cleanup.json');cold=read(OUT/'cold/cold/output.json')
 for arm in ['probe','evict','cold']:
  p=OUT/arm/('cold' if arm=='cold' else 'strict');s=read(p/'summary.json');m=s['native_match'];post=read(p/'post_request.json')
  assert s['status']=='PASS' and cleanup[arm]['status']=='PASS'
  row={'arm':arm,'H':m['physical_fa_hit'],'E':m['executable_prefix'],'G':m['replay_gap'],'node':m['best_match_node'],'source_slots':m['cow_source_slots'],'working_slots':m['working_slots'],'input_length':m['input_token_count'],'execution':post['trace']['execution'],'copies':post['trace']['copies'],'output_comparison':compare_outputs(read(p/'output.json'),cold)}
  if arm!='cold':
   r=read(p/'reconciliation.json');assert r['ok'];row['operations']=r['operations'];row['plan']=r['plan'];row['fa_content_postcheck']=r['fa_content_postcheck']
   row['physical_before']=[{'node':n['node_id'],'L':len(n['token_ids']),'slots':n['slots']} for n in r['before']['nodes'] if n['slots']]
   row['physical_after']=[{'node':n['node_id'],'L':len(n['token_ids']),'slots':n['slots']} for n in r['after']['nodes'] if n['slots']]
   row['free_before']=len(r['before']['free_slots']);row['free_after']=len(r['after']['free_slots'])
  rows.append(row)
 assert rows[0]['E']==4096 and rows[1]['E']==4032 and rows[2]['E']==0
 assert not rows[0]['operations'] and len(rows[1]['operations'])==1 and rows[1]['operations'][0]['returned']
 lock=read(OUT/'frozen_input_lock.json')
 # 冻结文件摘要的键结构原样保存，另核对完整来源锁。
 source=read(OUT/'source_identity.json')
 for p,h in source['sha256'].items():assert sha(Path(p))==h,p
 result={'gate':'PARTIAL','runtime_closure':'PASS','rows':rows,'cleanup':cleanup,'frozen_sources_unchanged':True,'blocker':'原CPU候选域仅包含10个末端，没有对10个可执行祖先规定共同资源计量和保留/驱逐合同；不能直接复用为正式GPU Baseline。','finished_unix':time.time()}
 (OUT/'diagnostic_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 report='''# KVFlow Synthetic Native Recovery 最小诊断

结论：**PARTIAL**。Agent00 的真实恢复与一次驱逐反事实通过；唯一剩余阻碍是原冻结 CPU 候选域没有纳入可执行祖先的资源与控制合同。

1. **4096 末端能够提供合法 Native Resume。** 原 `WARM_1_00` 只有4096个输入Token，源码 `schedule_batch_source.py:1281、1310、1367–1372` 将匹配上限设为输入长度减一。其 H=4095/E=0 发生在第二次预热匹配时；4032状态是在该次执行后观察到。它不是后续请求的恢复结论。新生命周期提交原冻结 `INVOKE_0_00` 的4128个Token，Native实际 H/E/G=4096/4096/0，源Node18/Slot3，COW到工作Slot7，实际Prefill含32个新Token。Node/Slot来自本轮观测，不是历史绑定。

2. **4032祖先会改变恢复和物理预算。** 原完整20次预热后的Census有10个4096末端及10个4032祖先，占用20个不同槽，组件切片无别名，28槽池空闲8槽。新Agent00生命周期中末端Node18/Slot3与祖先Node20/Slot6分别占槽。只驱逐末端后，祖先仍提供E=4032，而非E=0。其独立槽与组件身份、无引用及原生释放接口已核查；本轮实际操作仅测试末端释放，没有再做祖先释放实验。

3. **一次实际驱逐→恢复→Cold对照通过。** 独立驱逐臂通过原生产Strict绑定、Preflight与逐项Postcheck；仅一次Recurrent-only操作，驻留从2变1、空闲从26变27。完整FA索引/内容在协调前后不变。随后实际H/E/G=4096/4032/64，Prefill为96 Token（64重算+32新输入），Native COW从祖先Slot6到工作Slot7。保留组件内容及非重算FA区域通过检查；请求正常新增状态与历史破坏分开检查。未驱逐臂和驱逐臂均与第三个干净Engine的Cold结果逐Token一致，均输出8 Token，停止行为一致、无回撤。全部生命周期清理通过。

4. **当前冻结工作负载不能直接用于正式GPU Baseline比较。** CPU的10个逻辑末端不能覆盖Runtime实际20个可恢复状态。若K=3只保留3个末端而任意留下10个祖先，就不是仅3个驻留槽，且未选Agent仍可从4032恢复。后续必须先明确覆盖全部真实状态的统一候选/预算/驱逐合同，再设计正式比较；本轮没有修改CPU结果或扩大实现。单Agent闭环成功不认证十Agent所有策略。

## 原始证据

- `historical_mapping.json`：冻结20状态占用、独立组件及Agent00预热时序。
- `probe/strict/reconciliation.json`、`summary.json`、`post_request.json`：只读绑定、直接Native Match、COW及执行。
- `evict/strict/reconciliation.json`、`summary.json`、`post_request.json`：真实一次驱逐及祖先恢复。
- `cold/cold/output.json`、`output_correctness.json`：独立Cold与精确输出比较。
- `diagnostic_results.json`：实际身份、H/E/G、逐事件及池记账。
- `gpu_cleanup.json`：各臂清理；`source_identity.json`和`implementation_lock.json`：输入及实现身份。

G采用H−E（本例Planning Target=H）。Cold缓存为空的H/E/G=0/0/0不表示没有计算，它执行完整4128 Token Prefill。请求执行后允许正常缓存插入和访问元数据变化；没有要求全树在请求之后完全不变。

未修改生产代码、第一阶段协议、输入或CPU结果；不作性能主张，不将本实验视为自然Agent工作流或跨Workflow物理血缘认证。完成后停止。
'''
 (OUT/'final_report.md').write_text(report)
 (OUT/'evidence_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='evidence_manifest.json' and 'triton_cache' not in str(p)},ensure_ascii=False,indent=2))
 print('PARTIAL：真实恢复及一次驱逐闭环通过；正式Baseline仍缺完整候选预算合同')
if __name__=='__main__':main()
