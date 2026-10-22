"""依据本轮真实运行、前后状态与清理记录生成结论，不填造缺失证据。"""
import csv,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def read(name):return json.loads((P/name).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,text):(P/name).write_text(text.strip()+'\n')
def main():
 source=read('source_identity.json');unchanged=all(sha(Path(p))==h for p,h in source['sha256'].items())
 cleanup=read('gpu_cleanup.json');engine=read('engine_cleanup.json') if (P/'engine_cleanup.json').exists() else {}
 process=read('process_result.json') if (P/'process_result.json').exists() else {}
 failure=read('failure.json') if (P/'failure.json').exists() else read('launch_failure.json') if (P/'launch_failure.json').exists() else None
 gate='BLOCKED'
 if failure is None and (P/'result.json').exists():
  result=read('result.json');integrity=read('state_integrity.json');comparison=read('shadow_plan_comparison.json');snapshot=read('live_epoch_snapshot.json')
  assert all(integrity['checks'].values()) and comparison['oracle']=='PASS'
  assert result['pending_dispatched']==0 and result['history_completed']==8
  gate='READY' if result['status']=='READY' and unchanged and cleanup['status']=='PASS' and engine.get('shutdown_returned') and process.get('returncode')==0 else 'BLOCKED'
  counts={x:sum(e['status']==x for e in snapshot['edges']) for x in ['EPOCH_ADMISSIBLE','REJECTED','UNKNOWN']}
  table='|求解器|选集|J（模型化ms）|最优性认证|\n|---|---|---:|---|\n'
  fronts=[]
  for row in comparison['plans']:
   p=row['plan'];table+=f"|{row['solver']}|{','.join(p['selected_ids'])}|{p['recovery_cost_ms']:.9f}|{row['optimality_certified']}|\n"
   fronts.append({'solver':row['solver'],'pending':p['pending']})
  times=list(csv.DictReader((P/'shadow_timing.csv').open()));tt='|阶段|归属|实测ms|\n|---|---|---:|\n'
  for t in times:tt+=f"|{t['stage']}|{t['solver']}|{float(t['ms']):.3f}|\n"
  write('state_integrity_report.md',f'''# Shadow前后状态完整性

本轮前观察窗口 `{integrity['before_window']}`，后观察窗口 `{integrity['after_window']}`。每次RPC各自在Scheduler单次同步安全窗口采集；两次之间CPU规划不持有跨RPC租约。

完整PhysicalView比较仅排除必然不同的barrier_id和lease_id标签，engine_id、配置、节点、Token、Slot、组件、空闲池、FA/非Recurrent摘要等全部保留比较。另独立比较semantic替换/引用元数据、component_slices、Conv/Temporal内容SHA、Native Match、COW及eviction事件记录，结果：

{json.dumps(integrity['checks'],ensure_ascii=False,indent=2)}

物理证据：physical_state_census.json与post_shadow_census.json；时间边界：state_integrity.json。八条历史执行期间存在原生缓存及COW正常副作用，但Shadow期间相关事件列表未新增。四个Pending未执行。

两端视图一致加上独占Engine、只读控制入口、CPU规划不持有Runtime引用，支持本次Shadow无观测到的状态变更；不推广为永久Pin/Lease或未来应用仍新鲜。源码及原始证据摘要不变：{unchanged}。GPU清理：{cleanup['status']}。
''')
  write('final_report.md',f'''# Phase B5-D2 最终报告

**Gate：{gate}**。

## 当前真实Epoch

一个新Engine生命周期完成固定八条历史回放；四个Controlled Pending均未提交。当前候选{len(snapshot['candidates'])}，Pending{len(snapshot['pendings'])}，完整配对{len(snapshot['edges'])}：{counts}。配置、实际接收Token与冻结B5-C2输入核验通过。Node/Slot均来自当前观察，未复用历史绑定。保持Trace-derived Controlled Composition语义。

主要且唯一规划预算K=2。新Epoch身份：`{snapshot['epoch_id']}`。完整StrictAllocationEpoch摘要、来源字节SHA和两份原StrictAllocationPlan见shadow_plan_comparison.json。两份计划通过既有validate_plan；无PhysicalBinding。

{table}

独立Oracle直接读取当前Token位置及严格边，用Decimal冻结Phi穷举所有大小≤2子集，目标与代表集合一致，见current_epoch_oracle.json。没有以历史B5-C4目标替代本次认证。详细每Pending的T/E/G及成本均保存在两份Plan中。

## 分段耗时

{tt}

这是一次实际调用的耗时，不是线上P95或重复统计。共同预处理只执行一次；观察RPC含GPU同步、摘要、传输与解码，Epoch构建认证含序列化、来源整理、严格验证及快照写出。CPU规划在前观察RPC返回后、后观察RPC之前执行，不占Scheduler连续控制窗口。Oracle为独立审计开销，不归入任一求解器。两个求解器连续运行的总时间不能作为单个规划器独立耗时；需要计算单个逻辑路径时仅加共同段和对应solve/plan两段。

## 无副作用与清理

两次完整物理Census、内容摘要及Match/COW/Eviction列表核验全部通过；候选和FA不变，四个Pending未派发。源文件摘要不变={unchanged}，Engine shutdown返回={engine.get('shutdown_returned')}，进程返回码={process.get('returncode')}，GPU清理={cleanup['status']}。专用容器移除及连续资源检查见gpu_cleanup.json；未终止其他用户进程。

退出日志中若存在CUDA IPC释放警告，保留runtime.log原文；清理判断以实际进程、容器和显存检查为准，不仅依据退出码。实际警告摘录见runtime_integration_risks.md。

本次证明新真实观察可进入纯规划合同，不证明未来应用时状态新鲜、Reconciliation安全、请求恢复正确或GPU净收益。EPOCH_ADMISSIBLE不是RESUME_VALIDATED。未调用驱逐、恢复或保留策略，不进入下一阶段。
''')
 else:
  write('final_report.md',f"# Phase B5-D2\n\n**BLOCKED**。\n\n原始失败：{failure}。\n\n清理：{cleanup['status']}；不重试、不更换输入，不将CPU预检查冒称Runtime成功。")
  write('state_integrity_report.md','# 状态完整性\n\nBLOCKED：真实运行未取得全部必要证据，见failure.json及runtime.log。')
 log=(P/'runtime.log').read_text(errors='replace')
 warnings=[line for line in log.splitlines() if 'CudaIPCTypes' in line or 'shared CUDA' in line]
 write('runtime_integration_risks.md',f'''# Runtime集成边界

1. 两次只读观察窗口是独立Scheduler同步调用，CPU规划在窗口外，不享有持续Admission隔离或租约。本轮独占Engine没有其他请求，实际前后状态一致；未来应用计划必须重新观察、Bind和Preflight。
2. D1只接受frozen_snapshot证据范围。本轮新Runtime观察在当前边界即时序列化成不可变证据，再交给原D1，不伪造current_runtime认证，不修改Strict语义。候选来源采用本生命周期首次Census与真实历史Session，不据此推断独占物理Owner。
3. 本次只能认证Epoch内准入及规划，没有Pending实际恢复、逐边RESUME_VALIDATED或策略性能证据。历史回放正常COW和缓存变化不属于Shadow无副作用期间。
4. Exact遵守n≤12、m≤4、K≤4、子集≤794；本轮只K2。缺失或UNKNOWN输入Fail-closed，超限明确拒绝，不调整候选或扩大范围。无硬deadline承诺。
5. 当前源文件与模型摘要检查结果：{unchanged}；GPU清理：{cleanup['status']}。实际CUDA IPC退出警告：{warnings if warnings else '未发现该类警告'}。保留原始日志，不能以警告或进程返回码单独替代资源清理证据。

本轮不修改生产Optimizer、Scheduler或StateController；复用C3已验证的专用观察进程入口，没有新增缓存动作。完成后停止。
''')
 (P/'validation.json').write_text(json.dumps({'gate':gate,'frozen_unchanged':unchanged,'gpu_cleanup':cleanup['status'],'engine_cleanup':engine,'process':process},ensure_ascii=False,indent=2)+'\n')
 (P/'evidence_manifest.json').write_text(json.dumps({'files':{str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and 'triton_cache' not in p.parts and '__pycache__' not in p.parts and p.name!='evidence_manifest.json'}},ensure_ascii=False,indent=2)+'\n')
 print('最终门禁',gate)
if __name__=='__main__':main()
