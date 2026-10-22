"""为首臂驱逐前停止生成明确的失败报告，保留实际日志且不伪造空缺测量。"""
import csv,json
from pathlib import Path
from .finalize import sha,save,write
P=Path(__file__).resolve().parent

def main():
 response=json.loads((P/'failed_control_response.json').read_text());audit=json.loads((P/'failure_source_audit.json').read_text())
 cleanup=json.loads((P/'gpu_cleanup.json').read_text());validation=json.loads((P/'validation.json').read_text())
 assert not validation['changed'] and response['operations']==[] and not (P/'cold').exists()
 input_data=json.loads((P/'prepared_inputs.json').read_text())
 save('paired_runtime_identity.json',{'status':'INCOMPLETE_NO_VALID_PAIR','executed':['p01_Greedy'],
  'gpu':json.loads((P/'p01_Greedy_gpu_before.json').read_text()),'configuration':input_data['configuration'],
  'implementation_lock':'implementation_lock.json','source_identity':'source_identity.json',
  'history_count':8,'pair_comparability':'UNKNOWN：Exact未运行','strict_epoch':'内部认证已完成但失败响应未返回完整Epoch；不据此补造快照'})
 save('resume_source_verification.json',{'status':'NOT_EXECUTED','target':'B5C2_C_TURN_003','native_match':None,'native_cow':None,
  'source_provenance_audit':'failure_source_audit.json','reason':'Greedy臂新增来源门禁在Apply前误拒绝，未派发目标'})
 save('cold_reference_correctness.json',{'status':'NOT_EXECUTED','cold_lifecycle_started':False,'token_comparison':None,'reason':'首臂失败即停止'})
 with (P/'paired_gpu_timings.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['run','status','valid_pair','prefill_gpu_ms','ttft_ms','request_wall_ms','reason']);w.writeheader()
  w.writerow({'run':'p01_Greedy','status':'FAILED_BEFORE_APPLY','valid_pair':False,'prefill_gpu_ms':'UNKNOWN','ttft_ms':'UNKNOWN','request_wall_ms':'UNKNOWN','reason':'固定C来源门禁错误；没有提交目标请求'})
 with (P/'control_cost_breakdown.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['run','stage','ms','boundary','status']);w.writeheader()
  for r in response['timing']:w.writerow({'run':'p01_Greedy',**r,'boundary':'失败控制RPC内部已完成Host区间，可能含GPU同步；不是GPU计算耗时','status':'OBSERVED_BEFORE_FAILURE'})
  for stage in ['apply_time_validation','reconciliation','native_eviction','target_prefill','target_decode','cold_reference']:
   w.writerow({'run':'p01_Greedy','stage':stage,'ms':'UNKNOWN','boundary':'未进入该阶段，不填充测量数值','status':'NOT_EXECUTED'})
 write('reconciliation_correctness.md','''# 首臂失败与状态控制边界

**未执行Reconciliation或Pending恢复。** 八条历史请求实际完成，提交与Runtime接收Token身份核对通过，保存于p01_Greedy/history_01.json至history_08.json。最终历史Census有7个驻留候选，位置3712/2432/3456/4160/4096/3904/3520，实际Slot为3/8/10/12/14/16/18。

控制函数完成观察、Strict Epoch认证、Greedy求解、Plan验证及当前Oracle；返回的失败对象见failed_control_response.json，其中operations=[]、observations=[]。失败位于control.py:97–99，早于control.py:101开始的Apply-time重验、bind/preflight与后续reconcile_strict。因此本轮实际策略驱逐0项、正式Pending 0项，不能声称协调、Native恢复或输出正确性通过。

现有CPU门禁验证Req/Batch/计时包装，没有覆盖“公共共享状态首次出现在目标同一来源Session的历史边界”这一目标选取情形。该缺口导致实验性来源门禁错误，不是已证实的生产Compatibility、Greedy或恢复计算错误。

在同一任务允许的范围内只做离线归因，不修改冻结执行代码、不重试Engine。清理见gpu_cleanup.json，唯一生命周期PASS。
''')
 write('measurement_validity.md','''# 测量有效性

**没有有效Greedy/Exact配对，没有目标C的GPU计算、TTFT或输出对照测量。** paired_gpu_timings.csv保留失败臂并将未执行指标标为UNKNOWN；不能把空缺当作0，也不能引用历史B5-D3或模型J替代本轮测量。

control_cost_breakdown.csv只包含失败前真实完成的重型观察、Epoch认证、矩阵、Greedy求解、Plan验证和Oracle的Host区间。观察包含GPU同步和哈希；没有设备事件执行数据。当前Oracle返回J=48.228728759089606 modeled ms、29个子集，但这不是Greedy实际运行耗时，更不是目标C GPU性能。

实现中的COW后EagerRunner CUDA事件、run_batch外围事件、流式TTFT和完整控制到请求完成Wall均未进入正式目标执行。CPU替身测试通过不证明GPU计时有效。缺少首对状态可比性、恢复来源、8Token对照，故不继续重复或报告性能差值。

完整当前Epoch/Greedy选集未随失败响应持久化：ControlClient._call对ok=false提前抛异常，runner.py:37的写盘未执行。失败RPC对象从原failure.json的traceback用ast.literal_eval无执行地提取到failed_control_response.json，原日志保持不变。不能将历史预期选集冒充当前已保存计划。
''')
 write('final_report.md','''# Phase B5-D4 最终报告

**Gate：BLOCKED。有效配对0/5。** 阻断是本轮新增实验性来源门禁错误，不是已证实的FlowState恢复错误或Exact/Greedy性能结论。

## 已完成与停止位置

冻结五对交替顺序、固定Pending C、K2、原输入/配置与8Token生成协议；真实Req/Batch及计时包装CPU门禁通过。仅启动首对的Greedy独立Engine，八条冻结历史全部成功。当前观察与Strict Epoch认证、Greedy规划/验证、独立Oracle完成后，在Apply-time验证之前停止。

实际Recurrent-only策略驱逐 **0项**；正式Pending **0项**；Exact生命周期 **未启动**；Cold Reference **未启动**；后续四对 **未启动**。没有可信成对GPU耗时，不能比较性能。

## 已确认的错误原因

`control.py:97`将 `h['workflow_id'] != p['workflow_id']` 施加于两种策略所有目标来源，然后在第99行要求唯一跨Workflow来源。这是从上一阶段“选择任意跨Workflow目标”流程迁移时引入的过强条件。

本任务要求Greedy从公共共享Checkpoint恢复，同时特别要求Exact的CD共享Checkpoint来源可追溯D；并未要求Greedy的公共共享源也必须来自与C不同的Session。

本轮实际Census：Node20/2432/Slot8首次在 `B5C2_C_TURN_001` 完成后可见（history_03.json），按当前归因规则其Workflow恰与目标C相同。其Token数组确为目标C前缀，却被上述新增来源条件排除。Node27/3520/Slot18首次在D2之后可见（history_08.json），但本轮未进入Exact执行。首次观察边界本身不是精确insert事件，也不单独认证源张量由哪个请求生成；本报告不把它升级为实际恢复来源证明。

证据见failure_source_audit.json、failed_control_response.json、p01_Greedy/failure.json及p01_Greedy.runtime.log。原始失败保留；未修改门禁后重跑，也未更换请求、候选或预算。

## 证据范围与记录缺口

失败前控制Host区间已保存到control_cost_breakdown.csv。当前Oracle枚举29个子集并返回Exact候选N18/N27与J=48.228728759 modeled ms，但该结果不能替代当前Greedy最终选集或GPU结果。

失败RPC未返回完整Epoch和Plan，且ControlClient抛异常使runner原计划的reconciliation.json写盘未发生。原异常完整含失败对象，本轮只将其解析归档，没有补造Snapshot、重算分配或推断实际恢复E/G。

CPU门禁缺少真实历史来源同目标Session的目标选择负面/正面覆盖；因此“CPU通过”没有保证该实验门禁符合任务范围。CUDA计时包装尚未在正式目标执行中验证。

## 清理与完整性

唯一Engine已关闭、专用容器不存在；连续三次GPU0观测均无计算进程、显存14MiB，清理PASS。退出CUDA IPC警告原样保留，不从警告推导状态损坏，也不只凭返回码认定清理成功。

冻结来源和启动时执行锁摘要复核不变。所有新产物仅在本目录。没有修改生产Greedy、Exact目标、Phi、Strict Compatibility或历史冻结证据。

## 唯一后续建议

先修正独立实验入口的目标来源条件：Greedy允许其合法公共共享源的首次观察历史来自同一Session；Exact仍必须证明D历史来源，双方都保留实际Native Match、COW、输出和状态保护门禁。同时用本轮真实Census增加CPU来源归因回归，并在失败时先保存完整Epoch/Plan/响应。之后另行冻结并授权新运行，不能覆盖此次失败。

本轮遵守失败即停；不自动修复、不重试、不扩展工作负载。
''')
 validation.update(failure_class='EXPERIMENT_PROVENANCE_GATE_ERROR',native_evictions=0,pending_executed=0,actual_cleanup_pass=all(v['status']=='PASS' for v in cleanup.values()))
 save('validation.json',validation)
 save('evidence_manifest.json',{'files':{str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and not any(x.startswith('triton_cache') or x=='__pycache__' for x in p.parts) and p.name!='evidence_manifest.json'}})
 print('失败归因与完整交付已保存，BLOCKED，未重试')
if __name__=='__main__':main()
