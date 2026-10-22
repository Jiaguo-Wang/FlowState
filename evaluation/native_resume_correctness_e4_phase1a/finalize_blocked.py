"""保存资源门禁失败证据；不启动Engine，不替代或推断Runtime成功。"""
import csv,json,time
from .prepare import OUT,ROOT,read,sha,write

def main():
 failure=read(OUT/'launch_failure.json');assert failure['attempted']==[] and failure['validated']==0 and 'GPU存在其他任务' in failure['error']
 write(OUT/'resource_preflight.json',{'status':'BLOCKED','recorded_unix_ns':time.time_ns(),'evidence':'失败后独立只读nvidia-smi及本轮容器查询；控制台原始输出如下','compute_apps_csv':'pid, process_name, used_gpu_memory [MiB]\n3059291, sglang::scheduler_ATTN_CP0_TP0, 79554 MiB\n','gpu_csv':'uuid, name, memory.used [MiB], memory.total [MiB]\nGPU-bfff4d31-0b94-7de3-9946-f593437ae566, NVIDIA H100 PCIe, 79578 MiB, 81559 MiB\n','experiment_container_query':'docker ps -a --filter name=flowstate-e4p1a --format {{.Names}} {{.Status}}','experiment_container_stdout':'','process_action':'仅观察；未终止、修改或占用该现有进程','gpu_engine_launched':False})
 plans=read(OUT/'frozen_plans.json');inputs=read(OUT/'prepared_inputs.json');rows=[]
 for p in inputs['pendings']:
  for solver,plan in plans.items():
   pred=next(x for x in plan['per_pending'] if x['continuation_id']==p['continuation_id'])
   rows.append({'pending':p['continuation_id'],'solver':solver,'status':'NOT_EXECUTED_RESOURCE_BLOCKED','frozen_selected_ids':plan['selected_ids'],'model_J_ms':plan['J_ms'],'model_T':pred['T'],'model_E':pred['E'],'model_G':pred['G'],'actual_native_node':'UNKNOWN_NOT_OBSERVED','actual_E':'UNKNOWN_NOT_OBSERVED','reconciliation':'NOT_EXECUTED','resume':'NOT_EXECUTED','cold_comparison':'NOT_EXECUTED','provenance':'PROVENANCE_UNVERIFIED'})
 with (OUT/'per_pending_status.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()})
 write(OUT/'gpu_cleanup_evidence.json',{'status':'NOT_REQUIRED_NO_ENGINE_STARTED','owned_engines':0,'owned_containers':0,'evictions':0,'pending_submissions':0,'cold_submissions':0,'external_gpu_process_untouched':True,'note':'launch产生的gpu_cleanup.json为空映射，不能解释为已执行生命周期的清理通过'})
 report=['# FlowState E4-Phase 1A：w1/K2 Native Resume正确性门禁','','## Gate：BLOCKED','','阻断来自执行前GPU资源门禁，不是已证实的恢复计算错误。GPU0存在不属于本轮的SGLang计算进程PID 3059291，进程显存79554 MiB；设备总占用79578/81559 MiB。本轮没有启动Engine、创建实验容器、驱逐状态或提交任何请求。遵循失败即停规则，未等待后自动重试、未干预该进程。','','## 完成的只读与CPU准备','','- 核验E2/E3冻结产物、w1完整输入与E1/B5-D4b实现摘要；准备12个独立生命周期的固定协议（每个A–D Pending分别Cold、Greedy、Exact）。\n- 新实验适配只扩展固定目标和冻结语义核验：完整Token前缀双射对齐当前与旧候选，禁止旧Node/Slot作为当前绑定；重放状态不等价则拒绝。\n- 新增CPU合同10/10通过，其中一个测试覆盖两策略×四Pending全部8个目标；既有生产Strict Epoch/Control测试54/54通过。uv312没有pytest，首次pytest启动失败日志保留，随后使用这些测试原本的unittest入口完成54项；未安装依赖或改变环境。\n- CPU证据仅验证代码接口与拒绝合同，不代表真实物理Bind、Reconciliation或Resume成功。','','## 冻结选集核对','','| 策略 | 冻结逻辑候选 | Model J (ms) | 实际物理应用 |','|---|---|---:|---|']
 for solver,plan in plans.items():report.append(f"| {solver} | {', '.join(plan['selected_ids'])} | {plan['J_ms']:.9f} | 未执行 |")
 report+=['','上述Node式ID仅是E2冻结逻辑命名，不代表新生命周期物理映射。E3的Greedy+1-Swap在w1/K2与Exact同集，无需单独Arm。','','## 四个Pending完整执行账本','','| Pending | 策略 | 模型T/E/G | Runtime、COW、Cold对照 |','|---|---|---|---|']
 for r in rows:report.append(f"| {r['pending']} | {r['solver']} | {r['model_T']}/{r['model_E']}/{r['model_G']} | 全部未执行 |")
 report+=['','完成范围：0/8个策略—Pending恢复；0/4个Cold；0项物理驱逐。所有真实前沿、COW与输出比较均UNKNOWN/NOT_EXECUTED，不用模型预测填充Runtime结果。','','## 实现与状态安全边界','','独立runner沿用E1重型路径；独立bridge只设置当前目标并接入冻结映射校验，原Scheduler隔离、Bind/Preflight、生产StateController.reconcile_strict、Native Recurrent-only控制、Admission前物理/内容核对、COW及FA保护逻辑均复用。重型参考和每个严格臂协调后内容审计保留；本轮不做性能调优。原生来源校验继续区分Token/严格准入、实际Node/Slot匹配及血缘。','','FA不变性分两个阶段：协调阶段要求全部原FA索引及内容不变；正常请求阶段允许目标恢复区间重算和新状态生成，要求保留源Recurrent内容及非重算FA内容不变。不能要求请求后全树机械不变，也不能把重算副作用掩盖为状态损坏或反之。以上是待执行合同，本轮未获得新GPU证据。','','## 证据等级','','- EPOCH_ADMISSIBLE：沿用E2已有冻结证据；未采集新Epoch。\n- RECONCILIATION_VALIDATED：本轮未取得。\n- RESUME_VALIDATED：本轮未取得。\n- PROVENANCE_UNVERIFIED：保持原级别，无新增来源血缘认证。','','## 关键文件','','`source_identity.json`、`execution_lock.json`、`experiment_protocol.json`、`frozen_snapshot.json`、`frozen_plans.json`记录来源与预定条件；`cpu_tests.log`和`production_contract_unittest.log`记录CPU结果；`launch_failure.json`保存启动失败堆栈，attempted为空；`resource_preflight.json`记录实际占用及无本轮容器证据；`per_pending_status.csv`列出全部8个未执行条件；`gpu_cleanup_evidence.json`明确无本轮资源需要清理。','','## 能证明和不能证明','','能证明冻结w1计划与目标适配通过CPU核验，且启动前资源安全检查拒绝了当前不满足条件的环境。不能证明两个选集在本轮真实安全落地、Pending实际恢复、输出正确或FA/组件保持。没有GPU性能或端到端收益结论。','','唯一下一步：在GPU0恢复空闲并获得继续执行安排后，以新独立执行目录重新进行相同w1/K2门禁；保留本次失败目录，不覆盖或复用execution_started标记。本轮已停止，不自动重试、不进入Phase 1B，也不扩展w2–w4。']
 (OUT/'final_report.md').write_text('\n'.join(report)+'\n')
 source=read(OUT/'source_identity.json');assert all(sha(p)==h for p,h in source['sha256'].items())
 lock=read(OUT/'execution_lock.json');assert all(sha(p)==h for p,h in lock['sha256'].items())
 write(OUT/'final_integrity.json',{'status':'PASS','source_files_unchanged':len(source['sha256']),'execution_inputs_code_unchanged':True,'gpu_started':False})
 files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
 write(OUT/'evidence_manifest.json',{'files':files});print('BLOCKED：GPU资源门禁；零生命周期；来源与代码保持不变')
if __name__=='__main__':main()
