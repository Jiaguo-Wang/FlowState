"""归档已停止的失败运行及可证实的部分证据，不重试实验。"""
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write
from .protocol import admission_gate

def main():
    trace=read(OUT/'strict/native_trace.json');rec=read(OUT/'strict/reconciliation_trace.json');contract=read(OUT/'target_contract.json')
    request=read(OUT/'strict/request.json');admission=trace['admission'][0]
    assert rec['status']=='READY' and rec['reconciliation']['status']=='PASS'
    admission_gate(rec,admission['state'],admission['input_ids'],request['pending']['token_ids'])
    assert len(rec['operations'])==10 and all(x['returned'] for x in rec['operations'])
    match=trace['matches'][0]['native'];source=next(x['node'] for x in rec['reconciliation']['binding']['candidates'] if x['candidate_id']=='RQ3_G001_B_TURN_001')
    assert (match['physical_fa_hit'],match['executable_prefix'],match['replay_gap'])==(contract['T'],contract['E'],contract['G'])
    assert match['best_match_node']==source['node_id'] and match['cow_source_slots']==source['slots']
    assert not trace['copies'] and not trace['execution'] and not (OUT/'strict/output.json').exists() and not (OUT/'cold').exists()
    log=(OUT/'strict.runtime.log').read_text();assert "AttributeError: 'Req' object has no attribute 'fill_ids'" in log
    cleanup=read(OUT/'gpu_cleanup.json');assert cleanup['strict']['status']=='PASS'
    for p,h in read(OUT/'implementation_lock.json')['sha256'].items():assert sha(p)==h,p
    identity=read(OUT/'source_identity.json')
    for row in identity['files']:assert sha(row['path'])==row['current_sha256'],row['path']
    for directory in (ROOT/'evaluation/runtime_control_closure_r2c0',ROOT/'evaluation/runtime_control_closure_r2c1a'):
        for rel,h in read(directory/'evidence_manifest.json')['files'].items():assert sha(directory/rel)==h,rel
    write(OUT/'resume_trace.json',{'gate':'BLOCKED','reason':'新增执行入口观测读取不存在的Req.fill_ids，Scheduler在调用原run_batch前退出','reconciliation_status':'PASS','completed_evictions':rec['reconciliation']['completed_evictions'],'admission_equivalence':'PASS','native_match':match,'copies':trace['copies'],'execution':trace['execution'],'raw_trace':'strict/native_trace.json','evidence_level':'Actual Runtime State Matched；未达到RESUME_VALIDATED','retry_run':False})
    write(OUT/'output_correctness.json',{'status':'NOT_EVALUATED','strict_output':'MISSING：执行入口观测异常，没有生成结果','cold_reference':'NOT_RUN：Strict失败后停止','expected_tokens':8,'sampling_params':contract['sampling_params'],'exact_match':'UNKNOWN','logits':'UNKNOWN','说明':'没有输出差异可分析，不能归因于数值非确定性或恢复计算错误'})
    write(OUT/'state_integrity.json',{'reconciliation_postcheck':'PASS','admission_before_match':'PASS','selected_payload_unchanged_until_admission':True,'native_match_source_matches_B1':True,'working_slot_assigned':match['working_slots'],'source_slot':match['cow_source_slots'],'cow_copy_executed':'UNKNOWN：没有复制调用完成记录','post_request_selected_integrity':'UNKNOWN：请求未完成，无执行后状态观测','admission_state':admission['state'],'pre_match_state':trace['matches'][0]['before'],'housekeeping':trace['housekeeping'],'gpu_cleanup':cleanup})
    write(OUT/'integrity_summary.json',{'status':'PASS','input_and_implementation_sha256':'PASS','prior_frozen_artifacts':'PASS','gpu_cleanup':'PASS','runtime_correctness_gate':'BLOCKED','说明':'完整性与清理通过不代表恢复或输出正确性通过'})
    report='''# Phase R2-C1b-1：单目标跨Workflow恢复正确性验证

## 最终门禁：BLOCKED

**本轮未完成恢复输出正确性闭环。阻断来自本轮新增观测代码错误，不是已经证实的FlowState恢复计算错误。**

新增 `bridge.py:121` 在Scheduler.run_batch调用前读取 `req.fill_ids`；当前冻结SGLang v0.5.17的Req没有该属性，触发AttributeError，Scheduler退出。该错误由本轮实现引入，应由本轮实现承担，不能归因于模型非确定性、输出容差或冻结数据。执行入口观测没有满足完整的无扰动要求。

已保留失败生命周期全部证据，未重试，也未继续启动Cold Reference。GPU0已清理至14MiB、无计算进程，本轮容器已消失。旧产物、算法与严格矩阵没有修改。

## 1. 输入与计划前提

目标固定为 `RQ3_G001_A_PENDING_TURN_004`，没有更换目标。

- 冻结B1→A_PENDING边：EPOCH_ADMISSIBLE，全部必要条件为true。
- 计划K=2，Selected=B1/B3；其中仅B1兼容A_PENDING。
- B1位置2432；Planning Target T=4224、计划E=2432、G=1792。
- 实际完整请求输入4394个Token。Planning Target与完整输入长度不是同一概念。
- B3位置5312，不能成为该目标的更深合法恢复来源。
- 预定协议沿用B1-C：temperature=0、max_new_tokens=8、ignore_eos=True，逐Token精确比较；未放宽输出规则。

来源：target_contract.json、source_identity.json、strict/prepared_inputs.json、strict/request.json。

## 2. 实际执行范围和环境

本轮只启动了一个Strict Engine生命周期，执行12条原冻结历史请求，再提交一个目标Pending；Cold未启动，其余三个Pending未提交。

Qwen3.5-9B、SGLang0.5.17、Transformers5.12.1、TP=1、28槽、冻结Engine参数。GPU0为当时空闲的H100 PCIe；镜像ID与C0/C1a一致，模型、Tokenizer、输入、生产实现与新观测脚本均在运行前锁定。独立新Engine，不使用C1a已结束的窗口或物理绑定。

见 implementation_lock.json、image_identity.json、strict_gpu_before.json 和 strict_command.json。执行后再次核对内容摘要，均一致。

## 3. 已经通过的部分

|阶段|结果|实际证据|
|---|---|---|
|12条历史输入与当前严格状态重建|PASS|strict/history_identity.json、history_replay.json|
|当前严格48对矩阵/计划/Bind/Preflight|PASS|strict/reconciliation_trace.json.initial_preflight|
|正式生产Reconciliation|PASS|10项驱逐完成，最终B1/B3驻留，16→26空闲槽|
|协调完成时Selected物理身份及内容|PASS|reconciliation_trace的before/after与Selected摘要|
|协调结束至Admission前状态|PASS|native_trace.admission与admission_baseline全量比较|
|首次观测的Native idle housekeeping|未改变语义及Selected内容|native_trace.housekeeping；并未声称枚举每次housekeeping|
|Native Match来源|PASS，仅匹配等级|下表及native_trace.matches|
|实际COW复制完成|UNKNOWN|没有copies完成记录|
|Prefill/Decode实际执行与输出|未完成|run_batch观测在调用原函数前异常|
|请求执行后Selected完整性|UNKNOWN|没有完成请求后的状态采集|
|Cold Reference与逐Token比较|未执行|Strict失败后停止|

Admission观测位于原 `Scheduler.process_input_requests` 分派目标请求之前。实际输入4394个Token与冻结输入逐项相等；当前缓存semantic、节点/slot和Selected内容摘要均与协调结束后的基线一致。因此这一已通过部分没有把旧控制窗口当作跨RPC租约。

## 4. 实际Native Match记录

|字段|观测结果|
|---|---:|
|Attention-side Prefix H|4224|
|Recurrent-aware executable prefix E|2432|
|H−E|1792|
|best_match_node / last_device_node|20 / 20|
|COW source slot|6|
|已分配working slot|27|
|cow_requested|true|
|输入Token数|4394|

新生命周期绑定的B1恰为Node20/Slot6；实际原生match返回一致。Match之前树中只有B1（2432、Slot6）与B3（5312、Slot22）驻留，没有更深兼容源，也没有已驱逐候选被记录为恢复源。

**working slot已分配与cow_requested=true不等于已完成COW复制。** 本轮 copies=[]、execution=[]，不能将这条边提升为RESUME_VALIDATED，也不能声称请求实际计算已从2432正确恢复。最多支持“当前严格协调及Admission状态有效，原生匹配选择了预期跨Workflow源”。

原始证据：strict/native_trace.json；摘要：resume_trace.json。

## 5. 精确故障链

`Scheduler.event_loop_normal:1704 → 本轮bridge.py:run（121行）→ len(req.fill_ids) → AttributeError → 原生Scheduler异常处理 → SIGQUIT → Engine退出`。

完整日志见 strict.runtime.log:723–764（runtime.log另带生命周期标题）。错误发生在调用保存的原生old_run之前，没有生成output.json，也没有Strict summary.json。

原生异常处理最终导致主进程SystemExit(0)，所以容器返回码0不能被视为实验成功。启动器随后因缺少严格成功summary.json停止，launch_failure.json保存了这一附加诊断。这里的summary缺失是前述Scheduler异常的后果，不是另一个恢复错误。

只读核查冻结镜像源码确认：`schedule_batch.py:1225–1226` 提供 `get_fill_ids()`，由 `full_untruncated_fill_ids[:extend_range.end]` 取得当前输入范围；`prepare_for_extend:2291` 使用 `r.get_fill_ids()[len(r.prefix_indices):]`。不能沿用旧接口名fill_ids，也不应未经核对继续假定Req上存在extend_input_len。

该源码核查使用不挂载GPU、不启动Engine的容器，仅为诊断；没有追加GPU实验。

## 6. 输出、状态保护与清理边界

输出比较为NOT_EVALUATED，不是PASS，也不是已发现Token不一致。没有可供比较的Strict输出，Cold未运行；无依据讨论数值非确定性或更改容差。

协调后、Admission前以及原生Match前保存了Selected身份和数值证据；执行后是否保持无法从本次未完成请求证明。state_integrity.json明确标注UNKNOWN，未用协调阶段的PASS替代请求执行后的保护证据。

gpu_cleanup.json：Strict容器不存在、GPU0无计算进程、14MiB；未操作其他用户进程。保留所有启动告警、Scheduler异常、NCCL与CUDA IPC清理告警，不省略失败结果。

CPU测试34项、3个子测试通过，但覆盖的是纯协议及Fake协调合同，没有覆盖新插桩与镜像中真实Req API的字段兼容性；这正是本次测试缺口，不能用CPU通过掩盖GPU失败。

## 7. 唯一建议与停止边界

先修正本轮执行入口观测：按冻结版本的 `get_fill_ids()`、`extend_range` 与实际batch字段记录Prefill区间，并在CPU侧使用真实Req API验证观测接口。可选诊断字段缺失应明确标为UNKNOWN，不能意外终止原生推理；必要证据缺失应在提交目标前阻断。

本轮保留已执行源码和锁不变，不覆盖失败证据、不静默修复后重跑。后续若重新开展验证，必须使用新的独立生命周期和独立输出目录，仍验证同一个Pending及原8-Token协议。**本轮到此停止，未进入其余Pending或R3。**
'''
    (OUT/'final_report.md').write_text(report)
    files=[p for p in OUT.rglob('*') if p.is_file() and not any(x.startswith('triton_cache') for x in p.parts) and p.name!='evidence_manifest.json']
    write(OUT/'evidence_manifest.json',{'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(files)},'说明':'失败原始记录和运行时观测完整保存，不覆盖旧证据'})
    print('BLOCKED：观测接口错误；部分证据及清理通过，Cold未执行，无重试')
if __name__=='__main__':main()
