"""只读核验已完成的两个生命周期并归档结果，不启动或重试实验。"""
from pathlib import Path
from .prepare import ROOT,OUT,read,sha,write
from .protocol import admission_gate,compare_outputs

SOURCE='RQ3_G001_B_TURN_001'
def main():
    cleanup=read(OUT/'gpu_cleanup.json')
    failures=[]
    for p,h in read(OUT/'implementation_lock.json')['sha256'].items():
        if sha(p)!=h:failures.append('当前实现摘要改变：'+p)
    identity=read(OUT/'source_identity.json')
    for row in identity['files']:
        if sha(row['path'])!=row['current_sha256']:failures.append('冻结来源改变：'+row['path'])
    for name in ('runtime_control_closure_r2c0','runtime_control_closure_r2c1a','runtime_control_closure_r2c1b1'):
        directory=ROOT/'evaluation'/name
        for rel,h in read(directory/'evidence_manifest.json')['files'].items():
            if sha(directory/rel)!=h:failures.append('历史产物改变：'+str(directory/rel))
    rec=read(OUT/'strict/reconciliation_trace.json')
    trace=read(OUT/'strict/native_trace.json')
    contract=read(OUT/'target_contract.json')
    request=read(OUT/'strict/request.json')
    admission_gate(rec,trace['admission'][0]['state'],trace['admission'][0]['input_ids'],request['pending']['token_ids'])
    assert rec['status']=='READY' and len(rec['operations'])==10 and all(x['returned'] for x in rec['operations'])
    assert rec['reconciliation']['status']=='PASS'
    assert sum(bool(n['slots']) for n in rec['before']['nodes'])==12
    assert sum(bool(n['slots']) for n in rec['after']['nodes'])==2
    assert len(rec['before']['free_slots'])==16 and len(rec['after']['free_slots'])==26
    match=next(x['native'] for x in trace['matches'] if x.get('native',{}).get('cow_requested'))
    source=next(x['node'] for x in rec['reconciliation']['binding']['candidates'] if x['candidate_id']==SOURCE)
    assert match['best_match_node']==source['node_id'] and match['cow_source_slots']==source['slots']
    assert (match['physical_fa_hit'],match['executable_prefix'],match['replay_gap'])==(contract['T'],contract['E'],contract['G'])
    output={'status':'NOT_EVALUATED','说明':'缺少完成输出或独立Cold结果'}
    protection={'status':'UNKNOWN','说明':'未完成执行后的Selected检查'}
    try:
        for arm in ('strict','cold'):
            assert read(OUT/arm/'summary.json')['status']=='PASS',arm
            assert cleanup[arm]['status']=='PASS',arm
        output=compare_outputs(read(OUT/'strict/output.json'),read(OUT/'cold/output.json'))
        strictstate=read(OUT/'strict/post_request.json')['state']
        assert strictstate['selected_payload']==rec['selected_payload_after']
        assert strictstate['selected_components']==rec['admission_baseline']['selected_components']
        copies=[c for c in trace['copies'] if c.get('returned') and c['source_slots']==source['slots'] and c['destination_slots']==match['working_slots']]
        assert copies and all(c['source_payload_before']==c['destination_payload_after']==c['source_payload_after'] for c in copies)
        assert trace['execution'] and all(x.get('native_run_returned') for x in trace['execution'])
        prefill=[x for x in trace['execution'] if isinstance(x['extend_start'],int)]
        assert prefill and prefill[0]['extend_start']==match['executable_prefix'] and prefill[0]['extend_end']==len(request['pending']['token_ids'])
        before=rec['admission_baseline'];after=strictstate
        def residents(s):return [{'node_id':n['node_id'],'position':len(n['token_ids']),'slots':n['slots']} for n in s['nodes'] if n['slots']]
        before_ids={x['node_id'] for x in before['nodes']};after_ids={x['node_id'] for x in after['nodes']}
        protection={'status':'PASS','selected_payload_before':rec['selected_payload_after'],'selected_payload_after':after['selected_payload'],'selected_components_before':before['selected_components'],'selected_components_after':after['selected_components'],'managed_resident_after_reconciliation':rec['plan']['selected_ids'],'resident_before_request':residents(before),'resident_after_request':residents(after),'newly_resident_existing_nodes':[n for n in residents(after) if not next((b['slots'] for b in before['nodes'] if b['node_id']==n['node_id']),[])],'new_node_ids':sorted(after_ids-before_ids),'removed_node_ids':sorted(before_ids-after_ids),'semantic_snapshot_equal':before['semantic']==after['semantic'],'accounting_before':before['semantic']['accounting'],'accounting_after':after['semantic']['accounting'],'说明':'请求后允许新增缓存节点与访问元数据变化；历史B1/B3切片身份与Conv/Temporal内容严格不变'}
    except (AssertionError,FileNotFoundError,KeyError,ValueError) as exc:
        failures.append(repr(exc))
    if (OUT/'launch_failure.json').exists():failures.append(read(OUT/'launch_failure.json'))
    gate='BLOCKED' if failures else 'READY'
    write(OUT/'output_correctness.json',output)
    write(OUT/'state_integrity.json',protection)
    write(OUT/'resume_trace.json',{'gate':gate,'reconciliation_status':rec['reconciliation']['status'],'completed_evictions':rec['reconciliation']['completed_evictions'],'admission_equivalence':'PASS','native_match':match,'copies':trace['copies'],'execution':trace['execution'],'raw_trace':'strict/native_trace.json','evidence_level':'RESUME_VALIDATED：仅此目标边及当前受控协议' if gate=='READY' else '仅按实际记录分级，未完成恢复认证','failures':failures})
    write(OUT/'integrity_summary.json',{'status':'PASS' if not failures else '核对结果见失败清单','source_files':len(identity['files']),'prior_frozen_manifests':['R2-C0','R2-C1a','R2-C1b-1'],'gpu_cleanup':cleanup,'failures':failures})
    coldmatch=read(OUT/'cold/summary.json')['native_match'] if (OUT/'cold/summary.json').exists() else {}
    rows='\n'.join('|'+str(x['forward_mode'])+'|'+str(x['extend_start'])+'|'+str(x['extend_end'])+'|'+str(x.get('native_run_returned',False))+'|' for x in trace['execution'])
    report=f'''# Phase R2-C1b-2：真实请求接口修复与跨Workflow恢复重验

## 最终门禁：{gate}

本轮只验证 `RQ3_G001_A_PENDING_TURN_004`，使用g001/K2严格计划B1/B3。没有运行其他Pending、策略性能比较或修改Greedy、Phi、严格兼容矩阵。R2-C1b-1的失败记录保持原样。

## 一、接口修复与CPU门禁

新建 `observation.py`，使用真实 `Req.get_fill_ids()`、`extend_range`、`prefix_indices` 及Batch的 `prefix_lens/extend_lens`。不再读取不存在的Req.fill_ids或Req.extend_input_len。包装原函数恰好一次，返回值与异常原样传递；可选字段缺失记UNKNOWN。没有额外Cache查找或决策修改。

冻结镜像中实际Req、ScheduleBatch与ForwardMode的CPU检查：43项测试及3个子测试通过。覆盖真实目标输入、Cold、分块、Decode、只读性、缺失诊断、必要字段冲突、调用次数与异常传播。没有用FakeReq代替真实类型。既有FakeRuntime测试仅用于协调合同。首次测试收集导入错误已修正，日志保留；不是GPU故障。源码、时机及CPU限度详见 `req_api_audit.md`、`req_api_source.json`、`cpu_tests.log`。

## 二、资源、身份与执行范围

首次宿主机预检查发现GPU0属于其他用户任务，未创建Engine；原始拒绝记录保留于 `preflight_gpu0/`。随后将宿主机GPU固定到空闲GPU1，同型号H100 PCIe，容器内cuda:0。没有操作其他用户进程。

Qwen3.5-9B、SGLang v0.5.17、TP=1、28槽及原Engine参数全部核对。12条历史请求使用冻结Token；Strict与Cold各自独立生命周期。Cold只提交相同完整目标输入，初始Recurrent Cache为空，不复用Strict状态。每臂均核对来源和实现SHA。没有使用旧Node/Slot作为当前映射。

当前源码身份：`implementation_lock.json`；来源身份：`source_identity.json`；实际命令及设备身份：`strict_command.json`、`cold_command.json`、各臂gpu_before文件。旧来源共{len(identity['files'])}项再次校验，另核对C0、C1a及C1b-1完整清单。

## 三、严格协调与Admission

同一次Scheduler同步控制调用内完成当前Strict Epoch/计划/Bind/Preflight、10项真实Recurrent-only驱逐及全部Postcheck。Selected=B1/B3，12→2驻留，16→26空闲槽；FA侧及非目标状态合同通过。当前B1映射Node {source['node_id']} / Slot {source['slots']}。

在真实 `process_input_requests` 分派目标之前再次读取缓存，完整节点/Slot、语义状态和Selected内容与协调结束基线相同，实际输入逐Token相同。原生Match前再次保存状态。没有把已结束的同步控制窗口解释为跨RPC租约。

## 四、实际恢复、COW与执行

|量|Strict实际值|Cold实际值|
|---|---:|---:|
|完整原始输入Token数|{match['input_token_count']}|{coldmatch.get('input_token_count','UNKNOWN')}|
|Attention前沿H|{match['physical_fa_hit']}|{coldmatch.get('physical_fa_hit','UNKNOWN')}|
|原生Executable Prefix E|{match['executable_prefix']}|{coldmatch.get('executable_prefix','UNKNOWN')}|
|H−E|{match['replay_gap']}|{coldmatch.get('replay_gap','UNKNOWN')}|
|匹配节点|{match['best_match_node']}|{coldmatch.get('best_match_node','UNKNOWN')}|
|COW源槽|{match['cow_source_slots']}|{coldmatch.get('cow_source_slots','UNKNOWN')}|
|工作槽|{match['working_slots']}|{coldmatch.get('working_slots','UNKNOWN')}|

COW完成记录数：{len(trace['copies'])}。完成认证要求原生 `pool.copy_from` 返回，复制前源内容、复制后目标内容与复制后源内容SHA逐项相同；CPU拷回读取同步观察实际Conv/Temporal张量内容，不能只用cow_requested作为证明。工作槽与Selected槽必须互斥。

实际原生执行记录如下；返回只表明本次run_batch返回，最终输出另行精确核验。

|模式|Extend起点|Extend终点|原函数返回|
|---|---:|---:|---|
{rows}

恢复输入区间与H−E分别解释：原始4394 Token，E=2432，首次Prefill为[2432,4394)，共1962 Token；其中1792对应H−E，另外170在H之后。Decode不以残留extend_range解释当前计算。没有把模型计算的E/G直接冒充实际执行记录。

## 五、输出正确性

冻结协议不变：temperature=0、max_new_tokens=8、ignore_eos=True；逐Token精确比较，停止行为和输出长度相同，禁止回撤。结果：{output['status']}。

Strict输出：{output.get('strict_output_ids','UNKNOWN')}。

Cold输出：{output.get('cold_output_ids','UNKNOWN')}。

停止原因：{output.get('finish_reason','UNKNOWN')}。

原协议没有Logits对照，本轮未增加或放宽任何数值容差。Cold只用于正确性，不据其Prefill时间宣称性能收益。

## 六、历史状态保护与正常副作用

执行后Selected组件内容与切片身份检查：{protection['status']}。B1/B3逐层Conv/Temporal内容使用SHA比较，源状态不得被COW或工作槽覆盖。

请求后新增节点：{protection.get('new_node_ids','UNKNOWN')}；移除节点：{protection.get('removed_node_ids','UNKNOWN')}。请求前后完整语义视图相等：{protection.get('semantic_snapshot_equal','UNKNOWN')}。新增驻留的原有节点：{protection.get('newly_resident_existing_nodes','UNKNOWN')}。此处实际出现的是原有Attention节点重新获得请求执行新产生的Recurrent缓存，而不是Admission前意外恢复被驱逐的旧状态。原生 `mamba_component.py:517–549, 608–620` 的extra_buffer路径按本次请求tracking位置插入状态并回收工作槽；当前请求前后证据与该路径一致。没有逐条插入事件，因此不把源码解释提升为新增节点每条写入的独立跟踪认证。完整视图不要求相等，因为正常请求可插入新前缀节点、缓存状态并改变访问元数据；以Selected历史状态不变、槽隔离及实际新增状态清单区分正常副作用和损坏，详见 `state_integrity.json` 与 `strict/post_request.json`。

## 七、清理、边界及结论

清理：{ {k:v['status'] for k,v in cleanup.items()} }。各臂独立Engine退出，本轮容器移除，实验GPU无计算进程且显存回到门限内；具体MiB见 `gpu_cleanup.json`。返回码不单独作为成功依据：要求每臂summary、匹配/COW/执行记录、输出比较及清理同时成立。

失败清单：{failures}。

此结果最多支持当前受控g001、当前计划、当前B1→A_PENDING边和8-Token确定性协议。不能推广到全部准入边、长期Pin/Lease、全部Pending、真实在线Agent并发、物理HBM容量压力或端到端性能净收益。

本轮到此停止，不运行其余三个Pending，不进入策略A/B或R3。
'''
    (OUT/'final_report.md').write_text(report)
    files=[p for p in OUT.rglob('*') if p.is_file() and not any(x.startswith('triton_cache') for x in p.parts) and p.name!='evidence_manifest.json']
    write(OUT/'evidence_manifest.json',{'files':{str(p.relative_to(OUT)):sha(p) for p in sorted(files)},'说明':'运行、失败预检查与完整证据均保留，未覆盖历史产物'})
    print(gate)
if __name__=='__main__':main()
