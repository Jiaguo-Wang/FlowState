"""保存中止后的逐臂账本和只读证据核对，不重跑生命周期或生成完整主矩阵结论。"""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)

def main():
    protocol = read(OUT / 'protocol.json')
    failure = read(OUT / 'launch_main_failure.json')
    lock = read(OUT / 'implementation_lock_main.json')
    mismatches = [p for p, h in lock['files'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert not mismatches, mismatches
    inputs = {p['continuation_id']: p for p in read(OUT / 'prepared_inputs.json')['pendings']}
    plans = read(OUT.parent / 't2g1_plans.json')['plans']
    ledger, checks = [], {}
    for arm in protocol['main_arms']:
        name = arm['name']
        folder = OUT / name / 'run'
        row = {'arm': name, 'pending': arm['pending'], 'policy': arm['policy'], 'rep': arm['rep'], 'status': 'NOT_RUN'}
        if (folder / 'summary.json').exists():
            s = read(folder / 'summary.json')
            r = read(folder / 'reconciliation.json')
            post = read(folder / 'post_request.json')
            clean = read(OUT / (name + '_cleanup.json'))
            cold = read(OUT / ('cold_reference_' + arm['pending'] + '.json'))
            output = read(folder / 'output.json')
            inp = inputs[arm['pending']]
            expected = plans[arm['policy']]
            eg = expected['expected_E_G'][arm['pending']]
            native = [m['native'] for m in post['trace']['matches'] if m.get('native', {}).get('request_id') == arm['pending']][-1]
            binding = {b['candidate_id']: b['node'] for b in r['reconciliation']['binding']['candidates']}
            source = binding.get(r['target']['source_checkpoint'])
            c = {
                '摘要与调度': s['status'] == 'PASS' and s['policy'] == arm['policy'] and s['pending'] == arm['pending'],
                '选集': s['selected_cpu'] == expected['selected_checkpoint_ids'],
                '当前前沿': s['E_executable_prefix'] == eg['E'] and s['G_replay_gap'] == eg['G'] and s['H_physical_fa_hit'] == inp['attention_frontier'],
                '实际Prefill': s['prefill_token_count'] == len(inp['token_ids']) - eg['E'] and s['batch_prefix_length'] == eg['E'],
                '独立Cold原始输出': output['output_ids'] == cold['output_ids'] == s['output_ids'] and len(output['output_ids']) == 8,
                '准确驱逐补集': [o['checkpoint_id'] for o in r['operations']] == r['plan']['logical_eviction_ids'] and len(r['operations']) == 13 and all(o['returned'] for o in r['operations']),
                '协调与窗口': r['reconciliation']['status'] == 'PASS' and r['apply_validation']['same_scheduler_call'],
                '保留组件内容': r['selected_payload_before'] == r['selected_payload_after'] == post['state']['selected_payload'],
                'FA树索引': r['before']['attention_digest'] == r['after']['attention_digest'],
                '受保护FA内容': post['state']['fa_protected_payload'] == r['fa_protected_payload'],
                '原生匹配身份': native['executable_prefix'] == eg['E'] and (eg['E'] == 0 or (source is not None and native['best_match_node'] == source['node_id'] and native['cow_source_slots'] == source['slots'])),
                '原生COW完成': eg['E'] == 0 or (source is not None and any(o['returned'] and o['source_slots'] == source['slots'] and o['destination_slots'] == native['working_slots'] for o in post['trace']['copies'])),
                '执行完成': all(o['native_run_returned'] for o in post['trace']['execution']),
                '请求窗零新增编译': s['triton_new_kernels'][-1]['new_entries'] == 0,
                '形状预热': s['shape_warm_b']['E'] == s['shape_warm_b']['fa_hit'] == eg['E'],
                '请求窗审计合同': s['audit_entries_total'] == 2 and s['measure_mode'] is True,
                '进程正常退出': read(OUT / (name + '_process.json'))['returncode'] == 0,
                'Engine关闭': read(folder / 'engine_cleanup.json')['shutdown_returned'],
                '本轮容器退出': not clean['own_container'],
                '三次空卡收尾': clean['status'] == 'PASS',
            }
            checks[name] = c
            row.update(status='PASS' if all(c.values()) else 'RESOURCE_GATE_FAIL', H=s['H_physical_fa_hit'], E=s['E_executable_prefix'], G=s['G_replay_gap'], prefill_tokens=s['prefill_token_count'], ttft_ms=s['ttft_ms'], cuda_prefill_ms=s['cuda_prefill_ms'], request_wall_ms=s['request_wall_ms'], control_rpc_ms=s['control_rpc_wall_ms'], epoch_wall_ms=s['epoch_wall_ms'], J_model_ms=s['plan_recovery_cost_ms'])
        ledger.append(row)
    result = {'gate': 'BLOCKED', 'cause': failure['error'], 'completed_arms': failure['completed_arms'], 'attempted': len(checks), 'not_run': sum(r['status'] == 'NOT_RUN' for r in ledger), 'frozen_hash_mismatches': mismatches, 'checks': checks, 'notes': ['未执行完整60臂验证或五轮统计。第四臂测量区间与外部进程启动是否重叠为UNKNOWN，不认定为有效完整臂。', '本复核仅覆盖已产生原始记录的四臂，不能替代完整主矩阵门禁。E=0时源Checkpoint与COW检查不适用，不声称发生COW。']}
    write(OUT / 'takeover/interrupted_main_verification.json', result)
    fields = ['arm','pending','policy','rep','status','H','E','G','prefill_tokens','J_model_ms','ttft_ms','cuda_prefill_ms','request_wall_ms','control_rpc_ms','epoch_wall_ms']
    with (OUT / 'takeover/interrupted_main_ledger.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ledger)
    lines = ['# T2-G2 正式GPU实验断点接管报告', '', '**最终状态：BLOCKED（执行期间出现外部GPU占用，已按冻结协议停止）。**', '', '## 接管与门禁', '', '原 verify_formal.py 的字符串/字典类型错误在接管前已经修复保存，本次没有再次修改。三个既有 Gate 未重跑；原CPU检查器33项与补充原始证据173项通过后，检查程序生成 gate_passed.json。门禁覆盖选集、T/E/G、Prefill、FA保护、13项Recurrent-only驱逐、Native来源/COW、Cold逐Token、形状预热、请求窗口编译和生命周期收尾。详见 formal_verification_gate.json 与 takeover/evidence_gate.json。', '', '接管时主矩阵尚无产物，未发现活动实验launcher或Engine，GPU0为H100 PCIe、14MiB、无计算进程。核对冻结协议、来源、模型镜像及实现摘要后，按授权启动原 launch_formal.py main；没有新增GPU预实验，没有修改策略、调度、选集或原生控制语义。结束后再次核验 implementation_lock_main.json，所有文件摘要一致。', '', '## 实际执行与停止原因', '', '正式主矩阵尝试4/60臂：3臂完整通过，1臂收尾资源门禁失败，56臂未执行。尚未完成任何完整四Pending轮次，更未完成五轮正式配对。', '', '|臂|状态|H/E/G|Prefill Tokens|原始TTFT ms|原始CUDA Prefill ms|', '|---|---|---|---:|---:|---:|']
    for row in ledger:
        if row['status'] != 'NOT_RUN':
            lines.append(f"|{row['arm']}|{row['status']}|{row['H']}/{row['E']}/{row['G']}|{row['prefill_tokens']}|{row['ttft_ms']:.6f}|{row['cuda_prefill_ms']:.6f}|")
    lines += ['', '上述数字只保留已经发生的尝试，不据此计算正式性能收益。第四臂不能因 summary.status=PASS 而升级为完整通过。', '', 'm1B_FS 进程返回0，Engine shutdown_returned=true，专用容器已不存在。其三次清理观察显存依次为14、17、1569MiB；第三次出现外部PID 1303777，程序路径为 /home/yangshuo/robot-learning/.venv/bin/python，占1546MiB。随后只读核查仍有该外部进程，占1652MiB，GPU总用量1675MiB；没有本轮专用容器或launcher残留。未终止或干预该用户任务。', '', '证据支持“收尾时失去独占空卡条件”，不支持“恢复计算错误”或“本轮Engine清理泄漏”。外部进程准确启动时刻及是否与第四臂计时窗口重叠为UNKNOWN；该臂保留为资源门禁失败，不用于正式收益统计。', '', '## 证据与统计边界', '', '- Gate三臂仅作正确性/有效性门禁，全部排除性能统计。', '- 已尝试四臂的局部只读复核见 takeover/interrupted_main_verification.json；唯一失败项为第四臂三次空卡收尾。原始文件不覆盖。', '- 全60臂账本见 takeover/interrupted_main_ledger.csv；缺测不填零、不估计、不以Gate替代。未调用要求完整60臂的分析程序，未生成五轮配对或四Pending等权总体结论。', '- 模型J、实际E/G/Prefill与实测TTFT分开保留；控制RPC及epoch_wall原始值见账本，不与其子区间重复相加，也不解释为生产净收益。', '- Marconi-Evict为Marconi-Evict Adaptation，KVFlow为KVFlow-inspired；不称完整原论文系统。当前实验应用冻结选集，不证明在线策略求解。跨Workflow物理血缘仍为PROVENANCE_UNVERIFIED。', '', '## 断点与保全', '', '最后完整通过：m1A_KV。失败对象：m1B_FS。原计划下一未执行臂：m1B_MC。本次没有重试或自动等待空卡。', '', '原launcher会独占创建 implementation_lock_main.json、execution_started_main.json 和逐臂目录，并不支持跳过已有主臂；因此不能直接再次执行 launch_formal.py main。任何后续恢复都需要先明确处理既有失败臂和不可覆盖的执行账本，不能绕过冻结测量条件。本轮不擅自实现续跑入口或补跑。', '', '关键证据：launch_main_failure.json、arm_status_main.json、m1B_FS_cleanup.json、m1B_FS_process.json、m1B_FS/run/{summary,output,engine_cleanup,reconciliation,post_request}.json、runtime_main.log、main_takeover_host.log、takeover/recovery_inventory.json。', '', '本轮已停止。没有修改冻结输入、生产代码或原有GPU产物。']
    with (OUT / 'formal_gpu_comparison_report.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps({'gate': result['gate'], 'attempted': result['attempted'], 'not_run': result['not_run'], 'failed_checks': {a: [k for k,v in c.items() if not v] for a,c in checks.items()}}, ensure_ascii=False))

if __name__ == '__main__':
    main()
