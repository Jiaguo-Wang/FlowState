"""按既有固定A/B来源准备完整输入，未运行Tokenizer、Engine或分配器。"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
from evaluation.agentic_executable_workload_cpu_gate.adapter import ROOT, PRIOR, OUT, read, sha, digest, require, save_new, causal_cut, tool_causality, input_histories, PreparedInput, PreparedPendingGate


def main():
    import pyarrow.parquet as pq
    require(not (OUT / 'prepared_inputs.json').exists(), '拒绝覆盖已有准备产物')
    source = read(PRIOR / 'source_selection_lock.json')
    config = read(PRIOR / 'input_configuration_lock.json')
    data = Path(source['source_path'])
    require(sha(data) == source['source_sha256'], '来源Parquet摘要变化')
    chosen = source['records'][:2]
    require([r['label'] for r in chosen] == ['A', 'B'], '固定A/B顺序变化')
    require(len({r['instance_id'] for r in chosen}) == 1, '固定任务身份变化')
    wanted = {r['session_id']: r for r in chosen}
    found = {}
    offset = 0
    for batch in pq.ParquetFile(data).iter_batches(batch_size=128):
        for i, row in enumerate(batch.to_pylist()):
            if row['session_id'] in wanted:
                require(row['session_id'] not in found, '源Session不唯一')
                found[row['session_id']] = (offset+i, row)
        offset += batch.num_rows
    require(set(found) == set(wanted), '固定来源缺失，不补位')
    prior_requests = read(PRIOR / 'requests.json')
    rows = []
    evidence = []
    source_paths = [PRIOR / 'source_selection_lock.json', PRIOR / 'requests.json', PRIOR / 'input_configuration_lock.json', ROOT / 'evaluation/openhands_single_workflow_smoke.py']
    configuration = config['b4_identity']
    configuration_digest = __import__('flowstate.strict_epoch', fromlist=['_digest'])._digest(configuration)
    for selection in chosen:
        label = selection['label']
        index, row = found[selection['session_id']]
        require(index == selection['parquet_row_index_zero_based'], '原始记录索引变化')
        require(json.loads(row['ground_truth_meta_json'])['instance_id'] == selection['instance_id'], '任务身份变化')
        raw = json.loads(row['messages_json'])
        history, cut = causal_cut(raw)
        require(cut == selection['pending_cut_index'], 'Pending截断位置变化')
        original = PRIOR / selection['source_history_file']
        require(sha(original) == selection['source_history_sha256'] and read(original) == history, '原始消息与冻结截断不同')
        source_paths.append(original)
        links = tool_causality(history)
        save_new(OUT / f'{label}_source_messages.json', history)
        histories = input_histories(history)
        prior_request = None
        for turn, messages in enumerate(histories, 1):
            old = next(r for r in prior_requests if r['label'] == label and r['turn'] == turn)
            f = PRIOR / old['token_file']
            require(sha(f) == old['file_sha256'], '冻结Token文件摘要不符')
            tokens = read(f)
            require(len(tokens) == old['token_count'], 'Token长度变化')
            token_sha = __import__('hashlib').sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
            require(token_sha == old['canonical_token_sha256'], 'Token内容摘要变化')
            item = {'request_id': old['request_id'], 'workflow_id': selection['session_id'], 'label': label, 'turn': turn, 'kind': 'PENDING' if turn == 3 else 'HISTORY', 'raw_messages': messages, 'token_ids': tokens, 'token_count': len(tokens), 'token_sha256': token_sha, 'configuration_digest': configuration_digest, 'source_message_indices': list(range(len(messages))), 'source_session_id': selection['session_id'], 'source_row': index, 'previous_request_id': prior_request, 'fork_parent_request_id': None, 'branch_id': selection['session_id'], 'branch_semantics': '独立真实执行；没有原生Fork证据', 'source_wall_clock': 'UNKNOWN'}
            PreparedInput(item['request_id'], item['workflow_id'], tuple(tokens), token_sha, configuration_digest).validate()
            rows.append(item)
            prior_request = old['request_id']
            source_paths.append(f)
        evidence.append({'label': label, 'session_id': selection['session_id'], 'task': selection['instance_id'], 'source_row': index, 'cut_index': cut, 'role_order': [m['role'] for m in history], 'tool_links': links, 'pending_output_consumed': False, 'later_messages_consumed_by_builder': False, 'original_timing': 'UNKNOWN'})
    for item in config['b4_identity']['tokenizer_files'] + config['b4_identity']['model_files']:
        if 'sha256' in item:
            require(sha(item['path']) == item['sha256'], '目标配置或Tokenizer摘要变化')
            source_paths.append(Path(item['path']))
    history_order = [f'B5C2_{label}_TURN_{turn:03d}' for turn in (1, 2) for label in 'AB']
    prepared = {'workload_type': 'Trace-derived Controlled Composition', 'run_scope': 'AB_ONLY', 'configuration': configuration, 'configuration_digest': configuration_digest, 'history': [next(r for r in rows if r['request_id'] == rid) for rid in history_order], 'pendings': [r for r in rows if r['kind'] == 'PENDING'], 'source_relation': '同任务的两次独立执行，不是原生Fork', 'gpu_started': False, 'current_physical_state': 'UNKNOWN', 'current_strict_epoch': None, 'current_allocation_plan': None, 'pending_dispatch_allowed': False}
    save_new(OUT / 'prepared_inputs.json', prepared)
    gate = PreparedPendingGate([PreparedInput(p['request_id'], p['workflow_id'], tuple(p['token_ids']), p['token_sha256'], configuration_digest) for p in prepared['pendings']], journal=OUT / 'pending_events.jsonl')
    gate.prepare()
    save_new(OUT / 'pending_boundary.json', {'state': gate.stage, 'pending_ids': [r.request_id for r in gate.requests], 'events': gate.events, 'allocation_executed': False, 'reconciliation_executed': False, 'requests_submitted': 0, 'native_resume_validated': False})
    save_new(OUT / 'source_identity.json', {'source_path': str(data), 'source_sha256': source['source_sha256'], 'records': evidence, 'files': {str(p): sha(p) for p in source_paths}, 'input_config_lock_sha256': sha(PRIOR / 'input_configuration_lock.json'), 'scope': '真实消息与冻结Token身份；不认证GPU物化'})
    print('A/B六份完整请求及Prepared-Pending证据保存完成；未发送任何请求。')


if __name__ == '__main__':
    main()
