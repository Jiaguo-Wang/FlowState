"""只读检查消息层结构及已有证据；不构建严格快照或调用分配算法。"""
import json
import hashlib
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sources = {}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def record(path):
    path = Path(path)
    sources[str(path)] = {'exists': path.exists()}
    if path.is_file():
        sources[str(path)].update(size_bytes=path.stat().st_size, sha256=digest(path))

def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

path = Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet')
record(path)
rows = pq.read_table(path, filters=[('source_dataset', '=', 'nebius-swe-rebench-openhands')]).to_pylist()
nodes = {}
tasks = {}
message_fields = Counter()
meta_fields = Counter()
session_rows = []
# 完整消息对象逐项规范化；仅搜索内容结构，不将摘要当作实际模型输入。
for row in rows:
    sid = row['session_id']
    msgs = json.loads(row['messages_json'])
    meta = json.loads(row['ground_truth_meta_json'])
    meta_fields.update(meta.keys())
    assert [m.get('role') for m in msgs[:3]] == ['system','user','assistant']
    tasks.setdefault(str(meta.get('instance_id')), []).append(sid)
    parent = ''
    history = False
    last = max((i for i,m in enumerate(msgs) if m.get('role')=='assistant'), default=-1)
    for i,m in enumerate(msgs[:last]):
        message_fields.update(m.keys())
        history |= m.get('role') not in ('system','developer')
        encoded = json.dumps(m, ensure_ascii=False, sort_keys=True, separators=(',',':'))
        key = hashlib.sha256((parent+'\n'+encoded).encode()).hexdigest()
        if key not in nodes:
            nodes[key] = {'parent':parent,'message_count':i+1,'beyond_system':history,'sessions':[]}
        nodes[key]['sessions'].append(sid)
        parent = key
    session_rows.append({'session_id':sid,'source_id':row['source_id'],'instance_id':meta.get('instance_id'),'recorded_model':row['recorded_model'],'agent_framework':row['agent_framework'],'first_assistant_index':next((i for i,m in enumerate(msgs) if m.get('role')=='assistant'),None),'messages':len(msgs),'assistant_messages':sum(m.get('role')=='assistant' for m in msgs)})
shared = {k:v for k,v in nodes.items() if len(set(v['sessions']))>=2}
relations = []
for key,v in shared.items():
    parent = v['parent']
    while parent:
        if parent in shared and set(shared[parent]['sessions']) != set(v['sessions']):
            relations.append({'ancestor':parent,'descendant':key,'ancestor_session_count':len(shared[parent]['sessions']),'descendant_session_count':len(v['sessions'])})
            break
        parent = nodes[parent]['parent']
by_session = {r['session_id']:r for r in session_rows}
assert all(len({by_session[s]['instance_id'] for s in v['sessions']})==1 for v in shared.values() if v['message_count']==2)
save('openhands_shared_message_prefixes.json', shared)
save('openhands_nested_supports.json', relations)
summary_nodes = {'shared_depth_counts':dict(Counter(v['message_count'] for v in shared.values())), 'nested_pairs_with_beyond_system_ancestor':sum(shared[r['ancestor']]['beyond_system'] for r in relations)}
save('message_structure_summary.json', summary_nodes)
save('openhands_session_inventory.json', session_rows)
summary = {'openhands':{'sessions':len(rows),'unique_sessions':len(set(r['session_id'] for r in rows)), 'message_prefix_nodes':len(nodes),'shared_message_prefix_nodes':len(shared),'shared_beyond_system_nodes':sum(v['beyond_system'] for v in shared.values()),'nested_distinct_shared_support_pairs':len(relations),'message_fields':dict(message_fields),'metadata_fields':dict(meta_fields),'multi_session_task_groups':{k:v for k,v in tasks.items() if len(v)>1},'scope':'完整消息对象前缀；不同会话是结构叶子，不是同边界Pending；不进行分词或候选构建'}}
fi = ROOT/'evaluation/new_dataset_suitability_audit/session_metrics.jsonl'
record(fi)
metrics = [json.loads(x) for x in fi.read_text().splitlines()]
qualified = [r for r in metrics if r['prefix_forks']>=2]
save('freeinference_structural_leads.json', qualified)
summary['freeinference'] = {'sessions':len(metrics),'at_least_two_block_forks':len(qualified),'qualified_with_subagents':sum(r['subagent_sessions']>0 for r in qualified),'scope':'复用已有逐会话统计；两个分叉不证明同一时刻两个共享Checkpoint，更不是Token认证'}
invpath=ROOT/'evaluation/p2_shared_cohort_feasibility_audit/source_inventory.json'
record(invpath)
inv=json.loads(invpath.read_text())
other=[]
for src in inv['other_local_sources']:
    p=Path(src['path']); item={'path':str(p),'exists':p.exists(),'scope':src['scope']}
    if p.name=='step_records.json' and p.exists():
        record(p); data=json.loads(p.read_text());item['container_type']=type(data).__name__
        item['top_keys_or_count']=list(data) if isinstance(data,dict) else len(data)
        if isinstance(data,list):
            item['step_fields']=sorted({k for r in data if isinstance(r,dict) for k in r})
            item['agent_ids']=sorted({str(r.get('agent_id')) for r in data if isinstance(r,dict)})
    other.append(item)
summary['other_local_sources']=other
agent=Path(inv['primary_source']['path']);record(agent)
with agent.open() as f:
    first=json.loads(next(f))
summary['agentx_first_record']={'id':first.get('id'),'fields':sorted(first),'request_fields':sorted({k for r in first['requests'] for k in r}),'historical_sha_matches':sources[str(agent)]['sha256']==inv['primary_source']['sha256']}
for p in [ROOT/'evaluation/new_dataset_suitability_audit/final_report.md', ROOT/'evaluation/gatc_real_session_sample_audit/final_report.md', ROOT/'evaluation/structural_exactness_gate_b5c0/verification.json', ROOT/'evaluation/strict_compatibility_b4a/final_report.md', ROOT/'evaluation/p2_shared_cohort_feasibility_audit/final_report.md', ROOT/'evaluation/allocation_structural_properties/t2/final_report.md', ROOT/'agentic_workflow_dataset_research/phase2_submission_inputs/selection_lock.json', ROOT/'evaluation/rq3_openhands_population.py', ROOT/'evaluation/rq3_openhands_neutral_collector.py']:
    record(p)
save('audit_summary.json',summary)
save('source_manifest.json',sources)
print(json.dumps({k:v for k,v in summary['openhands'].items() if k not in ('multi_session_task_groups','message_fields','metadata_fields')},ensure_ascii=False))
print(json.dumps(summary['freeinference'],ensure_ascii=False))
