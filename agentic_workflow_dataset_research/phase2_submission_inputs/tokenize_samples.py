"""用本地目标模型模板处理冻结样本，仅运行 CPU。"""
import csv
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
import transformers
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = Path('/home/wjg/models/qwen3.5-9b')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
def normalize(messages):
    result = []
    for original in messages:
        message = {k:v for k,v in original.items() if v is not None}
        for call in message.get('tool_calls', []):
            function = call['function']
            if isinstance(function.get('arguments'), str):
                function['arguments'] = json.loads(function['arguments'])
            if not isinstance(function.get('arguments', {}), dict):
                raise ValueError('工具参数不是映射，禁止静默转换')
        result.append(message)
    return result

tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
def tokens(messages, generation=True):
    value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
    return list(value['input_ids'] if isinstance(value, Mapping) else value)
lock = json.loads((ROOT/'selection_lock.json').read_text())
identity = {name:digest(MODEL/name) for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json']}
assert tokenizer.chat_template == (MODEL/'chat_template.jinja').read_text()
rows, manifests = [], []
for item in lock['selected']:
    source = Path(item['input_path'])
    assert digest(source) == item['input_sha256']
    original = json.loads(source.read_text())
    row = {'case_id':item['case_id'], 'task_id':item['task_id'], 'stratum':item['stratum']}
    try:
        sessions = original['sessions']
        messages = [normalize(s['messages']) for s in sessions]
        ids = [tokens(m) for m in messages]
        a,b = ids
        lcp = next((i for i,(x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)))
        common = next((i for i,(x,y) in enumerate(zip(*messages)) if x != y), min(map(len,messages)))
        boundaries = []
        for n in range(1, common+1):
            try:
                prefix = tokens(messages[0][:n], False)
            except Exception:
                # 部分消息前缀不是合法独立对话，不能据此否定完整输入。
                continue
            if len(prefix) <= lcp and a[:len(prefix)] == prefix:
                boundaries.append({'messages':n, 'tokens':len(prefix)})
        candidate = [lcp//256*256] if lcp >= 256 else []
        paths=[]
        for i in range(2):
            path=ROOT/'inputs'/f"{item['case_id']}_p{i+1}_tokens.json"
            save(path, {'session_id':sessions[i]['session_id'],'input_ids':ids[i], 'messages':messages[i]})
            paths.append(str(path))
        row.update(status='PASS', session_id_1=sessions[0]['session_id'],session_id_2=sessions[1]['session_id'], original_model_1=sessions[0]['model'],original_model_2=sessions[1]['model'],target_model=str(MODEL),p1_total_tokens=len(a),p2_total_tokens=len(b),prefix_tokens=lcp,prefix_ratio=lcp/min(len(a),len(b)),first_divergence_position=lcp if lcp<min(len(a),len(b)) else 'NONE',different_suffix=lcp<min(len(a),len(b)),common_messages=common,verified_message_boundaries=json.dumps(boundaries),checkpoint_candidate_positions=json.dumps(candidate),source_type='实验回放',prefix_scope='跨同源会话',tool_calls=sum(len(m.get('tool_calls',[])) for mm in messages for m in mm),tool_results=sum(m['role']=='tool' for mm in messages for m in mm),roles=','.join(sorted({m['role'] for mm in messages for m in mm})))
        manifests.append(dict(case_id=item['case_id'],dataset_name='sammshen/lmcache-agentic-traces',source_task_id=item['task_id'],source_session_ids=[s['session_id'] for s in sessions],source_trace_ids=[],source_type='实验回放',target_model=str(MODEL),tokenizer_revision={'hub_revision':'UNKNOWN','transformers':transformers.__version__,'files_sha256':identity},chat_template_identity=identity['chat_template.jinja'],tokenization_method='apply_chat_template(tokenize=True, add_generation_prompt=True)，不传 tools 或 enable_thinking，不截断',prefix_tokens=lcp,p1_total_tokens=len(a),p2_total_tokens=len(b),first_divergence_position=row['first_divergence_position'],checkpoint_candidate_positions=candidate,native_fork_evidence='UNKNOWN',pending_event_evidence='UNKNOWN',physical_checkpoint_evidence='UNKNOWN',input_artifact_paths=paths,input_sha256=[digest(Path(p)) for p in paths],evidence_paths=[str(source),str(ROOT/'selection_lock.json')],different_suffix=row['different_suffix']))
    except Exception as exc:
        row.update(status='FAIL',error=str(exc))
    rows.append(row)
    print(item['case_id'], row.get('prefix_tokens'),row.get('p1_total_tokens'),row.get('p2_total_tokens'),row['status'],flush=True)
keys=list(dict.fromkeys(k for row in rows for k in row))
with (ROOT/'prefix_statistics.csv').open('w') as f:
    writer=csv.DictWriter(f, fieldnames=keys);writer.writeheader();writer.writerows(rows)
with (ROOT/'runtime_candidate_manifest.jsonl').open('w') as f:
    for value in manifests:f.write(json.dumps(value,ensure_ascii=False)+'\n')
save(ROOT/'tokenizer_identity.json',{'model':str(MODEL),'class':type(tokenizer).__name__,'transformers':transformers.__version__,'files_sha256':identity})
