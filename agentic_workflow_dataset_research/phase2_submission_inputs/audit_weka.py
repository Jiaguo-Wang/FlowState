"""流式核对本地原始语料身份，只输出结构和摘要，不输出原始内容。"""
import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
raw=Path('/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl')
remote=json.loads((OUT/'official'/'weka_tree.json').read_text());expected=next(x for x in remote if x['path']=='traces.jsonl')['lfs']['oid']
h=hashlib.sha256();traces=[];counts={'invocations':0,'subagent_groups':0};representative=None
def visit(rs):
 for r in rs:
  if r.get('type')=='subagent':counts['subagent_groups']+=1;visit(r['requests'])
  else:counts['invocations']+=1
with raw.open('rb') as f:
 for i,line in enumerate(f,1):
  h.update(line);r=json.loads(line);visit(r['requests']);traces.append(r['id'])
  if r['id']=='02bc0afb13f7a2d9efa86c28511261d85c0e':
   representative=dict(trace_id=r['id'],source_line=i,record_sha256=hashlib.sha256(line).hexdigest(),scope=r['hash_id_scope'],block_size=r['block_size'],models=r['models'],requests_sha256=hashlib.sha256(json.dumps(r['requests'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
assert h.hexdigest()==expected
(OUT/'official'/'weka_tree.json').write_text(json.dumps(remote,ensure_ascii=False,indent=2)+'\n')
(OUT/'weka_identity.json').write_text(json.dumps(dict(source=str(raw),local_sha256=h.hexdigest(),official_lfs_sha256=expected,whole_file_identical=True,traces=len(traces),trace_ids=traces,**counts,representative=representative),ensure_ascii=False,indent=2)+'\n')
print('全文件摘要一致',len(traces),counts)
