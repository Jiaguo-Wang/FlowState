"""仅读取已有五个分片，以文本结构冻结样本，不读取任何token收益。"""
import json,hashlib,itertools
from pathlib import Path
from collections import defaultdict
OUT=Path(__file__).resolve().parent
FILES=[f'/tmp/lmcache_check/{i}.parquet' for i in range(5)]
SEED='20261013'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def lcp(a,b):
 for i,(x,y) in enumerate(zip(a,b)):
  if x!=y:return i
 return min(len(a),len(b))
import pyarrow.parquet as pq
best={}
for file in FILES:
 offset=0
 for batch in pq.ParquetFile(file).iter_batches(batch_size=16):
  for row,v in enumerate(batch.to_pylist(),offset):
   sid=v['session_id'];key=(len(v['input']),-FILES.index(file),row)
   if sid not in best or key>best[sid][0]:best[sid]=(key,file,row,v['model'])
  offset+=batch.num_rows
groups=defaultdict(list)
for sid,(key,file,row,model) in best.items():groups[sid.rsplit('__',1)[0]].append(dict(session_id=sid,model=model,file=file,row=row))
needed={}
for task,ss in groups.items():
 if len({s['model'] for s in ss})<2:continue
 ss.sort(key=lambda s:s['session_id']);a,b=next((a,b) for a,b in itertools.combinations(ss,2) if a['model']!=b['model'])
 for x in [a,b]:needed[(x['file'],x['row'])]=x
for file in FILES:
 offset=0
 for batch in pq.ParquetFile(file).iter_batches(batch_size=16):
  for row,v in enumerate(batch.to_pylist(),offset):
   if (file,row) in needed:needed[(file,row)]['messages']=v['input']
  offset+=batch.num_rows
rows=best
meta=[];pairs={}
for task,ss in sorted(groups.items()):
 if len({s['model'] for s in ss})<2:continue
 ss=sorted(ss,key=lambda s:s['session_id']);a,b=next((a,b) for a,b in itertools.combinations(ss,2) if a['model']!=b['model'])
 x=json.dumps(a['messages'],ensure_ascii=False,separators=(',',':'));y=json.dumps(b['messages'],ensure_ascii=False,separators=(',',':'))
 same=0
 for m,n in zip(a['messages'],b['messages']):
  if m!=n:break
  same+=1
 meta.append(dict(task_id=task,sessions=[a['session_id'],b['session_id']],total_sessions=len(ss),char_lcp=lcp(x,y),same_messages=same,max_chars=max(len(x),len(y)),message_counts=[len(a['messages']),len(b['messages'])]));pairs[task]=[a,b]
assert len(rows)==767 and len(groups)==629 and len(meta)==120
rank=sorted(meta,key=lambda x:(x['char_lcp'],x['task_id']));chosen=[]
for q in range(4):
 bucket=rank[q*30:(q+1)*30];picked=sorted(bucket,key=lambda x:hashlib.sha256((SEED+x['task_id']).encode()).hexdigest())[:3]
 for x in picked:chosen.append((x,'文本前缀四分位'+str(q+1)))
for x in sorted(meta,key=lambda x:(-x['max_chars'],x['task_id'])):
 if x['task_id'] not in {v['task_id'] for v,label in chosen}:chosen.append((x,'长会话补充'))
 if len(chosen)==16:break
(OUT/'inputs').mkdir(exist_ok=True)
selection=[]
for i,(m,label) in enumerate(chosen,1):
 cid=f'lmcache_{i:02d}';p=OUT/'inputs'/f'{cid}_messages.json';p.write_text(json.dumps(dict(case_id=cid,task_id=m['task_id'],sessions=pairs[m['task_id']]),ensure_ascii=False)+'\n');selection.append(dict(case_id=cid,stratum=label,task_id=m['task_id'],input_path=str(p),input_sha256=sha(p)))
(OUT/'all_120_structure.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
(OUT/'selection_lock.json').write_text(json.dumps(dict(seed=SEED,rule='每session取消息数最多的已有输入，同数按filename升序、row降序；每task取session字典序最早的异模型对。120组按字符前缀四分位，每层按seed+task的SHA256取3组，再补4个最长会话。先冻结再tokenize，不替补失败。',sessions=767,tasks=629,multi_tasks=120,selected=selection,parquet_sha256={p:sha(p) for p in FILES}),ensure_ascii=False,indent=2)+'\n')
print('冻结样本',len(selection))
