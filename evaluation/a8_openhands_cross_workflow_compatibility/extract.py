"""流式读取已有消息，只保存冻结轮次以内的重建材料，不执行请求。"""
import hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
FORMAL=ROOT/'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017'
needs={};snapshots=[]
for p in sorted((FORMAL/'snapshots').glob('g*.json')):
    r=json.loads(p.read_text());s=json.loads(r['canonical_snapshot']);snapshots.append(str(p))
    for sid in r['session_ids']:needs[sid]=max(needs.get(sid,0),r['allocation_round']+1)
assert len(snapshots)==168
source=Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet');found={}
for batch in pq.ParquetFile(source).iter_batches(batch_size=8,columns=['session_id','messages_json']):
    for r in batch.to_pylist():
        sid=r['session_id']
        if sid not in needs:continue
        messages=json.loads(r['messages_json']);kept=[];turn=0
        for m in messages:
            if m.get('role')=='assistant':
                turn+=1
                if turn==needs[sid]:break
            kept.append(m)
        assert turn==needs[sid]
        assert sid not in found
        found[sid]=dict(messages=kept,pending_turn=turn)
assert set(found)==set(needs)
(OUT/'reconstruction_messages.json').write_text(json.dumps(found,ensure_ascii=False))
h=hashlib.sha256()
with source.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
(OUT/'extraction_identity.json').write_text(json.dumps(dict(source=str(source),sha256=h.hexdigest(),sessions=len(found),snapshots=snapshots,note='仅为当前消息重建材料，未认证完整历史 Pending token 身份'),ensure_ascii=False,indent=2))
print('已提取',len(found),'个会话的有界历史')
