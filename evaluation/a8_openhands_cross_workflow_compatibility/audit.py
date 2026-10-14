"""重建并逐 token 比较，严格分开重建结果与历史输入认证。"""
import ast,csv,hashlib,json
from array import array
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import transformers
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
FORMAL=ROOT/'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017';MODEL=Path('/home/wjg/models/qwen3.5-9b')
hashes={}
def read(p):
    p=Path(p);b=p.read_bytes();hashes[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(ids):return hashlib.sha256(array('q',ids).tobytes()).hexdigest()
def csvsave(n,rs):
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
# 只提取纯消息规范化函数，避免导入任何 Runtime 模块。
source=ROOT/'evaluation/openhands_single_workflow_smoke.py';tree=ast.parse(source.read_text());funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_message','_parse_tool_calls']]
ns={'json':json,'Mapping':Mapping};exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),ns)
normalize=ns['normalize_message'];tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
def tokenize(messages,generation=True):
    r=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation)
    return list(r['input_ids'] if isinstance(r,Mapping) else r)
materials=read(OUT/'reconstruction_messages.json');inputs={};histories={}
for sid,data in materials.items():
    history=[];turn=0
    for raw in data['messages']:
        if raw.get('role')=='assistant':
            turn+=1;inputs[sid,turn]=tokenize(history);histories[sid,turn]=list(history)
        history.append(normalize(raw))
    turn+=1;assert turn==data['pending_turn'];inputs[sid,turn]=tokenize(history);histories[sid,turn]=history
print('有界重建完成',len(inputs),flush=True)
versions=read(FORMAL/'COLLECTOR_VERSION.json');source_checks=[]
for f in versions['source_files']:
    b=Path(f['path']).read_bytes();actual=hashlib.sha256(b).hexdigest();hashes[f['path']]=actual;source_checks.append(dict(path=f['path'],matches=actual==f['sha256']))
protocol=read(FORMAL/'protocol.json');edges=[];snaps=[];candidate_rows=[]
for sp in sorted((FORMAL/'snapshots').glob('g*.json')):
    payload=read(sp);s=json.loads(payload['canonical_snapshot']);ordinal=payload['group_ordinal'];round_=payload['allocation_round'];rt=read(FORMAL/f'runtime_correctness/g{ordinal:03d}.json');verdict=read(FORMAL/f'verdicts/g{ordinal:03d}.json')
    last=rt['census_rows'][-1];nodes={n['node_id']:n for n in last['resident_mamba_nodes']};ev={e['checkpoint_id']:e for e in s['runtime_evidence']};pending=s['pending_continuations']
    canonical_ok=hashlib.sha256(payload['canonical_snapshot'].encode()).hexdigest()==payload['snapshot_digest'];assert canonical_ok
    pend_lengths=all(len(inputs[p['workflow_id'],round_+1])==p['anchor_pos'] for p in pending)
    sr=dict(snapshot_id=s['snapshot_id'],ordinal=ordinal,candidates=len(s['eligible_candidates']),pending=len(pending),canonical_identity=canonical_ok,verdict=verdict['status'],pending_length_matches=pend_lengths,historical_pending_token_ids='NOT_ARCHIVED',historical_pending_input_digest='NOT_ARCHIVED',historical_transformers=protocol['environment']['transformers_version'],reconstruction_transformers=transformers.__version__,source=str(sp),barrier_source=str(FORMAL/f'runtime_correctness/g{ordinal:03d}.json'))
    for c in s['eligible_candidates']:
        name=c['checkpoint_id'];turn=int(name.rsplit('_',1)[1]);ids=inputs[c['workflow_id'],turn];pos=c['token_pos'];prefix=ids[:pos];e=ev[name];node=nodes.get(e['node_id']);match=digest(prefix)==e['runtime_identity_digest'];physical=bool(node and node['token_position']==pos and node['slots'] and c['recurrent_resident'] and c['fa_resident']);free=last['raw_accounting']['mamba_free_slots'];slotok=bool(node) and not any(slot in free for slot in node['slots'])
        candidate_rows.append(dict(snapshot_id=s['snapshot_id'],checkpoint_id=name,L=pos,node_id=e['node_id'],slots=json.dumps(node['slots'] if node else []),historical_digest=e['runtime_identity_digest'],reconstructed_digest=digest(prefix),prefix_digest_match=match,barrier_resident=physical,slots_not_free=slotok,aggregate_protected=last['raw_accounting']['mamba_protected'],generation='UNKNOWN',individual_lock_flags='NOT_ARCHIVED',source_request_turn=turn,source_workflow=c['workflow_id']))
        for p in pending:
            target=inputs[p['workflow_id'],round_+1];T=min(p['anchor_pos'],p['resident_fa_frontier']);cross=c['workflow_id']!=p['workflow_id'];lcp=next((i for i,(x,y) in enumerate(zip(prefix,target)) if x!=y),min(len(prefix),len(target)));token_equal=len(prefix)==pos and lcp==pos;within=pos<=T
            old=c['workflow_id']==p['workflow_id'] and tuple(p['lineage_path'][:len(c['lineage_path'])])==tuple(c['lineage_path']) and within
            kind='无完整候选前缀匹配'
            if token_equal:
                ha=histories[c['workflow_id'],turn];hb=histories[p['workflow_id'],round_+1];common=next((i for i,(x,y) in enumerate(zip(ha,hb)) if x!=y),min(len(ha),len(hb)));boundaries=[]
                for count in range(1,common+1):
                    try: pref=tokenize(ha[:count],False)
                    except Exception:continue
                    if len(pref)<=pos and prefix[:len(pref)]==pref:boundaries.append(count)
                if any(any(m['role'] in ['assistant','tool'] for m in ha[:n]) for n in boundaries):kind='重建：更长共同执行历史'
                elif any(any(m['role']=='user' for m in ha[:n]) for n in boundaries):kind='重建：共享完整用户消息'
                else:kind='重建：模板、system 或不完整交互片段，未认证完整交互'
            edges.append(dict(snapshot_id=s['snapshot_id'],checkpoint_id=name,pending_id=p['continuation_id'],cross_workflow=cross,L=pos,T=T,first_divergence_or_prefix_end=lcp,reconstructed_token_prefix=token_equal,L_le_T=within,candidate_digest_match=match,barrier_resident=physical,slot_not_free=slotok,old_compatible=old,reconstruction_potential_omission=cross and token_equal and within and match and physical and slotok,qualified_historical_edge=False,qualification='BLOCKED_PENDING_INPUT_IDENTITY' if cross else '未重新认证旧边',sharing_type=kind,model_identity='同历史 engine /model；不可变权重摘要未归档',tokenizer_identity='本次文件摘要固定；历史模板摘要未归档',pending_identity='仅重建及长度匹配；历史完整 token 摘要缺失',source=str(sp)))
    snaps.append(sr)
    print('核对',ordinal,flush=True)
cross=[e for e in edges if e['cross_workflow']];candidate_ok=sum(c['prefix_digest_match'] for c in candidate_rows)
summary=dict(status='BLOCKED',snapshots=len(snaps),pending_inputs=sum(s['pending'] for s in snaps),candidates=len(candidate_rows),all_edges=len(edges),cross_edges=len(cross),candidate_prefix_digest_matches=candidate_ok,pending_length_matching_snapshots=sum(s['pending_length_matches'] for s in snaps),reconstructed_cross_token_edges=sum(e['reconstructed_token_prefix'] for e in cross),reconstructed_cross_token_and_target_edges=sum(e['reconstructed_token_prefix'] and e['L_le_T'] for e in cross),reconstructed_cross_edges_with_physical_candidates=sum(e['reconstruction_potential_omission'] for e in cross),qualified_cross_edges=0,historical_cross_token_edge_count='UNKNOWN',unresolved_historical_cross_edges=len(cross),physical_resident_candidates=sum(c['barrier_resident'] for c in candidate_rows),source_checks=source_checks,sharing_types=dict(Counter(e['sharing_type'] for e in cross if e['reconstructed_token_prefix'])),note='BLOCKED 不表示共享不存在；重建边数不等于历史认证边数')
save('summary.json',summary);csvsave('snapshot_identity_audit.csv',snaps);csvsave('candidate_identity_audit.csv',candidate_rows);csvsave('edge_audit.csv',edges)
save('reconstruction_identity.json',dict(model=str(MODEL),transformers=transformers.__version__,tokenizer_class=type(tokenizer).__name__,files_sha256={f:hashlib.sha256((MODEL/f).read_bytes()).hexdigest() for f in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json']},historical_environment=protocol['environment'],historical_token_ids_archived=False))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());save('evidence_manifest.json',dict(source_sha256=hashes,source_integrity='PASS',gpu_used=False,selectors_run=False))
print(json.dumps(summary,ensure_ascii=False,indent=2))
