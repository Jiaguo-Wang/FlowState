"""无Runtime依赖的输入锁、证据准入、分层和不可变边界合同。"""
import hashlib
import json
from pathlib import Path
from evaluation.strict_compatibility_b1c.protocol import token_digest, require
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
B1=ROOT/'evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json'
PROTOCOL=ROOT/'evaluation/strict_compatibility_b4a/sampling_protocol.json'
MANIFEST=ROOT/'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/population_manifest.json'
DATA=Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet')
GROUPS=tuple(range(1,7))

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_new(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def append_event(out,event,**fields):
    import time
    with (Path(out)/'audit.jsonl').open('a') as f:f.write(json.dumps({'event':event,'unix_ns':time.time_ns(),**fields},ensure_ascii=False)+'\n')
def verify_hashes(hashes):
    for p,h in hashes.items():require(sha(p)==h,'摘要不一致：'+p)
def verify_lock():
    lock=read(HERE/'execution_lock.json');verify_hashes(lock['sha256']);return lock

def group_contract(ordinal,manifest=None):
    require(ordinal in GROUPS,'只允许g001至g006')
    m=manifest or read(MANIFEST)
    row=next(g for g in m['groups'] if g['group_ordinal']==ordinal)
    require(row['allocation_round']==(2,3,4,5)[ordinal%4],'round变化')
    require(row['session_ids']==m['ordered_session_ids'][4*ordinal:4*ordinal+4],'分组或顺序变化')
    require(len(set(row['session_ids']))==4,'组内Session重复')
    return row

def safe_barrier(state,native_eviction=False):
    s=state.get('scope',{})
    require(s.get('scheduler_fully_idle') is True and s.get('running_requests')==0 and s.get('waiting_requests')==0 and s.get('chunked_request_present') is False,'屏障非静止或缺失证据')
    require(state.get('observation_noninterference') is True,'观测干扰或缺失证据')
    require(not native_eviction,'出现非预期Native Eviction')

def token_identity(row):
    t=row.get('token_ids');d=row.get('token_prefix_sha256')
    if t is None or d is None:return None
    return isinstance(t,list) and all(type(x) is int for x in t) and token_digest(t)==d

def edge(c,p,state,alias,config_digest):
    """必要条件未知时保留UNKNOWN；明确违例优先标为REJECTED。"""
    identity=c.get('state_identity',{});length=c.get('token_position_L');target=p.get('planning_target_T')
    nodes={n['node_id']:n for n in state.get('nodes',[])};node=nodes.get(identity.get('node_id'))
    comp=state.get('components');slots=identity.get('target_mamba_slots')
    complete=None
    if comp is not None and slots is not None:
        tensors=comp.get('conv',[])+([comp['temporal']] if comp.get('temporal') else [])
        complete=bool(comp.get('conv') and comp.get('temporal') and slots) and all(len(t.get('shape',[]))>1 and all(0<int(s)<t['shape'][1] for s in slots) for t in tensors)
    refs=identity.get('references');required_refs={'target_full_lock_ref','target_mamba_lock_ref','target_mamba_host_lock_ref','target_mamba_session_ref'}
    ref_ok=None if refs is None or not required_refs<=set(refs) or any(v is None for v in refs.values()) else all(v==0 for v in refs.values())
    conditions={
        'candidate_token_identity':token_identity(c),'pending_token_identity':token_identity(p),
        'L_le_T':None if length is None or target is None else 0<=length<=target,
        'exact_token_prefix':None if c.get('token_ids') is None or p.get('token_ids') is None or length is None else len(c['token_ids'])==length and c['token_ids']==p['token_ids'][:length],
        'same_configuration':None if not config_digest or not c.get('configuration_digest') or not p.get('configuration_digest') else c['configuration_digest']==p['configuration_digest']==config_digest,
        'same_boundary':None if not c.get('barrier_id') or not p.get('barrier_id') else c['barrier_id']==p['barrier_id'],
        'known_pending':p.get('controlled_input_ready'),
        'full_path_resident':identity.get('path_full_all_present'),'target_full_resident':identity.get('target_full_present'),
        'recurrent_resident':identity.get('target_mamba_present'),
        'complete_components':complete,'references_clear':ref_ok,'no_slot_alias':not alias,
        'physical_identity':None if node is None or length is None or c.get('token_ids') is None or c.get('token_prefix_sha256') is None or slots is None else node.get('position')==length and node.get('token_ids')==c.get('token_ids') and node.get('prefix_digest')==c.get('token_prefix_sha256') and node.get('mamba_slots')==slots,
    }
    status='REJECTED' if any(x is False for x in conditions.values()) else 'UNKNOWN' if any(x is None for x in conditions.values()) else 'EPOCH_ADMISSIBLE'
    return {'checkpoint_id':c['checkpoint_id'],'continuation_id':p['continuation_id'],'status':status,'conditions':conditions,'reasons':[k for k,v in conditions.items() if v is not True],'resume_validated':False}

def build_edges(candidates,pendings,state,digest):
    require(len({c['checkpoint_id'] for c in candidates})==len(candidates),'候选ID重复')
    require(len({p['continuation_id'] for p in pendings})==len(pendings),'Pending ID重复')
    slots=[s for c in candidates for s in c.get('state_identity',{}).get('target_mamba_slots',[])]
    duplicate={s for s in slots if slots.count(s)>1}
    return [edge(c,p,state,bool(duplicate.intersection(c.get('state_identity',{}).get('target_mamba_slots',[]))),digest) for c in candidates for p in pendings]

def structure_labels(candidates,pendings,edges):
    """只读取位置与准入边；不读取Phi、收益或策略结果。"""
    pos={c['checkpoint_id']:c['token_position_L'] for c in candidates}
    support={cid:{e['continuation_id'] for e in edges if e['checkpoint_id']==cid and e['status']=='EPOCH_ADMISSIBLE' and pos[cid]>0} for cid in pos}
    targets={p['continuation_id']:p['planning_target_T'] for p in pendings}
    labels=[]
    if any(pos[a]<pos[b] and support[a]&support[b] and support[c]-support[a] for a in pos for b in pos for c in pos):labels.append('A')
    if any(len(support[a])>=2 and len(support[b])==1 and support[b]<support[a] and pos[b]>pos[a] for a in pos for b in pos):labels.append('B')
    if any(support[a] and support[b] and not support[a]&support[b] and pos[a]!=pos[b] and any(targets[p]!=targets[q] for p in support[a] for q in support[b]) for a in pos for b in pos):labels.append('C')
    return {'all_labels':labels,'primary':labels[0] if labels else None}

def freeze_snapshot(out,snapshot):
    path=Path(out)/'snapshot.json';write_new(path,snapshot);return sha(path)
def unchanged_snapshot(out,digest):require(sha(Path(out)/'snapshot.json')==digest,'Pending执行污染冻结Snapshot')
def finalize_manifest(out):
    out=Path(out)
    write_new(out/'manifest.json',{'files':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
