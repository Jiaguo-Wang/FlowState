"""用历史只读 FA 命中和唯一首段约束认证必要前缀，不认证完整 Pending 重建。"""
import ast,csv,hashlib,json
from collections.abc import Mapping
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
FORMAL=ROOT/'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017'
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvwrite(p,rs):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
rows=list(csv.DictReader((OUT/'edge_audit.csv').open()));certs=[]
for sp in sorted((FORMAL/'snapshots').glob('g*.json')):
    payload=read(sp);s=json.loads(payload['canonical_snapshot']);ordinal=payload['group_ordinal'];rt=read(FORMAL/f'runtime_correctness/g{ordinal:03d}.json');last=rt['census_rows'][-1];tree=last['raw_tree']['structure_rows'];roots=[n for n in tree if n[1] is None]
    assert len(roots)==1 and roots[0][3]==0
    root=roots[0];assert len(root[2])==1
    first=next(n for n in tree if n[0]==root[2][0]);length=first[3]
    ev={e['checkpoint_id']:e for e in s['runtime_evidence']};node=next(n for n in last['resident_mamba_nodes'] if n['node_id']==first[0]);assert node['token_position']==length
    candidates=[c for c in s['eligible_candidates'] if c['token_pos']==length and ev[c['checkpoint_id']]['node_id']==first[0]]
    assert len(candidates)==1 and length==2432
    c=candidates[0];assert c['recurrent_resident'] and c['fa_resident'];assert not set(node['slots'])&set(last['raw_accounting']['mamba_free_slots'])
    assert last['raw_accounting']['mamba_protected']==0 and rt['abort_reason'] is None and all(r['request_completed'] for r in rt['request_rows'])
    pcert=[]
    for p in s['pending_continuations']:
        R=p['resident_fa_frontier'];assert R>=length
        pcert.append(dict(pending_id=p['continuation_id'],recorded_FA_frontier=R,T=min(R,p['anchor_pos']),necessary_prefix_inference='根只有该首段；逐 token FA 遍历返回 R 不小于该段长度，故真实查询输入必须完整匹配该段',complete_pending_input_identity='UNKNOWN'))
    certs.append(dict(snapshot_id=s['snapshot_id'],candidate_id=c['checkpoint_id'],root_id=root[0],sole_first_segment_node=first[0],prefix_tokens=length,prefix_digest=ev[c['checkpoint_id']]['runtime_identity_digest'],slots=node['slots'],pending_certificates=pcert,physical_scope='历史 TP=1 串行 idle barrier，非现在驻留、非跨请求恢复实测',source=str(sp),runtime_source=str(FORMAL/f'runtime_correctness/g{ordinal:03d}.json')))
    for r in rows:
        if r['snapshot_id']!=s['snapshot_id']:continue
        proven=(r['checkpoint_id']==c['checkpoint_id'] and r['reconstructed_token_prefix']=='True' and r['candidate_digest_match']=='True' and r['L_le_T']=='True')
        r['historical_required_prefix_proven']=proven
        r['qualified_historical_edge']=bool(proven and r['cross_workflow']=='True')
        r['qualification']='HISTORICAL_REQUIRED_PREFIX_AND_RESIDENCY_CERTIFICATE' if r['qualified_historical_edge'] else ('REJECTED_L_GT_T' if r['L_le_T']=='False' else 'UNKNOWN_FULL_PENDING_IDENTITY')
        r['pending_identity']='完整输入未知；必要首段由历史 FA 查询与唯一首段联合认证' if proven else r['pending_identity']
# 仅检查共同首段内容类型，不把模板内容打印到报告。
mats=read(OUT/'reconstruction_messages.json');sample=certs[0];snap=read(sample['source']);canonical=json.loads(snap['canonical_snapshot']);candidate=next(c for c in canonical['eligible_candidates'] if c['checkpoint_id']==sample['candidate_id']);raw=mats[candidate['workflow_id']]['messages']
source=ROOT/'evaluation/openhands_single_workflow_smoke.py';tree=ast.parse(source.read_text());ns={'json':json,'Mapping':Mapping};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_message','_parse_tool_calls']],type_ignores=[]),str(source),'exec'),ns)
h=[]
for m in raw:
    if m.get('role')=='assistant':break
    h.append(ns['normalize_message'](m))
t=AutoTokenizer.from_pretrained('/home/wjg/models/qwen3.5-9b',local_files_only=True);render=t.apply_chat_template(h,tokenize=False,add_generation_prompt=True)
index=render.index('<|im_start|>user\n');system_end=len(t.encode(render[:index],add_special_tokens=False));header_end=len(t.encode(render[:index+len('<|im_start|>user\n')],add_special_tokens=False))
full=t.encode(render,add_special_tokens=False);assert t.encode(render[:index],add_special_tokens=False)==full[:system_end]
classification='仅共享 System Prompt 或模板' if 2432<=header_end else '含首条用户消息的不完整片段，未确认完整交互'
for r in rows:
    if r['qualified_historical_edge']:r['sharing_type']=classification
csvwrite(OUT/'edge_audit.csv',rows);save('historical_prefix_certificates.json',certs)
s=read(OUT/'summary.json');save('reconstruction_summary.json',s);s.update(status='SHARING_FOUND',qualified_cross_edges=sum(r['qualified_historical_edge'] is True for r in rows),historical_cross_token_edge_count='至少504条必要前缀已认证；其余不以重建代替原输入',unresolved_historical_cross_edges=sum(r['cross_workflow']=='True' and r['qualification']=='UNKNOWN_FULL_PENDING_IDENTITY' for r in rows),rejected_cross_L_gt_T=sum(r['cross_workflow']=='True' and r['qualification']=='REJECTED_L_GT_T' for r in rows),certified_sharing_type=classification,system_tokens=system_end,system_and_user_header_tokens=header_end,note='认证限必要的2432-token前缀与历史驻留，不宣称获得672条历史完整Pending token输入，不是原生fork或新GPU恢复认证')
save('summary.json',s)
print(json.dumps(s,ensure_ascii=False,indent=2))
