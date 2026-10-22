"""只解析冻结结果与结构，不导入或调用任何分配策略。"""
import csv,hashlib,itertools,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RAW=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01'
B4=ROOT/'evaluation/strict_compatibility_b4c';B5=ROOT/'evaluation/sota_baseline_audit_b5a'
EPS=1e-9
sources={};checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    sources[str(p)]=sha(p)
    return json.loads(p.read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def main():
    for root,name in [(RAW,'manifest.json'),(B4,'evidence_manifest.json')]+[(RAW/f'g{i:03d}','manifest.json') for i in range(1,5)]:
        m=read(root/name)
        for rel,h in m['files'].items():
            p=Path(rel) if Path(rel).is_absolute() else root/rel
            assert sha(p)==h,str(p)
        checks.append({'manifest':str(root/name),'files':len(m['files']),'status':'PASS'})
    lock=read(B4/'input_integrity.json');mismatches=[]
    for p,h in lock['sha256'].items():
        actual=sha(Path(p));sources[p]=actual
        if actual!=h:mismatches.append({'path':p,'frozen_sha256':h,'current_sha256':actual})
    for key,commit in [('marconi','08016617b1524e6bf6ac29b680641cc945bda7f0'),('kvflow','7ef897eaedc16626a281ccfe0d453fae80251e8f')]:
        path=B5/'raw'/f'{key}_repo'
        head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
        assert head==commit and not dirty
        checks.append({'repository':str(path),'commit':head,'status':'PASS'})
    r=read(B4/'results.json');assert r['status']=='PASS'
    batch=read(RAW/'batch_summary.json');assert [x['group'] for x in batch['groups'] if x['decision']=='入选']==[1,2,3,4]
    structures=[];conditions=[];access=[];equal=[]
    for x in r['snapshots']:
        group=x['group'];assert group in ['g001','g002','g003','g004']
        s=read(RAW/group/'snapshot.json');summary=read(RAW/group/'summary.json')
        assert sha(RAW/group/'snapshot.json')==summary['snapshot_sha256']
        cs={c['checkpoint_id']:c for c in s['candidates']};ps=s['pendings'];bm=x['benefit_matrix_ms']
        pids=[p['identifier'] for p in x['input']['pending']];assert pids==[p['continuation_id'] for p in ps]
        assert set(bm)==set(cs) and len(s['edges'])==len(cs)*len(ps)
        edges={(e['checkpoint_id'],e['continuation_id']):e['status'] for e in s['edges']}
        support={c:[p for i,p in enumerate(pids) if b[i]>EPS] for c,b in bm.items()}
        for c,b in bm.items():
            for i,value in enumerate(b):
                assert value>=0
                if value>EPS:assert edges[c,pids[i]]=='EPOCH_ADMISSIBLE'
                if edges[c,pids[i]]=='REJECTED':assert value==0
        byp={p:[c for c in cs if p in support[c]] for p in pids}
        maxima={p:max([cs[c]['token_position_L'] for c in byp[p]]+[0]) for p in pids}
        tops={p:[c for c in byp[p] if cs[c]['token_position_L']==maxima[p]] for p in pids}
        kmin=None;witness=None
        for k in range(len(ps)+1):
            witness=next((list(sub) for sub in itertools.combinations(cs,k) if all(max([cs[c]['token_position_L'] for c in sub if p in support[c]]+[0])==maxima[p] for p in pids)),None)
            if witness is not None:kmin=k;break
        assert kmin==x['K_min']
        dominated=[{'dominated':a,'dominating':b} for a,b in itertools.permutations(cs,2) if all(v<=w+EPS for v,w in zip(bm[a],bm[b])) and any(v<w-EPS for v,w in zip(bm[a],bm[b]))]
        node_candidates={c['state_identity']['node_id']:cid for cid,c in cs.items()}
        events=[]
        orders={row['rid']:row['request_ordinal'] for row in s['history_request_rows']}
        for h in s['history']:
            order=orders[h['request']['rid']]
            for e in h['native_match_and_copy']['records']:
                cid=node_candidates.get(e['best_match_node'])
                if cid and e['cow_requested'] and e['cow_source_slots']==cs[cid]['state_identity']['target_mamba_slots'] and e['executable_prefix']==cs[cid]['token_position_L']:
                    events.append({'group':group,'ordinal':order,'request_id':h['request']['rid'],'checkpoint_id':cid,'cross_workflow':h['request']['workflow_id']!=cs[cid]['workflow_id'],'monotonic_ns':e['monotonic_ns']})
        access+=events
        structures.append({'group':group,'candidate_count':len(cs),'pending_count':len(ps),'admissible':sum(v=='EPOCH_ADMISSIBLE' for v in edges.values()),'unknown':sum(v=='UNKNOWN' for v in edges.values()),'positive_edges':sum(map(len,support.values())),'shared_candidates':[c for c,v in support.items() if len(v)>1],'positive_support':support,'positive_candidates_by_pending':byp,'cross_workflow_positive_edges':sum(cs[c]['workflow_id']!=ps[pids.index(p)]['workflow_id'] for c,v in support.items() for p in v),'K_min':kmin,'witness':witness,'maximum_frontiers':maxima,'maximum_frontier_candidates':tops,'domination_pairs':dominated,'structure':s['structure'],'history_count':len(s['history']),'native_mapped_accesses':len(events),'native_cross_workflow_accesses':sum(e['cross_workflow'] for e in events),'metadata':[{'checkpoint_id':cid,**c['metadata']} for cid,c in cs.items()]})
        for k in range(1,5):
            rows={z['policy']:z for z in x['rows'] if z['K']==k};opt=rows['Exact OPT']['J_ms']
            for name,z in rows.items():assert abs(z['gap_to_exact_ms']-(z['J_ms']-opt))<=EPS
            conditions.append({'group':group,'K':k,'K_min':kmin,'tight':k<kmin,'costs':{p:z['J_ms'] for p,z in rows.items()},'gaps':{p:z['J_ms']-opt for p,z in rows.items()},'selected':{p:z['selected'] for p,z in rows.items()},'optimal_policies':[p for p,z in rows.items() if abs(z['J_ms']-opt)<=EPS]})
            for a,b in itertools.combinations(rows,2):
                if abs(rows[a]['J_ms']-rows[b]['J_ms'])<=EPS and set(rows[a]['selected'])!=set(rows[b]['selected']):equal.append({'group':group,'K':k,'policy_a':a,'policy_b':b,'selected_a':rows[a]['selected'],'selected_b':rows[b]['selected'],'J_ms':rows[a]['J_ms']})
    write('structure_statistics.json',structures);write('condition_review.json',conditions);write('same_objective_different_sets.json',equal);write('native_access_evidence.json',access)
    write('integrity_review.json',{'checks':checks,'historical_input_hash_mismatches':mismatches,'说明':'当前生产源码变化单列；冻结结果与快照清单必须完整通过，不运行新策略'})
    with (OUT/'condition_policy_review.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['group','K','policy','J_ms','gap_to_exact_ms','selected_ids'])
        for c in conditions:
            for p in ('FlowState','PCB','WFC','LDR','Exact OPT'):w.writerow([c['group'],c['K'],p,c['costs'][p],c['gaps'][p],json.dumps(c['selected'][p])])
    for p in [B5/'source_lock.json',B5/'final_report.md',B5/'adaptation_contract.md',B5/'marconi_audit.md',B5/'kvflow_audit.md',ROOT/'evaluation/strict_compatibility_b4b0/capture.py',ROOT/'evaluation/rq3_openhands_neutral_collector.py',ROOT/'evaluation/sota_metadata.py',ROOT/'evaluation/strict_compatibility_b4a/final_report.md',Path('/home/wjg/models/qwen3.5-9b/config.json')]:sources[str(p)]=sha(p)
    write('source_hashes.json',sources)
    print('结构',[(x['group'],x['candidate_count'],x['positive_edges'],x['K_min']) for x in structures])
    print('最优条件数',{p:sum(p in c['optimal_policies'] for c in conditions) for p in ('FlowState','PCB','WFC')})
    print('PCB/WFC覆盖',sum(any(p in c['optimal_policies'] for p in ('PCB','WFC')) for c in conditions))
    print('输入历史摘要差异',mismatches)
    print('同目标异集合',len(equal),'跨Workflow实际历史命中',sum(e['cross_workflow'] for e in access))
if __name__=='__main__':main()
