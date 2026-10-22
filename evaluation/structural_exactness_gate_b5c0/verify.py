"""独立验证单共享候选的两分支公式，只读取冻结矩阵和选择日志。"""
import csv,hashlib,json,math
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RAW=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01';B4=ROOT/'evaluation/strict_compatibility_b4c'
EPS=1e-9
sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):sources[str(p)]=sha(p);return json.loads(p.read_text())
def write(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def top(items,k):
    """正额外收益按值降序，同值按ID确定代表，不添加零收益项。"""
    return sorted((x for x in items if x[0]>0),key=lambda x:(-x[0],x[1]))[:max(0,k)]
def main():
    checks=[]
    for base,name in [(RAW,'manifest.json'),(B4,'evidence_manifest.json')]+[(RAW/f'g{i:03d}','manifest.json') for i in range(1,5)]:
        m=read(base/name)
        for name,h in m['files'].items():
            p=Path(name) if Path(name).is_absolute() else base/name
            assert sha(p)==h,str(p)
        checks.append({'path':str(base),'files':len(m['files']),'status':'PASS'})
    data=read(B4/'results.json');assert data['status']=='PASS'
    model=data['cost_model'];rows=[];contracts=[];failures=[]
    for x in data['snapshots']:
        g=x['group'];assert g in ['g001','g002','g003','g004']
        s=read(RAW/g/'snapshot.json');summary=read(RAW/g/'summary.json')
        assert sha(RAW/g/'snapshot.json')==summary['snapshot_sha256']
        ps=x['input']['pending'];ids=[p['identifier'] for p in ps];bm=x['benefit_matrix_ms'];cs={c['checkpoint_id']:c for c in s['candidates']}
        assert set(cs)==set(bm) and len(set(ids))==len(ids)==len(s['pendings'])
        assert all(math.isfinite(v) and v>=0 for values in bm.values() for v in values)
        support={c:[i for i,v in enumerate(values) if v>0] for c,values in bm.items()}
        shared=[c for c,v in support.items() if len(v)>1];assert shared==[f'RQ3_{g.upper()}_B_TURN_001'];b=shared[0]
        assert len(support[b])==4
        edges={(e['checkpoint_id'],e['continuation_id']):e for e in s['edges']};assert len(edges)==len(cs)*len(ids)
        sizes={c['metadata']['memory_bytes'] for c in cs.values()};assert len(sizes)==1 and next(iter(sizes))>0
        config={c['configuration_digest'] for c in cs.values()};barriers={c['barrier_id'] for c in cs.values()}
        assert len(config)==1 and barriers=={s['barrier_id']}
        def phi(gap,target):
            a=Decimal(str(model['coefficient_a']));bb=Decimal(str(model['coefficient_b']));c=Decimal(str(model['coefficient_c']))
            gg=Decimal(gap)/1024;tt=Decimal(target)/1024
            return a*gg+bb*gg*tt+c*gg*gg
        for i,p in enumerate(ps):
            original=next(z for z in s['pendings'] if z['continuation_id']==p['identifier'])
            assert p['target']==original['planning_target_T'] and original['barrier_id']==s['barrier_id'] and original['configuration_digest'] in config
            for c,values in bm.items():
                edge=edges[c,p['identifier']];assert edge['status'] in ['EPOCH_ADMISSIBLE','REJECTED']
                if edge['status']=='EPOCH_ADMISSIBLE':
                    assert all(v is True for v in edge['conditions'].values())
                    expected=float(phi(p['target'],p['target'])-phi(p['target']-cs[c]['token_position_L'],p['target']))
                else:expected=0
                assert abs(values[i]-expected)<=EPS,(g,c,p['identifier'])
        qr=[]
        for i,p in enumerate(ps):
            choices=sorted(((-bm[c][i],c) for c in cs if c!=b and support[c]==[i]))
            q,cid=(-choices[0][0],choices[0][1]) if choices else (0,None)
            r=bm[b][i];qr.append({'pending':p['identifier'],'q':q,'r':r,'residual':max(0,q-r),'best_exclusive':cid})
        def benefit(chosen):return sum(max([bm[c][i] for c in chosen]+[0]) for i in range(len(ps)))
        assert benefit(())==0
        contracts.append({'group':g,'shared':b,'candidate_count':len(cs),'pending_count':len(ps),'support_sizes':{c:len(v) for c,v in support.items()},'memory_bytes':next(iter(sizes)),'barrier_id':s['barrier_id'],'configuration_digest':next(iter(config)),'qr':qr,'snapshot_sha256':sha(RAW/g/'snapshot.json'),'J_empty_ms':x['J_empty_ms']})
        for k in range(1,5):
            t0=top([(z['q'],z['best_exclusive']) for z in qr if z['best_exclusive']],k)
            t1=top([(z['residual'],z['best_exclusive']) for z in qr if z['best_exclusive']],k-1)
            set0=[c for _,c in t0];set1=[b]+[c for _,c in t1]
            f0=sum(v for v,_ in t0);f1=sum(z['r'] for z in qr)+sum(v for v,_ in t1)
            assert abs(f0-benefit(set0))<=EPS and abs(f1-benefit(set1))<=EPS
            frozen={r['policy']:r for r in x['rows'] if r['K']==k};opt=frozen['Exact OPT']
            winner='F0' if f0>f1+EPS else 'F1' if f1>f0+EPS else 'TIE'
            best=max(f0,f1);j=x['J_empty_ms']-best
            assert abs(j-opt['J_ms'])<=EPS and abs(best-opt['F_ms'])<=EPS
            row={'group':g,'K':k,'shared_id':b,'F0_ms':f0,'J0_ms':x['J_empty_ms']-f0,'set0':set0,'F1_ms':f1,'J1_ms':x['J_empty_ms']-f1,'set1':set1,'winning_branch':winner,'F_opt_ms':best,'J_opt_ms':j,'frozen_exact_J_ms':opt['J_ms'],'difference_to_frozen_exact_ms':j-opt['J_ms']}
            for name in ['PCB','WFC','FlowState']:
                z=frozen[name];branch='F1' if b in z['selected'] else 'F0';targetset=set1 if branch=='F1' else set0
                row.update({name+'_set':z['selected'],name+'_branch':branch,name+'_J_ms':z['J_ms'],name+'_gap_to_exact_ms':z['J_ms']-opt['J_ms'],name+'_same_branch_set':set(z['selected'])==set(targetset),name+'_branch_optimal':abs(z['F_ms']-(f1 if branch=='F1' else f0))<=EPS})
            steps=x['traces'][f'{k}/FlowState'];assert [z['chosen'] for z in steps]==frozen['FlowState']['selected']
            for n,st in enumerate(steps):
                assert st['prefix']==frozen['FlowState']['selected'][:n]
                for c,delta in st['marginals_ms'].items():assert abs(delta-(benefit(st['prefix']+[c])-benefit(st['prefix'])))<=EPS
            rows.append(row)
            if frozen['FlowState']['gap_to_exact_ms']>EPS:
                assert steps[0]['chosen']==b and all(b not in st for st in opt['exact_opt_sets'])
                gp=frozen['FlowState']['per_pending'];ep=opt['per_pending']
                costs=[{'pending':a['continuation_id'],'greedy_E':a['E'],'greedy_G':a['G'],'greedy_cost_ms':a['cost_ms'],'exact_E':z['E'],'exact_G':z['G'],'exact_cost_ms':z['cost_ms'],'extra_cost_ms':a['cost_ms']-z['cost_ms']} for a,z in zip(gp,ep)]
                assert abs(sum(z['extra_cost_ms'] for z in costs)-frozen['FlowState']['gap_to_exact_ms'])<=EPS
                failures.append({'group':g,'K':k,'branches':row,'frozen_steps':steps,'exact_sets':opt['exact_opt_sets'],'per_pending_cost_difference':costs,'first_irreversible_departure_step':1,'shared_first_gain':steps[0]['marginals_ms'][b],'best_exclusive_first_gain':max(v for c,v in steps[0]['marginals_ms'].items() if c!=b)})
    assert len(rows)==16 and len(failures)==4
    with (OUT/'branch_verification.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in row.items()})
    write('structural_contract.json',contracts);write('branch_details.json',rows);write('failure_details.json',failures)
    for p,h in sources.items():assert sha(Path(p))==h,p
    write('verification.json',{'gate':'READY','conditions_verified':16,'failures_explained':4,'epsilon_ms':EPS,'maximum_absolute_error_ms':max(abs(r['difference_to_frozen_exact_ms']) for r in rows),'manifest_checks':checks,'source_sha256':sources,'说明':'独立两分支核验；未调用生产Optimizer或冻结策略，不测性能'})
    print('READY',[(r['group'],r['K'],r['winning_branch']) for r in rows]);print('误差',max(abs(r['difference_to_frozen_exact_ms']) for r in rows))
if __name__=='__main__':main()
