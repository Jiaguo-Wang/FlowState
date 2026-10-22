"""只读结构核查与既有求解代码计时，不启动 Runtime。"""
import ast
import array
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from evaluation.strict_compatibility_b4c import evaluate as old
from evaluation.strict_compatibility_b4b0.contracts import build_edges
OUT=Path(__file__).resolve().parent
RAW=old.RAW

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(array.array('q',x).tobytes()).hexdigest()
def write(name,data):
    with (OUT/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def validate_hashes():
    """核对最终执行锁、既有 Manifest 和当前实际文件内容。"""
    records=[];cache={}
    manifests=[(RAW/'execution_lock.json','sha256',ROOT),(RAW/'manifest.json','files',RAW),(old.OUT/'input_integrity.json','sha256',ROOT),(old.OUT/'evidence_manifest.json','files',ROOT)]
    for path,key,base in manifests:
        for name,h in read(path)[key].items():
            p=Path(name) if Path(name).is_absolute() else base/name
            if str(p) not in cache:cache[str(p)]=sha(p)
            records.append(dict(manifest=str(path),path=str(p),expected=h,actual=cache[str(p)],match=h==cache[str(p)]))
    assert sha(RAW/'execution_lock.json')==sha(ROOT/'evaluation/strict_compatibility_b4b0/execution_lock.json')
    assert all(r['match'] for r in records),'冻结内容发生变化，停止'
    return records,cache

def structure(group):
    """逐边验证必要性和充分性；历史缺失组件证据不推定为完整。"""
    if group==0:
        source=ROOT/'evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json'
        whole=read(source);s=whole['positive_arm'];cs=s['checkpoints'];ps=s['pendings'];edges=s['edges'];recomputed=None
        physical='UNDETERMINED'
        boundary=s['allocation_boundary']['concurrent_request_execution_during_capture'] is False
        config=bool(whole['configuration_identity']['engine_configuration_sha256'])
        note='B1旧准入只核对驻留标志与slot非空，未保存conv/temporal组件形状证据；配置为单Engine公共身份，非逐对象摘要。不能把旧布尔字段升级为B4完整合同。'
    else:
        source=RAW/f'g{group:03d}'/'snapshot.json';s=read(source);cs=s['candidates'];ps=s['pendings'];edges=s['edges']
        cfg=hashlib.sha256(json.dumps(s['configuration'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        config=all(x['configuration_digest']==cfg for x in cs+ps)
        recomputed=build_edges(cs,ps,s['barrier_start'],cfg)
        assert [e['conditions'] for e in recomputed]==[e['conditions'] for e in edges]
        physical='VALIDATED' if all(all(e['conditions'][key] is True for key in ['complete_components','references_clear','no_slot_alias','physical_identity','recurrent_resident']) for e in recomputed) else 'UNDETERMINED'
        boundary=all(s[b]['scope']['scheduler_fully_idle'] and s[b]['scope']['running_requests']==0 and s[b]['scope']['waiting_requests']==0 for b in ['barrier_start','barrier_end']) and s['barrier_start']['tree']==s['barrier_end']['tree'] and s['barrier_start']['accounting']==s['barrier_end']['accounting']
        note='限定TP=1静止边界的组件完整性与物理映射；slot generation和逐边实际恢复仍未认证。'
    cd={c['checkpoint_id']:c for c in cs};pd={p['continuation_id']:p for p in ps}
    assert len(edges)==len(cs)*len(ps)==len({(e['checkpoint_id'],e['continuation_id']) for e in edges})
    token_valid=all(isinstance(x.get('token_ids'),list) and digest(x['token_ids'])==x['token_prefix_sha256'] for x in cs+ps)
    pairs=[];supports={c:[] for c in cd};benefits={}
    for e in edges:
        c=cd[e['checkpoint_id']];p=pd[e['continuation_id']];l=c['token_position_L'];t=p['planning_target_T']
        ancestor=len(c['token_ids'])==l and 0<=l<=t<=len(p['token_ids']) and c['token_ids']==p['token_ids'][:t][:l]
        admitted=e['status']=='EPOCH_ADMISSIBLE'
        value=old.MODEL.estimate(t,t)-old.MODEL.estimate(t-l,t) if admitted else 0.
        benefits[(e['checkpoint_id'],e['continuation_id'])]=value
        if value>0:supports[e['checkpoint_id']].append(e['continuation_id'])
        pairs.append(dict(checkpoint_id=e['checkpoint_id'],pending_id=e['continuation_id'],L=l,T=t,token_ancestor=ancestor,status=e['status'],necessary=(not admitted or ancestor),sufficient=(not ancestor or admitted),failed_conditions=[k for k,v in e['conditions'].items() if v is not True],benefit_ms=value,evidence=str(source)))
    pair_types=[];violations=[];duplicates=[];dominance=[]
    for a,b in itertools.combinations(cs,2):
        ca=a['checkpoint_id'];cb=b['checkpoint_id'];sa=set(supports[ca]);sb=set(supports[cb])
        kind='EQUAL' if sa==sb else 'DISJOINT' if not sa&sb else 'NESTED' if sa<=sb or sb<=sa else 'VIOLATED'
        entry=dict(a=ca,b=cb,kind=kind);pair_types.append(entry)
        if kind=='VIOLATED':violations.append(entry)
        if a['token_ids']==b['token_ids'] and a['token_position_L']==b['token_position_L']:duplicates.append([ca,cb])
        shallow,deep=sorted([a,b],key=lambda c:c['token_position_L'])
        if shallow['token_ids']==deep['token_ids'][:shallow['token_position_L']]:
            for p in set(supports[ca])&set(supports[cb]):
                lo=benefits[(shallow['checkpoint_id'],p)];hi=benefits[(deep['checkpoint_id'],p)]
                dominance.append(dict(shallow=shallow['checkpoint_id'],deep=deep['checkpoint_id'],pending=p,shallow_benefit=lo,deep_benefit=hi,valid=hi>=lo-1e-9))
    necessary=all(p['necessary'] for p in pairs);sufficient=all(p['sufficient'] for p in pairs)
    sizes={c.get('metadata',{}).get('memory_bytes') for c in cs}
    contract='VALIDATED' if all([necessary,sufficient,token_valid,config,boundary,not violations,all(d['valid'] for d in dominance),physical=='VALIDATED',len(sizes)==1,None not in sizes]) else 'UNDETERMINED'
    if not necessary or not sufficient or violations or not all(d['valid'] for d in dominance):contract='VIOLATED'
    return dict(group=f'g{group:03d}',source=str(source),source_sha256=sha(source),candidates=len(cs),pendings=len(ps),pairs=pairs,supports=supports,shared_candidates=sum(len(v)>1 for v in supports.values()),pair_types=pair_types,laminar_violations=violations,duplicates=duplicates,depth_checks=dominance,checks=dict(token_identity=token_valid,configuration=config,boundary=boundary,necessity=necessary,sufficiency=sufficient,laminar=not violations,depth_dominance=all(d['valid'] for d in dominance),physical=physical,tree_contract=contract),说明=note)

def extract_exact():
    """从冻结函数抽取原有缓存构建与最优集合查询语句，不改写选择逻辑。"""
    tree=ast.parse((old.OUT/'evaluate.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='evaluate')
    prepare=[];query=[]
    for node in fn.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('all_sets','scores'):prepare.append(node)
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='k':
            for child in node.body:
                if isinstance(child,ast.Assign) and isinstance(child.targets[0],ast.Name) and child.targets[0].id in ('available','optimum','ties','exact'):query.append(child)
    assert len(prepare)==2 and len(query)==4
    return tuple(compile(ast.Module(body=x,type_ignores=[]),str(old.OUT/'evaluate.py'),'exec') for x in (prepare,query))

def stats(values):
    ordered=sorted(values);return statistics.mean(values),statistics.median(values),ordered[math.ceil(.95*len(values))-1]

def benchmark():
    """30次独立重复，缓存只在一次四预算批次内按既有规则复用。"""
    prepare,query=extract_exact();frozen={x['group']:x for x in read(old.OUT/'results.json')['snapshots']}
    rows=[];raw=[]
    for group in range(1,5):
        s,epoch,core=old.load(group);ids=tuple(c.identifier for c in epoch.candidates)
        bm=old.common.benefit_matrix(epoch);assert bm==frozen[f'g{group:03d}']['benefit_matrix_ms']
        optimizer=old.MatrixOptimizer(epoch);samples={k:dict(greedy=[],prepare=[],query=[],cold=[],amortized=[]) for k in range(1,5)}
        for repetition in range(-3,30):
            env=dict(common=old.common,subsets=old.subsets,ids=ids,epoch=epoch,EPS=old.EPS)
            greedy={};times={}
            def run_greedy():
                for k in range(1,5):
                    start=time.perf_counter_ns();answer=optimizer.select((),core,k*core[0].memory_bytes);stop=time.perf_counter_ns()
                    greedy[k]=tuple(c.checkpoint_id for c in answer.selected);times[k]=(stop-start)/1e6
            if repetition%2==0:run_greedy()
            start=time.perf_counter_ns();exec(prepare,env);stop=time.perf_counter_ns();prep=(stop-start)/1e6
            exact={};qt={}
            for k in range(1,5):
                env['k']=k;start=time.perf_counter_ns();exec(query,env);stop=time.perf_counter_ns()
                exact[k]=env['exact'];qt[k]=(stop-start)/1e6
            if repetition%2:run_greedy()
            for k in range(1,5):
                reference={r['policy']:r for r in frozen[f'g{group:03d}']['rows'] if r['K']==k}
                for name,selected in [('FlowState',greedy[k]),('Exact OPT',exact[k])]:
                    assert list(selected)==reference[name]['selected']
                    assert abs(old.common.score(epoch,selected)['cost_ms']-reference[name]['J_ms'])<1e-9
                if repetition>=0:
                    for key,value in [('greedy',times[k]),('prepare',prep),('query',qt[k]),('cold',prep+qt[k]),('amortized',prep/4+qt[k])]:samples[k][key].append(value)
                    raw.append(dict(group=group,K=k,repetition=repetition,greedy_ms=times[k],exact_prepare_ms=prep,exact_query_ms=qt[k]))
        for k,data in samples.items():
            reference={r['policy']:r for r in frozen[f'g{group:03d}']['rows'] if r['K']==k}
            row=dict(group=f'g{group:03d}',K=k,candidates=len(ids),eligible_subsets=sum(math.comb(len(ids),j) for j in range(k+1)),cached_subsets=len(env['all_sets']),repetitions=30,greedy_J_ms=reference['FlowState']['J_ms'],exact_J_ms=reference['Exact OPT']['J_ms'],optimality_gap_ms=reference['FlowState']['gap_to_exact_ms'])
            for key,values in data.items():
                for label,value in zip(['mean','median','p95'],stats(values)):row[f'{key}_{label}_ms']=value
            row['greedy_over_exact_cold_median']=row['greedy_median_ms']/row['cold_median_ms']
            rows.append(row)
        print('已完成CPU计时',group,flush=True)
    with (OUT/'solver_timing.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return raw

def main():
    before,hashes=validate_hashes();print('内容锁核验通过',len(before),flush=True)
    structures=[structure(i) for i in range(5)]
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]})
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0],allowed_affinity=affinity,benchmark_affinity=sorted(os.sched_getaffinity(0)),gc_enabled=gc.isenabled(),thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},timer='perf_counter_ns',warmup=3,repetitions=30,p95='最近秩法：升序第ceil(0.95n)项',说明='重复调用不是独立工作负载；无后台负载与频率隔离保证。')
    raw=benchmark()
    after={p:sha(p) for p in hashes};assert after==hashes,'执行期间输入发生变化'
    write('structure_audit.json',dict(snapshots=structures,integrity=before,inputs_unchanged=True,environment=environment,timing_samples=raw,script_sha256=sha(__file__),说明='g000仅结构；g001至g004计时；g005/g006只核对根Manifest完整性，不评估。'))
if __name__=='__main__':main()
