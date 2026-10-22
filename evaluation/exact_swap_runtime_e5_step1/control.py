"""内容审计独立RPC，轻量控制保持单一串行窗口与生产协调合同。"""
import json,os,time,threading,traceback,hashlib
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
from evaluation.lightweight_runtime_e1 import light_capture
from evaluation.multishare_runtime_gate_b5c3 import bridge as heavy_capture
from evaluation.runtime_control_closure_r2c0 import bridge as prior
from evaluation.runtime_control_closure_r2c1a.bridge import RecordedAdapter,selected_payload
from evaluation.exact_greedy_paired_gpu_fix_b5d4a import control as heavy
from .source_contract import choose_source
from . import planner
from flowstate.strict_control import bind,preflight,StrictRuntimeBridge,StrictReconciliationError
from flowstate.controller import StateController
require=prior.require
fa_map=heavy.fa_map
fa_payload=heavy.fa_payload
OUT=Path('/b5out')
REFERENCE=None

def equivalent(a,b):
    """只忽略边界标识和内容审计载荷；状态与条件逐项比较，不重新定义兼容。"""
    def canonical(x):
        x=json.loads(json.dumps(x))
        x.pop('epoch_id',None)
        for c in x['candidates']:
            c.pop('barrier_id',None);c.pop('payload',None)
        for p in x['pendings']:p.pop('barrier_id',None)
        return x
    require(canonical(a)==canonical(b),'轻重Strict输入或状态发生变化')

def certify(initial,inputs):
    raw=initial['strict_epoch'];origins=[]
    histories=[json.loads((OUT/os.environ['B5D4_RUN']/f'history_{i:02d}.json').read_text()) for i in range(1,9)]
    for c in raw['candidates']:
        seen=[h['request']['request_id'] for h in histories if any(n['node_id']==c['state_identity']['node_id'] and n['token_ids']==c['token_ids'] and n['slots'] for n in h['state']['before']['nodes'])]
        require(bool(seen),'缺少本生命周期来源线索')
        origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':seen[0]})
    data=heavy.encode(raw);physical=heavy.encode(initial)
    evidence=planner.FrozenEvidence(data,hashlib.sha256(data).hexdigest(),physical_snapshot=physical,physical_sha256=hashlib.sha256(physical).hexdigest())
    ep=planner.from_frozen(evidence,3,(initial,origins,inputs));prepared=planner.prepare(ep,evidence)
    require(planner.allowed(ep),'超出Exact冻结规模')
    return ep,prepared,origins

def reference(scheduler,request):
    """控制计时之前的独立重型认证，不执行状态变更，开销另计且保留整体总量。"""
    global REFERENCE
    started=time.monotonic_ns()
    initial=heavy_capture.observe_control(scheduler,{'final':True});require(initial.get('ok'),'独立重型观察失败')
    inputs=json.loads((OUT/'prepared_inputs.json').read_text())
    ep,prepared,origins=certify(initial,inputs);matrix=planner.benefits(prepared)
    plans={s:planner.finish(prepared,planner.solve(prepared,matrix,s)) for s in ('Swap','Exact')}
    plan=plans[os.environ['B5D4_SOLVER']];target=choose_source(initial['strict_epoch'],ep,plan,origins,inputs,os.environ['B5D4_SOLVER'])
    scheduler._r2c0_owner=threading.get_ident()
    try:
        view,_=prior.observe(scheduler,ep,initial['window'])
        index=fa_map(scheduler.tree_cache);all_indices=set(index.values());recomputed=set();d=hashlib.sha256()
        for pos,t in enumerate(target['pending']['token_ids']):
            d.update(int(t).to_bytes(8,'little'))
            if pos>=target['expected']['E'] and d.hexdigest() in index:recomputed.add(index[d.hexdigest()])
        protected=sorted(all_indices-recomputed)
        REFERENCE={'ok':True,'initial':initial,'plan':asdict(plan),'plans':{s:asdict(p) for s,p in plans.items()},'selected_payload':selected_payload(scheduler.tree_cache,view,plan,ep),'fa_map':index,'fa_content':fa_payload(scheduler.tree_cache,all_indices),'fa_protected_indices':protected,'fa_protected_payload':fa_payload(scheduler.tree_cache,protected),'wall_ms':(time.monotonic_ns()-started)/1e6}
        return REFERENCE
    finally:scheduler._r2c0_owner=None

def apply(scheduler,request):
    if os.environ['E1_MODE']=='heavy':return heavy.apply(scheduler,request)
    result={'ok':False,'status':'BLOCKED','operations':[],'timing':[],'observations':[],'phase':'start'}
    def timed(stage,fn):
        t=time.monotonic_ns();v=fn();result['timing'].append({'stage':stage,'ms':(time.monotonic_ns()-t)/1e6});return v
    try:
        initial=timed('metadata_observation',lambda:light_capture.observe_control(scheduler,{'final':True}));require(initial.get('ok'),'轻量观察失败')
        window=initial['window'];scheduler._r2c0_owner=threading.get_ident();prior.scope(scheduler)
        raw=initial['strict_epoch'];result['raw_epoch']=raw;result['physical_observation']=initial
        require(REFERENCE is not None,'缺少独立重型对照')
        equivalent(raw,REFERENCE['initial']['strict_epoch'])
        inputs=json.loads((OUT/'prepared_inputs.json').read_text())
        ep,prepared,origins=timed('strict_epoch_authentication',lambda:certify(initial,inputs));result.update(epoch=asdict(ep),origins=origins)
        matrix=timed('benefit_matrix',lambda:planner.benefits(prepared));solver=os.environ['B5D4_SOLVER']
        selected=timed('solver',lambda:planner.solve(prepared,matrix,solver))
        plan=timed('plan_build_validate',lambda:planner.finish(prepared,selected));result.update(plan=asdict(plan),solver=solver)
        require(plan.selected_ids==tuple(REFERENCE['plan']['selected_ids']) and plan.recovery_cost_ms==REFERENCE['plan']['recovery_cost_ms'],'轻重分配不同')
        target=choose_source(raw,ep,plan,origins,inputs,solver);result['target']=target
        def check():
            before,audit=prior.observe(scheduler,ep,window)
            require(asdict(before)==initial['before'] and audit['semantic']==initial['audit_before']['semantic'],'Apply前元数据发生变化')
            from evaluation.barrier_fa_frontier_control import inspect_resident_fa_frontier
            for p in raw['pendings']:
                require(inspect_resident_fa_frontier(scheduler.tree_cache,p['token_ids'])['resident_fa_frontier']==p['attention_frontier'],'FA前沿改变')
            b=bind(ep,plan,before,window);preflight(ep,plan,b,before,window)
            return before,audit,b
        before,audit,b=timed('apply_time_freshness_preflight',check)
        result.update(before=asdict(before),audit_before=audit,apply_validation={'status':'PASS','binding':asdict(b),'window':window,'same_scheduler_call':True})
        @contextmanager
        def held():
            prior.scope(scheduler)
            yield window
            prior.scope(scheduler)
        def observe():
            view,a=prior.observe(scheduler,ep,window)
            require(a['state']['tree']['structure_sha256']==audit['state']['tree']['structure_sha256'],'协调改变FA树结构')
            return view
        def invalidate(ledger):scheduler._r2c0_failed=True
        bridge=StrictRuntimeBridge(RecordedAdapter(scheduler.tree_cache,result['operations']),held,observe,invalidate)
        result['phase']='reconciliation'
        ledger=timed('strict_reconciliation_with_postchecks',lambda:StateController.reconcile_strict(ep,plan,bridge))
        after,last=prior.observe(scheduler,ep,window)
        require([o['checkpoint_id'] for o in result['operations']]==list(plan.logical_eviction_ids) and all(o['returned'] for o in result['operations']),'驱逐不完整')
        index=fa_map(scheduler.tree_cache);require(index==REFERENCE['fa_map'],'FA索引改变')
        result.update(ok=True,status='READY',phase='complete',reconciliation=ledger,after=asdict(after),audit_after=last,fa_map_before=index,fa_protected_indices=REFERENCE['fa_protected_indices'],fa_protected_payload=REFERENCE['fa_protected_payload'],content_audit_scope='来自独立前置审计；本控制仅元数据验证，非当前内容读取证明')
        prior.scope(scheduler);return result,ep,plan
    except BaseException as exc:
        scheduler._r2c0_failed=True
        if isinstance(exc,StrictReconciliationError):result['reconciliation']=exc.ledger
        result.update(error=repr(exc),traceback=traceback.format_exc())
        return result,None,None
    finally:scheduler._r2c0_owner=None
