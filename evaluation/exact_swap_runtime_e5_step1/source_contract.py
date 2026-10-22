"""实验来源条件独立于严格兼容和物理门禁；本模块不操作Runtime。"""
from dataclasses import asdict
import os,json
from pathlib import Path
from flowstate.strict_control import validate_plan

def require(ok,message):
    if not ok:raise ValueError(message)

def choose_source(raw,epoch,plan,origins,inputs,solver):
    """调用方仍须执行生产Bind/Preflight；来源线索不能授权恢复或驱逐。"""
    require(solver in ('Swap','Exact'),'未知策略')
    validate_plan(epoch,plan)
    validate_frozen(raw,plan,solver)
    require(raw['epoch_id']==epoch.epoch_id,'原始Epoch身份不同')
    require(all(e['status'] in ('EPOCH_ADMISSIBLE','REJECTED') for e in raw['edges']),'UNKNOWN或未知关系状态')
    frozen={(e.candidate_id,e.pending_id):(e.status,tuple(e.conditions)) for e in epoch.edges}
    observed={(e['checkpoint_id'],e['continuation_id']):(e['status'],tuple(sorted(e['conditions'].items()))) for e in raw['edges']}
    require(len(observed)==len(raw['edges']) and observed==frozen,'原始与Strict关系不一致')
    ps=[p for p in raw['pendings'] if p['continuation_id']==os.environ['E4_PENDING']]
    require(len(ps)==1,'目标Pending不唯一')
    p=ps[0];ep=next(x for x in epoch.pendings if x.pending_id==p['continuation_id'])
    require(tuple(p['token_ids'])==ep.token_ids and p['planning_target_T']==ep.target,'Pending身份不一致')
    expected=next(x for x in plan.pending if x.pending_id==ep.pending_id)
    cs={c['checkpoint_id']:c for c in raw['candidates']};choices=[]
    for cid in sorted(plan.selected_ids):
        c=cs[cid];ec=next(x for x in epoch.candidates if x.candidate.checkpoint_id==cid)
        require(tuple(c['token_ids'])==ec.token_ids and c['token_position_L']==ec.candidate.token_pos,'候选身份不一致')
        if observed[cid,ep.pending_id][0]!='EPOCH_ADMISSIBLE' or c['token_position_L']!=expected.E:continue
        require(0<expected.E<=ep.target and tuple(p['token_ids'][:expected.E])==ec.token_ids,'恢复位置或Token前缀不合法')
        support=sum(c0==cid and value[0]=='EPOCH_ADMISSIBLE' for (c0,_),value in observed.items())
        require(support>=1,'选中恢复来源没有合法支持')
        found=[o for o in origins if o['checkpoint_id']==cid]
        require(len(found)==1,'首次观察线索不唯一')
        origin=found[0];hs=[h for h in inputs['history'] if h['request_id']==origin['first_observed_after']]
        require(len(hs)==1,'来源线索缺少历史请求')
        h=hs[0]
        provenance={'status':'PROVENANCE_UNVERIFIED','cross_workflow_certified':False,
            'first_observed_after':origin['first_observed_after'],'observed_workflow_hint':h['workflow_id'],
            'different_workflow_hint':h['workflow_id']!=p['workflow_id'],
            'requested_origin':'不推断原生物理血缘',
            'reason':'首次Census不证明张量创建、继承或COW血缘；本接口没有已验证的原生来源链'}
        choices.append({'pending':p,'source_checkpoint':cid,'source_history':h,
            'source_first_observed':origin,'expected':asdict(expected),'provenance':provenance})
    require(len(choices)==1,'固定Pending没有唯一已准入最深选中状态')
    return choices[0]

def validate_native_source(target,bound,actual):
    """仅核对实际匹配与本生命周期物理绑定，不认证张量来源血缘。"""
    require(actual['best_match_node']==bound['node_id'] and actual['cow_source_slots']==bound['slots'],'Native来源与当前绑定不一致')
    require(actual['executable_prefix']==target['expected']['E'] and actual['physical_fa_hit']==target['pending']['attention_frontier'],'实际恢复前沿不一致')
    require(actual['input_token_digest']==target['pending']['token_prefix_sha256'],'实际请求Token不一致')
    return {'native_source':'MATCHED','provenance':target['provenance'],
            'cross_workflow_resume_certified':False,
            'reason':'匹配认证与来源血缘认证分开；仍需原有COW、输出及状态后验检查'}


def validate_frozen(raw,plan,solver,root=None):
    """用完整Token身份对齐当前与冻结候选，拒绝语义漂移或计划偏离。"""
    root=Path('/b5out') if root is None else Path(root)
    frozen=json.loads((root/'frozen_snapshot.json').read_text());plans=json.loads((root/'frozen_plans.json').read_text())
    require(raw['configuration']==frozen['configuration'] and raw['configuration_digest']==frozen['configuration_digest'],'模型执行配置偏离冻结输入')
    old={tuple(c['token_ids']):c for c in frozen['candidates']};now={tuple(c['token_ids']):c for c in raw['candidates']}
    require(len(old)==len(frozen['candidates']) and len(now)==len(raw['candidates']) and old.keys()==now.keys(),'候选Token域不等价')
    mapping={c['checkpoint_id']:old[t]['checkpoint_id'] for t,c in now.items()}
    for tokens,c in now.items():require(c['token_position_L']==old[tokens]['token_position_L'],'候选位置漂移')
    oldp={p['continuation_id']:p for p in frozen['pendings']};newp={p['continuation_id']:p for p in raw['pendings']}
    require(oldp.keys()==newp.keys(),'Pending域不等价')
    for pid,p in newp.items():
        require(all(p[k]==oldp[pid][k] for k in ('token_ids','token_prefix_sha256','planning_target_T','attention_frontier','workflow_id')),'Pending输入或前沿偏离冻结合同')
    olde={(e['checkpoint_id'],e['continuation_id']):(e['status'],e['conditions']) for e in frozen['edges']}
    newe={(mapping[e['checkpoint_id']],e['continuation_id']):(e['status'],e['conditions']) for e in raw['edges']}
    require(len(newe)==len(raw['edges']) and newe==olde,'严格关系偏离冻结矩阵')
    expected=plans[solver]
    require(set(mapping[c] for c in plan.selected_ids)==set(expected['selected_ids']),'当前计划偏离冻结逻辑选集')
    require(abs(plan.recovery_cost_ms-expected['J_ms'])<=1e-9 and plan.K==3,'计划预算或模型目标偏离')
    return mapping
