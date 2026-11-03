"""三策略的纯CPU接线；复用既有规则，不创建新策略或物理执行入口。"""
from copy import deepcopy
from math import isfinite
from time import perf_counter_ns
from evaluation.kvflow_multiworkflow_cpu_main import run_cpu as legacy
from evaluation.kvflow_inspired_retention_cpu import policy as kvflow
from evaluation.sota_policies import MarconiStylePolicy, _build_marconi_metrics
from flowstate.optimizer import GlobalOptimizer
from flowstate.state_catalog import CheckpointCandidate

POLICIES = ('FlowState-Greedy', 'Marconi Adaptation', 'KVFlow-inspired')


def validate_declared_case(case, k):
    """验证冻结声明输入，返回值不代表Runtime认证。"""
    cs = kvflow.index_unique(case['candidates'], 'checkpoint_id')
    calls = kvflow.index_unique(case['calls'], 'continuation_id')
    graph = kvflow.index_unique(case['graph'], 'invocation_id')
    declared = kvflow.index_unique(case['declarations'], 'invocation_id')
    if type(k) is not int or not 0 <= k <= len(cs):
        raise ValueError('非法逻辑预算')
    if set(calls) != set(graph) or set(calls) != set(declared):
        raise ValueError('调用、ASG与固定前缀声明不完整')
    if set(case['predeclared_evaluation_order']) != set(calls) or len(case['predeclared_evaluation_order']) != len(calls):
        raise ValueError('评价次序域不完整')
    if case['common_retained_state_ids']:
        raise ValueError('此冻结合同不允许免费公共状态')
    for pid, p in calls.items():
        if type(p['ready']) is not bool or graph[pid]['ready'] != p['ready']:
            raise ValueError('Ready身份冲突')
        if p['token_ids'] != declared[pid]['input_token_ids']:
            raise ValueError('调用Token与固定前缀声明输入不同')
        if type(p['planning_target_T']) is not int or not 0 <= p['planning_target_T'] <= len(p['token_ids']):
            raise ValueError('Planning Target非法')
    events = {}
    orders = []
    for e in case['synthetic_access_events']:
        if e['checkpoint_id'] not in cs or e['runtime_observed'] is not False or e['event_type'] != 'DECLARED_SYNTHETIC_ACCESS':
            raise ValueError('不得将声明访问升级为实际访问')
        if type(e['event_order']) is not int:
            raise ValueError('访问次序非法')
        orders.append(e['event_order'])
        events[e['checkpoint_id']] = max(events.get(e['checkpoint_id'], -1), e['event_order'])
    if len(set(orders)) != len(orders):
        raise ValueError('重复访问事件次序')
    for cid, c in cs.items():
        if c['logical_cost'] != 1 or c['expected_independent_slot_cost'] != 1:
            raise ValueError('等大小逻辑预算合同不成立')
        if c['runtime_materialized'] is not False or c['runtime_handle'] is not None:
            raise ValueError('声明输入不能复用历史物理绑定')
        if cid not in events or c['last_access_order'] != events[cid] or not isfinite(c['last_access_order']):
            raise ValueError('访问元数据与声明事件不一致')
        expected = {pid for pid, p in calls.items() if c['token_position_L'] <= p['planning_target_T'] and kvflow.prefix(c['token_ids'], p['token_ids'])}
        if expected != set(c['token_compatible_calls']):
            raise ValueError('Token关系与声明兼容域冲突')
        parent = c.get('parent_candidate')
        if parent is not None and (parent not in cs or not len(cs[parent]['token_ids']) < len(c['token_ids']) or not kvflow.prefix(cs[parent]['token_ids'], c['token_ids'])):
            raise ValueError('祖先关系缺少Token依据')
    # 直接复用原策略验证STE、前缀、动态后缀和UNKNOWN，不增加新的计算公式。
    priorities = kvflow.priorities(case['candidates'], case['graph'], case['declarations'], case['epoch_id'], case['configuration_digest'])
    if any(r['priority'] is None for r in priorities):
        raise kvflow.IncompleteEvidence('存在UNKNOWN优先级')
    return priorities


def plan_declared(case, k, policy_name):
    """共同输出选集、评分、E/G及J；仅在冻结声明输入上执行回归。"""
    if policy_name not in POLICIES:
        raise ValueError('未登记的策略')
    original = deepcopy(case)
    start = perf_counter_ns()
    priority_rows = validate_declared_case(case, k)
    ep = legacy.epoch(case, k)
    core = tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'], 1, True) for c in case['candidates'])
    # 此处驻留标志只供旧CPU选择内核使用，不能作为物理操作凭据。
    recency = {c['checkpoint_id']: c['last_access_order'] for c in case['candidates']}
    proxy = {c['checkpoint_id']: c['token_position_L'] for c in case['candidates']}
    ready = perf_counter_ns()
    if policy_name == 'FlowState-Greedy':
        result = GlobalOptimizer(legacy.common.MODEL)._select_candidates(core, k, lambda chosen: legacy.common.score(ep, tuple(c.checkpoint_id for c in chosen))['cost_ms'])
        selected = tuple(c.checkpoint_id for c in result.selected)
    elif policy_name == 'Marconi Adaptation':
        selected = MarconiStylePolicy().select(core, k, recency, proxy, 1.0).selected_checkpoint_ids
    else:
        selected, priority_rows = kvflow.select(case['candidates'], case['graph'], case['declarations'], case['epoch_id'], case['configuration_digest'], k)
    solved = perf_counter_ns()
    score = legacy.common.score(ep, selected)
    if len(selected) > k or len(set(selected)) != len(selected) or not set(selected) <= {c.checkpoint_id for c in core}:
        raise ValueError('策略输出违反候选/预算合同')
    if case != original:
        raise ValueError('纯规划修改了输入')
    diagnostics = {'KVFlow': priority_rows, 'Marconi': {'recency': recency, 'value_token_proxy': proxy, 'memory_unit': '等大小逻辑单位，不是实际字节', 'alpha': 1.0, 'utility': _build_marconi_metrics(core, recency, proxy, 1.0)}}
    return {'policy': policy_name, 'epoch_id': case['epoch_id'], 'K': k, 'selected': list(selected), 'per_pending': score['per_pending'], 'J_ms': score['cost_ms'], 'evidence_level': 'DECLARED_SYNTHETIC_CPU_ONLY', 'diagnostics': diagnostics, 'timing': {'input_validation_and_prepare_ms': (ready-start)/1e6, 'policy_call_ms': (solved-ready)/1e6, 'scope': '单次接口回归诊断；非正式求解微基准'}}


def require_runtime_execution(plan):
    """声明CPU计划永远不能直接作为物理协调授权。"""
    raise ValueError('必须新建当前Strict Epoch、重映射并验证原生访问/组件合同；本接口不执行GPU')
