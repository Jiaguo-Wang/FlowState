"""只精简已完成认证的基线响应，不跳过原始生成器或设备内容读取。"""
import copy
import hashlib
import json
from types import SimpleNamespace
from flowstate.strict_control import PhysicalNode, PhysicalView, validate_view

SCHEMA = 'admission-selected-components-v1'
MODES = ('validation', 'serving', 'shadow_validation')
FULL_FIELDS = {'nodes', 'semantic', 'ns', 'selected_payload', 'selected_components',
               'fa_map', 'fa_protected_payload'}
MINIMAL_FIELDS = {'schema', 'context_sha256', 'selected_components'}


def require(value, message):
    if not value:
        raise ValueError(message)


def mode(value):
    """模式必须显式合法，未知模式不能回退。"""
    require(value in MODES, '未知基线接口模式')
    return value



def validate_runtime_mode(selected_mode, separation_mode):
    """精简模式必须使用现有异常隔离包装，不能误走无包装实验分支。"""
    mode(selected_mode)
    require(selected_mode == 'validation' or separation_mode == 'serving',
            '精简接口必须启用现有Serving异常隔离包装')


def context(result):
    """绑定当前响应已有的生命周期、计划及物理视图；不认证跨窗口新鲜度。"""
    require(result.get('ok') is True and result['status'] == 'READY', '协调未成功')
    require(result['reconciliation']['status'] == 'PASS', '物理协调未通过')
    epoch, plan, view = result['epoch'], result['plan'], result['after']
    require((plan['epoch_id'], plan['epoch_digest'], plan['K']) ==
            (epoch['epoch_id'], epoch['content_digest'], epoch['K']), '计划身份或预算错误')
    require(plan['snapshot_sha256'] == epoch['snapshot_sha256'], '输入身份错误')
    require(view['configuration_digest'] == epoch['configuration_digest'], '物理配置错误')
    require(view['idle'] is True and view['admission_closed'] is True and
            view['transfers_pending'] is False, '静止控制条件不成立')
    for key in ('engine_id', 'barrier_id', 'lease_id'):
        require(type(view[key]) is str and bool(view[key]), '缺失生命周期身份')
    nodes = tuple(PhysicalNode(**dict(n, token_ids=tuple(n['token_ids']),
                                     slots=tuple(n['slots']), component_ids=tuple(n['component_ids'])))
                  for n in view['nodes'])
    physical = PhysicalView(**dict(view, nodes=nodes, free_slots=tuple(view['free_slots'])))
    validate_view(physical, SimpleNamespace(configuration_digest=epoch['configuration_digest']), view['lease_id'])
    selected = plan['selected_ids']
    require(type(selected) in (list, tuple) and len(selected) == len(set(selected)) and
            len(selected) == plan['selected_count'] and len(selected) <= plan['K'], '保留集非法')
    return {'epoch_digest': epoch['content_digest'], 'plan': plan, 'view': view}


def context_digest(result):
    """此摘要只绑定响应元数据，不等于GPU内容认证或租约。"""
    return hashlib.sha256(json.dumps(context(result), sort_keys=True,
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def selected_components(result):
    """复用当前后验视图中的真实组件身份，不使用历史Slot或Workflow推断。"""
    context(result)
    epoch, view = result['epoch'], result['after']
    selected = set(result['plan']['selected_ids'])
    output = {}
    for c in epoch['candidates']:
        cid = c['candidate']['checkpoint_id']
        if cid not in selected:
            continue
        found = [n for n in view['nodes'] if n['token_ids'] == c['token_ids']]
        require(len(found) == 1, '保留候选缺少唯一物理映射')
        n = found[0]
        require(n['slots'] and n['conv_complete'] is True and n['temporal_complete'] is True
                and n['attention_present'] is True, '保留组件不完整')
        require(n['references'] == 0 and n['protected'] is False and n['host_copy'] is False,
                '组件引用或所有权冲突')
        require(len(n['component_ids']) >= 2 and all(type(x) is str and x for x in n['component_ids']),
                '组件身份不完整')
        output[cid] = list(n['component_ids'])
    require(set(output) == selected, '选中候选身份缺失')
    return output


def project(full, result):
    """完整生成及隐式检查先执行，之后才舍弃无在线消费者的返回字段。"""
    require(type(full) is dict and set(full) == FULL_FIELDS, '完整基线字段合同变化')
    require(type(full['ns']) is int and full['ns'] > 0 and type(full['semantic']) is dict,
            '完整基线损坏')
    expected = selected_components(result)
    require(full['selected_components'] == expected, '当前基线组件与协调后映射不同')
    require(full['selected_payload'] == result['selected_payload_after'], '保留组件内容变化')
    require(full['fa_map'] == result['fa_map_before'], '基线FA映射变化')
    require(full['fa_protected_payload'] == result['fa_protected_payload'], '保护FA内容变化')
    by_id = {n['node_id']: n for n in full['nodes']}
    require(len(by_id) == len(full['nodes']), '基线Node别名')
    for n in result['after']['nodes']:
        current = by_id.get(n['node_id'])
        require(current is not None and current['token_ids'] == n['token_ids'] and
                current['slots'] == n['slots'] and current['full_present'] is True and
                current['references'] == [0, 0, 0], '基线物理身份、驻留或引用变化')
    return {'schema': SCHEMA, 'context_sha256': context_digest(result),
            'selected_components': copy.deepcopy(expected)}


def validate_response(result, selected_mode):
    """客户端在提交任何Pending之前核验响应；损坏时不得静默回退。"""
    mode(selected_mode)
    baseline = result['admission_baseline']
    if selected_mode == 'serving':
        require(type(baseline) is dict and set(baseline) == MINIMAL_FIELDS, '精简基线字段不完整')
        require(baseline['schema'] == SCHEMA, '基线接口版本错误')
        require(baseline['context_sha256'] == context_digest(result), '基线响应身份不一致')
        require(baseline['selected_components'] == selected_components(result), '必要组件字段损坏')
    else:
        require(type(baseline) is dict and set(baseline) == FULL_FIELDS, 'Validation基线不完整')
        if selected_mode == 'shadow_validation':
            require(result['admission_baseline_serving'] == project(baseline, result),
                    '双接口不一致')
    return result


def build(result, full_builder, selected_mode):
    """生产者调用恰一次；全量内容SHA、同步及异常均保留。"""
    mode(selected_mode)
    full = full_builder()
    if selected_mode == 'validation':
        result['admission_baseline'] = full
    elif selected_mode == 'serving':
        result['admission_baseline'] = project(full, result)
    else:
        result['admission_baseline_serving'] = project(full, result)
        result['admission_baseline'] = full
    return validate_response(result, selected_mode)
