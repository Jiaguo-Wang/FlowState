"""为 Kimi 双 rank 复用批量观测和循环组件协调原语。"""
from __future__ import annotations

import rq5b_kimi_phase1_transport as phase1
import rq6_batched_control_transport as batch
from flowstate.adapters.sglang import RuntimeCheckpointHandle, SGLangAdapter


def replacement(cache):
    """只读记录节点访问字段及全部替换链表指针。"""
    core = cache.tree_core
    nodes = {node.id: node for _, node in phase1.probe._tree_items(core)}
    for lru in core.lru_lists.values():
        for name in ('head', 'tail', 'mid', 'cursor'):
            node = getattr(lru, name, None)
            if node is not None:
                nodes[node.id] = node
    return [[key, node.last_access_time, node.creation_time, node.hit_count,
             node.priority, [getattr(n, 'id', None) for n in node.lru_prev],
             [getattr(n, 'id', None) for n in node.lru_next]]
            for key, node in sorted(nodes.items())]


def witness(cache):
    """记录观测前后的驻留、分配器和替换元数据。"""
    return {'tree': phase1.probe._global_maps(cache),
            'accounting': phase1.probe._accounting_snapshot(cache),
            'replacement': replacement(cache)}


def parse(request):
    """允许首次发现节点身份，协调时由完整句柄再次验证。"""
    handles = {}
    for row in request['handles']:
        handle = RuntimeCheckpointHandle(
            checkpoint_id=row['checkpoint_id'], token_ids=tuple(row['token_ids']),
            expected_node_id=row.get('expected_node_id'),
            expected_prefix_digest=row['expected_prefix_digest'])
        if handle.checkpoint_id in handles:
            raise RuntimeError('候选身份重复')
        handles[handle.checkpoint_id] = handle
    if not handles or list(handles) != request['candidate_ids']:
        raise RuntimeError('候选顺序或集合不一致')
    return handles


def validate(handles, view):
    """核对全部身份与 MLA 路径，并拒绝物理节点别名。"""
    nodes = []
    for key, handle in handles.items():
        path = view['paths'][key]
        if handle.expected_node_id is not None and path['node_id'] != handle.expected_node_id:
            raise RuntimeError('物理节点身份不匹配')
        if path['prefix_sha256'] != handle.expected_prefix_digest:
            raise RuntimeError('前缀身份不匹配')
        if not path['target_full_present'] or not path['path_full_all_present']:
            raise RuntimeError('MLA 路径不完整')
        nodes.append(path['node_id'])
    if len(nodes) != len(set(nodes)):
        raise RuntimeError('多个逻辑候选映射到同一物理节点')


def install_events(scheduler):
    """在实验包装层记录实际组件驱逐，不修改底层实现。"""
    scheduler._rq5b2_events = []
    scheduler._rq5b2_authorized = False
    core = scheduler.tree_cache.tree_core
    original = core._evict_component_and_detach_lru

    def observed(node, comp, *args, **kwargs):
        scheduler._rq5b2_events.append({'node_id': int(node.id),
            'component': type(comp).__name__,
            'authorized': scheduler._rq5b2_authorized})
        return original(node, comp, *args, **kwargs)

    core._evict_component_and_detach_lru = observed


def control(scheduler, request):
    """每个 rank 接收一次整批请求，并返回一次后验验证证据。"""
    cache = scheduler.tree_cache
    phase1.runtime_scope(scheduler, cache)
    action = request['action']
    if action == 'runtime_gate':
        install_events(scheduler)
        return {'ok': True, 'scope': phase1.runtime_scope(scheduler, cache),
                'initial': witness(cache)}
    handles = parse(request)
    if action == 'flowstate_batch_introspection':
        before = witness(cache)
        result = batch._batch_view(scheduler, handles)
        after = witness(cache)
        validate(handles, result['view'])
        if before != after:
            raise RuntimeError('只读观测改变了缓存或替换元数据')
        return {'ok': True, **result, 'read_only_before': before,
                'read_only_after': after, 'read_only': True,
                'eviction_events': scheduler._rq5b2_events,
                'checkpoint_bytes_per_rank': sum(cache.req_to_token_pool.mamba_pool.get_contiguous_buf_infos()[2])}
    if action != 'flowstate_batch_reconciliation':
        raise RuntimeError('不支持的阶段二控制动作')
    selected = tuple(request['selected_ids'])
    if len(selected) != len(set(selected)) or not set(selected).issubset(handles):
        raise RuntimeError('所选集合含重复或未知候选')
    before = batch._batch_view(scheduler, handles)
    if before['view_digest'] != request['expected_view_digest']:
        raise RuntimeError('观测后运行时状态发生漂移')
    if replacement(cache) != request['expected_replacement']:
        raise RuntimeError('观测后替换元数据发生漂移')
    if any(handle.expected_node_id is None for handle in handles.values()):
        raise RuntimeError('协调缺少精确物理句柄')
    validate(handles, before['view'])
    resident = {key for key, path in before['view']['paths'].items() if path['target_mamba_present']}
    evicted = tuple(sorted(resident - set(selected)))
    absent = tuple(sorted(set(handles) - resident))
    adapter = SGLangAdapter(cache)
    try:
        scheduler._rq5b2_authorized = True
        for key in evicted:
            adapter.evict_mamba_only(handles[key])
    finally:
        scheduler._rq5b2_authorized = False
    cache.sanity_check()
    after = batch._batch_view(scheduler, handles)
    proof = batch._proof(before['view'], after['view'], tuple(k for k in selected if k in resident), evicted)
    proof['already_absent_unchanged'] = all(not after['view']['paths'][key]['target_mamba_present'] for key in absent)
    proof['no_native_eviction'] = all(event['authorized'] for event in scheduler._rq5b2_events)
    if proof['status'] != 'PASS' or not proof['already_absent_unchanged'] or not proof['no_native_eviction']:
        raise RuntimeError(f'批量协调验证失败：{proof}')
    return {'ok': True, 'before': before['view'], 'after': after['view'],
            'proof': proof, 'evicted_ids': evicted, 'already_absent_ids': absent,
            'selected_ids': selected, 'post_validation_count': 1,
            'eviction_events': scheduler._rq5b2_events}


def run_scheduler(*args, **kwargs):
    """复用阶段一的双 rank 安全时点传输安装流程。"""
    phase1.control = control
    return phase1.wrapped_run_scheduler_process(*args, **kwargs)


class KimiPhase2Engine(phase1.SGLangEngine):
    """阶段二的独立引擎生命周期入口。"""
    run_scheduler_process_func = staticmethod(run_scheduler)
