"""同一次只读观察内批量复制内容，保留逐组件字节摘要与原有时刻边界。"""
import hashlib
import time
from types import SimpleNamespace

METRICS = []


def _digest(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _recurrent(cache, nodes):
    """每个组件一次复制；不复用任何上一次调用的内容或身份。"""
    import torch
    start = time.monotonic_ns()
    tensors = cache.req_to_token_pool.mamba_pool.mamba_cache
    components = list(tensors.conv) + [tensors.temporal]
    slots = sorted({s for n in nodes for s in n.slots})
    if not slots:
        return {}
    assert all(isinstance(s, int) and s >= 0 for s in slots), '槽位编号非法'
    hashes = {}
    total = 0
    for i, tensor in enumerate(components):
        assert tensor.ndim >= 2 and max(slots) < tensor.shape[1], '组件槽位越界'
        indices = torch.tensor(slots, device=tensor.device, dtype=torch.long)
        host = tensor.detach().index_select(1, indices).contiguous().cpu()
        total += host.numel() * host.element_size()
        for j, slot in enumerate(slots):
            hashes[slot, i] = _digest(host[:, j])
    METRICS.append({'kind': 'recurrent', 'elapsed_ns': time.monotonic_ns()-start,
                    'device_to_host_calls': len(components), 'bytes': total,
                    'slots': len(slots), 'cross_call_cache': False})
    return hashes


def payload(cache, view):
    hashes = _recurrent(cache, view.nodes)
    count = len(cache.req_to_token_pool.mamba_pool.mamba_cache.conv) + 1
    return {str(n.node_id): [{'slot': s, 'component': i, 'sha256': hashes[s, i]}
                            for s in n.slots for i in range(count)]
            for n in view.nodes if n.slots}


def selected_payload(cache, view, plan, epoch):
    nodes = [(c.candidate.checkpoint_id,
              next(n for n in view.nodes if n.token_ids == c.token_ids))
             for c in epoch.candidates if c.candidate.checkpoint_id in plan.selected_ids]
    hashes = _recurrent(cache, [n for _, n in nodes])
    count = len(cache.req_to_token_pool.mamba_pool.mamba_cache.conv) + 1
    return {cid: [hashes[s, i] for s in n.slots for i in range(count)] for cid, n in nodes}


def fa_payload(cache, indices):
    """按原层序计算摘要，最多聚集六十四 MiB 后统一复制到主机。"""
    import torch
    start = time.monotonic_ns()
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    assert type(pool).__name__ == 'HybridLinearKVPool', '不支持该FA池类型'
    assert not pool.full_kv_pool.is_quantized_kv_cache, '不支持量化FA池'
    assert pool.layer_transfer_counter is None and pool.full_kv_pool.layer_transfer_counter is None, '传输未完成'
    rows, group = [], []
    size = 0
    copies = 0
    total = 0

    def flush():
        nonlocal group, size, copies, total
        if not group:
            return
        host = torch.cat([data for _, _, data in group]).cpu()
        offset = 0
        for layer, kind, data in group:
            length = data.numel()
            blob = host[offset:offset+length].numpy().tobytes()
            rows.append({'layer': layer, 'component': kind,
                         'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': length})
            offset += length
        copies += 1
        total += size
        group, size = [], 0

    for layer in sorted(pool.full_attention_layer_id_mapping):
        for kind, getter in [('K', pool.get_key_buffer), ('V', pool.get_value_buffer)]:
            tensor = getter(layer)
            ix = torch.tensor(sorted(indices), device=tensor.device, dtype=torch.long)
            data = tensor.detach().index_select(0, ix).contiguous().view(torch.uint8).reshape(-1)
            if group and size + data.numel() > 64*1024*1024:
                flush()
            group.append((layer, kind, data))
            size += data.numel()
    flush()
    METRICS.append({'kind': 'fa', 'elapsed_ns': time.monotonic_ns()-start,
                    'device_to_host_calls': copies, 'bytes': total, 'cross_call_cache': False})
    return rows


def install(control, verify=False):
    """只在隔离 Scheduler 子进程替换读取函数；验证模式额外逐调用对照原实现。"""
    from evaluation.multishare_runtime_gate_b5c3 import bridge as capture
    from evaluation.runtime_control_closure_r2c1a import bridge as recurrent
    from evaluation.lightweight_runtime_e1 import control as light

    def wrap(old, new):
        def read(*args, **kwargs):
            value = new(*args, **kwargs)
            if verify:
                assert value == old(*args, **kwargs), '批量与原始摘要不一致'
                METRICS.append({'kind': 'equivalence', 'verified': True})
            return value
        return read

    capture.payload = wrap(capture.payload, payload)
    control.fa_payload = wrap(control.fa_payload, fa_payload)
    control.selected_payload = wrap(control.selected_payload, selected_payload)
    recurrent.selected_payload = wrap(recurrent.selected_payload, selected_payload)
    light.fa_payload = wrap(light.fa_payload, fa_payload)
