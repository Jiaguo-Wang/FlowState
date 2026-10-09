"""仅用 CPU 核验现有延后证据能否保持请求间正确性门禁，不启动实验。"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from collections.abc import Mapping
from enum import IntFlag
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    """从已有源码提取原函数体执行，避免导入模型、设备或运行时初始化。"""
    tree = ast.parse(path.read_text())
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names):
        raise RuntimeError('函数覆盖不唯一，拒绝继续')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace


def census_fixture(slots):
    """构造明确标为合成反例的 CPU 状态，不代表任何正式请求或实验结果。"""
    return {'tree': {'structure_rows': [[0, None, [1], 0], [1, 0, [], 16]],
                     'full_rows': [[0, 0, '根'], [1, 16, '固定索引']],
                     'mamba_rows': [] if slots is None else [[1, slots]],
                     'mamba_node_count': int(slots is not None), 'node_count': 2},
            'accounting': {'mamba_available': 15 if slots is not None else 16,
                          'mamba_schedulable_available': 15, 'mamba_evictable': int(slots is not None),
                          'mamba_protected': 0, 'full_evictable': 16,
                          'full_protected': 0, 'full_allocator': {'固定': True}}}


def check_temporal_gap():
    """直接运行旧 census 判定，验证末态相同不能替代中间边界证据。"""
    path = ROOT / 'evaluation/openhands_4workflow_occupancy_calibration.py'
    namespace = load_functions(path, {'reconstruct_node_positions', 'compact_census'},
                               {'Mapping': Mapping, 'PHYSICAL_MAX_MAMBA_CACHE_SIZE': 16})
    census = namespace['compact_census']
    baseline = census(census_fixture([7]), ordinal=0, request=None, previous=None)
    missing = census(census_fixture(None), ordinal=1, request=None, previous=baseline)
    replaced = census(census_fixture([8]), ordinal=1, request=None, previous=baseline)
    final = census(census_fixture([7]), ordinal=4, request=None, previous=baseline)
    assert missing['native_mamba_capacity_eviction_inferred']
    assert replaced['native_mamba_capacity_eviction_inferred']
    assert not final['native_mamba_capacity_eviction_inferred']
    return {'中途丢失被旧门禁拒绝': True, '中途换槽被旧门禁拒绝': True,
            '仅比较相同末态无法识别两种反例': True,
            '性质': '合成 CPU 反例；不宣称正式运行曾出现这些异常'}


def check_native_eviction(runtime_source):
    """以 CPU 对象执行原 Mamba 驱逐函数体，确认现有事件队列不覆盖此写入。"""
    class Layer(IntFlag):
        """提供原函数要求的层枚举，不创建任何设备对象。"""
        DEVICE = 1
        HOST = 2
    path = runtime_source / 'components/mamba_component.py'
    namespace = load_functions(path, {'evict_component'}, {'EvictLayer': Layer})
    events = []
    core = SimpleNamespace(component_evictable_size_={'mamba': 1},
                           host_lru_lists={'mamba': SimpleNamespace()}, kv_event_queue=events,
                           _record_remove_event=lambda *args, **kwargs: events.append('删除'))
    component = SimpleNamespace(component_type='mamba', tree_core=core)
    node = SimpleNamespace(component_data={'mamba': SimpleNamespace(value=[7], host_value=None)})
    device_frees, host_frees = defaultdict(list), defaultdict(list)
    result = namespace['evict_component'](component, node, device_frees, host_frees)
    assert result == (1, 0) and node.component_data['mamba'].value is None
    assert events == []
    return {'原函数释放循环状态': True, '现有事件队列未产生对应记录': True,
            '源码': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            '性质': '仅执行提取的函数体；不初始化 SGLang，不运行 GPU'}


def qualify(runtime_source, output):
    """证据不足时封闭失败，不创建一个静默降低门禁的 serving runner。"""
    results = {'中间状态证据': check_temporal_gap(),
               '原生事件覆盖': check_native_eviction(runtime_source)}
    result = {'status': 'BLOCKED_BY_CORRECTNESS_GATE', 'checks': results,
              '正式生命周期': 0, '正式请求': 0,
              '阻塞门禁': ['逐请求前候选驻留与身份一致性', '逐请求后无循环状态丢失或换槽',
                           '无中途 native replacement contamination'],
              '说明': '现有 H/E/G 记录、末态 census 和原生事件不足以替代原中间态验证。尚未建立并验证全覆盖的被动变更日志；不宣称这种实现原则上不可能。',
              '禁止推论': 'CPU 反例不是正式 workload 的测量，不能用于性能统计或推断真实异常率。'}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='核验现有被动证据是否保持原正确性门禁')
    parser.add_argument('--runtime-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.runtime_source, args.output), ensure_ascii=False, indent=2))
