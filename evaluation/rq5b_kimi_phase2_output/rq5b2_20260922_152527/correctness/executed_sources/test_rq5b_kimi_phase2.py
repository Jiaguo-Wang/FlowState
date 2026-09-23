"""验证双 rank 映射边界、隐藏祖先状态及只读批量消息约束。"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from evaluation.rq5b_kimi_phase2 import workload, logical_view, heg, TPBatchClient
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel


def view():
    """构造两个工作流的真实路径格式。"""
    frozen, prefixes = workload()
    paths = {}
    for index, row in enumerate(frozen['candidates'], 1):
        paths[row['checkpoint_id']] = {'node_id': index, 'prefix_tokens': row['token_pos'],
            'path_full_all_present': True, 'target_mamba_present': True,
            'path_mamba_positions': [{'node_id': index, 'position': row['token_pos']}]}
    return frozen, {'paths': paths}


def test_existing_allocator_and_frontier():
    """相同冻结逻辑在四候选与预算二下产生非平凡选择。"""
    frozen, observed = view()
    candidates, pending = logical_view(observed, frozen, 32)
    result = GlobalOptimizer(RecoveryCostModel()).select(pending, candidates, 64)
    assert [c.checkpoint_id for c in result.selected] == ['A2', 'B2']
    assert all(row['G'] == 0 for row in heg(candidates, pending))
    for key in ('A1', 'B1'):
        observed['paths'][key]['target_mamba_present'] = False
        observed['paths'][key]['path_mamba_positions'] = []
    candidates, pending = logical_view(observed, frozen, 32)
    assert [(row['E'], row['G']) for row in heg(candidates, pending)] == [(0, 2048), (4096, 0), (0, 2048), (4096, 0)]


def test_hidden_ancestor_must_fail():
    """不能忽略未编入候选的物理祖先检查点来制造前沿一致。"""
    frozen, observed = view()
    observed['paths']['A2']['path_mamba_positions'].append({'node_id': 99, 'position': 3072})
    with pytest.raises(RuntimeError, match='祖先'):
        logical_view(observed, frozen, 32)


def test_two_logical_batches_four_rank_messages():
    """每个 epoch 两次逻辑提交，各自覆盖两个 rank。"""
    calls = []
    class Client:
        def _call(self, request):
            calls.append(request)
            return {'ok': True}
    batch = TPBatchClient({0: Client(), 1: Client()})
    for action in ('flowstate_batch_introspection', 'flowstate_batch_reconciliation'):
        assert set(batch.submit(action, {0: {}, 1: {}})) == {0, 1}
    assert batch.counts == {'introspection': 1, 'reconciliation': 1, 'rank_rpc': 4}
    assert len(calls) == 4


def test_rank_failure_is_not_partial_success():
    """任一 rank 返回错误时不能汇报整批成功。"""
    batch = TPBatchClient({0: SimpleNamespace(_call=lambda request: {'ok': True}),
                           1: SimpleNamespace(_call=lambda request: {'ok': False})})
    with pytest.raises(RuntimeError):
        batch.submit('flowstate_batch_introspection', {0: {}, 1: {}})


def transport_control():
    """仅加载传输控制函数，注入纯内存运行时用于失败路径测试。"""
    tree = ast.parse(Path('tests/runtime/rq5b_kimi_phase2_transport.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'control')
    calls = []
    handles = {key: SimpleNamespace(expected_node_id=i) for i, key in enumerate(('a', 'b'))}
    state = {'paths': {'a': {'target_mamba_present': True}, 'b': {'target_mamba_present': False}}}
    scheduler = SimpleNamespace(tree_cache=SimpleNamespace(sanity_check=lambda: None), _rq5b2_events=[])
    namespace = {'phase1': SimpleNamespace(runtime_scope=lambda *a: None),
        'parse': lambda r: handles, 'validate': lambda *a: None,
        'replacement': lambda c: [],
        'batch': SimpleNamespace(_batch_view=lambda *a: {'view': state, 'view_digest': 'digest'},
            _proof=lambda *a: {'status': 'PASS'}),
        'SGLangAdapter': lambda cache: SimpleNamespace(evict_mamba_only=lambda h: calls.append(h))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<测试传输>', 'exec'), namespace)
    request = {'action': 'flowstate_batch_reconciliation', 'selected_ids': ['a', 'b'],
               'expected_view_digest': 'digest', 'expected_replacement': []}
    return namespace['control'], scheduler, request, calls


def test_already_absent_is_never_recreated():
    """所选但已缺失的状态保持缺失，协调包装不得重物化。"""
    control, scheduler, request, calls = transport_control()
    result = control(scheduler, request)
    assert result['already_absent_ids'] == ('b',)
    assert not result['after']['paths']['b']['target_mamba_present']
    assert calls == []


@pytest.mark.parametrize('field,value', [('expected_view_digest', 'stale'),
    ('expected_replacement', [1]), ('selected_ids', ['a', 'a']), ('selected_ids', ['unknown'])])
def test_reject_invalid_batch_before_mutation(field, value):
    """过期视图、替换元数据漂移及非法选择必须在驱逐前失败。"""
    control, scheduler, request, calls = transport_control()
    request[field] = value
    with pytest.raises(RuntimeError):
        control(scheduler, request)
    assert calls == []
