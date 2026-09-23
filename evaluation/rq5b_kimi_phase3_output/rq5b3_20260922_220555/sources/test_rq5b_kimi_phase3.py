"""核对恢复实验的真实前沿、公共准备和配对统计边界。"""
from __future__ import annotations

import pytest
from evaluation.rq5b_kimi_phase3 import H, FALLBACK, workload, state_heg, selected_for_common


def view(positions):
    """构造包含全部实际祖先状态的观测。"""
    return {'paths': {f'c{H}': {'path_full_all_present': True, 'prefix_tokens': H,
        'path_mamba_positions': [{'node_id': position, 'position': position} for position in positions]}}}


def test_raw_phase1_fallback_is_not_eight_k():
    """原始状态仅删除 16K 时不能误报 8K 恢复间隔。"""
    assert state_heg(view(range(2048, H, 2048))) == {'H': H, 'E': 14336, 'G': 2048}


def test_common_preparation_produces_exact_conditions():
    """公共准备后 A/B 只差 16K 状态且真实回退为 8K。"""
    selected = [int(key[1:]) for key in selected_for_common()]
    assert state_heg(view(selected)) == {'H': H, 'E': H, 'G': 0}
    assert state_heg(view([p for p in selected if p != H])) == {'H': H, 'E': FALLBACK, 'G': H - FALLBACK}
    assert set(range(2048, H + 1, 2048)) - set(selected) == {10240, 12288, 14336}


def test_workload_has_same_history_and_novel_tail():
    """测量请求只共享已物化的 16K 历史，不命中已有分叉后缀。"""
    prefixes, history, target = workload()
    assert target[:H] == prefixes[f'c{H}']
    assert len(target) == H + 256
    assert all(target[:min(len(values), H) + 1] != values[:min(len(values), H) + 1] for values in history)
    assert workload() == workload()


def test_missing_mla_rejected_before_measurement():
    """缺少 attention 历史时不得把计算开销当作纯 KDA 条件差异。"""
    state = view([FALLBACK, H])
    state['paths'][f'c{H}']['path_full_all_present'] = False
    with pytest.raises(RuntimeError, match='MLA'):
        state_heg(state)
