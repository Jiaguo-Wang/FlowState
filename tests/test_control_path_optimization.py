"""验证控制优化不丢失信息、不绕过非干扰和动态协调检查。"""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from evaluation.control_path_codec import pack,unpack
from evaluation import control_path_optimized_client as client
from evaluation import resume_epoch_e2e as frozen
import control_path_optimized_transport as transport


def test_codec_preserves_complete_values():
    """嵌套JSON、长token序列及空值均必须逐值还原。"""
    value={'token_ids':list(range(30000)),'extra_key':None,'中文':'保持语义','values':[True,False,{'ids':[]} ]}
    assert unpack(pack(value))==value
    with pytest.raises(Exception):unpack({'codec':'错误','payload':'x'})
    with pytest.raises(Exception):unpack({'codec':'json-zlib-base64-v1','payload':'损坏'})


def test_epoch_binding_and_single_consumption():
    """禁止跨epoch、错集合、错view或重复使用句柄。"""
    context=transport.EpochHandles({'A':object(),'B':object()},'digest')
    for token,ids,digest in [('错误',['A','B'],'digest'),(context.token,['B','A'],'digest'),(context.token,['A','B'],'错误')]:
        with pytest.raises(RuntimeError):context.take(token,ids,digest)
    assert context.take(context.token,['A','B'],'digest')==context.handles
    with pytest.raises(RuntimeError):context.take(context.token,['A','B'],'digest')


def test_reconciliation_preserves_original_dynamic_validation():
    """即使句柄已缓存，也必须完整调用原协调函数；异常后清除上下文。"""
    handles={'A':object()}
    context=transport.EpochHandles(handles,'digest')
    scheduler=SimpleNamespace(_control_epoch_handles=context)
    request={'epoch_token':context.token,'candidate_ids':['A'],'expected_view_digest':'digest'}
    original=transport.base._parse_handles
    called=[]
    def reconcile(s,r):
        called.append(True)
        assert transport.base._parse_handles(r)==handles
        raise RuntimeError('当前view已变化，原动态验证拒绝')
    with patch.object(transport.base,'_batch_reconciliation',reconcile):
        with pytest.raises(RuntimeError,match='当前view'):transport.reconciliation(scheduler,request)
    assert called==[True] and transport.base._parse_handles is original
    assert scheduler._control_epoch_handles is None


def test_frontier_batch_checks_each_query_and_reuses_only_adjacent_boundary():
    """四次查询仍有四次比较；相邻边界复用只减少三次相同快照读取。"""
    request={'packed_queries':pack([{'query_id':str(i),'token_ids':[i],'extra_key':None,'limit':None} for i in range(4)])}
    scheduler=SimpleNamespace(tree_cache=object())
    checks=[]
    with patch.object(transport.probe,'_validate_runtime_scope',return_value={'idle':True}), \
         patch.object(transport,'semantic_state',return_value={'state':[1,2]}) as state, \
         patch.object(transport.frontier,'inspect_resident_fa_frontier',side_effect=lambda c,t,**kw:{'resident_fa_frontier':t[0]}) as inspect, \
         patch.object(transport.frontier,'semantic_snapshot_differences',side_effect=lambda a,b:checks.append((a,b)) or []):
        result=transport.frontier_batch(scheduler,request)
    assert inspect.call_count==4 and state.call_count==5 and len(checks)==4
    assert [r['resident_fa_frontier'] for r in result['observations']]==[0,1,2,3]


def test_frontier_batch_rejects_transient_mutation():
    """中间查询改变状态时立即失败，不能被后续恢复原值掩盖。"""
    request={'packed_queries':pack([{'query_id':str(i),'token_ids':[i]} for i in range(4)])}
    snapshots=[{'state':0},{'state':0},{'state':1},{'state':0},{'state':0}]
    with patch.object(transport.probe,'_validate_runtime_scope',return_value={'idle':True}), \
         patch.object(transport,'semantic_state',side_effect=snapshots), \
         patch.object(transport.frontier,'inspect_resident_fa_frontier',return_value={'resident_fa_frontier':1}):
        with pytest.raises(RuntimeError,match='改变语义状态'):transport.frontier_batch(SimpleNamespace(tree_cache=object()),request)


def test_client_rejects_mapping_change_before_rpc():
    """客户端句柄集合改变时不得向服务端提交驱逐。"""
    c=client.BatchedControlClient(SimpleNamespace(port=1,timeout_s=1))
    c.handle_identity=('原句柄',)
    with patch.object(c,'call') as call:
        with pytest.raises(RuntimeError,match='句柄映射改变'):
            c.reconcile(nonce='n',candidates=[SimpleNamespace(checkpoint_id='A')],handles={'A':'新句柄'},selected_ids=[],expected_view_digest='d')
        call.assert_not_called()


def test_client_rejects_frontier_order_change():
    """批量结果错序必须失败，不能把其他请求的frontier用于当前continuation。"""
    c=client.BatchedControlClient(SimpleNamespace(port=1,timeout_s=1))
    pending=[{'rid':str(i),'input_ids':[i]} for i in range(4)]
    result={'observations':[{'query_id':str(i)} for i in [1,0,2,3]],'non_interference_checks':4,'semantic_state':{}}
    with patch.object(c,'call',return_value=result):
        with pytest.raises(RuntimeError,match='顺序'):c.observe_frontiers(pending,nonce='n')


def test_frozen_runner_diff_is_bounded():
    """请求执行、时钟、选择器和正确性主体必须与冻结runner逐字相同。"""
    import inspect
    from evaluation import control_path_optimized_e2e as optimized
    for name in ['construction','selection','stream_request','epoch_metrics']:
        assert inspect.getsource(getattr(frozen,name))==inspect.getsource(getattr(optimized,name))
    original=inspect.getsource(frozen.worker)
    actual=inspect.getsource(optimized.worker)
    actual=actual.replace('from control_path_optimized_transport import OptimizedControlEngine as RQ6BatchedControlEngine, requested_control_port',
                          'from rq6_batched_control_transport import RQ6BatchedControlEngine, requested_control_port')
    actual=actual.replace("            batch_observations = client.observe_frontiers(pending, nonce=run_id+':frontiers')\n            for request, obs in zip(pending, batch_observations):",
                          "            for request in pending:\n                obs = runtime.inspect_fa_frontier(request['input_ids'], nonce=run_id+':frontier:'+request['workflow_label'])")
    assert actual==original
