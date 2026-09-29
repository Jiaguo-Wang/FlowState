"""在同一epoch内复用控制连接与不可变句柄，保持独立的三段控制计时。"""
import json
import socket
from evaluation import rq6_batched_control as base
from evaluation.control_path_codec import pack

_state_from_view=base._state_from_view
compact_census=base.compact_census


class BatchedControlClient:
    """三种策略共用相同的观测与协调接口，只有FlowState调用frontier批量读取。"""
    def __init__(self,delegate):
        self.delegate=delegate
        self.connection=None
        self.stream=None
        self.epoch_token=None
        self.handle_identity=None
        self.rpc_count=0

    def close(self):
        """结束本轮控制连接，后续请求验证恢复原客户端。"""
        if self.stream is not None:self.stream.close()
        if self.connection is not None:self.connection.close()
        self.stream=None
        self.connection=None

    def call(self,request):
        """同一连接串行发送消息，异常立即关闭且不重试。"""
        try:
            if self.connection is None:
                self.connection=socket.create_connection(('127.0.0.1',self.delegate.port),timeout=self.delegate.timeout_s)
                self.connection.settimeout(self.delegate.timeout_s)
                self.stream=self.connection.makefile('rb')
            self.rpc_count+=1
            self.connection.sendall((json.dumps(request)+'\n').encode())
            line=self.stream.readline()
            if not line:raise RuntimeError('控制响应为空')
            result=json.loads(line.decode())
            if not result.get('ok'):raise RuntimeError('控制操作失败：'+str(result))
            return result
        except Exception:
            self.close()
            raise

    def introspect(self,*,nonce,candidates,handles):
        """完整传输并验证一次候选句柄，随后只保存本epoch引用。"""
        self.handle_identity=tuple(handles[c.checkpoint_id] for c in candidates)
        result=self.call({'op':'checkpoint_control','nonce':nonce,'action':'control_epoch_introspection',
             'candidate_ids':[c.checkpoint_id for c in candidates],
             'packed_handles':pack(base._handle_rows(candidates,handles))})
        self.epoch_token=result['epoch_token']
        return result

    def observe_frontiers(self,pending,*,nonce):
        """真实查询全部四个pending，再恢复原非干扰证据的完整逻辑字段。"""
        queries=[{'query_id':r['rid'],'token_ids':list(r['input_ids']),'extra_key':None,'limit':None} for r in pending]
        result=self.call({'op':'checkpoint_control','nonce':nonce,'action':'control_epoch_frontiers','packed_queries':pack(queries)})
        observations=result['observations']
        if [r['query_id'] for r in observations]!=[q['query_id'] for q in queries] or result['non_interference_checks']!=4:
            raise RuntimeError('frontier结果数量、顺序或验证次数不一致')
        return [{**r,'semantic_snapshot_before':result['semantic_state'],'semantic_snapshot_after':result['semantic_state']} for r in observations]

    def reconcile(self,*,nonce,candidates,handles,selected_ids,expected_view_digest):
        """验证客户端映射未变，发送epoch引用；服务端仍重新读取并验证动态状态。"""
        try:
            if tuple(handles[c.checkpoint_id] for c in candidates)!=self.handle_identity:
                raise RuntimeError('本epoch客户端句柄映射改变')
            return self.call({'op':'checkpoint_control','nonce':nonce,'action':'control_epoch_reconciliation',
                'candidate_ids':[c.checkpoint_id for c in candidates],'selected_ids':list(selected_ids),
                'epoch_token':self.epoch_token,'expected_view_digest':expected_view_digest})
        finally:
            self.close()
