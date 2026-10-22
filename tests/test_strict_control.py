"""只使用全新模拟物理身份验证严格协调，不执行任何真实 Runtime 操作。"""
from contextlib import contextmanager
from dataclasses import replace
import unittest
from unittest.mock import patch

from flowstate.controller import StateController
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.strict_control import PhysicalNode, PhysicalView, StrictReconciliationError, StrictRuntimeBridge
from tests.test_strict_epoch import load


class FakeRuntime:
    """提供明确的模拟连续隔离、当前物理观测与可故障注入的单组件驱逐。"""
    def __init__(self,epoch):
        nodes=tuple(PhysicalNode(10000+i,c.token_ids,(i+1,),(f'模拟conv:{i+1}',f'模拟temporal:{i+1}'),True,True,True,0,False,False) for i,c in enumerate(epoch.candidates))
        self.view=PhysicalView('模拟新Engine','模拟新Barrier','模拟连续lease',epoch.configuration_digest,True,True,False,nodes,28,tuple(range(len(nodes)+1,29)),'模拟FA完整摘要','模拟其他状态摘要')
        self.operations=[];self.held=False;self.invalid=False;self.used=False;self.failure_ledger=None;self.fail_at=None;self.break_fa=False;self.mutate_at=None;self.observations=0;self.skip_evict=False;self.corrupt_selected=False;self.damage_node_id=None;self.fail_after_mutation=False

    @contextmanager
    def control_epoch(self):
        if self.invalid or self.used:raise RuntimeError('模拟控制区间失效或已消费')
        self.used=True;self.held=True
        try:yield self.view.lease_id
        finally:self.held=False

    def observe(self):
        self.observations+=1
        if self.mutate_at==self.observations:self.view=replace(self.view,attention_digest='模拟外部变化')
        return self.view

    def evict_mamba_only(self,handle):
        assert self.held and not self.invalid
        self.operations.append(handle.checkpoint_id)
        if self.fail_at==len(self.operations):raise RuntimeError('模拟驱逐失败')
        nodes=[];free=set(self.view.free_slots)
        for n in self.view.nodes:
            if n.node_id==handle.expected_node_id:
                assert n.token_ids==handle.token_ids
                if not self.skip_evict:
                    free.update(n.slots);n=replace(n,slots=(),component_ids=(),conv_complete=False,temporal_complete=False)
            nodes.append(n)
        if self.corrupt_selected:
            target=next(i for i,n in enumerate(nodes) if n.node_id==self.damage_node_id)
            n=nodes[target];free.update(n.slots);nodes[target]=replace(n,slots=(),component_ids=(),conv_complete=False,temporal_complete=False)
        self.view=replace(self.view,nodes=tuple(nodes),free_slots=tuple(sorted(free)),attention_digest='模拟FA意外改变' if self.break_fa else self.view.attention_digest)

        if self.fail_after_mutation:raise RuntimeError('模拟驱逐已经修改状态后异常')

    def invalidate(self,ledger):
        self.invalid=True;self.failure_ledger=ledger;ledger['mock_invalidated_while_guarded']=self.held

    def bridge(self):
        return StrictRuntimeBridge(self,self.control_epoch,self.observe,self.invalidate)


class StrictControlTests(unittest.TestCase):
    def setUp(self):
        self.epoch=load(1,2);self.plan=GlobalOptimizer(RecoveryCostModel()).plan_strict(self.epoch);self.runtime=FakeRuntime(self.epoch)
    def run_plan(self,plan=None):
        return StateController.reconcile_strict(self.epoch,plan or self.plan,self.runtime.bridge())
    def rejects_before(self,plan=None):
        with self.assertRaises(StrictReconciliationError) as ctx:self.run_plan(plan)
        self.assertEqual(self.runtime.operations,[]);self.assertTrue(self.runtime.invalid)
        self.assertTrue(ctx.exception.ledger['mock_invalidated_while_guarded'])
        return ctx.exception.ledger
    def change_node(self,index,**kw):
        nodes=list(self.runtime.view.nodes);nodes[index]=replace(nodes[index],**kw);self.runtime.view=replace(self.runtime.view,nodes=tuple(nodes))

    def test_success_shared_ancestor_and_branches(self):
        with patch.object(GlobalOptimizer,'select',side_effect=AssertionError('不得重新选择')),patch('flowstate.executable_state.is_compatible',side_effect=AssertionError('不得回退旧兼容')):
            result=self.run_plan()
        self.assertEqual(result['status'],'PASS');self.assertEqual(tuple(self.runtime.operations),self.plan.logical_eviction_ids)
        resident={c.candidate.checkpoint_id for c,n in zip(self.epoch.candidates,self.runtime.view.nodes) if n.slots}
        self.assertEqual(resident,set(self.plan.selected_ids));self.assertEqual(len(resident),2)
        self.assertTrue(all(b['node']['node_id']>=10000 for b in result['binding']['candidates']))
    def test_slot_alias(self):
        self.change_node(1,slots=self.runtime.view.nodes[0].slots);self.rejects_before()
    def test_node_alias(self):
        self.change_node(1,node_id=self.runtime.view.nodes[0].node_id);self.rejects_before()
    def test_component_alias(self):
        self.change_node(1,component_ids=self.runtime.view.nodes[0].component_ids);self.rejects_before()
    def test_selected_missing(self):
        cid=self.plan.selected_ids[0];i=next(i for i,c in enumerate(self.epoch.candidates) if c.candidate.checkpoint_id==cid)
        self.change_node(i,token_ids=(999999,));self.rejects_before()
    def test_evicted_protected(self):
        cid=self.plan.logical_eviction_ids[0];i=next(i for i,c in enumerate(self.epoch.candidates) if c.candidate.checkpoint_id==cid)
        self.change_node(i,protected=True);self.rejects_before()
    def test_reference(self):
        self.change_node(0,references=1);self.rejects_before()
    def test_host_copy(self):
        self.change_node(0,host_copy=True);self.rejects_before()
    def test_epoch_digest(self):self.rejects_before(replace(self.plan,epoch_digest='0'*64))
    def test_budget(self):self.rejects_before(replace(self.plan,K=1))
    def test_selected_outside(self):self.rejects_before(replace(self.plan,selected_ids=('域外',)))
    def test_wrong_complement(self):self.rejects_before(replace(self.plan,logical_eviction_ids=()))
    def test_incomplete_components(self):
        self.change_node(0,temporal_complete=False);self.rejects_before()
    def test_stale_view(self):
        self.runtime.mutate_at=2;self.rejects_before()
    def test_active(self):
        self.runtime.view=replace(self.runtime.view,idle=False);self.rejects_before()
    def test_admission(self):
        self.runtime.view=replace(self.runtime.view,admission_closed=False);self.rejects_before()
    def test_transfer(self):
        self.runtime.view=replace(self.runtime.view,transfers_pending=True);self.rejects_before()
    def test_unmanaged(self):
        node=PhysicalNode(99999,(123456,),(28,),('额外conv','额外temporal'),True,True,True,0,False,False)
        self.runtime.view=replace(self.runtime.view,nodes=self.runtime.view.nodes+(node,),free_slots=tuple(v for v in self.runtime.view.free_slots if v!=28));self.rejects_before()
    def test_fa_corruption(self):
        self.runtime.break_fa=True
        with self.assertRaises(StrictReconciliationError) as ctx:self.run_plan()
        self.assertEqual(len(self.runtime.operations),1);self.assertEqual(ctx.exception.ledger['phase'],'postcheck');self.assertTrue(self.runtime.invalid)
    def test_partial_failure(self):
        self.runtime.fail_at=2
        with self.assertRaises(StrictReconciliationError) as ctx:self.run_plan()
        self.assertEqual(ctx.exception.ledger['completed_evictions'],list(self.plan.logical_eviction_ids[:1]));self.assertEqual(ctx.exception.ledger['failed_checkpoint_id'],self.plan.logical_eviction_ids[1]);self.assertEqual(len(self.runtime.operations),2)
        self.assertTrue(ctx.exception.ledger['mock_invalidated_while_guarded'])
        with self.assertRaises(StrictReconciliationError):self.run_plan()
        self.assertEqual(len(self.runtime.operations),2)
    def test_midflight_change(self):
        self.runtime.mutate_at=5
        with self.assertRaises(StrictReconciliationError) as ctx:self.run_plan()
        self.assertEqual(len(self.runtime.operations),1);self.assertEqual(ctx.exception.ledger['phase'],'before_operation')
    def test_silent_noop(self):
        self.runtime.skip_evict=True
        with self.assertRaises(StrictReconciliationError):self.run_plan()
        self.assertEqual(len(self.runtime.operations),1)
    def test_selected_damaged_by_broken_primitive(self):
        self.runtime.corrupt_selected=True
        self.runtime.damage_node_id=10000+next(i for i,c in enumerate(self.epoch.candidates) if c.candidate.checkpoint_id==self.plan.selected_ids[0])
        with self.assertRaises(StrictReconciliationError):self.run_plan()
        self.assertEqual(len(self.runtime.operations),1)
    def test_failed_operation_may_have_mutated(self):
        self.runtime.fail_after_mutation=True
        with self.assertRaises(StrictReconciliationError) as ctx:self.run_plan()
        self.assertEqual(ctx.exception.ledger['completed_evictions'],[])
        self.assertEqual(len(self.runtime.operations),1)
        self.assertIn('failure_view',ctx.exception.ledger)
        self.assertTrue(self.runtime.invalid)
    def test_selected_exceeds_budget(self):
        selected=tuple(c.candidate.checkpoint_id for c in self.epoch.candidates[:3])
        self.rejects_before(replace(self.plan,selected_ids=selected,selected_count=3))
    def test_engine_identity_changed(self):
        original=self.runtime.observe
        def observe():
            result=original()
            return replace(result,engine_id='另一Engine') if self.runtime.observations==2 else result
        self.runtime.observe=observe;self.rejects_before()
    def test_empty_slot(self):
        self.change_node(0,slots=());self.rejects_before()
    def test_accounting(self):
        self.runtime.view=replace(self.runtime.view,free_slots=());self.rejects_before()

if __name__=='__main__':unittest.main()
