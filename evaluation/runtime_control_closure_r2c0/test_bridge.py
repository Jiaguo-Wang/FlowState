"""只读预检查与组件区间的CPU合同，不启动Engine。"""
import unittest
from dataclasses import replace
from evaluation.runtime_control_closure_r2c0.bridge import load,assert_nonoverlap
from flowstate.strict_control import bind,preflight
from tests.test_strict_control import FakeRuntime

class ReadonlyTests(unittest.TestCase):
    def setUp(self):
        self.raw,self.epoch,self.plan=load();self.runtime=FakeRuntime(self.epoch);self.view=self.runtime.view
    def test_readonly_binding(self):
        b=bind(self.epoch,self.plan,self.view,self.view.lease_id)
        preflight(self.epoch,self.plan,b,self.view,self.view.lease_id)
        self.assertEqual(self.runtime.operations,[]);self.assertEqual(len(b.evicted_ids),10)
    def test_fail_closed_views(self):
        variants=[replace(self.view,admission_closed=False),replace(self.view,lease_id='失效'),replace(self.view,nodes=self.view.nodes[1:]),replace(self.view,free_slots=()),replace(self.view,configuration_digest='不一致')]
        nodes=list(self.view.nodes);nodes[0]=replace(nodes[0],component_ids=());variants.append(replace(self.view,nodes=tuple(nodes)))
        nodes=list(self.view.nodes);nodes[0]=replace(nodes[0],slots=nodes[1].slots);variants.append(replace(self.view,nodes=tuple(nodes)))
        for v in variants:
            with self.subTest(v=v.lease_id),self.assertRaises(ValueError):bind(self.epoch,self.plan,v,self.view.lease_id)
        self.assertEqual(self.runtime.operations,[])
    def test_stale_view(self):
        b=bind(self.epoch,self.plan,self.view,self.view.lease_id)
        with self.assertRaises(ValueError):preflight(self.epoch,self.plan,b,replace(self.view,attention_digest='变化'),self.view.lease_id)
    def test_plan_digest(self):
        with self.assertRaises(ValueError):bind(self.epoch,replace(self.plan,epoch_digest='变化'),self.view,self.view.lease_id)
    def test_actual_byte_overlap(self):
        assert_nonoverlap({'a':{'byte_ranges':[(100,20),(200,20)]},'b':{'byte_ranges':[(120,20),(220,20)]}})
        with self.assertRaises(RuntimeError):assert_nonoverlap({'a':{'byte_ranges':[(100,20)]},'b':{'byte_ranges':[(110,20)]}})
if __name__=='__main__':unittest.main()
