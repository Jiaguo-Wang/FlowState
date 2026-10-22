"""新增入口的CPU接线与失败即停测试，不导入GPU执行模块。"""
import unittest
from types import SimpleNamespace
from dataclasses import asdict
from unittest.mock import patch
from evaluation.runtime_control_closure_r2c1a import bridge
from tests.test_strict_control import FakeRuntime

class BridgeTests(unittest.TestCase):
    def execute(self,failed=False,preflight=False):
        raw,epoch,plan=bridge.load();runtime=FakeRuntime(epoch)
        if failed:runtime.fail_at=2
        scheduler=SimpleNamespace(tree_cache=object())
        initial={'ok':not preflight,'continuous_window':runtime.view.lease_id,'plan':asdict(plan)}
        def observe(*args):return runtime.observe(),{'semantic':{},'state':{'tree':{'structure_sha256':'模拟固定树'}}}
        with runtime.control_epoch(),patch.object(bridge,'readonly_control',return_value=initial),patch.object(bridge.prior,'scope'),patch.object(bridge.prior,'observe',side_effect=observe),patch.object(bridge,'selected_payload',return_value={'模拟':'不变'}),patch.object(bridge.SGLangAdapter,'evict_mamba_only',side_effect=runtime.evict_mamba_only):
            result=bridge.control(scheduler,{})
        return result,runtime,scheduler
    def test_success(self):
        r,rt,s=self.execute();self.assertEqual(r['status'],'READY');self.assertEqual(len(r['operations']),10)
        self.assertEqual(r['reconciliation']['status'],'PASS');self.assertEqual(len(r['after']['free_slots']),26)
    def test_partial_failure_stops(self):
        r,rt,s=self.execute(failed=True);self.assertEqual(r['status'],'BLOCKED');self.assertEqual(len(r['operations']),2)
        self.assertEqual(len(r['reconciliation']['completed_evictions']),1);self.assertTrue(s._r2c0_failed)
        self.assertTrue(r['invalidated_while_in_scheduler_call']);self.assertFalse(r['operations'][1]['returned'])
    def test_preflight_failure_no_operation(self):
        r,rt,s=self.execute(preflight=True);self.assertEqual(r['status'],'BLOCKED');self.assertEqual(r['operations'],[])
        self.assertEqual(rt.operations,[]);self.assertTrue(s._r2c0_failed)
if __name__=='__main__':unittest.main()
