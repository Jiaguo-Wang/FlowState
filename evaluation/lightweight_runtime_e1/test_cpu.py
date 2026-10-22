"""CPU门禁使用冻结真实Census及模拟故障，不启动Engine。"""
import copy,json,io,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import asdict,replace
from . import light_capture,control
from evaluation.exact_greedy_paired_gpu_fix_b5d4a.test_regression import epoch_from,view_from
from evaluation.exact_greedy_paired_gpu_fix_b5d4a import test_regression
from evaluation.exact_planner_contract_gate_b5d1 import planner
from tests.test_strict_control import StrictControlTests
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
class LightTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.r=json.loads((ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4b/p01_Exact/reconciliation.json').read_text())
  cls.initial=cls.r['physical_observation'];cls.inputs=json.loads((P/'prepared_inputs.json').read_text())
 def capture(self,modify=None):
  initial=copy.deepcopy(self.initial);view=view_from(initial['before']);audit=initial['audit_before']
  scheduler=SimpleNamespace(server_args=SimpleNamespace(**self.inputs['configuration']['engine_configuration']),tree_cache=object())
  def frontier(cache,tokens):return {'resident_fa_frontier':next(p['attention_frontier'] for p in initial['strict_epoch']['pendings'] if p['token_ids']==tokens)}
  if modify:modify(audit)
  with patch.object(light_capture,'OUT',P),patch.object(light_capture.prior,'scope',return_value={}),patch.object(light_capture.prior,'observe',return_value=(view,audit)),patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))}),patch('evaluation.barrier_fa_frontier_control.inspect_resident_fa_frontier',side_effect=frontier),patch.object(light_capture,'payload',side_effect=AssertionError('轻量路径禁止内容哈希')):
   return light_capture.observe_control(scheduler,{'final':True})
 def test_01_real_fixture_contract_equal(self):
  current=self.capture();self.assertTrue(current['ok']);control.equivalent(current['strict_epoch'],self.initial['strict_epoch'])
  self.assertIsNone(current['payload_before'])
 def test_02_both_plans_equal(self):
  current=self.capture();raw=current['strict_epoch'];data=control.heavy.encode(raw);physical=control.heavy.encode(current)
  import hashlib
  evidence=planner.FrozenEvidence(data,hashlib.sha256(data).hexdigest(),physical_snapshot=physical,physical_sha256=hashlib.sha256(physical).hexdigest())
  ep=planner.from_frozen(evidence,2,(current,self.r['origins'],self.inputs));prep=planner.prepare(ep,evidence)
  old=epoch_from(self.r['epoch']);od=control.heavy.encode(self.r['raw_epoch']);op=control.heavy.encode(self.initial)
  prior=planner.prepare(old,planner.FrozenEvidence(od,hashlib.sha256(od).hexdigest(),physical_snapshot=op,physical_sha256=hashlib.sha256(op).hexdigest()))
  for solver in ('Greedy','Exact'):
   a=planner.finish(prep,planner.solve(prep,planner.benefits(prep),solver));b=planner.finish(prior,planner.solve(prior,planner.benefits(prior),solver))
   self.assertEqual((a.selected_ids,a.pending,a.recovery_cost_ms),(b.selected_ids,b.pending,b.recovery_cost_ms))
 def test_03_unknown_component(self):
  current=self.capture(lambda a:a['state'].pop('components'))
  self.assertTrue(any(e['status']=='UNKNOWN' for e in current['edges']))
 def test_04_frontier_change(self):
  r=copy.deepcopy(self.initial['strict_epoch']);r['pendings'][0]['attention_frontier']-=1
  with self.assertRaises(RuntimeError):control.equivalent(r,self.initial['strict_epoch'])
 def test_05_slot_change(self):
  r=copy.deepcopy(self.initial['strict_epoch']);r['candidates'][0]['state_identity']['target_mamba_slots']=[28]
  with self.assertRaises(RuntimeError):control.equivalent(r,self.initial['strict_epoch'])
 def test_06_reference_change(self):
  r=copy.deepcopy(self.initial['strict_epoch']);r['candidates'][0]['state_identity']['references']['target_mamba_lock_ref']=1
  with self.assertRaises(RuntimeError):control.equivalent(r,self.initial['strict_epoch'])
 def test_07_no_mutation_generation_shortcut(self):
  self.assertNotIn('mutation_generation',Path(control.__file__).read_text())
 def test_08_production_reconciler_unchanged(self):
  self.assertIs(control.StateController.reconcile_strict, __import__('flowstate.controller',fromlist=['StateController']).StateController.reconcile_strict)

def main():
 test_regression.P=P
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (LightTests,StrictControlTests,test_regression.Regression)])
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 (P/'cpu_tests.log').write_text(stream.getvalue())
 report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'errors':[(str(t),e) for t,e in result.errors+result.failures],'gpu_started':False}
 (P/'cpu_regression_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(stream.getvalue());assert result.wasSuccessful()
if __name__=='__main__':main()
