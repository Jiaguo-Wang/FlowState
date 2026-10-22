"""CPU情景输入、预算、复用策略和评价口径的合同测试。"""
import copy,unittest,json
from . import run_cpu as r
from evaluation.kvflow_inspired_retention_cpu.policy import select,compute_ste,IncompleteEvidence
class ContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cases=r.read(r.OUT/'workloads.json')['cases']
 def test_lock(self):r.verify_lock()
 def test_sizes(self):self.assertEqual([(len(c['candidates']),sum(p['ready'] for p in c['calls'])) for c in self.cases],[(20,1),(12,3),(16,4)])
 def test_original_sequential_inputs(self):
  old=r.read(r.ROOT/'evaluation/kvflow_synthetic_stage1/workload.json')['cycles'][0]
  self.assertEqual([p['token_ids'] for p in self.cases[0]['calls']],[p['token_ids'] for p in old['pendings']])
 def test_proposed_only(self):
  for case in self.cases:
   self.assertEqual(case['runtime_actual_candidates'],[])
   for c in case['candidates']:self.assertFalse(c['runtime_materialized']);self.assertIsNone(c['runtime_handle']);self.assertEqual(c['logical_cost'],1)
 def test_prefixes_positions(self):
  for case in self.cases:
   for c in case['candidates']:
    self.assertEqual(len(c['token_ids']),c['token_position_L'])
    for p in case['calls']:
     valid=c['token_position_L']<=p['planning_target_T'] and c['token_ids']==p['token_ids'][:c['token_position_L']]
     self.assertEqual(valid,p['continuation_id'] in c['token_compatible_calls'])
 def test_graph_known(self):
  for case in self.cases:
   values=compute_ste(case['graph'],case['epoch_id']);self.assertTrue(all(v is not None for v in values.values()))
   for g in case['graph']:
    if g['ready']:self.assertEqual(values[g['invocation_id']],0)
  self.assertEqual(compute_ste(self.cases[2]['graph'],self.cases[2]['epoch_id'])['C_JOIN'],1)
 def test_current_scope(self):
  for case in self.cases:self.assertEqual({p.identifier for p in r.epoch(case,3).pending},{p['continuation_id'] for p in case['calls'] if p['ready']})
 def test_future_isolation(self):
  for case in self.cases:
   other=copy.deepcopy(case)
   for p in other['calls']:
    if not p['ready']:p['planning_target_T']+=64
   self.assertEqual(r.epoch(case,3),r.epoch(other,3))
 def test_ancestor_recovers(self):
  c=self.cases[0];ancestor=min(c['candidates'],key=lambda c:c['checkpoint_id']);score=r.common.score(r.epoch(c,3),[ancestor['checkpoint_id']]);self.assertEqual((score['per_pending'][0]['E'],score['per_pending'][0]['G']),(4032,64))
 def test_overlap_is_max(self):
  case=self.cases[0];cs=case['candidates'][:2];s=r.common.score(r.epoch(case,3),[c['checkpoint_id'] for c in cs]);self.assertEqual(s['per_pending'][0]['E'],4096);self.assertEqual(s['cost_ms'],0)
 def test_shared_support(self):
  c=self.cases[2];root=next(x for x in c['candidates'] if x['prefix_owner']=='ROOT' and x['token_position_L']==1024)
  self.assertEqual(len(root['token_compatible_calls']),5)
  ab=next(x for x in c['candidates'] if x['prefix_owner']=='AB' and x['token_position_L']==2048);self.assertEqual(set(ab['token_compatible_calls']),{'C_A','C_B'})
 def test_budget_and_repeatability(self):
  for case in self.cases:
   for k in [3,5]:
    one=r.solve(case,k)[0];self.assertEqual(one,r.solve(case,k)[0])
    for s in one.values():self.assertEqual(len(s),len(set(s)));self.assertLessEqual(len(s),k);r.common.score(r.epoch(case,k),s)
 def test_current_future_partition(self):
  for case in self.cases:
   for s in r.solve(case,3)[0].values():self.assertAlmostEqual(r.common.score(r.epoch(case,3,'all'),s)['cost_ms'],sum(r.common.score(r.epoch(case,3,v),s)['cost_ms'] for v in ['current','future']),places=8)
 def test_unknown_fail_closed(self):
  c=copy.deepcopy(self.cases[0]);c['graph'][0]['ready']=False
  with self.assertRaises(IncompleteEvidence):select(c['candidates'],c['graph'],c['declarations'],c['epoch_id'],c['configuration_digest'],3)
 def test_bad_token_fail_closed(self):
  c=copy.deepcopy(self.cases[0]);c['candidates'][0]['token_ids'][0]=-1
  with self.assertRaises(IncompleteEvidence):select(c['candidates'],c['graph'],c['declarations'],c['epoch_id'],c['configuration_digest'],3)
 def test_epoch_conflict(self):
  c=copy.deepcopy(self.cases[0]);c['candidates'][0]['barrier_id']='OTHER'
  with self.assertRaises(IncompleteEvidence):select(c['candidates'],c['graph'],c['declarations'],c['epoch_id'],c['configuration_digest'],3)
 def test_complete_access_events(self):
  for case in self.cases:
   for c in case['candidates']:
    events=[e for e in case['synthetic_access_events'] if e['checkpoint_id']==c['checkpoint_id']]
    self.assertEqual(len(events),2);self.assertEqual(c['last_access_order'],events[-1]['event_order']);self.assertTrue(all(not e['runtime_observed'] for e in events))
if __name__=='__main__':unittest.main()
