"""以冻结真实w1状态核验目标泛化与语义漂移拒绝，不模拟GPU成功。"""
import copy,json,os,unittest
from unittest.mock import patch
from dataclasses import asdict
from .prepare import ROOT,OUT,read,sha,write,E2
from . import source_contract as sc
from evaluation.exact_planner_contract_gate_b5d1 import planner as d
class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=read(OUT/'frozen_snapshot.json');cls.inputs=read(OUT/'prepared_inputs.json');cls.physical=read(E2/'w1/physical_state_census.json');cls.origins=read(E2/'w1/allocation_results.json')['origins']
  f=E2/'w1/strict_epoch_snapshot.json';p=E2/'w1/physical_state_census.json';ev=d.FrozenEvidence(f.read_bytes(),sha(f),physical_snapshot=p.read_bytes(),physical_sha256=sha(p));cls.ep=d.from_frozen(ev,2,(cls.physical,cls.origins,cls.inputs));pr=d.prepare(cls.ep,ev);bm=d.benefits(pr);cls.plans={s:d.finish(pr,d.solve(pr,bm,s)) for s in ('Greedy','Exact')}
 def choose(self,solver,pid):
  original=sc.validate_frozen
  with patch.dict(os.environ,{'E4_PENDING':pid}),patch.object(sc,'validate_frozen',lambda raw,plan,s:original(raw,plan,s,OUT)):
   return sc.choose_source(self.raw,self.ep,self.plans[solver],self.origins,self.inputs,solver)
 def test_all_eight_targets(self):
  for solver in self.plans:
   for p in self.inputs['pendings']:
    result=self.choose(solver,p['continuation_id']);self.assertEqual(result['provenance']['status'],'PROVENANCE_UNVERIFIED');self.assertIn(result['source_checkpoint'],self.plans[solver].selected_ids)
 def test_changed_candidate_tokens(self):
  raw=copy.deepcopy(self.raw);raw['candidates'][0]['token_ids'][0]+=1
  with self.assertRaises(ValueError):sc.validate_frozen(raw,self.plans['Exact'],'Exact',OUT)
 def test_changed_target(self):
  raw=copy.deepcopy(self.raw);raw['pendings'][0]['planning_target_T']+=1
  with self.assertRaises(ValueError):sc.validate_frozen(raw,self.plans['Exact'],'Exact',OUT)
 def test_unknown(self):
  raw=copy.deepcopy(self.raw);raw['edges'][0]['status']='UNKNOWN'
  with self.assertRaises(ValueError):sc.validate_frozen(raw,self.plans['Exact'],'Exact',OUT)
 def test_wrong_policy_plan(self):
  with self.assertRaises(ValueError):sc.validate_frozen(self.raw,self.plans['Greedy'],'Exact',OUT)
 def test_node_slot_renumbering_not_binding(self):
  raw=copy.deepcopy(self.raw)
  for c in raw['candidates']:c['state_identity']['node_id']+=100;c['state_identity']['target_mamba_slots']=[99]
  self.assertEqual(len(sc.validate_frozen(raw,self.plans['Exact'],'Exact',OUT)),7)
  # 此测试只说明逻辑映射不绑定历史Slot；物理冲突由原Bind/Preflight拒绝。
 def test_native_wrong_node(self):
  target=self.choose('Exact',self.inputs['pendings'][0]['continuation_id']);bound={'node_id':1,'slots':[2]};actual={'best_match_node':3,'cow_source_slots':[2]}
  with self.assertRaises(ValueError):sc.validate_native_source(target,bound,actual)
 def test_missing_edges(self):
  raw=copy.deepcopy(self.raw);raw['edges'].pop()
  with self.assertRaises(ValueError):sc.validate_frozen(raw,self.plans['Exact'],'Exact',OUT)
 def test_duplicate_candidates(self):
  raw=copy.deepcopy(self.raw);raw['candidates'].append(raw['candidates'][0])
  with self.assertRaises(ValueError):sc.validate_frozen(raw,self.plans['Exact'],'Exact',OUT)
 def test_configuration_mismatch(self):
  raw=copy.deepcopy(self.raw);raw['configuration_digest']='错误摘要'
  with self.assertRaises(ValueError):sc.validate_frozen(raw,self.plans['Exact'],'Exact',OUT)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
 write(OUT/'cpu_regression_results.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'eight_target_contracts':True,'gpu_started':False})
 raise SystemExit(0 if result.wasSuccessful() else 1)
