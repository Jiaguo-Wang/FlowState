"""真实冻结输入与独立模拟物理身份检查；不创建Engine或使用GPU。"""
import json,hashlib,os,copy,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from . import planner,source_contract
from tests.test_strict_control import FakeRuntime
from flowstate.controller import StateController
from flowstate.strict_control import bind,preflight,StrictReconciliationError
P=Path(__file__).resolve().parent
OLD=P.parent/'native_resume_performance_e4_phase1b/smoke_A_Exact'

class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  r=json.loads((OLD/'reconciliation.json').read_text());ref=json.loads((OLD/'reference_audit.json').read_text())
  cls.raw=ref['initial']['strict_epoch'];cls.initial=ref['initial'];cls.inputs=json.loads((P/'prepared_inputs.json').read_text());cls.origins=r['origins']
  data=planner.original.json.dumps(cls.raw).encode();phy=json.dumps(cls.initial).encode()
  cls.evidence=planner.FrozenEvidence(data,hashlib.sha256(data).hexdigest(),physical_snapshot=phy,physical_sha256=hashlib.sha256(phy).hexdigest())
  cls.epoch=planner.from_frozen(cls.evidence,3,(cls.initial,cls.origins,cls.inputs));cls.prepared=planner.prepare(cls.epoch,cls.evidence)
 def make(self,solver):
  matrix=planner.benefits(self.prepared)
  with patch.object(planner,'frozen_selected',side_effect=lambda pr,s:planner_frozen(pr,s,P)):
   plan=planner.finish(self.prepared,planner.solve(self.prepared,matrix,solver))
  source_contract.validate_frozen(self.raw,plan,solver,P)
  return plan
 def test_both_sets_and_fresh_binding(self):
  for solver in ('Exact','Swap'):
   plan=self.make(solver);rt=FakeRuntime(self.epoch)
   with rt.control_epoch() as lease:
    b=bind(self.epoch,plan,rt.view,lease);preflight(self.epoch,plan,b,rt.view,lease)
   rt=FakeRuntime(self.epoch);result=StateController.reconcile_strict(self.epoch,plan,rt.bridge())
   self.assertEqual(result['status'],'PASS');self.assertEqual(len(rt.operations),4)
   self.assertEqual(sum(bool(n.slots) for n in rt.view.nodes),3);self.assertEqual(len(rt.view.free_slots),25)
 def test_all_eight_sources(self):
  for solver in ('Exact','Swap'):
   plan=self.make(solver)
   for label in 'ABCD':
    with patch.dict(os.environ,{'E4_PENDING':f'IMW1_{label}_TURN_003'}),patch.object(source_contract,'validate_frozen',side_effect=lambda raw,pl,s:validate_frozen(raw,pl,s,P)):
     target=source_contract.choose_source(self.raw,self.epoch,plan,self.origins,self.inputs,solver)
    self.assertEqual(target['expected']['G'],dict(Exact=dict(A=256,B=320,C=0,D=0),Swap=dict(A=0,B=0,C=384,D=448))[solver][label])
 def test_fail_closed(self):
  plan=self.make('Exact')
  for kind in ('reference','alias','component','stale','admission','selected_missing','FA_change','partial_failure'):
   rt=FakeRuntime(self.epoch);nodes=list(rt.view.nodes)
   if kind=='reference':nodes[0]=replace(nodes[0],references=1)
   if kind=='alias':nodes[1]=replace(nodes[1],slots=nodes[0].slots)
   if kind=='component':nodes[0]=replace(nodes[0],temporal_complete=False)
   if kind=='selected_missing':nodes[0]=replace(nodes[0],token_ids=(999,))
   rt.view=replace(rt.view,nodes=tuple(nodes))
   if kind=='stale':rt.mutate_at=2
   if kind=='admission':rt.view=replace(rt.view,admission_closed=False)
   if kind=='FA_change':rt.break_fa=True
   if kind=='partial_failure':rt.fail_at=2
   with self.assertRaises(StrictReconciliationError):StateController.reconcile_strict(self.epoch,plan,rt.bridge())
   self.assertTrue(rt.invalid)
   if kind=='partial_failure':self.assertEqual(len(rt.operations),2)
 def test_unknown_and_budget(self):
  with self.assertRaises(ValueError):
   bad=replace(self.epoch,edges=(replace(self.epoch.edges[0],status='UNKNOWN'),)+self.epoch.edges[1:])
   planner.prepare(bad,self.evidence)
  with self.assertRaises(ValueError):source_contract.validate_frozen(self.raw,replace(self.make('Exact'),K=2),'Exact',P)
 def test_native_source_mismatch(self):
  target={'expected':{'E':100},'pending':{'attention_frontier':200,'token_prefix_sha256':'正确'},'provenance':{'status':'PROVENANCE_UNVERIFIED'}}
  bound={'node_id':10000,'slots':[1]};actual={'best_match_node':10001,'cow_source_slots':[1],'executable_prefix':100,'physical_fa_hit':200,'input_token_digest':'正确'}
  with self.assertRaises(ValueError):source_contract.validate_native_source(target,bound,actual)

planner_frozen=planner.frozen_selected
validate_frozen=source_contract.validate_frozen
if __name__=='__main__':unittest.main()
