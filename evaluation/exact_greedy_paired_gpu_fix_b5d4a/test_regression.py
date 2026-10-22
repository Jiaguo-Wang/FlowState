"""仅CPU：失败Census来源回归、生产物理合同拒绝和失败落盘，不伪造GPU成功。"""
import copy,hashlib,io,json,os,tempfile,unittest
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from flowstate.strict_epoch import StrictAllocationEpoch,StrictCandidate,StrictPending,StrictEdge
from flowstate.state_catalog import CheckpointCandidate
from flowstate.strict_control import PhysicalView,PhysicalNode,bind,preflight
from flowstate.recovery_model import RecoveryCostModel
from evaluation.exact_planner_contract_gate_b5d1.planner import finish
from .source_contract import choose_source,validate_native_source
from .persistence import call_saved,Journal
P=Path(__file__).resolve().parent;ROOT=P.parents[1];OLD=ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4'
def read(p):return json.loads(Path(p).read_text())

def epoch_from(x):
 x=copy.deepcopy(x)
 for c in x['candidates']:
  c['candidate']['lineage_path']=tuple(c['candidate']['lineage_path']);c['candidate']=CheckpointCandidate(**c['candidate']);c['token_ids']=tuple(c['token_ids'])
 x['candidates']=tuple(StrictCandidate(**c) for c in x['candidates'])
 for p in x['pendings']:p['token_ids']=tuple(p['token_ids'])
 x['pendings']=tuple(StrictPending(**p) for p in x['pendings'])
 for e in x['edges']:e['conditions']=tuple(tuple(v) for v in e['conditions'])
 x['edges']=tuple(StrictEdge(**e) for e in x['edges']);return StrictAllocationEpoch(**x)

def view_from(x):
 x=copy.deepcopy(x)
 for n in x['nodes']:
  for k in ('token_ids','slots','component_ids'):n[k]=tuple(n[k])
 x['nodes']=tuple(PhysicalNode(**n) for n in x['nodes']);x['free_slots']=tuple(x['free_slots']);return PhysicalView(**x)

class Regression(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.r=read(ROOT/'evaluation/exact_reconciliation_correctness_gate_b5d3/strict/reconciliation.json')
  cls.ep=epoch_from(cls.r['epoch']);cls.ep.validate();cls.raw=cls.r['raw_epoch'];cls.inputs=read(OLD/'prepared_inputs.json')
  cls.histories=[read(OLD/'p01_Greedy'/f'history_{i:02d}.json') for i in range(1,9)]
  cls.view=view_from(cls.histories[-1]['state']['before']);cls.origins=[]
  for c in cls.raw['candidates']:
   seen=[h['request']['request_id'] for h in cls.histories if any(n['token_ids']==c['token_ids'] and n['slots'] for n in h['state']['before']['nodes'])]
   assert seen;cls.origins.append({'checkpoint_id':c['checkpoint_id'],'first_observed_after':seen[0]})
  prep=SimpleNamespace(epoch=cls.ep,ids=tuple(c.candidate.checkpoint_id for c in cls.ep.candidates),costs=tuple(RecoveryCostModel().estimate(p.target,p.target) for p in cls.ep.pendings))
  # 仅构造测试计划，不重跑策略；该计划不是失败D4内部未保存的计划证明。
  cls.plans={s:finish(prep,ids) for s,ids in [('Greedy',('B5C3_NODE_20','B5C3_NODE_18')),('Exact',('B5C3_NODE_18','B5C3_NODE_27'))]}
 def choose(self,solver='Greedy',raw=None,ep=None,plan=None):return choose_source(raw or self.raw,ep or self.ep,plan or self.plans[solver],self.origins,self.inputs,solver)
 def test_01_greedy_same_workflow_hint(self):
  t=self.choose();self.assertEqual(t['source_checkpoint'],'B5C3_NODE_20');self.assertFalse(t['provenance']['different_workflow_hint']);self.assertEqual(t['expected']['E'],2432)
 def test_02_exact_different_workflow_hint(self):
  t=self.choose('Exact');self.assertEqual(t['source_checkpoint'],'B5C3_NODE_27');self.assertTrue(t['provenance']['different_workflow_hint']);self.assertEqual(t['source_first_observed']['first_observed_after'],'B5C2_D_TURN_002')
 def test_03_no_provenance_promotion(self):
  for solver in self.plans:
   p=self.choose(solver)['provenance'];self.assertEqual(p['status'],'PROVENANCE_UNVERIFIED');self.assertFalse(p['cross_workflow_certified'])
 def test_04_incompatible_selected(self):
  prep=SimpleNamespace(epoch=self.ep,ids=tuple(c.candidate.checkpoint_id for c in self.ep.candidates),costs=tuple(RecoveryCostModel().estimate(p.target,p.target) for p in self.ep.pendings))
  with self.assertRaises(ValueError):self.choose(plan=finish(prep,('B5C3_NODE_18',)))
 def test_05_unknown(self):
  raw=copy.deepcopy(self.raw);raw['edges'][0]['status']='UNKNOWN'
  with self.assertRaises(ValueError):self.choose(raw=raw)
 def test_06_changed_strict_relation(self):
  raw=copy.deepcopy(self.raw);next(e for e in raw['edges'] if e['checkpoint_id']=='B5C3_NODE_20' and e['continuation_id']=='B5C2_C_TURN_003')['status']='REJECTED'
  with self.assertRaises(ValueError):self.choose(raw=raw)
 def test_07_missing_pair(self):
  raw=copy.deepcopy(self.raw);raw['edges'].pop()
  with self.assertRaises(ValueError):self.choose(raw=raw)
 def test_08_snapshot_stale(self):
  raw=copy.deepcopy(self.raw);raw['epoch_id']='错误边界'
  with self.assertRaises(ValueError):self.choose(raw=raw)
 def test_09_false_condition(self):
  raw=copy.deepcopy(self.raw);raw['edges'][0]['conditions']['complete_components']=False
  with self.assertRaises(ValueError):self.choose(raw=raw)
 def test_10_recorded_physical_fixture(self):
  b=bind(self.ep,self.plans['Greedy'],self.view,self.view.lease_id);preflight(self.ep,self.plans['Greedy'],b,self.view,self.view.lease_id)
 def altered(self,node):return replace(self.view,nodes=tuple(node if n.node_id==node.node_id else n for n in self.view.nodes))
 def source_node(self):return next(n for n in self.view.nodes if n.token_ids==tuple(self.raw['candidates'][1]['token_ids']))
 def reject_view(self,current):
  b=bind(self.ep,self.plans['Greedy'],self.view,self.view.lease_id)
  with self.assertRaises(ValueError):preflight(self.ep,self.plans['Greedy'],b,current,self.view.lease_id)
 def test_11_evicted(self):
  n=next(n for n in self.view.nodes if n.node_id==20);current=self.altered(replace(n,slots=(),component_ids=(),conv_complete=False,temporal_complete=False));current=replace(current,free_slots=tuple(sorted(current.free_slots+n.slots)));self.reject_view(current)
 def test_12_wrong_slot(self):
  n=next(n for n in self.view.nodes if n.node_id==20);self.reject_view(self.altered(replace(n,slots=(27,))))
 def test_13_stale_engine(self):self.reject_view(replace(self.view,engine_id='新生命周期'))
 def test_14_stale_window(self):self.reject_view(replace(self.view,lease_id='过期窗口'))
 def test_15_conv_missing(self):
  n=next(n for n in self.view.nodes if n.node_id==20);self.reject_view(self.altered(replace(n,conv_complete=False)))
 def test_16_temporal_missing(self):
  n=next(n for n in self.view.nodes if n.node_id==20);self.reject_view(self.altered(replace(n,temporal_complete=False)))
 def test_17_protected(self):
  n=next(n for n in self.view.nodes if n.node_id==20);self.reject_view(self.altered(replace(n,protected=True)))
 def test_18_reference(self):
  n=next(n for n in self.view.nodes if n.node_id==20);self.reject_view(self.altered(replace(n,references=1)))
 def test_19_native_wrong_node(self):
  t=self.choose();n=next(n for n in self.view.nodes if n.node_id==20)
  # 显式模拟拒绝输入，不冒充实际请求恢复记录。
  actual={'best_match_node':999,'cow_source_slots':list(n.slots)}
  with self.assertRaises(ValueError):validate_native_source(t,asdict(n),actual)
 def test_20_native_wrong_slot(self):
  t=self.choose();n=next(n for n in self.view.nodes if n.node_id==20)
  with self.assertRaises(ValueError):validate_native_source(t,{'node_id':20,'slots':list(n.slots)},{'best_match_node':20,'cow_source_slots':[999]})
 def test_21_native_wrong_frontier(self):
  t=self.choose();n=next(n for n in self.view.nodes if n.node_id==20)
  with self.assertRaises(ValueError):validate_native_source(t,{'node_id':20,'slots':list(n.slots)},{'best_match_node':20,'cow_source_slots':list(n.slots),'executable_prefix':0,'physical_fa_hit':3904})
 def test_22_failure_journal(self):
  with tempfile.TemporaryDirectory(dir=P) as d:
   j=Journal(Path(d)/'journal');r={'raw_epoch':self.raw,'epoch':asdict(self.ep),'plan':asdict(self.plans['Greedy']),'phase':'source_selection'}
   j.record('plan',r)
   try:raise ValueError('模拟来源条件失败')
   except ValueError as exc:j.fail(r,exc)
   x=read(Path(d)/'journal/001_failure.json');self.assertEqual(x['plan'],asdict_json(self.plans['Greedy']));self.assertEqual(x['failed_stage'],'source_selection');self.assertEqual(x['raw_epoch'],self.raw)
 def socket_fixture(self,line):
  class Socket:
   def __enter__(s):return s
   def __exit__(s,*a):pass
   def settimeout(s,t):pass
   def sendall(s,data):pass
   def makefile(s,mode):return io.BytesIO(line)
  return lambda *a,**k:Socket()
 def test_23_raw_response_before_error(self):
  response=read(OLD/'failed_control_response.json');line=json.dumps(response).encode()+b'\n'
  with tempfile.TemporaryDirectory(dir=P) as d:
   q=Path(d)/'rpc'
   with self.assertRaises(RuntimeError):call_saved(0,1,{'op':'CPU测试'},q,self.socket_fixture(line))
   self.assertEqual((q/'response.raw').read_bytes(),line);self.assertEqual(read(q/'response.json'),response);self.assertTrue((q/'exception.json').exists())
 def test_24_malformed_response_saved(self):
  with tempfile.TemporaryDirectory(dir=P) as d:
   q=Path(d)/'rpc'
   with self.assertRaises(ValueError):call_saved(0,1,{'op':'CPU测试'},q,self.socket_fixture(b'not-json'))
   self.assertEqual((q/'response.raw').read_bytes(),b'not-json');self.assertTrue((q/'exception.json').exists())
 def test_25_no_overwrite(self):
  with tempfile.TemporaryDirectory(dir=P) as d:
   q=Path(d)/'rpc';q.mkdir()
   with self.assertRaises(FileExistsError):call_saved(0,1,{},q,self.socket_fixture(b'{}'))
 def test_26_control_failure_integration(self):
  from . import control
  # 回放静态观察夹具至失败分支；全部真实Runtime读写接口被拒绝，不启动Engine。
  initial={'ok':True,'strict_epoch':self.raw,'before':self.r['before'],'audit_before':self.r['audit_before'],'window':self.raw['epoch_id'],'unchanged':True}
  with tempfile.TemporaryDirectory(dir=P) as d:
   root=Path(d);(root/'fixture').mkdir();(root/'prepared_inputs.json').write_text(json.dumps(self.inputs))
   for i,h in enumerate(self.histories,1):(root/'fixture'/f'history_{i:02d}.json').write_text(json.dumps(h))
   scheduler=SimpleNamespace()
   with patch.object(control,'OUT',root),patch.dict(os.environ,{'B5D4_RUN':'fixture','B5D4_SOLVER':'Greedy'}),patch.object(control.capture,'observe_control',return_value=initial),patch.object(control.prior,'scope'),patch.object(control,'choose_source',side_effect=ValueError('CPU来源失败注入')),patch.object(control.prior,'observe',side_effect=AssertionError('不准观察GPU')) as observe,patch.object(control.StateController,'reconcile_strict',side_effect=AssertionError('不准驱逐')) as reconcile:
    response,ep,plan=control.apply(scheduler,{})
   self.assertFalse(response['ok']);self.assertEqual(response['phase'],'source_selection');self.assertIn('plan',response);self.assertIn('epoch',response);self.assertIn('raw_epoch',response);self.assertTrue(scheduler._r2c0_failed)
   self.assertEqual(observe.call_count,0);self.assertEqual(reconcile.call_count,0)
   failures=list((root/'fixture/control_evidence').glob('*failure.json'));self.assertEqual(len(failures),1);saved=read(failures[0]);self.assertEqual(saved['plan'],json.loads(json.dumps(response['plan'])));self.assertEqual(saved['raw_epoch'],self.raw)
   rpc=root/'saved_rpc'
   with self.assertRaises(RuntimeError):call_saved(0,1,{'op':'CPU失败传输'},rpc,self.socket_fixture(json.dumps(response).encode()+b'\n'))
   persisted=read(rpc/'response.json');self.assertEqual(persisted['epoch'],json.loads(json.dumps(response['epoch'])));self.assertEqual(persisted['plan'],saved['plan'])
 def test_27_no_claim_from_injected_hint(self):
  origins=copy.deepcopy(self.origins)
  for o in origins:o['cross_workflow_certified']=True
  t=choose_source(self.raw,self.ep,self.plans['Exact'],origins,self.inputs,'Exact')
  self.assertFalse(t['provenance']['cross_workflow_certified'])
 def test_28_empty_response_persisted(self):
  with tempfile.TemporaryDirectory(dir=P) as d:
   q=Path(d)/'rpc'
   with self.assertRaises(RuntimeError):call_saved(0,1,{},q,self.socket_fixture(b''))
   self.assertEqual((q/'response.raw').read_bytes(),b'');self.assertTrue((q/'exception.json').exists())
 def test_29_native_input_digest_wrong(self):
  t=self.choose();n=next(n for n in self.view.nodes if n.node_id==20)
  actual={'best_match_node':20,'cow_source_slots':list(n.slots),'executable_prefix':t['expected']['E'],'physical_fa_hit':t['pending']['attention_frontier'],'input_token_digest':'错误输入'}
  with self.assertRaises(ValueError):validate_native_source(t,{'node_id':20,'slots':list(n.slots)},actual)

 def test_30_archived_native_match_not_provenance(self):
  target=copy.deepcopy(self.r['target'])
  target['provenance']={'status':'PROVENANCE_UNVERIFIED','cross_workflow_certified':False}
  bound=next(x['node'] for x in self.r['reconciliation']['binding']['candidates'] if x['candidate_id']==target['source_checkpoint'])
  actual=read(ROOT/'evaluation/exact_reconciliation_correctness_gate_b5d3/strict/summary.json')['native_match']
  result=validate_native_source(target,bound,actual)
  self.assertEqual(result['native_source'],'MATCHED');self.assertFalse(result['cross_workflow_resume_certified'])

def asdict_json(x):return json.loads(json.dumps(asdict(x)))

class Recorder(unittest.TextTestResult):
 def startTest(self,test):super().startTest(test);self.current=test.id()
 def addSuccess(self,test):super().addSuccess(test);ROWS.append({'test':test.id(),'status':'PASS','scope':'CPU静态夹具/模拟故障，非Runtime恢复认证'})
 def addFailure(self,test,err):super().addFailure(test,err);ROWS.append({'test':test.id(),'status':'FAIL'})
 def addError(self,test,err):super().addError(test,err);ROWS.append({'test':test.id(),'status':'ERROR'})
ROWS=[]
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Regression)
 result=unittest.TextTestRunner(verbosity=2,resultclass=Recorder).run(suite)
 (P/'cpu_regression_results.json').write_text(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'gpu_started':False,'rows':ROWS,'fixture_scope':'D4真实Census和首次观察记录；D3既有严格Epoch仅用于CPU计划合同夹具，未构造新Runtime成功证据'},ensure_ascii=False,indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful() else 1)
