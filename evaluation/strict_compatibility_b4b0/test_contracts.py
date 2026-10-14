"""仅用静态证据和模拟对象验证采集合同，禁止启动Runtime。"""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from .contracts import *
from .launch import execute_plan

class Contracts(unittest.TestCase):
    def setUp(self):
        old=read(B1)['positive_arm'];self.state=read(ROOT/'evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2/retained/evidence.json')['barrier']
        self.c=copy.deepcopy(old['checkpoints'][0]);self.p=copy.deepcopy(old['pendings'][0]);self.c.update(configuration_digest='x',barrier_id='b');self.p.update(configuration_digest='x',barrier_id='b',controlled_input_ready=True)
        node=next(n for n in self.state['nodes'] if n['token_ids']==self.c['token_ids'])
        self.c['state_identity']['node_id']=node['node_id']
        self.c['state_identity']['target_mamba_slots']=node['mamba_slots']
    def result(self,c=None,p=None,state=None,alias=False):return edge(c or self.c,p or self.p,state or self.state,alias,'x')
    def test_valid_static_evidence(self):self.assertEqual(self.result()['status'],'EPOCH_ADMISSIBLE')
    def test_exact_scope_and_round(self):
        self.assertEqual(GROUPS,(1,2,3,4,5,6))
        for n in GROUPS:group_contract(n)
        for n in (0,7,-1):
            with self.assertRaises(RuntimeError):group_contract(n)
        m=read(MANIFEST);m['groups'][1]['allocation_round']=2
        with self.assertRaises(RuntimeError):group_contract(1,m)
    def test_hash_change_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.write_text('原内容');h=sha(p);verify_hashes({str(p):h});p.write_text('改变')
            with self.assertRaises(RuntimeError):verify_hashes({str(p):h})
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f';write_new(p,{'a':1})
            with self.assertRaises(FileExistsError):write_new(p,{'a':2})
    def test_token_identity_and_prefix(self):
        self.c['token_ids'][0]+=1
        self.assertEqual(self.result()['status'],'REJECTED')
        self.c['token_prefix_sha256']=token_digest(self.c['token_ids'])
        self.assertEqual(self.result()['status'],'REJECTED')
    def test_L_greater_T(self):
        self.p['planning_target_T']=1;self.assertEqual(self.result()['status'],'REJECTED')
    def test_components_refs_alias(self):
        self.state['components']['conv']=[];self.assertEqual(self.result()['status'],'REJECTED')
        self.setUp();self.c['state_identity']['references']['target_mamba_lock_ref']=1
        self.assertEqual(self.result()['status'],'REJECTED')
        self.setUp();self.assertEqual(self.result(alias=True)['status'],'REJECTED')
    def test_missing_is_unknown(self):
        del self.p['token_prefix_sha256'];self.assertEqual(self.result()['status'],'UNKNOWN')
        self.setUp();del self.state['components'];self.assertEqual(self.result()['status'],'UNKNOWN')
        self.setUp();del self.c['token_ids'];self.assertEqual(self.result()['status'],'UNKNOWN')
        self.setUp();del self.c['state_identity']['references']['target_mamba_lock_ref'];self.assertEqual(self.result()['status'],'UNKNOWN')
    def test_not_idle_native_eviction(self):
        safe_barrier(self.state)
        with self.assertRaises(RuntimeError):safe_barrier(self.state,True)
        self.state['scope']['running_requests']=1
        with self.assertRaises(RuntimeError):safe_barrier(self.state)
    def test_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            original={'edges':[self.result()]};h=freeze_snapshot(d,original)
            original['edges'][0]['status']='污染';unchanged_snapshot(d,h)
            Path(d,'snapshot.json').write_text('{}')
            with self.assertRaises(RuntimeError):unchanged_snapshot(d,h)
    def test_failure_ledger_and_no_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            seen=[]
            def fake(g):
                seen.append(g)
                if g==3:raise RuntimeError('模拟失败')
                return {'status':'CAPTURED','structure':{'primary':'A'}}
            with self.assertRaises(RuntimeError):execute_plan(d,fake)
            self.assertEqual(seen,[1,2,3])
            rows=[json.loads(x) for x in Path(d,'audit.jsonl').read_text().splitlines()]
            self.assertEqual([r['group'] for r in rows if r['event']=='未执行'],[4,5,6])
    def test_quota_does_not_reassign(self):
        with tempfile.TemporaryDirectory() as d:
            r=execute_plan(d,lambda g:{'status':'CAPTURED','structure':{'primary':'A','all_labels':['A','B']}})
            self.assertEqual(r['accepted_by_stratum'],{'A':4,'B':0,'C':0})
            self.assertEqual(len(r['groups']),6)
    def test_classification_no_policy_inputs(self):
        cs=[{'checkpoint_id':'a','token_position_L':1},{'checkpoint_id':'b','token_position_L':2},{'checkpoint_id':'c','token_position_L':3}]
        ps=[{'continuation_id':'p','planning_target_T':4},{'continuation_id':'q','planning_target_T':5}]
        edges=[{'checkpoint_id':c,'continuation_id':p,'status':'EPOCH_ADMISSIBLE'} for c,p in [('a','p'),('b','p'),('c','q')]]
        result=structure_labels(cs,ps,edges);self.assertEqual(result['primary'],'A')
        for c in cs:c['FlowState_wins']=False;c['benefit_ms']=999999
        self.assertEqual(result,structure_labels(cs,ps,edges))
        tree=ast.parse((HERE/'contracts.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='structure_labels')
        self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(node)))
    def test_actual_materializer_future_boundary(self):
        source=ROOT/'evaluation/rq3_openhands_neutral_collector.py';tree=ast.parse(source.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='materialize_group_requests')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        env={'CollectionAbort':RuntimeError,'request_id_for':lambda *x:str(x)}
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),env)
        class Tokenizer:
            def apply_chat_template(self,history,**kw):return [len(history)]
        class Poison(dict):
            def get(self,*a):raise AssertionError('读取了未来消息')
        raw=[{'role':'user','content':'开场'},{'role':'assistant','content':'历史1'},{'role':'tool','content':'已发生工具结果'},{'role':'assistant','content':'历史2'},{'role':'tool','content':'当前已记录结果'},{'role':'assistant','content':'禁止读取Pending输出'},Poison()]
        normalized=[]
        def norm(x):
            self.assertNotEqual(x.get('content'),'禁止读取Pending输出');normalized.append(x);return dict(x)
        group=SimpleNamespace(allocation_round=2,group_ordinal=1,session_by_label={'A':'s'})
        req,audit=env['materialize_group_requests'](Tokenizer(),{'A':raw},group=group,normalize_message=norm,template_input_ids=list)
        self.assertEqual(req[('A',3)]['input_ids'],[5]);self.assertFalse(audit[0]['pending_turn_output_read'])
    def test_frozen_execution_lock(self):
        lock=verify_lock()
        self.assertIn(str(DATA),lock['sha256'])
        self.assertIn(str(PROTOCOL),lock['sha256'])
        self.assertIn(str(HERE/'capture.py'),lock['sha256'])
    def test_lazy_capture_import(self):
        import sys
        before=set(sys.modules)
        from . import capture
        self.assertFalse(any(x.startswith('sglang') for x in set(sys.modules)-before))

if __name__=='__main__':unittest.main()
