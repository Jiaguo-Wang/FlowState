"""真实冻结输入的接线回归与拒绝测试，不启动Runtime。"""
import copy
import csv
import json
import unittest
from pathlib import Path
from .interface import plan_declared, validate_declared_case, require_runtime_execution, POLICIES

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'

class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((MAIN/'workloads.json').read_text())['cases']
        cls.frozen = {(r['structure'],int(r['K']),r['policy']):r for r in csv.DictReader((MAIN/'cpu_comparison.csv').open())}

    def test_all_frozen_conditions(self):
        results=[]
        for case in self.cases:
            for k in [3,5]:
                for policy in POLICIES:
                    with self.subTest(structure=case['structure'],K=k,policy=policy):
                        got=plan_declared(case,k,policy);results.append(got);old=self.frozen[case['structure'],k,policy]
                        self.assertEqual(got['selected'],json.loads(old['selected']))
                        self.assertAlmostEqual(got['J_ms'],float(old['current_J_ms']),places=9)
                        self.assertEqual([(p['E'],p['G']) for p in got['per_pending']],[(p['E'],p['G']) for p in json.loads(old['current_E_G'])])

        (Path(__file__).resolve().parent/'adapter_regression.json').write_text(json.dumps({'说明':'18项接口回归；耗时仅为单次调用诊断，不是正式性能比较。','results':results},ensure_ascii=False,indent=2,allow_nan=False)+'\n')

    def test_corrupted_inputs(self):
        changes = [lambda c:c['graph'][0].update(known_at_epoch=False),lambda c:c['declarations'][0].update(configuration_digest='错误'),lambda c:c['candidates'][0].update(last_access_order=-1),lambda c:c['candidates'][0].update(logical_cost=2),lambda c:c['candidates'][0].update(runtime_handle=27),lambda c:c['candidates'][0].update(token_compatible_calls=[]),lambda c:c['calls'][0].update(ready=False),lambda c:c['candidates'].append(copy.deepcopy(c['candidates'][0])),lambda c:c['declarations'][0].update(fixed_token_ids=[-1]),lambda c:c['candidates'][0].update(parent_candidate='缺失'),lambda c:c['synthetic_access_events'][0].update(runtime_observed=True),lambda c:c['calls'][0].update(planning_target_T=999999)]
        for i,change in enumerate(changes):
            with self.subTest(case=i):
                c=copy.deepcopy(self.cases[2]);change(c)
                with self.assertRaises(ValueError):validate_declared_case(c,3)

    def test_budget(self):
        for k in [-1, True, 99, 1.5]:
            with self.subTest(k=k),self.assertRaises(ValueError):validate_declared_case(self.cases[2],k)

    def test_no_physical_execution(self):
        with self.assertRaises(ValueError):require_runtime_execution(plan_declared(self.cases[2],3,'KVFlow-inspired'))

    def test_marconi_equal_actual_bytes(self):
        from dataclasses import replace
        from evaluation.sota_policies import MarconiStylePolicy
        from flowstate.state_catalog import CheckpointCandidate
        c=self.cases[2]['candidates']
        core=tuple(CheckpointCandidate(x['checkpoint_id'],x['checkpoint_id'],(),x['token_position_L'],1,True) for x in c)
        real=tuple(replace(x,memory_bytes=51511296) for x in core)
        rec={x['checkpoint_id']:x['last_access_order'] for x in c}
        values={x['checkpoint_id']:x['token_position_L'] for x in c}
        for k in [3,5]:
            a=MarconiStylePolicy().select(core,k,rec,values,1.0)
            b=MarconiStylePolicy().select(real,k,rec,values,1.0)
            self.assertEqual(a.selected_checkpoint_ids,b.selected_checkpoint_ids)

    def test_marconi_invalid_metadata(self):
        from evaluation.sota_policies import MarconiStylePolicy
        from flowstate.state_catalog import CheckpointCandidate
        core=(CheckpointCandidate('c','w',(),64,1,True),)
        for rec,values,alpha in [({}, {'c':64},1),({'c':0},{},1),({'c':float('nan')},{'c':64},1),({'c':0},{'c':float('inf')},1),({'c':0},{'c':64},-1)]:
            with self.subTest(rec=rec,values=values,alpha=alpha),self.assertRaises(ValueError):
                MarconiStylePolicy().select(core,1,rec,values,alpha)

    def test_unknown_policy(self):
        with self.assertRaises(ValueError):plan_declared(self.cases[2],3,'完整官方Marconi')

if __name__=='__main__':unittest.main()
