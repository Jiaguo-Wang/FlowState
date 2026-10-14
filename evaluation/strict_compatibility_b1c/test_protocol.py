"""用冻结证据和替身对象验证隔离拒绝条件及观察器不重复调用原操作。"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from evaluation.strict_compatibility_b1c.protocol import isolation_plan
from evaluation.strict_compatibility_b1c import runner

B1 = Path('evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json')


class Values:
    def __init__(self, values): self.values = values
    def detach(self): return self
    def cpu(self): return self
    def reshape(self, *args): return self
    def tolist(self): return self.values
    def numel(self): return len(self.values)


class ProtocolChecks(unittest.TestCase):
    def setUp(self):
        data = json.loads(B1.read_text())['positive_arm']
        self.candidates = [{'checkpoint_id': c['checkpoint_id'], 'token_ids': c['token_ids'], 'path': c['state_identity']} for c in data['checkpoints']]
        self.nodes = [{'node_id': c['path']['node_id'], 'token_ids': c['token_ids'], 'mamba_slots': c['path']['target_mamba_slots']} for c in self.candidates]
        self.tokens = next(p['token_ids'] for p in data['pendings'] if p['continuation_id'] == runner.PENDING)

    def test_frozen_plan(self):
        plan = isolation_plan(self.candidates, self.nodes, self.tokens, runner.TARGET)
        self.assertEqual([x['checkpoint_id'] for x in plan], ['RQ3_G000_A_TURN_002', 'RQ3_G000_A_TURN_001'])

    def test_unmanaged_state_rejected(self):
        nodes = self.nodes + [{'node_id': 999, 'mamba_slots': [99], 'token_ids': self.tokens[:4000]}]
        with self.assertRaises(RuntimeError): isolation_plan(self.candidates, nodes, self.tokens, runner.TARGET)

    def test_alias_rejected(self):
        nodes = copy.deepcopy(self.nodes)
        nodes[1]['mamba_slots'] = nodes[0]['mamba_slots']
        with self.assertRaises(RuntimeError): isolation_plan(self.candidates, nodes, self.tokens, runner.TARGET)

    def test_actual_prefix_conflict_rejected(self):
        nodes = copy.deepcopy(self.nodes)
        nodes[0]['token_ids'][0] += 1
        with self.assertRaises(RuntimeError): isolation_plan(self.candidates, nodes, self.tokens, runner.TARGET)

    def test_observer_calls_once(self):
        calls = []
        result = NS(full_kv_hit_length=6, device_indices=Values([0,1,2]), best_match_node=101,
            last_device_node=101,last_host_node=101,mamba_host_hit_length=0,mamba_branching_seqlen=None)
        def native_match(params):
            calls.append('match')
            params.req.mamba_cow_src_index = Values([7])
            params.req.mamba_pool_idx = Values([9])
            return result
        def native_copy(src,dst): calls.append('copy')
        pool = NS(copy_from=native_copy)
        cache = NS(match_prefix=native_match, req_to_token_pool=NS(mamba_pool=pool))
        runner.install_observation(cache)
        req = NS(rid='测试请求', origin_input_ids=[1,2,3])
        actual = cache.match_prefix(NS(req=req,cow_mamba=True))
        self.assertIs(actual,result)
        pool.copy_from(Values([7]),Values([9]))
        self.assertEqual(calls,['match','copy'])
        self.assertEqual(runner.MATCHES['测试请求'][-1]['best_match_node'],101)
        self.assertEqual(runner.COPIES[-1]['request_ids'],['测试请求'])


if __name__ == '__main__': unittest.main()
