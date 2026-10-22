"""纯CPU合同测试；小型Token数组仅为单测，不冒充工作负载。"""
import copy
import unittest
from .policy import compute_ste, priorities, select, IncompleteEvidence


def node(name, parents=(), mode='AND', ready=False):
    return dict(invocation_id=name, predecessors=list(parents), aggregation=mode, ready=ready, epoch_id='e', known_at_epoch=True, evidence_path='单测声明')


def declaration(name, fixed, full=None):
    return dict(invocation_id=name, fixed_token_ids=fixed, input_token_ids=full or fixed, dynamic_suffix_confirmed=True, epoch_id='e', configuration_digest='cfg', known_at_epoch=True, evidence_path='单测声明')


def candidate(name, tokens):
    return dict(checkpoint_id=name, token_ids=tokens, token_position_L=len(tokens), barrier_id='e', configuration_digest='cfg')


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.g = [node('a', ready=True), node('b', ['a'])]
        self.d = [declaration('a', [1,2], [1,2,9]), declaration('b', [1,3])]
        self.c = [candidate('shared', [1]), candidate('a', [1,2]), candidate('b', [1,3]), candidate('suffix', [1,2,9])]

    def test_ready_and_or(self):
        g = self.g + [node('c',['a','b']),node('d',['a','b'],'OR')]
        self.assertEqual(compute_ste(g,'e'),dict(a=0,b=1,c=2,d=1))

    def test_unknown_missing_parent(self):
        self.assertIsNone(compute_ste([node('a',['missing'])],'e')['a'])

    def test_cycle_unknown(self):
        self.assertEqual(compute_ste([node('a',['b']),node('b',['a'])],'e'),dict(a=None,b=None))

    def test_future_not_used(self):
        self.g[0]['known_at_epoch']=False
        self.assertEqual(compute_ste(self.g,'e'),dict(a=None,b=None))

    def test_shared_min_and_dynamic_last(self):
        selected, rows=select(self.c,self.g,self.d,'e','cfg',3)
        self.assertEqual(selected,('a','shared','b'))
        self.assertEqual(rows[0]['priority'],0)
        self.assertEqual(rows[-1]['priority'],float('inf'))

    def test_tie_is_deterministic(self):
        self.assertEqual(select(self.c[::-1],self.g,self.d,'e','cfg',1)[0],('a',))

    def test_unknown_never_guessed(self):
        self.g[0]['ready']=False
        with self.assertRaises(IncompleteEvidence): select(self.c,self.g,self.d,'e','cfg',1)

    def test_missing_fixed(self):
        self.d[0]['fixed_token_ids']=None
        with self.assertRaises(IncompleteEvidence): select(self.c,self.g,self.d,'e','cfg',1)

    def test_prefix_not_workflow_label(self):
        self.c.append(candidate('unrelated',[7]))
        with self.assertRaises(IncompleteEvidence): select(self.c,self.g,self.d,'e','cfg',1)

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError): select(self.c+self.c[:1],self.g,self.d,'e','cfg',1)

    def test_epoch_configuration_and_position(self):
        for key,value in [('barrier_id','other'),('configuration_digest','other'),('token_position_L',100)]:
            with self.subTest(key=key):
                cs=copy.deepcopy(self.c);cs[0][key]=value
                with self.assertRaises(IncompleteEvidence):select(cs,self.g,self.d,'e','cfg',1)

    def test_budget(self):
        for k in [-1,5,True,1.5]:
            with self.subTest(k=k),self.assertRaises(ValueError):select(self.c,self.g,self.d,'e','cfg',k)
        self.assertEqual(select(self.c,self.g,self.d,'e','cfg',0)[0],())

    def test_no_mutation(self):
        before=copy.deepcopy((self.c,self.g,self.d));select(self.c,self.g,self.d,'e','cfg',2)
        self.assertEqual(before,(self.c,self.g,self.d))

    def test_no_dynamic_inference(self):
        self.d[0]['dynamic_suffix_confirmed']=False
        self.assertIsNone(priorities(self.c,self.g,self.d,'e','cfg')[-1]['priority'])

if __name__=='__main__':unittest.main()
