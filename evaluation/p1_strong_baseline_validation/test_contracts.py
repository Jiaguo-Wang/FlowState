"""用确定的边界和解析案例验证冻结算法定义，避免仅检验正式结果是否有利。"""
import unittest
from unittest.mock import patch
from common import Candidate as C, Pending as P, Epoch, score, MODEL
import pcb,ldr,wfc

class Contracts(unittest.TestCase):
    """对选择规则、模型隔离和数值排序进行独立检查。"""
    def setUp(self):
        self.modules=(pcb,ldr,wfc)
    def test_empty_and_zero_budget(self):
        for mod in self.modules:
            self.assertEqual(mod.select(Epoch((),(P('p',10),),2))[0],())
            self.assertEqual(mod.select(Epoch((C('c',5,('p',)),),(P('p',10),),0))[0],())
    def test_zero_and_incompatible_stop(self):
        e=Epoch((C('a',0,('p',)),C('b',5,())),(P('p',10),),8)
        for mod in self.modules:self.assertEqual(mod.select(e)[0],())
    def test_exact_tie_uses_identifier(self):
        e=Epoch((C('z',5,('p',)),C('a',5,('p',))),(P('p',10),),3)
        for mod in self.modules:self.assertEqual(mod.select(e)[0],('a',))
    def test_shared_id_dedup_continues_scan(self):
        e=Epoch((C('s',6,('p1','p2')),C('b',3,('p3',))),(P('p1',10),P('p2',10),P('p3',10)),2)
        for mod in self.modules:self.assertEqual(set(mod.select(e)[0]),{'s','b'})
    def test_partial_frontier_counts_as_improvement(self):
        e=Epoch((C('s',4,('p1','p2')),C('a',6,('p1',)),C('b',6,('p2',))),(P('p1',10),P('p2',10)),2)
        self.assertEqual(wfc.select(e)[0],('s','a'))
        self.assertEqual(pcb.select(e)[0],('a','b'))
        self.assertEqual(ldr.select(e)[0],('a','b'))
        step=wfc.select(e)[1]['steps'][1];self.assertEqual((step['q'],step['r']),(1,2))
    def test_coverage_count_has_priority_over_length(self):
        e=Epoch((C('s',1,('p1','p2')),C('a',9,('p1',))),(P('p1',10),P('p2',10)),1)
        self.assertEqual(wfc.select(e)[0],('s',))
    def test_position_aware_reversal(self):
        e=Epoch((C('a',1024,('p1',)),C('b',1088,('p2',))),(P('p1',131072),P('p2',8192)),1)
        self.assertEqual(pcb.select(e)[0],('a',))
        self.assertEqual(ldr.select(e)[0],('b',))
        self.assertEqual(wfc.select(e)[0],('b',))
        self.assertLess(score(e,('a',))['cost_ms'],score(e,('b',))['cost_ms'])
    def test_proxy_selectors_never_call_phi(self):
        e=Epoch((C('a',5,('p',)),),(P('p',10),),1)
        with patch.object(type(MODEL),'estimate',side_effect=AssertionError('代理选择器调用了 Phi')):
            self.assertEqual(ldr.select(e)[0],('a',));self.assertEqual(wfc.select(e)[0],('a',))
    def test_static_recommendation_does_not_add_shared_ancestor(self):
        e=Epoch((C('s',4,('p1','p2')),C('a',6,('p1',)),C('b',6,('p2',))),(P('p1',10),P('p2',10)),1)
        self.assertEqual(pcb.select(e)[0],('a',));self.assertEqual(ldr.select(e)[0],('a',))
    def test_scoring_rejects_illegal_sets(self):
        e=Epoch((C('a',5,('p',)),C('b',6,('p',))),(P('p',10),),1)
        for ids in [('x',),('a','a'),('a','b')]:
            with self.assertRaises(ValueError):score(e,ids)

if __name__=='__main__':unittest.main()
