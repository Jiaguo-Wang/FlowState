"""对冻结单快照的矩阵接口与集合目标进行穷举 CPU 检查。"""
import unittest
from evaluation.strict_compatibility_b2 import evaluate as e

class MatrixTests(unittest.TestCase):
    def test_rejected_edges(self):
        for row in e.EDGES:
            i=next(i for i,p in enumerate(e.EPOCH.pending) if p.identifier==row['continuation_id'])
            b=e.BM[row['checkpoint_id']][i]
            self.assertGreaterEqual(b,0)
            if row['status']=='REJECTED':self.assertEqual(b,0)
    def test_cross_workflow_admission(self):
        cid='RQ3_G000_B_TURN_001'
        self.assertEqual(sum(v>0 for v in e.BM[cid]),4)
        self.assertGreater(e.BM[cid][0],0)
        self.assertEqual(sum(r['B1C_evidence_level']=='RESUME_VALIDATED' for r in e.EDGES),1)
    def test_empty_and_objective_equivalence(self):
        self.assertEqual(e.benefit(()),0)
        for s in e.subsets(8):
            self.assertAlmostEqual(e.benefit(s)+e.common.score(e.EPOCH,s)['cost_ms'],e.J0,places=9)
    def test_overlap_not_additive(self):
        a,b='RQ3_G000_A_TURN_002','RQ3_G000_B_TURN_001'
        self.assertLess(e.benefit((a,b)),e.benefit((a,))+e.benefit((b,)))
    def test_all_marginals(self):
        for s in e.subsets(8):
            for c in e.IDS:
                if c in s:continue
                explicit=sum(max(0,e.BM[c][i]-max([e.BM[x][i] for x in s]+[0])) for i in range(4))
                self.assertAlmostEqual(explicit,e.benefit(s+(c,))-e.benefit(s),places=9)
    def test_greedy_against_independent_marginal(self):
        selected=[]
        for k in range(1,5):
            values={c:e.benefit(selected+[c])-e.benefit(selected) for c in e.IDS if c not in selected}
            best=max(values.values())
            selected.append(min(c for c in values if best-values[c]<=e.EPS))
            self.assertEqual(tuple(selected),e.greedy(k))
    def test_all_budgets_and_optimality(self):
        rows,_=e.run()
        for r in rows:
            self.assertEqual(len(r['selected']),len(set(r['selected'])))
            self.assertLessEqual(len(r['selected']),r['K'])
            self.assertGreaterEqual(r['gap_vs_opt_ms'],-e.EPS)
        for k in range(1,5):
            best,_=e.exact(k)
            self.assertTrue(all(e.benefit(s)<=e.benefit(best)+e.EPS for s in e.subsets(k)))

if __name__=='__main__':unittest.main()
