"""冻结输入适配与共同目标的独立CPU核验。"""
import unittest
from decimal import Decimal as D
from . import evaluate as e

class EvaluationTests(unittest.TestCase):
    def test_scope(self):
        self.assertEqual(e.GROUPS,(1,2,3,4))
        for g in (0,5,6):
            with self.assertRaises(AssertionError):e.load(g)
    def test_all_results_decimal_and_constraints(self):
        for g in e.GROUPS:
            x=e.evaluate(g);raw,epoch,_=e.load(g);ids={c.identifier for c in epoch.candidates}
            for r in x['rows']:
                self.assertLessEqual(len(r['selected']),r['K']);self.assertEqual(len(set(r['selected'])),len(r['selected']));self.assertTrue(set(r['selected'])<=ids)
                value=D(0)
                for p in r['per_pending']:
                    gap=D(p['G'])/1024;target=D(p['T'])/1024
                    value+=D('37.828150')*gap+D('0.345974143')*gap*target-D('0.156201917')*gap*gap
                self.assertAlmostEqual(float(value),r['J_ms'],places=9)
                self.assertGreaterEqual(r['gap_to_exact_ms'],-e.EPS)
    def test_rejected_and_cross_edges(self):
        for g in e.GROUPS:
            raw,epoch,_=e.load(g);bm=e.common.benefit_matrix(epoch)
            cp={c['checkpoint_id']:c for c in raw['candidates']};pp={p['continuation_id']:p for p in raw['pendings']};cross=0
            for edge in raw['edges']:
                c,p=edge['checkpoint_id'],edge['continuation_id'];i=next(i for i,x in enumerate(epoch.pending) if x.identifier==p)
                if edge['status']=='REJECTED':self.assertEqual(bm[c][i],0)
                elif cp[c]['workflow_id']!=pp[p]['workflow_id']:self.assertGreater(bm[c][i],0);cross+=1
            self.assertEqual(cross,3)
    def test_marginal_and_overlap(self):
        for g in e.GROUPS:
            result=e.evaluate(g);bm=result['benefit_matrix_ms']
            for step in result['traces']['4/FlowState']:
                for cid,gain in step['marginals_ms'].items():
                    expected=sum(max(0,bm[cid][i]-max([bm[c][i] for c in step['prefix']]+[0])) for i in range(4))
                    self.assertAlmostEqual(expected,gain,places=9)
    def test_marconi_original_raw_recency(self):
        for g in e.GROUPS:
            result=e.evaluate(g);md=result['raw_metadata'];rec={c:float(v['last_access_order']) for c,v in md.items()};eff={c:v['marconi_flop_saved']/v['memory_bytes'] for c,v in md.items()}
            def norm(v):return {c:0 if max(v.values())==min(v.values()) else (x-min(v.values()))/(max(v.values())-min(v.values())) for c,x in v.items()}
            nr,ne=norm(rec),norm(eff);ranked=sorted(md,key=lambda c:(-(nr[c]+ne[c]),c))
            for r in result['rows']:
                if r['policy']=='Marconi Adaptation':self.assertEqual(r['selected'],ranked[:r['K']])

if __name__=='__main__':unittest.main()
