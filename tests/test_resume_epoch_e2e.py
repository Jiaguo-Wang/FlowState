"""验证新恢复实验的计时边界和冻结选择等价性。"""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from evaluation import resume_epoch_e2e as e2e


class FakeEngine:
    """返回固定的单令牌流，以隔离时钟与校验逻辑。"""
    def generate(self, **kwargs):
        return iter([{'output_ids': []}, {'output_ids': [7], 'meta_info': {
            'completion_tokens': 1, 'prompt_tokens': 2, 'num_retractions': 0}}])


class EpochTests(unittest.TestCase):
    """检验真实选择、串行累计等待和原始时间戳。"""
    def test_stream_boundaries(self):
        clock = iter([1000000, 5000000, 8000000])
        row = e2e.stream_request(FakeEngine(), {'rid':'r','workflow_id':'w','input_ids':[1,2]}, lambda: next(clock))
        self.assertEqual((row['submit_ts'],row['first_token_ts'],row['completion_ts']), (1,5,8))
        self.assertEqual((row['ttft_ms'],row['request_latency_ms']), (4,7))

    def test_epoch_queue_and_control(self):
        rows = [{'submit_ts':s,'first_token_ts':s+1,'completion_ts':s+2,'request_latency_ms':2}
                for s in [10,13,16,19]]
        result = e2e.epoch_metrics(0,10,rows)
        self.assertEqual(result['control_inclusive_epoch_latency_ms'],21)
        self.assertEqual(result['mean_barrier_to_completion_ms'],16.5)
        self.assertEqual(result['between_request_validation_and_dispatch_ms'],3)
        rows[1]['submit_ts'] = 11
        with self.assertRaises(RuntimeError):
            e2e.epoch_metrics(0,10,rows)

    def test_all_frozen_selectors(self):
        plans = json.loads((e2e.FORMAL/'manifest.json').read_text())['runs']
        checked = set()
        for plan in plans:
            key = (plan['snapshot_id'],plan['policy'])
            if key in checked:
                continue
            checked.add(key)
            snapshot = e2e.rq4.create_budget_variant(e2e.rq4._load_allocation_snapshot(Path(plan['snapshot_artifact'])),plan['logical_k'])
            states = {c.checkpoint_id:{'recurrent_resident':True,'fa_resident':True} for c in snapshot.eligible_candidates}
            history = {m.checkpoint_id:m for m in snapshot.candidate_metadata}
            frontiers = {p.workflow_id:p.resident_fa_frontier for p in snapshot.pending_continuations}
            candidates,inputs = e2e.construction(snapshot,states,history,frontiers,plan['policy'])
            actual = e2e.selection(snapshot,candidates,inputs,plan['policy'])
            self.assertEqual(tuple(plan['selected_candidate_ids']),actual,key)
        self.assertEqual(len(checked),72)

    def test_truncation_rejected(self):
        with self.assertRaises(RuntimeError):
            e2e.stream_request(FakeEngine(),{'rid':'r','workflow_id':'w','input_ids':[1]})


if __name__ == '__main__':
    unittest.main()


class AnalysisTests(unittest.TestCase):
    """检查配对方向、分层区间与盈亏平衡边界。"""
    def test_paired_constant_difference(self):
        from evaluation.resume_epoch_e2e_analysis import paired
        rows={str(i):{'round':2+i//6,'LRU':{'m':8},'FlowState':{'m':3}} for i in range(24)}
        result=paired(rows,'LRU','m')
        self.assertEqual(result['mean_difference_ms'],5)
        self.assertEqual(result['bootstrap_95_ci_ms'],[5,5])
        self.assertEqual((result['win'],result['tie'],result['loss']),(24,0,0))

    def test_break_even_strict_gain(self):
        from evaluation.resume_epoch_e2e_analysis import break_even, METRICS
        base={k:0 for k in METRICS}
        fs=dict(base)
        fs['total_control_latency_ms']=10
        fs['policy_observation_ms']=10
        base['mean_request_execution_ms']=2
        result=break_even({'LRU':base,'FlowState':fs},'LRU')
        self.assertEqual(result['total_control_break_even_count'],6)
        self.assertEqual(result['policy_specific_break_even_count'],6)
