"""严格生产规划的冻结回归、拒绝输入与 Legacy 隔离测试。"""
import hashlib
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from flowstate.optimizer import GlobalOptimizer, _exact_admission
from flowstate.recovery_model import RecoveryCostModel
from flowstate.strict_epoch import StrictAllocationEpoch, _digest
from flowstate.workflow import PendingContinuation

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01'
FROZEN=ROOT/'evaluation/strict_compatibility_b4c'


def load(group=1,k=1):
    path=RAW/f'g{group:03d}'/'snapshot.json'
    expected=json.loads((FROZEN/'input_integrity.json').read_text())['sha256'][str(path)]
    return StrictAllocationEpoch.from_snapshot_bytes(path.read_bytes(),expected_sha256=expected,K=k)


def reseal(epoch, **changes):
    """仅用于负例：更新外层摘要，确认内部不变量仍被独立核验。"""
    fields={k:getattr(epoch,k) for k in epoch.__dataclass_fields__}
    fields.update(changes)
    payload={k:tuple(asdict(x) for x in v) if k in ('candidates','pendings','edges') else v for k,v in fields.items() if k!='content_digest'}
    fields['content_digest']=_digest(payload)
    return StrictAllocationEpoch(**fields)


class StrictTests(unittest.TestCase):
    def setUp(self):
        self.epoch=load();self.optimizer=GlobalOptimizer(RecoveryCostModel())

    def test_frozen_16_conditions(self):
        frozen=json.loads((FROZEN/'results.json').read_text())
        count=0
        for snapshot in frozen['snapshots']:
            g=int(snapshot['group'][1:])
            for k in range(1,5):
                with self.subTest(group=g,K=k):
                    epoch=load(g,k)
                    with patch('flowstate.executable_state.is_compatible',side_effect=AssertionError('严格入口调用旧谓词')),patch.object(self.optimizer,'_recovery_cost',side_effect=AssertionError('严格入口调用Legacy评分')):
                        plan=self.optimizer.plan_strict(epoch)
                    # 混合求解合同：规模门禁内走 Exact（对照冻结 Exact OPT 参照，字典序终集），
                    # 超域走原有 Marginal Greedy（对照冻结 FlowState 参照与贪心轨迹顺序）。
                    allowed,_gate=_exact_admission(len(epoch.candidates),len(epoch.pendings),k)
                    self.assertEqual(self.optimizer.last_strict_decision['solver'],'Exact' if allowed else 'Marginal Greedy')
                    policy='Exact OPT' if allowed else 'FlowState'
                    ref=next(r for r in snapshot['rows'] if r['K']==k and r['policy']==policy)
                    self.assertEqual(list(plan.selected_ids),sorted(ref['selected']) if allowed else ref['selected'])
                    if allowed:
                        # 门禁内质量不得劣于冻结 Greedy 参照
                        greedy_ref=next(r for r in snapshot['rows'] if r['K']==k and r['policy']=='FlowState')
                        self.assertLessEqual(plan.recovery_cost_ms,greedy_ref['J_ms']+1e-9)
                    else:
                        self.assertEqual(list(plan.selected_ids),[x['chosen'] for x in snapshot['traces'][f'{k}/FlowState']])
                    self.assertLessEqual(plan.selected_count,k)
                    self.assertAlmostEqual(plan.recovery_cost_ms,ref['J_ms'],places=9)
                    self.assertEqual(set(plan.logical_eviction_ids),{c.candidate.checkpoint_id for c in epoch.candidates}-set(plan.selected_ids))
                    for actual,expected in zip(plan.pending,ref['per_pending']):
                        self.assertEqual((actual.pending_id,actual.T,actual.E,actual.G),(expected['continuation_id'],expected['T'],expected['E'],expected['G']))
                        self.assertAlmostEqual(actual.cost_ms,expected['cost_ms'],places=9)
                    for step in snapshot['traces'][f'{k}/FlowState']:
                        def cost(ids):
                            return sum(RecoveryCostModel().estimate(p.target-e,p.target) for p,e in zip(epoch.pendings,epoch.frontiers(ids)))
                        before=cost(step['prefix'])
                        for cid,gain in step['marginals_ms'].items():
                            self.assertAlmostEqual(before-cost(step['prefix']+[cid]),gain,places=9)
                    count+=1
        self.assertEqual(count,16)

    def test_all_cross_workflow_edges(self):
        count=0
        for g in range(1,5):
            epoch=load(g);cs={c.candidate.checkpoint_id:c for c in epoch.candidates};ps={p.pending_id:p for p in epoch.pendings}
            for e in epoch.edges:
                if e.status=='EPOCH_ADMISSIBLE' and cs[e.candidate_id].candidate.workflow_id!=ps[e.pending_id].workflow_id:
                    i=list(ps).index(e.pending_id)
                    self.assertEqual(epoch.frontiers([e.candidate_id])[i],cs[e.candidate_id].candidate.token_pos)
                    count+=1
        self.assertEqual(count,12)

    def test_unknown(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):reseal(self.epoch,edges=(replace(self.epoch.edges[0],status='UNKNOWN'),)+self.epoch.edges[1:])
    def test_missing_pair(self):
        with self.assertRaisesRegex(ValueError,'缺失配对'):reseal(self.epoch,edges=self.epoch.edges[:-1])
    def test_duplicate_pair(self):
        with self.assertRaisesRegex(ValueError,'重复或冲突'):reseal(self.epoch,edges=self.epoch.edges+(self.epoch.edges[0],))
    def test_conflicting_pair(self):
        with self.assertRaisesRegex(ValueError,'重复或冲突'):reseal(self.epoch,edges=self.epoch.edges+(replace(self.epoch.edges[0],status='REJECTED'),))
    def test_duplicate_candidate(self):
        with self.assertRaisesRegex(ValueError,'Candidate ID'):reseal(self.epoch,candidates=self.epoch.candidates+(self.epoch.candidates[0],))
    def test_duplicate_pending(self):
        with self.assertRaisesRegex(ValueError,'Pending ID'):reseal(self.epoch,pendings=self.epoch.pendings+(self.epoch.pendings[0],))
    def test_out_of_domain(self):
        with self.assertRaisesRegex(ValueError,'超出当前域'):reseal(self.epoch,edges=(replace(self.epoch.edges[0],pending_id='不存在'),)+self.epoch.edges[1:])
    def test_position(self):
        c=self.epoch.candidates[0]
        with self.assertRaisesRegex(ValueError,'位置非法'):reseal(self.epoch,candidates=(replace(c,candidate=replace(c.candidate,token_pos=c.candidate.token_pos+1)),)+self.epoch.candidates[1:])
    def test_target(self):
        p=self.epoch.pendings[0]
        with self.assertRaisesRegex(ValueError,'位置非法'):reseal(self.epoch,pendings=(replace(p,target=len(p.token_ids)+1),)+self.epoch.pendings[1:])
    def test_epoch_mismatch(self):
        with self.assertRaisesRegex(ValueError,'Epoch 身份'):reseal(self.epoch,pendings=(replace(self.epoch.pendings[0],epoch_id='另一快照'),)+self.epoch.pendings[1:])
    def test_config_mismatch(self):
        with self.assertRaisesRegex(ValueError,'配置身份'):reseal(self.epoch,pendings=(replace(self.epoch.pendings[0],configuration_digest='0'*64),)+self.epoch.pendings[1:])
    def test_token_digest(self):
        with self.assertRaisesRegex(ValueError,'输入摘要'):reseal(self.epoch,pendings=(replace(self.epoch.pendings[0],token_digest='0'*64),)+self.epoch.pendings[1:])
    def test_source_digest(self):
        with self.assertRaisesRegex(ValueError,'Snapshot 输入摘要'):StrictAllocationEpoch.from_snapshot_bytes((RAW/'g001/snapshot.json').read_bytes(),expected_sha256='0'*64,K=1)
    def test_epoch_digest(self):
        with self.assertRaisesRegex(ValueError,'内容摘要'):replace(self.epoch,K=2)
    def test_incomplete_conditions(self):
        e=self.epoch.edges[0]
        with self.assertRaisesRegex(ValueError,'条件缺失'):reseal(self.epoch,edges=(replace(e,conditions=e.conditions[:-1]),)+self.epoch.edges[1:])
    def test_admissible_rejected_conflict(self):
        with self.assertRaisesRegex(ValueError,'证据冲突'):reseal(self.epoch,edges=(replace(self.epoch.edges[0],status='REJECTED'),)+self.epoch.edges[1:])
    def test_missing_component(self):
        c=self.epoch.candidates[0]
        with self.assertRaisesRegex(ValueError,'状态不完整'):reseal(self.epoch,candidates=(replace(c,candidate=replace(c.candidate,recurrent_resident=False)),)+self.epoch.candidates[1:])
    def test_strict_legacy_mixing(self):
        with self.assertRaises(TypeError):self.optimizer.plan_strict(())
        with self.assertRaises(TypeError):self.optimizer.select(self.epoch.pendings,[c.candidate for c in self.epoch.candidates],1)
        with self.assertRaises(TypeError):self.optimizer.select([],self.epoch.candidates,1)
    def test_legacy_still_rejects_cross_workflow(self):
        epoch=load();p=epoch.pendings[0];c=next(c for c in epoch.candidates if c.candidate.checkpoint_id=='RQ3_G001_B_TURN_001')
        pending=PendingContinuation(p.pending_id,p.workflow_id,(p.workflow_id,),p.target,p.target)
        self.assertEqual(self.optimizer.select([pending],[c.candidate],c.candidate.memory_bytes).selected,())
        self.assertGreater(epoch.frontiers([c.candidate.checkpoint_id])[0],0)
    def test_no_physical_execution(self):
        with patch('flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only',side_effect=AssertionError('规划不允许驱逐')):
            self.assertEqual(self.optimizer.plan_strict(self.epoch).selected_count,1)
    def test_mutable_input(self):
        with self.assertRaisesRegex(ValueError,'不可变'):reseal(self.epoch,candidates=list(self.epoch.candidates))
    def test_zero_budget(self):
        plan=self.optimizer.plan_strict(load(k=0));self.assertEqual(plan.selected_ids,());self.assertTrue(all(p.E==0 for p in plan.pending))
    def test_saturation_stop_and_tie(self):
        epoch=load(k=100);plan=self.optimizer.plan_strict(epoch)
        self.assertEqual(plan.selected_count,5)
        self.assertEqual(plan.recovery_cost_ms,0)
        self.assertEqual(plan.selected_ids[3],'RQ3_G001_A_TURN_003')
        self.assertEqual(plan.selected_ids[4],'RQ3_G001_D_TURN_003')
    def test_rejected_edges_zero(self):
        for e in self.epoch.edges:
            if e.status=='REJECTED':
                index=[p.pending_id for p in self.epoch.pendings].index(e.pending_id)
                self.assertEqual(self.epoch.frontiers([e.candidate_id])[index],0)

if __name__=='__main__':unittest.main()
