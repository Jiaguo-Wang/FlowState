"""核验严格输出规则和Admission证据拒绝条件，不启动Engine。"""
import unittest
from copy import deepcopy
from evaluation.runtime_control_closure_r2c1b1.protocol import admission_gate,compare_outputs
from evaluation.runtime_control_closure_r2c1b1.bridge import load,TARGET,SOURCE
class ProtocolTests(unittest.TestCase):
    def test_frozen_target(self):
        raw,e,p=load();ids={x['checkpoint_id'] for x in raw['edges'] if x['continuation_id']==TARGET and x['status']=='EPOCH_ADMISSIBLE'}
        self.assertEqual(set(p.selected_ids)&ids,{SOURCE})
        x=next(x for x in p.pending if x.pending_id==TARGET);self.assertEqual((x.T,x.E,x.G),(4224,2432,1792))
    def test_admission_gate(self):
        state={'semantic':{'tree':1},'nodes':[2],'selected_payload':{'b':3}}
        base={'admission_baseline':state,'selected_payload_after':state['selected_payload']}
        admission_gate(base,state,[1],[1])
        for key in state:
            bad=deepcopy(state);bad[key]=None
            with self.subTest(key=key),self.assertRaises(ValueError):admission_gate(base,bad,[1],[1])
        with self.assertRaises(ValueError):admission_gate(base,state,[2],[1])
    def test_output_gate(self):
        out={'output_ids':list(range(8)),'meta_info':{'completion_tokens':8,'finish_reason':{'type':'length','length':8}}}
        self.assertEqual(compare_outputs(out,out)['status'],'PASS')
        for bad in (dict(out,output_ids=list(range(7))),dict(out,output_ids=[9]*8),dict(out,meta_info={'completion_tokens':8,'finish_reason':{'type':'stop'}})):
            with self.assertRaises(ValueError):compare_outputs(out,bad)
if __name__=='__main__':unittest.main()
