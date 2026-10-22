"""在冻结镜像中使用真实Req/Batch进行CPU合同验证，不创建Engine。"""
import inspect
import json
import unittest
from array import array
from pathlib import Path
import torch
from sglang.srt.managers.schedule_batch import Req,ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from evaluation.runtime_control_closure_r2c1b2.observation import TARGET,observe_batch,wrap_run_batch,real_api_preflight

class RealApiTests(unittest.TestCase):
    def make(self,start=2432,end=4394):
        root=Path(__file__).resolve().parents[2]
        raw=json.loads((root/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g001/snapshot.json').read_text())
        ids=next(x['token_ids'] for x in raw['pendings'] if x['continuation_id']==TARGET)
        req=Req(TARGET,'',array('q',ids),SamplingParams(temperature=0,max_new_tokens=8,ignore_eos=True))
        req._refresh_fill_ids();req.set_extend_range(start,end);req.prefix_indices=torch.arange(start,device='cpu')
        batch=ScheduleBatch(reqs=[req],device='cpu',forward_mode=ForwardMode.EXTEND,prefix_lens=[start],extend_lens=[end-start])
        return req,batch
    def state(self,req,batch):
        return (tuple(req.origin_input_ids),tuple(req.full_untruncated_fill_ids),tuple(req.output_ids),req.extend_range,tuple(req.prefix_indices.tolist()),tuple(batch.prefix_lens),tuple(batch.extend_lens),batch.forward_mode,tuple(vars(req)),tuple(vars(batch)))
    def test_preflight_real_classes(self):
        self.assertEqual(real_api_preflight()['status'],'PASS')
    def test_actual_target_readonly(self):
        req,batch=self.make();before=self.state(req,batch);row=observe_batch(batch)[0]
        self.assertEqual(self.state(req,batch),before)
        self.assertEqual((row['original_input_length'],row['extend_start'],row['extend_end'],row['extend_token_count']),(4394,2432,4394,1962))
        self.assertFalse(hasattr(req,'fill_ids'));self.assertFalse(hasattr(req,'extend_input_len'))
    def test_cold_and_chunk(self):
        for start,end in ((0,4394),(2432,4096)):
            req,batch=self.make(start,end);before=self.state(req,batch)
            self.assertEqual(list(req.get_fill_ids()[len(req.prefix_indices):]),list(req.origin_input_ids[start:end]))
            self.assertEqual(observe_batch(batch)[0]['extend_token_count'],end-start)
            self.assertEqual(self.state(req,batch),before)
    def test_decode_does_not_infer_stale_extend(self):
        req,batch=self.make();batch.forward_mode=ForwardMode.DECODE;req.extend_range=None
        self.assertEqual(observe_batch(batch)[0]['extend_start'],'不适用：非Prefill')
    def test_wrapper_once_and_passthrough(self):
        req,batch=self.make();before=self.state(req,batch);calls=[];records=[];saved=[];sentinel=object()
        def native(owner,b,*a,**kw):calls.append((owner,b,a,kw));return sentinel
        run=wrap_run_batch(native,records,lambda:saved.append(1))
        self.assertIs(run('scheduler',batch,5,key=7),sentinel)
        self.assertEqual(len(calls),1);self.assertEqual(len(saved),2)
        self.assertTrue(records[0]['native_run_returned']);self.assertEqual(self.state(req,batch),before)
    def test_native_exception_propagates_once(self):
        req,batch=self.make();calls=[];records=[];error=RuntimeError('原生错误')
        def native(*a,**kw):calls.append(1);raise error
        with self.assertRaises(RuntimeError) as raised:wrap_run_batch(native,records,lambda:None)(None,batch)
        self.assertIs(raised.exception,error);self.assertEqual(len(calls),1);self.assertNotIn('native_run_returned',records[0])
    def test_optional_missing_unknown(self):
        req,batch=self.make();del req.extend_batch_idx
        self.assertEqual(observe_batch(batch)[0]['optional_extend_batch_idx'],'UNKNOWN')
    def test_required_mismatch_refused(self):
        req,batch=self.make();batch.extend_lens=[1];calls=[]
        with self.assertRaises(ValueError):wrap_run_batch(lambda *a:calls.append(1),[],lambda:None)(None,batch)
        self.assertEqual(calls,[])
    def test_native_source_contract(self):
        source=inspect.getsource(ScheduleBatch.prepare_for_extend)
        self.assertIn('r.get_fill_ids()[len(r.prefix_indices) :]',source)
        self.assertIn('r.extend_range.length',source)
        self.assertIn('resolve_forward_inputs(batch, self.future_map)',inspect.getsource(Scheduler.run_batch))
if __name__=='__main__':unittest.main()
