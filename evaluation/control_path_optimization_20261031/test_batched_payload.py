"""真实CPU张量对照、变化可见性及冻结严格规划回归。"""
import unittest
from types import SimpleNamespace as NS
import torch
from . import batched_payload as batch
from evaluation.multishare_runtime_gate_b5c3.bridge import payload as original_payload
from evaluation.runtime_control_closure_r2c1a.bridge import selected_payload as original_selected
from evaluation.exact_reconciliation_correctness_gate_b5d3.control import fa_payload as original_fa


class HybridLinearKVPool:
    def __init__(self, dtype, noncontiguous):
        self.full_attention_layer_id_mapping = {3: 0, 1: 1}
        self.layer_transfer_counter = None
        self.full_kv_pool = NS(is_quantized_kv_cache=False, layer_transfer_counter=None)
        self.data = {}
        for layer in [1, 3]:
            for kind in ['K', 'V']:
                value = torch.arange(384).reshape(12, 4, 8).to(dtype)
                self.data[layer, kind] = value.transpose(1, 2) if noncontiguous else value
    def get_key_buffer(self, layer):
        return self.data[layer, 'K']
    def get_value_buffer(self, layer):
        return self.data[layer, 'V']


class BatchTests(unittest.TestCase):
    def setup_case(self, dtype=torch.bfloat16, noncontiguous=True):
        values = [torch.arange(360).reshape(3, 10, 12).to(dtype) + i for i in range(3)]
        if noncontiguous:
            values = [t.transpose(0, 2) for t in values]
        self.pool = HybridLinearKVPool(dtype, noncontiguous)
        self.cache = NS(req_to_token_pool=NS(mamba_pool=NS(mamba_cache=NS(conv=values[:2], temporal=values[2]))),
                        token_to_kv_pool_allocator=NS(get_kvcache=lambda: self.pool))
        self.view = NS(nodes=[NS(node_id=4, token_ids=(1, 2), slots=(7, 2)),
                              NS(node_id=9, token_ids=(1, 3), slots=(4,)),
                              NS(node_id=12, token_ids=(1,), slots=())])
        self.plan = NS(selected_ids=('c9',))
        self.epoch = NS(candidates=[NS(candidate=NS(checkpoint_id='c9'), token_ids=(1, 3))])

    def test_exact_bytes(self):
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for layout in [True, False]:
                self.setup_case(dtype, layout)
                self.assertEqual(original_payload(self.cache, self.view), batch.payload(self.cache, self.view))
                self.assertEqual(original_selected(self.cache, self.view, self.plan, self.epoch),
                                 batch.selected_payload(self.cache, self.view, self.plan, self.epoch))
                for indices in [[], [0, 2, 8], [11, 2, 2, 5]]:
                    self.assertEqual(original_fa(self.cache, indices), batch.fa_payload(self.cache, indices))

    def test_mutation_is_not_cached(self):
        self.setup_case()
        old = batch.payload(self.cache, self.view)
        self.cache.req_to_token_pool.mamba_pool.mamba_cache.temporal[:, 4].fill_(99)
        self.assertNotEqual(old, batch.payload(self.cache, self.view))
        old = batch.fa_payload(self.cache, [2])
        self.pool.data[1, 'K'][2].fill_(42)
        self.assertNotEqual(old, batch.fa_payload(self.cache, [2]))

    def test_invalid_slot(self):
        self.setup_case()
        for slot in [-1, 10]:
            with self.assertRaises((AssertionError, IndexError)):
                batch.payload(self.cache, NS(nodes=[NS(node_id=1, slots=(slot,))]))

    def test_transfer_and_quantization(self):
        self.setup_case()
        self.pool.layer_transfer_counter = object()
        with self.assertRaises(AssertionError):
            batch.fa_payload(self.cache, [1])
        self.pool.layer_transfer_counter = None
        self.pool.full_kv_pool.is_quantized_kv_cache = True
        with self.assertRaises(AssertionError):
            batch.fa_payload(self.cache, [1])

    def test_frozen_plan(self):
        from evaluation.p0_control_inclusive_harness import select_entry
        ctx = select_entry.build_ck3_epoch()
        first, _ = select_entry.select_flowstate(ctx)
        second, _ = select_entry.select_flowstate(ctx)
        self.assertEqual(first, second)
        self.assertEqual(len(first.selected_ids), 3)
        self.assertEqual(len(first.logical_eviction_ids), 13)
        self.assertEqual(len(ctx['epoch'].candidates), 16)


if __name__ == '__main__':
    unittest.main()
