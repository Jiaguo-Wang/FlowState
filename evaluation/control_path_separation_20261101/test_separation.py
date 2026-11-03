"""取证无损性、失败拒绝、时间快照和控制调用等价合同。"""
import ast
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from . import codec
from .control import DeferredJournal

PKG = Path(__file__).resolve().parent
ORIGINAL = PKG.parent/'p0_control_cycle_gpu'


class SeparationTests(unittest.TestCase):
    def sample(self):
        return {'ok': True, 'epoch_id': '当前', 'tuple': (1, 2),
                'plan': {'selected': ['A', 'B'], 'J': 1.253}, 'view': [{'slot': 7, 'hash': 'abc'}]}

    def test_original_wire_equivalence(self):
        value = self.sample()
        packet = codec.pack(value, verify=True)
        decoded, metrics = codec.unpack(packet)
        self.assertEqual(decoded, json.loads(json.dumps(value, sort_keys=True)))
        self.assertTrue(metrics['digest_verified'])
        self.assertTrue(packet['encoding']['equivalence_verified'])

    def test_full_real_snapshot(self):
        path = ORIGINAL/'arms/m1_ck3_fs/run/reconciliation.json'
        value = json.loads(path.read_text())
        packet = codec.pack(value)
        decoded, _ = codec.unpack(packet)
        self.assertEqual(decoded, value)
        self.assertLess(len(packet['payload_b64']), len(path.read_bytes()))

    def test_corruptions_rejected(self):
        packet = codec.pack(self.sample())
        cases = [dict(packet, raw_bytes=0), dict(packet, raw_bytes=codec.MAX_RAW+1),
                 dict(packet, raw_bytes=packet['raw_bytes']-1), dict(packet, raw_sha256='0'*64),
                 dict(packet, payload_b64='!'), dict(packet, codec='其他协议'),
                 dict(packet, ok=False), dict(packet, payload_b64=packet['payload_b64'][:-8])]
        for case in cases:
            with self.subTest(case=next(k for k in case if case[k] != packet.get(k))):
                with self.assertRaises((ValueError, zlib.error)):
                    codec.unpack(case)

    def test_trailing_stream_rejected(self):
        packet = codec.pack(self.sample())
        blob = base64.b64decode(packet['payload_b64']) + zlib.compress(b'{}')
        packet['payload_b64'] = base64.b64encode(blob).decode()
        with self.assertRaises(ValueError):
            codec.unpack(packet)

    def test_duplicate_keys_rejected(self):
        raw = b'{"ok":true,"epoch":"old","epoch":"new"}'
        packet = codec.pack(self.sample())
        packet.update(payload_b64=base64.b64encode(zlib.compress(raw)).decode(), raw_bytes=len(raw),
                      raw_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            codec.unpack(packet)

    def test_no_cross_epoch_or_slot_cache(self):
        value = self.sample()
        first = codec.pack(value)
        value['epoch_id'] = '新窗口'
        value['view'][0].update(slot=8, hash='新的组件内容')
        second = codec.pack(value)
        self.assertNotEqual(first['raw_sha256'], second['raw_sha256'])
        self.assertEqual(codec.unpack(first)[0]['view'][0]['slot'], 7)
        self.assertEqual(codec.unpack(second)[0]['view'][0]['slot'], 8)

    def test_journal_time_snapshot_and_format_equivalence(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'FS_SEPARATION_MODE': 'serving', 'FS_SEPARATION_VERIFY': '1'}):
                new = DeferredJournal(Path(temp)/'new')
                value = self.sample()
                new.record('first', value)
                expected = json.loads(json.dumps(value))
                value['view'][0]['slot'] = 9
                new.mark_window_closed()
                new.record('second', value)
                self.assertFalse(list((Path(temp)/'new').iterdir()))
                new.flush_after_window()
                self.assertEqual(json.loads((Path(temp)/'new/000_first.json').read_text()), expected)
                self.assertEqual(json.loads((Path(temp)/'new/001_second.json').read_text()), json.loads(json.dumps(value)))

    def test_durable_failure_propagates(self):
        with tempfile.TemporaryDirectory() as temp:
            new = DeferredJournal(Path(temp)/'new')
            new.record('test', self.sample())
            with patch('os.fsync', side_effect=OSError('模拟磁盘失败')):
                with self.assertRaises(OSError):
                    new.flush_after_window()

    def test_persistence_precedes_decode(self):
        tree = ast.parse((PKG/'runner.py').read_text())
        client = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'AuditedControlClient')
        text = ast.get_source_segment((PKG/'runner.py').read_text(), client)
        self.assertLess(text.index('call_saved('), text.index('unpack(value)'))
        self.assertIn("report_timing['start_ns'] >= t4_ns", (PKG/'runner.py').read_text())

    def test_original_safety_call_sequence(self):
        for name, function in [('control.py', 'apply'), ('bridge.py', 'dispatch'), ('bridge.py', 'inputs'),
                               ('bridge.py', 'install_observation')]:
            def body(base):
                text = (base/name).read_text()
                node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == function)
                return ast.dump(node, include_attributes=False)
            self.assertEqual(body(ORIGINAL), body(PKG), function)

    def test_frozen_strict_plan_route(self):
        from evaluation.p0_control_inclusive_harness import select_entry
        from flowstate.strict_control import validate_plan
        ctx = select_entry.build_ck3_epoch()
        plan, meta = select_entry.select_flowstate(ctx)
        decoded, _ = codec.unpack(codec.pack({'ok': True, 'epoch': asdict(ctx['epoch']), 'plan': asdict(plan)}))
        self.assertEqual(decoded['epoch'], json.loads(json.dumps(asdict(ctx['epoch']))))
        self.assertEqual(decoded['plan'], json.loads(json.dumps(asdict(plan))))
        self.assertEqual(meta['decision']['solver'], 'Marginal Greedy')
        validate_plan(ctx['epoch'], plan)


if __name__ == '__main__':
    unittest.main()
