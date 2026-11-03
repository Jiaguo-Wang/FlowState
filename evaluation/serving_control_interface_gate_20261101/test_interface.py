"""冻结真实响应投影、异常拒绝及隔离调用链的CPU回归。"""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import interface, codec
from evaluation.exact_greedy_paired_gpu_fix_b5d4a.persistence import call_saved

PKG = Path(__file__).resolve().parent
SOURCE = PKG.parent / 'control_path_separation_20261101'


def frozen(policy='fs', repeat=1):
    """读取真实GPU产物，仅保留本接口测试所需字段，不制造成功响应。"""
    path = SOURCE / f'arms/t{repeat}_{policy}/run/reconciliation.json'
    value = json.loads(path.read_text())
    keys = ('ok', 'status', 'epoch', 'plan', 'after', 'reconciliation',
            'selected_payload_after', 'fa_map_before', 'fa_protected_payload', 'admission_baseline')
    return {k: value[k] for k in keys}


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = frozen()

    def setUp(self):
        self.result = copy.deepcopy(self.fixture)

    def serving(self):
        return interface.build(self.result, lambda: self.result['admission_baseline'], 'serving')

    def test_frozen_three_policies_five_rounds(self):
        for repeat in range(1, 6):
            for policy in ('fs', 'mc', 'kv'):
                with self.subTest(repeat=repeat, policy=policy):
                    result = frozen(policy, repeat)
                    original = copy.deepcopy(result)
                    full = result['admission_baseline']
                    interface.build(result, lambda: full, 'serving')
                    post = json.loads((SOURCE / f'arms/t{repeat}_{policy}/run/post_request.json').read_text())
                    self.assertEqual(result['admission_baseline']['selected_components'],
                                     post['state']['selected_components'])
                    for key in original.keys() - {'admission_baseline'}:
                        self.assertEqual(result[key], original[key], key)
                    self.assertEqual(post['state']['selected_payload'], result['selected_payload_after'])
                    self.assertEqual(post['state']['fa_protected_payload'], result['fa_protected_payload'])
                    self.assertEqual(len(post['trace']['audit']), 2)

    def test_validation_original_object(self):
        full = self.result['admission_baseline']
        interface.build(self.result, lambda: full, 'validation')
        self.assertIs(self.result['admission_baseline'], full)
        self.assertEqual(set(full), interface.FULL_FIELDS)

    def test_shadow_preserves_complete_validation(self):
        full = self.result['admission_baseline']
        original = copy.deepcopy(full)
        interface.build(self.result, lambda: full, 'shadow_validation')
        self.assertEqual(full, original)
        self.assertEqual(self.result['admission_baseline_serving'], interface.project(full, self.result))

    def test_full_builder_once_each_mode(self):
        for selected_mode in interface.MODES:
            with self.subTest(mode=selected_mode):
                calls = []
                result = copy.deepcopy(self.fixture)
                full = result['admission_baseline']
                def build():
                    calls.append('完整生产者与全部SHA')
                    return full
                interface.build(result, build, selected_mode)
                self.assertEqual(calls, ['完整生产者与全部SHA'])

    def test_producer_failure_never_falls_back(self):
        for selected_mode in interface.MODES:
            with self.subTest(mode=selected_mode):
                with self.assertRaisesRegex(RuntimeError, '内容读取失败'):
                    interface.build(self.result, lambda: (_ for _ in ()).throw(RuntimeError('内容读取失败')), selected_mode)

    def test_invalid_mode_before_builder(self):
        with self.assertRaises(ValueError):
            interface.build(self.result, lambda: self.fail('不应调用生产者'), '未知模式')

    def test_missing_each_full_field(self):
        for key in interface.FULL_FIELDS:
            with self.subTest(field=key):
                full = dict(self.result['admission_baseline'])
                del full[key]
                with self.assertRaises(ValueError):
                    interface.project(full, self.result)

    def test_missing_each_minimal_field(self):
        self.serving()
        full = self.result['admission_baseline']
        for key in interface.MINIMAL_FIELDS:
            with self.subTest(field=key):
                self.result['admission_baseline'] = dict(full)
                del self.result['admission_baseline'][key]
                with self.assertRaises(ValueError):
                    interface.validate_response(self.result, 'serving')

    def test_unknown_schema_and_mixed_legacy_reject(self):
        self.serving()
        self.result['admission_baseline']['schema'] = '未知协议'
        with self.assertRaises(ValueError):
            interface.validate_response(self.result, 'serving')
        self.result['admission_baseline']['schema'] = interface.SCHEMA
        self.result['admission_baseline']['nodes'] = []
        with self.assertRaises(ValueError):
            interface.validate_response(self.result, 'serving')

    def test_missing_required_component_identity(self):
        self.result['admission_baseline']['selected_components'].pop(next(iter(self.result['admission_baseline']['selected_components'])))
        with self.assertRaises(ValueError):
            self.serving()

    def test_content_protection_not_removed(self):
        for baseline_key in ('selected_payload', 'fa_map', 'fa_protected_payload'):
            with self.subTest(field=baseline_key):
                result = copy.deepcopy(self.fixture)
                full = dict(result['admission_baseline'], **{baseline_key: {}})
                with self.assertRaises(ValueError):
                    interface.project(full, result)

    def test_epoch_plan_scope_and_configuration_mismatch(self):
        cases = [('plan', 'epoch_digest', '错误摘要'), ('plan', 'epoch_id', '过期Epoch'),
                 ('plan', 'K', 2), ('after', 'configuration_digest', '错误配置'),
                 ('after', 'idle', False), ('after', 'admission_closed', False),
                 ('after', 'transfers_pending', True), ('after', 'engine_id', '')]
        for parent, key, value in cases:
            with self.subTest(parent=parent, key=key):
                result = copy.deepcopy(self.fixture)
                result[parent][key] = value
                with self.assertRaises(ValueError):
                    interface.project(result['admission_baseline'], result)

    def test_stale_identity_receipt(self):
        for key in ('engine_id', 'barrier_id', 'lease_id'):
            with self.subTest(key=key):
                result = copy.deepcopy(self.fixture)
                full = result['admission_baseline']
                interface.build(result, lambda: full, 'serving')
                result['after'][key] += '新的生命周期'
                with self.assertRaises(ValueError):
                    interface.validate_response(result, 'serving')

    def test_physical_faults_rejected(self):
        def selected_node(result):
            cid = result['plan']['selected_ids'][0]
            tokens = next(c['token_ids'] for c in result['epoch']['candidates'] if c['candidate']['checkpoint_id'] == cid)
            return next(n for n in result['after']['nodes'] if n['token_ids'] == tokens)
        cases = [('conv_complete', False), ('temporal_complete', False), ('slots', []),
                 ('references', 1), ('host_copy', True), ('protected', True),
                 ('component_ids', []), ('node_id', 999999)]
        for key, value in cases:
            with self.subTest(field=key):
                result = copy.deepcopy(self.fixture)
                selected_node(result)[key] = value
                with self.assertRaises(ValueError):
                    interface.project(result['admission_baseline'], result)
        for key in ('slots', 'component_ids', 'node_id'):
            with self.subTest(alias=key):
                result = copy.deepcopy(self.fixture)
                nodes = [n for n in result['after']['nodes'] if n['slots']]
                nodes[1][key] = nodes[0][key]
                with self.assertRaises(ValueError):
                    interface.project(result['admission_baseline'], result)
        for key, value in [('pool_capacity', 27), ('free_slots', [])]:
            with self.subTest(accounting=key):
                result = copy.deepcopy(self.fixture)
                result['after'][key] = value
                with self.assertRaises(ValueError):
                    interface.project(result['admission_baseline'], result)

    def test_fresh_baseline_slot_and_reference_changes_reject(self):
        for key, value in [('slots', [999]), ('references', [1, 0, 0]), ('full_present', False)]:
            with self.subTest(field=key):
                result = copy.deepcopy(self.fixture)
                n = next(n for n in result['admission_baseline']['nodes'] if n['slots'])
                n[key] = value
                with self.assertRaises(ValueError):
                    interface.project(result['admission_baseline'], result)

    def test_failed_or_partial_reconciliation_reject(self):
        for key, value in [('ok', False), ('status', 'BLOCKED')]:
            result = copy.deepcopy(self.fixture)
            result[key] = value
            with self.assertRaises(ValueError):
                interface.project(result['admission_baseline'], result)
        self.result['reconciliation']['status'] = 'FAIL'
        with self.assertRaises(ValueError):
            self.serving()

    def test_json_wire_equivalence_and_corruption(self):
        self.serving()
        decoded, metrics = codec.unpack(codec.pack(self.result, verify=True))
        self.assertEqual(decoded, self.result)
        interface.validate_response(decoded, 'serving')
        packet = codec.pack(self.result)
        packet['raw_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            codec.unpack(packet)

    def test_client_persistence_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('os.fsync', side_effect=OSError('模拟持久化失败')):
                with self.assertRaisesRegex(OSError, '持久化失败'):
                    call_saved(1, 1, {'action': 'b5d4_apply'}, Path(directory) / 'rpc',
                               connect=lambda *a, **k: self.fail('落盘失败后不得发送请求'))

    def test_corrupt_raw_reply_saved_before_reject(self):
        class Socket:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def settimeout(self, value): pass
            def sendall(self, value): pass
            def makefile(self, value): return io.BytesIO('损坏响应\n'.encode())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'rpc'
            with self.assertRaises((ValueError, UnicodeError)):
                call_saved(1, 1, {'action': 'b5d4_apply'}, target, connect=lambda *a, **k: Socket())
            self.assertEqual((target / 'response.raw').read_bytes(), '损坏响应\n'.encode())
            self.assertTrue((target / 'exception.json').exists())

    def test_core_files_identical_and_hooks_preserved(self):
        for name in ('control.py', 'codec.py', 'protocol.py', 'observation.py'):
            self.assertEqual((PKG / name).read_bytes(), (SOURCE / name).read_bytes(), name)
        def body(directory, file, name):
            tree = ast.parse((directory / file).read_text())
            return ast.dump(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name))
        for name in ('_cache_state', 'cache_state', 'install_observation', 'inputs', 'idle'):
            self.assertEqual(body(PKG, 'bridge.py', name), body(SOURCE, 'bridge.py', name), name)
        source = (PKG / 'runner.py').read_text()
        self.assertLess(source.index('call_saved('), source.index('interface.validate_response('))
        self.assertIn('len(audit_entries) == 2', source)
        for expression in ("post['state']['selected_payload'] == result['selected_payload_after']",
                           "post['state']['fa_protected_payload'] == result['fa_protected_payload']"):
            self.assertIn(expression, source)

    def test_no_cross_epoch_component_reuse(self):
        self.serving()
        old = copy.deepcopy(self.result['admission_baseline'])
        result = frozen('fs', 2)
        result['admission_baseline'] = old
        with self.assertRaises(ValueError):
            interface.validate_response(result, 'serving')

    def test_projection_does_not_alias_original_components(self):
        full = self.result['admission_baseline']
        compact = interface.project(full, self.result)
        cid = next(iter(compact['selected_components']))
        compact['selected_components'][cid][0] = '模拟响应损坏'
        self.assertNotEqual(compact['selected_components'], full['selected_components'])
        self.assertEqual(full, self.fixture['admission_baseline'])

    def test_device_gate_only_changes_runner_destination(self):
        old = (SOURCE / 'device_entry.py').read_text()
        new = (PKG / 'device_entry.py').read_text()
        self.assertEqual(new, old.replace('evaluation.control_path_separation_20261101.runner',
                                         'evaluation.serving_control_interface_gate_20261101.runner'))
        self.assertLess(new.index('torch.cuda.device_count() == 1'), new.index('runpy.run_module('))
        self.assertIn("os.environ['EXPECTED_GPU_UUID']", new)

    def test_serving_requires_existing_failstop_wrapper(self):
        for selected_mode in ('serving', 'shadow_validation'):
            with self.subTest(mode=selected_mode):
                with self.assertRaises(ValueError):
                    interface.validate_runtime_mode(selected_mode, 'original')
                interface.validate_runtime_mode(selected_mode, 'serving')
        interface.validate_runtime_mode('validation', 'original')
        self.assertIn('interface.validate_runtime_mode(BASELINE_MODE', (PKG / 'bridge.py').read_text())
        source = (PKG / 'bridge.py').read_text()
        self.assertIn('except BaseException:\n                scheduler._r2c0_failed = True\n                raise', source)

    def test_original_sources_unchanged(self):
        manifest = json.loads((PKG / 'source_manifest.json').read_text())
        for path, digest in manifest['sources_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest, path)


if __name__ == '__main__':
    unittest.main()
