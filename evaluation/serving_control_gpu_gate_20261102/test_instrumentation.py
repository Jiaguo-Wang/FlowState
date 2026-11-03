"""只对新增计时与副本一致性做CPU回归，不触碰设备。"""
import ast
import copy
import json
from pathlib import Path
import unittest
from . import interface, codec
from evaluation.serving_control_interface_gate_20261101 import interface as old
from evaluation.serving_control_interface_gate_20261101.test_interface import frozen

PKG = Path(__file__).resolve().parent
SOURCE = PKG.parent / 'serving_control_interface_gate_20261101'


class InstrumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = frozen()

    def test_old_new_modes_equivalent(self):
        for selected_mode in old.MODES:
            with self.subTest(mode=selected_mode):
                a, b = copy.deepcopy(self.fixture), copy.deepcopy(self.fixture)
                aa, bb = a['admission_baseline'], b['admission_baseline']
                old.build(a, lambda: aa, selected_mode)
                calls = []
                def build():
                    calls.append('全部原始SHA')
                    return bb
                interface.build(b, build, selected_mode)
                m = b.pop('admission_interface_metrics')
                self.assertEqual(calls, ['全部原始SHA'])
                self.assertEqual(a, b)
                self.assertEqual(m['full_baseline']['end_ns'], m['projection']['start_ns'])
                self.assertEqual(m['projection']['end_ns'], m['server_validation']['start_ns'])
                self.assertTrue(all(m[k]['wall_ms'] >= 0 for k in ('full_baseline','projection','server_validation')))

    def test_original_content_and_safety_functions_identical(self):
        for file in ('control.py','observation.py','protocol.py','codec.py','bridge.py'):
            self.assertEqual((PKG/file).read_bytes(), (SOURCE/file).read_bytes(), file)
        def functions(path):
            return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        a, b = functions(SOURCE/'interface.py'), functions(PKG/'interface.py')
        for name in a.keys()-{'build', 'project'}:
            self.assertEqual(a[name], b[name], name)
        expected = (SOURCE/'interface.py').read_text().replace("current['token_ids'] == n['token_ids']",
            "tuple(current['token_ids']) == tuple(n['token_ids'])").replace("current['slots'] == n['slots']",
            "tuple(current['slots']) == tuple(n['slots'])")
        expected_project = next(n for n in ast.parse(expected).body if isinstance(n, ast.FunctionDef) and n.name=='project')
        self.assertEqual(ast.dump(expected_project), b['project'])

    def test_real_scheduler_tuple_representation(self):
        for selected_mode in ('serving', 'shadow_validation'):
            with self.subTest(mode=selected_mode):
                r = copy.deepcopy(self.fixture)
                for c in r['epoch']['candidates']:
                    c['token_ids'] = tuple(c['token_ids'])
                r['epoch']['candidates'] = tuple(r['epoch']['candidates'])
                for n in r['after']['nodes']:
                    for key in ('token_ids','slots','component_ids'):
                        n[key] = tuple(n[key])
                r['after']['nodes'] = tuple(r['after']['nodes'])
                r['after']['free_slots'] = tuple(r['after']['free_slots'])
                r['plan']['selected_ids'] = tuple(r['plan']['selected_ids'])
                full = r['admission_baseline']
                with self.assertRaises(ValueError):
                    old.project(full, r)
                interface.build(r, lambda: full, selected_mode)
                interface.validate_response(r, selected_mode)
                decoded, _ = codec.unpack(codec.pack(r))
                interface.validate_response(decoded, selected_mode)

    def test_builder_failure_propagates(self):
        for selected_mode in ('serving','shadow_validation'):
            with self.subTest(mode=selected_mode), self.assertRaises(RuntimeError):
                interface.build(copy.deepcopy(self.fixture), lambda: (_ for _ in ()).throw(RuntimeError('原SHA读取失败')), selected_mode)

    def test_persistence_and_all_gate_assertions_preserved(self):
        a, b = (SOURCE/'runner.py').read_text(), (PKG/'runner.py').read_text()
        def guards(text):
            return [ast.dump(n) for n in ast.walk(ast.parse(text))
                    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require']
        self.assertTrue(set(guards(a)) <= set(guards(b)))
        self.assertLess(b.index('value = call_saved('), b.index('interface.validate_response('))
        self.assertIn('engine.shutdown()', b)
        self.assertIn("len(audit_entries) == 2", b)
        self.assertEqual((PKG/'device_entry.py').read_text(), (SOURCE/'device_entry.py').read_text().replace(
            'evaluation.serving_control_interface_gate_20261101.runner', 'evaluation.serving_control_gpu_gate_20261102.runner'))

    def test_new_metrics_wire_roundtrip(self):
        b = copy.deepcopy(self.fixture)
        full = b['admission_baseline']
        interface.build(b, lambda: full, 'shadow_validation')
        value, _ = codec.unpack(codec.pack(b, verify=True))
        self.assertEqual(value, b)
        interface.validate_response(value, 'shadow_validation')


if __name__ == '__main__':
    unittest.main()
