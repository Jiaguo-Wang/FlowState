"""T2-G1 统一接入的 CPU 验证测试：三策略合同、决策输入合法性、冻结不变、确定性。

运行方式（仓库根目录）：python3 -m unittest evaluation.t2g1_three_policy_gpu.test_adapter -v
"""
import hashlib
import json
import unittest
from pathlib import Path

from evaluation.kvflow_inspired_retention_cpu import policy as kvflow
from evaluation.t2g1_three_policy_gpu.adapter import K, THREE_POLICIES, expected_E_G, load_case, select_three

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
PILOT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case, cls.gpu_pendings = load_case()
        cls.sha_before = hashlib.sha256((MAIN / 'workloads.json').read_bytes()).hexdigest()
        cls.pilot = json.loads((PILOT / 'policy_plans.json').read_text())

    def test_three_policies_contract(self):
        """三策略：K=3、唯一、⊆16 候选域、⊆物理物化域、E/G 覆盖恰为四 GPU Pending。"""
        results, _ = select_three(self.case)
        mapping = json.loads((ROOT / 'evaluation/baseline_reuse_t2r/historical_physical_mapping.json').read_text())
        physical = {r['cpu_checkpoint_id'] for r in mapping['rows']}
        for policy, (selected, _) in results.items():
            with self.subTest(policy=policy):
                self.assertEqual(len(selected), K)
                self.assertEqual(len(set(selected)), K)
                self.assertTrue(set(selected) <= {c['checkpoint_id'] for c in self.case['candidates']})
                self.assertTrue(set(selected) <= physical)
                eg = expected_E_G(self.case, self.gpu_pendings, list(selected))
                self.assertEqual(sorted(eg), ['C_A', 'C_B', 'C_C', 'C_D'])
                for pid, e in eg.items():
                    self.assertEqual(e['T'], self.gpu_pendings[pid]['planning_target_T'])
                    self.assertEqual(e['G'], e['T'] - e['E'])

    def test_fs_reproduces_pilot(self):
        """FS 必须逐位复现已通过 GPU 验证的 pilot 计划（选集与四 Pending E/G 预期）。"""
        results, _ = select_three(self.case)
        sel = list(results['FlowState-Greedy'][0])
        self.assertEqual(sel, self.pilot['FlowState-Greedy']['selected_checkpoint_ids'])
        self.assertEqual(expected_E_G(self.case, self.gpu_pendings, sel),
                         self.pilot['FlowState-Greedy']['expected_E_G'])

    def test_decision_inputs_legal(self):
        """决策时刻输入合法：全部来自冻结声明；C_JOIN（CPU 独有 future）不进入决策评分域。"""
        case = self.case
        results, ep = select_three(case)
        self.assertEqual([p.identifier for p in ep.pending], ['C_A', 'C_B', 'C_C', 'C_D'], '评分域必须恰为四 GPU Pending')
        # Marconi：recency 全有限且来自声明访问事件；父子链合法（select 内部已验证，此处复核存在性）
        for c in case['candidates']:
            self.assertIsInstance(c['last_access_order'], int)
        # KVFlow：STE 无 UNKNOWN
        pri = kvflow.priorities(case['candidates'], case['graph'], case['declarations'], case['epoch_id'], case['configuration_digest'])
        self.assertTrue(all(r['priority'] is not None for r in pri))

    def test_determinism(self):
        """两次运行三策略选集完全一致。"""
        a, _ = select_three(self.case)
        b, _ = select_three(self.case)
        for policy in THREE_POLICIES:
            self.assertEqual(a[policy][0], b[policy][0])

    def test_frozen_unchanged(self):
        """接入与测试全部运行后，冻结 workloads.json 摘要不变。"""
        self.assertEqual(hashlib.sha256((MAIN / 'workloads.json').read_bytes()).hexdigest(), self.sha_before)


if __name__ == '__main__':
    unittest.main()
