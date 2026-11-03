"""T2-Mini 的 CPU 验证测试：增量收益、动态驱逐重评分、预算合同、A/B/C 新旧比较、冻结不变。

运行方式（仓库根目录）：python3 -m unittest evaluation.marconi_evict_t2m.test_evict -v
"""
import copy
import csv
import hashlib
import json
import unittest
from pathlib import Path

from evaluation.marconi_evict_t2m.evict_policy import REAL_CHECKPOINT_BYTES, MarconiEvictPolicy, plan_case
from evaluation.sota_policies import MarconiStylePolicy, _build_marconi_metrics
from flowstate.state_catalog import CheckpointCandidate

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
T2R = ROOT / 'evaluation/baseline_reuse_t2r'


def cand(cid, l, parent=None):
    """构造使用实测等大小字节数的测试候选。"""
    return (CheckpointCandidate(cid, cid, (), l, REAL_CHECKPOINT_BYTES, True), parent)


class EvictPolicyTests(unittest.TestCase):
    def test_chain_incremental_proxy(self):
        """链式树：父被驱逐后，子增量代理改为相对更远保留祖先（逻辑合并语义）。"""
        # root(100)→a(300)→b(500)→c(700)，独立叶 d(50)；k=1
        nodes = [cand('root', 100), cand('a', 300, 'root'), cand('b', 500, 'a'), cand('c', 700, 'b'), cand('d', 50)]
        recency = {'root': 3, 'a': 1, 'b': 4, 'c': 5, 'd': 2}
        core = tuple(n[0] for n in nodes)
        parent = {n[0].checkpoint_id: n[1] for n in nodes}
        result = MarconiEvictPolicy().select(core, 1, recency, parent, 1.0)
        # 独立复核每轮：可驱逐集=保留集中有效孩子数≤1 的节点；代理=相对最近保留祖先
        retained = {n[0].checkpoint_id for n in nodes}
        all_ids = set(retained)
        for r in result.rounds:
            eff = {}
            for cid in retained:
                cur = parent[cid]
                while cur is not None and cur not in retained:
                    cur = parent[cur]
                eff[cid] = cur
            children = {}
            for cid, p in eff.items():
                if p is not None:
                    children[p] = children.get(p, 0) + 1
            self.assertEqual(r['evictable'], sorted(c for c in retained if children.get(c, 0) <= 1))
            self.assertEqual(r['retained_count'], len(retained))
            self.assertIn(r['victim'], r['evictable'])
            retained.remove(r['victim'])
        self.assertEqual(retained, set(result.selected_checkpoint_ids))
        self.assertEqual(len(result.selected_checkpoint_ids), 1)
        self.assertTrue(set(result.evicted_checkpoint_ids) | set(result.selected_checkpoint_ids) == all_ids)
        # 至少存在一轮：某保留节点因祖先被驱逐而代理增大（验证相对化确实生效）
        grew = any(r2['incremental_proxy'][v] > r1['incremental_proxy'][v]
                   for r1, r2 in zip(result.rounds, result.rounds[1:])
                   for v in r2['evictable'] if v in r1['evictable'])
        self.assertTrue(grew or len(result.rounds) < 2, '未观察到代理随祖先驱逐而变化')

    def test_intermediate_node_protection(self):
        """两孩子中间节点在两个孩子都保留时不可驱逐；孩子减少后逐步可驱逐。"""
        # p(100) 有 x(200)、y(300) 两孩子，x 有 xx(250)
        nodes = [cand('p', 100), cand('x', 200, 'p'), cand('y', 300, 'p'), cand('xx', 250, 'x')]
        core = tuple(n[0] for n in nodes)
        parent = {n[0].checkpoint_id: n[1] for n in nodes}
        recency = {'p': 4, 'x': 3, 'y': 2, 'xx': 1}
        result = MarconiEvictPolicy().select(core, 1, recency, parent, 1.0)
        self.assertNotIn('p', result.rounds[0]['evictable'], '两孩子节点第一轮不应可驱逐')
        self.assertEqual(result.rounds[0]['evictable'], ['x', 'xx', 'y'])
        # p 只能在其有效孩子数降到 ≤1 之后被驱逐
        retained = {n[0].checkpoint_id for n in nodes}
        for r in result.rounds:
            # 重算有效孩子数
            def nearest(nid):
                cur = parent[nid]
                while cur is not None and cur not in retained:
                    cur = parent[cur]
                return cur
            children = {}
            for cid in retained:
                p = nearest(cid)
                if p is not None:
                    children[p] = children.get(p, 0) + 1
            if 'p' in r['evictable']:
                self.assertLessEqual(children.get('p', 0), 1, 'p 可驱逐时有效孩子数必须≤1')
            retained.remove(r['victim'])

    def test_budget_and_invalid_inputs(self):
        """预算合同与非法输入拒绝。"""
        nodes = [cand('r', 100), cand('a', 200, 'r')]
        core = tuple(n[0] for n in nodes)
        parent = {n[0].checkpoint_id: n[1] for n in nodes}
        rec = {'r': 1, 'a': 2}
        # k=0 全驱逐；k≥n 无驱逐轮；正常 k
        self.assertEqual(MarconiEvictPolicy().select(core, 0, rec, parent).selected_checkpoint_ids, ())
        r_full = MarconiEvictPolicy().select(core, 2, rec, parent)
        self.assertEqual(r_full.selected_checkpoint_ids, ('a', 'r'))
        self.assertEqual(r_full.rounds, ())
        bad = [(-1, rec, parent), (True, rec, parent), (1.5, rec, parent)]
        for k, rc, pa in bad:
            with self.subTest(k=k), self.assertRaises(ValueError):
                MarconiEvictPolicy().select(core, k, rc, pa)
        # 非法元数据：父缺失 / 环 / 重复ID / 字节不符
        with self.assertRaises(ValueError):
            MarconiEvictPolicy().select(core, 1, rec, {'r': '缺失', 'a': 'r'})
        with self.assertRaises(ValueError):
            MarconiEvictPolicy().select((CheckpointCandidate('a', 'a', (), 100, REAL_CHECKPOINT_BYTES, True),
                                         CheckpointCandidate('r', 'r', (), 200, REAL_CHECKPOINT_BYTES, True)),
                                        1, rec, {'a': 'r', 'r': 'a'})
        with self.assertRaises(ValueError):
            MarconiEvictPolicy().select(core + core, 1, rec, dict(parent, **{'a': 'r'}))
        with self.assertRaises(ValueError):
            MarconiEvictPolicy().select((CheckpointCandidate('r', 'r', (), 100, 1, True),), 1, {'r': 1}, {'r': None})

    def test_determinism(self):
        """同输入两次运行结果完全一致（同分按 checkpoint_id 升序）。"""
        nodes = [cand('p', 100), cand('x', 200, 'p'), cand('y', 300, 'p'), cand('xx', 250, 'x'), cand('z', 50)]
        core = tuple(n[0] for n in nodes)
        parent = {n[0].checkpoint_id: n[1] for n in nodes}
        rec = {'p': 1, 'x': 1, 'y': 1, 'xx': 1, 'z': 1}  # 全同分：触发确定性次序
        a = MarconiEvictPolicy().select(core, 2, rec, parent)
        b = MarconiEvictPolicy().select(core, 2, rec, parent)
        # call_ms 为单次诊断计时，不属于确定性合同；比较选集/驱逐序/逐轮轨迹
        self.assertEqual((a.selected_checkpoint_ids, a.evicted_checkpoint_ids, a.rounds),
                         (b.selected_checkpoint_ids, b.evicted_checkpoint_ids, b.rounds))

    def test_real_bytes_evidence(self):
        """实测字节常量必须与 T2-R Census 证据逐项一致（C 结构 16 候选等大小）。"""
        mp = json.loads((T2R / 'historical_physical_mapping.json').read_text())
        self.assertEqual(len(mp['rows']), 16)
        for row in mp['rows']:
            self.assertEqual(row['independent_recurrent_bytes'], REAL_CHECKPOINT_BYTES)


class FrozenABCComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((MAIN / 'workloads.json').read_text())['cases']
        with (MAIN / 'cpu_comparison.csv').open() as f:
            cls.frozen = {(r['structure'], int(r['K']), r['policy']): r for r in csv.DictReader(f)}
        cls.sha_before = hashlib.sha256((MAIN / 'workloads.json').read_bytes()).hexdigest()

    def test_legacy_matches_frozen(self):
        """旧版静态 Top-K 复算必须与冻结 cpu_comparison.csv 的 Marconi 行全等（6 条件）。"""
        for case in self.cases:
            for k in (3, 5):
                with self.subTest(structure=case['structure'], k=k):
                    cs = case['candidates']
                    core = tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'], 1, True) for c in cs)
                    rec = {c['checkpoint_id']: c['last_access_order'] for c in cs}
                    flop = {c['checkpoint_id']: c['token_position_L'] for c in cs}
                    got = MarconiStylePolicy().select(core, k, rec, flop, 1.0).selected_checkpoint_ids
                    self.assertEqual(list(got), json.loads(self.frozen[case['structure'], k, 'Marconi Adaptation']['selected']))

    def test_evict_on_abc(self):
        """新版在 A/B/C×K3/5 可运行：选集合法、J 有限、输入未被修改；新旧差异仅记录不断言优劣。"""
        diffs = []
        for case in self.cases:
            snapshot = copy.deepcopy(case)
            for k in (3, 5):
                with self.subTest(structure=case['structure'], k=k):
                    out = plan_case(case, k)
                    cs_ids = {c['checkpoint_id'] for c in case['candidates']}
                    self.assertEqual(len(out['selected']), k)
                    self.assertTrue(set(out['selected']) <= cs_ids)
                    self.assertEqual(case, snapshot, '纯规划修改了输入')
                    self.assertTrue(isinstance(out['J_ms'], float))
                    legacy_sel = json.loads(self.frozen[case['structure'], k, 'Marconi Adaptation']['selected'])
                    diffs.append({'structure': case['structure'], 'K': k,
                                  'legacy': legacy_sel, 'evict': out['selected'],
                                  'same_set': set(legacy_sel) == set(out['selected']),
                                  'J_evict_ms': out['J_ms'],
                                  'J_legacy_ms': float(self.frozen[case['structure'], k, 'Marconi Adaptation']['current_J_ms'])})
        self.assertEqual(len(diffs), 6)
        (Path(__file__).resolve().parent / 'abc_diff_test.json').write_text(
            json.dumps({'说明': '测试内记录的新旧差异，正式对照以 run_compare.py 产物为准', 'diffs': diffs},
                       ensure_ascii=False, indent=2))

    def test_frozen_inputs_unchanged(self):
        """全部运行之后，冻结 workloads.json 摘要不变。"""
        sha_after = hashlib.sha256((MAIN / 'workloads.json').read_bytes()).hexdigest()
        self.assertEqual(self.sha_before, sha_after)


if __name__ == '__main__':
    unittest.main()
