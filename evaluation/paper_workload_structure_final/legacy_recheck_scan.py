"""任务 1+2：历史 573 条件只读复核 + AgentX 快照固定绝对预算扫描（CPU-only）。

数据源（只读，哈希留档）：
- gate1_step1_allocation_equivalence/{gate1_step1_instance_statistics.csv, matrices.jsonl}

口径（与 T1 结构审计 / e3 冻结语义一致，全部为 modeled Phi 恢复成本，非 GPU 延迟）：
- 目标：J(S)=Σ_p Phi(T_p−E_p, T_p)，E_p=max(选中且兼容 p 的候选 position)；
  benefit_ms[p][c]=Phi(T_p,T_p)−Phi(T_p−pos_c,T_p) 为其线性化证书矩阵（max-E 与 max-benefit 等价）；
- K_min=逐 Pending 最深合法前沿唯一实现者并集大小（与 statistics.k_min_frontier 复核）；
- 证书：支持可分 ⇒ 逐 Pending 行最大值 TopK 恰为最优；否则 K≥K_min ⇒ 全域前沿可达最优。

四方法（同一 modeled 目标；Marginal Greedy 与 WFC 已在原 budget 下 69/69 复现冻结选集）：
- Exact：K<K_min 枚举 C(n,K)（AgentX K_min≤2，故仅 K=1，n≤149）；K≥K_min 用代数最优
  Σ Phi(T−E*)（并对 C(n,K)≤200,000 且 K≤4 的条件额外枚举交叉验证代数式）；
  并列终集取 (大小, ID 字典序) 最小；
- Marginal Greedy：每步选 Φ 边际收益最大（并列取索引序最小），零边际即停；
- WFC（P1 B3 直译）：按 (可推进 Pending 数 q, token 推进量 r, ID) 排序，q==0 停，前沿动态更新；
- StandaloneTopK（e3 直译）：按独立总 benefit 排序取前 K 个正收益候选（不感知共享）。

性能注记：Phi 用 Decimal 系数逐点求值昂贵，本脚本对每实例预计算「Pending×前沿值 → 成本」
查表（前沿值域=该 Pending 的合法 position 集合∪{0}），枚举与贪心评价全部走整数比较+查表；
组合计数用 math.comb（不迭代生成器）。

扫描设计：23 个 AgentX 快照（同快照跨 budget_ratio 矩阵逐字节一致已验）× 固定绝对预算
K∈{1..8} 共 184 格 × 4 方法；重点 K<K_min（本数据仅 K_min=2 的快照在 K=1）。
不筛选、不改任何冻结输入。
"""
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
GATE1 = ROOT / 'evaluation/gate1_step1_allocation_equivalence'
OUT = ROOT / 'evaluation/paper_workload_structure_final'
EPS = 1e-9
SCAN_K = list(range(1, 9))          # 固定绝对预算扫描集合
ENUM_CAP = 200_000                  # Exact 交叉验证枚举的组合数上限
PHI_A, PHI_B, PHI_C = Decimal('37.828150'), Decimal('0.345974143'), Decimal('0.156201917')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def phi(g, t):
    """冻结 Phi 成本模型（g/t 为 Token，内部按 1024 块换算；与 G2 控制路径同系数）。"""
    g, t = Decimal(g) / 1024, Decimal(t) / 1024
    return float(PHI_A * g + PHI_B * g * t - PHI_C * g * g)


def make_inst(x):
    """从 matrices.jsonl 行构建实例（含逐 Pending 前沿→成本查表与兼容 position 列表）。"""
    n, m = len(x['candidate_ids']), len(x['targets'])
    pos, comp, T = x['positions'], x['compatibility'], x['targets']
    legal = [[pos[i] for i in range(n) if comp[p][i]] for p in range(m)]
    cost_tab = [dict((e, phi(T[p] - e, T[p])) for e in sorted(set(legal[p]) | {0})) for p in range(m)]
    return {'ids': x['candidate_ids'], 'bm': x['benefit_ms'], 'T': T, 'pos': pos, 'comp': comp,
            'n': n, 'm': m, 'legal': legal, 'cost_tab': cost_tab,
            'j0': sum(phi(T[p], T[p]) for p in range(m))}


def cost_of(inst, sel):
    """Phi 域 modeled 成本（查表版）：sel 为候选索引集合。"""
    tot = 0.0
    for p in range(inst['m']):
        e = 0
        for i in sel:
            if inst['comp'][p][i] and inst['pos'][i] > e:
                e = inst['pos'][i]
        tot += inst['cost_tab'][p][e]
    return tot


def greedy_sel(inst, k):
    """Marginal Greedy：Φ 边际收益最大，并列取索引序最小，零边际停（查表版）。"""
    E = [0] * inst['m']
    sel = []
    while len(sel) < k:
        gains = []
        for c in range(inst['n']):
            if c in sel:
                continue
            d = 0.0
            for p in range(inst['m']):
                if inst['comp'][p][c] and inst['pos'][c] > E[p]:
                    d += inst['cost_tab'][p][E[p]] - inst['cost_tab'][p][inst['pos'][c]]
            gains.append((d, c))
        d, c = max(gains, key=lambda t: (t[0], -t[1]))
        if d <= EPS:
            break
        sel.append(c)
        for p in range(inst['m']):
            if inst['comp'][p][c]:
                E[p] = max(E[p], inst['pos'][c])
    return sel


def wfc_sel(inst, k):
    """WFC（P1 B3 直译）：(可推进 Pending 数, token 推进量, ID) 排序；q==0 停。"""
    E = [0] * inst['m']
    sel = []
    while len(sel) < k:
        ch = [(sum(1 for p in range(inst['m']) if inst['comp'][p][c] and inst['pos'][c] > E[p]),
               sum(max(0, inst['pos'][c] - E[p]) for p in range(inst['m']) if inst['comp'][p][c]), c)
              for c in range(inst['n']) if c not in sel]
        ch.sort(key=lambda t: (-t[0], -t[1], t[2]))
        if not ch or ch[0][0] == 0:
            break
        win = ch[0][2]
        sel.append(win)
        for p in range(inst['m']):
            if inst['comp'][p][win]:
                E[p] = max(E[p], inst['pos'][win])
    return sel


def sa_sel(inst, k):
    """StandaloneTopK（e3 直译）：按独立总 benefit 排序取前 K 个正收益候选。"""
    bm = inst['bm']
    tot = [sum(bm[p][c] for p in range(inst['m'])) for c in range(inst['n'])]
    return [c for c in sorted(range(inst['n']), key=lambda c: (-tot[c], c)) if tot[c] > EPS][:k]


def exact_best(inst, k, kmin):
    """Exact：K<K_min 枚举；K≥K_min 代数最优（小组合数时枚举交叉验证）。

    返回 (最优 J, 终集, 方式)。并列终集与 e3 同规则：最优 J 下取 (大小, ID 字典序) 最小。
    """
    n, m = inst['n'], inst['m']
    estar = [max(inst['legal'][p] + [0]) for p in range(m)]
    j_alg = sum(inst['cost_tab'][p][estar[p]] for p in range(m))
    witness = sorted({i for p in range(m) for i in range(n)
                      if inst['comp'][p][i] and inst['pos'][i] == estar[p] and estar[p] > 0})
    n_comb = math.comb(n, k) if k <= n else 0
    if k >= kmin:
        if n_comb <= ENUM_CAP and k <= 4:
            best = min(cost_of(inst, S) for S in itertools.combinations(range(n), k))
            assert abs(best - j_alg) < 1e-9, '代数最优与枚举不一致'
        return j_alg, witness, 'algebraic(k>=Kmin)'
    scored = [(cost_of(inst, S), S) for S in itertools.combinations(range(n), k)]
    bj = min(v for v, _ in scored)
    S = min((S for v, S in scored if abs(v - bj) <= EPS), key=lambda S: (len(S), S))
    return bj, list(S), 'enumerate(k<Kmin)'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {'stats': GATE1 / 'gate1_step1_instance_statistics.csv',
              'matrices': GATE1 / 'matrices.jsonl'}
    (OUT / 'source_hashes.json').write_text(json.dumps(
        {k: {'path': str(v), 'sha256': sha(v)} for k, v in inputs.items()}, ensure_ascii=False, indent=2))
    stats = {r['instance_id']: r for r in csv.DictReader(inputs['stats'].open())}
    mats = {}
    for line in inputs['matrices'].read_text().splitlines():
        x = json.loads(line)
        mats[x['instance_id']] = x
    checks, fails = [], []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})
        if not ok:
            fails.append(name)

    # ══ 任务 1：573 条历史条件只读复核 ══
    agg = defaultdict(lambda: defaultdict(int))
    policy_cost_ok = defaultdict(int)
    for iid, r in stats.items():
        x = mats[iid]
        inst = make_inst(x)
        ids, bm = inst['ids'], inst['bm']
        n, m, k = inst['n'], inst['m'], int(r['budget'])
        # 结构指标重算
        supports = [{p for p in range(m) if bm[p][c] > 0} for c in range(n)]
        shared = sum(len(s) > 1 for s in supports)
        overlap = sum(bool(a & b) for a, b in itertools.combinations(supports, 2))
        separable = all(len(s) <= 1 for s in supports)
        # K_min 重算（逐 Pending 最深合法前沿唯一实现者并集）
        forced = set()
        for p in range(m):
            e = max(inst['legal'][p] + [0])
            if e > 0:
                forced.update(i for i in range(n) if inst['comp'][p][i] and inst['pos'][i] == e)
        # 证书公式重算（benefit 域线性化）
        row_max = [max([bm[p][c] for c in range(n)] + [0.]) for p in range(m)]
        if separable:
            opt_ben = sum(sorted(row_max, reverse=True)[:k])
            cert = 'SEPARABLE_SUPPORT_TOPK'
        else:
            opt_ben = sum(row_max)
            cert = 'ALL_CANDIDATE_FRONTIERS_ATTAINED'
        ok_struct = (shared == int(r['num_shared_checkpoints'])
                     and overlap == int(r['num_overlapping_candidate_pairs'])
                     and len(forced) == int(r['k_min_frontier']))
        # 四策略 cost/benefit 复算（Phi 域 cost 与 benefit 域线性化互证）
        idx = {c: i for i, c in enumerate(ids)}
        pol_ok = True
        fs_ben = None
        for pol, sf, cf in (('flowstate', 'flowstate_selected_ids', 'flowstate_cost_ms'),
                            ('pcb', 'pcb_selected_ids', 'pcb_cost_ms'),
                            ('ldr', 'ldr_selected_ids', 'ldr_cost_ms'),
                            ('wfc', 'wfc_selected_ids', 'wfc_cost_ms')):
            sel = [idx[c] for c in json.loads(r[sf])]
            jc = cost_of(inst, sel)
            ben = sum(max([bm[p][c] for c in sel] + [0.]) for p in range(m))
            pol_ok &= abs(jc - float(r[cf])) < 1e-6 and abs((inst['j0'] - ben) - jc) < 1e-6
            policy_cost_ok[pol] += abs(jc - float(r[cf])) < 1e-6
            if pol == 'flowstate':
                fs_ben = ben
        greedy_opt = abs(fs_ben - opt_ben) < 1e-6
        check('recheck_' + iid, ok_struct and pol_ok, 'struct=%s pol=%s' % (ok_struct, pol_ok))
        ds = r['workload']
        agg[ds]['conditions'] += 1
        agg[ds]['shared'] += shared > 0
        agg[ds]['overlap'] += overlap > 0
        agg[ds]['K_lt_Kmin'] += k < len(forced)
        agg[ds]['separable'] += separable
        agg[ds]['greedy_optimal'] += greedy_opt
        agg[ds]['cert_' + cert] += 1
    check('agg_OpenHands', dict(agg['OpenHands']) == {
        'conditions': 504, 'shared': 0, 'overlap': 504, 'K_lt_Kmin': 90, 'separable': 504,
        'greedy_optimal': 504, 'cert_SEPARABLE_SUPPORT_TOPK': 504}, dict(agg['OpenHands']))
    check('agg_AgentX', dict(agg['AgentX']) == {
        'conditions': 69, 'shared': 66, 'overlap': 33, 'K_lt_Kmin': 0, 'separable': 3,
        'greedy_optimal': 69, 'cert_ALL_CANDIDATE_FRONTIERS_ATTAINED': 66,
        'cert_SEPARABLE_SUPPORT_TOPK': 3}, dict(agg['AgentX']))
    check('policy_cost_recheck_573x4', all(v == 573 for v in policy_cost_ok.values()),
          dict(policy_cost_ok))

    # ══ 任务 2：AgentX 固定绝对预算扫描 ══
    ax = {iid: r for iid, r in stats.items() if r['workload'] == 'AgentX'}
    snap = defaultdict(list)
    for iid in ax:
        snap[(ax[iid]['source_file'], ax[iid]['snapshot_id'])].append(iid)
    mk = lambda x: (tuple(x['candidate_ids']), tuple(map(tuple, x['benefit_ms'])),
                    tuple(map(tuple, x['compatibility'])), tuple(x['positions']), tuple(x['targets']))
    check('snapshot_matrix_identical', all(len({mk(mats[i]) for i in v}) == 1 for v in snap.values()))
    scan_rows, calib = [], {'greedy': 0, 'wfc': 0, 'n': len(ax)}
    exact_sets = defaultdict(dict)  # 快照 -> {K: 终集}（嵌套性分析）
    for (src, sid), iids in sorted(snap.items()):
        inst = make_inst(mats[iids[0]])
        ids = inst['ids']
        kmin = int(ax[iids[0]]['k_min_frontier'])
        snap_key = '%s::%s' % (sid, iids[0].split('::')[0])
        # 校准：原 budget 行 Greedy/WFC 复现冻结选集
        for iid in iids:
            r = ax[iid]
            K0 = int(r['budget'])
            g = sorted(ids[i] for i in greedy_sel(inst, K0))
            w = sorted(ids[i] for i in wfc_sel(inst, K0))
            calib['greedy'] += g == sorted(json.loads(r['flowstate_selected_ids']))
            calib['wfc'] += w == sorted(json.loads(r['wfc_selected_ids']))
        for K in SCAN_K:
            je, Se, how = exact_best(inst, K, kmin)
            exact_sets[snap_key][K] = sorted(ids[i] for i in Se)
            for pol, sel in (('Exact', Se), ('Greedy', greedy_sel(inst, K)),
                             ('WFC', wfc_sel(inst, K)), ('StandaloneTopK', sa_sel(inst, K))):
                j = cost_of(inst, sel)
                scan_rows.append({'snapshot': snap_key, 'K': K, 'K_min': kmin,
                                  'K_lt_Kmin': int(K < kmin), 'policy': pol,
                                  'J_ms': round(j, 9), 'gap_vs_exact': round(j - je, 9),
                                  'selected': '|'.join(sorted(ids[i] for i in sel)),
                                  'same_set_as_exact': sorted(ids[i] for i in sel) == sorted(ids[i] for i in Se),
                                  'n_candidate': inst['n'], 'n_pending': inst['m'],
                                  'exact_method': how if pol == 'Exact' else ''})
    check('calibration_greedy_69', calib['greedy'] == calib['n'], calib)
    check('calibration_wfc_69', calib['wfc'] == calib['n'], calib)
    nest_viol = []
    for sk, d in exact_sets.items():
        for K in SCAN_K[:-1]:
            if not set(d[K]) <= set(d[K + 1]):
                nest_viol.append({'snapshot': sk, 'from_K': K, 'set_k': d[K], 'set_k1': d[K + 1]})
    with (OUT / 'agentx_budget_scan.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        w.writeheader()
        w.writerows(scan_rows)
    (OUT / 'legacy573_recheck.json').write_text(json.dumps(
        {'checks': checks, 'fails': fails, 'aggregate': {k: dict(v) for k, v in agg.items()},
         'calibration': calib, 'snapshot_count': len(snap),
         'exact_nesting_violations': nest_viol}, ensure_ascii=False, indent=2))
    print('复核 %d 检查（失败 %s）；扫描 %d 行（%d 快照×%d K×4）' % (
        len(checks), fails or '无', len(scan_rows), len(snap), len(SCAN_K)))
    print('聚合：', json.dumps({k: dict(v) for k, v in agg.items()}, ensure_ascii=False))
    print('校准：', calib, '| Exact 非嵌套违反数：', len(nest_viol))


if __name__ == '__main__':
    main()
