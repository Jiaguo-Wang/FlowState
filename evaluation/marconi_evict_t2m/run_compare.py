"""T2-Mini 正式对照：A/B/C × K3/5 上 Marconi-Legacy（静态 Top-K）与 Marconi Adaptation (evict)。

只读冻结输入；输出当前调用 J 与未来静态 J（与冻结表同口径：epoch 的 future 域），
并交叉核验旧版复算与 cpu_comparison.csv 全等、冻结文件摘要前后不变。产物禁止覆盖。
"""
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter_ns

from evaluation.baseline_reuse_t2r.interface import validate_declared_case
from evaluation.kvflow_multiworkflow_cpu_main import run_cpu as legacy
from evaluation.marconi_evict_t2m.evict_policy import REAL_CHECKPOINT_BYTES, MarconiEvictPolicy
from evaluation.sota_policies import MarconiStylePolicy
from flowstate.state_catalog import CheckpointCandidate

OUT = Path(__file__).resolve().parent
MAIN = OUT.parent / 'kvflow_multiworkflow_cpu_main'


def main():
    cases = json.loads((MAIN / 'workloads.json').read_text())['cases']
    with (MAIN / 'cpu_comparison.csv').open() as f:
        frozen = {(r['structure'], int(r['K']), r['policy']): r for r in csv.DictReader(f)}
    sha_before = {p: hashlib.sha256((MAIN / p).read_bytes()).hexdigest() for p in ('workloads.json', 'cpu_comparison.csv')}
    rows, cross_fail = [], []

    for case in cases:
        snapshot = deepcopy(case)
        for k in (3, 5):
            validate_declared_case(case, k)
            cs = case['candidates']
            rec = {c['checkpoint_id']: c['last_access_order'] for c in cs}
            flop = {c['checkpoint_id']: c['token_position_L'] for c in cs}
            parent = {c['checkpoint_id']: c['parent_candidate'] for c in cs}
            # 旧版：静态 Top-K（逻辑单位字节；T2-R 已证等大小实字节选集不变）
            t0 = perf_counter_ns()
            leg = MarconiStylePolicy().select(
                tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'], 1, True) for c in cs),
                k, rec, flop, 1.0).selected_checkpoint_ids
            leg_ms = (perf_counter_ns() - t0) / 1e6
            # 新版：增量收益 + 动态驱逐（实测等大小字节）
            ev = MarconiEvictPolicy().select(
                tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'], REAL_CHECKPOINT_BYTES, True) for c in cs),
                k, rec, parent, 1.0)
            # 同口径计分：当前域与未来静态域（与冻结表同用 legacy.epoch/score）
            ep_cur, ep_fut = legacy.epoch(case, k), legacy.epoch(case, k, 'future')
            for name, sel, evicted, ms in (('Marconi-Legacy', list(leg), [], leg_ms),
                                            ('Marconi Adaptation (evict)', list(ev.selected_checkpoint_ids), list(ev.evicted_checkpoint_ids), ev.call_ms)):
                cur, fut = legacy.common.score(ep_cur, sel), legacy.common.score(ep_fut, sel)
                rows.append({'structure': case['structure'], 'K': k, 'policy': name, 'selected': sel,
                             'current_J_ms': cur['cost_ms'], 'current_E_G': cur['per_pending'],
                             'future_static_J_ms': fut['cost_ms'], 'evicted_in_order': evicted,
                             'n_eviction_rounds': len(evicted), 'call_ms': round(ms, 6),
                             'same_set_as_legacy': set(sel) == set(leg)})
            # 交叉核验：旧版当前/未来 J 必须与冻结表全等
            fz = frozen[case['structure'], k, 'Marconi Adaptation']
            if not (list(leg) == json.loads(fz['selected'])
                    and abs(rows[-2]['current_J_ms'] - float(fz['current_J_ms'])) < 1e-9
                    and abs(rows[-2]['future_static_J_ms'] - float(fz['future_static_J_ms'])) < 1e-9):
                cross_fail.append((case['structure'], k))
            if case != snapshot:
                raise ValueError('纯规划修改了输入')

    for p, h in sha_before.items():
        if hashlib.sha256((MAIN / p).read_bytes()).hexdigest() != h:
            raise ValueError('冻结文件被修改：' + p)
    with (OUT / 'evict_vs_legacy.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k2: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k2, v in r.items()})
    summary = {'说明': 'A/B/C×K3/5 新旧 Marconi 对照；J 为 modeled ms（声明合成输入，非实测延迟）；call_ms 为单次诊断非基准。',
               'legacy_matches_frozen': not cross_fail, 'cross_fail': cross_fail,
               'frozen_sha_unchanged': True, 'rows': len(rows),
               'conditions': [{'structure': r['structure'], 'K': r['K'], 'same_set': r['same_set_as_legacy']}
                              for r in rows if r['policy'] == 'Marconi Adaptation (evict)']}
    (OUT / 'compare_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('对照完成：%d 行；旧版与冻结全等=%s' % (len(rows), not cross_fail))
    for r in rows:
        print('%s K%d %-26s cur=%9.3f fut=%9.3f same=%s rounds=%d' % (
            r['structure'], r['K'], r['policy'], r['current_J_ms'], r['future_static_J_ms'], r['same_set_as_legacy'], r['n_eviction_rounds']))


if __name__ == '__main__':
    main()
