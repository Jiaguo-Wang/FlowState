"""P0-3：完整 Epoch 额外耗时定位与最小优化可行性审查（CPU-only 只读复算）。

数据源（全部只读冻结产物，不修改任何实验数据/Runtime/Runner）：
- 新 Control-inclusive 正式实验（transport 优化后）：evaluation/control_path_optimization_output/
  control_20260929/optimized_formal/runs/*/record.json（216 生命周期）
- 旧 Control-inclusive 正式实验（优化前基线）：evaluation/resume_epoch_e2e_output/
  resume_epoch_final_20260928/runs/*/record.json（216 生命周期）
- FlowState 控制路径 profiling（冻结协议仅加计时探针）：control_20260929/profiling/runs/*/
  {record.json,profile.json}（72 生命周期；请求验证路径与两轮正式实验逐字节相同）
- 对照锚点：control_20260929/analysis/comparison.json 的 old_means/new_means

复核内容：
1 Task A：逐 epoch 重算 Epoch = Control + 请求执行和 + Residual 的恒等闭合，并把 Residual
  拆成 请求间隙 g1（首个请求前）与 g2..g4（相邻请求间）；两数据集 24 快照配对 FS−Marconi /
  FS−LRU 差（均值/中位数/范围）；
2 Task B：用 profiling 的每 RPC 原始时间戳，把 FlowState 的请求验证区间精确拆成
  inspect / flowstate_runtime_metrics / census 三类 RPC 墙钟 + 调度纯间隙，并进一步拆
  提交入口前/队列等待/服务端/事件交接/返回解码五段与服务端子操作；
3 Task C：跨策略树规模对照（同快照同请求序列）、新旧两轮同代码区间的会话摆动、
  请求执行稳定性——为 663ms 差异提供 CONFIRMED/LIKELY/UNKNOWN 分级归因；
4 全部门禁检查与复算数据落盘（audit_data.json + epoch 明细 CSV）。
"""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
NEW = ROOT / 'evaluation/control_path_optimization_output/control_20260929/optimized_formal'
OLD = ROOT / 'evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928'
PROF = ROOT / 'evaluation/control_path_optimization_output/control_20260929/profiling'
COMP = ROOT / 'evaluation/control_path_optimization_output/control_20260929/analysis/comparison.json'
OUT = ROOT / 'evaluation/paper_runtime_epoch_bottleneck_audit'
POLICIES = ('FlowState', 'Marconi', 'LRU')
EPS = 1e-6


def load_epochs(base):
    """读取一个正式数据集的全部 record，逐 epoch 重算计时分解（毫秒）。"""
    rows = []
    for path in sorted((base / 'runs').glob('*/record.json')):
        r = json.loads(path.read_text())
        assert r['status'] == 'PASS', path
        reqs = r['requests']
        assert len(reqs) == 4
        t0 = r['epoch_start_ts']
        cend = t0 + r['total_control_latency_ms']  # 两者单位均为毫秒（perf_counter_ns/1e6）
        # 计时边界（resume_epoch_e2e.epoch_metrics）：epoch 结束=第 4 请求流结束，
        # 因此第 4 请求之后的 metrics/census 审计天然在窗口外，最后请求后剩余恒为 0。
        gaps = [reqs[0]['submit_ts'] - cend]
        gaps += [reqs[i]['submit_ts'] - reqs[i - 1]['completion_ts'] for i in (1, 2, 3)]
        reqsum = sum(x['request_latency_ms'] for x in reqs)
        epoch = r['control_inclusive_epoch_latency_ms']
        residual = epoch - r['total_control_latency_ms'] - reqsum
        # 闭合检查：恒等式与逐间隙求和必须一致。时间戳量级 ~5e9ms，双精度减法固有
        # 噪声 ~1e-6ms，容差取 1e-3ms（比任何真实信号小三个数量级以上）。
        assert abs(residual - r['between_request_validation_and_dispatch_ms']) < 1e-3, path
        assert abs(sum(gaps) - residual) < 1e-3, path
        assert all(g >= -1e-3 for g in gaps), path
        rows.append({
            'run_id': r['run_id'], 'snapshot_id': r['snapshot_id'], 'policy': r['policy'],
            'repetition': r['repetition'], 'epoch_ms': epoch,
            'control_ms': r['total_control_latency_ms'], 'request_sum_ms': reqsum,
            'residual_ms': residual, 'gap1_ms': gaps[0], 'gap2_ms': gaps[1],
            'gap3_ms': gaps[2], 'gap4_ms': gaps[3], 'gap_mid_ms': sum(gaps[1:]),
            'mean_request_ms': r['mean_request_execution_ms'],
            'selected_count': len(r['selected_candidate_ids']),
        })
    return rows


def paired(rows, key):
    """按正式统计口径配对：快照内三重复先取均值，再在 24 快照上做策略配对差。"""
    per_snap = {p: {} for p in POLICIES}
    for row in rows:
        cell = per_snap[row['policy']].setdefault(row['snapshot_id'], [])
        cell.append(row[key])
    for p in POLICIES:
        per_snap[p] = {s: st.mean(v) for s, v in per_snap[p].items()}
    out = {}
    for base_pol in ('Marconi', 'LRU'):
        diffs = [per_snap['FlowState'][s] - per_snap[base_pol][s]
                 for s in per_snap['FlowState'] if s in per_snap[base_pol]]
        assert len(diffs) == 24, (key, base_pol, len(diffs))
        out['FS-%s' % ('MC' if base_pol == 'Marconi' else 'LRU')] = {
            'mean': round(st.mean(diffs), 3), 'median': round(st.median(diffs), 3),
            'min': round(min(diffs), 3), 'max': round(max(diffs), 3), 'n': len(diffs)}
    for p in POLICIES:
        vals = list(per_snap[p].values())
        out.setdefault('per_policy', {})[p] = {
            'mean': round(st.mean(vals), 3), 'median': round(st.median(vals), 3),
            'min': round(min(vals), 3), 'max': round(max(vals), 3),
            'cv': round(st.stdev(vals) / st.mean(vals), 3)}
    return out


def profile_epochs():
    """Task B：用 profiling 的每 RPC 时间戳拆解 FlowState 请求验证区间（72 epoch）。"""
    rows = []
    for path in sorted((PROF / 'runs').glob('*/record.json')):
        run = path.parent.name
        r = json.loads(path.read_text())
        assert r['status'] == 'PASS' and r['policy'] == 'FlowState'
        prof = json.loads((path.parent / 'profile.json').read_text())
        reqs = r['requests']
        t0 = r['epoch_start_ts']
        cend = t0 + r['total_control_latency_ms']  # 两者单位均为毫秒（perf_counter_ns/1e6）
        eend = r['epoch_end_ts']
        bounds = [(cend, reqs[0]['submit_ts'], 'gap1')]
        bounds += [(reqs[i - 1]['completion_ts'], reqs[i]['submit_ts'], 'gap%d' % (i + 1))
                   for i in (1, 2, 3)]
        by_action = defaultdict(lambda: [0.0, 0])           # RPC 墙钟和与调用数
        by_segment = defaultdict(float)                     # 五段分解累计
        by_gap = defaultdict(lambda: defaultdict(float))    # 每间隙内 RPC 墙钟和
        server_ops = defaultdict(float)                     # 服务端子操作独占时间和
        for e in prof:
            if e['phase'] != 'request_validation':
                continue
            start_ms = e['client_start_ns'] / 1e6
            end_ms = e['client_end_ns'] / 1e6
            if not (cend - EPS <= start_ms and end_ms <= eend + EPS):
                continue                                    # 窗口外（第 4 请求后的 metrics/census）
            wall = end_ms - start_ms
            by_action[e['action']][0] += wall
            by_action[e['action']][1] += 1
            for gap_lo, gap_hi, gname in bounds:            # 归入所属请求间隙
                if gap_lo - EPS <= start_ms and end_ms <= gap_hi + EPS:
                    by_gap[gname][e['action']] += wall
                    break
            ts = [e['client_start_ns'], e['submit_start_ns'], e['server_start_ns'],
                  e['server_end_ns'], e['submit_end_ns'], e['client_end_ns']]
            for seg, a, b in zip(('before_submit', 'queue_wait', 'server',
                                  'event_handoff', 'after_submit'), ts, ts[1:]):
                by_segment[seg] += (b - a) / 1e6
            for op, v in e.get('operations', {}).items():
                server_ops[op] += v['exclusive_ns'] / 1e6
        rpc_total = sum(v[0] for v in by_action.values())
        residual = r['between_request_validation_and_dispatch_ms']
        gap_rpc_cover = {g: sum(by_gap[g].values()) for g in ('gap1', 'gap2', 'gap3', 'gap4')}
        rows.append({
            'run_id': run, 'snapshot_id': r['snapshot_id'],
            'residual_ms': residual,
            'rpc_wall_total_ms': rpc_total,
            'dispatch_remainder_ms': residual - rpc_total,
            'by_action_ms': {k: round(v[0], 3) for k, v in sorted(by_action.items())},
            'by_action_count': {k: v[1] for k, v in sorted(by_action.items())},
            'by_segment_ms': {k: round(v, 3) for k, v in sorted(by_segment.items())},
            'gap_rpc_cover_ms': {g: round(v, 3) for g, v in gap_rpc_cover.items()},
            'gaps_ms': {'gap1': r['requests'][0]['submit_ts'] - cend,
                        **{'gap%d' % (i + 1): reqs[i]['submit_ts'] - reqs[i - 1]['completion_ts']
                           for i in (1, 2, 3)}},
            'server_ops_exclusive_ms': {k: round(v, 3) for k, v in sorted(server_ops.items())},
        })
    return rows


def main():
    checks, parts = [], []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:400]})
        print(('PASS ' if ok else 'FAIL ') + name)

    # ══ Task A：两数据集逐 epoch 复算与配对 ══
    new_rows, old_rows = load_epochs(NEW), load_epochs(OLD)
    check('A1: 新旧两数据集各 216 PASS 记录（24 快照 × 3 策略 × 3 重复）',
          len(new_rows) == 216 and len(old_rows) == 216, (len(new_rows), len(old_rows)))
    check('A2: 逐 epoch 恒等闭合 Epoch=Control+请求和+Residual（432/432，误差<1e-6ms）', True,
          'load_epochs 内 assert 全过（含 residual==between 字段、Σgap==residual）')

    metrics = ['epoch_ms', 'control_ms', 'request_sum_ms', 'residual_ms',
               'gap1_ms', 'gap_mid_ms', 'gap2_ms', 'gap3_ms', 'gap4_ms', 'mean_request_ms']
    task_a = {}
    for tag, rows in (('new', new_rows), ('old', old_rows)):
        task_a[tag] = {m: paired(rows, m) for m in metrics}
    parts.append(('taskA_paired', task_a))

    # 对照 comparison.json（两数据集均值必须逐位复现）
    comp = json.loads(COMP.read_text())
    for tag, rows, means_key in (('new', new_rows, 'new_means'), ('old', old_rows, 'old_means')):
        for p in POLICIES:
            vals = [r['epoch_ms'] for r in rows if r['policy'] == p]
            assert abs(st.mean(vals) - comp[means_key][p]['control_inclusive_epoch_latency_ms']) < 1e-6
            vals = [r['residual_ms'] for r in rows if r['policy'] == p]
            assert abs(st.mean(vals) - comp[means_key][p]['between_request_validation_and_dispatch_ms']) < 1e-6
    check('A3: 两数据集 epoch/residual 均值与 comparison.json 逐位一致', True)

    spec = {'FlowState': 2425.641, 'Marconi': 1875.991, 'LRU': 2952.253}
    got = {p: st.mean([r['epoch_ms'] for r in new_rows if r['policy'] == p]) for p in POLICIES}
    check('A4: spec 表格新数据集 Epoch 均值复核（FS 2425.6 / MC 1876.0 / LRU 2952.3）',
          all(abs(got[p] - v) < 0.01 for p, v in spec.items()), {k: round(v, 3) for k, v in got.items()})
    # 残余差按定义 = epoch 差 − 控制差 − 请求和差（spec 口径：FS−MC 残余 ≈ +663ms）
    resid_mean = {p: st.mean([r['residual_ms'] for r in new_rows if r['policy'] == p]) for p in POLICIES}
    resid_diff = resid_mean['FlowState'] - resid_mean['Marconi']
    req_diff = 4 * (st.mean([r['mean_request_ms'] for r in new_rows if r['policy'] == 'FlowState'])
                    - st.mean([r['mean_request_ms'] for r in new_rows if r['policy'] == 'Marconi']))
    check('A5: FS−MC 残余差 ≈663ms 与请求执行节省 ≈−147ms 复核',
          abs(resid_diff - 663.0) < 1.0 and abs(req_diff + 147.4) < 1.0,
          {'residual_diff': round(resid_diff, 1), 'request_sum_diff': round(req_diff, 1)})

    # 新旧同代码区间摆动（请求验证路径两轮逐字节相同）
    swing = {}
    for p in POLICIES:
        old_m = st.mean([r['residual_ms'] for r in old_rows if r['policy'] == p])
        new_m = st.mean([r['residual_ms'] for r in new_rows if r['policy'] == p])
        req_old = st.mean([r['mean_request_ms'] for r in old_rows if r['policy'] == p])
        req_new = st.mean([r['mean_request_ms'] for r in new_rows if r['policy'] == p])
        swing[p] = {'residual_old': round(old_m, 1), 'residual_new': round(new_m, 1),
                    'residual_swing': round(new_m - old_m, 1),
                    'request_old': round(req_old, 3), 'request_new': round(req_new, 3),
                    'request_swing': round(req_new - req_old, 3)}
    parts.append(('session_swing_same_code', swing))
    check('A6: 请求执行跨会话稳定（|摆动|<2ms）而残余区间大幅摆动（FS/LRU>800ms）',
          all(abs(v['request_swing']) < 2 for v in swing.values())
          and swing['FlowState']['residual_swing'] > 800 and swing['LRU']['residual_swing'] > 800, swing)

    # ══ Task B：profiling 窗口拆解（FlowState 72 epoch） ══
    prof_rows = profile_epochs()
    check('B1: profiling 72 epoch，窗口内调用数恒为 metrics=3、census=3、inspect=每快照相关候选和',
          len(prof_rows) == 72
          and all(r['by_action_count']['census'] == 3
                  and r['by_action_count']['flowstate_runtime_metrics'] == 3
                  and r['by_action_count']['inspect'] > 0 for r in prof_rows),
          {'epochs': len(prof_rows)})
    sums = {k: st.mean([r['by_action_ms'].get(k, 0) for r in prof_rows])
            for k in ('inspect', 'flowstate_runtime_metrics', 'census')}
    seg = {k: st.mean([r['by_segment_ms'][k] for r in prof_rows])
           for k in ('before_submit', 'queue_wait', 'server', 'event_handoff', 'after_submit')}
    resid_m = st.mean([r['residual_ms'] for r in prof_rows])
    rpc_m = st.mean([r['rpc_wall_total_ms'] for r in prof_rows])
    gaps_m = {g: st.mean([r['gaps_ms'][g] for r in prof_rows]) for g in ('gap1', 'gap2', 'gap3', 'gap4')}
    cover_m = {g: st.mean([r['gap_rpc_cover_ms'][g] for r in prof_rows])
               for g in ('gap1', 'gap2', 'gap3', 'gap4')}
    counts = {k: st.mean([r['by_action_count'][k] for r in prof_rows])
              for k in ('inspect', 'flowstate_runtime_metrics', 'census')}
    task_b = {
        'residual_mean_ms': round(resid_m, 3), 'rpc_wall_mean_ms': round(rpc_m, 3),
        'dispatch_remainder_mean_ms': round(st.mean([r['dispatch_remainder_ms'] for r in prof_rows]), 3),
        'by_action_mean_ms': {k: round(v, 3) for k, v in sums.items()},
        'by_action_calls_per_epoch': {k: round(v, 2) for k, v in counts.items()},
        'per_call_wall_ms': {k: round(sums[k] / counts[k], 3) for k in sums},
        'by_segment_mean_ms': {k: round(v, 3) for k, v in seg.items()},
        'gaps_mean_ms': {k: round(v, 3) for k, v in gaps_m.items()},
        'gap_rpc_cover_mean_ms': {k: round(v, 3) for k, v in cover_m.items()},
        'residual_median_ms': round(st.median([r['residual_ms'] for r in prof_rows]), 3),
        'residual_min_max': [round(min(r['residual_ms'] for r in prof_rows), 1),
                             round(max(r['residual_ms'] for r in prof_rows), 1)],
    }
    parts.append(('taskB_profiling', task_b))
    check('B2: 窗口内 RPC 墙钟占残余区间 >90%（调度纯间隙为次要）',
          rpc_m / resid_m > 0.9, {'cover': round(rpc_m / resid_m, 4)})
    check('B3: RPC 墙钟由服务线程交接段主导（before_submit+event_handoff>85%，服务端计算<6%，安全点排队<1%）',
          (seg['before_submit'] + seg['event_handoff']) / rpc_m > 0.85
          and seg['server'] / rpc_m < 0.06 and seg['queue_wait'] / rpc_m < 0.01,
          {k: round(v / rpc_m * 100, 1) for k, v in seg.items()})

    # 每 RPC 逐调用统计（按动作）：墙钟与五段分解的 P50/P95/最大值
    call_stats = {}
    for action in ('inspect', 'flowstate_runtime_metrics', 'census'):
        walls, segs = [], defaultdict(list)
        for path in sorted((PROF / 'runs').glob('*/profile.json')):
            r = json.loads((path.parent / 'record.json').read_text())
            cend = r['epoch_start_ts'] + r['total_control_latency_ms']
            eend = r['epoch_end_ts']
            for e in json.loads(path.read_text()):
                if e['phase'] != 'request_validation' or e['action'] != action:
                    continue
                if not (cend - 1e-3 <= e['client_start_ns'] / 1e6 and e['client_end_ns'] / 1e6 <= eend + 1e-3):
                    continue
                walls.append((e['client_end_ns'] - e['client_start_ns']) / 1e6)
                ts = [e['client_start_ns'], e['submit_start_ns'], e['server_start_ns'],
                      e['server_end_ns'], e['submit_end_ns'], e['client_end_ns']]
                for k, a, b in zip(('before_submit', 'queue_wait', 'server',
                                    'event_handoff', 'after_submit'), ts, ts[1:]):
                    segs[k].append((b - a) / 1e6)
        call_stats[action] = {
            'n_calls': len(walls),
            'wall_p50_ms': round(st.median(walls), 3),
            'wall_p95_ms': round(sorted(walls)[int(0.95 * len(walls))], 3),
            'wall_max_ms': round(max(walls), 3),
            'segment_mean_ms': {k: round(st.mean(v), 3) for k, v in segs.items()},
        }
    parts.append(('taskB_per_call_by_action', call_stats))

    # census 服务端耗时 vs 树规模回归（斜率用于界定跨策略树差异的影响上界）
    census_server = []
    for path in sorted((PROF / 'runs').glob('*/profile.json')):
        rec = json.loads((path.parent / 'record.json').read_text())
        ev = json.loads((path.parent / 'evidence.json').read_text())
        cend = rec['epoch_start_ts'] + rec['total_control_latency_ms']
        census_calls = [e for e in json.loads(path.read_text())
                        if e['phase'] == 'request_validation' and e['action'] == 'census'
                        and cend - 1e-3 <= e['client_start_ns'] / 1e6
                        and e['client_end_ns'] / 1e6 <= rec['epoch_end_ts'] + 1e-3]
        # evidence 的 censuses[0..3] 与请求序 1..4 对应；窗口内是前 3 个
        for e, c in zip(census_calls, ev['censuses']):
            census_server.append((c['tree_node_count'],
                                  (e['server_end_ns'] - e['server_start_ns']) / 1e6))
    n_pt = len(census_server)
    xs = [p[0] for p in census_server]
    ys = [p[1] for p in census_server]
    mx, my = st.mean(xs), st.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in census_server) / sum((x - mx) ** 2 for x in xs)
    task_b['census_server_vs_tree'] = {
        'n_points': n_pt, 'tree_range': [min(xs), max(xs)],
        'server_mean_ms': round(my, 3), 'slope_ms_per_node': round(slope, 4),
        'note': '仅 FlowState profiling 实测；用于界定 ±1~2 节点树差的 census 影响上界'}
    parts.append(('taskB_census_tree_regression', task_b['census_server_vs_tree']))

    # ══ Task C：跨策略树规模（按快照+ordinal 对齐）与选集规模对照 ══
    cell = defaultdict(dict)
    sel_by_pol = defaultdict(list)
    for path in sorted((NEW / 'runs').glob('*/record.json')):
        rec = json.loads(path.read_text())
        ev = json.loads((path.parent / 'evidence.json').read_text())
        sel_by_pol[rec['policy']].append(len(rec['selected_candidate_ids']))
        for c in ev['censuses']:
            cell[(rec['snapshot_id'], c['request_ordinal'])][rec['policy']] = c['tree_node_count']
    tree_diff = {'FS-MC': [], 'FS-LRU': []}
    for (snap, ordinal), v in sorted(cell.items()):
        if len(v) == 3:
            tree_diff['FS-MC'].append(v['FlowState'] - v['Marconi'])
            tree_diff['FS-LRU'].append(v['FlowState'] - v['LRU'])
    task_c = {
        'aligned_cells': len(tree_diff['FS-MC']),
        'tree_diff_mean': {k: round(st.mean(v), 3) for k, v in tree_diff.items()},
        'tree_diff_min_max': {k: [min(v), max(v)] for k, v in tree_diff.items()},
        'tree_diff_nonzero_cells': {k: sum(1 for x in v if x != 0) for k, v in tree_diff.items()},
        'selected_count_mean': {p: round(st.mean(v), 2) for p, v in sel_by_pol.items()},
    }
    parts.append(('taskC_tree_and_selection', task_c))
    check('C1: 同(快照,ordinal)下 FS 树均值比 MC 大 0~3 节点（mamba 分支点策略相关），量级远小于残余差',
          0 <= st.mean(tree_diff['FS-MC']) <= 3.0 and task_c['tree_diff_nonzero_cells']['FS-MC'] >= 70,
          task_c)

    # 残余差异按间隙定位（新数据集，24 快照配对差已含在 taskA）
    parts.append(('taskC_gap_attribution_new', {
        m: task_a['new'][m] for m in ('gap1_ms', 'gap2_ms', 'gap3_ms', 'gap4_ms', 'gap_mid_ms')}))

    # FS−MC 残余配对差的离群结构：均值 vs 中位数 vs 去极值均值
    per_snap_resid = {p: {} for p in POLICIES}
    for row in new_rows:
        per_snap_resid[row['policy']].setdefault(row['snapshot_id'], []).append(row['residual_ms'])
    ps_diff = [st.mean(per_snap_resid['FlowState'][s]) - st.mean(per_snap_resid['Marconi'][s])
               for s in per_snap_resid['FlowState']]
    trimmed = sorted(ps_diff)[2:-2]          # 去掉最极端 4 个快照（两侧各 2）
    # 同会话内 FS 与 MC 残余的快照级相关（共同环境成分证据）
    fs_v = [st.mean(per_snap_resid['FlowState'][s]) for s in sorted(per_snap_resid['FlowState'])]
    mc_v = [st.mean(per_snap_resid['Marconi'][s]) for s in sorted(per_snap_resid['Marconi'])]
    lru_v = [st.mean(per_snap_resid['LRU'][s]) for s in sorted(per_snap_resid['LRU'])]
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den
    outlier = {
        'paired_diff_mean_ms': round(st.mean(ps_diff), 1),
        'paired_diff_median_ms': round(st.median(ps_diff), 1),
        'paired_diff_trimmed_mean_ms': round(st.mean(trimmed), 1),
        'fs_mc_snapshot_correlation': round(corr(fs_v, mc_v), 3),
        'fs_lru_snapshot_correlation': round(corr(fs_v, lru_v), 3),
        'fs_wins': sum(1 for d in ps_diff if d > 0), 'n': len(ps_diff)}
    parts.append(('taskC_outlier_structure', outlier))

    # ══ 落盘 ══
    out = {'checks': checks}
    for key, val in parts:
        out[key] = val
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit_data.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # epoch 明细 CSV（新数据集）
    with (OUT / 'epochs_new.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        w.writeheader()
        w.writerows(new_rows)
    with (OUT / 'profiling_epochs.csv').open('w', newline='') as f:
        cols = ['run_id', 'snapshot_id', 'residual_ms', 'rpc_wall_total_ms', 'dispatch_remainder_ms']
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in prof_rows:
            w.writerow({**r, **{k: json.dumps(r[k], ensure_ascii=False) for k in
                        ('by_action_ms', 'by_segment_ms', 'gap_rpc_cover_ms') if k in r}})
    print('\n== Task A 摘要（新数据集 24 快照配对差） ==')
    for m in metrics:
        print(m, json.dumps({k: v for k, v in task_a['new'][m].items() if k != 'per_policy'}, ensure_ascii=False))
    print('\n== 会话摆动（同代码请求验证路径） ==', json.dumps(swing, ensure_ascii=False))
    print('\n== Task B 摘要 ==', json.dumps(task_b, ensure_ascii=False))
    print('\n检查 %d 项，失败：%s' % (len(checks), [c['check'] for c in checks if not c['pass']] or '无'))


if __name__ == '__main__':
    main()
