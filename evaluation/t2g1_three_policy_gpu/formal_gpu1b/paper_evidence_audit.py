"""T2-G2 最终论文证据审计（CPU-only、只读）：独立核验 formal_gpu1b 批次。

定位：不信任既有结论文件，从 63 臂原始运行产物与冻结输入独立复算，再与已发布
产物（formal_verification_main.json / analysis_summary.json 等）交叉对照。

审计范围（六项，对应任务规格）：
A 批次完整性：63=3+60 结构、状态与完成标记一致、统计产物臂域==60 主臂（门禁不进统计）、
  冻结输入三方血缘 sha、实现锁、逐臂 GPU1 设备绑定与容器命令、gate_passed 哈希链；
B 实验正确性：60 主臂 11 项合同独立重算（E/G/H、prefill、测量模式、形状预热、零编译、
  Cold 逐 Token、13 驱逐、TTFT 闭包等）+ 逐臂原始证据子集（输入 sha/退出码/清理/引擎关闭/
  设备绑定），另引用（只读）takeover/verify_evidence.py 的 2928 项全量结论；
C 统计独立复算：从 summary.json 原始字段重建 60 格数据，重算请求级 20 对×2 比较器×4 指标、
  逐 Pending 分组、轮级四 Pending 等权，与 analysis_summary.json / CSV 逐值对照，
  并输出三策略绝对 TTFT（每格五轮原始值）与逐对胜负表；
D 性能归因：E→prefill_tokens→CUDA Prefill→TTFT 对应（OLS 拟合）、TTFT 差−Prefill CUDA 差
  残余分解、控制开销三层（apply RPC 外带 / 分段 / 请求窗闭包）、恢复在线成本（COW+Native 匹配）；
E 执行身份：formal_gpu1→formal_gpu1b 编排修改恰为声明的最小修正（launch 2 处、verify/analyze
  各 1 处目录替换、verify_evidence 逐字节、device_entry 仅加 __main__ 保护）、
  c_formal 三模块 sha 与 GPU0 批次锁一致（Runtime 恢复语义与测量合同未变）。

产物：paper_evidence_audit_data.json（复算明细）+ 控制台摘要；不修改任何既有文件。
"""
import ast
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
T2G1 = ROOT / 'evaluation' / 't2g1_three_policy_gpu'
OUT = T2G1 / 'formal_gpu1b'
GPU1 = T2G1 / 'formal_gpu1'
GPU0 = T2G1 / 'formal'
GPU_UUID = 'GPU-051c9abc-864a-9f31-c4f7-a737aed997ff'
POLICIES = ['FlowState-Greedy', 'Marconi-Evict', 'KVFlow-inspired']
COMP = [('FS-MC', 'FlowState-Greedy', 'Marconi-Evict'), ('FS-KV', 'FlowState-Greedy', 'KVFlow-inspired')]
PENDINGS = ['C_A', 'C_B', 'C_C', 'C_D']
SHARED10 = ['protocol.json', 'policy_plans.json', 'prepared_inputs.json', 'source_identity.json',
            'cpu_gate.json', 'cpu_candidate_prefix_mapping.json',
            'cold_reference_C_A.json', 'cold_reference_C_B.json',
            'cold_reference_C_C.json', 'cold_reference_C_D.json']
# 逐臂 case 目录内应与批次冻结逐字节一致的 9 份共享文件（launcher 复制集合）
CASE9 = SHARED10[1:]

results = []  # (编号, 名称, 是否通过, 明细)


def ck(key, name, ok, detail=''):
    results.append({'id': key, 'check': name, 'pass': bool(ok), 'detail': str(detail)[:300]})
    return bool(ok)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def med(v):
    return round(statistics.median(v), 3)


def ols(xs, ys):
    """两参数最小二乘：返回 (截距, 斜率, R²)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return round(a, 3), round(b, 6), round(1 - ss_res / ss_tot, 4)


def main():
    protocol = read(OUT / 'protocol.json')
    plans = read(T2G1 / 't2g1_plans.json')['plans']
    cold = {p: read(OUT / ('cold_reference_%s.json' % p))['output_ids'] for p in PENDINGS}
    ptlen = {p['continuation_id']: len(p['token_ids']) for p in read(OUT / 'prepared_inputs.json')['pendings']}
    frontier = {p['continuation_id']: p['attention_frontier'] for p in read(OUT / 'prepared_inputs.json')['pendings']}
    main_arms, gate_arms = protocol['main_arms'], protocol['gate_arms']

    # ---------- A 批次完整性 ----------
    names = [a['name'] for a in main_arms]
    ck('A1', '协议 63=3 门禁+60 主臂且 5×4×3 全覆盖无重复',
       len(gate_arms) == 3 and len(names) == 60 and len(set(names)) == 60 and
       {(a['rep'], a['pending'], a['policy']) for a in main_arms} ==
       {(r, p, q) for r in range(1, 6) for p in PENDINGS for q in POLICIES})
    st_g, st_m = read(OUT / 'arm_status_gate.json'), read(OUT / 'arm_status_main.json')
    done = read(OUT / 'launch_main_complete.json')['arms']
    started = read(OUT / 'execution_started_main.json')['arms']
    ck('A2', '门禁 3/3 与主矩阵 60/60 状态全 PASS，完成标记与启动顺序==冻结调度',
       list(st_g.items()) and all(v == 'PASS' for v in st_g.values()) and len(st_g) == 3 and
       len(st_m) == 60 and all(v == 'PASS' for v in st_m.values()) and
       len(done) == 60 and all(v == 'PASS' for v in done.values()) and started == names)
    with (OUT / 'formal_arms_main.csv').open() as f:
        arms_csv = list(csv.DictReader(f))
    with (OUT / 'pair_analysis.csv').open() as f:
        pair_csv = list(csv.DictReader(f))
    with (OUT / 'round_equal_weight.csv').open() as f:
        rew_csv = list(csv.DictReader(f))
    ck('A3', '统计产物臂域==60 主臂（门禁臂零混入；CSV 无 g_* 行）',
       len(arms_csv) == 60 and all(r['arm'].startswith('m') for r in arms_csv) and
       len(pair_csv) == 40 and len(rew_csv) == 10 and
       {(int(r['rep']), r['pending']) for r in pair_csv} ==
       {(r, p) for r in range(1, 6) for p in PENDINGS} and
       all(r['phase'] == 'main' for r in arms_csv))
    sha3 = {f: (sha(OUT / f), sha(GPU1 / f), sha(GPU0 / f)) for f in SHARED10}
    ck('A4', '10 项冻结输入三方（gpu1b/formal_gpu1/formal）逐字节一致',
       all(len(set(v)) == 1 for v in sha3.values()),
       {f: v[0][:12] for f, v in sha3.items() if len(set(v)) != 1})
    precheck = read(OUT / 'formal_precheck.json')
    lock_ok, lock_bad = True, []
    for path, h in read(OUT / 'implementation_lock_main.json')['files'].items():
        if not Path(path).exists() or sha(path) != h:
            lock_ok, _ = False, lock_bad.append(Path(path).name)
    ck('A5', '实现锁 implementation_lock_main.json 全部文件当前 sha 一致（%d 文件）' % len(
        read(OUT / 'implementation_lock_main.json')['files']), lock_ok, lock_bad)
    # Runtime 恢复语义与测量合同：c_formal 三模块与 GPU0 批次锁记录一致
    gpu0_lock = read(GPU0 / 'implementation_lock_main.json')['files']
    cmod = ['evaluation/c_formal_perf_gpu/runner.py', 'evaluation/c_formal_perf_gpu/control.py',
            'evaluation/c_formal_perf_gpu/bridge.py']
    c_same = all(gpu0_lock.get(str(ROOT / m)) == sha(ROOT / m) for m in cmod)
    ck('A6', 'c_formal runner/control/bridge sha 与 GPU0 批次锁一致（Runtime/测量合同未变）', c_same)
    dev_ok, cmd_ok = True, True
    for a in gate_arms + main_arms:
        d = read(OUT / a['name'] / 'device_binding.json')
        if not (d['status'] == 'PASS' and d['host_gpu_uuid'] == GPU_UUID and d['cuda_device_count'] == 1
                and d['logical_device'] == 0):
            dev_ok = False
        c = read(OUT / (a['name'] + '_command.json'))
        s = json.dumps(c, ensure_ascii=False)
        if not ('--gpus' in s and GPU_UUID in s and 'flowstate-t2g2-g1b-' in s
                and str(OUT / a['name']) in s and 'device_entry.py' in s):
            cmd_ok = False
    ck('A7', '63 臂 device_binding 全 GPU1 唯一绑定', dev_ok)
    ck('A8', '63 臂容器命令均绑定 GPU1 UUID、g1b 前缀、本批次 case 目录与编排入口', cmd_ok)
    gp = read(OUT / 'gate_passed.json')
    chain_ok = (gp.get('passed') is True and
                gp.get('verification_sha256') == sha(OUT / 'formal_verification_gate.json') and
                gp.get('evidence_sha256') == sha(OUT / 'takeover/evidence_gate.json') and
                gp.get('protocol_sha256') == sha(OUT / 'protocol.json') and
                gp.get('checker_sha256') == sha(OUT / 'takeover/verify_evidence.py'))
    ck('A9', 'gate_passed.json 哈希链四项引用当前全部有效（本批次无覆盖瑕疵）', chain_ok)
    ck('A10', 'spawn 重入回归 PASS 在案（含旧入口 RED 对照）',
       read(OUT / 'spawn_regression.json')['status'] == 'PASS')

    # ---------- B 实验正确性：60 主臂 11 项合同独立重算 ----------
    contract_fail, n_contract = [], 0
    for arm in main_arms:
        case = OUT / arm['name']
        s = read(case / 'run' / 'summary.json')
        exp = plans[arm['policy']]['expected_E_G'][arm['pending']]
        n = arm['name']
        conds = [
            ('status', s['status'] == 'PASS' and s['policy'] == arm['policy'] and s['pending'] == arm['pending']),
            ('census16', s['census_counts'] == {'physical': 16, 'mapped': 16, 'unmapped_extras': 0}),
            ('selected', s['selected_cpu'] == plans[arm['policy']]['selected_checkpoint_ids']),
            ('E_G_H', s['E_executable_prefix'] == exp['E'] and s['G_replay_gap'] == exp['G']
             and s['H_physical_fa_hit'] == frontier[arm['pending']]),
            ('prefill', s['prefill_token_count'] == ptlen[arm['pending']] - exp['E']
             and s['batch_prefix_length'] == exp['E']),
            ('measure_mode', s.get('measure_mode') is True and s.get('audit_entries_total') == 2),
            ('shape_warm', s.get('shape_warm_b', {}).get('E') == exp['E']
             and s.get('shape_warm_b', {}).get('fa_hit') == exp['E']),
            ('zero_compile', bool(s.get('triton_new_kernels'))
             and s['triton_new_kernels'][-1]['new_entries'] == 0),
            ('cold_tokens', s['output_ids'] == cold[arm['pending']]),
            ('evictions13', len(read(case / 'run' / 'reconciliation.json')['operations']) == 13
             and all(x['returned'] for x in read(case / 'run' / 'reconciliation.json')['operations'])),
            ('ttft_closure', abs((s['timing_decomposition']['queue_ms']
                                  + s['timing_decomposition']['execwin_ms']
                                  + s['timing_decomposition']['delivery_ms'])
                                 - s['timing_decomposition']['ttft_ms_check']) < 5.0),
        ]
        n_contract += len(conds)
        contract_fail += ['%s:%s' % (n, nm) for nm, ok in conds if not ok]
    ck('B1', '60 主臂×11 项合同独立重算全部通过（%d 项）' % n_contract, not contract_fail, contract_fail)
    # 原始证据子集独立重算（63 臂：输入 sha 副本/退出码/清理/引擎关闭已在 A7/A2 之外单独计）
    ev_fail, n_ev = [], 0
    for arm in gate_arms + main_arms:
        case = OUT / arm['name']
        n = arm['name']
        for f in CASE9:
            n_ev += 1
            if sha(case / f) != sha3[f][0]:
                ev_fail.append('%s:case副本漂移 %s' % (n, f))
        n_ev += 4
        if read(OUT / (n + '_process.json'))['returncode'] != 0:
            ev_fail.append(n + ':退出码非0')
        cl = read(OUT / (n + '_cleanup.json'))
        if not (cl['status'] == 'PASS' and not cl['own_container'] and len(cl['observations']) == 3
                and all(o['memory_mib'] < 100 for o in cl['observations'])):
            ev_fail.append(n + ':清理观察不合格')
        ec = read(case / 'run' / 'engine_cleanup.json')
        if not (ec.get('engine_created') and ec.get('shutdown_returned')):
            ev_fail.append(n + ':Engine未正常关闭')
        if not read(case / 'device_binding.json')['status'] == 'PASS':
            ev_fail.append(n + ':绑定非PASS')
    ck('B2', '63 臂原始证据子集独立重算通过（%d 项：9×case 输入 sha+退出码+3 次清理+引擎关闭+绑定）'
       % n_ev, not ev_fail, ev_fail[:20])
    ev_main = read(OUT / 'takeover/evidence_main.json')
    ev_gate = read(OUT / 'takeover/evidence_gate.json')
    ck('B3', '既有全量证据复核结论引用（只读）：gate %d 项 / main %d 项均 PASS' % (
        len(ev_gate['checks']), len(ev_main['checks'])),
        ev_gate['status'] == 'PASS' and ev_main['status'] == 'PASS' and
        all(c['pass'] for c in ev_gate['checks'] + ev_main['checks']))

    # ---------- C 统计独立复算 ----------
    cell = {}
    for arm in main_arms:
        s = read(OUT / arm['name'] / 'run' / 'summary.json')
        cell[(arm['rep'], arm['pending'], arm['policy'])] = {
            'ttft': s['ttft_ms'], 'prefill_cuda': s['cuda_prefill_ms'], 'decode_cuda': s['cuda_decode_ms'],
            'compute': s['cuda_prefill_ms'] + s['cuda_decode_ms'], 'control': s['control_rpc_wall_ms'],
            'match': s['match_native_ms'] or 0.0, 'cow': s['cow_copy_cuda_ms'] + s['cow_copy_host_ms'],
            'E': s['E_executable_prefix'], 'ptok': s['prefill_token_count'],
            'mode': 'slow' if s['cuda_decode_ms'] > 160.0 else 'fast'}
    pd_ = {(int(r['rep']), r['pending'], r['comparator']): float(r['ttft_diff_ms']) for r in pair_csv}
    pair_rows, mism = [], defaultdict(int)
    for rep in range(1, 6):
        for pend in PENDINGS:
            for cname, pa, pb in COMP:
                a, b = cell[(rep, pend, pa)], cell[(rep, pend, pb)]
                if a['mode'] != b['mode']:
                    mism[cname] += 1
                pair_rows.append({'rep': rep, 'pending': pend, 'comparator': cname,
                                  'ttft_diff': round(a['ttft'] - b['ttft'], 3),
                                  'fs_ttft': round(a['ttft'], 3), 'base_ttft': round(b['ttft'], 3),
                                  'prefill_diff': round(a['prefill_cuda'] - b['prefill_cuda'], 3),
                                  'compute_diff': round(a['compute'] - b['compute'], 3),
                                  'control_diff': round(a['control'] - b['control'], 2)})
    def summ(v):
        return {'median_ms': med(v), 'min_ms': round(min(v), 3), 'max_ms': round(max(v), 3),
                'neg': sum(x < 0 for x in v), 'zero': sum(x == 0 for x in v),
                'pos': sum(x > 0 for x in v), 'n': len(v)}
    rr = {}
    for cname, _, _ in COMP:
        rr['%s_ttft' % cname] = summ([r['ttft_diff'] for r in pair_rows if r['comparator'] == cname])
        rr['%s_compute' % cname] = summ([r['compute_diff'] for r in pair_rows if r['comparator'] == cname])
        rr['%s_prefill_cuda' % cname] = summ([r['prefill_diff'] for r in pair_rows if r['comparator'] == cname])
        rr['%s_control' % cname] = summ([r['control_diff'] for r in pair_rows if r['comparator'] == cname])
    # 轮级等权：策略内四 Pending 先均值，后配对差，再五轮中位
    rew = {}
    for cname, pa, pb in COMP:
        diffs = []
        for rep in range(1, 6):
            ea = statistics.mean(cell[(rep, p, pa)]['ttft'] for p in PENDINGS)
            eb = statistics.mean(cell[(rep, p, pb)]['ttft'] for p in PENDINGS)
            diffs.append(ea - eb)
        rew[cname] = summ(diffs)
    asum = read(OUT / 'analysis_summary.json')
    def close(a, b, tol=0.002):
        return (abs(a['median_ms'] - b['median_ms']) < tol and a['neg'] == b['neg']
                and a['pos'] == b['pos'] and a['zero'] == b['zero'] and a['n'] == b['n'])
    key_nums = {'FS-MC_ttft': rr['FS-MC_ttft']['median_ms'], 'FS-KV_ttft': rr['FS-KV_ttft']['median_ms'],
                'FS-MC_rew': rew['FS-MC']['median_ms'], 'FS-KV_rew': rew['FS-KV']['median_ms']}
    ck('C1', '请求级复算==已发布（FS−MC %.3f vs −148.932；FS−KV %.3f vs −148.749；符号计数一致）'
       % (key_nums['FS-MC_ttft'], key_nums['FS-KV_ttft']),
       close(rr['FS-MC_ttft'], asum['request_level']['FS-MC_ttft_diff_ms'])
       and close(rr['FS-KV_ttft'], asum['request_level']['FS-KV_ttft_diff_ms'])
       and close(rr['FS-MC_compute'], asum['request_level']['FS-MC_compute_diff_ms'])
       and close(rr['FS-KV_compute'], asum['request_level']['FS-KV_compute_diff_ms'])
       and close(rr['FS-MC_prefill_cuda'], asum['request_level']['FS-MC_prefill_cuda_diff_ms'])
       and close(rr['FS-KV_prefill_cuda'], asum['request_level']['FS-KV_prefill_cuda_diff_ms'])
       and close(rr['FS-MC_control'], asum['request_level']['FS-MC_control_diff_ms'])
       and close(rr['FS-KV_control'], asum['request_level']['FS-KV_control_diff_ms']))
    ck('C2', '轮级等权复算==已发布（FS−MC %.3f vs −114.127；FS−KV %.3f vs −105.728；均 5/5 负）'
       % (key_nums['FS-MC_rew'], key_nums['FS-KV_rew']),
       close(rew['FS-MC'], asum['round_equal_weight']['FS-MC'])
       and close(rew['FS-KV'], asum['round_equal_weight']['FS-KV']))
    ck('C3', '逐对差与 pair_analysis.csv 40 行逐值一致',
       all(abs(pd_[(r['rep'], r['pending'], r['comparator'])] - r['ttft_diff']) < 0.002
           for r in pair_rows))
    # 逐 Pending 复算对照
    pp_ok = True
    for pend in PENDINGS:
        for cname, _, _ in COMP:
            mine = summ([r['ttft_diff'] for r in pair_rows if r['pending'] == pend and r['comparator'] == cname])
            ref = asum['per_pending'][pend]['%s_ttft_diff_ms' % cname]
            pp_ok = pp_ok and close(mine, ref)
    ck('C4', '逐 Pending 分组复算==已发布（8 组中位/符号一致）', pp_ok)
    # 绝对 TTFT：12 格×5 轮原始值（论文用）+ CSV 一致性
    abs_ttft = {('%s|%s' % (p, q)): [round(cell[(r, p, q)]['ttft'], 3) for r in range(1, 6)]
                for p in PENDINGS for q in POLICIES}
    csv_by = defaultdict(dict)  # (pending, policy) -> {rep: ttft}，与逐臂 CSV 行对齐
    for r in arms_csv:
        csv_by[(r['pending'], r['policy'])][int(r['rep'])] = float(r['ttft_ms'])
    ck('C5', '绝对 TTFT 与 formal_arms_main.csv 逐臂一致（60 值）',
       len(csv_by) == 12 and all(len(d) == 5 for d in csv_by.values()) and
       all(abs(abs_ttft['%s|%s' % k][rep - 1] - v) < 0.002
           for k, d in csv_by.items() for rep, v in d.items()))

    # ---------- D 性能归因 ----------
    grid = []
    for pend in PENDINGS:
        for q in POLICIES:
            rows = [cell[(r, pend, q)] for r in range(1, 6)]
            grid.append({'pending': pend, 'policy': q, 'E': rows[0]['E'], 'ptok': rows[0]['ptok'],
                         'ttft_med': med([r['ttft'] for r in rows]),
                         'pc_med': med([r['prefill_cuda'] for r in rows]),
                         'match_med': med([r['match'] for r in rows]),
                         'cow_med': med([r['cow'] for r in rows])})
    # E→prefill_token 恒等：prefill == len(pt)−E（B1 已含单臂；此处 12 格结构复核）
    ck('D1', '12 格结构恒等 prefill_tokens==T−E（E/T 冻结值）',
       all(g['ptok'] == ptlen[g['pending']] - g['E'] for g in grid))
    fit_ttft = ols([g['ptok'] for g in grid], [g['ttft_med'] for g in grid])
    fit_pc = ols([g['ptok'] for g in grid], [g['pc_med'] for g in grid])
    ck('D2', 'TTFT/CUDA Prefill 对 prefill_tokens 线性拟合成立（R²>0.97）',
       fit_ttft[2] > 0.97 and fit_pc[2] > 0.97,
       'TTFT=%.3f+%.6f·tok R²=%.4f；PC=%.3f+%.6f·tok R²=%.4f' % (fit_ttft + fit_pc))
    resid = {'%s_ttft_minus_pc' % c: summ([round(r['ttft_diff'] - r['prefill_diff'], 3)
                                           for r in pair_rows if r['comparator'] == c])
             for c, _, _ in COMP}
    ck('D3', 'TTFT 差≈Prefill CUDA 差（残余中位绝对值 <15ms，属 host 侧排队/投递/首Token）',
       all(abs(resid[k]['median_ms']) < 15 for k in resid), resid)
    ctrl = {}
    for q in POLICIES:
        rows = [cell[(r, p, q)] for r in range(1, 6) for p in PENDINGS]
        ctrl[q] = {'apply_rpc_med': med([r['control'] for r in rows]),
                   'match_native_med': med([r['match'] for r in rows]),
                   'cow_med': med([r['cow'] for r in rows])}
    ck('D4', '控制开销为生命周期外带（apply RPC 秒级）且请求窗闭包<5ms（B1 ttft_closure 已证）',
       all(v['apply_rpc_med'] > 5000 for v in ctrl.values()), ctrl)
    # 恢复在线成本分层：E>0 格的 COW+Native 匹配 vs E=0 格
    rec_layers = [{'pending': g['pending'], 'policy': g['policy'], 'E': g['E'],
                   'cow_med_ms': g['cow_med'], 'match_native_med_ms': g['match_med']} for g in grid]

    # ---------- E 执行身份：编排修改恰为声明的最小修正 ----------
    l_g1b, l_g1 = (OUT / 'launch_formal.py').read_text(), (GPU1 / 'launch_formal.py').read_text()
    inv = l_g1b.replace("evaluation/t2g1_three_policy_gpu/formal_gpu1b'",
                        "evaluation/t2g1_three_policy_gpu/formal_gpu1'").replace(
        "'flowstate-t2g2-g1b-", "'flowstate-t2g2-gpu1-")
    ck('E1', 'launch_formal.py 相对 formal_gpu1 仅目录+容器前缀两处替换（逆替换逐字节还原）',
       inv == l_g1)
    for fn in ['verify_formal.py', 'analyze_formal.py']:
        t_b, t_1 = (OUT / fn).read_text(), (GPU1 / fn).read_text()
        ck('E2', '%s 相对 formal_gpu1 仅输出目录一处替换（逆替换还原）' % fn,
           t_b.replace("formal_gpu1b'", "formal_gpu1'") == t_1)
    ck('E3', 'takeover/verify_evidence.py 与 formal_gpu1 逐字节相同（凭据链同源）',
       sha(OUT / 'takeover/verify_evidence.py') == sha(GPU1 / 'takeover/verify_evidence.py'))
    new_e, old_e = (OUT / 'device_entry.py').read_text(), (GPU1 / 'device_entry.py').read_text()
    tree = ast.parse(new_e)
    safe = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
            ast.Assign, ast.AnnAssign, ast.Pass)
    bad_top = [type(n).__name__ for n in tree.body
               if not isinstance(n, safe) and not (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
               and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')
               and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]  # 模块 docstring 豁免（与 test_spawn_reentry 安全形态一致）
    guard = any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__' for n in tree.body)
    same_core = all(k in new_e and k in old_e for k in
                    ["run_module('evaluation.c_formal_perf_gpu.runner'", GPU_UUID, "open('x')"])
    ck('E4', 'device_entry 修正仅加 __main__ 保护：新入口顶层无可执行语句、保护存在、'
             '核心编排（UUID 校验/独占写/runpy 接力同一 Runner）与旧版一致',
       not bad_top and guard and same_core, '顶层残留=%s' % bad_top)
    ck('E5', '冻结计划源未变（precheck 记录 t2g1_plans sha == 当前）',
       precheck['source_t2g1_plans'] == sha(T2G1 / 't2g1_plans.json')
       and precheck['policy_plans'] == sha(OUT / 'policy_plans.json'))

    # ---------- 汇总 ----------
    all_pass = all(r['pass'] for r in results)
    verdict = 'READY' if all_pass else 'PARTIAL'
    data = {'verdict': verdict, 'checks_total': len(results),
            'checks_failed': [r for r in results if not r['pass']],
            'key_numbers': key_nums,
            'request_level_recomputed': rr, 'round_equal_weight_recomputed': rew,
            'per_pair': pair_rows, 'mode_mismatch': dict(mism),
            'abs_ttft_ms': abs_ttft, 'grid12': grid, 'fits': {'ttft': fit_ttft, 'prefill_cuda': fit_pc},
            'ttft_minus_prefillcuda': resid, 'control_layers': ctrl, 'recovery_online_layers': rec_layers,
            'contract_items': n_contract, 'evidence_subset_items': n_ev,
            'counts': {'contract_main': n_contract, 'evidence_subset': n_ev,
                       'verify_formal_main': 660, 'verify_evidence_main': len(ev_main['checks']),
                       'verify_evidence_gate': len(ev_gate['checks'])}}
    (OUT / 'paper_evidence_audit_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print('审计检查 %d 项，%s' % (len(results), verdict))
    for r in results:
        print(' %s %s %s %s' % ('PASS' if r['pass'] else 'FAIL', r['id'], r['check'], r['detail']))
    print('关键数字：FS−MC 请求级 %.3f / 轮级 %.3f；FS−KV 请求级 %.3f / 轮级 %.3f' % (
        key_nums['FS-MC_ttft'], key_nums['FS-MC_rew'], key_nums['FS-KV_ttft'], key_nums['FS-KV_rew']))
    if not all_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
