"""G2-A 门禁复核（宿主机执行，只读证据）：8 臂逐臂合同 + 跨臂 D4b TTFT 反向专项检查。

复核链（证据零漂移；任何不满足即该臂 FAIL）：
1. 生命周期：进程 returncode==0、cleanup PASS、执行标记与锁齐全；
2. 求解臂语义：Exact 臂=生产 plan_strict 自动 Exact（gate_allowed）；Greedy 臂=greedy_adapter
   生产路径且域满足门禁（决策记录含“未篡改门禁/候选/预算”）；两臂事后 e3 参照全精度一致，
   独立 Decimal Oracle（29 子集）PASS 且 Exact 臂 winner==在线选集；
3. B5-C4 Prefill 合同：FA命中==冻结前沿、E==计划、Prefill Token==len(pt)−E、batch_prefix==E、
   G_model==T−E 分列；未选 5 状态全驱逐；
4. 输出正确性：8 Token 与独立 Cold 参照逐 Token 一致；
5. 测量有效性：审计==2（apply 基线+请求后）、形状预热在正式计时前且 E==计划 E、
   请求窗口零新增编译、TTFT 六段分解闭合（<5ms）、控制分段齐全；
6. D4b 专项：逐 Pending 检查 ΔCUDA Prefill 与 ΔTTFT（Greedy−Exact）方向——两量均超出 ±3ms
   噪声带时不得反向（历史上 D4b 曾出现 Prefill 改善而 TTFT 反向，此处必须显式排除）。
"""
import json
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
GATE = ROOT / 'evaluation/b5c4_formal_gpu_g2/preflight/gate'
PRE = ROOT / 'evaluation/b5c4_formal_gpu_g2/preflight'
ARMS = [(s, l) for s in ('Exact', 'Greedy') for l in 'ABCD']
REF = json.loads((PRE / 'g2_reference.json').read_text())
COLD = {p: json.loads((PRE / ('cold_reference_B5C2_%s_TURN_003.json' % p)).read_text())['output_ids'] for p in 'ABCD'}
NOISE_MS = 3.0
EXPECTED_STAGE_PREFIX = ('heavy_observation_gpu_sync', 'strict_epoch_authentication', 'online_solve', 'independent_oracle',
                         'apply_time_freshness_preflight', 'fa_content_audit_before', 'strict_reconciliation_with_postchecks',
                         'fa_content_postcheck_and_resume_baseline')


def read(p):
    return json.loads(Path(p).read_text())


def verify_arm(solver, letter):
    run = '%s_%s' % (solver.lower(), letter.lower())
    d = GATE / run
    row = {'run': run, 'solver': solver, 'pending': 'B5C2_%s_TURN_003' % letter, 'checks': [], 'fail': []}

    def check(name, ok, detail=''):
        row['checks'].append({'check': name, 'pass': bool(ok), 'detail': str(detail)[:200]})
        if not ok:
            row['fail'].append(name + (': ' + str(detail)[:120] if detail else ''))

    try:
        s = read(d / 'summary.json')
        check('summary_pass', s.get('status') == 'PASS' and s.get('arm') == 'strict' and s.get('phase') == 'gate')
        check('lifecycle', read(GATE / (run + '_process.json'))['returncode'] == 0 and
              read(GATE / (run + '_cleanup.json'))['status'] == 'PASS' and (GATE / (run + '_execution_started.json')).exists())
        rec = read(d / 'reconciliation.json')
        # ── 求解臂语义 ──
        dec = s['strict_decision']
        if solver == 'Exact':
            check('exact_branch', dec['solver'] == 'Exact' and dec['gate_allowed'] is True and
                  'plan_strict' in s['selection_source'], dec.get('gate'))
            check('exact_oracle_winner', rec['oracle']['status'] == 'PASS' and rec['oracle']['subsets'] == 29 and
                  set(rec['oracle']['selected']) == set(s['selected_ids']))
        else:
            check('greedy_domain_admissible', dec['gate_allowed'] is True and '未篡改门禁/候选/预算' in str(dec.get('note', '')), dec.get('gate'))
            check('greedy_oracle_bound', rec['oracle']['status'] == 'PASS' and rec['oracle']['subsets'] == 29 and
                  rec['plan']['recovery_cost_ms'] >= rec['oracle']['J_ms'] - 1e-9)
        rrow = REF['Exact' if solver == 'Exact' else 'Marginal Greedy']
        check('e3_selection_match', set(s['selected_ids']) == set(rrow['selected']) and s['oracle_e3']['match'] is True)
        check('e3_J_match', abs(s['plan_recovery_cost_ms'] - rrow['J_ms']) <= 1e-9)
        raw = rec['raw_epoch']
        check('domain_7x4', len(raw['candidates']) == 7 and len(raw['pendings']) == 4 and rec['epoch']['K'] == 2)
        pend = {p['continuation_id']: p for p in raw['pendings']}
        plan_pend = {p['pending_id']: p for p in rec['plan']['pending']}
        # ── B5-C4 Prefill 合同（逐 Pending）──
        prefill_ok, evict_ok = True, True
        for pid, p in pend.items():
            e = plan_pend[pid]
            prefill_ok &= (e['E'], e['T'], e['G']) == (rrow['per_pending'][pid]['E'], rrow['per_pending'][pid]['T'], rrow['per_pending'][pid]['G'])
            prefill_ok &= e['G'] == e['T'] - e['E'] and 0 <= e['E'] <= len(p['token_ids'])
        tp = rec['target']['pending']
        check('prefill_contract_all_pendings', prefill_ok)
        check('prefill_token_count', s['prefill_token_count'] == len(tp['token_ids']) - s['E_executable_prefix'] and
              s['batch_prefix_length'] == s['E_executable_prefix'])
        check('fa_hit_frontier', s['H_physical_fa_hit'] == tp['attention_frontier'] and
              s['E_executable_prefix'] == plan_pend[tp['continuation_id']]['E'] and
              s['G_model_replay_gap'] == plan_pend[tp['continuation_id']]['G'])
        selected = set(s['selected_ids'])
        evicted_expected = {c['checkpoint_id'] for c in raw['candidates']} - selected
        check('evictions_complete', {c for c, _ in s['evictions']} == evicted_expected and
              all(r for _, r in s['evictions']) and len(s['evictions']) == 5)
        # ── 输出正确性 ──
        check('cold_token_match', s['output_ids'] == COLD[letter] and s['cold_output_match'] is True)
        # ── 测量有效性 ──
        check('audit_two', s['audit_entries_total'] == 2)
        check('request_window_zero_compile', s['triton_new_kernels'][-1]['new_entries'] == 0 and
              s['triton_new_kernels'][-1]['window'].endswith('post_shape_warm→post_request'))
        check('shape_warm_before_timing', s['shape_warm_b']['E'] == s['E_executable_prefix'] and
              read(d / 'shape_warm_b.json')['planned_E'] == s['E_executable_prefix'])
        dd = s['timing_decomposition']
        check('ttft_decompose_closed', abs(dd['queue_ms'] + dd['execwin_ms'] + dd['delivery_ms'] - dd['ttft_ms_check']) < 5.0 and
              abs(dd['ttft_ms_check'] - s['ttft_ms']) < 1.0)
        stages = {t['stage'] for t in s['control_timing_stages']}
        check('control_stages_complete', all(any(st.startswith(x) for st in stages) for x in EXPECTED_STAGE_PREFIX), sorted(stages))
        streams = read(d / 'stream_chunks.json')
        check('output_timestamps', len(s['output_ids']) == 8 and streams['first_token_ns'] > streams['start_ns'] and
              len(streams['arrivals']) >= 1)
        check('state_preserved', rec.get('fa_content_postcheck') == 'PASS' and
              rec.get('selected_payload_after') == rec.get('selected_payload_before'))
        row['metrics'] = {'E': s['E_executable_prefix'], 'H': s['H_physical_fa_hit'],
                          'cuda_prefill_ms': s['cuda_prefill_ms'], 'ttft_ms': s['ttft_ms'],
                          'request_wall_ms': s['request_wall_ms'], 'cuda_decode_ms': s['cuda_decode_ms'],
                          'control_rpc_wall_ms': s['control_rpc_wall_ms']}
    except BaseException as exc:
        row['fail'].append('exception:' + repr(exc))
    row['status'] = 'PASS' if not row['fail'] else 'FAIL'
    return row


def main():
    rows = [verify_arm(s, l) for s, l in ARMS]
    by = {(r['solver'], r['pending'].split('_')[1]): r for r in rows}
    # ── D4b 专项：ΔCUDA Prefill 与 ΔTTFT 方向一致性（Greedy−Exact；±3ms 噪声带外不得反向）──
    d4b = []
    d4b_ok = True
    for letter in 'ABCD':
        e, g = by[('Exact', letter)], by[('Greedy', letter)]
        if e['status'] != 'PASS' or g['status'] != 'PASS':
            d4b.append({'pending': letter, 'skip': '臂未PASS'})
            continue
        dp = g['metrics']['cuda_prefill_ms'] - e['metrics']['cuda_prefill_ms']
        dt = g['metrics']['ttft_ms'] - e['metrics']['ttft_ms']
        reversed_dir = (dp < -NOISE_MS and dt > NOISE_MS) or (dp > NOISE_MS and dt < -NOISE_MS)
        d4b_ok &= not reversed_dir
        d4b.append({'pending': letter, 'delta_cuda_prefill_ms': round(dp, 2), 'delta_ttft_ms': round(dt, 2),
                    'both_beyond_noise': abs(dp) > NOISE_MS and abs(dt) > NOISE_MS, 'reversed': reversed_dir})
    all_pass = all(r['status'] == 'PASS' for r in rows)
    verdict = 'PASS' if all_pass and d4b_ok else ('BLOCKED' if all(not r['checks'] for r in rows) else 'PARTIAL')
    result = {'verdict': verdict, 'arms': rows, 'd4b_ttft_reversal_check': {'noise_band_ms': NOISE_MS, 'rows': d4b, 'pass': d4b_ok,
               'note': '两量均超出噪声带时不得反向；A/B 为 E 相同对照（应近零差），C/D 为结构差异主效应'},
              'cold_references': {p: 'preflight/cold_reference_B5C2_%s_TURN_003.json' % p for p in 'ABCD'},
              'gate_scope': 'G2-A 门禁：8/8 臂合同 + 测量有效性 + D4b 专项；性能数值仅记录，不在此宣称结论'}
    out = GATE / 'verify.json'
    assert not out.exists(), '拒绝覆盖 verify.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('G2-A 门禁复核：', verdict)
    for r in rows:
        print(' ', r['run'], r['status'], ('FAIL:' + ';'.join(r['fail'])) if r['fail'] else '')
    for x in d4b:
        print('  D4b', x)


if __name__ == '__main__':
    main()
