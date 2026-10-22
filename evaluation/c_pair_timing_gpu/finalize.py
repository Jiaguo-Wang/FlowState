"""离线汇总：逐生命周期计时账本 + 配对差分析 + 计时口径文档，只读各生命周期产物。

- lifecycle_results.csv：逐次 E/G、Prefill Token、TTFT 六段分解、CUDA Prefill、首 decode、
  COW、控制分段、Triton 各窗口新增编译数；
- pair_analysis.csv：CB 与 CC 各 5 对的 FS−SA 配对差（TTFT/execwin/compute/pregap/queue/delivery），
  CC 为相同恢复前沿的零假设带宽；不把 5 次重复解释为显著性；
- timing_scopes.md：每个计时指标覆盖事件的口径表（自动生成，含本轮实测中位数）。
"""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def qmed(vals):
    """中位数（小样本如实返回，不做分布假设）。"""
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    protocol = read(OUT / 'protocol.json')
    arms = [a for a in protocol['phase1_arms']] + \
           [a for a in protocol['phase2_arms'] if (OUT / a['name'] / 'run' / 'summary.json').exists()]
    rows = []
    by_pair = {}
    for arm in arms:
        case = OUT / arm['name']
        sp = case / 'run' / 'summary.json'
        if not sp.exists():
            rows.append({'name': arm['name'], 'status': 'NOT_RUN'})
            continue
        s = read(sp)
        if s.get('status') != 'PASS':
            rows.append({'name': arm['name'], 'status': s.get('status', 'UNKNOWN')})
            continue
        d = s['timing_decomposition']
        tw = {w['window']: w['new_entries'] for w in s['triton_new_kernels']}
        row = {'name': arm['name'], 'status': 'PASS', 'mode': s['mode'], 'pair': arm['pair'],
               'experiment': arm['experiment'], 'policy': s['policy'], 'pending': s['pending'],
               'E': s['E_executable_prefix'], 'G': s['G_replay_gap'], 'H': s['H_physical_fa_hit'],
               'prefill_token_count': s['prefill_token_count'],
               'ttft_ms': round(s['ttft_ms'], 1), 'request_wall_ms': round(s['request_wall_ms'], 1),
               'queue_ms': round(d['queue_ms'], 1), 'pregap_ms': round(d['pregap_ms'], 1),
               'execwin_ms': round(d['execwin_ms'], 1), 'compute_cuda_ms': round(d['compute_cuda_ms'], 1),
               'postgap_ms': round(d['postgap_ms'], 1), 'delivery_ms': round(d['delivery_ms'], 1),
               'first_decode_cuda_ms': round(d['first_decode_cuda_ms'], 1),
               'cuda_prefill_ms': round(s['cuda_prefill_ms'], 1), 'cuda_decode_ms': round(s['cuda_decode_ms'], 1),
               'cow_copy_count': s['cow_copy_count'], 'cow_copy_cuda_ms': round(s['cow_copy_cuda_ms'], 2),
               'control_rpc_wall_ms': round(s['control_rpc_wall_ms'], 1),
               'engine_boot_ms': round(s['engine_boot_ms'], 1), 'warmup1_wall_ms': round(s['warmup1_wall_ms'], 1),
               'warmup2_wall_ms': round(s['warmup2_wall_ms'], 1) if s['warmup2_wall_ms'] is not None else None,
               'epoch_wall_ms': round(s['epoch_wall_ms'], 1),
               'triton_new_warmup': tw.get('engine_ready→post_warmup'),
               'triton_new_throwaway': tw.get('post_warmup→post_throwaway'),
               'triton_new_warmup2': tw.get('post_flush→post_warmup2'),
               'triton_new_control': tw.get(('post_warmup2→post_control' if s['warmup2_wall_ms'] is not None
                                             else 'post_warmup→post_control')),
               'triton_new_request': tw.get('post_control→post_request'),
               'cold_output_match': s['cold_output_match']}
        rows.append(row)
        by_pair.setdefault((arm['experiment'].replace('_warm', ''), arm['pair']), {})[arm['policy']] = row

    # 配对差：每对 FS−SA（CB=收益检验，CC=零假设带宽；w 对只入 CB_warm 记录不并入 p1 统计）
    pair_rows = []
    deltas = {'CB': {k: [] for k in ['ttft_ms', 'execwin_ms', 'compute_cuda_ms', 'pregap_ms', 'queue_ms',
                                     'delivery_ms', 'request_wall_ms']},
              'CC': {k: [] for k in ['ttft_ms', 'execwin_ms', 'compute_cuda_ms', 'pregap_ms', 'queue_ms',
                                     'delivery_ms', 'request_wall_ms']}}
    for (exp, pair), policies in sorted(by_pair.items()):
        if 'FlowState-Greedy' not in policies or 'StandaloneTopK' not in policies:
            continue
        fs, sa = policies['FlowState-Greedy'], policies['StandaloneTopK']
        rec = {'experiment': exp, 'pair': pair,
               'fs_name': fs['name'], 'sa_name': sa['name'],
               'fs_E': fs['E'], 'sa_E': sa['E']}
        for k in deltas[exp] if exp in deltas else []:
            rec['d_' + k] = round(fs[k] - sa[k], 1)
            if exp in deltas:
                deltas[exp][k].append(fs[k] - sa[k])
        pair_rows.append(rec)

    with (OUT / 'lifecycle_results.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    with (OUT / 'pair_analysis.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in pair_rows for k in r}))
        w.writeheader()
        w.writerows(pair_rows)

    # 汇总统计：CB/CC 各指标配对差中位数与范围；CC 为零假设带宽
    stats = {}
    for exp in ['CB', 'CC']:
        stats[exp] = {}
        for k, vals in deltas[exp].items():
            if vals:
                stats[exp][k] = {'median': round(qmed(vals), 1), 'min': round(min(vals), 1),
                                 'max': round(max(vals), 1), 'n_pairs': len(vals)}
    passed = [r for r in rows if r.get('status') == 'PASS']
    result = {'protocol_version': protocol['version'],
              'lifecycles_total': len(arms), 'lifecycles_passed': len(passed),
              'pair_delta_stats': stats,
              'conclusion': 'READY' if len(passed) == len(arms) else 'PARTIAL'}
    with (OUT / 'finalize_results.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 计时口径文档（含本轮实测中位数，便于对照阶段归属）
    med = lambda rows_, k: qmed([r[k] for r in rows_ if k in r and r.get('status') == 'PASS']) \
        if any(k in r for r in rows_) else None
    scopes = [
        ('TTFT（ttft_ms）', 'client 发送请求 → 首个含 Token 的流式 chunk 到达；含 RPC/排队/准入/编译间隙/前向/采样/解词全部', 'ttft_ms'),
        ('queue（queue_ms）', 'client 发送 → 引擎首 batch 开始（execution[0].ns_before）；含调度等待、radix/FA 匹配、'
                             'mamba 准入（E>0 时含 COW 暂存与 payload 哈希审计）', 'queue_ms'),
        ('pregap（pregap_ms）', '首 batch 开始 → 首个 kernel CUDA 事件；host 侧 kernel 编译/autotune 主要落在此段', 'pregap_ms'),
        ('execwin（execwin_ms）', '首 batch 整窗（execution[0] 事件 elapsed）；事件间 GPU 空闲（等 host 编译）计入', 'execwin_ms'),
        ('compute（compute_cuda_ms）', '紧凑 kernel 事件窗（compute[0].cuda_elapsed_ms）；含事件间由编译造成的空隙，'
                                      '非纯 kernel 执行时间之和', 'compute_cuda_ms'),
        ('postgap（postgap_ms）', '末 kernel 事件 → batch 结束', 'postgap_ms'),
        ('delivery（delivery_ms）', 'batch 结束 → 首 chunk 到达；含采样、解词、流式下发', 'delivery_ms'),
        ('cuda_prefill_ms', '全部 is_extend compute 事件 elapsed 之和（=compute_cuda_ms，单 extend batch）', 'cuda_prefill_ms'),
        ('first_decode（first_decode_cuda_ms）', '首个 decode 步事件窗；首 decode 形状编译证据（引擎 disable_cuda_graph=true，无 Graph Capture）', 'first_decode_cuda_ms'),
        ('control_rpc_wall_ms', 'b5d4_apply RPC 墙钟；审计增强型研究仪表（重型观察+FA 内容审计），与请求计时完全分离', 'control_rpc_wall_ms'),
        ('cow_copy_cuda_ms', 'COW 设备拷贝事件 elapsed（宿主暂存与哈希审计不计入，落在 queue 内）', 'cow_copy_cuda_ms'),
    ]
    with (OUT / 'timing_scopes.md').open('x') as f:
        f.write('# 计时口径表：每个指标覆盖的事件范围\n\n'
                'client 与 engine 同用 CLOCK_MONOTONIC，跨进程时间戳直接衔接；分解恒等式 '
                'TTFT ≈ queue + execwin + delivery（容差 5ms，runner 内闭合断言）。\n\n'
                '| 指标 | 覆盖事件 | 本轮中位数(ms) |\n|---|---|---|\n')
        for title, scope, key in scopes:
            v = med(rows, key)
            f.write(f'| {title} | {scope} | {round(v, 1) if v is not None else "—"} |\n')
    print('汇总完成：', result['conclusion'], '生命周期', len(passed), '/', len(arms),
          '| CB 配对差中位数 ttft =', stats['CB'].get('ttft_ms', {}).get('median'),
          '| CC 零假设带宽 ttft =', stats['CC'].get('ttft_ms', {}))


if __name__ == '__main__':
    main()
